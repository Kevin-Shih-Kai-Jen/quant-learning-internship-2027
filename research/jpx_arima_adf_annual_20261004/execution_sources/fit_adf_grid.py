"""Fit only new d=0/2 stock-years; workers receive no labels or realized returns."""
import argparse
import io
import json
import sqlite3
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from common import EXP, INPUT, OUT, CONFIG, ORDERS, YEARS, save, sha, hashes

DATA = None
DECISIONS = None


def initialize():
    global DATA, DECISIONS
    with np.load(INPUT/'prices.npz') as z:
        DATA = {k:z[k] for k in z.files}
    doc = json.loads((OUT/'adf_decisions.json').read_text())
    assert doc['specification_hashes'] == hashes()
    DECISIONS = {(r['year'],r['ci']): r for r in doc['records']}


def forecast_states(result, positions):
    fr = result.filter_results
    assert fr.transition.shape[-1] == 1 and fr.design.shape[-1] == 1
    trans, design = fr.transition[:,:,0], fr.design[:,:,0]
    state_intercept = fr.state_intercept[:,0]
    obs_intercept = fr.obs_intercept[:,0]
    if fr.obs_intercept.shape[1] > 1:
        np.testing.assert_allclose(fr.obs_intercept, obs_intercept[:,None] * np.ones_like(fr.obs_intercept))
    s1 = fr.predicted_state[:,positions+1]
    h1 = (design @ s1 + obs_intercept[:,None])[0]
    h2 = (design @ (trans @ s1 + state_intercept[:,None]) + obs_intercept[:,None])[0]
    return np.column_stack([h1,h2])


def fit_one(key):
    oi, yi, ci = key
    year, (p,q) = YEARS[yi], ORDERS[oi]
    decision = DECISIONS[(year,ci)]
    d = decision['chosen_d']
    assert d in (0,2)
    positions = DATA['valid_positions'][DATA['years']==year]
    train_end, end = int(positions[0]), int(positions[-1])+1
    prices = DATA['prices'][:end,ci].copy()
    dates = DATA['dates'][:end]
    train = prices[:train_end]
    code = int(DATA['codes'][ci])
    trend = CONFIG['d0_trend'] if d == 0 else CONFIG['positive_d_trend']
    order = (p,d,q)
    assert decision['train_end_exclusive'] == train_end
    assert decision['train_last'] == str(dates[train_end-1]) < str(dates[positions[0]])
    started = time.monotonic()
    audit = {'code':code,'year':year,'p':p,'d':d,'q':q,'trend':trend,
             'train_start':str(dates[0]),'train_last':str(dates[train_end-1]),
             'first_signal':str(dates[positions[0]]),'last_signal':str(dates[positions[-1]]),
             'training_slots':train_end,'finite_training_prices':int(np.isfinite(train).sum()),
             'attempts':[],'warnings':[],'causal_checks':[],'status':'pending'}
    score = np.zeros(len(positions))
    fallback = np.ones(len(positions),dtype=bool)
    forecasts = np.full((len(positions),2),np.nan)
    params = np.empty(0)
    anchor = float(train[np.isfinite(train)][0])
    scale = float(np.nanstd(np.diff(train),ddof=1))
    audit.update(anchor=anchor,scale=scale if np.isfinite(scale) else None)
    fitted = None
    accepted = False
    if not np.isfinite(scale) or scale <= 1e-12*max(1.,abs(anchor)):
        audit['status'] = 'constant_or_invalid_scale'
    else:
        normalized = (prices-anchor)/scale
        # Only this training prefix is supplied to likelihood optimization.
        model = ARIMA(normalized[:train_end],order=order,trend=trend,
                      enforce_stationarity=True,enforce_invertibility=True)
        try:
            for iterations in CONFIG['optimizer_maxiter']:
                kwargs = {'method_kwargs':{'maxiter':iterations,'disp':0},'cov_type':'none'}
                if fitted is not None and np.isfinite(fitted.params).all():
                    kwargs['start_params'] = fitted.params
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    fitted = model.fit(**kwargs)
                audit['warnings'].extend(str(w.message) for w in caught)
                ret = fitted.mle_retvals
                audit['attempts'].append({'converged':bool(ret.get('converged',False)),
                    'iterations':int(ret.get('iterations',0)),
                    'llf':float(fitted.llf) if np.isfinite(fitted.llf) else None})
                if ret.get('converged',False): break
            accepted = (bool(fitted.mle_retvals.get('converged',False)) and
                        np.isfinite(fitted.params).all() and np.isfinite(fitted.llf)
                        and fitted.params[-1] > 0 and fitted.llf != 0)
            if accepted:
                measured = np.isfinite(normalized[:train_end])
                measured[:int(fitted.loglikelihood_burn)] = False
                variances = fitted.filter_results.forecasts_error_cov[0,0,:]
                accepted = bool(np.isfinite(variances[measured]).all() and (variances[measured]>0).all())
            if not accepted: audit['status'] = 'fit_not_converged_or_invalid'
        except Exception as exc:
            audit.update(status='training_exception',exception=f'{type(exc).__name__}: {exc}')
            accepted = False
        if accepted:
            params = np.asarray(fitted.params)
            audit['params'] = dict(zip(fitted.param_names,map(float,params)))
            audit['min_ar_root'] = float(np.min(np.abs(fitted.arroots)))
            audit['min_ma_root'] = float(np.min(np.abs(fitted.maroots)))
            assert audit['min_ar_root'] > 1-1e-8 and audit['min_ma_root'] > 1-1e-8
            # Unexpected forward-filter exceptions are fatal, never retroactive fallback.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                full = ARIMA(normalized,order=order,trend=trend).filter(params,cov_type='none')
            forecasts = forecast_states(full,positions)*scale+anchor
            obs = np.isfinite(normalized)
            obs[:int(full.loglikelihood_burn)] = False
            fv = full.filter_results.forecasts_error_cov[0,0,:]
            invalid = obs & (~np.isfinite(fv)|(fv<=0))
            causal_bad = np.maximum.accumulate(invalid)[positions]
            good = np.isfinite(forecasts).all(axis=1)&(forecasts>0).all(axis=1)&~causal_bad
            with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
                score[good] = forecasts[good,1]/forecasts[good,0]-1.
            good &= np.isfinite(score)
            fallback = ~good
            score[fallback] = 0.
            audit.update(status='ok',invalid_forward_observations=int(invalid.sum()),
                         forward_numerical_fallback_days=int(causal_bad.sum()))
            # Deterministic price-only audit schedule, independent of model performance.
            check_samples = ci % 29 == 0 or code in [1301,1332,7203,9984]
            if check_samples:
                for j in sorted(set([0,len(positions)//2,len(positions)-1])):
                    pos = int(positions[j])
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        prefix = ARIMA(normalized[:pos+1],order=order,trend=trend).filter(params,cov_type='none')
                        direct = np.asarray(prefix.forecast(2))*scale+anchor
                        changed = normalized.copy()
                        changed[pos+1:] = 10000.+127*np.arange(len(changed)-pos-1)
                        perturbed = ARIMA(changed,order=order,trend=trend).filter(params,cov_type='none')
                    np.testing.assert_allclose(direct,forecasts[j],rtol=2e-10,atol=2e-8)
                    np.testing.assert_allclose(prefix.filter_results.predicted_state[:,-1],
                                               perturbed.filter_results.predicted_state[:,pos+1],rtol=1e-11,atol=1e-10)
                    audit['causal_checks'].append({'date':str(dates[pos]),
                        'max_price_error':float(np.max(np.abs(direct-forecasts[j])))})
            # Residual diagnostics for newly estimated fits, not a selection criterion.
            innovations = fitted.filter_results.standardized_forecasts_error[0].copy()
            innovations[~np.isfinite(train)] = np.nan
            innovations[:int(fitted.loglikelihood_burn)] = np.nan
            from prepare_adf import longest_segment
            lo,hi = longest_segment(innovations)
            if hi-lo>40:
                lb=acorr_ljungbox(innovations[lo:hi],lags=[20],model_df=p+q,return_df=True)
                audit['ljung_box_training_lag20_pvalue']=float(lb.lb_pvalue.iloc[0])
    audit['warnings'] = sorted(set(audit['warnings']))
    audit['fallback_days'] = int(fallback.sum())
    audit['seconds'] = time.monotonic()-started
    buffer=io.BytesIO()
    np.savez_compressed(buffer,score=score,fallback=fallback,forecast_prices=forecasts,params=params)
    return key,audit,buffer.getvalue()


def connect():
    connection=sqlite3.connect(OUT/'new_fits.sqlite',timeout=60)
    connection.execute('PRAGMA journal_mode=WAL')
    connection.execute('PRAGMA synchronous=NORMAL')
    connection.execute('CREATE TABLE IF NOT EXISTS fits (oi INTEGER, yi INTEGER, ci INTEGER, status TEXT, seconds REAL, audit TEXT, payload BLOB, PRIMARY KEY(oi,yi,ci))')
    return connection


def run(workers, pilot):
    initialize()
    specification = {'hashes':hashes(),'runner_sha256':sha(__file__),
                     'adf_decisions_sha256':sha(OUT/'adf_decisions.json'),
                     'price_sha256':sha(INPUT/'prices.npz')}
    manifest=OUT/'fit_manifest.json'
    if manifest.exists():
        assert json.loads(manifest.read_text())==specification, 'Changed specification: preserve old version and create new checkpoints.'
    else: save(manifest,specification)
    con=connect()
    done=set(con.execute('SELECT oi,yi,ci FROM fits'))
    all_keys=[(oi,yi,ci) for yi,year in enumerate(YEARS) for ci in range(len(DATA['codes']))
              if DECISIONS[(year,ci)]['chosen_d'] in [0,2] for oi in range(len(ORDERS))]
    if pilot:
        stock_years=[]
        for yi,year in enumerate(YEARS):
            for d in [0,2]:
                cis=[ci for ci in range(len(DATA['codes'])) if DECISIONS[(year,ci)]['chosen_d']==d]
                stock_years.extend((yi,cis[i]) for i in sorted(set([0,len(cis)//2])) if cis)
        keys=[key for key in all_keys if (key[1],key[2]) in stock_years and ORDERS[key[0]] in [(1,1),(3,1),(5,5)]]
    else: keys=all_keys
    todo=[key for key in keys if key not in done]
    started=time.monotonic();completed=0;counts=Counter();last_report=started
    print(json.dumps({'mode':'pilot' if pilot else 'full','tasks':len(todo),'all_new_fits':len(all_keys),'workers':workers}),flush=True)
    iterator=iter(todo)
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize) as pool:
        pending={pool.submit(fit_one,next(iterator)) for _ in range(min(workers*2,len(todo)))}
        while pending:
            finished,pending=wait(pending,timeout=1,return_when=FIRST_COMPLETED)
            for future in finished:
                key,audit,payload=future.result()
                con.execute('INSERT INTO fits VALUES (?,?,?,?,?,?,?)',(*key,audit['status'],audit['seconds'],json.dumps(audit,allow_nan=False),payload))
                completed+=1;counts[audit['status']]+=1
                if completed%10==0:con.commit()
                try:pending.add(pool.submit(fit_one,next(iterator)))
                except StopIteration:pass
            now=time.monotonic()
            if now-last_report>=15 or not pending:
                con.commit()
                progress={'mode':'pilot' if pilot else 'full','session_completed':completed,'session_total':len(todo),
                          'total_completed':len(done)+completed,'all_new_fits':len(all_keys),
                          'seconds':round(now-started,1),'statuses':dict(counts),
                          'estimated_remaining_seconds':round((len(todo)-completed)*(now-started)/completed,1) if completed else None}
                save(OUT/('pilot_progress.json' if pilot else 'fit_progress.json'),progress)
                print(json.dumps(progress),flush=True);last_report=now
    con.commit();con.execute('PRAGMA wal_checkpoint(TRUNCATE)');con.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=6);parser.add_argument('--pilot',action='store_true')
    args=parser.parse_args();run(args.workers,args.pilot)
