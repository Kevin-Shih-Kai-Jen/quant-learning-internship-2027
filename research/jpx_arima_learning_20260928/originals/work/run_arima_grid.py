"""Checkpointed 25-model JPX grid. Workers receive prices only, never Target."""
from pathlib import Path
import os,sys,io,time,json,hashlib,sqlite3,warnings,argparse,shutil
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from statsmodels.tsa.arima.model import ARIMA
ROOT=Path(__file__).resolve().parent
RUN=ROOT/'arima_grid'
SOURCE=ROOT/'arima012_expanding'
BASE=Path('/Users/coolguy/.codex/.chatgpt-projects/g-p-6a8ad7dc606481919474d5424342a1c3')
ORDERS=[(p,q) for p in range(1,6) for q in range(1,6)]
YEARS=[2018,2019,2020,2021]
MIN_OBS=126
DATA=None

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def save(path,value):
    tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False));tmp.replace(path)

def initialize():
    global DATA
    with np.load(RUN/'prices.npz') as z:DATA={k:z[k] for k in z.files}

def connect():
    con=sqlite3.connect(RUN/'fits.sqlite',timeout=60)
    con.execute('PRAGMA journal_mode=WAL');con.execute('PRAGMA synchronous=NORMAL')
    con.execute('CREATE TABLE IF NOT EXISTS fits (oi INTEGER, yi INTEGER, ci INTEGER, status TEXT, seconds REAL, audit TEXT, payload BLOB, PRIMARY KEY(oi,yi,ci))')
    return con

def prepare():
    RUN.mkdir(exist_ok=True)
    for n in ['prices.npz','labels.pkl','baseline.pkl','calendar.csv']:
        if not (RUN/n).exists():shutil.copy2(SOURCE/n,RUN/n)
    prep=json.loads((SOURCE/'preparation.json').read_text())
    prep.update(plan_sha256=sha(ROOT/'arima_grid_plan.md'),runner_sha256=sha(__file__),
                cached_from=str(SOURCE),input_hashes={n:sha(RUN/n) for n in ['prices.npz','labels.pkl','baseline.pkl','calendar.csv']},
                orders=ORDERS,total_fits=200000)
    if (RUN/'preparation.json').exists():
        prior=json.loads((RUN/'preparation.json').read_text())
        assert prior['runner_sha256']==prep['runner_sha256'] and prior['plan_sha256']==prep['plan_sha256'], 'Do not mix changed specifications with existing checkpoints'
    save(RUN/'preparation.json',prep)
    connect().close()

def forecast_states(result, positions):
    """Use forward predicted states only; smoothed states must never be used."""
    fr = result.filter_results
    trans = fr.transition[:,:,0]
    design = fr.design[:,:,0]
    intercept = fr.state_intercept[:,0]
    obsint = fr.obs_intercept[:,0]
    state1 = fr.predicted_state[:,positions+1]
    h1 = (design @ state1 + obsint[:,None])[0]
    h2 = (design @ (trans@state1+intercept[:,None])+obsint[:,None])[0]
    return np.column_stack([h1,h2])


def fit_one(key):
    oi, yi, ci = key
    p, q = ORDERS[oi]; order=(p,1,q)
    year=int(YEARS[yi]);code=int(DATA['codes'][ci])
    positions=DATA['valid_positions'][DATA['years']==year]
    train_end=int(positions[0]);end=int(positions[-1])+1
    y=DATA['prices'][:end,ci].copy();dates=DATA['dates'][:end]
    start=0
    t0 = time.monotonic()
    audit = dict(p=p,q=q,code=int(code), year=int(year), status='pending',
                 train_start=str(dates[0]),train_last=str(dates[train_end-1]),
                 first_signal=str(dates[positions[0]]),last_signal=str(dates[positions[-1]]),
                 training_slots=int(train_end), valid_training_prices=int(np.isfinite(y[:train_end]).sum()),
                 missing_training_prices=int(np.isnan(y[:train_end]).sum()),
                 attempts=[], warnings=[], prefix_checks=[])
    assert train_end <= positions[0] and dates[train_end-1] < dates[positions[0]]
    score = np.zeros(len(positions)); fallback=np.ones(len(positions),dtype=bool)
    forecasts = np.full((len(positions),2),np.nan)
    pars = np.empty(0)
    train=y[:train_end]
    if audit['valid_training_prices'] < MIN_OBS:
        audit['status']='insufficient_history'
    else:
        anchor=float(train[np.isfinite(train)][0])
        scale=float(np.nanstd(np.diff(train),ddof=1))
        audit.update(anchor=anchor,scale=scale)
        if not np.isfinite(scale) or scale <= 1e-12*max(1.,abs(anchor)):
            audit['status']='constant_or_invalid_scale'
        else:
            normalized=(y-anchor)/scale
            model=ARIMA(normalized[:train_end],order=order,trend='n',
                        enforce_stationarity=True,enforce_invertibility=True)
            fitted=None
            try:
                for iterations in [500,1000]:
                    kwargs={'method_kwargs':{'maxiter':iterations,'disp':0},'cov_type':'none'}
                    if fitted is not None and np.isfinite(fitted.params).all():
                        kwargs['start_params']=fitted.params
                    with warnings.catch_warnings(record=True) as ws:
                        warnings.simplefilter('always')
                        fitted=model.fit(**kwargs)
                    audit['warnings'].extend(sorted(set(str(w.message) for w in ws)))
                    ret=fitted.mle_retvals
                    audit['attempts'].append({'converged':bool(ret.get('converged',False)),
                                             'iterations':int(ret.get('iterations',0)),
                                             'llf':float(fitted.llf) if np.isfinite(fitted.llf) else None})
                    if ret.get('converged',False):break
                good=bool(fitted.mle_retvals.get('converged',False)) and np.isfinite(fitted.params).all()
                good=good and np.isfinite(fitted.llf) and fitted.params[-1]>0
                if not good:
                    audit['status']='fit_not_converged'
                else:
                    measured=np.isfinite(normalized[:train_end])
                    measured[:int(fitted.loglikelihood_burn)]=False
                    fv=fitted.filter_results.forecasts_error_cov[0,0,:]
                    numerical_ok=np.isfinite(fv[measured]).all() and (fv[measured]>0).all()
                    if not numerical_ok or fitted.llf==0:
                        raise FloatingPointError('Non-positive/non-finite innovation variance or degenerate zero log likelihood')
                    pars=np.asarray(fitted.params)
                    audit['params']=dict(zip(fitted.param_names,map(float,pars)))
                    audit['min_ar_root']=float(np.min(np.abs(fitted.arroots)))
                    audit['min_ma_root']=float(np.min(np.abs(fitted.maroots)))
                    assert audit['min_ma_root']>1-1e-8 and audit['min_ar_root']>1-1e-8
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        full=ARIMA(normalized,order=order,trend='n').filter(pars,cov_type='none')
                    forecasts=forecast_states(full,positions)*scale+anchor
                    obs=np.isfinite(normalized)
                    obs[:int(full.loglikelihood_burn)]=False
                    fv=full.filter_results.forecasts_error_cov[0,0,:]
                    invalid_obs=obs&(~np.isfinite(fv)|(fv<=0))
                    causal_bad=np.maximum.accumulate(invalid_obs)[positions]
                    audit['invalid_forward_observations']=int(invalid_obs.sum())
                    audit['forward_numerical_fallback_days']=int(causal_bad.sum())
                    good_fc=np.isfinite(forecasts).all(axis=1)&(forecasts>0).all(axis=1)&~causal_bad
                    score[good_fc]=forecasts[good_fc,1]/forecasts[good_fc,0]-1.
                    good_fc &= np.isfinite(score)
                    fallback=~good_fc;score[fallback]=0.
                    audit['status']='ok'
                    audit['invalid_forecasts']=int(fallback.sum())
                    # Price-only deterministic audit samples; no performance selection.
                    if code in [1301,7203]:
                        for j in sorted(set([0,len(positions)//2,len(positions)-1])):
                            p=int(positions[j])
                            prefix=ARIMA(normalized[:p+1],order=order,trend='n').filter(pars,cov_type='none')
                            direct=np.asarray(prefix.forecast(2))*scale+anchor
                            np.testing.assert_allclose(direct,forecasts[j],rtol=2e-10,atol=2e-8)
                            audit['prefix_checks'].append({'origin':str(dates[p]),
                                'max_price_error':float(np.max(np.abs(direct-forecasts[j])))})
            except Exception as exc:
                # A failed mathematical check is fatal; numerical model failures are logged.
                if isinstance(exc,AssertionError):raise
                audit['status']='fit_or_filter_exception'
                audit['exception']=f'{type(exc).__name__}: {exc}'
                score[:]=0.;fallback[:]=True;forecasts[:]=np.nan
    audit['fallback_forecasts']=int(fallback.sum())
    audit['forecast_days']=len(positions)
    audit['seconds']=time.monotonic()-t0
    buf=io.BytesIO()
    np.savez_compressed(buf,score=score,fallback=fallback,forecast_prices=forecasts,params=pars)
    return key,audit,buf.getvalue()

def run(workers=6,pilot=False):
    prep=json.loads((RUN/'preparation.json').read_text())
    assert prep['runner_sha256']==sha(__file__) and prep['plan_sha256']==sha(ROOT/'arima_grid_plan.md')
    initialize();con=connect()
    finished=set(con.execute('SELECT oi,yi,ci FROM fits'))
    selected=[int(np.where(DATA['codes']==c)[0][0]) for c in [1301,1332,7203,9984]] if pilot else range(len(DATA['codes']))
    # Small batches across orders keep workloads mixed and expose failures early.
    todo=[(oi,yi,ci) for ci in selected for yi in range(4) for oi in range(25) if (oi,yi,ci) not in finished]
    begun=time.monotonic();done=0;counts={};last_write=begun
    total_initial=len(finished);total=total_initial+len(todo)
    iterator=iter(todo)
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize) as pool:
        pending=set()
        for _ in range(min(workers*3,len(todo))):pending.add(pool.submit(fit_one,next(iterator)))
        while pending:
            complete,pending=wait(pending,timeout=1,return_when=FIRST_COMPLETED)
            for f in complete:
                key,a,payload=f.result()
                con.execute('INSERT INTO fits VALUES (?,?,?,?,?,?,?)',(*key,a['status'],a['seconds'],json.dumps(a,allow_nan=False),payload))
                done+=1;counts[a['status']]=counts.get(a['status'],0)+1
                if done%10==0:con.commit()
                try:pending.add(pool.submit(fit_one,next(iterator)))
                except StopIteration:pass
            now=time.monotonic()
            if now-last_write>=10 or not pending:
                con.commit();elapsed=now-begun
                progress=dict(completed=total_initial+done,total=total,session_completed=done,session_seconds=round(elapsed,1),
                    session_statuses=counts,estimated_remaining_seconds=round((len(todo)-done)*elapsed/done,1) if done else None,
                    mode='pilot' if pilot else 'full')
                save(RUN/'progress.json',progress);print(json.dumps(progress),flush=True);last_write=now
    con.commit();con.execute('PRAGMA wal_checkpoint(TRUNCATE)');con.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','pilot','run']);p.add_argument('--workers',type=int,default=6)
    args=p.parse_args()
    if args.mode=='prepare':prepare()
    else:run(args.workers,args.mode=='pilot')
