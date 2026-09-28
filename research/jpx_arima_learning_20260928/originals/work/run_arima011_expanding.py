"""JPX ARIMA(0,1,1), expanding annual fits, official t+1 to t+2 horizon.

Run with BLAS threads=1. Forecast workers receive prices and date positions only;
Target is loaded only by prepare/evaluate, never by fit_stock_year.
"""
from pathlib import Path
import os, sys, json, time, warnings, hashlib, zipfile, argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from statsmodels.tsa.arima.model import ARIMA
import statsmodels

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'arima011_expanding'
BASE = Path('/Users/coolguy/.codex/.chatgpt-projects/g-p-6a8ad7dc606481919474d5424342a1c3')
RAW = Path('/Users/coolguy/Desktop/JPX_data/modelling/JPX_data.zip')
ORDER = (0, 1, 1)
WINDOW = None  # Expanding: retain all available history
MIN_OBS = 126


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def prepare():
    RUN.mkdir(exist_ok=True)
    (RUN / 'jobs').mkdir(exist_ok=True)
    cal = pd.read_csv(BASE/'jpx_v5_daily_returns_20260912/daily_spread_returns.csv',
                      usecols=['Date', 'ValidationYear'], parse_dates=['Date'])
    assert len(cal) == 953
    member = 'JPX_data/raw/train_files/stock_prices.csv'
    with zipfile.ZipFile(RAW) as z:
        raw = pd.read_csv(z.open(member), usecols=['Date', 'SecuritiesCode', 'Close',
                          'AdjustmentFactor', 'SupervisionFlag', 'Target'], parse_dates=['Date'])
    raw = raw.sort_values(['SecuritiesCode', 'Date']).reset_index(drop=True)
    raw['FactorBefore'] = raw.groupby('SecuritiesCode').AdjustmentFactor.transform(
        lambda a: a.fillna(1.).cumprod().shift(1, fill_value=1.))
    raw['AdjustedPrice'] = raw.Close/raw.FactorBefore
    old = pd.read_pickle(BASE/'jpx_stock_returns_20260910/features.pkl')
    check = raw.merge(old[['SignalDate','SecuritiesCode','CumulativeFactor','EntryDate','ExitDate']],
                     left_on=['Date','SecuritiesCode'],right_on=['SignalDate','SecuritiesCode'],
                     validate='one_to_one')
    np.testing.assert_allclose(check.FactorBefore, check.CumulativeFactor, rtol=1e-13)
    timeline = check[['Date','EntryDate','ExitDate']].drop_duplicates().sort_values('Date')
    assert timeline.Date.is_unique
    labels = raw.loc[raw.Date.isin(cal.Date) & ~raw.SupervisionFlag,
                     ['Date','SecuritiesCode','Target']].merge(cal,on='Date',validate='many_to_one')
    labels = labels.sort_values(['Date','SecuritiesCode']).reset_index(drop=True)
    assert len(labels) == 1864363 and not labels.duplicated(['Date','SecuritiesCode']).any()
    labels.to_pickle(RUN/'labels.pkl')
    oldranks = pd.concat([pd.read_csv(BASE/f'jpx_v7_daily_mse_20260912/v7_equal/ranks_{y}.csv.gz',
                                    usecols=['Date','SecuritiesCode','g','Rank'],parse_dates=['Date'])
                          for y in [2018,2019,2020,2021]],ignore_index=True)
    oldranks = oldranks.sort_values(['Date','SecuritiesCode']).reset_index(drop=True)
    pd.testing.assert_frame_equal(oldranks[['Date','SecuritiesCode']], labels[['Date','SecuritiesCode']])
    oldranks.to_pickle(RUN/'baseline.pkl')
    wide = raw.loc[raw.Date.le(cal.Date.max())].pivot(index='Date',columns='SecuritiesCode',values='AdjustedPrice')
    assert wide.index.is_monotonic_increasing
    valid_positions = wide.index.get_indexer(cal.Date)
    assert (valid_positions >= 0).all()
    timing = cal.merge(timeline,on='Date',validate='one_to_one')
    # Keep the original t+1/t+2 data calendar, including its documented closure.
    full_dates = pd.DatetimeIndex(sorted(raw.Date.unique()))
    pos = full_dates.get_indexer(timing.Date)
    np.testing.assert_array_equal(full_dates[pos+1], timing.EntryDate)
    np.testing.assert_array_equal(full_dates[pos+2], timing.ExitDate)
    np.savez_compressed(RUN/'prices.npz', prices=wide.to_numpy(), codes=wide.columns.to_numpy(),
                        dates=wide.index.to_numpy(dtype='datetime64[D]'),
                        valid_positions=valid_positions, years=cal.ValidationYear.to_numpy())
    timing.to_csv(RUN/'calendar.csv',index=False)
    save(RUN/'preparation.json', {'raw_member_read':member, 'raw_zip_sha256':sha(RAW),
         'input_price_matrix_shape':list(wide.shape), 'validation_rows':len(labels),
         'validation_days':len(cal), 'stock_count':len(wide.columns),
         'first_signal':str(cal.Date.min().date()),'last_signal':str(cal.Date.max().date()),
         'missing_targets':int(labels.Target.isna().sum()),
         'adjustment_matches_existing':True,'calendar_matches_existing':True,
         'baseline_keys_match':True, 'formal_test_read':False,
         'python':sys.version.split()[0], 'statsmodels':statsmodels.__version__,
         'numpy':np.__version__, 'pandas':pd.__version__,
         'plan_sha256':sha(ROOT/'arima011_expanding_plan.md')})
    print('PREPARED',json.dumps(json.loads((RUN/'preparation.json').read_text())),flush=True)


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


def fit_stock_year(task):
    code, year, start, train_end, positions, y, dates = task
    stem = RUN/'jobs'/f'{year}_{code}'
    t0 = time.monotonic()
    audit = dict(code=int(code), year=int(year), status='pending',
                 train_start=str(dates[0]),train_last=str(dates[train_end-1]),
                 first_signal=str(dates[positions[0]]),last_signal=str(dates[positions[-1]]),
                 training_slots=int(train_end), valid_training_prices=int(np.isfinite(y[:train_end]).sum()),
                 missing_training_prices=int(np.isnan(y[:train_end]).sum()),
                 attempts=[], warnings=[], prefix_checks=[])
    assert train_end <= positions[0] and dates[train_end-1] < dates[positions[0]]
    score = np.zeros(len(positions)); fallback=np.ones(len(positions),dtype=bool)
    forecasts = np.full((len(positions),2),np.nan)
    pars = np.empty(0)
    raw_score=np.zeros(len(positions));one_step_score=np.zeros(len(positions))
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
            model=ARIMA(normalized[:train_end],order=ORDER,trend='n',
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
                    audit['min_ar_root']=None  # No AR term
                    audit['min_ma_root']=float(np.min(np.abs(fitted.maroots)))
                    assert audit['min_ma_root']>1-1e-8
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        full=ARIMA(normalized,order=ORDER,trend='n').filter(pars,cov_type='none')
                    forecasts=forecast_states(full,positions)*scale+anchor
                    good_fc=np.isfinite(forecasts).all(axis=1)&(forecasts>0).all(axis=1)
                    score[good_fc]=forecasts[good_fc,1]/forecasts[good_fc,0]-1.
                    good_fc &= np.isfinite(score)
                    fallback=~good_fc;score[fallback]=0.
                    audit['status']='ok'
                    audit['invalid_forecasts']=int(fallback.sum())
                    # MA(1) increments have zero conditional mean beyond one step.
                    np.testing.assert_allclose(forecasts[:,1],forecasts[:,0],rtol=2e-12,atol=2e-8)
                    raw_score=score.copy()
                    audit['horizon_checks']=len(positions)
                    audit['horizon_max_price_gap']=float(np.max(np.abs(forecasts[:,1]-forecasts[:,0])))
                    audit['raw_score_max_abs']=float(np.max(np.abs(raw_score)))
                    # Use the analytical zero, so numerical roundoff never creates ranks.
                    score[:]=0.
                    valid_current=good_fc&np.isfinite(y[positions])&(y[positions]>0)
                    one_step_score[valid_current]=forecasts[valid_current,0]/y[positions[valid_current]]-1.
                    audit['one_step_nonzero_forecasts']=int(np.count_nonzero(one_step_score))
                    # Price-only deterministic audit samples; no performance selection.
                    if code in [1301,1332,7203,9984]:
                        for j in sorted(set([0,len(positions)//2,len(positions)-1])):
                            p=int(positions[j])
                            prefix=ARIMA(normalized[:p+1],order=ORDER,trend='n').filter(pars,cov_type='none')
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
                raw_score[:]=0.;one_step_score[:]=0.
    audit['fallback_forecasts']=int(fallback.sum())
    audit['forecast_days']=len(positions)
    audit['seconds']=time.monotonic()-t0
    np.savez_compressed(stem.with_suffix('.npz'),score=score,fallback=fallback,
                        forecast_prices=forecasts,params=pars,raw_score=raw_score,one_step_score=one_step_score)
    save(stem.with_suffix('.json'),audit)
    return {'code':int(code),'year':int(year),'status':audit['status'],
            'seconds':round(audit['seconds'],3),'retries':max(0,len(audit['attempts'])-1)}


def tasks(codes_limit=None):
    data=np.load(RUN/'prices.npz')
    prices,codes,dates=data['prices'],data['codes'],data['dates']
    vp,years=data['valid_positions'],data['years']
    for year in sorted(set(years)):
        positions=vp[years==year];boundary=int(positions[0]);start=0
        for k,code in enumerate(codes):
            if codes_limit is not None and code not in codes_limit:continue
            end=int(positions[-1])+1
            yield (int(code),int(year),start,boundary-start,positions-start,
                   prices[start:end,k].copy(),dates[start:end].copy())


def run(workers, pilot=False):
    todo=list(tasks([1301,1332,7203,9984] if pilot else None))
    todo=[t for t in todo if not (RUN/'jobs'/f'{t[1]}_{t[0]}.json').exists()]
    print('START',len(todo),'jobs',workers,'workers',flush=True)
    begun=time.monotonic();counts={};retries=0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending=[pool.submit(fit_stock_year,t) for t in todo]
        for done,job in enumerate(as_completed(pending),1):
            r=job.result();counts[r['status']]=counts.get(r['status'],0)+1;retries+=r['retries']
            if pilot or done%50==0 or done==len(todo):
                progress={'completed':done,'total':len(todo),'elapsed_seconds':round(time.monotonic()-begun,1),
                          'statuses':counts,'retries':retries,'last':r}
                save(RUN/'progress.json',progress)
                print(json.dumps(progress),flush=True)


def daily_metrics(frame, scorecol, rankcol):
    records=[];weights=np.linspace(2.,1.,200)
    for date,g in frame.groupby('Date',sort=True):
        g=g.sort_values(rankcol)
        assert np.array_equal(g[rankcol].to_numpy(),np.arange(len(g)))
        target=g.Target.to_numpy();score=g[scorecol].to_numpy();known=np.isfinite(target)
        selected=np.r_[np.arange(200),np.arange(len(g)-1,len(g)-201,-1)]
        missing=int((~known[selected]).sum())
        spread=None if missing else float((target[:200]@weights-target[-200:][::-1]@weights)/weights.mean())
        a=rankdata(score[known]);b=rankdata(target[known]);ic=None
        if np.ptp(a)>0 and np.ptp(b)>0:ic=float(np.corrcoef(a,b)[0,1])
        fallback=g.Fallback.to_numpy() if 'Fallback' in g else np.zeros(len(g),bool)
        records.append({'Date':date,'ValidationYear':int(g.ValidationYear.iloc[0]),'StocksRanked':len(g),
                        'SelectedMissingTargets':missing,'OfficialDailySpread':spread,'RankIC':ic,
                        'ForecastMSE':float(np.mean((score[known]-target[known])**2)),
                        'FallbackStocks':int(fallback.sum()),'SelectedFallbackStocks':int(fallback[selected].sum())})
    return pd.DataFrame(records)


def stats(d):
    s=d.OfficialDailySpread.dropna()
    return {'days':len(d),'scored_days':len(s),'sharpe':float(s.mean()/s.std(ddof=1)),
            'mean_rank_ic':float(d.RankIC.mean()) if d.RankIC.notna().any() else None,'rank_ic_days':int(d.RankIC.notna().sum()),
            'mean_spread':float(s.mean()),'std_spread':float(s.std(ddof=1)),
            'forecast_mse':float(d.ForecastMSE.mean())}


def evaluate():
    data=np.load(RUN/'prices.npz');codes=data['codes'];years=data['years'];vp=data['valid_positions']
    cal=pd.read_csv(RUN/'calendar.csv',parse_dates=['Date'])
    pieces=[];audits=[]
    for year in sorted(set(years)):
        dates=cal.loc[cal.ValidationYear.eq(year),'Date'].to_numpy()
        for code in codes:
            stem=RUN/'jobs'/f'{year}_{code}'
            a=json.loads(stem.with_suffix('.json').read_text());audits.append(a)
            pred=np.load(stem.with_suffix('.npz'))
            pieces.append(pd.DataFrame({'Date':dates,'SecuritiesCode':code,'Score':pred['score'],
                          'Fallback':pred['fallback'],'ForecastEntryPrice':pred['forecast_prices'][:,0],
                          'ForecastExitPrice':pred['forecast_prices'][:,1],
                          'RawScore':pred['raw_score'],'OneStepScore_DiagnosticOnly':pred['one_step_score']}))
    frame=pd.read_pickle(RUN/'labels.pkl').merge(pd.concat(pieces,ignore_index=True),
                                               on=['Date','SecuritiesCode'],validate='one_to_one')
    assert len(frame)==1864363 and np.isfinite(frame.Score).all()
    frame=frame.sort_values(['Date','Score','SecuritiesCode'],ascending=[True,False,True])
    frame['Rank']=frame.groupby('Date').cumcount()
    frame=frame.sort_values(['Date','SecuritiesCode']).reset_index(drop=True)
    baseline=pd.read_pickle(RUN/'baseline.pkl').merge(frame[['Date','SecuritiesCode','Target','ValidationYear']],
                                                      on=['Date','SecuritiesCode'],validate='one_to_one')
    daily=daily_metrics(frame,'Score','Rank');base=daily_metrics(baseline,'g','Rank')
    assert (frame.Score==0).all()
    assert frame.groupby('Date').Score.nunique().eq(1).all()
    # An all-zero score baseline must have exactly these code-order ranks.
    np.testing.assert_array_equal(frame.Rank,frame.groupby('Date').cumcount())
    olddaily=pd.read_csv(BASE/'jpx_v7_daily_mse_20260912/v7_equal/daily_spread_returns.csv',parse_dates=['Date'])
    np.testing.assert_allclose(base.OfficialDailySpread,olddaily.OfficialDailySpread,rtol=1e-12,atol=1e-12)
    valid=daily.SelectedMissingTargets.eq(0)&base.SelectedMissingTargets.eq(0)
    # Import the saved official formula as an independent full recomputation.
    sys.path.insert(0,str(BASE/'jpx_official_ranking_20260912'))
    from official_metric import calc_spread_return_sharpe
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        official=float(calc_spread_return_sharpe(frame.loc[frame.Date.isin(daily.loc[valid,'Date']),
                                                     ['Date','Rank','Target']]))
    manual=stats(daily.loc[valid])['sharpe']
    np.testing.assert_allclose(official,manual,rtol=0,atol=1e-12)
    summary={'model':'ARIMA(0,1,1), no drift; expanding annual fit, daily filtering; official two-step horizon',
             'window':'expanding from first available date','refit':'annual; fixed parameters within each validation year',
             'score_formula':'forecast_P_t_plus_2 / forecast_P_t_plus_1 - 1','target_interval':'t+1 to t+2 (official JPX)',
             'arima':stats(daily),'v7':stats(base),'common_days':int(valid.sum()),
             'common_arima':stats(daily.loc[valid]),'common_v7':stats(base.loc[valid]),
             'annual':[],'fit_statuses':pd.Series([a['status'] for a in audits]).value_counts().to_dict(),
             'stock_year_fits':len(audits),'retried_fits':sum(len(a['attempts'])>1 for a in audits),
             'fallback_stock_days':int(frame.Fallback.sum()),
             'fallback_fraction':float(frame.Fallback.mean()),
             'selected_fallback_stock_days':int(daily.SelectedFallbackStocks.sum()),
             'forecast_rows':len(frame),'official_formula_max_error':abs(official-manual),
             'prefix_checks':sum(len(a['prefix_checks']) for a in audits),
             'horizon_checks':sum(a.get('horizon_checks',0) for a in audits),
             'horizon_max_price_gap':max(a.get('horizon_max_price_gap',0.) for a in audits),
             'raw_score_max_abs':float(frame.RawScore.abs().max()),
             'one_step_nonzero_stock_days':int(frame.OneStepScore_DiagnosticOnly.ne(0).sum()),
             'all_scores_zero':True,'constant_score_days':int(frame.groupby('Date').Score.nunique().eq(1).sum()),
             'ranks_equal_zero_score_code_order_control':True,
             'metric_interpretation':'Sharpe reflects SecuritiesCode tie-breaking only, not model ranking information',
             'formal_test_used':False,'annualized':False,'costs_included':False,
             'plan_sha256':sha(ROOT/'arima011_expanding_plan.md'),'run_code_sha256':sha(Path(__file__))}
    for year in sorted(set(years)):
        mask=valid&daily.ValidationYear.eq(year)
        summary['annual'].append({'year':int(year),'arima':stats(daily.loc[mask]),'v7':stats(base.loc[mask])})
    frame.to_pickle(RUN/'evaluated_predictions.pkl')
    frame.to_csv(RUN/'predictions.csv.gz',index=False,compression='gzip')
    daily.to_csv(RUN/'daily_metrics.csv',index=False);base.to_csv(RUN/'baseline_daily_metrics.csv',index=False)
    save(RUN/'results.json',summary)
    save(RUN/'fit_audits.json',audits)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','pilot','run','evaluate'])
    p.add_argument('--workers',type=int,default=6);args=p.parse_args()
    if args.mode=='prepare':prepare()
    elif args.mode in ['pilot','run']:run(args.workers,args.mode=='pilot')
    else:evaluate()
