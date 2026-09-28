"""Evaluate complete grid checkpoints, with independent causal and metric audits."""
from pathlib import Path
import io,json,sys,warnings,sqlite3,argparse,time
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from statsmodels.tsa.arima.model import ARIMA
from run_arima_grid import ROOT,RUN,BASE,ORDERS,YEARS,sha,save
sys.path.insert(0,str(BASE/'jpx_official_ranking_20260912'))
from official_metric import calc_spread_return_sharpe


def stats(d):
    s=d.OfficialDailySpread.dropna()
    return {'days':len(d),'scored_days':len(s),'sharpe':float(s.mean()/s.std(ddof=1)),
            'mean_rank_ic':float(d.RankIC.mean()) if d.RankIC.notna().any() else None,
            'rank_ic_days':int(d.RankIC.notna().sum()),'mean_spread':float(s.mean()),
            'std_spread':float(s.std(ddof=1))}


def verify_samples(oi,records,con,data):
    p,q=ORDERS[oi];checks=[]
    for yi,year in enumerate(YEARS):
        good=[(key,a) for key,a in records if key[1]==yi and a['status']=='ok']
        selected=[good[i] for i in np.linspace(0,len(good)-1,8,dtype=int)]
        positions=data['valid_positions'][data['years']==year]
        for key,a in selected:
            payload=con.execute('SELECT payload FROM fits WHERE oi=? AND yi=? AND ci=?',key).fetchone()[0]
            pred=np.load(io.BytesIO(payload));params=pred['params']
            assert len(params)==p+q+1 and np.isfinite(params).all()
            y=(data['prices'][:int(positions[-1])+1,key[2]]-a['anchor'])/a['scale']
            for j in [0,len(positions)//2,len(positions)-1]:
                pos=int(positions[j])
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    f=ARIMA(y[:pos+1],order=(p,1,q),trend='n').filter(params,cov_type='none')
                    expected=np.asarray(f.forecast(2))*a['scale']+a['anchor']
                    modified=y.copy();modified[pos+1:]=5000+173*np.arange(len(y)-pos-1)
                    g=ARIMA(modified,order=(p,1,q),trend='n').filter(params,cov_type='none')
                np.testing.assert_allclose(expected,pred['forecast_prices'][j],rtol=2e-10,atol=2e-8)
                np.testing.assert_allclose(f.filter_results.predicted_state[:,-1],g.filter_results.predicted_state[:,pos+1],rtol=1e-12,atol=1e-12)
                # Validate the fallback using only covariance observations through this origin.
                measured=np.isfinite(y[:pos+1]);measured[:int(f.loglikelihood_burn)]=False
                variance=f.filter_results.forecasts_error_cov[0,0,:]
                bad_cov=bool((measured&(~np.isfinite(variance)|(variance<=0))).any())
                bad_price=not (np.isfinite(expected).all() and (expected>0).all())
                with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
                    score=expected[1]/expected[0]-1 if not bad_price else 0.
                use_fallback=bad_cov or bad_price or not np.isfinite(score)
                if use_fallback:score=0.
                assert bool(pred['fallback'][j])==use_fallback
                np.testing.assert_allclose(score,pred['score'][j],rtol=1e-10,atol=1e-12)
                error=np.abs(expected-pred['forecast_prices'][j]);finite_error=error[np.isfinite(error)]
                checks.append({'year':year,'code':a['code'],'date':str(data['dates'][pos]),
                    'price_error':float(finite_error.max()) if len(finite_error) else 0.,'fallback':use_fallback})
    return {'passed':True,'checks':checks,'count':len(checks),'max_price_error':max(c['price_error'] for c in checks)}


def evaluate_order(oi,con,data,labels,baseline,cal):
    p,q=ORDERS[oi];dest=RUN/'models'/f'p{p}_q{q}';dest.mkdir(parents=True,exist_ok=True)
    n=len(cal);codes=data['codes'];codeindex={int(c):i for i,c in enumerate(codes)}
    score=np.zeros((n,len(codes)));fallback=np.ones(score.shape,bool)
    records=[];params=[]
    for yi,ci,status,seconds,audit,payload in con.execute('SELECT yi,ci,status,seconds,audit,payload FROM fits WHERE oi=? ORDER BY yi,ci',(oi,)):
        a=json.loads(audit);key=(oi,yi,ci);records.append((key,a))
        assert a['train_start']==str(data['dates'][0]) and a['train_last']<a['first_signal']
        positions=data['valid_positions'][data['years']==YEARS[yi]]
        assert a['training_slots']==int(positions[0])
        if status=='ok':
            assert a['attempts'][-1]['converged'] and a['valid_training_prices']>=126
            assert a['params']['sigma2']>0 and a['min_ar_root']>1-1e-8 and a['min_ma_root']>1-1e-8
        z=np.load(io.BytesIO(payload));mask=cal.ValidationYear.to_numpy()==YEARS[yi]
        if status!='ok':
            assert 'params' not in a, 'Exception after accepted training fit requires a separate causal fallback audit'
            assert (z['score']==0).all() and z['fallback'].all()
        score[mask,ci]=z['score'];fallback[mask,ci]=z['fallback']
        assert len(z['score'])==mask.sum() and np.isfinite(z['score']).all()
        params.append({'p':p,'q':q,'Year':YEARS[yi],'SecuritiesCode':int(codes[ci]),'Status':status,**a.get('params',{})})
    assert len(records)==8000
    frame=labels.copy();date_ix=pd.Index(cal.Date).get_indexer(frame.Date);code_ix=frame.SecuritiesCode.map(codeindex).to_numpy()
    frame['Score']=score[date_ix,code_ix];frame['Fallback']=fallback[date_ix,code_ix]
    frame=frame.sort_values(['Date','Score','SecuritiesCode'],ascending=[True,False,True])
    frame['Rank']=frame.groupby('Date').cumcount()
    weights=np.linspace(2,1,200);daily=[]
    for date,g in frame.groupby('Date',sort=True):
        assert np.array_equal(g.Rank.to_numpy(),np.arange(len(g)))
        t=g.Target.to_numpy();s=g.Score.to_numpy();known=np.isfinite(t);fb=g.Fallback.to_numpy()
        selected=np.r_[np.arange(200),np.arange(len(g)-1,len(g)-201,-1)]
        missing=int((~known[selected]).sum())
        spread=(t[:200]@weights-t[-200:][::-1]@weights)/weights.mean() if not missing else np.nan
        ar=rankdata(s[known]);br=rankdata(t[known]);ic=np.corrcoef(ar,br)[0,1] if np.ptp(ar)>0 and np.ptp(br)>0 else np.nan
        daily.append({'Date':date,'ValidationYear':int(g.ValidationYear.iloc[0]),'OfficialDailySpread':spread,'RankIC':ic,
            'SelectedMissingTargets':missing,'FallbackStocks':int(fb.sum()),'SelectedFallbackStocks':int(fb[selected].sum()),
            'DistinctScores':int(g.Score.nunique()),'StocksRanked':len(g)})
    daily=pd.DataFrame(daily)
    valid=daily.SelectedMissingTargets.eq(0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        official=float(calc_spread_return_sharpe(frame.loc[frame.Date.isin(daily.loc[valid,'Date']),['Date','Rank','Target']]))
    metric_error=abs(official-stats(daily.loc[valid])['sharpe']);assert metric_error<1e-12
    sample=verify_samples(oi,records,con,data)
    status_counts=pd.Series([a['status'] for _,a in records]).value_counts().to_dict()
    summary={'p':p,'q':q,'model':f'ARIMA({p},1,{q})','raw_metrics':stats(daily),'fit_statuses':status_counts,
        'retried_fits':sum(len(a['attempts'])>1 for _,a in records),
        'fallback_stock_days':int(frame.Fallback.sum()),'fallback_fraction':float(frame.Fallback.mean()),
        'selected_fallback_stock_days':int(daily.SelectedFallbackStocks.sum()),
        'constant_score_days':int(daily.DistinctScores.eq(1).sum()),'minimum_distinct_scores':int(daily.DistinctScores.min()),
        'total_fit_seconds':sum(a['seconds'] for _,a in records),'official_metric_error':metric_error,
        'prefix_checks':sum(len(a['prefix_checks']) for _,a in records),
        'independent_audit_checks':sample['count'],'independent_audit_max_price_error':sample['max_price_error'],
        'invalid_forward_observations':sum(a.get('invalid_forward_observations',0) for _,a in records),
        'forward_numerical_fallback_days':sum(a.get('forward_numerical_fallback_days',0) for _,a in records),
        'all_training_boundaries_valid':True,'stock_year_fits':len(records),'forecast_rows':len(frame),
        'runner_sha256':sha(ROOT/'run_arima_grid.py'),'plan_sha256':sha(ROOT/'arima_grid_plan.md')}
    daily.to_csv(dest/'daily.csv',index=False)
    pd.DataFrame(params).to_csv(dest/'parameters.csv.gz',index=False,compression='gzip')
    # Compact, aligned predictions: keys shared once; SQLite retains prices and full audits.
    frame=frame.sort_values(['Date','SecuritiesCode'])
    np.savez_compressed(dest/'predictions.npz',score=frame.Score.to_numpy(),rank=frame.Rank.to_numpy(dtype=np.int16),fallback=frame.Fallback.to_numpy())
    save(dest/'audit.json',sample);save(dest/'summary.json',summary)
    print(json.dumps({'evaluated':summary['model'],'raw_sharpe':summary['raw_metrics']['sharpe'],'statuses':status_counts}),flush=True)


def main():
    con=sqlite3.connect(f'file:{RUN}/fits.sqlite?mode=ro',uri=True)
    with np.load(RUN/'prices.npz') as z:data={k:z[k] for k in z.files}
    labels=pd.read_pickle(RUN/'labels.pkl').sort_values(['Date','SecuritiesCode']).reset_index(drop=True)
    baseline=pd.read_pickle(RUN/'baseline.pkl').sort_values(['Date','SecuritiesCode']).reset_index(drop=True)
    pd.testing.assert_frame_equal(labels[['Date','SecuritiesCode']],baseline[['Date','SecuritiesCode']])
    cal=pd.read_csv(RUN/'calendar.csv',parse_dates=['Date'])
    original=pd.read_csv(BASE/'jpx_v7_daily_mse_20260912/v7_equal/daily_spread_returns.csv',parse_dates=['Date'])
    # Independent daily baseline reconstruction.
    merged=labels.merge(baseline[['Date','SecuritiesCode','Rank']],on=['Date','SecuritiesCode'],validate='one_to_one')
    reconstructed=[];w=np.linspace(2,1,200)
    for _,g in merged.groupby('Date',sort=True):
        t=g.sort_values('Rank').Target.to_numpy();reconstructed.append((t[:200]@w-t[-200:][::-1]@w)/w.mean())
    np.testing.assert_allclose(reconstructed,original.OfficialDailySpread,rtol=1e-12,atol=1e-12)
    baseline_daily=pd.read_csv(SOURCE_BASELINE,parse_dates=['Date'])
    pd.testing.assert_series_equal(baseline_daily.Date,cal.Date,check_names=False)
    np.testing.assert_allclose(baseline_daily.OfficialDailySpread,reconstructed,rtol=1e-12,atol=1e-12)
    baseline_daily.to_csv(RUN/'baseline_daily.csv',index=False)
    # These shared keys pair with all per-model prediction arrays.
    np.savez_compressed(RUN/'prediction_keys.npz',date=labels.Date.to_numpy(dtype='datetime64[D]'),code=labels.SecuritiesCode.to_numpy(),target=labels.Target.to_numpy())
    for oi in range(25):
        p,q=ORDERS[oi];dest=RUN/'models'/f'p{p}_q{q}'
        if (dest/'summary.json').exists():continue
        count=con.execute('SELECT COUNT(*) FROM fits WHERE oi=?',(oi,)).fetchone()[0]
        if count!=8000:continue
        evaluate_order(oi,con,data,labels,baseline,cal)
    complete=[(RUN/'models'/f'p{p}_q{q}'/'summary.json').exists() for p,q in ORDERS]
    if not all(complete):
        print('PARTIAL_EVALUATION',sum(complete),'/25; full report waits for every fit',flush=True);return
    common=baseline_daily.OfficialDailySpread.notna().to_numpy(copy=True)
    dailylist=[];models=[]
    for p,q in ORDERS:
        dest=RUN/'models'/f'p{p}_q{q}'
        d=pd.read_csv(dest/'daily.csv',parse_dates=['Date']);dailylist.append(d);common &= d.OfficialDailySpread.notna().to_numpy()
        models.append(json.loads((dest/'summary.json').read_text()))
    assert common.sum()>0
    base=stats(baseline_daily.loc[common]);rows=[];annual=[]
    for m,d in zip(models,dailylist):
        cm=stats(d.loc[common]);m['common_metrics']=cm;m['annual']=[]
        for year in YEARS:
            mask=common&cal.ValidationYear.eq(year).to_numpy();a=stats(d.loc[mask]);b=stats(baseline_daily.loc[mask])
            m['annual'].append({'year':year,'model':a,'v7':b});annual.append({'p':m['p'],'q':m['q'],'Year':year,**a,'v7_sharpe':b['sharpe']})
        rows.append({'p':m['p'],'q':m['q'],'Model':m['model'],**cm,'RawScoredDays':m['raw_metrics']['scored_days'],
            'FallbackFraction':m['fallback_fraction'],'SelectedFallbackStocks':m['selected_fallback_stock_days'],
            'FitOK':m['fit_statuses'].get('ok',0),'InsufficientHistory':m['fit_statuses'].get('insufficient_history',0),
            'OtherFitFailures':8000-m['fit_statuses'].get('ok',0)-m['fit_statuses'].get('insufficient_history',0),
            'PositiveYears':sum(a['model']['sharpe']>0 for a in m['annual']),
            'YearsAboveV7':sum(a['model']['sharpe']>a['v7']['sharpe'] for a in m['annual'])})
    values=np.column_stack([d.loc[common,'OfficialDailySpread'].to_numpy() for d in dailylist]+[baseline_daily.loc[common,'OfficialDailySpread'].to_numpy()])
    observed=values.mean(axis=0)/values.std(axis=0,ddof=1);diff=observed[:-1]-observed[-1]
    rng=np.random.default_rng(20260923);n=len(values);boot=[]
    for _ in range(2000):
        starts=rng.integers(0,n,size=int(np.ceil(n/20)));ix=((starts[:,None]+np.arange(20))%n).ravel()[:n]
        sampled=values[ix];sh=sampled.mean(axis=0)/sampled.std(axis=0,ddof=1);boot.append(sh[:-1]-sh[-1])
    boot=np.asarray(boot);lower,upper=np.quantile(boot,[.025,.975],axis=0)
    radius=float(np.quantile(np.max(np.abs(boot-diff),axis=1),.95))
    for k,row in enumerate(rows):row.update(SharpeMinusV7=float(diff[k]),Marginal95Low=float(lower[k]),Marginal95High=float(upper[k]),GridSimultaneous95Low=float(diff[k]-radius),GridSimultaneous95High=float(diff[k]+radius))
    table=pd.DataFrame(rows).sort_values(['sharpe','mean_rank_ic','p','q'],ascending=[False,False,True,True]).reset_index(drop=True)
    table.insert(0,'GridRank',np.arange(1,len(table)+1));table.to_csv(RUN/'leaderboard.csv',index=False)
    pd.DataFrame(annual).to_csv(RUN/'annual.csv',index=False)
    combined={'complete':True,'models':models,'baseline_common':base,'common_days':int(common.sum()),
        'excluded_dates':cal.loc[~common,'Date'].dt.strftime('%Y-%m-%d').tolist(),'total_fits':200000,'grid_size':25,
        'bootstrap':{'replicates':2000,'block_days':20,'seed':20260923,'simultaneous_centered_max_deviation_radius':radius,
            'scope':'approximate fixed 25-model grid only; does not correct all prior adaptive research'},
        'top_model':{k:(v.item() if isinstance(v,np.generic) else v) for k,v in table.iloc[0].to_dict().items()},'formal_test_used':False,'annualized':False,'costs_included':False,
        'runner_sha256':sha(ROOT/'run_arima_grid.py'),'evaluator_sha256':sha(__file__),'plan_sha256':sha(ROOT/'arima_grid_plan.md'),
        'all_official_metric_checks_passed':True,'independent_causal_checks':sum(m['independent_audit_checks'] for m in models)}
    save(RUN/'results.json',combined);print(json.dumps({'complete':True,'common_days':combined['common_days'],'top_model':combined['top_model']},indent=2),flush=True)

SOURCE_BASELINE=ROOT/'arima012_expanding/baseline_daily_metrics.csv'
if __name__=='__main__':main()
