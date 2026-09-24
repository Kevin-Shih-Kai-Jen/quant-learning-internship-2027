from pathlib import Path
import sys,json,zipfile,hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
PREV=ROOT.parent/'jpx_stock_returns_20260910'
sys.path.insert(0,str(PREV/'source'))
from jpx_common import allocate,performance
from jpx_models import matrix
from jpx_execution import execute_hold,ORDER_COLUMNS
FEATURES=['PR5','PR22','PR60','VR5','VR22','VR60','PR5xVR5','PR22xVR22','PR60xVR60']
MARKET=ROOT.parent/'jpx_signed_band_20260910/market_signals.csv'

def save(name,x): (ROOT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))

def fit_raw(x,y):
    gram=x.T@x/len(x);xy=x.T@y/len(x)
    theta=np.zeros(len(xy));res=xy.copy();direction=res.copy();rr=res@res
    for step in range(1000):
        ad=gram@direction;alpha=rr/(direction@ad)
        theta+=alpha*direction;res-=alpha*ad
        if np.max(np.abs(gram@theta-xy))<1e-14:break
        newrr=res@res;direction=res+(newrr/rr)*direction;rr=newrr
    else:raise RuntimeError('Conjugate gradients did not converge')
    reference=np.linalg.lstsq(x,y,rcond=None)[0]
    np.testing.assert_allclose(theta,reference,rtol=1e-7,atol=1e-9)
    return theta,{'solver':'conjugate gradient; raw features; no preconditioning',
        'iterations':step+1,'gradient_inf_norm':float(np.max(np.abs(2*(gram@theta-xy)))),
        'ols_coefficient_max_error':float(np.max(np.abs(theta-reference))),
        'gram_condition_number':float(np.linalg.cond(gram)),
        'train_mse':float(np.mean((x@theta-y)**2))}

def main():
    f=pd.read_pickle(PREV/'features.pkl')
    signals=pd.read_csv(MARKET,parse_dates=['SignalDate','EntryDate','ExitDate'])
    priorfits=json.loads((PREV/'model_fits.json').read_text())
    prevdaily=pd.read_csv(PREV/'daily_returns.csv',parse_dates=['SignalDate','EntryDate','ExitDate'])
    prevorders=pd.read_csv(PREV/'requested_orders.csv',parse_dates=['SignalDate','EntryDate','ExitDate'])
    with zipfile.ZipFile('/Users/coolguy/Desktop/JPX_data/modelling/JPX-market-regime-experiment.zip') as z:
        benchmark=pd.read_csv(z.open('jpx_inputs/nikkei_benchmark.csv'),parse_dates=['Date']).set_index('Date').Close
    fits=[];orders=[];intervals=[];previous_mask=pd.Series(False,index=f.index)
    for year in (2018,2019,2020,2021):
        s=signals.loc[signals.ValidationYear.eq(year)].copy();cutoff=s.SignalDate.min()
        tm=f.SignalDate.ge('2017-01-01')&f.SignalDate.lt(cutoff)&f.ExitDate.le(cutoff)&f.Target.notna()&f.ReturnEligible
        assert not (previous_mask&~tm).any()
        train=f.loc[tm]
        assert train.ExitDate.max()<=cutoff
        theta,info=fit_raw(matrix(train,FEATURES),train.Target.to_numpy())
        if year==2018:
            assert len(train)==priorfits[0]['train_rows']
            np.testing.assert_allclose(theta,list(priorfits[0]['coefficients'].values()),rtol=0,atol=1e-12)
        v=f.loc[f.SignalDate.isin(s.SignalDate)&f.ReturnEligible].copy()
        assert len(v)==priorfits[year-2018]['validation_rows']
        v['Prediction']=matrix(v,FEATURES)@theta
        v['RawPrediction']=v.Prediction;v['PreviousRawPrediction']=v.Prediction
        v['LagFallback']=False;v['U']=1.;v['ValidationYear']=year
        oldtheta=np.array(list(priorfits[year-2018]['coefficients'].values()))
        oldpred=matrix(v,FEATURES)@oldtheta
        dailyic=[]
        for date,day in v.groupby('SignalDate'):
            sr=s.loc[s.SignalDate.eq(date)].iloc[0]
            long=.7 if sr.Regime==1 else .3 if sr.Regime==-1 else .5
            chosen=allocate(day,(np.array([.5,.3,.2])*long,np.array([.5,.3,.2])*(1-long)))
            orders.append(chosen[ORDER_COLUMNS])
            valid=day.loc[day.Target.notna()]
            ic=valid.Prediction.rank().corr(valid.Target.rank()) if valid.Target.nunique()>1 and valid.Prediction.nunique()>1 else np.nan
            dailyic.append(ic)
            intervals.append({**sr[['SignalDate','EntryDate','ExitDate','ValidationYear']].to_dict(),
                'DailyRankIC':ic,'RawDailyRankIC':ic,'V2DailyRankIC':np.nan,'EligibleStocks':len(day)})
        labeled=v.Target.notna()
        fits.append({'validation_year':year,'fit_asof':str(cutoff.date()),'train_rows':len(train),
            'train_rows_by_signal_year':{str(k):int(n) for k,n in train.groupby(train.SignalDate.dt.year).size().items()},
            'training_first_signal':str(train.SignalDate.min().date()),'training_last_signal':str(train.SignalDate.max().date()),
            'last_training_label_exit':str(train.ExitDate.max().date()),'added_training_rows':int((tm&~previous_mask).sum()),
            'coefficients':dict(zip(['alpha']+FEATURES,map(float,theta))),'fit':info,
            'validation_rows':len(v),'validation_mse':float(np.mean((v.loc[labeled,'Prediction']-v.loc[labeled,'Target'])**2)),
            'previous_year_model_mse_same_rows':float(np.mean((oldpred[labeled]-v.loc[labeled,'Target'])**2)),
            'zero_prediction_mse':float(np.mean(v.loc[labeled,'Target']**2)),
            'mean_daily_rank_ic':float(np.nanmean(dailyic))})
        previous_mask=tm
        print('Fitted',year,'rows',len(train),'IC',fits[-1]['mean_daily_rank_ic'],flush=True)
    save('model_fits.json',fits)
    selected=pd.concat(orders,ignore_index=True);daily=pd.DataFrame(intervals).sort_values('EntryDate')
    selected.to_csv(ROOT/'requested_orders.csv',index=False)
    pd.testing.assert_frame_equal(daily[['SignalDate','EntryDate','ExitDate','ValidationYear','EligibleStocks']].reset_index(drop=True),
        prevdaily[['SignalDate','EntryDate','ExitDate','ValidationYear','EligibleStocks']].reset_index(drop=True),check_dtype=False)
    keys=['SignalDate','SecuritiesCode','SourceSide','SourceRank']
    a=selected.loc[selected.ValidationYear.eq(2018)].sort_values(keys).reset_index(drop=True)
    b=prevorders.loc[prevorders.ValidationYear.eq(2018)].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(a[keys],b[keys],check_dtype=False)
    np.testing.assert_allclose(a[['Prediction','Weight']],b[['Prediction','Weight']],atol=1e-12,rtol=0)
    bars=pd.read_pickle(PREV/'bars.pkl')
    bars=bars.loc[bars.SecuritiesCode.isin(selected.SecuritiesCode.unique())&bars.Date.between(daily.EntryDate.min(),daily.ExitDate.max())]
    t,d,e=execute_hold(selected,bars,daily,benchmark)
    np.testing.assert_allclose(d.StrategyReturn,e.groupby('IntervalEntryDate').Contribution.sum().reindex(d.EntryDate).fillna(0),atol=1e-12,rtol=0)
    np.testing.assert_allclose(d.loc[d.ValidationYear.eq(2018),'StrategyReturn'],prevdaily.loc[prevdaily.ValidationYear.eq(2018),'StrategyReturn'],atol=1e-12,rtol=0)
    np.testing.assert_allclose(d.IndexReturn,prevdaily.IndexReturn,atol=1e-12,rtol=0)
    assert d.CashWeight.ge(-1e-12).all() and not t.duplicated(['EntryDate','SecuritiesCode']).any()
    merged=selected.merge(signals[['SignalDate','Regime']],on='SignalDate',validate='many_to_one')
    long=np.where(merged.Regime.eq(1),.7,np.where(merged.Regime.eq(-1),.3,.5))
    budget=np.where(merged.SourceSide.eq('long'),long,1-long)
    np.testing.assert_allclose(merged.Weight.abs(),budget*merged.SourceRank.map({1:.5,2:.3,3:.2}),atol=1e-14,rtol=0)
    assert (selected.Weight*selected.Prediction>0).all()
    for name,frame in [('selected_trades',t),('daily_returns',d),('position_events',e)]:frame.to_csv(ROOT/(name+'.csv'),index=False)
    priorresults=json.loads((PREV/'results.json').read_text())
    result={'expanding_stock_returns':performance(d.StrategyReturn),'previous_year_stock_returns':performance(prevdaily.StrategyReturn),
        'old_T_reference':priorresults['old_T'],'benchmark':performance(d.IndexReturn),
        'expanding_mean_rank_ic':float(d.DailyRankIC.mean()),'previous_year_mean_rank_ic':float(prevdaily.DailyRankIC.mean()),
        'annual':[{'year':int(y),'expanding':performance(g.StrategyReturn),'previous_year':performance(prevdaily.loc[prevdaily.ValidationYear.eq(y),'StrategyReturn'])} for y,g in d.groupby('ValidationYear')],
        'fits':fits,'checks':{'training_sets_nested':True,'label_availability_checked':True,'same_validation_universe':True,
            'first_year_coefficients_selections_returns_match':True,'same_market_signals_and_budgets':True,'event_pnl_reconciled':True,'no_open_positions_at_end':True},
        'test_used':False,'costs':0,'market_signals_sha256':hashlib.sha256(MARKET.read_bytes()).hexdigest(),
        'plan_sha256':hashlib.sha256((ROOT/'experiment_plan.md').read_bytes()).hexdigest(),
        'all_checks_passed':True}
    assert result['market_signals_sha256']==priorresults['market_signals_sha256']
    save('results.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='fits'},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
