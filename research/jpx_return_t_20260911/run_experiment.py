from pathlib import Path
import sys,json,zipfile,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'jpx_stock_returns_20260910'
MARKET=ROOT.parent/'jpx_signed_band_20260910/market_signals.csv'
sys.path.insert(0,str(BASE/'source'))
from jpx_common import allocate,performance
from jpx_models import matrix
from jpx_execution import execute_hold,ORDER_COLUMNS
from feature_builder import prepare,FEATURES,OLD

def save(name,x): (ROOT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))

def fit_raw(x,y):
    # Reversible numerical column equilibration; coefficients are returned in RAW units.
    scale=np.max(np.abs(x),axis=0);scale=np.where(scale==0,1,scale)
    z=x/scale
    # Use a column-count tolerance so the large row count does not silently
    # discard the weak but nonzero direction created by an extreme ratio.
    rcond=np.finfo(float).eps*x.shape[1]
    gamma,_,rank,singular=np.linalg.lstsq(z,y,rcond=rcond)
    assert rank==x.shape[1],('Design lost rank',rank,singular)
    q,r=np.linalg.qr(z,mode='reduced')
    qr_gamma=np.linalg.solve(r,q.T@y)
    theta=gamma/scale
    predicted=x@theta
    np.testing.assert_allclose(predicted,z@gamma,atol=1e-11,rtol=1e-9)
    np.testing.assert_allclose(predicted,z@qr_gamma,atol=1e-10,rtol=1e-8)
    residual=predicted-y
    gradient=z.T@residual/len(y)
    assert np.max(np.abs(gradient))<1e-10
    return theta,{'solver':'SVD with reversible max-absolute column equilibration; QR cross-check; coefficients restored to raw units',
        'rank':int(rank),'svd_rcond':float(rcond),'equilibrated_condition_number':float(singular[0]/singular[-1]),
        'internal_column_scales':list(map(float,scale)),
        'qr_max_prediction_difference':float(np.max(np.abs(predicted-z@qr_gamma))),
        'equilibrated_normal_residual':float(np.max(np.abs(gradient))),
        'train_mse':float(np.mean(residual**2))}

def main():
    if '--rebuild-features' in sys.argv or not (ROOT/'return_t_features.pkl').exists():prepare()
    f=pd.read_pickle(ROOT/'return_t_features.pkl')
    signals=pd.read_csv(MARKET,parse_dates=['SignalDate','EntryDate','ExitDate'])
    prevfits=json.loads((BASE/'model_fits.json').read_text())
    prevdaily=pd.read_csv(BASE/'daily_returns.csv',parse_dates=['SignalDate','EntryDate','ExitDate'])
    with zipfile.ZipFile('/Users/coolguy/Desktop/JPX_data/modelling/JPX-market-regime-experiment.zip') as z:
        benchmark=pd.read_csv(z.open('jpx_inputs/nikkei_benchmark.csv'),parse_dates=['Date']).set_index('Date').Close
    fits=[];orders=[];intervals=[]
    for year in (2018,2019,2020,2021):
        s=signals.loc[signals.ValidationYear.eq(year)].copy();cutoff=s.SignalDate.min()
        train=f.loc[f.SignalDate.dt.year.eq(year-1)&f.ExitDate.le(cutoff)&f.Target.notna()&f.TEligible]
        assert train.ExitDate.max()<=cutoff and train.SignalDate.max()<cutoff
        theta,info=fit_raw(matrix(train,FEATURES),train.Target.to_numpy())
        v=f.loc[f.SignalDate.isin(s.SignalDate)&f.TEligible].copy()
        assert np.isfinite(v[OLD]).all().all()
        v['Prediction']=matrix(v,FEATURES)@theta
        v['RawPrediction']=v.Prediction;v['PreviousRawPrediction']=v.Prediction
        v['LagFallback']=False;v['U']=1.;v['ValidationYear']=year
        oldpred=matrix(v,OLD)@np.array(list(prevfits[year-2018]['coefficients'].values()))
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
            'training_first_signal':str(train.SignalDate.min().date()),'training_last_signal':str(train.SignalDate.max().date()),
            'last_training_label_exit':str(train.ExitDate.max().date()),
            'coefficients':dict(zip(['alpha']+FEATURES,map(float,theta))),'fit':info,
            'validation_rows':len(v),'previous_validation_rows':prevfits[year-2018]['validation_rows'],
            'validation_mse':float(np.mean((v.loc[labeled,'Prediction']-v.loc[labeled,'Target'])**2)),
            'previous_model_mse_same_rows':float(np.mean((oldpred[labeled]-v.loc[labeled,'Target'])**2)),
            'zero_prediction_mse':float(np.mean(v.loc[labeled,'Target']**2)),
            'prediction_abs_quantiles':{str(k):float(n) for k,n in v.Prediction.abs().quantile([.5,.99,1]).items()},
            'mean_daily_rank_ic':float(np.nanmean(dailyic))})
        print('Fitted',year,json.dumps(fits[-1]),flush=True)
    save('model_fits.json',fits)
    selected=pd.concat(orders,ignore_index=True);daily=pd.DataFrame(intervals).sort_values('EntryDate')
    selected.to_csv(ROOT/'requested_orders.csv',index=False)
    pd.testing.assert_frame_equal(daily[['SignalDate','EntryDate','ExitDate','ValidationYear']].reset_index(drop=True),signals[['SignalDate','EntryDate','ExitDate','ValidationYear']].reset_index(drop=True),check_dtype=False)
    bars=pd.read_pickle(BASE/'bars.pkl')
    bars=bars.loc[bars.SecuritiesCode.isin(selected.SecuritiesCode.unique())&bars.Date.between(daily.EntryDate.min(),daily.ExitDate.max())]
    t,d,e=execute_hold(selected,bars,daily,benchmark)
    np.testing.assert_allclose(d.StrategyReturn,e.groupby('IntervalEntryDate').Contribution.sum().reindex(d.EntryDate).fillna(0),atol=1e-12,rtol=0)
    np.testing.assert_allclose(d.IndexReturn,prevdaily.IndexReturn,atol=1e-12,rtol=0)
    assert d.CashWeight.ge(-1e-12).all() and not t.duplicated(['EntryDate','SecuritiesCode']).any()
    assert (selected.Weight*selected.Prediction>0).all()
    for name,frame in [('selected_trades',t),('daily_returns',d),('position_events',e)]:frame.to_csv(ROOT/(name+'.csv'),index=False)
    priorresults=json.loads((BASE/'results.json').read_text())
    result={'return_t':performance(d.StrategyReturn),'baseline_raw_return':performance(prevdaily.StrategyReturn),
        'old_price_level_T_reference':priorresults['old_T'],'previous_ratio':json.loads((ROOT.parent/'jpx_return_mean_20260911/results.json').read_text())['ratio_plus_pr1'],'benchmark':performance(d.IndexReturn),
        'new_mean_rank_ic':float(d.DailyRankIC.mean()),'baseline_mean_rank_ic':float(prevdaily.DailyRankIC.mean()),
        'annual':[{'year':int(y),'new':performance(g.StrategyReturn),'baseline':performance(prevdaily.loc[prevdaily.ValidationYear.eq(y),'StrategyReturn'])} for y,g in d.groupby('ValidationYear')],
        'fits':fits,'test_used':False,'costs':0,'market_signals_sha256':hashlib.sha256(MARKET.read_bytes()).hexdigest(),
        'plan_sha256':hashlib.sha256((ROOT/'experiment_plan.md').read_bytes()).hexdigest(),'execution_checks_passed':True}
    assert result['market_signals_sha256']==priorresults['market_signals_sha256']
    save('results.json',result)
    print('RESULT',json.dumps({k:v for k,v in result.items() if k!='fits'},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
