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
RAW=['PR5','PR22','PR60','VR5','VR22','VR60']
RATIO=['Q'+c for c in RAW]
PRODUCTS=[f'QPR{k}xQVR{k}' for k in (5,22,60)]
FEATURES=RATIO+PRODUCTS+['PR1']
OLD=RAW+['PR5xVR5','PR22xVR22','PR60xVR60']
def save(name,x): (ROOT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))

def prepare():
    f=pd.read_pickle(BASE/'features.pkl')
    p1=pd.read_pickle(ROOT/'price_return_1d.pkl')
    pd.testing.assert_frame_equal(f[['SecuritiesCode','SignalDate']],p1[['SecuritiesCode','SignalDate']])
    f['PR1']=p1.PR1
    closed=pd.to_datetime(json.loads((BASE/'feature_audit.json').read_text())['price']['market_wide_closures_excluded_from_lookbacks_only'])
    a=f.loc[~f.SignalDate.isin(closed)].copy()
    means=pd.DataFrame(np.nan,index=a.index,columns=RAW)
    for code,indices in a.groupby('SecuritiesCode').groups.items():
        vals=a.loc[indices,RAW].to_numpy(dtype=np.longdouble)
        if len(vals)<=22:continue
        windows=np.lib.stride_tricks.sliding_window_view(vals,22,axis=0)
        # Window starting at j predicts the denominator for row j+22.
        means.loc[indices[22:],RAW]=np.mean(windows[:-1],axis=2,dtype=np.longdouble).astype(float)
    # Near-zero values are CHECKED, never clipped. Recompute from exact decimal
    # source prices, volumes and split factors to distinguish real tiny means
    # from roundoff when positive/negative returns cancel to mathematical zero.
    from fractions import Fraction
    suspicious=means.abs().lt(1e-12)&means.notna()
    exact_checks=[]
    if suspicious.any().any():
        with zipfile.ZipFile('/Users/coolguy/Desktop/JPX_data/modelling/JPX_data.zip') as z:
            raw=pd.read_csv(z.open('JPX_data/raw/train_files/stock_prices.csv'),usecols=['SecuritiesCode','Date','Close','Volume','AdjustmentFactor'],parse_dates=['Date'])
        for code,indices in a.loc[suspicious.any(axis=1)].groupby('SecuritiesCode').groups.items():
            bars=raw.loc[raw.SecuritiesCode.eq(code)&~raw.Date.isin(closed)].sort_values('Date').set_index('Date')
            cumulative=[];factor=Fraction(1)
            for x in bars.AdjustmentFactor:
                cumulative.append(factor);factor*=Fraction(str(x)) if pd.notna(x) else Fraction(1)
            for idx in indices:
                t=bars.index.get_loc(a.loc[idx,'SignalDate'])
                for col in RAW:
                    if not suspicious.loc[idx,col]:continue
                    k=int(col[2:]);vals=[]
                    for j in range(t-22,t):
                        now=bars.iloc[j];before=bars.iloc[j-k]
                        adjustment=cumulative[j]/cumulative[j-k]
                        if col.startswith('PR'):
                            value=Fraction(str(now.Close))/(Fraction(str(before.Close))*adjustment)-1
                        else:value=Fraction(str(now.Volume))*adjustment/Fraction(str(before.Volume))-1
                        vals.append(value)
                    exact=sum(vals,Fraction(0))/22
                    exact_checks.append({'security':int(code),'signal_date':str(a.loc[idx,'SignalDate'].date()),'feature':col,
                        'floating_mean':float(means.loc[idx,col]),'exact_mean_fraction':str(exact),'exact_zero':exact==0})
                    means.loc[idx,col]=float(exact)
    denominator_stats={}
    for col,q in zip(RAW,RATIO):
        den=means[col]
        ratio=a[col]/den.where(den.ne(0))-1
        f[q]=ratio
        f['Mean22_'+col]=den
        denominator_stats[col]={'exact_zero':int(den.eq(0).sum()),'missing':int(den.isna().sum()),
            'negative':int(den.lt(0).sum()),'nonzero_abs_below_1e_12':int((den.abs().gt(0)&den.abs().lt(1e-12)).sum()),
            'min_nonzero_abs':float(den.loc[den.ne(0)].abs().min()),'max_abs_ratio':float(ratio.abs().max())}
    cm=f.SignalDate.isin(closed)
    cols=RATIO+['Mean22_'+c for c in RAW]
    f.loc[cm,cols]=f.groupby('SecuritiesCode')[cols].shift(1).loc[cm]
    for k in (5,22,60):f[f'QPR{k}xQVR{k}']=f[f'QPR{k}']*f[f'QVR{k}']
    f['RatioEligible']=np.isfinite(f[FEATURES]).all(axis=1)&f.Close.gt(0)&f.Volume.gt(0)
    f.loc[cm,'RatioEligible']=f.groupby('SecuritiesCode').RatioEligible.shift(1,fill_value=False).loc[cm]
    keep=['SecuritiesCode','SignalDate','PreviousSignalDate','EntryDate','ExitDate','Target','RatioEligible','ReturnEligible']+FEATURES+OLD+['Mean22_'+c for c in RAW]
    f=f[keep]
    f.to_pickle(ROOT/'ratio_features.pkl')
    save('feature_audit.json',{'denominators':denominator_stats,'rows':len(f),'eligible':int(f.RatioEligible.sum()),
        'features':FEATURES,'mean_window':22,'mean_excludes_current':True,'zero_policy':'undefined; omit candidate','nonzero_floor':None,
        'near_zero_exact_source_checks':exact_checks})
    print('Prepared features',json.dumps(denominator_stats),flush=True)

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
    if '--rebuild-features' in sys.argv or not (ROOT/'ratio_features.pkl').exists():prepare()
    f=pd.read_pickle(ROOT/'ratio_features.pkl')
    signals=pd.read_csv(MARKET,parse_dates=['SignalDate','EntryDate','ExitDate'])
    prevfits=json.loads((BASE/'model_fits.json').read_text())
    prevdaily=pd.read_csv(BASE/'daily_returns.csv',parse_dates=['SignalDate','EntryDate','ExitDate'])
    with zipfile.ZipFile('/Users/coolguy/Desktop/JPX_data/modelling/JPX-market-regime-experiment.zip') as z:
        benchmark=pd.read_csv(z.open('jpx_inputs/nikkei_benchmark.csv'),parse_dates=['Date']).set_index('Date').Close
    fits=[];orders=[];intervals=[]
    for year in (2018,2019,2020,2021):
        s=signals.loc[signals.ValidationYear.eq(year)].copy();cutoff=s.SignalDate.min()
        train=f.loc[f.SignalDate.dt.year.eq(year-1)&f.ExitDate.le(cutoff)&f.Target.notna()&f.RatioEligible]
        assert train.ExitDate.max()<=cutoff and train.SignalDate.max()<cutoff
        theta,info=fit_raw(matrix(train,FEATURES),train.Target.to_numpy())
        v=f.loc[f.SignalDate.isin(s.SignalDate)&f.RatioEligible].copy()
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
    result={'ratio_plus_pr1':performance(d.StrategyReturn),'baseline_raw_return':performance(prevdaily.StrategyReturn),
        'old_T_reference':priorresults['old_T'],'benchmark':performance(d.IndexReturn),
        'new_mean_rank_ic':float(d.DailyRankIC.mean()),'baseline_mean_rank_ic':float(prevdaily.DailyRankIC.mean()),
        'annual':[{'year':int(y),'new':performance(g.StrategyReturn),'baseline':performance(prevdaily.loc[prevdaily.ValidationYear.eq(y),'StrategyReturn'])} for y,g in d.groupby('ValidationYear')],
        'fits':fits,'test_used':False,'costs':0,'market_signals_sha256':hashlib.sha256(MARKET.read_bytes()).hexdigest(),
        'plan_sha256':hashlib.sha256((ROOT/'experiment_plan.md').read_bytes()).hexdigest(),'execution_checks_passed':True}
    assert result['market_signals_sha256']==priorresults['market_signals_sha256']
    save('results.json',result)
    print('RESULT',json.dumps({k:v for k,v in result.items() if k!='fits'},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
