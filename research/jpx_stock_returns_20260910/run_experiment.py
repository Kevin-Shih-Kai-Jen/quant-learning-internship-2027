from pathlib import Path
import sys,json,zipfile,hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))
from jpx_common import build_features,gd_fit,allocate,performance
from jpx_volume_features import add_volume_features
from jpx_models import matrix
from jpx_execution import execute_hold,ORDER_COLUMNS
DATA=Path('/Users/coolguy/Desktop/JPX_data/modelling/JPX_data.zip')
EXP=Path('/Users/coolguy/Desktop/JPX_data/modelling/JPX-market-regime-experiment.zip')
OLD=['T5','T22','T60','V5','V22','V60','P5xV5','P22xV22','P60xV60']
NEW=['PR5','PR22','PR60','VR5','VR22','VR60','PR5xVR5','PR22xVR22','PR60xVR60']

def save(name,x): (ROOT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))

def features():
    with zipfile.ZipFile(DATA) as z:
        member='JPX_data/raw/train_files/stock_prices.csv'
        h=hashlib.sha256()
        with z.open(member) as f:
            while chunk:=f.read(1024*1024):h.update(chunk)
        assert h.hexdigest()=='bf774a86f834e5338bba74f2356ebb750f93a1cc4e89b064ee07251e8ed851db'
        raw=pd.read_csv(z.open(member),parse_dates=['Date'])
    raw=raw.sort_values(['SecuritiesCode','Date'])
    raw['CumulativeFactor']=raw.groupby('SecuritiesCode').AdjustmentFactor.transform(lambda x:x.fillna(1.).cumprod().shift(1,fill_value=1.))
    old,cal,audit=build_features(raw)
    closed=pd.to_datetime(audit['market_wide_closures_excluded_from_lookbacks_only'])
    old,va=add_volume_features(old,raw,closed)
    old=old.merge(raw[['SecuritiesCode','Date','CumulativeFactor']],left_on=['SecuritiesCode','SignalDate'],right_on=['SecuritiesCode','Date'],validate='one_to_one').drop(columns='Date')
    a=old.loc[~old.SignalDate.isin(closed)].copy()
    a['AdjustedPrice']=a.Close/a.CumulativeFactor
    a['AdjustedVolume']=a.Volume*a.CumulativeFactor
    g=a.groupby('SecuritiesCode')
    for k in (5,22,60):
        for src,dst in [('AdjustedPrice','PR'),('AdjustedVolume','VR')]:
            denom=g[src].shift(k)
            a[f'{dst}{k}']=a[src]/denom.where(denom>0)-1
            old[f'{dst}{k}']=a[f'{dst}{k}']
    cols=NEW[:6];cm=old.SignalDate.isin(closed)
    old.loc[cm,cols]=old.groupby('SecuritiesCode')[cols].shift(1).loc[cm]
    for k in (5,22,60):old[f'PR{k}xVR{k}']=old[f'PR{k}']*old[f'VR{k}']
    old['ReturnEligible']=np.isfinite(old[NEW]).all(axis=1)&old.Close.gt(0)&old.Volume.gt(0)
    old.loc[cm,'ReturnEligible']=old.groupby('SecuritiesCode').ReturnEligible.shift(1,fill_value=False).loc[cm]
    prev=pd.Series(cal[:-1],index=cal[1:])
    old['PreviousSignalDate']=old.SignalDate.map(prev)
    old.to_pickle(ROOT/'features.pkl')
    raw[['Date','SecuritiesCode','Open','High','Low','Close','Volume','CumulativeFactor']].to_pickle(ROOT/'bars.pkl')
    save('feature_audit.json',{'price':audit,'volume':va,'raw_sha256':h.hexdigest(),'rows':len(old),'new_eligible':int(old.ReturnEligible.sum()),'old_eligible':int(old.Eligible.sum())})
    print('Features built',len(old),flush=True)

def main():
    if not (ROOT/'features.pkl').exists():features()
    f=pd.read_pickle(ROOT/'features.pkl')
    signals=pd.read_csv(ROOT.parent/'jpx_signed_band_20260910/market_signals.csv',parse_dates=['SignalDate','EntryDate','ExitDate'])
    with zipfile.ZipFile(EXP) as z:
        oldresults=json.loads(z.read('jpx_hold_same_results/results.json'))
        saved=pd.read_csv(z.open('jpx_hold_same_results/selected_trades.csv'),parse_dates=['SignalDate','EntryDate'])
        benchmark=pd.read_csv(z.open('jpx_inputs/nikkei_benchmark.csv'),parse_dates=['Date']).set_index('Date').Close
    fits=[];orders=[];intervals=[];reproduction=[]
    for year in (2018,2019,2020,2021):
        s=signals.loc[signals.ValidationYear.eq(year)].copy();cutoff=s.SignalDate.min()
        tm=f.SignalDate.dt.year.eq(year-1)&f.ExitDate.le(cutoff)&f.Target.notna()
        oldtrain=f.loc[tm&f.Eligible]
        ref=oldresults['model'][year-2018]
        print('Old train rows',year,len(oldtrain),'expected',ref['train_rows'],flush=True)
        assert len(oldtrain)==ref['train_rows']
        theta,info=gd_fit(matrix(oldtrain,OLD),oldtrain.Target.to_numpy())
        err=float(np.max(np.abs(theta-np.array(list(ref['coefficients'].values())))))
        assert err<1e-9
        val=f.loc[f.SignalDate.isin(s.SignalDate)&f.Eligible].copy()
        val['Prediction']=matrix(val,OLD)@theta
        oldorders=pd.concat([allocate(day) for _,day in val.groupby('SignalDate')])
        keys=['SignalDate','SecuritiesCode','SourceSide','SourceRank']
        rr=saved.loc[saved.EntryDate.dt.year.eq(year)]
        # Saved trades include every scheduled slot, including entries held as cash.
        assert len(oldorders)==len(rr),(len(oldorders),len(rr))
        comp=oldorders.merge(rr,on=keys,suffixes=('_new','_ref'),validate='one_to_one')
        assert len(comp)==len(rr)
        pe=float(np.max(np.abs(comp.Prediction_new-comp.Prediction_ref)))
        assert pe<1e-9
        reproduction.append({'year':year,'train_rows':len(oldtrain),'coefficient_max_error':err,'selection_rows':len(comp),'prediction_max_error':pe})
        train=f.loc[tm&f.ReturnEligible]
        x=matrix(train,NEW);y=train.Target.to_numpy()
        try:
            nt,ni=gd_fit(x,y);ni['solver']='original full-batch GD'
        except (RuntimeError,AssertionError):
            # Conjugate gradients on the unchanged raw-feature normal equations.
            gram=x.T@x/len(x);xy=x.T@y/len(x)
            nt=np.zeros(len(xy));res=xy.copy();direction=res.copy();rr=res@res
            for step in range(1000):
                ad=gram@direction;alpha=rr/(direction@ad)
                nt+=alpha*direction;res-=alpha*ad
                if np.max(np.abs(gram@nt-xy))<1e-14:break
                newrr=res@res;direction=res+(newrr/rr)*direction;rr=newrr
            else:raise RuntimeError('Conjugate gradients did not converge')
            reference=np.linalg.lstsq(x,y,rcond=None)[0]
            np.testing.assert_allclose(nt,reference,rtol=1e-7,atol=1e-9)
            ni={'solver':'conjugate gradient; raw features; no preconditioning','iterations':step+1,
                'gradient_inf_norm':float(np.max(np.abs(2*(gram@nt-xy)))),
                'ols_coefficient_max_error':float(np.max(np.abs(nt-reference))),
                'gram_condition_number':float(np.linalg.cond(gram)),
                'train_mse':float(np.mean((x@nt-y)**2))}
        nv=f.loc[f.SignalDate.isin(s.SignalDate)&f.ReturnEligible].copy()
        nv['Prediction']=matrix(nv,NEW)@nt
        nv['RawPrediction']=nv.Prediction;nv['PreviousRawPrediction']=nv.Prediction
        nv['LagFallback']=False;nv['U']=1.;nv['ValidationYear']=year
        common=nv.merge(val[['SignalDate','SecuritiesCode','Prediction']],on=['SignalDate','SecuritiesCode'],suffixes=('','_old'),validate='one_to_one')
        common=common.loc[common.Target.notna()]
        dailyic=[]
        for date,day in nv.groupby('SignalDate'):
            sr=s.loc[s.SignalDate.eq(date)].iloc[0]
            long=.7 if sr.Regime==1 else .3 if sr.Regime==-1 else .5
            chosen=allocate(day,(np.array([.5,.3,.2])*long,np.array([.5,.3,.2])*(1-long)))
            orders.append(chosen[ORDER_COLUMNS])
            icday=day.loc[day.Target.notna()]
            ic=icday.Prediction.rank().corr(icday.Target.rank());dailyic.append(ic)
            intervals.append({**sr[['SignalDate','EntryDate','ExitDate','ValidationYear']].to_dict(),
                'DailyRankIC':ic,'RawDailyRankIC':ic,'V2DailyRankIC':np.nan,'EligibleStocks':len(day)})
        labeled=nv.loc[nv.Target.notna()]
        fits.append({'validation_year':year,'fit_asof':str(cutoff.date()),'train_rows':len(train),
            'last_training_label_exit':str(train.ExitDate.max().date()),'fit':ni,
            'coefficients':dict(zip(['alpha']+NEW,map(float,nt))),
            'validation_rows':len(nv),'old_validation_rows':len(val),
            'new_only_validation_rows':len(nv)-len(common),'old_only_validation_rows':len(val)-len(common),
            'validation_mse':float(np.mean((labeled.Prediction-labeled.Target)**2)),
            'validation_zero_mse':float(np.mean(labeled.Target**2)),
            'common_rows':len(common),'common_new_mse':float(np.mean((common.Prediction-common.Target)**2)),
            'common_old_mse':float(np.mean((common.Prediction_old-common.Target)**2)),
            'mean_daily_rank_ic':float(np.nanmean(dailyic)),
            'validation_max_abs_prediction':float(nv.Prediction.abs().max())})
        print('Fit done',year,fits[-1],flush=True)
    save('model_fits.json',fits);save('old_reproduction.json',reproduction)
    selected=pd.concat(orders,ignore_index=True);daily=pd.DataFrame(intervals).sort_values('EntryDate')
    selected.to_csv(ROOT/'requested_orders.csv',index=False)
    bars=pd.read_pickle(ROOT/'bars.pkl')
    bars=bars.loc[bars.SecuritiesCode.isin(selected.SecuritiesCode.unique())&bars.Date.between(daily.EntryDate.min(),daily.ExitDate.max())]
    t,d,e=execute_hold(selected,bars,daily,benchmark)
    np.testing.assert_allclose(d.StrategyReturn,e.groupby('IntervalEntryDate').Contribution.sum().reindex(d.EntryDate).fillna(0),atol=1e-12,rtol=0)
    assert d.CashWeight.ge(-1e-12).all() and not t.duplicated(['EntryDate','SecuritiesCode']).any()
    for name,frame in [('selected_trades',t),('daily_returns',d),('position_events',e)]:frame.to_csv(ROOT/(name+'.csv'),index=False)
    baseline=pd.read_csv(ROOT.parent/'jpx_signed_band_20260910/signed_smoothed/daily_returns.csv')
    np.testing.assert_allclose(d.IndexReturn,baseline.IndexReturn,rtol=0,atol=1e-12)
    result={'new_returns':performance(d.StrategyReturn),'old_T':performance(baseline.StrategyReturn),'benchmark':performance(d.IndexReturn),
        'new_mean_rank_ic':float(d.DailyRankIC.mean()),'old_mean_rank_ic':float(baseline.DailyRankIC.mean()),
        'annual':[{'year':int(y),'new_returns':performance(g.StrategyReturn),
            'old_T':performance(baseline.loc[baseline.ValidationYear.eq(y),'StrategyReturn']),
            'benchmark':performance(g.IndexReturn)} for y,g in d.groupby('ValidationYear')],
        'fits':fits,'old_reproduction':reproduction,'all_checks_passed':True,'test_used':False,'costs':0,
        'market_signals_sha256':hashlib.sha256((ROOT.parent/'jpx_signed_band_20260910/market_signals.csv').read_bytes()).hexdigest(),
        'plan_sha256':hashlib.sha256((ROOT/'experiment_plan.md').read_bytes()).hexdigest()}
    save('results.json',result)
    print('RESULT',json.dumps({k:v for k,v in result.items() if k not in ['fits','old_reproduction']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
