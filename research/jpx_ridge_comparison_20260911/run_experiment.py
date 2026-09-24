from pathlib import Path
import sys,json,zipfile,hashlib,gc
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'jpx_stock_returns_20260910'
RT=ROOT.parent/'jpx_return_t_20260911'
sys.path.insert(0,str(ROOT.parent/'jpx_ridge_top10_20260911'))
from run_experiment import choose_lambda,fit_check,predict,LAMBDAS
sys.path.insert(0,str(BASE/'source'))
from jpx_common import allocate,performance
from jpx_execution import execute_hold,ORDER_COLUMNS
EXP=Path('/Users/coolguy/Desktop/JPX_data/modelling/JPX-market-regime-experiment.zip')
DATES=['SignalDate','EntryDate','ExitDate','ValidationYear']

def save(path,x):
    path.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))

def allocate_variant(orders,budget):
    out=orders.copy()
    b=out.SignalDate.map(budget)
    assert b.notna().all()
    out.Weight=np.sign(out.Weight)*np.where(out.SourceSide.eq('long'),b,1-b)*out.SourceRank.map({1:.5,2:.3,3:.2})
    assert out.groupby('SignalDate').Weight.apply(lambda a:a.abs().sum()).le(1+1e-12).all()
    return out

def readcsv(z,path):return pd.read_csv(z.open(path),parse_dates=['SignalDate','EntryDate','ExitDate'])

def fit_family(family,signals,z):
    out=ROOT/family;out.mkdir(exist_ok=True)
    if family=='level_t':
        f=pd.read_pickle(BASE/'features.pkl');refs=json.loads(z.read('jpx_hold_same_results/results.json'))['model'];elig='Eligible'
    else:
        f=pd.read_pickle(RT/'return_t_features.pkl');refs=json.loads((RT/'model_fits.json').read_text());elig='TEligible'
    fits=[];tunes=[];new_orders=[];old_orders=[];timeline=[]
    for year,ref in zip([2018,2019,2020,2021],refs):
        cols=list(ref['coefficients'])[1:]
        s=signals.loc[signals.ValidationYear.eq(year)];cutoff=s.SignalDate.min()
        train=f.loc[f.SignalDate.dt.year.eq(year-1)&f.ExitDate.le(cutoff)&f.Target.notna()&f[elig]]
        assert len(train)==ref['train_rows']
        penalty,tune=choose_lambda(train,cols,cutoff)
        theta,ols,checks=fit_check(train,cols,penalty,ref['coefficients'])
        tune.update(validation_year=year,fit_asof=str(cutoff.date()));tunes.append(tune)
        v=f.loc[f.SignalDate.isin(s.SignalDate)&f[elig]].copy()
        expected=ref['validation_mse']['rows'] if family=='level_t' else ref['validation_rows']
        assert len(v)==expected
        v['Prediction']=predict(v[cols].to_numpy(),theta)
        v['OldPrediction']=predict(v[cols].to_numpy(),ols)
        v['RawPrediction']=v.Prediction;v['PreviousRawPrediction']=v.Prediction
        v['LagFallback']=False;v['U']=1.;v['ValidationYear']=year
        ics=[]
        for date,day in v.groupby('SignalDate'):
            new_orders.append(allocate(day)[ORDER_COLUMNS])
            od=day.copy();od['Prediction']=od.OldPrediction;od['RawPrediction']=od.Prediction;od['PreviousRawPrediction']=od.Prediction
            old_orders.append(allocate(od)[ORDER_COLUMNS])
            valid=day.loc[day.Target.notna()]
            ic=valid.Prediction.rank().corr(valid.Target.rank()) if valid.Target.nunique()>1 and valid.Prediction.nunique()>1 else np.nan
            ics.append(ic)
            row=s.loc[s.SignalDate.eq(date)].iloc[0]
            timeline.append({**row[DATES].to_dict(),'DailyRankIC':ic,'RawDailyRankIC':ic,'V2DailyRankIC':np.nan,'EligibleStocks':len(day)})
        labeled=v.Target.notna()
        fits.append({'validation_year':year,'lambda':penalty,'train_rows':len(train),'validation_rows':len(v),
            'fit_asof':str(cutoff.date()),'last_training_label_exit':str(train.ExitDate.max().date()),
            'coefficients':dict(zip(['alpha']+cols,map(float,theta))),'checks':checks,
            'validation_mse':float(np.mean((v.loc[labeled,'Prediction']-v.loc[labeled,'Target'])**2)),
            'ols_validation_mse':float(np.mean((v.loc[labeled,'OldPrediction']-v.loc[labeled,'Target'])**2)),
            'mean_daily_rank_ic':float(np.nanmean(ics))})
        print('FIT',family,year,'lambda',penalty,'norm ratio',checks['coefficient_l2_ratio'],flush=True)
    save(out/'model_fits.json',fits);save(out/'lambda_selection.json',tunes)
    new=pd.concat(new_orders,ignore_index=True);old=pd.concat(old_orders,ignore_index=True)
    new.to_csv(out/'ranked_orders_50_50.csv',index=False)
    old.to_csv(out/'ols_ranked_orders_50_50.csv',index=False)
    time=pd.DataFrame(timeline).sort_values('EntryDate').reset_index(drop=True)
    time.to_csv(out/'intervals.csv',index=False)
    # Compare original stock picks, ranks and sign transfers independently of initial side budget.
    if family=='level_t':saved=pd.read_csv(z.open('jpx_hold_same_results/selected_trades.csv'),parse_dates=['SignalDate'])
    else:saved=pd.read_csv(RT/'requested_orders.csv',parse_dates=['SignalDate'])
    keys=['SignalDate','SecuritiesCode','SourceSide','SourceRank','Transferred']
    comp=old.merge(saved,on=keys,suffixes=('_new','_old'),validate='one_to_one')
    assert len(comp)==len(old)==len(saved)
    np.testing.assert_allclose(comp.Prediction_new,comp.Prediction_old,atol=1e-9,rtol=0)
    save(out/'old_selection_check.json',{'rows':len(comp),'max_prediction_error':float(abs(comp.Prediction_new-comp.Prediction_old).max())})
    del f,v,train;gc.collect()
    return new,old,time,fits

def run_portfolio(orders,bars,time,benchmark,close):
    subset=bars.loc[bars.SecuritiesCode.isin(orders.SecuritiesCode.unique())&bars.Date.between(time.EntryDate.min(),time.ExitDate.max())]
    t,d,e=execute_hold(orders,subset,time,benchmark,close_only=close)
    np.testing.assert_allclose(d.StrategyReturn,e.groupby('IntervalEntryDate').Contribution.sum().reindex(d.EntryDate).fillna(0),atol=1e-12,rtol=0)
    assert d.CashWeight.ge(-1e-12).all() and not t.duplicated(['EntryDate','SecuritiesCode']).any()
    assert t.loc[t.Executed,'ExecutedWeight'].abs().le(.35+1e-12).all()
    if close:assert not t.Touched.any()
    return t,d,e

def main():
    signed=pd.read_csv(ROOT.parent/'jpx_signed_band_20260910/market_signals.csv',parse_dates=DATES[:3])
    z=zipfile.ZipFile(EXP)
    original=pd.read_csv(z.open('jpx_market_regime_results/market_signals.csv'),parse_dates=DATES[:3])
    pd.testing.assert_frame_equal(original[DATES],signed[DATES])
    benchmark=pd.read_csv(z.open('jpx_inputs/nikkei_benchmark.csv'),parse_dates=['Date']).set_index('Date').Close
    budget={
        'original':pd.Series(np.where(original.Regime.eq(1),.7,.3),index=original.SignalDate),
        'signed':pd.Series(np.where(signed.Regime.eq(1),.7,np.where(signed.Regime.eq(-1),.3,.5)),index=signed.SignalDate),
        'expanding':pd.Series(np.where(signed.MarketPrediction.gt(0),.7,np.where(signed.MarketPrediction.lt(0),.3,.5)),index=signed.SignalDate)}
    for name,b in [('30_70',.3),('50_50',.5),('70_30',.7)]:budget[name]=pd.Series(b,index=signed.SignalDate)
    specs=[
        ('original_dynamic','原始動態配置','level_t','original','regime',False),
        ('return_t_signed','Return T＋1日價格T','return_t','signed',None,False),
        ('fixed_30_70','固定多30%／空70%','level_t','30_70','fixed_30_70',False),
        ('level_t_signed','水準T＋expanding＋中性帶','level_t','signed',None,False),
        ('fixed_50_50','固定多50%／空50%','level_t','50_50','baseline_replay',False),
        ('level_t_expanding','水準T＋expanding無中性帶','level_t','expanding',None,False),
        ('original_dynamic_close','原始動態配置／收盤出場','level_t','original','regime',True),
        ('fixed_30_70_close','固定多30%／空70%／收盤出場','level_t','30_70','fixed_30_70',True),
        ('fixed_50_50_close','固定多50%／空50%／收盤出場','level_t','50_50','baseline_replay',True),
        ('fixed_70_30_close','固定多70%／空30%／收盤出場','level_t','70_30','fixed_70_30',True)]
    family_data={}
    for family in ['level_t','return_t']:
        out=ROOT/family
        if (out/'old_selection_check.json').exists():
            family_data[family]=(pd.read_csv(out/'ranked_orders_50_50.csv',parse_dates=DATES[:3]+['PreviousSignalDate']),pd.read_csv(out/'ols_ranked_orders_50_50.csv',parse_dates=DATES[:3]+['PreviousSignalDate']),pd.read_csv(out/'intervals.csv',parse_dates=DATES[:3]),json.loads((out/'model_fits.json').read_text()))
        else:family_data[family]=fit_family(family,signed,z)
    bars=pd.read_pickle(BASE/'bars.pkl');results=[]
    for key,name,family,bkey,zipref,close in specs:
        new,old,time,fits=family_data[family]
        if zipref:reference=readcsv(z,'jpx_market_regime_results/'+zipref+'/daily_returns.csv')
        elif family=='return_t':reference=pd.read_csv(RT/'daily_returns.csv',parse_dates=DATES[:3])
        else:reference=pd.read_csv(ROOT.parent/'jpx_signed_band_20260910'/('signed_smoothed' if bkey=='signed' else 'expanding_no_neutral')/'daily_returns.csv',parse_dates=DATES[:3])
        pd.testing.assert_frame_equal(time[DATES],reference[DATES])
        oo=allocate_variant(old,budget[bkey]);no=allocate_variant(new,budget[bkey])
        # The original archived level-T runner uses the actual-market interval map.
        # The later return-T baseline retained raw-calendar order dates around the
        # 2020-10-01 closure. Preserve that for exact paired reproduction; audit its
        # corrected-calendar sensitivity separately below.
        if family=='level_t':
            for order in [oo,no]:
                for c in ['EntryDate','ExitDate']:order[c]=order.SignalDate.map(time.set_index('SignalDate')[c])
        _,od,_=run_portfolio(oo,bars,time,benchmark,close)
        expected=reference.CloseOnlySamePositionsReturn if close else reference.StrategyReturn
        error=float(abs(od.StrategyReturn-expected).max())
        np.testing.assert_allclose(od.StrategyReturn,expected,atol=1e-9,rtol=0)
        t,d,e=run_portfolio(no,bars,time,benchmark,close)
        np.testing.assert_allclose(d.IndexReturn,reference.IndexReturn,atol=1e-12,rtol=0)
        out=ROOT/key;out.mkdir(exist_ok=True)
        for leaf,frame in [('requested_orders',no),('selected_trades',t),('daily_returns',d),('position_events',e)]:frame.to_csv(out/(leaf+'.csv'),index=False)
        a=performance(expected);b=performance(d.StrategyReturn)
        r={'key':key,'name':name,'feature_family':family,'close_only':close,'ols':a,'ridge':b,
            'drawdown_reduction_percentage_points':100*(abs(a['max_drawdown'])-abs(b['max_drawdown'])),
            'drawdown_lower':bool(abs(b['max_drawdown'])<abs(a['max_drawdown'])),
            'old_daily_return_reproduction_max_error':error,
            'lambdas':{str(f['validation_year']):f['lambda'] for f in fits},
            'mean_long_exposure':float(d.LongExposure.mean()),'mean_short_exposure':float(d.ShortExposure.mean()),
            'one_sided_days':int(((d.LongExposure.lt(1e-12))|(d.ShortExposure.lt(1e-12))).sum()),
            'old_one_sided_days':int(((od.LongExposure.lt(1e-12))|(od.ShortExposure.lt(1e-12))).sum()),
            'annual':[{'year':int(y),'ridge':performance(g.StrategyReturn),
                'ols':performance(expected.loc[reference.ValidationYear.eq(y)]),
                'mean_long_exposure':float(g.LongExposure.mean()),'mean_short_exposure':float(g.ShortExposure.mean())} for y,g in d.groupby('ValidationYear')]}
        results.append(r);save(out/'results.json',r);save(ROOT/'results.json',{'variants':results,'complete':len(results)==len(specs)})
        print('RESULT',key,'old',a['cumulative_return'],abs(a['max_drawdown']),'ridge',b['cumulative_return'],abs(b['max_drawdown']),'dd reduction pp',r['drawdown_reduction_percentage_points'],flush=True)
        gc.collect()
    save(ROOT/'results.json',{'variants':results,'complete':True,'lambda_grid':LAMBDAS,'benchmark':performance(d.IndexReturn),'all_checks_passed':True,'test_used':False,'costs':0,'stock_ridge_only':True})
    new,old,time,fits=family_data['return_t']
    corrected={}
    for label,orders in [('ols',old),('ridge',new)]:
        order=allocate_variant(orders,budget['signed'])
        for c in ['EntryDate','ExitDate']:order[c]=order.SignalDate.map(time.set_index('SignalDate')[c])
        t,d,e=run_portfolio(order,bars,time,benchmark,False)
        out=ROOT/('return_t_calendar_corrected_'+label);out.mkdir(exist_ok=True)
        for leaf,frame in [('requested_orders',order),('selected_trades',t),('daily_returns',d),('position_events',e)]:frame.to_csv(out/(leaf+'.csv'),index=False)
        corrected[label]=performance(d.StrategyReturn)
    save(ROOT/'return_t_calendar_sensitivity.json',{'description':'Both OLS and Ridge order entry/exit dates remapped to the same actual-market timeline, correcting the inherited 2020-10-01 closure mismatch. Models and training unchanged.',**corrected})
    print('CALENDAR SENSITIVITY',json.dumps(corrected),flush=True)
    paths=[ROOT/'experiment_plan.md',ROOT/'run_experiment.py',BASE/'source/jpx_execution.py',BASE/'source/jpx_common.py',ROOT.parent/'jpx_ridge_top10_20260911/run_experiment.py',ROOT.parent/'jpx_signed_band_20260910/market_signals.csv']
    save(ROOT/'source_hashes.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    print('COMPLETE',len(results),flush=True)

if __name__=='__main__':main()
