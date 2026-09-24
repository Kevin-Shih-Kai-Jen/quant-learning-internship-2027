"""User-specified signed 10% order-statistic bands, with 60/40 annual raw-band smoothing."""
from pathlib import Path
import hashlib
import json
import math
import sys
import zipfile

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
DATA=Path('/Users/coolguy/Desktop/JPX_data/modelling/JPX_data.zip')
EXPERIMENT=Path('/Users/coolguy/Desktop/JPX_data/modelling/JPX-market-regime-experiment.zip')
FRACTION=.10
CURRENT_WEIGHT=.60
INITIAL_UPPER=.001
INITIAL_LOWER=-.001


def digest(stream):
    h=hashlib.sha256()
    while chunk:=stream.read(1024*1024): h.update(chunk)
    return h.hexdigest()


def save(name,obj):
    (ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))


def main():
    plan_hash=hashlib.sha256((ROOT/'experiment_plan.md').read_bytes()).hexdigest()
    source=ROOT/'source';source.mkdir(exist_ok=True)
    with zipfile.ZipFile(EXPERIMENT) as z:
        manifest=json.loads(z.read('jpx_market_regime_results/run_manifest.json'))
        for name in ['jpx_common.py','jpx_models.py','jpx_market_proxy.py','jpx_regime.py','jpx_execution.py']:
            content=z.read(name)
            assert hashlib.sha256(content).hexdigest()==manifest['source_sha256'][name]
            (source/name).write_bytes(content)
        def read(name):return pd.read_csv(z.open(name))
        old=read('jpx_market_regime_results/baseline_replay/daily_returns.csv')
        old_dynamic=read('jpx_market_regime_results/regime/daily_returns.csv')
        saved=read('jpx_hold_same_results/selected_trades.csv')
        benchmark=read('jpx_inputs/nikkei_benchmark.csv')
    sys.path.insert(0,str(source))
    from jpx_common import FEATURES,performance
    from jpx_market_proxy import build_proxy
    from jpx_regime import MarketPriceModel
    from jpx_execution import ORDER_COLUMNS,execute_hold

    hashes={}
    with zipfile.ZipFile(DATA) as z:
        for leaf in ['stock_prices.csv','financials.csv']:
            with z.open('JPX_data/raw/train_files/'+leaf) as f: hashes[leaf]=digest(f)
            assert hashes[leaf]==manifest['source_sha256']['jpx_inputs/'+leaf]
        raw=pd.read_csv(z.open('JPX_data/raw/train_files/stock_prices.csv'),
            usecols=['Date','SecuritiesCode','Open','High','Low','Close','Volume','AdjustmentFactor'],parse_dates=['Date'])
        financials=pd.read_csv(z.open('JPX_data/raw/train_files/financials.csv'),low_memory=False)
    market=build_proxy(raw,financials,ROOT/'market_proxy');del financials
    market['LabelExit']=pd.Series(market.index[2:],index=market.index[:-2])
    market['Target']=market.Close.shift(-2)/market.Close.shift(-1)-1
    interval_cols=['SignalDate','EntryDate','ExitDate','ValidationYear','DailyRankIC','RawDailyRankIC','V2DailyRankIC','EligibleStocks']
    intervals=old[interval_cols].copy()
    for c in ['SignalDate','EntryDate','ExitDate']:intervals[c]=pd.to_datetime(intervals[c])
    signals=intervals[['SignalDate','EntryDate','ExitDate','ValidationYear']].copy()
    signals['MarketPrediction']=np.nan
    fits=[]
    for year in [2018,2019,2020,2021]:
        mask=signals.ValidationYear.eq(year)
        dates=pd.DatetimeIndex(signals.loc[mask,'SignalDate']);cutoff=dates.min()
        train=market.loc[(market.index>=pd.Timestamp('2017-01-01'))&(market.index<=cutoff)
            &market.LabelExit.le(cutoff)&market[FEATURES+['Target']].notna().all(axis=1)]
        assert train.LabelExit.max()<=cutoff
        model=MarketPriceModel().fit(train)
        signals.loc[mask,'MarketPrediction']=model.predict(market.loc[dates]).Prediction.to_numpy()
        fits.append({'validation_year':year,'train_rows':len(train),'fit_asof':str(cutoff.date()),
            'last_training_label_exit':str(train.LabelExit.max().date()),
            'coefficients':dict(zip(['alpha']+FEATURES,map(float,model.theta)))})
    reference=pd.read_csv(ROOT.parent/'jpx_expanding_threshold_20260910'/'expanding_market_signals.csv')
    np.testing.assert_allclose(signals.MarketPrediction,reference.MarketPrediction,rtol=0,atol=1e-12)
    assert signals.MarketPrediction.notna().all()

    bounds=[];previous_raw_upper=INITIAL_UPPER;previous_raw_lower=INITIAL_LOWER
    for year in [2018,2019,2020,2021]:
        mask=signals.ValidationYear.eq(year);cutoff=signals.loc[mask,'SignalDate'].min()
        history=signals.loc[(signals.SignalDate<cutoff)&signals.ExitDate.le(cutoff)].copy()
        rec={'validation_year':year,'asof':str(cutoff.date()),'historical_rows':len(history),
            'previous_raw_upper':previous_raw_upper,'previous_raw_lower':previous_raw_lower}
        if history.empty:
            upper,lower=INITIAL_UPPER,INITIAL_LOWER
            rec.update({'initialization':True,'raw_upper':None,'raw_lower':None,
                        'positive_count':0,'negative_count':0,'positive_selected_count':0,'negative_selected_count':0})
        else:
            pos=history.loc[history.MarketPrediction>0,'MarketPrediction'].sort_values()
            neg=history.loc[history.MarketPrediction<0,'MarketPrediction'].sort_values(ascending=False)
            assert len(pos)>0 and len(neg)>0,'Missing one sign: user must specify a fallback policy'
            kp=math.ceil(FRACTION*len(pos));kn=math.ceil(FRACTION*len(neg))
            raw_upper=float(pos.iloc[kp-1]);raw_lower=float(neg.iloc[kn-1])
            upper=CURRENT_WEIGHT*raw_upper+(1-CURRENT_WEIGHT)*previous_raw_upper
            lower=CURRENT_WEIGHT*raw_lower+(1-CURRENT_WEIGHT)*previous_raw_lower
            rec.update({'initialization':False,'raw_upper':raw_upper,'raw_lower':raw_lower,
                'positive_count':len(pos),'negative_count':len(neg),'zero_count':int(history.MarketPrediction.eq(0).sum()),
                'positive_selected_count':kp,'negative_selected_count':kn,
                'positive_inclusive_count':int(pos.le(raw_upper).sum()),'negative_inclusive_count':int(neg.ge(raw_lower).sum()),
                'history_last_signal':str(history.SignalDate.max().date()),'history_last_exit':str(history.ExitDate.max().date())})
            assert history.ExitDate.max()<=cutoff and history.SignalDate.max()<cutoff
            previous_raw_upper,previous_raw_lower=raw_upper,raw_lower
        assert lower<0<upper
        g=signals.loc[mask,'MarketPrediction']
        reg=np.where(g>upper,1,np.where(g<lower,-1,0))
        signals.loc[mask,'UpperBound']=upper;signals.loc[mask,'LowerBound']=lower;signals.loc[mask,'Regime']=reg
        rec.update({'used_upper':upper,'used_lower':lower,'bull_days':int((reg==1).sum()),
                    'bear_days':int((reg==-1).sum()),'neutral_days':int((reg==0).sum()),'validation_days':len(reg)})
        bounds.append(rec)
    signals.Regime=signals.Regime.astype(int)
    signals.to_csv(ROOT/'market_signals.csv',index=False)
    save('annual_bounds.json',bounds);save('market_fits.json',fits)
    print('Bounds set using only prior history',flush=True)

    orders=saved.copy();orders['ExitDate']=orders.PlannedExitDate;orders['Weight']=orders.RequestedWeight
    for c in ['SignalDate','PreviousSignalDate','EntryDate','ExitDate']:orders[c]=pd.to_datetime(orders[c])
    orders=orders[ORDER_COLUMNS].copy()
    raw=raw.sort_values(['SecuritiesCode','Date'])
    raw['CumulativeFactor']=raw.groupby('SecuritiesCode').AdjustmentFactor.transform(lambda a:a.cumprod().shift(1,fill_value=1.))
    bars=raw.loc[raw.SecuritiesCode.isin(orders.SecuritiesCode.unique())
        &raw.Date.between(intervals.EntryDate.min(),intervals.ExitDate.max()),
        ['Date','SecuritiesCode','Open','High','Low','Close','Volume','CumulativeFactor']].copy();del raw
    benchmark.Date=pd.to_datetime(benchmark.Date);benchmark=benchmark.set_index('Date').Close
    scenarios={};annual=[];reference_error=None
    for name,reg in [('signed_smoothed',signals.Regime),('expanding_no_neutral',np.sign(signals.MarketPrediction)),('baseline_50_50',np.zeros(len(signals)))]:
        selected=orders.copy();r=selected.EntryDate.map(pd.Series(np.asarray(reg),index=signals.EntryDate))
        assert r.notna().all()
        long=np.where(r>0,.7,np.where(r<0,.3,.5));budget=np.where(selected.SourceSide.eq('long'),long,1-long)
        selected.Weight=np.sign(selected.Weight)*budget*selected.SourceRank.map({1:.5,2:.3,3:.2})
        t,d,e=execute_hold(selected,bars,intervals,benchmark)
        np.testing.assert_allclose(d.StrategyReturn,e.groupby('IntervalEntryDate').Contribution.sum().reindex(d.EntryDate).fillna(0),rtol=0,atol=1e-12)
        assert (d.CashWeight>=-1e-12).all() and not t.duplicated(['EntryDate','SecuritiesCode']).any()
        if name=='baseline_50_50':
            reference_error=float(np.max(np.abs(d.StrategyReturn-old.StrategyReturn)))
            np.testing.assert_allclose(d.StrategyReturn,old.StrategyReturn,rtol=0,atol=1e-12)
        out=ROOT/name;out.mkdir(exist_ok=True)
        for leaf,frame in [('daily_returns',d),('selected_trades',t),('position_events',e)]:frame.to_csv(out/(leaf+'.csv'),index=False)
        perf=performance(d.StrategyReturn);idx=performance(d.IndexReturn)
        scenarios[name]={'strategy':perf,'benchmark':idx,
            'mean_long_exposure':float(d.LongExposure.mean()),'mean_short_exposure':float(d.ShortExposure.mean()),
            'both_conditions_met':bool(perf['cumulative_return']>idx['cumulative_return'] and abs(perf['max_drawdown'])<abs(idx['max_drawdown']))}
        if name=='signed_smoothed':
            for year,v in d.groupby('ValidationYear'):
                p=performance(v.StrategyReturn);b=performance(v.IndexReturn)
                annual.append({'year':int(year),'strategy':p,'benchmark':b,
                    'both_conditions_met':bool(p['cumulative_return']>b['cumulative_return'] and abs(p['max_drawdown'])<abs(b['max_drawdown']))})
        print(name,perf['cumulative_return'],abs(perf['max_drawdown']),flush=True)
    result={'plan_sha256':plan_hash,'input_sha256':hashes,'fraction_each_sign':FRACTION,
        'current_weight':CURRENT_WEIGHT,'previous_raw_weight':1-CURRENT_WEIGHT,
        'initial_upper':INITIAL_UPPER,'initial_lower':INITIAL_LOWER,
        'market_fits':fits,'annual_bounds':bounds,'scenarios':scenarios,'annual':annual,
        'original_annual_model_dynamic_reference':performance(old_dynamic.StrategyReturn),
        'baseline_max_daily_return_abs_difference':reference_error,
        'all_checks_passed':True,'test_data_used':False,
        'scope':'Fixed user-specified rule; no parameter search; expanding market model; frozen stock model and original execution; zero costs; development data'}
    save('results.json',result)
    print(json.dumps({'bounds':bounds,'scenarios':scenarios,'annual':annual},ensure_ascii=False,indent=2,allow_nan=False),flush=True)


if __name__=='__main__':main()
