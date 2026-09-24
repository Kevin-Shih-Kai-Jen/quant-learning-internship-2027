from pathlib import Path
import sys,json,zipfile,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
MODEL=ROOT.parent/'jpx_return_t_20260911'
BASE=ROOT.parent/'jpx_stock_returns_20260910'
MARKET=ROOT.parent/'jpx_signed_band_20260910/market_signals.csv'
sys.path.insert(0,str(BASE/'source'))
from jpx_common import allocate,performance
from jpx_models import matrix
from jpx_execution import execute_hold,ORDER_COLUMNS
from allocation import allocate_slots

def save(name,obj): (ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))

def main():
    f=pd.read_pickle(MODEL/'return_t_features.pkl')
    fits=json.loads((MODEL/'model_fits.json').read_text())
    signals=pd.read_csv(MARKET,parse_dates=['SignalDate','EntryDate','ExitDate'])
    olddaily=pd.read_csv(MODEL/'daily_returns.csv',parse_dates=['SignalDate','EntryDate','ExitDate'])
    oldorders=pd.read_csv(MODEL/'requested_orders.csv',parse_dates=['SignalDate','EntryDate','ExitDate'])
    with zipfile.ZipFile('/Users/coolguy/Desktop/JPX_data/modelling/JPX-market-regime-experiment.zip') as z:
        benchmark=pd.read_csv(z.open('jpx_inputs/nikkei_benchmark.csv'),parse_dates=['Date']).set_index('Date').Close
    orders=[];oldreplay=[];audit=[]
    key=['SignalDate','SecuritiesCode','SourceSide','SourceRank']
    for fit in fits:
        year=fit['validation_year'];cols=list(fit['coefficients'])[1:]
        dates=signals.loc[signals.ValidationYear.eq(year),'SignalDate']
        v=f.loc[f.SignalDate.isin(dates)&f.TEligible].copy()
        assert len(v)==fit['validation_rows']
        v['Prediction']=matrix(v,cols)@np.array(list(fit['coefficients'].values()))
        v['RawPrediction']=v.Prediction;v['PreviousRawPrediction']=v.Prediction
        v['LagFallback']=False;v['U']=1.;v['ValidationYear']=year
        year_old=[]
        for date,day in v.groupby('SignalDate'):
            sr=signals.loc[signals.SignalDate.eq(date)].iloc[0]
            long=.7 if sr.Regime==1 else .3 if sr.Regime==-1 else .5
            legacy_weights=(np.array([.5,.3,.2])*long,np.array([.5,.3,.2])*(1-long))
            legacy=allocate(day,legacy_weights)
            generalized=allocate_slots(day,legacy_weights)
            pd.testing.assert_frame_equal(legacy[ORDER_COLUMNS],generalized[ORDER_COLUMNS])
            year_old.append(legacy[ORDER_COLUMNS])
            weights=np.array([np.full(10,long/10),np.full(10,(1-long)/10)])
            selected=allocate_slots(day,weights)
            assert selected.SourceRank.between(1,10).all()
            assert selected.Weight.abs().max()<=.07+1e-12
            orders.append(selected[ORDER_COLUMNS])
        regenerated=pd.concat(year_old,ignore_index=True).sort_values(key).reset_index(drop=True)
        reference=oldorders.loc[oldorders.ValidationYear.eq(year)].sort_values(key).reset_index(drop=True)
        pd.testing.assert_frame_equal(regenerated[key+['Transferred']],reference[key+['Transferred']],check_dtype=False)
        error=float(np.max(np.abs(regenerated.Prediction-reference.Prediction)))
        np.testing.assert_allclose(regenerated[['Weight','Prediction']],reference[['Weight','Prediction']],atol=1e-12,rtol=0)
        oldreplay.append(regenerated)
        audit.append({'year':year,'unchanged_candidate_rows':len(v),'original_orders_matched':len(regenerated),'max_prediction_difference':error})
        print('Frozen model and old allocation verified',year,len(v),flush=True)
    selected=pd.concat(orders,ignore_index=True)
    selected.to_csv(ROOT/'requested_orders.csv',index=False)
    timeline=olddaily[['SignalDate','EntryDate','ExitDate','ValidationYear','DailyRankIC','RawDailyRankIC','V2DailyRankIC','EligibleStocks']].copy()
    pd.testing.assert_frame_equal(timeline[['SignalDate','EntryDate','ExitDate','ValidationYear']],signals[['SignalDate','EntryDate','ExitDate','ValidationYear']],check_dtype=False)
    bars=pd.read_pickle(BASE/'bars.pkl')
    bars=bars.loc[bars.SecuritiesCode.isin(selected.SecuritiesCode.unique())&bars.Date.between(timeline.EntryDate.min(),timeline.ExitDate.max())]
    trades,daily,events=execute_hold(selected,bars,timeline,benchmark)
    np.testing.assert_allclose(daily.StrategyReturn,events.groupby('IntervalEntryDate').Contribution.sum().reindex(daily.EntryDate).fillna(0),atol=1e-12,rtol=0)
    np.testing.assert_allclose(daily.IndexReturn,olddaily.IndexReturn,atol=1e-12,rtol=0)
    assert daily.CashWeight.ge(-1e-12).all() and not trades.duplicated(['EntryDate','SecuritiesCode']).any()
    assert trades.loc[trades.Executed,'ExecutedWeight'].abs().le(.07+1e-12).all()
    # Replaying the verified old orders through the same engine checks that the
    # comparison is unaffected by execution, prices, or time boundaries.
    _,oldcheck,_=execute_hold(pd.concat(oldreplay,ignore_index=True),bars,timeline,benchmark)
    np.testing.assert_allclose(oldcheck.StrategyReturn,olddaily.StrategyReturn,atol=1e-12,rtol=0)
    for name,frame in [('selected_trades',trades),('daily_returns',daily),('position_events',events)]:frame.to_csv(ROOT/(name+'.csv'),index=False)
    o=selected.merge(signals[['SignalDate','Regime']],on='SignalDate',validate='many_to_one')
    long=np.where(o.Regime.eq(1),.7,np.where(o.Regime.eq(-1),.3,.5))
    budget=np.where(o.SourceSide.eq('long'),long,1-long)
    np.testing.assert_allclose(o.Weight.abs(),budget/10,atol=1e-14,rtol=0)
    prior=json.loads((MODEL/'results.json').read_text())
    result={'top10_equal':performance(daily.StrategyReturn),'top3_original':performance(olddaily.StrategyReturn),'benchmark':performance(daily.IndexReturn),
        'annual':[{'year':int(y),'top10':performance(g.StrategyReturn),'top3':performance(olddaily.loc[olddaily.ValidationYear.eq(y),'StrategyReturn'])} for y,g in daily.groupby('ValidationYear')],
        'mean_daily_rank_ic_unchanged':float(daily.DailyRankIC.mean()),
        'allocation_audit':{'slots_each_side':10,'within_side_equal':True,'max_requested_slot_weight':float(selected.Weight.abs().max()),
            'old_max_requested_slot_weight':float(oldorders.Weight.abs().max()),'requested_positions':len(selected),'executed_positions':int(trades.Executed.sum()),
            'delayed_positions':int(trades.DelaySessions.gt(0).sum()),'transferred_slots':int(selected.Transferred.sum()),
            'mean_long_exposure':float(daily.LongExposure.mean()),'mean_short_exposure':float(daily.ShortExposure.mean()),
            'worst_single_interval':float(daily.StrategyReturn.min()),'old_worst_single_interval':float(olddaily.StrategyReturn.min()),
            'old_replay_max_return_error':float(np.max(np.abs(oldcheck.StrategyReturn-olddaily.StrategyReturn)))},
        'frozen_model_checks':audit,'training_unchanged':True,'features_unchanged':True,'execution_unchanged':True,
        'model_fits_sha256':hashlib.sha256((MODEL/'model_fits.json').read_bytes()).hexdigest(),
        'market_signals_sha256':hashlib.sha256(MARKET.read_bytes()).hexdigest(),
        'plan_sha256':hashlib.sha256((ROOT/'experiment_plan.md').read_bytes()).hexdigest(),
        'test_used':False,'costs':0,'all_checks_passed':True}
    assert result['market_signals_sha256']==prior['market_signals_sha256']
    save('results.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
