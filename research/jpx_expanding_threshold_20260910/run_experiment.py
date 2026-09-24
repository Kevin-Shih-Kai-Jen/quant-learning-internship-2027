"""Predeclared expanding-window market model and chronological quantile-threshold selection."""
from pathlib import Path
import hashlib
import json
import sys
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = Path('/Users/coolguy/Desktop/JPX_data/modelling/JPX_data.zip')
EXPERIMENT = Path('/Users/coolguy/Desktop/JPX_data/modelling/JPX-market-regime-experiment.zip')
QUANTILES = [.25, .50, .75]


def sha(stream):
    h = hashlib.sha256()
    while chunk := stream.read(1024*1024):
        h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))


def main():
    source = ROOT / 'source'
    source.mkdir(exist_ok=True)
    plan_hash = hashlib.sha256((ROOT/'experiment_plan.md').read_bytes()).hexdigest()
    with zipfile.ZipFile(EXPERIMENT) as z:
        manifest = json.loads(z.read('jpx_market_regime_results/run_manifest.json'))
        for name in ['jpx_common.py', 'jpx_models.py', 'jpx_market_proxy.py', 'jpx_regime.py', 'jpx_execution.py']:
            content = z.read(name)
            assert hashlib.sha256(content).hexdigest() == manifest['source_sha256'][name]
            (source/name).write_bytes(content)
        def read(name): return pd.read_csv(z.open(name))
        saved = read('jpx_hold_same_results/selected_trades.csv')
        old = read('jpx_market_regime_results/baseline_replay/daily_returns.csv')
        old_signals = read('jpx_market_regime_results/market_signals.csv')
        benchmark = read('jpx_inputs/nikkei_benchmark.csv')
    sys.path.insert(0, str(source))
    from jpx_market_proxy import build_proxy
    from jpx_regime import MarketPriceModel
    from jpx_common import FEATURES, performance
    from jpx_execution import ORDER_COLUMNS, execute_hold
    hashes = {}
    with zipfile.ZipFile(DATA) as z:
        for leaf in ['stock_prices.csv', 'financials.csv']:
            with z.open('JPX_data/raw/train_files/'+leaf) as f:
                hashes[leaf] = sha(f)
            assert hashes[leaf] == manifest['source_sha256']['jpx_inputs/'+leaf]
        raw = pd.read_csv(z.open('JPX_data/raw/train_files/stock_prices.csv'),
            usecols=['Date','SecuritiesCode','Open','High','Low','Close','Volume','AdjustmentFactor'], parse_dates=['Date'])
        financials = pd.read_csv(z.open('JPX_data/raw/train_files/financials.csv'), low_memory=False)
    market = build_proxy(raw, financials, ROOT/'market_proxy')
    del financials
    market['LabelExit'] = pd.Series(market.index[2:], index=market.index[:-2])
    market['Target'] = market.Close.shift(-2)/market.Close.shift(-1)-1
    cols = ['SignalDate','EntryDate','ExitDate','ValidationYear','DailyRankIC','RawDailyRankIC','V2DailyRankIC','EligibleStocks']
    intervals = old[cols].copy()
    for c in ['SignalDate','EntryDate','ExitDate']:
        intervals[c] = pd.to_datetime(intervals[c])
    signals = intervals[['SignalDate','EntryDate','ExitDate','ValidationYear']].copy()
    signals['MarketPrediction'] = np.nan
    fits = []
    for year in [2018,2019,2020,2021]:
        mask = signals.ValidationYear.eq(year)
        signal_dates = pd.DatetimeIndex(signals.loc[mask,'SignalDate'])
        cutoff = signal_dates.min()
        train = market.loc[(market.index >= pd.Timestamp('2017-01-01'))
            & (market.index <= cutoff) & market.LabelExit.le(cutoff)
            & market[FEATURES+['Target']].notna().all(axis=1)]
        assert train.LabelExit.max() <= cutoff
        model = MarketPriceModel().fit(train)
        signals.loc[mask,'MarketPrediction'] = model.predict(market.loc[signal_dates]).Prediction.to_numpy()
        fits.append({'validation_year':year, 'train_rows':len(train),
            'train_first_signal':str(train.index.min().date()),'train_last_signal':str(train.index.max().date()),
            'last_label_exit':str(train.LabelExit.max().date()),'fit_asof':str(cutoff.date()),
            'coefficients':dict(zip(['alpha']+FEATURES,map(float,model.theta)))})
    assert signals.MarketPrediction.notna().all()
    np.testing.assert_allclose(signals.loc[signals.ValidationYear.eq(2018),'MarketPrediction'],
        old_signals.loc[old_signals.ValidationYear.eq(2018),'MarketPrediction'],rtol=0,atol=1e-10)
    signals.to_csv(ROOT/'expanding_market_signals.csv',index=False)
    write_json(ROOT/'market_fits.json',fits)
    print('Expanding market fits:',[(f['validation_year'],f['train_rows']) for f in fits],flush=True)

    orders = saved.copy()
    orders['ExitDate'] = orders.PlannedExitDate
    orders['Weight'] = orders.RequestedWeight
    for c in ['SignalDate','PreviousSignalDate','EntryDate','ExitDate']:
        orders[c] = pd.to_datetime(orders[c])
    orders = orders[ORDER_COLUMNS].copy()
    raw = raw.sort_values(['SecuritiesCode','Date'])
    raw['CumulativeFactor'] = raw.groupby('SecuritiesCode').AdjustmentFactor.transform(lambda a:a.cumprod().shift(1,fill_value=1.))
    needed_codes = orders.SecuritiesCode.unique()
    bars = raw.loc[raw.SecuritiesCode.isin(needed_codes)
        & raw.Date.between(intervals.EntryDate.min(),intervals.ExitDate.max()),
        ['Date','SecuritiesCode','Open','High','Low','Close','Volume','CumulativeFactor']].copy()
    del raw
    benchmark.Date = pd.to_datetime(benchmark.Date)
    benchmark = benchmark.set_index('Date').Close

    def replay(rows, regime, folder=None):
        timeline = intervals.loc[intervals.EntryDate.isin(rows.EntryDate)].copy()
        selected = orders.loc[orders.EntryDate.isin(rows.EntryDate)].copy()
        mapping = pd.Series(np.asarray(regime),index=rows.EntryDate)
        reg = selected.EntryDate.map(mapping)
        assert reg.notna().all()
        long = np.where(reg>0,.7,np.where(reg<0,.3,.5))
        budgets = np.where(selected.SourceSide.eq('long'),long,1-long)
        selected.Weight = np.sign(selected.Weight)*budgets*selected.SourceRank.map({1:.5,2:.3,3:.2})
        # Full date coverage for selected symbols; no cached exit prices or dates.
        subbars = bars.loc[bars.SecuritiesCode.isin(selected.SecuritiesCode.unique())
                          & bars.Date.between(timeline.EntryDate.min(),timeline.ExitDate.max())]
        t,d,e = execute_hold(selected,subbars,timeline,benchmark)
        np.testing.assert_allclose(d.StrategyReturn,e.groupby('IntervalEntryDate').Contribution.sum().reindex(d.EntryDate).fillna(0),rtol=0,atol=1e-12)
        assert (d.CashWeight>=-1e-12).all() and not t.duplicated(['EntryDate','SecuritiesCode']).any()
        if folder:
            folder.mkdir(exist_ok=True,parents=True)
            for name,frame in [('daily_returns',d),('selected_trades',t),('position_events',e)]:
                frame.to_csv(folder/(name+'.csv'),index=False)
        return d,performance(d.StrategyReturn),performance(d.IndexReturn)

    def regimes(g,tau):
        return np.where(g>tau,1,np.where(g < -tau,-1,0))

    np.testing.assert_array_equal(regimes(np.array([-.02,-.01,0,.01,.02]),.01),[-1,0,0,0,1])
    base,base_perf,_ = replay(signals,np.zeros(len(signals)),ROOT/'baseline_replay')
    base_error = float(np.max(np.abs(base.StrategyReturn-old.StrategyReturn)))
    np.testing.assert_allclose(base.StrategyReturn,old.StrategyReturn,rtol=0,atol=1e-12)

    annual=[]; selection_rows=[]; deployment=[]
    for year in [2019,2020,2021]:
        future = signals.loc[signals.ValidationYear.eq(year)].copy()
        cutoff = future.SignalDate.min()
        history = signals.loc[(signals.SignalDate<cutoff)&signals.ExitDate.le(cutoff)].sort_values('SignalDate').copy()
        split = len(history)//2
        calibration,scoring = history.iloc[:split].copy(),history.iloc[split:].copy()
        assert calibration.SignalDate.max()<scoring.SignalDate.min()
        assert scoring.ExitDate.max()<=cutoff
        # Threshold calibration reads g only; its generating model uses earlier labels.
        record={'validation_year':year,'selection_asof':str(cutoff.date()),'historical_rows':len(history),
            'calibration_rows':len(calibration),'scoring_rows':len(scoring),
            'calibration_signal_start':str(calibration.SignalDate.min().date()),
            'calibration_signal_end':str(calibration.SignalDate.max().date()),
            'scoring_entry_start':str(scoring.EntryDate.min().date()),
            'scoring_exit_end':str(scoring.ExitDate.max().date()),'candidates':[]}
        for q in QUANTILES:
            tau=float(calibration.MarketPrediction.abs().quantile(q))
            _,perf,bperf=replay(scoring,regimes(scoring.MarketPrediction,tau),ROOT/f'inner_{year}'/f'q{int(q*100)}')
            return_pass=perf['cumulative_return']>bperf['cumulative_return']+1e-12
            dd_pass=abs(perf['max_drawdown'])<abs(bperf['max_drawdown'])-1e-12
            candidate={'q':q,'threshold_decimal_return':tau,'strategy':perf,'benchmark':bperf,
                'return_pass':bool(return_pass),'drawdown_pass':bool(dd_pass),'qualified':bool(return_pass and dd_pass)}
            record['candidates'].append(candidate)
            selection_rows.append({'validation_year':year,'q':q,'calibration_tau_pct':tau*100,
                'inner_annualized_return':perf['annualized_return_252'],'inner_cumulative_return':perf['cumulative_return'],
                'inner_max_drawdown_magnitude':abs(perf['max_drawdown']),
                'benchmark_cumulative_return':bperf['cumulative_return'],'benchmark_max_drawdown_magnitude':abs(bperf['max_drawdown']),
                'return_pass':return_pass,'drawdown_pass':dd_pass,'qualified':return_pass and dd_pass})
        qualified=[c for c in record['candidates'] if c['qualified']]
        if not qualified:
            record['status']='no_qualified_candidate'
            record['selected_q']=None
            record['deployment_tau_decimal_return']=None
            record['outer_strategy']=None
            record['outer_benchmark']=performance(benchmark.reindex(future.ExitDate).to_numpy()/benchmark.reindex(future.EntryDate).to_numpy()-1)
        else:
            winner=sorted(qualified,key=lambda c:(-c['strategy']['annualized_return_252'],abs(c['strategy']['max_drawdown']),c['q']))[0]
            q=winner['q'];tau=float(history.MarketPrediction.abs().quantile(q))
            _,perf,bperf=replay(future,regimes(future.MarketPrediction,tau),ROOT/f'outer_{year}')
            record.update({'status':'selected','selected_q':q,'deployment_tau_decimal_return':tau,
                'outer_strategy':perf,'outer_benchmark':bperf,
                'outer_both_conditions_met':bool(perf['cumulative_return']>bperf['cumulative_return']+1e-12 and abs(perf['max_drawdown'])<abs(bperf['max_drawdown'])-1e-12),
                'outer_neutral_days':int((future.MarketPrediction.abs()<=tau).sum()),'outer_days':len(future)})
            deployment.append(future.assign(Threshold=tau,Quantile=q,Regime=regimes(future.MarketPrediction,tau)))
        annual.append(record)
        print('Annual selection',year,record['status'],record['selected_q'],record.get('outer_both_conditions_met'),flush=True)
    pd.DataFrame(selection_rows).to_csv(ROOT/'candidate_selection.csv',index=False)
    write_json(ROOT/'annual_selection.json',annual)
    combined=None
    if deployment:
        dep=pd.concat(deployment).sort_values('EntryDate')
        dep.to_csv(ROOT/'deployment_signals.csv',index=False)
        years=[r['validation_year'] for r in annual if r['status']=='selected']
        if years==list(range(min(years),max(years)+1)):
            _,perf,bperf=replay(dep,dep.Regime,ROOT/'combined_selected_years')
            combined={'years':years,'strategy':perf,'benchmark':bperf,
                'first_entry':str(dep.EntryDate.min().date()),'last_exit':str(dep.ExitDate.max().date())}
    result={'plan_sha256':plan_hash,'raw_source_sha256':hashes,
        'quantile_candidates':QUANTILES,'benchmark':'Nikkei 225 spot price index from original experiment',
        'first_threshold_evaluation_year':2019,'2018_role':'historical OOS market-prediction calibration; no 2017 stock-order history in supplied experiment',
        'market_training':'expanding historical samples; annual fixed coefficients',
        'stock_model_refit':False,'cost_model':'original zero costs and target-touch assumptions',
        'test_data_used':False,'result_type':'development evaluation on previously examined years',
        'market_fits':fits,'annual_selection':annual,'combined_selected_years':combined,
        'original_baseline_replay_max_daily_abs_error':base_error,'all_checks_passed':True}
    write_json(ROOT/'results.json',result)
    compact={'market_train_rows':[(f['validation_year'],f['train_rows']) for f in fits],
        'annual':[{'year':a['validation_year'],'status':a['status'],'q':a['selected_q'],
                   'tau_pct':None if a['deployment_tau_decimal_return'] is None else a['deployment_tau_decimal_return']*100,
                   'strategy':a['outer_strategy'],'benchmark':a['outer_benchmark'],
                   'both_pass':a.get('outer_both_conditions_met')} for a in annual],
        'combined':combined,'baseline_error':base_error}
    print(json.dumps(compact,ensure_ascii=False,indent=2,allow_nan=False),flush=True)


if __name__=='__main__':
    main()
