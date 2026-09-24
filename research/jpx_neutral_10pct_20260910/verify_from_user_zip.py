"""Rebuild market signals and replay frozen stock orders using the supplied data ZIP."""
from pathlib import Path
import hashlib
import json
import sys
import zipfile

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / 'zip_verification'
DATA_ZIP = Path('/Users/coolguy/Desktop/JPX_data/modelling/JPX_data.zip')
EXPERIMENT_ZIP = Path('/Users/coolguy/Desktop/JPX_data/modelling/JPX-market-regime-experiment.zip')
BAND = .10


def digest_stream(f):
    h = hashlib.sha256()
    while chunk := f.read(1024 * 1024):
        h.update(chunk)
    return h.hexdigest()


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def main():
    OUT.mkdir(exist_ok=True)
    source = OUT / 'source'
    source.mkdir(exist_ok=True)
    with zipfile.ZipFile(EXPERIMENT_ZIP) as z:
        manifest = json.loads(z.read('jpx_market_regime_results/run_manifest.json'))
        for name in ['jpx_common.py', 'jpx_execution.py', 'jpx_market_proxy.py', 'jpx_models.py', 'jpx_regime.py']:
            content = z.read(name)
            assert hashlib.sha256(content).hexdigest() == manifest['source_sha256'][name]
            (source / name).write_bytes(content)
        def read(name):
            return pd.read_csv(z.open(name))
        old_proxy = read('jpx_market_proxy_results/market_proxy_daily.csv').set_index('Date')
        old_proxy.index = pd.to_datetime(old_proxy.index)
        old_signals = read('jpx_market_regime_results/market_signals.csv')
        old_folds = json.loads(z.read('jpx_market_regime_results/results.json'))['market_folds']
        old_baseline = read('jpx_market_regime_results/baseline_replay/daily_returns.csv')
        old_dynamic = read('jpx_market_regime_results/regime/daily_returns.csv')
        saved_orders = read('jpx_hold_same_results/selected_trades.csv')
        benchmark = read('jpx_inputs/nikkei_benchmark.csv')
    sys.path.insert(0, str(source))
    from jpx_common import FEATURES, performance
    from jpx_execution import ORDER_COLUMNS, execute_hold
    from jpx_market_proxy import build_proxy
    from jpx_regime import MarketPriceModel

    with DATA_ZIP.open('rb') as f:
        zip_hash = digest_stream(f)
    hashes = {}
    with zipfile.ZipFile(DATA_ZIP) as z:
        names = z.namelist()
        for leaf in ['stock_prices.csv', 'financials.csv']:
            member = 'JPX_data/raw/train_files/' + leaf
            with z.open(member) as f:
                h = digest_stream(f)
            expected = manifest['source_sha256']['jpx_inputs/' + leaf]
            assert h == expected, f'Input differs from original experiment: {leaf}'
            hashes[leaf] = {'zip_member': member, 'sha256': h, 'matches_original_manifest': True}
        raw = pd.read_csv(z.open('JPX_data/raw/train_files/stock_prices.csv'),
            usecols=['Date', 'SecuritiesCode', 'Open', 'High', 'Low', 'Close', 'Volume', 'AdjustmentFactor'],
            parse_dates=['Date'])
        financials = pd.read_csv(z.open('JPX_data/raw/train_files/financials.csv'), low_memory=False)
    assert not raw.duplicated(['Date', 'SecuritiesCode']).any()
    input_info = {'stock_rows': len(raw), 'stock_count': int(raw.SecuritiesCode.nunique()),
                  'stock_start': str(raw.Date.min().date()), 'stock_end': str(raw.Date.max().date()),
                  'financial_rows': len(financials),
                  'test_named_members': [n for n in names if 'test' in n.lower() and not n.endswith('/')]}
    print('ZIP input hashes match original stock/financial sources', flush=True)

    # Rebuild the market index and its three features directly from ZIP inputs.
    market = build_proxy(raw, financials, OUT / 'rebuilt_proxy')
    pd.testing.assert_index_equal(market.index, old_proxy.index)
    proxy_errors = {}
    for col in old_proxy.columns:
        np.testing.assert_allclose(market[col], old_proxy[col], rtol=1e-10, atol=1e-10, equal_nan=True)
        proxy_errors[col] = float(np.nanmax(np.abs(market[col] - old_proxy[col])))
    del financials
    print('Market proxy and T5/T22/T60 rebuilt and reconciled', flush=True)

    interval_cols = ['SignalDate', 'EntryDate', 'ExitDate', 'ValidationYear',
                     'DailyRankIC', 'RawDailyRankIC', 'V2DailyRankIC', 'EligibleStocks']
    intervals = old_baseline[interval_cols].copy()
    for c in ['SignalDate', 'EntryDate', 'ExitDate']:
        intervals[c] = pd.to_datetime(intervals[c])
    dates = market.index
    market['EntryDate'] = pd.Series(dates[1:], index=dates[:-1])
    market['ExitDate'] = pd.Series(dates[2:], index=dates[:-2])
    market['Target'] = market.Close.shift(-2) / market.Close.shift(-1) - 1
    market['Prediction'] = np.nan
    folds = []
    for year in [2017, 2018, 2019, 2020]:
        cutoff = dates[dates.year == year].max()
        train = market.loc[(dates.year == year) & market.ExitDate.le(cutoff)
                           & market[FEATURES + ['Target']].notna().all(axis=1)]
        val_dates = pd.DatetimeIndex(intervals.loc[intervals.ValidationYear == year+1, 'SignalDate'])
        assert train.ExitDate.max() <= val_dates.min()
        model = MarketPriceModel().fit(train)
        market.loc[val_dates, 'Prediction'] = model.predict(market.loc[val_dates]).Prediction.to_numpy()
        reference = next(x for x in old_folds if x['train_year'] == year)
        theta_old = np.array([reference['coefficients'][c] for c in ['alpha']+FEATURES])
        np.testing.assert_allclose(model.theta, theta_old, rtol=0, atol=1e-10)
        assert len(train) == reference['train_rows']
        folds.append({'train_year': year, 'validation_year': year+1, 'train_rows': len(train),
                      'last_training_label_exit': str(train.ExitDate.max().date()),
                      'first_validation_signal': str(val_dates.min().date()),
                      'coefficient_max_abs_difference': float(np.max(np.abs(model.theta-theta_old)))})
    signals = intervals[['SignalDate', 'EntryDate', 'ExitDate', 'ValidationYear']].copy()
    signals['MarketPrediction'] = signals.SignalDate.map(market.Prediction)
    signals['MarketTarget'] = signals.SignalDate.map(market.Target)
    assert signals.MarketPrediction.notna().all()
    np.testing.assert_allclose(signals.MarketPrediction, old_signals.MarketPrediction, rtol=0, atol=1e-10)
    signals['OriginalRegime'] = np.sign(signals.MarketPrediction).astype(int)
    signals['Regime'] = np.where(signals.MarketPrediction > BAND, 1,
                                np.where(signals.MarketPrediction < -BAND, -1, 0))
    assert len(signals) == 953 and signals.Regime.eq(0).all()
    signal_error = float(np.max(np.abs(signals.MarketPrediction-old_signals.MarketPrediction)))
    print('All four market fits and 953 signals reconciled; all days neutral', flush=True)

    # Individual-stock coefficients and candidate ordering remain frozen, as in the user's experiment.
    orders = saved_orders.copy()
    orders['ExitDate'] = orders.PlannedExitDate
    orders['SignalDate'] = pd.to_datetime(orders.SignalDate)
    daily_regime = orders.SignalDate.map(signals.set_index('SignalDate').Regime)
    long_budget = np.where(daily_regime > 0, .7, np.where(daily_regime < 0, .3, .5))
    side_budget = np.where(orders.SourceSide.eq('long'), long_budget, 1-long_budget)
    orders['Weight'] = np.sign(orders.RequestedWeight) * side_budget * orders.SourceRank.map({1: .5, 2: .3, 3: .2})
    orders = orders[ORDER_COLUMNS].copy()
    raw = raw.sort_values(['SecuritiesCode', 'Date'])
    raw['CumulativeFactor'] = raw.groupby('SecuritiesCode').AdjustmentFactor.transform(
        lambda a: a.cumprod().shift(1, fill_value=1.))
    # Include every raw date for selected symbols, so exits do not depend on cached exit dates.
    bars = raw.loc[raw.SecuritiesCode.isin(orders.SecuritiesCode.unique())
                   & raw.Date.between(intervals.EntryDate.min(), intervals.ExitDate.max()),
                   ['Date', 'SecuritiesCode', 'Open', 'High', 'Low', 'Close', 'Volume', 'CumulativeFactor']].copy()
    del raw
    benchmark.Date = pd.to_datetime(benchmark.Date)
    trades, daily, events = execute_hold(orders, bars, intervals, benchmark.set_index('Date').Close)
    prior_result = pd.read_csv(BASE / 'daily_returns.csv')
    for expected in [prior_result, old_baseline]:
        np.testing.assert_array_equal(daily.EntryDate.to_numpy(), pd.to_datetime(expected.EntryDate).to_numpy())
        np.testing.assert_allclose(daily.StrategyReturn, expected.StrategyReturn, rtol=0, atol=1e-12)
    daily_error = float(np.max(np.abs(daily.StrategyReturn-prior_result.StrategyReturn)))
    np.testing.assert_allclose(daily.Pnl.sum(), trades.RealizedPnl.sum(), rtol=0, atol=1e-10)
    np.testing.assert_allclose(daily.StrategyReturn,
        events.groupby('IntervalEntryDate').Contribution.sum().reindex(daily.EntryDate).fillna(0), rtol=0, atol=1e-12)
    assert (daily.CashWeight >= -1e-12).all()
    assert not trades.duplicated(['EntryDate', 'SecuritiesCode']).any()

    # Independently derive each executed trade's first tradable exit and target/close fill from raw bars.
    by_code = {c: v.set_index('Date').sort_index() for c,v in bars.groupby('SecuritiesCode')}
    max_fill_error = 0.
    for p in trades.loc[trades.Executed].itertuples():
        b = by_code[p.SecuritiesCode]
        entry = b.loc[p.EntryDate]
        candidates = b.loc[b.index >= p.PlannedExitDate]
        eligible = candidates.loc[candidates.Volume > 0]
        exit_date = eligible.index[0]
        assert exit_date == p.ExitDate
        ex = eligible.iloc[0]
        entry_on_exit_scale = entry.Close * ex.CumulativeFactor / entry.CumulativeFactor
        target = entry_on_exit_scale * (1+p.Prediction)
        fill = target if ex.Low <= target <= ex.High else ex.Close
        err = abs(float(fill) - p.ExitPrice)
        max_fill_error = max(max_fill_error, err)
        np.testing.assert_allclose(fill, p.ExitPrice, rtol=1e-12, atol=1e-10)
        pnl = np.sign(p.Weight)*p.EntryNotional*(fill/entry_on_exit_scale-1)
        np.testing.assert_allclose(pnl, p.RealizedPnl, rtol=1e-10, atol=1e-10)
    summary = {
        'data_zip': str(DATA_ZIP), 'data_zip_sha256': zip_hash,
        'inputs': hashes, 'input_dimensions': input_info,
        'proxy_max_abs_differences': proxy_errors, 'market_folds': folds,
        'market_signal_max_abs_difference': signal_error,
        'threshold_decimal_return': BAND, 'neutral_days': int(signals.Regime.eq(0).sum()),
        'prediction_min_pct': float(signals.MarketPrediction.min()*100),
        'prediction_max_pct': float(signals.MarketPrediction.max()*100),
        'strategy': performance(daily.StrategyReturn),
        'previous_dynamic_reference': performance(old_dynamic.StrategyReturn),
        'daily_return_max_abs_difference_from_previous_run': daily_error,
        'independent_fill_max_abs_difference': max_fill_error,
        'selected_positions': len(trades), 'executed_positions': int(trades.Executed.sum()),
        'delayed_positions': int(trades.DelaySessions.gt(0).sum()),
        'stock_model_refit': False, 'market_model_reproduced': True,
        'test_data_used': False,
        'scope': 'ZIP stock/financial sources; rebuilt market proxy and market fits; frozen original stock orders; original target-touch/carry engine; zero trading costs',
        'all_checks_passed': True,
    }
    for name, frame in [('market_signals', signals), ('daily_returns', daily),
                         ('selected_trades', trades), ('position_events', events)]:
        frame.to_csv(OUT / (name+'.csv'), index=False)
    dump('verification.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
