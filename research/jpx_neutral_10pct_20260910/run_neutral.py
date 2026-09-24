"""Replay the existing execution engine with the user-specified +/-10% band."""
from pathlib import Path
import hashlib
import json
import sys
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ARCHIVE = Path('/Users/coolguy/Desktop/JPX_data/modelling/JPX-market-regime-experiment.zip')
RAW = Path('/Users/coolguy/Desktop/JPX_data/raw/train_files/stock_prices.csv')
BAND = 0.10


def main():
    archive_hash = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    source = ROOT / 'source'
    source.mkdir(exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as z:
        for name in ['jpx_common.py', 'jpx_execution.py']:
            (source / name).write_bytes(z.read(name))
        sys.path.insert(0, str(source))
        from jpx_common import performance
        from jpx_execution import ORDER_COLUMNS, execute_hold

        def read(name):
            return pd.read_csv(z.open(name))

        manifest = json.loads(z.read('jpx_market_regime_results/run_manifest.json'))
        with RAW.open('rb') as f:
            raw_hash = hashlib.file_digest(f, 'sha256').hexdigest()
        assert raw_hash == manifest['source_sha256']['jpx_inputs/stock_prices.csv']
        signals = read('jpx_market_regime_results/market_signals.csv')
        saved = read('jpx_hold_same_results/selected_trades.csv')
        reference = read('jpx_market_regime_results/baseline_replay/daily_returns.csv')
        old_dynamic = read('jpx_market_regime_results/regime/daily_returns.csv')
        old_events = read('jpx_hold_same_results/position_events.csv')
        benchmark = read('jpx_inputs/nikkei_benchmark.csv')

    # Inclusive neutral boundary, with all model forecasts left unchanged.
    def classify(g):
        return np.where(g > BAND, 1, np.where(g < -BAND, -1, 0))

    np.testing.assert_array_equal(classify(np.array([-.100001, -.1, 0, .1, .100001])), [-1, 0, 0, 0, 1])
    signals['OriginalRegime'] = signals.Regime
    signals['Regime'] = classify(signals.MarketPrediction.to_numpy())
    signals['NeutralBandDecimalReturn'] = BAND
    signal_map = signals.set_index('SignalDate').Regime
    orders = saved.copy()
    orders['ExitDate'] = orders.PlannedExitDate
    orders['Weight'] = orders.RequestedWeight
    regime = orders.SignalDate.map(signal_map)
    assert regime.notna().all()
    long_budget = np.where(regime > 0, .7, np.where(regime < 0, .3, .5))
    side_budget = np.where(orders.SourceSide.eq('long'), long_budget, 1 - long_budget)
    rank_fraction = orders.SourceRank.map({1: .5, 2: .3, 3: .2})
    # Keep final sign and candidate identity from the original sign-transfer rule.
    orders.Weight = np.sign(orders.Weight) * side_budget * rank_fraction
    orders = orders[ORDER_COLUMNS].copy()
    interval_cols = ['SignalDate', 'EntryDate', 'ExitDate', 'ValidationYear',
                     'DailyRankIC', 'RawDailyRankIC', 'V2DailyRankIC', 'EligibleStocks']
    intervals = reference[interval_cols].copy()
    fields = ['Open', 'High', 'Low', 'Close', 'Volume']
    raw = pd.read_csv(RAW, usecols=['Date', 'SecuritiesCode', 'AdjustmentFactor'] + fields,
                      parse_dates=['Date']).sort_values(['SecuritiesCode', 'Date'])
    raw['CumulativeFactor'] = raw.groupby('SecuritiesCode').AdjustmentFactor.transform(
        lambda a: a.cumprod().shift(1, fill_value=1.))
    # The frozen orders require entry, actual exit, and carried valuation bars only.
    required = pd.concat([
        saved[['EntryDate', 'SecuritiesCode']].rename(columns={'EntryDate': 'Date'}),
        saved[['ExitDate', 'SecuritiesCode']].rename(columns={'ExitDate': 'Date'}),
        old_events[['ValuationDate', 'SecuritiesCode']].rename(columns={'ValuationDate': 'Date'})
    ]).drop_duplicates()
    required.Date = pd.to_datetime(required.Date)
    bars = raw.merge(required, on=['Date', 'SecuritiesCode'], how='inner', validate='one_to_one')
    bars = bars[['Date', 'SecuritiesCode'] + fields + ['CumulativeFactor']]
    del raw
    benchmark.Date = pd.to_datetime(benchmark.Date)
    trades, daily, events = execute_hold(orders, bars, intervals, benchmark.set_index('Date').Close)
    assert len(daily) == len(signals) == 953
    assert signals.Regime.eq(0).all()
    error = float(np.max(np.abs(daily.StrategyReturn.to_numpy() - reference.StrategyReturn.to_numpy())))
    np.testing.assert_allclose(daily.StrategyReturn, reference.StrategyReturn, rtol=0, atol=1e-12)
    np.testing.assert_allclose(daily.Pnl.sum(), trades.RealizedPnl.sum(), rtol=0, atol=1e-10)
    np.testing.assert_allclose(daily.StrategyReturn,
        events.groupby('IntervalEntryDate').Contribution.sum().reindex(daily.EntryDate).fillna(0), rtol=0, atol=1e-12)
    assert not trades.duplicated(['EntryDate', 'SecuritiesCode']).any()
    assert (daily.CashWeight >= -1e-12).all()
    daily = daily.merge(signals[['EntryDate', 'MarketPrediction', 'Regime']].assign(
        EntryDate=lambda x: pd.to_datetime(x.EntryDate)), on='EntryDate', validate='one_to_one')
    summary = {
        'neutral_band_decimal_return': BAND,
        'boundaries_inclusive': True,
        'prediction_min_pct': float(signals.MarketPrediction.min()*100),
        'prediction_max_pct': float(signals.MarketPrediction.max()*100),
        'regime_counts': {str(int(k)): int(v) for k,v in signals.Regime.value_counts().items()},
        'neutral_band_strategy': performance(daily.StrategyReturn),
        'original_dynamic_strategy': performance(old_dynamic.StrategyReturn),
        'original_v3_strategy': performance(reference.StrategyReturn),
        'mean_long_exposure': float(daily.LongExposure.mean()),
        'mean_short_exposure': float(daily.ShortExposure.mean()),
        'annual': {str(int(y)): performance(v.StrategyReturn) for y,v in daily.groupby('ValidationYear')},
        'baseline_max_daily_return_abs_difference': error,
        'execution': 'Original target-touch/carry/transfer rules; no new fees, slippage, or borrowing constraints',
        'model_refit': False,
        'test_data_used': False,
        'source_raw_sha256': raw_hash,
        'source_archive_sha256': archive_hash,
    }
    for name, frame in [('market_signals', signals), ('daily_returns', daily),
                        ('selected_trades', trades), ('position_events', events)]:
        frame.to_csv(ROOT / (name+'.csv'), index=False)
    (ROOT/'results.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
