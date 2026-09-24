"""Causal volume z-scores and price-volume products for the shared runner."""
import numpy as np
import pandas as pd
from jpx_common import WINDOWS, FEATURES

VOLUME_FEATURES = [f'V{k}' for k in WINDOWS]


def add_volume_features(data, raw, closed):
    # Use the same reindexed stock calendar as price features. Price adjustment
    # divides by the prior cumulative factor; volume uses its reciprocal action.
    result = data.copy()
    factors = raw.sort_values(['SecuritiesCode', 'Date']).copy()
    factors['PriorFactor'] = factors.groupby('SecuritiesCode').AdjustmentFactor.transform(
        lambda x: x.cumprod().shift(1, fill_value=1.))
    result = result.merge(factors[['Date', 'SecuritiesCode', 'PriorFactor']],
                          left_on=['SignalDate', 'SecuritiesCode'],
                          right_on=['Date', 'SecuritiesCode'], how='left',
                          validate='one_to_one').drop(columns='Date')
    result = result.sort_values(['SecuritiesCode', 'SignalDate'])
    audit = {'constant_volume_window_rows': {}, 'volume_adjustment':
             'Volume * cumulative AdjustmentFactor strictly before signal date',
             'constant_window_policy': 'zero z-score for a complete constant window',
             'missing_policy': 'never impute unavailable volume; fail if baseline-eligible row lacks a feature'}
    actual = result.loc[~result.SignalDate.isin(closed)].copy()
    actual['AdjustedVolume'] = actual.Volume * actual.PriorFactor
    grouped = actual.groupby('SecuritiesCode').AdjustedVolume
    for k in WINDOWS:
        mean = grouped.transform(lambda x: x.rolling(k, min_periods=k).mean())
        std = grouped.transform(lambda x: x.rolling(k, min_periods=k).std(ddof=1))
        constant = std <= 1e-12 * mean.abs().clip(lower=1.)
        actual[f'V{k}'] = ((actual.AdjustedVolume-mean)/std.mask(constant)).mask(constant, 0.)
        audit['constant_volume_window_rows'][k] = int(constant.sum())
        result[f'V{k}'] = actual[f'V{k}']
    closed_mask = result.SignalDate.isin(closed)
    result.loc[closed_mask, VOLUME_FEATURES] = result.groupby('SecuritiesCode')[VOLUME_FEATURES].shift(1).loc[closed_mask]
    missing = result.Eligible & ~np.isfinite(result[VOLUME_FEATURES]).all(axis=1)
    audit['baseline_eligible_rows_missing_volume_features'] = int(missing.sum())
    if missing.any():
        raise ValueError(f'{missing.sum()} baseline-eligible rows lack volume features; explicitly resolve universe before running')
    for p in WINDOWS:
        for v in WINDOWS:
            result[f'P{p}xV{v}'] = result[f'T{p}'] * result[f'V{v}']
    return result.drop(columns='PriorFactor'), audit
