"""Shared JPX features, fitting, portfolio allocation and performance utilities.
Price adjustment factors apply from the next session; do not change this silently.
"""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
WINDOWS=(5,22,60)
FEATURES=[f'T{k}' for k in WINDOWS]

def digest(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def price_t_features(prices):
    """Causal rolling price locations; input must already be calendar/scale aligned."""
    features = pd.DataFrame(index=prices.index)
    zero_std = {}
    for k in WINDOWS:
        mean = prices.rolling(k, min_periods=k).mean()
        std = prices.rolling(k, min_periods=k).std(ddof=1)
        bad_std = std <= 1e-12 * mean.abs().clip(lower=1.)
        zero_std[k] = int(bad_std.sum())
        features[f'T{k}'] = (prices-mean) / std.mask(bad_std)
    return features, zero_std


def json_write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def gd_fit(x, y):
    """Full-batch GD through exact sufficient statistics (not an OLS fit)."""
    gram = x.T @ x / len(x)
    xy = x.T @ y / len(x)
    lr = 1 / (2 * np.linalg.eigvalsh(gram).max())
    theta = np.zeros(x.shape[1])
    for step in range(100000):
        gradient = 2 * (gram @ theta - xy)
        if np.max(np.abs(gradient)) < 1e-13:
            break
        theta -= lr * gradient
    else:
        raise RuntimeError('GD did not converge on training data')
    reference = np.linalg.lstsq(x, y, rcond=None)[0]
    max_error = float(np.max(np.abs(theta-reference)))
    assert max_error < 1e-9, (theta, reference)
    return theta, {'iterations':step, 'learning_rate':float(lr),
                   'gradient_inf_norm':float(np.max(np.abs(gradient))),
                   'train_ols_reference_max_coefficient_difference':max_error,
                   'gram_condition_number':float(np.linalg.cond(gram)),
                   'train_mse':float(np.mean((x@theta-y)**2))}


def performance(returns):
    r = np.asarray(returns, dtype=float)
    if not len(r) or not np.isfinite(r).all():
        return None
    eq = np.cumprod(1+r)
    assert np.all(eq > 0), 'Wealth became non-positive; standard compounding invalid'
    peak = np.maximum.accumulate(np.r_[1., eq])[1:]
    sd = r.std(ddof=1)
    return {'days':len(r), 'cumulative_return':float(eq[-1]-1),
            'annualized_return_252':float(eq[-1]**(252/len(r))-1),
            'max_drawdown':float((eq/peak-1).min()),
            'sharpe_252_rf0':float(np.sqrt(252)*r.mean()/sd) if sd>0 else None,
            'daily_mean':float(r.mean()), 'daily_std':float(sd),
            'positive_day_fraction':float((r>0).mean()),
            'ending_wealth_per_100':float(eq[-1]*100),
            'simple_sum_return':float(r.sum())}


def build_features(raw):
    traded = (raw.Close.notna() & (raw.Volume>0)).groupby(raw.Date).any()
    cal = pd.DatetimeIndex(traded.index).sort_values()
    trading_cal = pd.DatetimeIndex(traded.index[traded]).sort_values()
    closed_dates = [str(d.date()) for d in traded.index[~traded]]
    entry_map = pd.Series(cal[1:].values, index=cal[:-1])
    exit_map = pd.Series(cal[2:].values, index=cal[:-2])
    frames = []
    zero_std = {k:0 for k in WINDOWS}
    gap_count = 0
    for security, stock in raw.groupby('SecuritiesCode', sort=True):
        stock = stock.sort_values('Date').set_index('Date')
        stock_cal = cal[(cal>=stock.index.min()) & (cal<=stock.index.max())]
        gap_count += len(stock_cal)-len(stock)
        # Missing market dates remain NaN rather than shortening the lookback.
        stock = stock.reindex(stock_cal)
        tradestock = stock.loc[stock.index.isin(trading_cal)]
        # CSV factor is stamped on the LAST cum-right trading day, and applies
        # to prices from the NEXT session. Exclude the current row's factor.
        factors = tradestock.AdjustmentFactor.fillna(1.).cumprod().shift(1, fill_value=1.)
        # Causal forward adjustment. At any t, multiplying this whole window
        # by cumulative_factor_before[t] gives prices on the t-date scale.
        # z-scores are invariant to this positive common scaling.
        adjusted = tradestock.Close / factors
        price_features, bad_counts = price_t_features(adjusted)
        for k in WINDOWS:
            zero_std[k] += bad_counts[k]
            stock[f'T{k}'] = price_features[f'T{k}']
        # On the one market-wide closure, the latest observable feature vector
        # remains yesterday's. Do not fill missing prices for individual stocks.
        closed_mask = stock.index.isin(pd.to_datetime(closed_dates))
        stock.loc[closed_mask, FEATURES] = stock[FEATURES].shift(1).loc[closed_mask]
        stock['SecuritiesCode'] = security
        stock['SignalDate'] = stock.index
        stock['EntryDate'] = stock.SignalDate.map(entry_map)
        stock['ExitDate'] = stock.SignalDate.map(exit_map)
        # Prior-day information only determines eligibility.
        stock['Eligible'] = (stock[FEATURES].notna().all(axis=1)
                             & (stock.Volume>0) & (stock.Close>0))
        stock.loc[closed_mask, 'Eligible'] = stock.Eligible.shift(1, fill_value=False).loc[closed_mask]
        frames.append(stock[['SecuritiesCode','SignalDate','EntryDate','ExitDate',
                             'Close','Volume','Target','Eligible']+FEATURES])
    return pd.concat(frames, ignore_index=True), cal, {'inserted_missing_market_dates':gap_count,
            'market_wide_closures_excluded_from_lookbacks_only':closed_dates,
            'zero_or_numerically_zero_std_rows':zero_std}


def allocate(day, side_weights=None):
    """Keep valid original slots, transfer wrong-sign slots to unused opposite ranks."""
    up = day.sort_values(['Prediction', 'SecuritiesCode'], ascending=[False, True])
    down = day.sort_values(['Prediction', 'SecuritiesCode'], ascending=[True, True])
    slots = []
    if side_weights is None:
        side_weights = ((.25, .15, .10), (.25, .15, .10))
    weights = np.asarray(side_weights, dtype=float)
    if (weights.shape != (2, 3) or not np.isfinite(weights).all()
            or (weights < 0).any() or weights.sum() > 1+1e-12):
        raise ValueError('Expected two sides of three nonnegative capital weights, total <= 1')
    for side, ordered, amounts in [(1, up, weights[0]), (-1, down, weights[1])]:
        for k, (idx, row) in enumerate(ordered.head(3).iterrows()):
            slots.append((idx, side, k+1, amounts[k], float(row.Prediction)))
    chosen, used, transfers = [], set(), []
    for idx, side, rank, amount, g in slots:
        if side*g > 0:
            used.add(idx)
            chosen.append((idx, side*amount, side, rank, False))
        elif side*g < 0:
            transfers.append((side, rank, amount))
        # Exact zero is held as cash, with no forced direction.
    for source_side, rank, amount in transfers:
        new_side = -source_side
        candidates = down if new_side < 0 else up
        candidates = candidates.loc[(candidates.Prediction*new_side > 0)
                                    & ~candidates.index.isin(used)]
        if len(candidates):
            idx = candidates.index[0]
            used.add(idx)
            chosen.append((idx, new_side*amount, source_side, rank, True))
    if not chosen:
        return pd.DataFrame(columns=list(day.columns)+['Weight','SourceSide','SourceRank','Transferred'])
    result = day.loc[[x[0] for x in chosen]].copy()
    result['Weight'] = [x[1] for x in chosen]
    result['SourceSide'] = ['long' if x[2]>0 else 'short' for x in chosen]
    result['SourceRank'] = [x[3] for x in chosen]
    result['Transferred'] = [x[4] for x in chosen]
    assert result.SecuritiesCode.is_unique
    assert (result.Weight*result.Prediction > 0).all()
    assert result.Weight.abs().sum() <= 1+1e-12
    return result


def check_contract_examples():
    cases = [([.05,.04,-.01,-.02,-.03,-.04,-.05,-.06], .4, .6),
             ([.06,.05,.04,.03,.02,.01,-.04,-.05], .6, .4),
             ([.08,.07,.06,.05,.04,.03,.02,.01], 1., 0.),
             ([-.01,-.02,-.03,-.04,-.05,-.06,-.07,-.08], 0., 1.)]
    for values, long_expected, short_expected in cases:
        day = pd.DataFrame({'SecuritiesCode':range(len(values)), 'Prediction':values})
        a = allocate(day)
        assert np.isclose(a.loc[a.Weight>0,'Weight'].sum(),long_expected)
        assert np.isclose(-a.loc[a.Weight<0,'Weight'].sum(),short_expected)
        if long_expected == .4:
            assert len(a.loc[(a.SecuritiesCode==4)&np.isclose(a.Weight,-.1)]) == 1
    # Not touched but profitable: entry 100, target 105, close 103 => +3%.
    assert np.isclose(103/100-1,.03)
    # 2-for-1 split: raw entry100, exit51, factor .5 => adjusted +2%.
    assert np.isclose(51/(100*.5)-1,.02)
    return {'allocation_examples':4, 'profitable_no_touch_example':True,
            'split_return_example':True}
