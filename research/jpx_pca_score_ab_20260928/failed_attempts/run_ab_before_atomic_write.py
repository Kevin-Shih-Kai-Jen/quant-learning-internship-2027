"""Causal rolling PCA score ablation with the original JPX rank-weight rule.

Generation receives prices and frozen predictions ONLY. Target is read after all
scores/ranks/directions have been saved. This is reused historical validation.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys
import time
import zipfile

import numpy as np
import pandas as pd
import scipy
from scipy.linalg import eigh
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent
VARIANTS = ['A', 'B_PC6', 'B_MAX']
W = np.linspace(2., 1., 200)
MISSING_RANK = np.iinfo(np.uint16).max
RAW_SHA = 'bf774a86f834e5338bba74f2356ebb750f93a1cc4e89b064ee07251e8ed851db'
RANK_SHA = {
    2018: '2a33754c695cc16579a959dc9e5d735a3fff323949a04cff82808e172b15679f',
    2019: '0178112bc51bdb28a3a25d8724647e2f76cbf662f6b0e6ecb5c81e14eb54606f',
    2020: '0bbe51096fa09438831b06cef196055cadc40fe08ef4588b31f6a98b92643403',
    2021: 'd0029f7dc9c89362183250321f6583a56df820b337a7233bdfba2ea65adfe42a',
}


def sha(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def records(frame):
    return json.loads(frame.to_json(orient='records', date_format='iso', double_precision=15))


def raw_member(z):
    names = [n for n in z.namelist() if n.endswith('raw/train_files/stock_prices.csv')]
    if len(names) != 1:
        raise ValueError('Expected exactly one TRAIN stock_prices.csv')
    return names[0]


def build_returns(price):
    assert 'Target' not in price
    assert not price.duplicated(['Date', 'SecuritiesCode']).any()
    b = price.sort_values(['SecuritiesCode', 'Date']).copy()
    factor = b.groupby('SecuritiesCode').AdjustmentFactor.transform(
        lambda x: x.fillna(1).cumprod().shift(1, fill_value=1))
    b['AdjustedClose'] = b.Close / factor
    active = (b.Close.gt(0) & b.Volume.gt(0)).groupby(b.Date).any().sort_index()
    cal = pd.DatetimeIndex(active.index[active])
    close = b.pivot(index='Date', columns='SecuritiesCode', values='AdjustedClose').reindex(cal)
    volume = b.pivot(index='Date', columns='SecuritiesCode', values='Volume').reindex(cal)
    good = close.gt(0) & np.isfinite(close) & volume.gt(0) & np.isfinite(volume)
    r = (close / close.shift(1) - 1).where(good & good.shift(1, fill_value=False))
    # The first market date cannot have an observed adjacent-day return.
    r = r.iloc[1:]
    return r, active.index[~active].strftime('%Y-%m-%d').tolist()


def pca_asof(panel, date, universe, lookback=252, minimum=120):
    train = panel.loc[:date, universe].tail(lookback)
    if len(train) < minimum:
        raise ValueError('Not enough past observations')
    a = train.to_numpy(float)
    keep = np.isfinite(a).all(axis=0)
    keep[keep] &= np.std(a[:, keep], axis=0, ddof=1) > 0
    if keep.sum() < 400:
        raise ValueError('Insufficient PCA coverage')
    x = np.ascontiguousarray(a[:, keep] - a[:, keep].mean(axis=0))
    # Dual covariance is T x T; recover the N-dimensional directions exactly.
    gram = x @ x.T / (len(x) - 1)
    l, u = eigh(gram, check_finite=False, driver='evd')
    order = np.argsort(l)[::-1]
    l, u = l[order], u[:, order]
    valid = l > l[0] * max(x.shape) * np.finfo(float).eps
    l, u = l[valid], u[:, valid]
    v = (x.T @ u) / np.sqrt((len(x) - 1) * l)
    sign = np.sign(v[np.argmax(np.abs(v), axis=0), np.arange(len(l))])
    v *= sign
    orth = float(np.max(np.abs(v.T @ v - np.eye(len(l)))))
    err = float(np.linalg.norm(x.T @ (x @ v) / (len(x) - 1) - v * l) / np.linalg.norm(v * l))
    total = float(np.sum(x * x) / (len(x) - 1))
    assert orth < 1e-8 and err < 1e-9
    np.testing.assert_allclose(l.sum(), total, rtol=1e-10, atol=1e-15)
    return keep, x, v, l, {'window_start': str(train.index[0].date()),
        'window_end': str(train.index[-1].date()), 'window_days': len(train),
        'pca_stocks': int(keep.sum()), 'rank': len(l), 'orth_error': orth,
        'eigen_equation_error': err, 'total_stock_variance': total}


def rank_weights(scores, codes):
    if len(scores) < 400 or not np.isfinite(scores).all():
        raise ValueError('Need finite predictions for at least 400 stocks')
    order = np.lexsort((np.asarray(codes), -np.asarray(scores)))
    rank = np.empty(len(order), dtype=np.uint16)
    rank[order] = np.arange(len(order), dtype=np.uint16)
    w = np.zeros(len(order))
    w[order[:200]] = .5 * W / W.sum()
    w[order[-200:][::-1]] = -.5 * W / W.sum()
    return rank, w


def ablate(scores, loading):
    q = loading - loading.mean()
    g = scores - scores.mean()
    den = float(q @ q)
    if den <= 1e-24:
        return scores.copy(), 0., 0.
    b = float(q @ g / den)
    result = scores - b * q
    relative_residual = abs(float(q @ (result - result.mean()))) / max(
        np.linalg.norm(q) * np.linalg.norm(g), 1e-30)
    np.testing.assert_allclose(result.mean(), scores.mean(), atol=1e-13, rtol=1e-11)
    assert relative_residual < 1e-9
    fraction = float(np.sum((b * q) ** 2) / np.sum(g ** 2)) if np.any(g) else 0.
    return result, b, fraction


def spearman(a, b):
    valid = np.isfinite(a) & np.isfinite(b)
    x, y = rankdata(a[valid]), rankdata(b[valid])
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def spread_from_ranks(ranks, target):
    order = np.argsort(ranks)
    selected = np.r_[order[:200], order[-200:]]
    missing = int(np.sum(~np.isfinite(target[selected])))
    if missing:
        return np.nan, missing
    return float((target[order[:200]] @ W - target[order[-200:][::-1]] @ W) / W.mean()), 0


def stationary_intervals(series, seed=20260928, samples=5000, block=20):
    # Paired resampling preserves the same dates across all variants.
    a = np.asarray(series)
    n = len(a)
    rng = np.random.default_rng(seed)
    idx = np.empty((samples, n), dtype=np.int32)
    idx[:, 0] = rng.integers(0, n, samples)
    for j in range(1, n):
        restart = rng.random(samples) < 1 / block
        idx[:, j] = np.where(restart, rng.integers(0, n, samples), (idx[:, j - 1] + 1) % n)
    sr = []
    for k in range(a.shape[1]):
        b = a[idx, k]
        sr.append(b.mean(axis=1) / b.std(axis=1, ddof=1))
    out = {}
    for k in range(1, len(sr)):
        diff = sr[k] - sr[0]
        out[VARIANTS[k]] = {'difference': float(a[:, k].mean() / a[:, k].std(ddof=1)
            - a[:, 0].mean() / a[:, 0].std(ddof=1)),
            'exploratory_95pct_interval': np.quantile(diff, [.025, .975]).tolist(),
            'bootstrap_fraction_positive': float(np.mean(diff > 0))}
    return out


def generate(args):
    out = args.out
    if out.exists():
        raise FileExistsError('Use a NEW output folder')
    out.mkdir(parents=True)
    (out / 'traces').mkdir()
    with zipfile.ZipFile(args.zip) as z:
        member = raw_member(z)
        with z.open(member) as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        assert digest == RAW_SHA
        # Target deliberately absent from all generation data structures.
        with z.open(member) as stream:
            prices = pd.read_csv(stream, usecols=['Date', 'SecuritiesCode', 'Close', 'Volume',
                'AdjustmentFactor', 'SupervisionFlag'], parse_dates=['Date'])
    assert len(prices) == 2332531
    panel, closures = build_returns(prices)
    codes = panel.columns.to_numpy(dtype=np.int32)
    active_source = prices.loc[~prices.SupervisionFlag, ['Date', 'SecuritiesCode']]
    source_groups = {d: np.sort(g.SecuritiesCode.to_numpy()) for d, g in active_source.groupby('Date')}
    del prices, active_source
    frames = []
    source_hashes = {'raw_member': digest, 'plan': sha(ROOT / 'experiment_plan.md')}
    for year in range(2018, 2022):
        path = args.input / 'research/jpx_v7_daily_mse_20260912/v7_equal' / f'ranks_{year}.csv.gz'
        source_hashes[str(year)] = sha(path)
        assert source_hashes[str(year)] == RANK_SHA[year]
        frames.append(pd.read_csv(path, usecols=['Date', 'SecuritiesCode', 'g', 'Rank', 'ValidationYear'],
                                  parse_dates=['Date'], float_precision='round_trip'))
    ranks = pd.concat(frames, ignore_index=True)
    assert len(ranks) == 1864363 and ranks.Date.nunique() == 953
    dates = pd.DatetimeIndex(sorted(ranks.Date.unique()))
    D, N = len(dates), len(codes)
    predictions = np.full((3, D, N), np.nan)
    rank_trace = np.full((3, D, N), MISSING_RANK, dtype=np.uint16)
    loadings = np.full((2, D, N), np.nan)
    pca_eligible = np.zeros((D, N), dtype=bool)
    eigenvalues = np.full((D, 251), np.nan)
    exposures = np.full((3, D, 251), np.nan)
    daily, components = [], []
    last_loading = None
    last_weights = np.full((3, N), np.nan)
    fold_year = []
    max_svd_error = 0.
    started = time.time()
    for day, (date, frame) in enumerate(ranks.groupby('Date', sort=True)):
        frame = frame.sort_values('SecuritiesCode')
        universe = frame.SecuritiesCode.to_numpy()
        np.testing.assert_array_equal(universe, source_groups[date])
        pos = np.searchsorted(codes, universe)
        g = frame.g.to_numpy()
        arank, aw = rank_weights(g, universe)
        np.testing.assert_array_equal(arank, frame.Rank.to_numpy())
        keep, x, v, l, info = pca_asof(panel, date, universe)
        assert info['window_end'] == str(date.date()) and len(l) >= 6
        fitpos = pos[keep]
        beta_a = aw[keep] @ v
        var_a = l * beta_a ** 2
        selected_k = int(np.argmax(var_a))
        g_all = [g]
        slopes, fractions = [], []
        for k in [5, selected_k]:
            transformed, slope, fraction = ablate(g[keep], v[:, k])
            b = g.copy()
            b[keep] = transformed
            g_all.append(b)
            slopes.append(slope)
            fractions.append(fraction)
        weights = []
        rows = []
        for variant, score in enumerate(g_all):
            rank, w = rank_weights(score, universe)
            weights.append(w)
            predictions[variant, day, pos] = score
            rank_trace[variant, day, pos] = rank
            beta = w[keep] @ v
            exposures[variant, day, :len(l)] = beta
            direct = float(np.var(x @ w[keep], ddof=1))
            decomp = float(np.sum(l * beta ** 2))
            np.testing.assert_allclose(direct, decomp, atol=1e-16, rtol=1e-10)
            whole = np.zeros(N)
            whole[pos] = w
            turn = float(np.sum(np.abs(whole - last_weights[variant]))) if day else np.nan
            last_weights[variant] = whole
            rows.append({'variant': VARIANTS[variant], 'Date': str(date.date()),
                'ValidationYear': int(frame.ValidationYear.iloc[0]),
                'covered_gross': float(np.sum(np.abs(w[keep]))),
                'covered_net': float(np.sum(w[keep])), 'covered_total_variance': direct,
                'PC6_exposure': float(beta[5]), 'PC6_variance': float(l[5] * beta[5] ** 2),
                'selected_max_exposure': float(beta[selected_k]),
                'selected_max_variance': float(l[selected_k] * beta[selected_k] ** 2),
                'rank_correlation_with_A': spearman(-rank.astype(float), -arank.astype(float)),
                'changed_ranks': int(np.sum(rank != arank)),
                'replaced_long': int(np.sum((rank < 200) & (arank >= 200))),
                'replaced_short': int(np.sum((rank >= len(rank) - 200) & (arank < len(rank) - 200))),
                'weight_change_L1_from_A': float(np.sum(np.abs(w - aw))),
                'daily_target_weight_change_L1': turn})
        daily.extend(rows)
        pca_eligible[day, fitpos] = True
        eigenvalues[day, :len(l)] = l
        loadings[0, day, fitpos] = v[:, 5]
        loadings[1, day, fitpos] = v[:, selected_k]
        similarity = np.nan
        if last_loading is not None:
            common = np.isfinite(last_loading) & np.isfinite(loadings[0, day])
            u0, u1 = last_loading[common], loadings[0, day, common]
            similarity = float(abs(u0 @ u1) / (np.linalg.norm(u0) * np.linalg.norm(u1)))
        last_loading = loadings[0, day].copy()
        gap6 = float(min(l[4] - l[5], l[5] - l[6]) / l[5])
        info.update(Date=str(date.date()), universe_stocks=len(universe), selected_max_PC=selected_k + 1,
            A_PC6_variance_share_covered=float(var_a[5] / var_a.sum()),
            A_max_variance_share_covered=float(var_a[selected_k] / var_a.sum()),
            PC6_neighbor_relative_gap=gap6, PC6_previous_abs_cosine=similarity,
            PC6_score_slope=slopes[0], max_score_slope=slopes[1],
            PC6_removed_score_variance_share=fractions[0], max_removed_score_variance_share=fractions[1])
        components.append(info)
        fold_year.append(int(frame.ValidationYear.iloc[0]))
        # Independent direct SVD at predetermined dates, not chosen from results.
        if day in [0, 244, 485, 727, 952]:
            _, s, vt = np.linalg.svd(x, full_matrices=False)
            np.testing.assert_allclose(l, s[:len(l)] ** 2 / (len(x) - 1), rtol=1e-8, atol=1e-13)
            cos = abs(float(vt[5] @ v[:, 5]))
            assert cos > 1 - 1e-8
            max_svd_error = max(max_svd_error, 1 - cos)
        if day % 50 == 0 or day == D - 1:
            print(json.dumps({'stage': 'generate', 'day': day + 1, 'of': D, 'date': str(date.date()),
                'seconds': round(time.time() - started, 1), 'covered_stocks': int(keep.sum())}), flush=True)
    # Full new predictions, ranks, directions, membership and risk decomposition.
    for year in range(2018, 2022):
        mask = np.asarray(fold_year) == year
        np.savez_compressed(out / 'traces' / f'predictions_{year}.npz',
            dates=dates[mask].values.astype('datetime64[D]'), codes=codes,
            variants=np.array(VARIANTS), scores=predictions[:, mask], ranks=rank_trace[:, mask],
            selected_loadings=loadings[:, mask], pca_eligible=pca_eligible[mask],
            eigenvalues=eigenvalues[mask], component_exposures=exposures[:, mask])
    pd.DataFrame(daily).to_csv(out / 'daily_diagnostics.csv', index=False)
    pd.DataFrame(components).to_csv(out / 'pca_windows.csv', index=False)
    audit = {'source_hashes': source_hashes, 'generated_days': D, 'ranked_rows': len(ranks),
        'raw_rows': 2332531, 'max_svd_direction_error': max_svd_error,
        'max_orthogonality_error': max(c['orth_error'] for c in components),
        'max_eigen_equation_error': max(c['eigen_equation_error'] for c in components),
        'market_closed_dates': closures, 'all_A_ranks_reproduced': True,
        'all_universes_match_contemporaneous_supervision_filter': True,
        'all_predictions_saved_before_target_read': True, 'target_used_in_generation': False,
        'formal_test_used': False, 'source_commit': '95592531e6c8132988d43cc87cd563da653ad28e',
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'pandas': pd.__version__, 'scipy': scipy.__version__}}
    save(out / 'generation_audit.json', audit)
    return audit


def evaluate(args):
    out = args.out
    if not (out / 'generation_audit.json').exists():
        raise ValueError('Generate and save predictions before reading any Target')
    with zipfile.ZipFile(args.zip) as z:
        with z.open(raw_member(z)) as stream:
            labels = pd.read_csv(stream, usecols=['Date', 'SecuritiesCode', 'Target'], parse_dates=['Date'])
    label_groups = {d: g.set_index('SecuritiesCode').Target for d, g in labels.groupby('Date')}
    result_rows, exact_frames = [], {v: [] for v in VARIANTS}
    trace_records = []
    for year in range(2018, 2022):
        path = out / 'traces' / f'predictions_{year}.npz'
        trace_records.append({'path': 'traces/' + path.name, 'size': path.stat().st_size, 'sha256': sha(path)})
        with np.load(path, allow_pickle=False) as z:
            codes, dates, scores, ranks = z['codes'], z['dates'], z['scores'], z['ranks']
            for day, date in enumerate(dates):
                date = pd.Timestamp(date)
                eligible = ranks[0, day] != MISSING_RANK
                daycodes = codes[eligible]
                target = label_groups[date].reindex(daycodes).to_numpy()
                for k, variant in enumerate(VARIANTS):
                    r, s = ranks[k, day, eligible], scores[k, day, eligible]
                    spread, missing = spread_from_ranks(r, target)
                    _, w = rank_weights(s, daycodes)
                    if not missing:
                        np.testing.assert_allclose(spread, 400 * np.nansum(w * target), rtol=1e-12, atol=1e-12)
                    result_rows.append({'Date': str(date.date()), 'ValidationYear': year, 'variant': variant,
                        'OfficialDailySpread': spread, 'SelectedMissingTargets': missing,
                        'RankIC': spearman(s, target), 'GrossOneScaledSpread': spread / 400})
                    exact_frames[variant].append(pd.DataFrame({'Date': date, 'Rank': r, 'Target': target}))
    d = pd.DataFrame(result_rows)
    reference = pd.read_csv(ROOT / 'references/baseline_daily_reference.csv')
    a = d.loc[d.variant.eq('A')].merge(reference, on='Date', suffixes=('', '_saved'), validate='one_to_one')
    assert len(a) == 953 and a.SelectedMissingTargets.eq(0).all()
    max_err = float(np.max(np.abs(a.OfficialDailySpread - a.OfficialDailySpread_saved)))
    np.testing.assert_allclose(a.OfficialDailySpread, a.OfficialDailySpread_saved, atol=1e-12, rtol=1e-12)
    availability = d.pivot(index='Date', columns='variant', values='SelectedMissingTargets')
    common = availability.index[availability.eq(0).all(axis=1)]
    d['CommonScorable'] = d.Date.isin(common)
    d.to_csv(out / 'daily_scores.csv', index=False)
    summary = []
    for period in ['all', 2018, 2019, 2020, 2021]:
        sample = d.loc[d.CommonScorable]
        if period != 'all':
            sample = sample.loc[sample.ValidationYear.eq(period)]
        for variant, group in sample.groupby('variant', sort=False):
            s = group.OfficialDailySpread
            summary.append({'period': str(period), 'variant': variant, 'days': len(s),
                'Sharpe': float(s.mean() / s.std(ddof=1)), 'RankIC': float(group.RankIC.mean()),
                'RankIC_days': int(group.RankIC.notna().sum()), 'mean_spread': float(s.mean()),
                'sd_spread': float(s.std(ddof=1)), 'mean_gross_one_scaled_spread': float(s.mean() / 400)})
    s = pd.DataFrame(summary)
    s.to_csv(out / 'summary.csv', index=False)
    sys.path.insert(0, str(ROOT / 'references'))
    from official_metric_reference import calc_spread_return_sharpe
    checks = {}
    for variant in VARIANTS:
        full = pd.concat(exact_frames[variant], ignore_index=True)
        full = full.loc[full.Date.dt.strftime('%Y-%m-%d').isin(common)]
        exact = float(calc_spread_return_sharpe(full))
        reported = float(s.loc[s.period.eq('all') & s.variant.eq(variant), 'Sharpe'].iloc[0])
        np.testing.assert_allclose(exact, reported, atol=1e-12, rtol=0)
        checks[variant] = {'official_reference_sharpe': exact, 'max_difference': abs(exact - reported)}
    wide = d.loc[d.CommonScorable].pivot(index='Date', columns='variant', values='OfficialDailySpread')[VARIANTS]
    intervals = stationary_intervals(wide.to_numpy())
    diag = pd.read_csv(out / 'daily_diagnostics.csv')
    windows = pd.read_csv(out / 'pca_windows.csv')
    effects = {}
    for variant, pc in [('B_PC6', 'PC6'), ('B_MAX', 'selected_max')]:
        orig = diag.loc[diag.variant.eq('A')].set_index('Date')
        b = diag.loc[diag.variant.eq(variant)].set_index('Date')
        riskcol = pc + '_variance'
        ratio = b[riskcol] / orig[riskcol]
        effects[variant] = {'days_selected_component_risk_decreased': int((ratio < 1).sum()),
            'median_selected_component_variance_ratio_B_over_A': float(ratio.median()),
            'mean_removed_score_variance_share': float(windows[
                'PC6_removed_score_variance_share' if variant == 'B_PC6' else 'max_removed_score_variance_share'].mean()),
            'mean_replaced_long': float(b.replaced_long.mean()), 'mean_replaced_short': float(b.replaced_short.mean()),
            'mean_rank_correlation_with_A': float(b.rank_correlation_with_A.mean()),
            'mean_target_weight_change_L1': float(b.daily_target_weight_change_L1.mean()),
            'mean_covered_total_variance_ratio_B_over_A': float((b.covered_total_variance / orig.covered_total_variance).mean())}
    result = {'scope': 'Reused historical validation; causal generation, no new holdout claim; not the same fixed PC6 over time',
        'original_days': 953, 'common_scorable_days': len(common),
        'excluded_missing_label_dates': availability.loc[~availability.index.isin(common)].reset_index().to_dict('records'),
        'summary': records(s), 'paired_stationary_bootstrap': {'samples': 5000, 'mean_block_days': 20,
            'seed': 20260928, 'intervals': intervals, 'caveat': 'Exploratory conditional intervals; do not correct repeated validation use or variant selection'},
        'effects': effects,
        'coverage': {'pca_stock_min': int(windows.pca_stocks.min()), 'pca_stock_median': float(windows.pca_stocks.median()),
            'A_gross_min': float(diag.loc[diag.variant.eq('A'), 'covered_gross'].min()),
            'A_gross_mean': float(diag.loc[diag.variant.eq('A'), 'covered_gross'].mean()),
            'window_days_min': int(windows.window_days.min()), 'shorter_than_252_days': int(windows.window_days.lt(252).sum())},
        'direction_stability': {'PC6_previous_abs_cosine_median': float(windows.PC6_previous_abs_cosine.median()),
            'PC6_previous_abs_cosine_below_half_days': int(windows.PC6_previous_abs_cosine.lt(.5).sum()),
            'B_MAX_selected_PC_counts': {str(k): int(v) for k, v in windows.selected_max_PC.value_counts().sort_index().items()}},
        'baseline_daily_spread_max_error': max_err, 'official_reference_checks': checks,
        'formal_test_used': False, 'costs_included': False, 'baseline_changed': False,
        'large_trace_files': trace_records}
    save(out / 'results.json', result)
    print(json.dumps({'stage': 'evaluated', 'summary': records(s.loc[s.period.eq('all')]),
                      'intervals': intervals, 'missing_dates': result['excluded_missing_label_dates']}, ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--zip', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--stage', choices=['generate', 'evaluate', 'all'], default='all')
    a = p.parse_args()
    if a.stage in ['generate', 'all']:
        generate(a)
    if a.stage in ['evaluate', 'all']:
        evaluate(a)


if __name__ == '__main__':
    main()
