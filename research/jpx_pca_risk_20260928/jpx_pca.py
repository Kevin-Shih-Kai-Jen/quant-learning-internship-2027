"""PCA risk diagnostic for a date-by-stock return panel.

This module does not train an alpha model or choose portfolio weights.
All fitting uses observations through an explicit after-close as-of date.
Originally validated on synthetic data on 2026-09-27. The accompanying
run_analysis.py applies it to verified JPX archives on 2026-09-28.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class PCAFit:
    columns: list[str]
    mean: np.ndarray
    directions: np.ndarray  # N stocks by rank; each column is one unit direction
    eigenvalues: np.ndarray
    total_variance: float
    training: np.ndarray
    dates: pd.DatetimeIndex
    excluded: dict[str, str]
    cutoff: pd.Timestamp

    @property
    def explained_variance_ratio(self):
        return self.eigenvalues / self.total_variance


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def check_panel(panel):
    if not isinstance(panel.index, pd.DatetimeIndex):
        raise ValueError('Index must be a DatetimeIndex of trading dates.')
    if panel.index.has_duplicates or panel.columns.has_duplicates:
        raise ValueError('Duplicate dates or stock codes are not allowed.')
    if panel.index.hasnans or panel.empty:
        raise ValueError('Empty panel or missing dates.')
    if panel.index.tz is not None:
        raise ValueError('Use timezone-naive exchange trading dates.')
    if not panel.index.equals(panel.index.normalize()):
        raise ValueError('Use daily dates without intraday timestamps.')


def fit_pca(panel, asof, lookback=252, min_observations=60):
    """Covariance PCA: center using training means; do not divide by stock SD.

    Selection is based ONLY on the training slice. Missing/constant stocks
    are excluded and listed. Any excluded holding will cause risk() to fail.
    """
    check_panel(panel)
    if min_observations < 3 or lookback < min_observations:
        raise ValueError('Require lookback >= min_observations >= 3.')
    cutoff = pd.Timestamp(asof)
    if cutoff not in panel.index:
        raise ValueError('asof must be an exact observed trading date.')
    train = panel.sort_index().loc[:cutoff].tail(lookback).copy()
    if len(train) < min_observations:
        raise ValueError(f'Only {len(train)} training observations.')
    train.columns = train.columns.map(str)
    if train.columns.has_duplicates:
        raise ValueError('Stock codes collide after string conversion.')
    a = train.to_numpy(dtype=float)
    finite = np.isfinite(a).all(axis=0)
    variable = np.zeros(a.shape[1], dtype=bool)
    variable[finite] = np.std(a[:, finite], axis=0, ddof=1) > 0
    keep = finite & variable
    excluded = {
        str(c): ('missing_or_nonfinite_training_return' if not finite[j]
                 else 'zero_training_variance')
        for j, c in enumerate(train.columns) if not keep[j]
    }
    if keep.sum() < 2:
        raise ValueError('Fewer than two complete, variable stocks in training.')
    train = train.iloc[:, keep]
    a = train.to_numpy(dtype=float)
    mean = a.mean(axis=0)
    x = a - mean
    # SVD gives the same principal directions as covariance eigendecomposition.
    # It avoids constructing a large N-by-N covariance matrix.
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    tol = max(x.shape) * np.finfo(float).eps * s[0]
    rank = int(np.count_nonzero(s > tol))
    if rank == 0:
        raise ValueError('No estimable variance.')
    eigenvalues = s[:rank] ** 2 / (len(x) - 1)
    directions = vt[:rank].T.copy()
    # Sign is arbitrary. This convention makes exports reproducible.
    for k in range(rank):
        j = np.argmax(np.abs(directions[:, k]))
        if directions[j, k] < 0:
            directions[:, k] *= -1
    total = float(np.sum(x * x) / (len(x) - 1))
    return PCAFit(list(train.columns), mean, directions, eigenvalues,
                  total, a, train.index, excluded, cutoff)


def aligned_weights(fit, weights):
    """Keep capital weights as supplied; never normalize or silently drop risk."""
    weights = weights.copy()
    weights.index = weights.index.map(str)
    if weights.index.has_duplicates:
        raise ValueError('Duplicate stock codes in weights.')
    if not np.isfinite(weights.to_numpy(dtype=float)).all():
        raise ValueError('Weights must be finite.')
    outside = weights[(weights != 0) & ~weights.index.isin(fit.columns)]
    if len(outside):
        raise ValueError('Held stocks lack a fitted loading: ' + ', '.join(outside.index))
    return weights.reindex(fit.columns, fill_value=0).to_numpy(dtype=float)


def risk(fit, weights):
    """Variance decomposition of a fixed portfolio under training covariance."""
    w = aligned_weights(fit, weights)
    exposure = w @ fit.directions
    contribution = fit.eigenvalues * exposure ** 2
    x = fit.training - fit.mean
    direct = float(np.var(x @ w, ddof=1))
    decomposed = float(contribution.sum())
    variance_scale = max(fit.total_variance * float(w @ w), 1e-30)
    if abs(direct - decomposed) > 1e-10 * variance_scale:
        raise ArithmeticError('Direct and PCA variance do not reconcile.')
    identifiable = direct > 1e-14 * variance_scale
    shares = contribution / direct if identifiable else np.full(len(contribution), np.nan)
    table = pd.DataFrame({
        'PC': np.arange(1, len(contribution) + 1),
        'exposure': exposure,
        'variance_contribution': contribution,
        'portfolio_variance_share': shares,
    })
    null_norm = float(np.linalg.norm(w - fit.directions @ exposure))
    summary = {
        'training_covariance_variance': direct,
        'training_covariance_daily_volatility': float(np.sqrt(max(direct, 0))),
        'pca_variance_sum': decomposed,
        'reconciliation_absolute_error': abs(direct - decomposed),
        'gross_weight': float(np.abs(w).sum()), 'net_weight': float(w.sum()),
        'weight_norm_in_unestimated_null_space': null_norm,
        'shares_defined': bool(identifiable),
        'scope': 'fixed weights under sample training covariance; not realized strategy risk',
    }
    return table, summary


def project(fit, panel):
    """Use the FROZEN training means/directions; no refit on later data."""
    check_panel(panel)
    data = panel.copy()
    data.columns = data.columns.map(str)
    if data.columns.has_duplicates:
        raise ValueError('Duplicate normalized stock codes.')
    absent = set(fit.columns) - set(data.columns)
    if absent:
        raise ValueError('Missing stock columns: ' + ', '.join(sorted(absent)))
    x = data[fit.columns].to_numpy(dtype=float) - fit.mean
    if not np.isfinite(x).all():
        raise ValueError('Projection needs complete observations; no silent imputation.')
    scores = x @ fit.directions
    residual = x - scores @ fit.directions.T
    return scores, residual


def evaluate_fixed_weights(fit, later, weights):
    """Later-period diagnostic, not an untouched-test or dynamic-strategy claim.

    Outside training, PC scores may be correlated and null-space residuals
    can be nonzero. Include both in the variance reconciliation.
    """
    if len(later) < 3 or (later.index <= fit.cutoff).any():
        raise ValueError('Need at least three strictly later daily observations.')
    later = later.copy()
    later.columns = later.columns.map(str)
    scores, residual = project(fit, later)
    w = aligned_weights(fit, weights)
    beta = w @ fit.directions
    component_pnl = scores * beta
    residual_pnl = residual @ w
    all_parts = np.column_stack([component_pnl, residual_pnl])
    diagonal = float(np.var(all_parts, axis=0, ddof=1).sum())
    total_pnl = (later[fit.columns].to_numpy(dtype=float) - fit.mean) @ w
    direct = float(np.var(total_pnl, ddof=1))
    cross = direct - diagonal
    reconstruction_error = float(np.max(np.abs(all_parts.sum(axis=1) - total_pnl)))
    return {
        'n_observations': len(later),
        'first_date': str(later.index.min().date()),
        'last_date': str(later.index.max().date()),
        'frozen_weight_realized_variance': direct,
        'sum_individual_component_and_residual_variances': diagonal,
        'sum_cross_covariance_terms': cross,
        'residual_variance': float(np.var(residual_pnl, ddof=1)),
        'max_pnl_reconstruction_error': reconstruction_error,
        'scope': 'later period, fixed weights; untouched status is not verified',
    }


def power_direction(covariance, previous=None, seed=27, tolerance=1e-10,
                    max_iterations=10000):
    """Teaching implementation: multiply, remove known directions, normalize.

    previous has already-converged orthonormal eigenvectors as columns.
    Near-tied eigenvalues may require more iterations; production uses SVD.
    """
    cov = np.asarray(covariance, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or not np.isfinite(cov).all():
        raise ValueError('Covariance must be a finite square matrix.')
    if not np.allclose(cov, cov.T):
        raise ValueError('Covariance must be symmetric.')
    if np.linalg.eigvalsh(cov).min() < -1e-12 * max(np.linalg.norm(cov), 1e-30):
        raise ValueError('Covariance must be positive semidefinite.')
    q = np.empty((len(cov), 0)) if previous is None else np.asarray(previous, dtype=float)
    if q.ndim != 2 or q.shape[0] != len(cov):
        raise ValueError('previous must be N by number_of_known_directions.')
    if not np.allclose(q.T @ q, np.eye(q.shape[1]), atol=1e-8):
        raise ValueError('Previously found directions must be orthonormal.')
    v = np.random.default_rng(seed).normal(size=len(cov))
    v -= q @ (q.T @ v)
    if np.linalg.norm(v) < 1e-14:
        raise ValueError('No remaining direction.')
    v /= np.linalg.norm(v)
    for iteration in range(1, max_iterations + 1):
        u = cov @ v
        u -= q @ (q.T @ u)
        u -= q @ (q.T @ u)  # reorthogonalize against roundoff
        length = np.linalg.norm(u)
        if length <= np.finfo(float).eps * np.linalg.norm(cov):
            raise ValueError('Remaining variance is zero or not identifiable.')
        u /= length
        distance = min(np.linalg.norm(u - v), np.linalg.norm(u + v))
        v = u
        if distance < tolerance:
            if v[np.argmax(np.abs(v))] < 0:
                v *= -1
            return v, float(v @ cov @ v), iteration
    raise RuntimeError('Power iteration did not converge; do not report a result.')


def returns_from_raw(raw, through, factor_timing):
    """Historical JPX archive adapter. Full price-universe input is required.

    The 2026-09-10 v4 archive stamps AdjustmentFactor on the last cum-rights
    session and applies it next session. This convention is NOT assumed for
    other J-Quants feeds. Explicit opt-in is required by the CLI.
    Price returns exclude dividends. Target is never read.
    """
    if factor_timing != 'next-session':
        raise ValueError('Verify the archive convention and set next-session explicitly.')
    needed = ['Date', 'SecuritiesCode', 'Close', 'Volume', 'AdjustmentFactor']
    if not set(needed).issubset(raw.columns):
        raise ValueError('Missing raw fields: ' + ', '.join(sorted(set(needed) - set(raw.columns))))
    data = raw[needed].copy()
    data['Date'] = pd.to_datetime(data['Date'], errors='raise')
    data = data.loc[data.Date <= pd.Timestamp(through)]
    if data.empty or data[['Date', 'SecuritiesCode']].isna().any().any():
        raise ValueError('No usable raw rows or missing identifiers.')
    data['SecuritiesCode'] = data.SecuritiesCode.astype(str)
    if data.duplicated(['Date', 'SecuritiesCode']).any():
        raise ValueError('Duplicate raw date-stock rows.')
    for name in ['Close', 'Volume', 'AdjustmentFactor']:
        data[name] = pd.to_numeric(data[name], errors='raise')
    factors = data.AdjustmentFactor.to_numpy()
    if not np.isfinite(factors).all() or (factors <= 0).any():
        raise ValueError('Missing or invalid split factor; no default substitution.')
    traded = (data.Close.gt(0) & np.isfinite(data.Close) & data.Volume.gt(0)
              & np.isfinite(data.Volume))
    market_open = traded.groupby(data.Date).any().sort_index()
    calendar = pd.DatetimeIndex(market_open.index[market_open])
    close = data.pivot(index='Date', columns='SecuritiesCode', values='Close').reindex(calendar)
    volume = data.pivot(index='Date', columns='SecuritiesCode', values='Volume').reindex(calendar)
    factor = data.pivot(index='Date', columns='SecuritiesCode', values='AdjustmentFactor').reindex(calendar)
    valid = close.gt(0) & np.isfinite(close) & volume.gt(0) & np.isfinite(volume)
    # Reindex before shift: never bridge a stock's missing market day.
    panel = close / (close.shift(1) * factor.shift(1)) - 1
    panel = panel.where(valid & valid.shift(1, fill_value=False))
    return panel, {
        'factor_timing': 'next-session (historical v4 archive convention)',
        'calendar_rule': 'any positive-volume valid close in supplied FULL universe',
        'excluded_marketwide_nontrading_dates': [str(d.date()) for d in market_open.index[~market_open]],
        'return_definition': 'split-adjusted close-to-close simple PRICE return; excludes dividends',
        'zero_volume_rule': 'missing return, never zero-filled or treated as an executable trade',
    }


def read_weights(path, asof):
    data = pd.read_csv(path, dtype={'SecuritiesCode': str})
    required = {'Date', 'SecuritiesCode', 'Weight'}
    if not required.issubset(data.columns):
        raise ValueError('Weights CSV needs Date,SecuritiesCode,Weight.')
    data['Date'] = pd.to_datetime(data.Date, errors='raise')
    data = data.loc[data.Date == pd.Timestamp(asof)]
    if data.empty or data.SecuritiesCode.isna().any():
        raise ValueError('No valid weights for exact asof date.')
    if data.SecuritiesCode.duplicated().any():
        raise ValueError('Duplicate stock code in asof weights.')
    return data.set_index('SecuritiesCode').Weight.astype(float)


def demo():
    """The user's handwritten matrix ONLY; these are not JPX observations."""
    cov = np.array([[2., 1.], [1., 3.]])
    basis = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]]) * np.sqrt(1.5)
    x = basis @ np.linalg.cholesky(cov).T
    panel = pd.DataFrame(x, index=pd.date_range('2000-01-01', periods=4), columns=['A', 'B'])
    fit = fit_pca(panel, '2000-01-04', lookback=4, min_observations=3)
    v1, l1, n1 = power_direction(cov)
    v2, l2, n2 = power_direction(cov, v1[:, None])
    result = {
        'status': 'HANDWRITTEN_MATRIX_DEMONSTRATION_NOT_JPX',
        'covariance': cov.tolist(), 'directions': fit.directions.tolist(),
        'eigenvalues': fit.eigenvalues.tolist(),
        'explained_variance_ratio': fit.explained_variance_ratio.tolist(),
        'power_iteration': {'v1': v1.tolist(), 'v2': v2.tolist(),
                            'lambda1': l1, 'lambda2': l2, 'iterations': [n1, n2]},
        'portfolios': {},
    }
    for name, w in [('equal_long', [.5, .5]), ('equal_long_short', [.5, -.5])]:
        table, summary = risk(fit, pd.Series(w, index=['A', 'B']))
        result['portfolios'][name] = {'weights': w, 'summary': summary,
                                      'components': table.to_dict(orient='records')}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', action='store_true')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--prices', type=Path, help='Full JPX stock_prices.csv universe')
    source.add_argument('--returns', type=Path, help='Wide CSV: Date,stock1,stock2,...; decimal returns')
    parser.add_argument('--asof', help='After-close fit/weight date, YYYY-MM-DD; required except demo')
    parser.add_argument('--lookback', type=int, default=252)
    parser.add_argument('--min-observations', type=int, default=60)
    parser.add_argument('--factor-timing', choices=['next-session'])
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--evaluate-through', help='Optional later diagnostic; fixed weights only')
    parser.add_argument('--out', type=Path, help='NEW output directory; existing directories are refused')
    args = parser.parse_args()
    if args.demo:
        print(json.dumps(demo(), ensure_ascii=False, indent=2, allow_nan=False))
        return
    if not (args.prices or args.returns) or not args.asof or not args.out:
        parser.error('Require --prices/--returns, --asof, and --out.')
    if args.out.exists():
        parser.error('Output directory exists; use a new versioned path.')
    if args.evaluate_through and (not args.weights or pd.Timestamp(args.evaluate_through) <= pd.Timestamp(args.asof)):
        parser.error('Later evaluation requires weights and a date later than asof.')
    through = args.evaluate_through or args.asof
    input_path = args.prices or args.returns
    if args.prices:
        if args.factor_timing is None:
            parser.error('Verify factor timing and explicitly pass --factor-timing next-session.')
        raw = pd.read_csv(input_path, usecols=['Date', 'SecuritiesCode', 'Close', 'Volume', 'AdjustmentFactor'],
                          dtype={'SecuritiesCode': str})
        panel, adapter = returns_from_raw(raw, through, args.factor_timing)
    else:
        data = pd.read_csv(input_path, parse_dates=['Date'])
        if any(c in data.columns for c in ['Target', 'Close', 'SecuritiesCode']):
            raise ValueError('Expected a wide stock-return matrix, not raw prices/targets.')
        panel = data.set_index('Date')
        adapter = {'return_definition': 'caller-supplied realized simple returns in decimals',
                   'calendar_rule': 'caller-supplied market trading dates; not inferred'}
    fit = fit_pca(panel, args.asof, args.lookback, args.min_observations)
    rank = len(fit.eigenvalues)
    names = [f'PC{k+1}' for k in range(rank)]
    summary = {
        'status': 'COMPUTED_FROM_SUPPLIED_DATA', 'asof_after_close': args.asof,
        'training_start': str(fit.dates.min().date()), 'training_end': str(fit.dates.max().date()),
        'n_dates': len(fit.dates), 'n_stocks': len(fit.columns), 'numerical_rank': rank,
        'maximum_centered_rank': min(len(fit.dates)-1, len(fit.columns)),
        'lookback_requested': args.lookback, 'excluded_stocks': fit.excluded,
        'input_sha256': sha256(input_path), 'script_sha256': sha256(__file__),
        'total_cross_section_variance': fit.total_variance,
        'adapter': adapter, 'portfolio_risk_computed': False,
        'limits': [
            'No alpha model, ranking, or trading-rule change.',
            'Sample covariance risk; zero sample directions are not future risk-free.',
            'Full-universe historical membership and untouched holdout status are not verified.',
            'No automatic clipping, future fill, scaling, or missing-holding renormalization.',
            'Later diagnostics hold weights fixed; daily-changing strategy risk requires dated holdings.',
        ],
    }
    portfolio_table = None
    if args.weights:
        weights = read_weights(args.weights, args.asof)
        portfolio_table, portfolio_summary = risk(fit, weights)
        summary.update(portfolio_risk_computed=True, portfolio=portfolio_summary,
                       weights_sha256=sha256(args.weights))
        if args.evaluate_through:
            sorted_panel = panel.sort_index()
            later = sorted_panel.loc[(sorted_panel.index > fit.cutoff) & (sorted_panel.index <= pd.Timestamp(through))]
            summary['later_fixed_weight_diagnostic'] = evaluate_fixed_weights(fit, later, weights)
    # Validate before creating output; never overwrite a prior experiment.
    args.out.mkdir(parents=True)
    np.savez_compressed(args.out / 'pca_fit.npz', columns=np.array(fit.columns), mean=fit.mean,
                        directions=fit.directions, eigenvalues=fit.eigenvalues)
    pd.DataFrame({'PC': names, 'eigenvalue': fit.eigenvalues,
                  'explained_variance_ratio': fit.explained_variance_ratio,
                  'cumulative_ratio': np.cumsum(fit.explained_variance_ratio)}).to_csv(args.out/'components.csv', index=False)
    display_n = min(10, rank)
    pd.DataFrame(fit.directions[:, :display_n], index=fit.columns,
                 columns=names[:display_n]).rename_axis('SecuritiesCode').to_csv(args.out/'loadings_first10.csv')
    scores = (fit.training - fit.mean) @ fit.directions[:, :display_n]
    pd.DataFrame(scores, index=fit.dates, columns=names[:display_n]).rename_axis('Date').to_csv(args.out/'training_scores_first10.csv')
    if portfolio_table is not None:
        portfolio_table.to_csv(args.out/'portfolio_components.csv', index=False)
    (args.out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
