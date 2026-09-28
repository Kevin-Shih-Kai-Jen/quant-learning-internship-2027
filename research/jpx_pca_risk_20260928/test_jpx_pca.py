"""Numerical identities and time/data-boundary checks, not JPX performance tests."""
import unittest
import numpy as np
import pandas as pd
from jpx_pca import fit_pca, risk, project, power_direction, returns_from_raw, evaluate_fixed_weights


class PCAContractTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(15)
        self.panel = pd.DataFrame(rng.normal(size=(90, 5))*.01,
                                  index=pd.bdate_range('2020-01-01', periods=90),
                                  columns=list('ABCDE'))

    def test_handwritten_matrix_and_power_iteration(self):
        s = np.array([[2., 1.], [1., 3.]])
        v, lam, _ = power_direction(s)
        np.testing.assert_allclose(v, [.525731112119, .850650808352], atol=1e-8)
        self.assertAlmostEqual(lam, (5+np.sqrt(5))/2)
        q, l2, _ = power_direction(s, v[:, None])
        self.assertLess(abs(q@v), 1e-8)
        np.testing.assert_allclose(s@q, l2*q, atol=1e-8)

    def test_variance_identity_sign_invariance_and_permutation(self):
        fit = fit_pca(self.panel, self.panel.index[59], lookback=60)
        w = pd.Series([.3, -.2, .1, -.4, .2], index=list('ABCDE'))
        table, summary = risk(fit, w.iloc[::-1])
        direct = w.to_numpy() @ np.cov(self.panel.iloc[:60].to_numpy(), rowvar=False) @ w.to_numpy()
        self.assertAlmostEqual(summary['pca_variance_sum'], direct, places=14)
        fit.directions[:, 0] *= -1
        flipped, _ = risk(fit, w)
        np.testing.assert_allclose(table.variance_contribution, flipped.variance_contribution)

    def test_future_changes_do_not_change_fit_or_eligibility(self):
        asof = self.panel.index[59]
        first = fit_pca(self.panel, asof, lookback=60)
        changed = self.panel.copy()
        changed.iloc[60:, :] = np.nan
        changed.iloc[-1, 0] = 100000
        second = fit_pca(changed, asof, lookback=60)
        np.testing.assert_array_equal(first.mean, second.mean)
        np.testing.assert_array_equal(first.directions, second.directions)
        self.assertEqual(first.columns, second.columns)

    def test_unknown_or_excluded_holdings_cannot_disappear(self):
        panel = self.panel.copy()
        panel.iloc[10, 0] = np.nan
        fit = fit_pca(panel, panel.index[59], lookback=60)
        self.assertIn('A', fit.excluded)
        with self.assertRaises(ValueError):
            risk(fit, pd.Series({'A': .5, 'B': -.5}))
        with self.assertRaises(ValueError):
            risk(fit, pd.Series({'UNKNOWN': .1}))

    def test_centered_rank_limit(self):
        panel = self.panel.iloc[:4]
        fit = fit_pca(panel, panel.index[-1], lookback=4, min_observations=3)
        self.assertLessEqual(len(fit.eigenvalues), 3)
        risk(fit, pd.Series([.3, -.2, .1, -.4, .2], index=list('ABCDE')))

    def test_split_timing_missing_day_and_no_future_target(self):
        dates = pd.bdate_range('2020-01-01', periods=4)
        raw = pd.DataFrame({'Date': list(dates)*2, 'SecuritiesCode': ['A']*4+['B']*4,
                            'Close': [100, 102, 52, 53, 20, 21, 22, 23],
                            'Volume': [10]*8, 'AdjustmentFactor': [1, .5, 1, 1]+[1]*4,
                            'Target': [999]*8})
        panel, _ = returns_from_raw(raw, dates[-1], 'next-session')
        self.assertAlmostEqual(panel.loc[dates[2], 'A'], 52/(102*.5)-1)
        self.assertAlmostEqual(panel.loc[dates[1], 'A'], .02)
        missing = raw.drop(index=1)
        panel, _ = returns_from_raw(missing, dates[-1], 'next-session')
        self.assertTrue(np.isnan(panel.loc[dates[1], 'A']))
        self.assertTrue(np.isnan(panel.loc[dates[2], 'A']))
        self.assertAlmostEqual(panel.loc[dates[3], 'A'], 53/52-1)

    def test_frozen_projection_and_later_covariance_terms(self):
        fit = fit_pca(self.panel, self.panel.index[59], lookback=60)
        later = self.panel.iloc[60:].copy()
        later['B'] = later['A']*2
        w = pd.Series([.3, -.2, .1, -.4, .2], index=list('ABCDE'))
        scores, residual = project(fit, later)
        np.testing.assert_allclose(scores @ fit.directions.T + residual,
                                   later.to_numpy()-fit.mean, atol=1e-12)
        diagnostic = evaluate_fixed_weights(fit, later, w)
        self.assertAlmostEqual(diagnostic['frozen_weight_realized_variance'],
                               np.var(later.to_numpy()@w.to_numpy(), ddof=1), places=14)
        self.assertGreater(abs(diagnostic['sum_cross_covariance_terms']), 1e-9)

    def test_later_null_space_is_not_risk_free(self):
        panel = self.panel.iloc[:4]
        fit = fit_pca(panel, panel.index[-1], lookback=4, min_observations=3)
        w = pd.Series([.3, -.2, .1, -.4, .2], index=list('ABCDE'))
        diagnostic = evaluate_fixed_weights(fit, self.panel.iloc[4:30], w)
        self.assertGreater(diagnostic['residual_variance'], 0)
        self.assertLess(diagnostic['max_pnl_reconstruction_error'], 1e-12)


if __name__ == '__main__':
    unittest.main(verbosity=2)
