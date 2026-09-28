import unittest
import numpy as np
import pandas as pd
from run_ab import ablate, pca_asof, rank_weights, spread_from_ranks, build_returns, W


class DiagnosticTests(unittest.TestCase):
    def test_ablation_removes_only_centered_linear_component(self):
        rng = np.random.default_rng(12)
        v = rng.normal(size=500)
        q = v - v.mean()
        noise = rng.normal(size=500)
        noise -= noise.mean()
        noise -= q * (q @ noise) / (q @ q)
        score = 4 + 2.5 * q + noise
        b, slope, _ = ablate(score, v)
        np.testing.assert_allclose(b, 4 + noise, atol=1e-12)
        self.assertAlmostEqual(slope, 2.5)

    def test_arbitrary_loading_sign_and_score_intercept_do_not_change_ranks(self):
        rng = np.random.default_rng(13)
        score, v = rng.normal(size=(2, 500))
        a = ablate(score, v)[0]
        b = ablate(score + 7, -v)[0]
        np.testing.assert_allclose(a + 7, b, atol=1e-12)
        np.testing.assert_array_equal(rank_weights(a, np.arange(500))[0],
                                      rank_weights(b, np.arange(500))[0])

    def test_future_returns_and_future_missingness_cannot_change_fit(self):
        rng = np.random.default_rng(14)
        dates = pd.bdate_range('2020-01-01', periods=145)
        cols = list(range(420))
        panel = pd.DataFrame(rng.normal(size=(145, 420)) * .01, index=dates, columns=cols)
        a = pca_asof(panel, dates[125], cols)
        panel.loc[dates[126]:] = np.nan
        b = pca_asof(panel, dates[125], cols)
        for j in range(4):
            np.testing.assert_allclose(a[j], b[j], atol=1e-12)
        self.assertEqual(a[4]['window_end'], str(dates[125].date()))

    def test_dual_covariance_agrees_with_direct_svd(self):
        rng = np.random.default_rng(15)
        panel = pd.DataFrame(rng.normal(size=(130, 420)), index=pd.bdate_range('2020-01-01', periods=130))
        keep, x, v, l, _ = pca_asof(panel, panel.index[-1], list(panel.columns))
        _, s, vt = np.linalg.svd(x, full_matrices=False)
        np.testing.assert_allclose(l, s[:len(l)]**2 / 129, rtol=1e-10)
        self.assertGreater(abs(vt[5] @ v[:, 5]), 1 - 1e-10)

    def test_official_rank_weights_and_spread(self):
        score = np.arange(500, dtype=float)
        codes = np.arange(1000, 1500)
        ranks, w = rank_weights(score, codes)
        target = np.arange(500, dtype=float) / 10000
        spread, missing = spread_from_ranks(ranks, target)
        self.assertEqual(missing, 0)
        self.assertAlmostEqual(spread, 400 * (w @ target))
        self.assertAlmostEqual(w[w > 0].sum(), .5)
        self.assertAlmostEqual(w[w < 0].sum(), -.5)
        self.assertEqual(np.count_nonzero(w), 400)
        self.assertAlmostEqual(w[-1], W[0] / 600)
        tied, _ = rank_weights(np.ones(500), codes[::-1])
        np.testing.assert_array_equal(tied, np.arange(500)[::-1])

    def test_missing_selected_target_is_not_silently_zero(self):
        ranks, w = rank_weights(np.arange(500, dtype=float), np.arange(500))
        y = np.zeros(500)
        y[0] = np.nan
        spread, missing = spread_from_ranks(ranks, y)
        self.assertTrue(np.isnan(spread))
        self.assertEqual(missing, 1)
        y[0] = 0
        y[250] = np.nan
        self.assertEqual(spread_from_ranks(ranks, y), (0., 0))

    def test_no_price_gap_bridging_or_target_in_generation(self):
        dates = pd.bdate_range('2020-01-01', periods=4)
        rows = []
        for code, prices in [(1, [100, np.nan, 110, 121]), (2, [100, 101, 102, 103])]:
            for d, c in zip(dates, prices):
                rows.append(dict(Date=d, SecuritiesCode=code, Close=c, Volume=1, AdjustmentFactor=1))
        raw = pd.DataFrame(rows)
        r, _ = build_returns(raw)
        self.assertTrue(np.isnan(r.loc[dates[2], 1]))
        self.assertAlmostEqual(r.loc[dates[3], 1], .1)
        raw['Target'] = 0
        with self.assertRaises(AssertionError):
            build_returns(raw)


if __name__ == '__main__':
    unittest.main()
