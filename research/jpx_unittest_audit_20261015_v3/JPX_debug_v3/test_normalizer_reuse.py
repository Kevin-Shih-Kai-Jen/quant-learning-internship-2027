"""Regression checks for the optional std/mean branch added during this audit."""
import unittest
import numpy as np
from test_audit import features, raw_frame, FEATURES


class NormalizerReuseTests(unittest.TestCase):
    def test_nonzero_scalar_std_returns_the_transformed_frame(self):
        frame = raw_frame()
        result = features.normalize_data(frame, FEATURES, std=2.0, mean=10.0)
        self.assertIsNotNone(result, "The supplied-std branch must return a result")
        transformed, mean, std = result
        np.testing.assert_allclose(transformed[FEATURES], (frame[FEATURES] - 10.0) / 2.0)

    def test_fitted_featurewise_metadata_can_be_reused(self):
        frame = raw_frame()
        transformed, means, stds = features.normalize_data(frame, FEATURES)
        # Isolate vector handling from the separate zero-variance metadata bug.
        safe_scales = np.where(stds == 0, 1.0, stds)
        result = features.normalize_data(frame, FEATURES, std=safe_scales, mean=means)
        self.assertIsNotNone(result)
        replayed, _, _ = result
        np.testing.assert_allclose(replayed[FEATURES], transformed[FEATURES], atol=1e-7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
