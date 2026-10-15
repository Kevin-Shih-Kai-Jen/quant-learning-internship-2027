"""Regression checks for revised interfaces and invalid-input boundaries."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from torch import nn

from test_audit import (
    application, features, learning, light_application, metrics, raw_frame,
    tiny_model,
)


class BugfixV6Tests(unittest.TestCase):
    def test_model2_sums_return_values_and_formats_percentage(self):
        with light_application() as (_, output, _):
            application.model_2()
        self.assertIn(
            "Total earnings (sum of daily returns): 24.00%",
            output.getvalue(),
        )

    def test_daily_earning_groups_dates_and_keeps_signed_contributions(self):
        frame = pd.DataFrame({
            "Date": ["2024-01-02", "2024-01-01", "2024-01-02", "2024-01-01"],
            "Weight": [0.5, 0.5, -0.5, -0.5],
            "Target": [0.1, 0.04, -0.06, -0.02],
        })
        actual = metrics.daily_earning(frame)
        np.testing.assert_allclose(actual, [0.03, 0.08])
        self.assertEqual(actual.dtype, np.dtype("float64"))

    def test_daily_earning_rejects_missing_dates_and_nonfinite_values(self):
        original = pd.DataFrame({"Date": ["2024-01-01"], "Weight": [1.0], "Target": [0.01]})
        for column, value in [("Date", None), ("Weight", np.nan), ("Target", np.inf)]:
            with self.subTest(column=column, value=value):
                frame = original.copy()
                frame.loc[0, column] = value
                with self.assertRaises(ValueError):
                    metrics.daily_earning(frame)

    def test_sequence_and_evaluation_reject_noninteger_windows(self):
        for window in [True, 1.5, "2"]:
            with self.subTest(window=window):
                with self.assertRaises(ValueError):
                    features.create_time_sequence(raw_frame(), window, ["Close"])
                with self.assertRaises(ValueError):
                    metrics.get_eval_df(raw_frame(), np.zeros(4), window)

    def test_eval_frame_rejects_nonfinite_or_matrix_predictions(self):
        for predictions in [np.array([0., 1., np.inf, 3.]), np.zeros((4, 1))]:
            with self.subTest(shape=predictions.shape):
                with self.assertRaises(ValueError):
                    metrics.get_eval_df(raw_frame(), predictions, 1)

    def test_invalid_epochs_do_not_step_optimizer(self):
        model = tiny_model()
        loader = learning.make_loader(np.ones((2, 6)), [0., 0.])
        optimizer = learning.torch.optim.SGD(model.parameters(), lr=0.01)
        for epochs in [0, -1, 1.5, True, "2"]:
            with self.subTest(epochs=epochs), patch.object(optimizer, "step") as step:
                with self.assertRaises(ValueError):
                    learning.train(model, loader, nn.L1Loss(), optimizer, epochs=epochs)
                step.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
