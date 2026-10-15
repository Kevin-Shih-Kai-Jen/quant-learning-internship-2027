# v3 regression copy: updated model module and metric return contract only.
# Original suites are retained at ../JPX_debug and ../JPX_debug_v2 on Desktop.
"""JPX debug v2: small, CPU-only unittest exercises; production code is unchanged.

Run from any directory:
    python -B /path/to/JPX_debug_v2/test_jpx_debug_v2.py
    python -B /path/to/JPX_debug_v2/test_jpx_debug_v2.py PredictionTests

This file lives in JPX/debug/JPX_debug_v2/ and always imports JPX/codes/.
An explicit JPX_CODES_DIR can select a diagnostic copy. There is no automatic
fallback to an old source snapshot. The active path is checked after imports.

Contracts: predictions are a 1-D ndarray in non-shuffled sample order;
get_eval_df matches each window's final row; Weight is signed, with +0.5/-0.5
per day; Sharpe uses mean / sample std (ddof=1), without annualization;
Rank IC is the unweighted mean of daily Spearman correlations.
overall_performance returns a dict with sharpe and rank_IC keys (2026-10-15 API).

These are synthetic correctness checks, not a validation of market data,
train/validation splits, AdjustmentFactor timing, or investment performance.
"""
from __future__ import annotations

import contextlib
import io
import unittest
from unittest.mock import patch
import warnings

import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
CODES_DIR = Path(os.environ.get("JPX_CODES_DIR", HERE.parent.parent / "codes")).expanduser().resolve()
if not (CODES_DIR / "deep_learning.py").is_file():
    raise RuntimeError(f"找不到待測程式：{CODES_DIR}。debug 和 codes 應放在同一層。")
sys.path.insert(0, str(CODES_DIR))


def verify_source_modules(*modules):
    """Refuse stale snapshots or unrelated modules already cached in Python."""
    for module in modules:
        actual = Path(module.__file__).resolve()
        expected = CODES_DIR / (module.__name__ + ".py")
        if actual != expected:
            raise RuntimeError(f"載入錯誤的程式：{actual}；預期：{expected}")

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

import deep_learning as learning
import models
import evaluation as metrics
import features

verify_source_modules(learning, metrics, features, models)

torch.set_num_threads(1)


def mixed_stocks() -> pd.DataFrame:
    """Three unequal histories; nonconsecutive labels and scrambled row order."""
    rows = []
    for code, count in [(1001, 4), (2002, 3), (3003, 1)]:
        for day in range(1, count + 1):
            rows.append({"SecuritiesCode": code,
                         "Date": pd.Timestamp(2024, 1, day),
                         "Close": float(code + day),
                         "Target": float(code * 10 + day)})
    df = pd.DataFrame(rows)
    df.index = [91, 13, 78, 24, 51, 69, 32, 88]
    return df.iloc[[7, 3, 5, 0, 6, 2, 4, 1]].copy()


def one_market_day(n: int = 402) -> pd.DataFrame:
    """Distinct predictions make tie-breaking irrelevant to weight assertions."""
    df = pd.DataFrame({
        "SecuritiesCode": np.arange(n) + 1000,
        "Date": pd.Timestamp("2024-01-01"),
        "Prediction": np.arange(n, 0, -1, dtype=float),
        "Target": np.linspace(-0.03, 0.04, n),
    })
    return df.sample(frac=1, random_state=7)


def daily_return_frame(returns=(0.01, 0.02, 0.03)) -> pd.DataFrame:
    """Signed holdings yield exactly the requested return on each date."""
    rows = []
    for day, value in enumerate(returns, start=1):
        rows.extend([
            {"Date": pd.Timestamp(2024, 1, day), "SecuritiesCode": 1001,
             "Weight": 0.5, "Target": 2 * value},
            {"Date": pd.Timestamp(2024, 1, day), "SecuritiesCode": 2002,
             "Weight": -0.5, "Target": 0.0},
        ])
    return pd.DataFrame(rows, columns=["Date", "SecuritiesCode", "Weight", "Target"])


class EchoModel(nn.Module):
    """Predict the first feature on the final day; no learned ranking involved."""
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.calls = 0
        self.grad_modes = []

    def forward(self, x):
        self.calls += 1
        self.grad_modes.append(torch.is_grad_enabled())
        return x[:, -1, :1] + self.bias


def prediction_loader():
    # Batch sizes 2, 2, 1; predictions are deliberately not sorted numerically.
    values = np.array([4., 1., 9., 2., 7.], dtype=np.float32)
    X = values.reshape(5, 1, 1)
    y = values + np.array([1., 1., 3., 3., 5.], dtype=np.float32)
    return learning.make_loader(X, y, batch_size=2, shuffle=False)


class PredictionTests(unittest.TestCase):
    """第一輪：batch 預測合併、回傳型別、loss 與空資料。"""

    def test_01_all_batches_return_in_original_order(self):
        result = learning.evaluation(EchoModel(), prediction_loader(), nn.L1Loss())
        self.assertIsInstance(result, tuple, "evaluation 應回傳 (loss, predictions)。")
        self.assertEqual(len(result), 2)
        loss, pred = result
        self.assertIsInstance(pred, np.ndarray, "交給 pandas 前請轉成 ndarray。")
        self.assertEqual(pred.shape, (5,), "最後一批只有一筆，也不能丟掉或變成純量。")
        np.testing.assert_allclose(pred, [4, 1, 9, 2, 7],
                                   err_msg="要合併每個 batch，而非只取最後一批。")
        self.assertTrue(np.issubdtype(pred.dtype, np.floating))

    def test_02_loss_is_sample_weighted_not_batch_weighted(self):
        loss, _ = learning.evaluation(EchoModel(), prediction_loader(), nn.L1Loss())
        self.assertAlmostEqual(loss, 2.6, places=6,
                               msg="五筆誤差為 1,1,3,3,5；平均應為 2.6。")

    def test_03_eval_mode_no_grad_and_no_parameter_update(self):
        model = EchoModel().train()
        before = model.bias.detach().clone()
        learning.evaluation(model, prediction_loader(), nn.L1Loss())
        self.assertFalse(model.training)
        self.assertEqual(model.grad_modes, [False, False, False])
        torch.testing.assert_close(model.bias, before)

    def test_04_one_sample_still_returns_one_dimensional_array(self):
        loader = learning.make_loader(np.array([[[8.]]]), [8.], batch_size=1)
        loss, pred = learning.evaluation(EchoModel(), loader, nn.L1Loss())
        self.assertEqual(pred.shape, (1,))
        np.testing.assert_allclose(pred, [8.])
        self.assertEqual(loss, 0.0)

    def test_05_empty_loader_raises_value_error_before_concatenating(self):
        loader = DataLoader(TensorDataset(torch.empty(0, 1, 1), torch.empty(0, 1)),
                            batch_size=2)
        with self.assertRaises(ValueError):
            learning.evaluation(EchoModel(), loader, nn.L1Loss())

    def test_06_default_loader_preserves_sample_order(self):
        loader = learning.make_loader(np.zeros((3, 2, 1)), [10, 20, 30], batch_size=2)
        self.assertIsInstance(loader.sampler, SequentialSampler)
        targets = torch.cat([y.reshape(-1) for _, y in loader]).numpy()
        np.testing.assert_array_equal(targets, [10, 20, 30])


class AlignmentTests(unittest.TestCase):
    """第二輪：外部建立 eval_df 可以，但列數、順序與窗口終點要一致。"""

    def test_01_scrambled_rows_align_to_window_end(self):
        df = mixed_stocks()
        saved = df.copy(deep=True)
        pred = np.array([11., 12., 13., 21., 22.])
        out = metrics.get_eval_df(df, pred, 2)
        self.assertEqual(out["SecuritiesCode"].tolist(), [1001] * 3 + [2002] * 2)
        self.assertEqual(pd.to_datetime(out["Date"]).dt.day.tolist(), [2, 3, 4, 2, 3])
        np.testing.assert_allclose(out["Target"], [10012, 10013, 10014, 20022, 20023])
        np.testing.assert_array_equal(out["Prediction"], pred)
        pd.testing.assert_frame_equal(df, saved)

    def test_02_window_one_retains_every_stock_row(self):
        df = mixed_stocks()
        pred = np.arange(len(df), dtype=float)
        out = metrics.get_eval_df(df, pred, 1)
        self.assertEqual(len(out), 8, "T=1 時，每支股票的第一筆也有預測。")
        self.assertEqual(out["SecuritiesCode"].tolist(), [1001] * 4 + [2002] * 3 + [3003])
        np.testing.assert_array_equal(out["Prediction"], pred)

    def test_03_exact_window_retains_one_row_and_short_stocks_have_none(self):
        out = metrics.get_eval_df(mixed_stocks(), np.array([0.7]), 4)
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["SecuritiesCode"], 1001)
        self.assertEqual(pd.Timestamp(out.iloc[0]["Date"]).day, 4)

    def test_04_predictions_of_wrong_length_are_rejected(self):
        with self.assertRaises(ValueError):
            metrics.get_eval_df(mixed_stocks(), np.zeros(4), 2)  # needs 5

    def test_05_actual_sequence_loader_predictions_and_metadata_agree(self):
        df = mixed_stocks()
        X, y = features.create_time_sequence(df, 2, ["Close"])
        loader = learning.make_loader(X, y, batch_size=2, shuffle=False)
        _, pred = learning.evaluation(EchoModel(), loader, nn.L1Loss())
        out = metrics.get_eval_df(df, pred, 2)
        np.testing.assert_allclose(out["Target"], y)
        np.testing.assert_allclose(out["Prediction"], [1003, 1004, 1005, 2004, 2005])


class WeightTests(unittest.TestCase):
    """第三輪：每日排名、200 多/200 空、符號與 2:1 相對權重。"""

    def test_01_ranking_restarts_on_each_date(self):
        df = pd.DataFrame({"Date": ["A", "B", "A", "B"],
                           "Prediction": [10., 100., 30., 50.]}, index=[9, 3, 8, 2])
        out = metrics.rank(df.copy(), "Prediction").reindex(df.index)
        np.testing.assert_allclose(out["Prediction_rank"], [2, 1, 1, 2])

    def test_02_signed_counts_totals_and_endpoint_ratio(self):
        out = metrics.calculate_weights(one_market_day()).set_index("SecuritiesCode")
        w = out["Weight"]
        self.assertEqual((w > 0).sum(), 200, "只能有 200 檔多頭。")
        self.assertEqual((w < 0).sum(), 200, "空頭要是負權重。")
        self.assertEqual((w == 0).sum(), 2)
        self.assertAlmostEqual(w[w > 0].sum(), 0.5)
        self.assertAlmostEqual(w[w < 0].sum(), -0.5)
        self.assertAlmostEqual(w.loc[1000], 1 / 300)
        self.assertAlmostEqual(w.loc[1199], 1 / 600)
        self.assertAlmostEqual(w.loc[1202], -1 / 600)
        self.assertAlmostEqual(w.loc[1401], -1 / 300)

    def test_03_all_rank_positions_have_the_linear_weights(self):
        out = metrics.calculate_weights(one_market_day()).set_index("SecuritiesCode")
        # Closed-form independent oracle, not a call to the function under test.
        for i in [0, 17, 99, 155, 199]:
            expected = (2 - i / 199) / 600
            self.assertAlmostEqual(out.loc[1000 + i, "Weight"], expected)
            self.assertAlmostEqual(out.loc[1401 - i, "Weight"], -expected)

    def test_04_less_than_400_stocks_is_rejected(self):
        with self.assertRaises(ValueError):
            metrics.calculate_weights(one_market_day(399))

    def test_05_signed_contribution_uses_the_Weight_column(self):
        df = pd.DataFrame({"Weight": [0.5, -0.5, 0.0],
                           "Target": [0.04, -0.02, 100.0]})
        self.assertAlmostEqual(metrics.one_day_earning(df), 0.03,
                               msg="空頭遇到負報酬應有正貢獻；欄位名稱區分大小寫。")


class SharpeTests(unittest.TestCase):
    """第四輪：跨日期 mean/std，不是股票間 std 或跨日期 sum/std。"""

    def test_01_daily_mean_over_sample_std(self):
        df = daily_return_frame().sample(frac=1, random_state=8)
        self.assertAlmostEqual(metrics.calculate_sharpe(df), 2.0, places=5,
                               msg="每日報酬 .01,.02,.03：平均 .02、樣本 std .01。")

    def test_02_formula_isolated_from_one_day_earning(self):
        # Stub only the already-tested daily aggregation, to expose formula bugs.
        with patch.object(metrics, "one_day_earning", side_effect=[0.01, 0.02, 0.03]):
            result = metrics.calculate_sharpe(daily_return_frame())
        self.assertAlmostEqual(result, 2.0, places=5)

    def test_03_base_number_is_daily_benchmark_return(self):
        with patch.object(metrics, "one_day_earning", side_effect=[0.01, 0.02, 0.03]):
            result = metrics.calculate_sharpe(daily_return_frame(), base_number=0.01)
        self.assertAlmostEqual(result, 1.0, places=5)

    def test_04_empty_dataframe_has_clear_error(self):
        with self.assertRaises(ValueError):
            metrics.calculate_sharpe(daily_return_frame(()))

    def test_05_single_day_is_undefined_not_a_finite_score(self):
        with patch.object(metrics, "one_day_earning", return_value=0.01):
            try:
                result = metrics.calculate_sharpe(daily_return_frame((0.01,)))
            except ValueError:
                return  # Either an explicit rejection or NaN is acceptable.
        self.assertTrue(np.isnan(result), "單日不足以估計樣本標準差。")

    def test_06_constant_daily_returns_are_explicitly_undefined(self):
        with patch.object(metrics, "one_day_earning", return_value=0.01):
            try:
                result = metrics.calculate_sharpe(daily_return_frame((0.01,) * 3))
            except ValueError:
                return
        self.assertTrue(np.isnan(result), "std=0 應回 NaN 或拒絕，不應靠 epsilon 產生巨大分數。")


class RankICTests(unittest.TestCase):
    """第五輪：Spearman 排名、同值平均排名、每天等權平均。"""

    def test_01_nonlinear_but_same_order_is_perfect(self):
        df = pd.DataFrame({"Date": ["A"] * 4,
                           "Prediction": [1., 2., 3., 4.],
                           "Target": [1., 4., 9., 16.]})
        self.assertAlmostEqual(metrics.calculate_rank_IC(df), 1.0,
                               msg="排序完全相同時 Rank IC=1；原始值 Pearson 不等於 1。")

    def test_02_opposite_order_is_negative_one(self):
        df = pd.DataFrame({"Date": ["A"] * 4,
                           "Prediction": [1., 2., 3., 4.],
                           "Target": [16., 9., 4., 1.]})
        self.assertAlmostEqual(metrics.calculate_rank_IC(df), -1.0)

    def test_03_ties_use_average_rank(self):
        df = pd.DataFrame({"Date": ["A"] * 4,
                           "Prediction": [1., 2., 2., 4.],
                           "Target": [1., 2., 3., 4.]})
        self.assertAlmostEqual(metrics.calculate_rank_IC(df), np.sqrt(0.9))

    def test_04_average_days_equally_with_different_stock_counts(self):
        df = pd.DataFrame({"Date": ["A"] * 4 + ["B"] * 3,
                           "Prediction": [1., 2., 3., 4., 10., 20., 30.],
                           "Target": [1., 4., 9., 16., -1., -2., -3.]})
        self.assertAlmostEqual(metrics.calculate_rank_IC(df), 0.0,
                               msg="每日 IC 為 +1 與 -1；平均為 0，不按股票數加權。")

    def test_05_missing_pairs_are_removed_before_ranking(self):
        df = pd.DataFrame({"Date": ["A"] * 5,
                           "Prediction": [1., 2., np.nan, 3., 4.],
                           "Target": [1., 4., 100., 9., np.nan]})
        self.assertAlmostEqual(metrics.calculate_rank_IC(df), 1.0)

    def test_06_constant_prediction_is_nan(self):
        df = pd.DataFrame({"Date": ["A"] * 3,
                           "Prediction": [1., 1., 1.], "Target": [1., 2., 3.]})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.assertTrue(np.isnan(metrics.calculate_rank_IC(df)))


def portfolio_history():
    """Four days, 402 stocks. Window=2 leaves three daily returns .01,.02,.03."""
    parts = []
    for day, r in enumerate([0.0, 0.01, 0.02, 0.03], start=1):
        df = one_market_day().sort_values("SecuritiesCode")
        df["Date"] = pd.Timestamp(2024, 1, day)
        df["Close"] = df["Prediction"]
        df["Target"] = np.r_[np.full(200, r), [0., 0.], np.full(200, -r)]
        parts.append(df.drop(columns="Prediction"))
    return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=11)


class PipelineTests(unittest.TestCase):
    """第六輪：overall_performance 需要先逐日配置權重，再交給指標。"""

    def test_01_overall_supplies_signed_weights_for_every_date(self):
        ready = pd.concat([
            one_market_day().assign(Date=pd.Timestamp("2024-01-01")),
            one_market_day().assign(Date=pd.Timestamp("2024-01-02")),
        ], ignore_index=True)
        seen = []

        def check_weighted_frame(frame, *args, **kwargs):
            self.assertIn("Weight", frame.columns,
                          "get_eval_df 只有預測，overall 還需先建立每日 Weight。")
            self.assertEqual(len(frame), len(ready))
            for _, day in frame.groupby("Date"):
                w = day["Weight"]
                self.assertEqual((w > 0).sum(), 200)
                self.assertEqual((w < 0).sum(), 200)
                self.assertAlmostEqual(w[w > 0].sum(), 0.5)
                self.assertAlmostEqual(w[w < 0].sum(), -0.5)
            seen.append(True)
            return 2.0

        with patch.object(metrics, "get_eval_df", return_value=ready), \
             patch.object(metrics, "calculate_sharpe", side_effect=check_weighted_frame), \
             patch.object(metrics, "calculate_rank_IC", return_value=0.4), \
             contextlib.redirect_stdout(io.StringIO()):
            metrics.overall_performance(ready, ready["Prediction"].to_numpy(), 1)
        self.assertEqual(len(seen), 1)

    def test_02_overall_returns_scores_in_declared_order(self):
        ready = one_market_day().assign(Weight=0.0)
        # Stub all lower-level computations to test only the public return value.
        with patch.object(metrics, "get_eval_df", return_value=ready), \
             patch.object(metrics, "calculate_weights", side_effect=lambda day: day), \
             patch.object(metrics, "calculate_sharpe", return_value=2.0), \
             patch.object(metrics, "calculate_rank_IC", return_value=0.4), \
             contextlib.redirect_stdout(io.StringIO()):
            result = metrics.overall_performance(ready, ready["Prediction"].to_numpy(), 1)
        self.assertIsInstance(result, dict)
        self.assertEqual(set(result), {"sharpe", "rank_IC"})
        np.testing.assert_allclose([result["sharpe"], result["rank_IC"]], [2.0, 0.4])

    def test_03_real_window_to_metrics_pipeline(self):
        raw = portfolio_history()
        X, y = features.create_time_sequence(raw, 2, ["Close"])
        loader = learning.make_loader(X, y, batch_size=128, shuffle=False)
        _, pred = learning.evaluation(EchoModel(), loader, nn.L1Loss())
        with contextlib.redirect_stdout(io.StringIO()):
            scores = metrics.overall_performance(raw, pred, 2)
        self.assertIsInstance(scores, dict)
        self.assertAlmostEqual(scores["sharpe"], 2.0, places=5)
        # Actual Target has 200 low ties, 2 middle ties, 200 high ties each day.
        pred_ranks = np.arange(402, 0, -1)
        target_ranks = np.r_[np.full(200, 302.5), [201.5] * 2, np.full(200, 100.5)]
        expected_ic = np.corrcoef(pred_ranks, target_ranks)[0, 1]
        self.assertAlmostEqual(scores["rank_IC"], expected_ic, places=6)


class RegressionTests(unittest.TestCase):
    """保留已修好的重要行為；採用新版 evaluation 的 tuple 回傳契約。"""

    def test_01_tensor_conversion_preserves_axes_and_float32(self):
        X = [np.array([[1., 2.], [3., 4.]]), np.array([[5., 6.], [7., 8.]])]
        out = learning.make_it_tensor(X)
        self.assertEqual(tuple(out.shape), (2, 2, 2))
        self.assertEqual(out.dtype, torch.float32)
        np.testing.assert_allclose(out.numpy(), X)

    def test_02_sequence_does_not_cross_stock_boundaries(self):
        X, y = features.create_time_sequence(mixed_stocks(), 3, ["Close"])
        np.testing.assert_allclose(X, [[[1002], [1003], [1004]],
                                      [[1003], [1004], [1005]],
                                      [[2003], [2004], [2005]]])
        np.testing.assert_allclose(y, [10013, 10014, 20023])

    def test_03_train_runs_full_epoch_with_sample_weighted_loss(self):
        model = EchoModel()
        result = learning.train(model, prediction_loader(), nn.L1Loss(),
                                torch.optim.SGD(model.parameters(), lr=0))
        self.assertEqual(model.calls, 3)
        self.assertAlmostEqual(result, 2.6, places=6)

    def test_04_actual_model_supports_current_42_day_configuration(self):
        torch.manual_seed(17)
        model = models.CNN_LSTM(num_features=6, cnn_out_channels=4,
                                  kernel_size=42, hidden_size=4).eval()
        with torch.no_grad():
            out = model(torch.ones(1, 42, 6))
        self.assertEqual(tuple(out.shape), (1, 1))
        self.assertTrue(torch.isfinite(out).all().item())


if __name__ == "__main__":
    print(f"Testing source: {CODES_DIR}", flush=True)
    unittest.main(verbosity=2)
