"""Independent bug reproductions and controls, using only synthetic CPU data.

Red tests assert intended behavior; known bugs are NOT marked expectedFailure.
Downstream orchestration tests stub training deliberately so a signature error
does not hide checkpoint/reporting bugs. They do not claim a full backtest ran.
"""
from __future__ import annotations

import ast
import contextlib
import inspect
import io
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
CODES = Path(os.environ.get("JPX_CODES_DIR", HERE.parent.parent / "codes")).expanduser().resolve()
if not (CODES / "deep_learning.py").is_file():
    raise RuntimeError(f"Cannot find JPX source: {CODES}")
sys.path.insert(0, str(CODES))

import numpy as np
import pandas as pd
import torch
from torch import nn
import data_loader
import data_processor as processor
import deep_learning as learning
import evaluation as metrics
import features
import JPX_deep_learning_models as application
import main as entrypoint
import models
import utils

for module in (data_loader, processor, learning, metrics, features, application,
               entrypoint, models, utils):
    if Path(module.__file__).resolve() != CODES / (module.__name__ + ".py"):
        raise RuntimeError(f"Wrong source module: {module.__file__}")
torch.set_num_threads(1)

FEATURES = ["Open", "High", "Low", "Close", "Volume", "ExpectedDividend"]
SPLIT = ["2017-01-01", "2017-01-03", "2017-01-03", "2017-01-05"]


def raw_frame(n=4):
    values = np.arange(1, n + 1, dtype=float) * 10
    return pd.DataFrame({
        "SecuritiesCode": [1001] * n,
        "Date": pd.date_range("2017-01-01", periods=n).strftime("%Y-%m-%d"),
        "Open": values, "High": values + 2, "Low": values - 2,
        "Close": values + 1, "Volume": values * 100,
        "AdjustmentFactor": np.ones(n), "ExpectedDividend": np.zeros(n),
        "Target": np.arange(n, dtype=float) / 100,
    })


def normalize_for_audit(frame, columns):
    # Supply the newly required std=None only for isolated downstream checks.
    # The separate integration test below uses the actual unmodified call and
    # catches its missing argument. This adapter never changes production files.
    function = features.normalize_data
    parameter = inspect.signature(function).parameters.get("std")
    if parameter is not None and parameter.default is inspect.Parameter.empty:
        return function(frame, columns, std=None)
    return function(frame, columns)


def preprocess(frame, *, isolate_normalizer_signature=True):
    with patch.object(application, "main_loader", side_effect=lambda *a, **k: frame.copy(deep=True)), \
         patch.object(torch.backends.mps, "is_available", return_value=False), \
         contextlib.redirect_stdout(io.StringIO()):
        if not isolate_normalizer_signature:
            return application.common_process(SPLIT, application.COL_NAMES)
        original = features.normalize_data
        parameter = inspect.signature(original).parameters.get("std")
        if parameter is not None and parameter.default is inspect.Parameter.empty:
            with patch.object(features, "normalize_data", side_effect=lambda frame, cols: original(frame, cols, std=None)):
                return application.common_process(SPLIT, application.COL_NAMES)
        return application.common_process(SPLIT, application.COL_NAMES)


def tiny_model():
    return nn.Sequential(nn.Flatten(), nn.Linear(6, 1))


@contextlib.contextmanager
def light_application(*, real_training=False, predictions=None):
    """Keep real sequence/loader code; isolate orchestration with tiny models.

    Production train is used by integration smoke tests. For downstream tests,
    intentionally bypass it and intercept save_model to avoid disk side effects.
    """
    with contextlib.ExitStack() as stack:
        capture = io.StringIO()
        stack.enter_context(contextlib.redirect_stdout(capture))
        frame = raw_frame(4)
        frame["Date"] = pd.to_datetime(frame["Date"])
        stack.enter_context(patch.object(application, "TIME_SPAN", [1]))
        stack.enter_context(patch.object(application, "common_process", return_value=(frame, frame, "cpu")))
        factory = stack.enter_context(patch.object(application.models, "CNN_LSTM", side_effect=lambda **kw: tiny_model()))
        save = stack.enter_context(patch.object(learning, "save_model", autospec=True))
        stack.enter_context(patch.object(metrics, "overall_performance", return_value={"sharpe": 1.0, "rank_IC": 0.2}))
        if not real_training:
            stack.enter_context(patch.object(learning, "train", return_value=0.0))
            pred = np.zeros(4) if predictions is None else np.asarray(predictions)
            stack.enter_context(patch.object(learning, "evaluation", return_value=(0.0, pred)))
        yield save, capture, factory


class TrainingContractTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(34)
        self.model = nn.Linear(1, 1)
        self.loader = learning.make_loader([[1], [2], [3], [4]], [0, 0, 0, 0], batch_size=2)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

    def test_epoch_twenty_runs_twenty_complete_passes(self):
        with patch.object(self.optimizer, "step", wraps=self.optimizer.step) as step:
            learning.train(self.model, self.loader, nn.L1Loss(), self.optimizer, epoch=20)
        self.assertEqual(step.call_count, 20 * len(self.loader), "20 epochs must make 40 updates, not 2")

    def test_zero_epoch_cannot_update_parameters(self):
        before = [p.detach().clone() for p in self.model.parameters()]
        try:
            learning.train(self.model, self.loader, nn.L1Loss(), self.optimizer, epoch=0)
        except ValueError:
            pass  # Either reject zero or make it a no-op; never train silently.
        for old, new in zip(before, self.model.parameters()):
            torch.testing.assert_close(old, new, rtol=0, atol=0)

    def test_real_model1_call_matches_train_signature(self):
        with light_application(real_training=True):
            application.model_1()

    def test_real_model2_call_matches_train_signature(self):
        with light_application(real_training=True):
            application.model_2()

    def test_normal_one_epoch_actually_changes_weights(self):
        before = self.model.weight.detach().clone()
        learning.train(self.model, self.loader, nn.L1Loss(), self.optimizer)
        self.assertFalse(torch.equal(before, self.model.weight))


class PreprocessingAuditTests(unittest.TestCase):
    def test_common_process_call_matches_normalize_signature(self):
        preprocess(raw_frame(), isolate_normalizer_signature=False)

    def test_validation_values_cannot_change_training_features(self):
        original = raw_frame()
        future_changed = original.copy()
        future_changed.loc[2:, ["Open", "High", "Low", "Close", "Volume"]] *= 100
        first, _, _ = preprocess(original)
        second, _, _ = preprocess(future_changed)
        np.testing.assert_allclose(first[FEATURES], second[FEATURES], atol=1e-7,
                                   err_msg="Only future values changed, but training features changed")

    def test_training_features_use_training_statistics(self):
        train, _, _ = preprocess(raw_frame())
        np.testing.assert_allclose(train["Close"], [-1, 1], atol=1e-7)

    def test_validation_uses_same_training_transform(self):
        _, val, _ = preprocess(raw_frame())
        # Train Close=[11,21], mean=16, population std=5.
        np.testing.assert_allclose(val["Close"], [3, 5], atol=1e-7)

    def test_returned_scaling_metadata_reproduces_constant_column(self):
        frame = raw_frame()
        normalized, mean, scale = normalize_for_audit(frame, FEATURES)
        with np.errstate(divide="ignore", invalid="ignore"):
            replay = (frame[FEATURES].to_numpy() - mean) / scale
        self.assertTrue(np.isfinite(replay).all(), "Returned std is zero although StandardScaler uses safe scale=1")
        np.testing.assert_allclose(replay, normalized[FEATURES], atol=1e-7)

    def test_nonconstant_scaling_metadata_reproduces_transform(self):
        frame = raw_frame()
        cols = ["Open", "Close", "Volume"]
        normalized, mean, scale = normalize_for_audit(frame, cols)
        np.testing.assert_allclose((frame[cols] - mean) / scale, normalized[cols], atol=1e-7)

    def test_dates_use_inclusive_start_exclusive_end(self):
        frame = raw_frame()
        frame["Date"] = pd.to_datetime(frame["Date"])
        result = processor.get_time_for_specific_period(frame, "2017-01-02", "2017-01-04")
        self.assertEqual(result["Date"].dt.day.tolist(), [2, 3])

    def test_required_filter_handles_repeated_index_labels_positionally(self):
        frame = raw_frame()
        frame.index = [10, 10, 20, 20]
        frame.iloc[1, frame.columns.get_loc("Open")] = np.nan
        result = features.get_valid_rows(frame, FEATURES)
        self.assertEqual(result["Close"].tolist(), [11, 31, 41])

    def test_duplicate_stock_dates_do_not_survive_valid_rows(self):
        frame = raw_frame()
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        try:
            result = features.get_valid_rows(frame, application.COL_NAMES)
        except ValueError:
            return
        self.assertFalse(result.duplicated(["SecuritiesCode", "Date"]).any(),
                         "Duplicate stock-days must be rejected or explicitly resolved before windows")

    def test_final_row_nonpositive_factor_is_rejected_or_repaired(self):
        for bad in [0.0, -0.5]:
            with self.subTest(factor=bad):
                frame = raw_frame()
                frame.loc[len(frame) - 1, "AdjustmentFactor"] = bad
                try:
                    adjusted = processor.correct_prices_with_factors(frame)
                except ValueError:
                    continue
                self.assertTrue((adjusted["AdjustmentFactor"] > 0).all(),
                                "Checking only shifted cumulative factors misses the final row")

    def test_missing_column_message_names_only_missing_columns(self):
        with self.assertRaises(ValueError) as err:
            features.get_features(raw_frame(), ["Close", "DoesNotExist"])
        self.assertIn("DoesNotExist", str(err.exception))
        self.assertNotIn("Close", str(err.exception), "An existing column must not be reported missing")

    def test_sequence_window_zero_is_rejected(self):
        with self.assertRaises(ValueError):
            features.create_time_sequence(raw_frame(), 0, ["Close"])

    def test_sequence_negative_window_is_rejected(self):
        with self.assertRaises(ValueError):
            features.create_time_sequence(raw_frame(), -1, ["Close"])

    def test_validation_does_not_mutate_source(self):
        frame = raw_frame()
        original = frame.copy(deep=True)
        preprocess(frame)
        pd.testing.assert_frame_equal(frame, original)


class CheckpointAuditTests(unittest.TestCase):
    def test_real_serializer_roundtrip_preserves_parameters_and_metadata(self):
        torch.manual_seed(53)
        original = models.CNN_LSTM(6, 4, 1, 4).eval()
        restored = models.CNN_LSTM(6, 4, 1, 4)
        payload = io.BytesIO()
        actual_save, actual_load = torch.save, torch.load
        def save_memory(obj, path):
            actual_save(obj, payload)
        def load_memory(path, **kwargs):
            payload.seek(0)
            return actual_load(payload, **kwargs)
        means = np.arange(6, dtype=float)
        scales = np.arange(1, 7, dtype=float)
        with patch.object(Path, "mkdir"), patch.object(Path, "is_file", return_value=True), \
             patch.object(torch, "save", side_effect=save_memory), \
             patch.object(torch, "load", side_effect=load_memory), \
             contextlib.redirect_stdout(io.StringIO()):
            path = learning.save_model(original, "synthetic.pt", means, scales, "2017-01-01", "2021-01-01", FEATURES)
            model, mean, scale, cols, start, end = learning.load_model(restored, path)
        self.assertGreater(payload.getbuffer().nbytes, 0)
        self.assertFalse(model.training)
        self.assertEqual(cols, FEATURES)
        self.assertEqual((start, end), ("2017-01-01", "2021-01-01"))
        np.testing.assert_array_equal(mean, means)
        np.testing.assert_array_equal(scale, scales)
        probe = torch.randn(3, 4, 6)
        with torch.no_grad():
            torch.testing.assert_close(original(probe), model(probe), rtol=0, atol=0)

    def test_model2_supplies_all_required_save_arguments(self):
        with light_application() as (save, _, _):
            application.model_2()
        save.assert_called_once()
        self.assertEqual(save.call_args.kwargs["feature_columns"], FEATURES)

    def test_model2_checkpoint_retains_one_scale_per_feature(self):
        with light_application() as (save, _, _):
            application.model_2()
        args = save.call_args.kwargs
        self.assertEqual(np.asarray(args["mean"]).shape, (6,), "Hard-coded scalar 0 loses six fitted means")
        self.assertEqual(np.asarray(args["std"]).shape, (6,))

    def test_load_model_path_is_required_or_has_a_real_path_default(self):
        default = inspect.signature(learning.load_model).parameters["model_path"].default
        self.assertTrue(default is inspect.Parameter.empty or isinstance(default, (str, os.PathLike)),
                        "model_path=str stores the type object, not a default pathname")

    def test_load_model_rejects_nonexistent_path(self):
        with patch.object(Path, "is_file", return_value=False):
            with self.assertRaises(FileNotFoundError):
                learning.load_model(tiny_model(), "definitely_missing_checkpoint.pt")

    def test_use_loaded_model_passes_a_nonempty_checkpoint_path(self):
        with patch.object(entrypoint, "CNN_LSTM", return_value=tiny_model()), \
             patch.object(entrypoint, "load_model", return_value=(object(),) * 6) as loader:
            entrypoint.use_loaded_model()
        self.assertTrue(str(loader.call_args.kwargs["model_path"]).strip(), "The helper always loads an empty pathname")

    def test_use_loaded_model_returns_the_loaded_result_for_use(self):
        sentinel = (object(),) * 6
        with patch.object(entrypoint, "CNN_LSTM", return_value=tiny_model()), \
             patch.object(entrypoint, "load_model", return_value=sentinel):
            result = entrypoint.use_loaded_model()
        self.assertIsNotNone(result, "Loaded model and normalization metadata are discarded")


class ReportingAuditTests(unittest.TestCase):
    def test_reported_earnings_are_invariant_to_prediction_level_shift(self):
        # Same ordering, same targets, same portfolio; adding a constant to all
        # scores cannot change realized earnings. Downstream metrics are stubbed.
        outputs = []
        for shift in [0, 10]:
            with light_application(predictions=np.arange(4) + shift) as (_, stdout, _):
                application.model_2()
            lines = [line for line in stdout.getvalue().splitlines() if "Total earnings:" in line]
            self.assertEqual(len(lines), 1)
            outputs.append(lines[0])
        self.assertEqual(outputs[0], outputs[1], "Sum of model scores is not realized portfolio earnings")

    def test_nonfinite_predictions_cannot_silently_produce_a_valid_score(self):
        rows = []
        for day, magnitude in [(1, 1), (2, 2), (3, 3)]:
            for code in range(401):
                rows.append({"SecuritiesCode": code + 1001, "Date": f"2017-01-0{day}",
                             "Target": (code - 200) * 0.0001 * magnitude})
        frame = pd.DataFrame(rows).sort_values(["SecuritiesCode", "Date"])
        pred = frame["SecuritiesCode"].to_numpy(dtype=float)
        pred[0] = np.nan
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                scores = metrics.overall_performance(frame, pred, 1)
        except (ValueError, FloatingPointError):
            return
        self.assertFalse(all(np.isfinite(value) for value in scores.values()),
                         "A NaN prediction silently received a position and produced finite scores")

    def test_missing_target_cannot_be_silently_treated_as_zero_earning(self):
        frame = pd.DataFrame({"Weight": [0.5, -0.5], "Target": [np.nan, 0.1]})
        try:
            value = metrics.one_day_earning(frame)
        except (ValueError, FloatingPointError):
            return
        self.assertFalse(np.isfinite(value), "pandas sum skipped a missing held-position Target")

    def test_small_but_nonzero_daily_variance_is_not_rounded_to_zero(self):
        returns = np.array([0.01, 0.01 + 1e-10, 0.01 + 2e-10], dtype=np.float64)
        frame = pd.DataFrame({"Date": pd.date_range("2017-01-01", periods=3), "Weight": 1.0, "Target": returns})
        expected = returns.mean() / returns.std(ddof=1)
        actual = metrics.calculate_sharpe(frame)
        self.assertAlmostEqual(actual / expected, 1.0, places=6)


class DesignObservationTests(unittest.TestCase):
    """Passing observations: these document choices, not required bug fixes."""
    def test_current_42_day_convolution_leaves_one_lstm_step(self):
        model = models.CNN_LSTM(6, 4, 42, 4).eval()
        seen = []
        handle = model.lstm.register_forward_pre_hook(lambda module, args: seen.append(tuple(args[0].shape)))
        try:
            with torch.no_grad():
                model(torch.ones(2, 42, 6))
        finally:
            handle.remove()
        self.assertEqual(seen, [(2, 1, 4)])

    def test_separate_validation_windows_omit_first_41_rows_per_stock(self):
        frame = raw_frame(60)
        X, y = features.create_time_sequence(frame, 42, FEATURES)
        self.assertEqual(len(X), 19)
        np.testing.assert_allclose(y, frame["Target"].to_numpy()[41:])

    def test_expanding_folds_reuse_one_model(self):
        with light_application() as (_, _, factory):
            application.model_2()
        self.assertEqual(factory.call_count, 1)

    def test_main_currently_selects_training_entry(self):
        with patch.object(entrypoint, "model_2") as train_entry:
            entrypoint.main()
        train_entry.assert_called_once_with()


class BoundaryControlTests(unittest.TestCase):
    def test_csv_missing_file_has_explicit_error(self):
        with patch.object(Path, "is_file", return_value=False):
            with self.assertRaises(FileNotFoundError):
                data_loader.load_data("missing.csv")

    def test_csv_empty_table_is_rejected(self):
        with patch.object(Path, "is_file", return_value=True), patch.object(pd, "read_csv", return_value=pd.DataFrame()):
            with self.assertRaises(ValueError):
                data_loader.load_data("empty.csv")

    def test_date_validator_rejects_impossible_date(self):
        self.assertFalse(processor.is_valid_date("2024-02-30"))
        self.assertTrue(processor.is_valid_date("2024-02-29"))

    def test_numeric_validator_rejects_nan_infinity_bool_and_array(self):
        for value in [np.nan, np.inf, -np.inf, True, [1, 2]]:
            with self.subTest(value=value):
                ok, error = processor.check_single_value(processor.NUMBER_TYPES, value, [])
                self.assertFalse(ok)
                self.assertIsInstance(error, (ValueError, TypeError))

    def test_sequence_empty_frame_is_rejected(self):
        with self.assertRaises(ValueError):
            features.create_time_sequence(raw_frame().iloc[:0], 1, FEATURES)

    def test_source_functions_are_importable_without_running_main(self):
        for module in (entrypoint, application, learning, models, features, processor, metrics, data_loader, utils):
            self.assertEqual(Path(module.__file__).resolve().parent, CODES)
        tree = ast.parse((CODES / "main.py").read_text())
        self.assertTrue(any(isinstance(node, ast.If) for node in tree.body))


if __name__ == "__main__":
    unittest.main(verbosity=2)
