from pathlib import Path

ROOT = Path(__file__).resolve().parent

def replace(path, old, new):
    content = path.read_text()
    count = content.count(old)
    if count != 1:
        raise ValueError(f'{path.name}: expected one match, got {count}: {old!r}')
    path.write_text(content.replace(old, new, 1))

code = ROOT / 'after'
path = code / 'deep_learning.py'
replace(path, 'from time import time\n', '')
replace(path, 'from data_loader import main_loader\nimport data_processor\nimport features\n', '')
replace(path, ') -> torch.tensor:', ') -> torch.Tensor:')
content = path.read_text()
start = content.index('\ndef common_process(')
end = content.index('\ndef save_model(', start)
path.write_text(content[:start] + '\n' + content[end:])
replace(path, 'def train(model, data_loader, loss_fn, optimizer, epochs=1, device="cpu"):\n', 'def train(model, data_loader, loss_fn, optimizer, epochs=1, device="cpu"):\n    if isinstance(epochs, bool) or not isinstance(epochs, (int, np.integer)) or epochs <= 0:\n        raise ValueError("epochs must be a positive integer")\n\n')
replace(path, 'model_path:str | Path,', 'model_path: str | Path,')

path = code / 'features.py'
replace(path, 'results, _ = check_JPX_stock_price_csv(df.copy())', 'results, duplicates = check_JPX_stock_price_csv(df.copy())\n    if not duplicates.empty:\n        raise ValueError("Duplicate SecuritiesCode and Date rows")')
replace(path, 'features_col_name for n in features_col_name if n not in df', 'n for n in features_col_name if n not in df')
replace(path, 'df: pd.DataFrame | pd.DataFrame,', 'df: pd.DataFrame,')
replace(path, ') -> tuple[list[list[float | int]], list[list[float | int]]]:\n', ') -> tuple[list[list[float | int]], list[list[float | int]]]:\n    if isinstance(time_sequence, bool) or not isinstance(time_sequence, (int, np.integer)) or time_sequence <= 0:\n        raise ValueError("time_sequence must be a positive integer")\n\n')

path = code / 'data_processor.py'
replace(path, '    #===========================================================================================================#\n', '    if (table["AdjustmentFactor"] <= 0).any():\n        raise ValueError("AdjustmentFactor must be greater than 0")\n\n    #===========================================================================================================#\n')
replace(path, 'if (prior_factor <= 0).any():\n        raise ValueError("AdjustmentFactor can not be smaller or equal to 0")', 'if not np.isfinite(prior_factor).all() or (prior_factor <= 0).any():\n        raise ValueError("Cumulative AdjustmentFactor must be finite and greater than 0")')

path = code / 'evaluation.py'
replace(path, 'from collections.abc import Callable\n', '')
replace(path, ') -> pd.DataFrame:    \n    df = df.copy().sort_values(["SecuritiesCode", "Date"])', ') -> pd.DataFrame:\n    if isinstance(time_window, bool) or not isinstance(time_window, (int, np.integer)) or time_window <= 0:\n        raise ValueError("time_window must be a positive integer")\n\n    predictions = np.asarray(predictions, dtype=np.float64)\n    if predictions.ndim != 1 or not np.isfinite(predictions).all():\n        raise ValueError("predictions must be a finite one-dimensional array")\n\n    df = df.copy().sort_values(["SecuritiesCode", "Date"])')
replace(path, '    ]\n    \n    eval_df["Prediction"] = predictions', '    ].copy()\n\n    if len(predictions) != len(eval_df):\n        raise ValueError("Prediction count must match the number of sequence targets")\n\n    eval_df["Prediction"] = predictions')
replace(path, '    if "Prediction_rank" not in df:\n', '    if not np.isfinite(df["Prediction"]).all():\n        raise ValueError("Prediction must contain only finite values")\n\n    if "Prediction_rank" not in df:\n')
start = path.read_text().index('def daily_earning(')
end = path.read_text().index('\ndef calculate_sharpe(', start)
content = path.read_text()
content = content[:start] + '''def daily_earning(
    df: pd.DataFrame,
) -> np.ndarray:
    """Return the sum of signed Weight * Target for each date."""
    if df["Date"].isna().any():
        raise ValueError("Date must not be missing")
    if not np.isfinite(df[["Weight", "Target"]].to_numpy(dtype=np.float64)).all():
        raise ValueError("Weight and Target must contain only finite values")

    return np.array([
        np.sum(
            daily_df["Weight"].to_numpy(dtype=np.float64)
            * daily_df["Target"].to_numpy(dtype=np.float64)
        )
        for _, daily_df in df.groupby("Date")
    ], dtype=np.float64)

''' + content[end:]
path.write_text(content)
replace(path, 'raise ValueError("Required more than 2 days\' earnings")', 'raise ValueError("Required at least 2 days\' earnings")')
replace(path, ') -> dict[str: float]:', ') -> dict[str, float]:')
replace(path, '    earnings = daily_earning(eval_df).sum()', '    earnings = float(daily_earning(eval_df).sum())')
replace(path, '"Earnings": earnings', '"earnings": earnings')

path = code / 'JPX_deep_learning_models.py'
replace(path, 'print(f"Total earnings: {total_earnings}%")', 'print(f"Total earnings (sum of daily returns): {total_earnings:.2%}")')

path = code / 'main.py'
replace(path, 'from models import CNN_LSTM', 'from pathlib import Path\n\nfrom models import CNN_LSTM')
replace(path, 'def use_loaded_model():', 'def use_loaded_model(model_path: str | Path):')
replace(path, '    load_model(\n', '    return load_model(\n')
replace(path, '        model_path="",', '        model_path=model_path,')

# Update tests for the current public API, preserving behavioral assertions.
tests = ROOT / 'tests_after'
path = tests / 'test_audit.py'
replace(path, 'epoch=20)', 'epochs=20)')
replace(path, 'epoch=0)', 'epochs=0)')
replace(path, '{"sharpe": 1.0, "rank_IC": 0.2}', '{"sharpe": 1.0, "rank_IC": 0.2, "earnings": 0.03}')
replace(path, '            entrypoint.use_loaded_model()\n        self.assertTrue(str(loader.call_args.kwargs["model_path"]).strip(), "The helper always loads an empty pathname")', '            entrypoint.use_loaded_model("checkpoints/CNN_LSTM")\n        self.assertEqual(loader.call_args.kwargs["model_path"], "checkpoints/CNN_LSTM")')
replace(path, '            result = entrypoint.use_loaded_model()\n        self.assertIsNotNone(result, "Loaded model and normalization metadata are discarded")', '            result = entrypoint.use_loaded_model(Path("checkpoints/CNN_LSTM"))\n        self.assertIs(result, sentinel, "Loaded model and normalization metadata must be returned")')
replace(path, 'if "Total earnings:" in line', 'if "Total earnings (sum of daily returns):" in line')
replace(path, 'frame = pd.DataFrame({"Weight": [0.5, -0.5], "Target": [np.nan, 0.1]})', 'frame = pd.DataFrame({"Date": ["2024-01-01"] * 2, "Weight": [0.5, -0.5], "Target": [np.nan, 0.1]})')
replace(path, 'value = metrics.one_day_earning(frame)', 'value = metrics.daily_earning(frame)[0]')

path = tests / 'test_regression_v2.py'
replace(path, 'metrics.one_day_earning(df), 0.03,', 'metrics.daily_earning(df.assign(Date="2024-01-01"))[0], 0.03,')
content = path.read_text()
content = content.replace('test_02_formula_isolated_from_one_day_earning', 'test_02_formula_isolated_from_daily_earning')
content = content.replace('patch.object(metrics, "one_day_earning", side_effect=[0.01, 0.02, 0.03])', 'patch.object(metrics, "daily_earning", return_value=np.array([0.01, 0.02, 0.03]))')
# The real aggregation handles the single-day and constant-return boundaries.
content = content.replace('with patch.object(metrics, "one_day_earning", return_value=0.01):', 'with contextlib.nullcontext():')
path.write_text(content)
replace(path, 'self.assertEqual(set(result), {"sharpe", "rank_IC"})', 'self.assertEqual(set(result), {"sharpe", "rank_IC", "earnings"})\n        self.assertEqual(result["earnings"], 0.0)')
replace(path, 'self.assertAlmostEqual(scores["sharpe"], 2.0, places=5)', 'self.assertAlmostEqual(scores["sharpe"], 2.0, places=5)\n        self.assertAlmostEqual(scores["earnings"], 0.06, places=8)')
print('Updated six source modules and two existing test modules.')
