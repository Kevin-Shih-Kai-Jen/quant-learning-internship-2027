# JPX training-only scaling fix — 2026-10-15

The user requested the previously proposed normalization changes, with English code comments and the existing naming/style preserved. Two production files were changed: features.py and JPX_deep_learning_models.py.

## Behavior

- common_process keeps the existing loading, filtering and price-adjustment steps, then splits train/validation by the existing dates.
- normalize_data fits only the training rows; validation receives the same fitted mean and scale.
- Both optional parameters may be absent (fit), or both present (transform). Supplying only one raises ValueError.
- The existing std variable now holds StandardScaler.scale_, which uses a safe denominator for constant features.
- common_process returns train_df, val_df, device, mean, std. Both model callers unpack the new return value; model_2 saves the actual final-fold statistics.
- Existing feature names, folds, model architecture, optimizer, training loss and scoring remain as before. No Chinese comments were added.
- Two existing unittest fixture files were updated for the five-value result. The checkpoint test now checks exact forwarded parameter values, not only vector shape.

## Verification

The complete existing suite ran against the candidate: 104 tests, 88 passed, 13 failed assertions and 3 errors. All 12 checks relevant to this change passed; they were then rerun against the actual installed Desktop code from another working directory and again passed 12/12. Two additional partial-parameter cases were rejected as intended. English-only comments in the two edited files and their Python syntax were checked.

Remaining unrelated failures are preserved in test_results. In particular, model_1/model_2 still call train with epochs, while train accepts epoch; epoch also still does not control repeated passes. This change does not claim a successful full training run. No real market CSV or held-out test was opened.

## Reproduction and preservation

- before/: exact pre-change copies of the two edited source files.
- after/: complete nine-file source snapshot after the change.
- tests_before/: original versions of the two updated unittest files.
- tests_after/: complete current v3 test Python files.
- changes.diff: exact code and fixture changes.
- source_before.json, source_after.json, installation.json: content hashes and installed-file verification.
- test_results/: full-suite outcome, including remaining failures.
- installed_results/: the 12 focused tests against /Users/coolguy/Desktop/JPX/codes.
- verification.json: focused checks, parameter-pair checks and scope.

Run the archived complete suite with:

    python3.12 -B tests_after/run_tests.py --codes after

The full suite is expected to remain red until unrelated issues are addressed. The prior audit is preserved at d9d157a30e03ddd06b9dfc17fdac4a18fc25c587; this work uses the new branch codex/jpx-train-only-scaling-20261015 and research/jpx_train_only_scaling_20261015_v1 directory. No existing cloud history was overwritten, no Release downloads were needed and no local data were deleted. Lightweight source, tests and result records are retained.
