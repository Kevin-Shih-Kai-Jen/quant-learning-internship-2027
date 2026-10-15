# Reapply JPX normalization fix — 2026-10-15, v2

The user requested reapplying the normalization changes after editor save conflicts. The current on-disk files had new user edits. All nine source files were copied to before/ before installation; after/ is the complete verified resulting source snapshot.

Only features.normalize_data changed. Both mean and std may be absent to fit the training data, or both present to reuse the fitted transform. Supplying exactly one raises ValueError. std uses StandardScaler.scale_ so constant features can be transformed safely. The parameter annotations match the returned NumPy arrays. The user's other functions and source files were preserved byte-for-byte. No Chinese comments were added.

The date split, training-only fit, validation reuse, five-value common_process return, and actual checkpoint metadata forwarding were already present in JPX_deep_learning_models.py. They did not need rewriting. The existing unittest fixtures already match that interface and were kept unchanged.

## Verification

The same 12 focused unittest checks ran before and after the change: before, 5 passed and 7 errored; after, all 12 passed. All 12 then passed against the actual Desktop files using the installed Desktop test runner. Two additional partial-metadata cases raised ValueError as intended. All nine installed source hashes were checked, and the unchanged functions in features.py were compared by AST. See focused_test_names.json, before_results/, after_results/, installed_results/, and verification.json.

These checks use synthetic data. No real market CSV, full training job, GPU run, or held-out test was used. This is not a claim that the full application is bug-free. In particular, the user's independently edited train function still has `for _ in epochs`, which cannot iterate an integer. Their added duplicate common_process and load_model default also remain unchanged; this request was limited to reapplying the normalization fix.

The initial Visual Studio Code accessibility state showed Close actions and no dirty markers on the relevant tabs. A subsequent screenshot was blocked because the Mac was locked. Unsaved editor buffers were not extracted and automatic editor refresh was not verified. Only the on-disk copies are claimed as backed up and repaired.

## Reproduction and preservation

Run tests/run_tests.py with --codes after and the names in focused_test_names.json. The complete test source is included for repeatability, but only the listed 12 tests were rerun in this revision.

The previous repair is preserved at commit d11f0a31470738d08b1422c9b85bd2a000d7257c. This revision uses a new branch, codex/jpx-scaling-reapply-20261015-v2, and a new experiment directory. No existing remote history was replaced, no Release downloads were required, and no local files or data were deleted.
