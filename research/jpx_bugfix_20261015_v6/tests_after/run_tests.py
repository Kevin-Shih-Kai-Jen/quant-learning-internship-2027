"""Run v3 JPX unittest checks against the current codes folder (CPU, synthetic data).

python3.12 -B debug/JPX_debug_v3/run_tests.py
python3.12 -B debug/JPX_debug_v3/run_tests.py test_audit.TrainingContractTests
"""
from __future__ import annotations
import argparse
import contextlib
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import sys
import time
import unittest
from zoneinfo import ZoneInfo

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def source_hashes(directory):
    return {p.name: {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in sorted(directory.glob("*.py"))}


class RecordedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.passed_ids = []
        self.failed_ids = set()
        self.errored_ids = set()

    def addSuccess(self, test):
        super().addSuccess(test)
        self.passed_ids.append(test.id())

    def addFailure(self, test, err):
        self.failed_ids.add(test.id())
        super().addFailure(test, err)

    def addError(self, test, err):
        self.errored_ids.add(test.id())
        super().addError(test, err)

    def addSubTest(self, test, subtest, err):
        if err is not None:
            target = self.failed_ids if issubclass(err[0], test.failureException) else self.errored_ids
            target.add(test.id())
        super().addSubTest(test, subtest, err)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codes", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("tests", nargs="*", help="Optional module.Class or module.Class.method selectors")
    args = parser.parse_args()
    codes = (args.codes or Path(os.environ.get("JPX_CODES_DIR", HERE.parent.parent / "codes"))).expanduser().resolve()
    if not (codes / "deep_learning.py").is_file():
        parser.error(f"Cannot find codes directory: {codes}; provide --codes explicitly")
    os.environ["JPX_CODES_DIR"] = str(codes)
    before = source_hashes(codes)
    print(f"Python: {sys.executable}\nTesting source: {codes}", flush=True)
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromNames(args.tests) if args.tests else loader.discover(str(HERE), pattern="test_*.py")
    if suite.countTestCases() == 0:
        raise RuntimeError("No tests found; zero tests is not a successful check")
    class Tee(io.StringIO):
        def write(self, value):
            sys.stderr.write(value)
            return super().write(value)
        def flush(self):
            sys.stderr.flush()
            super().flush()
    output, stdout = Tee(), io.StringIO()
    start = time.monotonic()
    with contextlib.redirect_stdout(stdout):
        result = unittest.TextTestRunner(stream=output, verbosity=2, resultclass=RecordedResult).run(suite)
    after = source_hashes(codes)
    changed = before != after
    failed = result.failed_ids - result.errored_ids
    summary = {
        "run_at": datetime.now(ZoneInfo("Asia/Taipei")).isoformat(),
        "duration_seconds": time.monotonic() - start,
        "python": sys.executable, "python_version": platform.python_version(),
        "versions": {n: getattr(sys.modules.get(n), "__version__", None) for n in ["numpy", "pandas", "torch", "scipy", "sklearn"]},
        "codes_directory": str(codes), "source_manifest": before,
        "source_changed_during_run": changed,
        "imported_module_paths": {n: str(Path(sys.modules[n].__file__).resolve()) for n in ["main", "JPX_deep_learning_models", "models", "deep_learning", "features", "evaluation", "data_loader", "data_processor", "utils"] if n in sys.modules},
        "total": result.testsRun, "passed": len(result.passed_ids),
        "failed_test_methods": len(failed), "errored_test_methods": len(result.errored_ids),
        "unittest_failure_entries_including_subtests": len(result.failures),
        "unittest_error_entries_including_subtests": len(result.errors),
        "skipped": len(result.skipped),
        "passed_tests": result.passed_ids, "failure_tests": sorted(failed),
        "error_tests": sorted(result.errored_ids),
        "details": [{"test": t.id(), "kind": kind, "traceback": trace} for kind, entries in [("failure", result.failures), ("error", result.errors)] for t, trace in entries],
        "scope": "Synthetic CPU checks; no market CSV, training job, MPS/CUDA run or investment-performance validation.",
    }
    if args.report_dir:
        args.report_dir.mkdir(parents=True, exist_ok=True)
        (args.report_dir / "test_results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        (args.report_dir / "test_results.txt").write_text(f"Python: {sys.executable}\nTesting source: {codes}\n" + output.getvalue())
        (args.report_dir / "captured_stdout.txt").write_text(stdout.getvalue())
    print(f"Test methods: {result.testsRun}; passed: {len(result.passed_ids)}; failures: {len(failed)}; errors: {len(result.errored_ids)}; source unchanged: {not changed}")
    return 0 if result.wasSuccessful() and not changed else 1


if __name__ == "__main__":
    raise SystemExit(main())
