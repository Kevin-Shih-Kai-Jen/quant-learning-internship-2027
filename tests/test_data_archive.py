"""Offline recovery smoke tests. No GitHub account or network is used."""

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "data_archive.py"
SPEC = importlib.util.spec_from_file_location("data_archive", SCRIPT)
archive = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(archive)


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="data-archive-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        (self.repo / "data/catalogs").mkdir(parents=True)
        self.destination = self.root / "downloaded"
        self.asset = self.root / "pack-0001.tar"
        self.contents = {"research/experiment_a/data.csv": b"x,y\n1,2\n" * 64,
                         "research/experiment_b/model.pkl": b"opaque-pickle-like-bytes\x00" * 32}
        self.entries = [{"path": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                         "mode": 0o640, "mtime_ns": 1789272000000000000}
                        for name, data in self.contents.items()]
        self.make_tar()
        self.pack = {"name": self.asset.name, "size": self.asset.stat().st_size,
                     "sha256": hashlib.sha256(self.asset.read_bytes()).hexdigest(), "files": self.entries}
        self.catalog = {"schema_version": 1, "repository": archive.REPOSITORY,
                        "tag": "snapshot-test", "packs": [self.pack]}
        (self.repo / "data/index.json").write_text(json.dumps({"schema_version": 1, "latest": "snapshot-test",
            "snapshots": [{"tag": "snapshot-test", "catalog": "data/catalogs/snapshot-test.json", "status": "verified"}]}))
        self.save_catalog()
        self.root_patch = patch.object(archive, "REPO_ROOT", self.repo)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.gh_patch = patch.object(archive.subprocess, "run", side_effect=self.mock_gh)
        self.gh = self.gh_patch.start()
        self.addCleanup(self.gh_patch.stop)
        self.http_patch = patch.object(archive, "urlopen", side_effect=lambda *a, **k: self.asset.open("rb"))
        self.http = self.http_patch.start()
        self.addCleanup(self.http_patch.stop)

    def make_tar(self, extra=None):
        with tarfile.open(self.asset, "w") as output:
            for name, data in self.contents.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                output.addfile(info, io.BytesIO(data))
            if extra is not None:
                output.addfile(extra, io.BytesIO(b"bad") if extra.size else None)

    def save_catalog(self):
        (self.repo / "data/catalogs/snapshot-test.json").write_text(json.dumps(self.catalog))

    def mock_gh(self, arguments, check):
        self.assertEqual(arguments[:3], ["gh", "release", "download"])
        self.assertTrue(check)
        self.assertEqual(arguments[arguments.index("--repo") + 1], archive.REPOSITORY)
        shutil.copyfile(self.asset, Path(arguments[arguments.index("--dir") + 1]) / self.asset.name)

    def run_cli(self, *arguments, expected=0):
        with contextlib.redirect_stdout(io.StringIO()) as output, contextlib.redirect_stderr(io.StringIO()) as errors:
            status = archive.main(list(arguments))
        self.assertEqual(status, expected, output.getvalue() + errors.getvalue())
        return output.getvalue() + errors.getvalue()

    def test_list_selective_fetch_verify_skip_and_metadata(self):
        self.assertIn("experiment_a", self.run_cli("list"))
        self.run_cli("fetch", "--experiment", "experiment_a", "--dest", str(self.destination))
        result = self.destination / "research/experiment_a/data.csv"
        self.assertEqual(result.read_bytes(), self.contents["research/experiment_a/data.csv"])
        self.assertEqual(result.stat().st_mode & 0o7777, 0o640)
        self.assertEqual(result.stat().st_mtime_ns, self.entries[0]["mtime_ns"])
        self.assertFalse((self.destination / "research/experiment_b/model.pkl").exists())
        self.run_cli("verify", "--experiment", "research/experiment_a", "--dest", str(self.destination))
        self.assertIn("SKIP", self.run_cli("fetch", "--experiment", "experiment_a", "--dest", str(self.destination)))
        self.assertEqual(self.http.call_count, 1)
        self.gh.assert_not_called()
        request = self.http.call_args.args[0]
        self.assertEqual(request.full_url,
                         f"https://github.com/{archive.REPOSITORY}/releases/download/snapshot-test/{self.asset.name}")
        self.assertNotIn("Authorization", request.headers)
        self.assertEqual(list(self.root.glob(".data-archive-*")), [])

    def test_existing_conflict_is_not_overwritten(self):
        target = self.destination / "research/experiment_a/data.csv"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"new local result")
        self.run_cli("fetch", "--experiment", "experiment_a", "--dest", str(self.destination), expected=1)
        self.assertEqual(target.read_bytes(), b"new local result")
        self.gh.assert_not_called()

    def test_corrupt_archive_fails_before_extract(self):
        self.asset.write_bytes(b"corrupt archive")
        self.run_cli("fetch", "--all", "--dest", str(self.destination), expected=1)
        self.assertFalse((self.destination / "research").exists())
        self.assertEqual(list(self.root.glob(".data-archive-*")), [])

    def test_unsafe_tar_member_fails_before_extract(self):
        extra = tarfile.TarInfo("../escaped.txt")
        extra.size = 3
        self.make_tar(extra)
        self.pack.update(size=self.asset.stat().st_size, sha256=hashlib.sha256(self.asset.read_bytes()).hexdigest())
        self.save_catalog()
        self.run_cli("fetch", "--all", "--dest", str(self.destination), expected=1)
        self.assertFalse((self.destination / "research").exists())
        self.assertFalse((self.root / "escaped.txt").exists())

    def test_tar_payload_hash_is_checked_independently(self):
        self.entries[0]["sha256"] = "0" * 64
        self.save_catalog()
        self.run_cli("fetch", "--all", "--dest", str(self.destination), expected=1)
        self.assertFalse((self.destination / "research").exists())

    def test_destination_symlink_is_rejected(self):
        self.destination.mkdir()
        elsewhere = self.root / "elsewhere"
        elsewhere.mkdir()
        (self.destination / "research").symlink_to(elsewhere, target_is_directory=True)
        self.run_cli("fetch", "--all", "--dest", str(self.destination), expected=1)
        self.assertEqual(list(elsewhere.iterdir()), [])

    def test_verify_detects_modified_local_file(self):
        self.run_cli("fetch", "--all", "--dest", str(self.destination))
        (self.destination / "research/experiment_a/data.csv").write_bytes(b"altered")
        self.run_cli("verify", "--experiment", "experiment_a", "--dest", str(self.destination), expected=1)

    def test_pack_may_reference_earlier_release(self):
        self.pack["release_tag"] = "snapshot-previous"
        self.save_catalog()
        self.run_cli("fetch", "--experiment", "experiment_a", "--dest", str(self.destination))
        self.assertIn("/releases/download/snapshot-previous/", self.http.call_args.args[0].full_url)

    def test_private_asset_falls_back_to_authenticated_gh(self):
        self.http.side_effect = archive.HTTPError("https://github.com/asset", 404, "Not Found", {}, None)
        self.pack["release_tag"] = "snapshot-previous"
        self.save_catalog()
        self.run_cli("fetch", "--experiment", "experiment_a", "--dest", str(self.destination))
        self.assertEqual(self.gh.call_args.args[0][3], "snapshot-previous")
        self.assertEqual((self.destination / "research/experiment_a/data.csv").read_bytes(),
                         self.contents["research/experiment_a/data.csv"])

    def test_oversized_public_download_is_rejected_before_extract(self):
        self.http.side_effect = lambda *a, **k: io.BytesIO(b"X" * (self.pack["size"] + 1))
        self.assertIn("exceeds catalog size",
                      self.run_cli("fetch", "--all", "--dest", str(self.destination), expected=1))
        self.assertFalse((self.destination / "research").exists())
        self.assertEqual(list(self.root.glob(".data-archive-*")), [])
        self.gh.assert_not_called()


if __name__ == "__main__":
    unittest.main()
