"""Offline publisher smoke tests: temporary fixtures, no GitHub or source data writes."""

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "publish_snapshot", REPOSITORY_ROOT / "scripts/publish_snapshot.py"
)
publisher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publisher)


class PublisherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="publisher-test-")
        self.addCleanup(temporary.cleanup)
        self.temporary = Path(temporary.name).resolve()
        self.workspace = self.temporary / "workspace"
        # Deliberately nested, with a name absent from excluded_directories.
        self.repo = self.workspace / "arbitrary-staging-name"
        self.repo.mkdir(parents=True)
        self.config = json.loads((REPOSITORY_ROOT / "archive-config.json").read_text())
        self.config.update(pack_max_bytes=2 * 1024 * 1024,
                           text_max_bytes=128, csv_git_max_bytes=64)
        root_patch = patch.object(publisher, "ROOT", self.repo)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        network_patch = patch.object(
            publisher, "gh", side_effect=AssertionError("Unexpected real GitHub operation")
        )
        self.no_network = network_patch.start()
        self.addCleanup(network_patch.stop)

    def write_source(self, name, data):
        path = self.workspace / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def plan_path(self, tag):
        return self.repo / "data/upload-plans" / (tag + ".json")

    def prepare(self, tag="snapshot-first"):
        with contextlib.redirect_stdout(io.StringIO()):
            return publisher.prepare(self.workspace, tag, self.config, self.plan_path(tag))

    def build(self, pack, name="payload.tar"):
        target = self.temporary / name
        publisher.build_pack(self.workspace, pack, target)
        return target

    def save_previous_catalog(self, tag, packs):
        publisher.write_json(self.repo / "data/catalogs" / (tag + ".json"), {
            "schema_version": 1, "repository": self.config["repository"],
            "tag": tag, "packs": packs,
        })
        publisher.write_json(self.repo / "data/index.json", {
            "schema_version": 1, "latest": tag,
            "snapshots": [{"tag": tag, "catalog": "data/catalogs/" + tag + ".json",
                           "status": "verified"}],
        })

    @staticmethod
    def remote_asset(pack, identifier=10):
        return {"id": identifier, "name": pack["name"], "state": "uploaded",
                "size": pack["size"], "digest": "sha256:" + pack["sha256"]}

    def test_prepare_covers_sources_and_pathspec_without_recursing_into_staging(self):
        contents = {
            "experiment/計畫.md": "保留實驗計畫\n".encode(),
            "experiment/predictions.npz": b"opaque-results\x00\xff",
            "site/.gitignore": b"/dist/\nnext-env.d.ts\n",
            "site/dist/generated.js": b"export const result = 1;\n",
            "site/next-env.d.ts": b"// generated declarations\n",
        }
        originals = {}
        for name, data in contents.items():
            path = self.write_source(name, data)
            originals[name] = path.stat().st_mtime_ns
        self.write_source("site/node_modules/dependency.js", b"exclude dependency")
        self.write_source("site/.git/config", b"exclude nested Git metadata")
        self.write_source("experiment/__pycache__/run.pyc", b"exclude bytecode")
        self.write_source("experiment/.DS_Store", b"exclude finder state")
        (self.repo / "do-not-archive.md").write_text("staging must not archive itself")

        plan = self.prepare()
        inventory = json.loads((self.repo / "data/inventories/snapshot-first.json").read_text())
        records = {entry["path"]: entry for entry in inventory["files"]}
        self.assertEqual(set(records), {"research/" + name for name in contents})
        pathspec = (self.plan_path(plan["tag"]).parent /
                    (plan["tag"] + "-git-paths.nul")).read_bytes().split(b"\0")
        self.assertEqual(pathspec[-1], b"")
        git_paths = {entry["path"] for entry in plan["git_files"]}
        self.assertEqual({path.decode() for path in pathspec[:-1]}, git_paths)
        # Ignored generated files must still reach the explicit force-add pathspec.
        self.assertIn("research/site/dist/generated.js", git_paths)
        self.assertIn("research/site/next-env.d.ts", git_paths)
        for name, data in contents.items():
            record = records["research/" + name]
            self.assertEqual(record["sha256"], hashlib.sha256(data).hexdigest())
            self.assertEqual(record["size"], len(data))
            self.assertEqual((self.workspace / name).read_bytes(), data)
            self.assertEqual((self.workspace / name).stat().st_mtime_ns, originals[name])
            if record["storage"] == "git":
                self.assertEqual((self.repo / record["path"]).read_bytes(), data)
        self.assertEqual(plan["excluded"]["directory:arbitrary-staging-name"], 1)
        self.no_network.assert_not_called()

    def test_pack_split_is_stable_and_every_payload_is_preserved_once(self):
        names = [f"experiment/data-{index}.npz" for index in range(6)]
        for index, name in enumerate(reversed(names)):
            self.write_source(name, bytes([index]) * (400 * 1024))
        first = self.prepare("snapshot-one")
        second = self.prepare("snapshot-two")
        groups = lambda plan: [[f["path"] for f in pack["files"]]
                               for pack in plan["new_packs"]]
        self.assertEqual(groups(first), groups(second))
        self.assertEqual(len(first["new_packs"]), 2)
        flattened = [name for group in groups(first) for name in group]
        self.assertEqual(flattened, ["research/" + name for name in names])
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertFalse({p["name"] for p in first["new_packs"]} &
                         {p["name"] for p in second["new_packs"]})
        for index, pack in enumerate(first["new_packs"]):
            target = self.build(pack, f"split-{index}.tar")
            self.assertLessEqual(target.stat().st_size, self.config["pack_max_bytes"])

    def test_unicode_tar_round_trip_is_deterministic_and_does_not_mutate_source(self):
        data = b"\x00\xffopaque-pickle-like-data\n" * 31
        source = self.write_source("財報實驗/季度 資料.pkl", data)
        source.chmod(0o640)
        os.utime(source, ns=(1789272000123456789, 1789272000123456789))
        before = source.stat()
        plan = self.prepare()
        pack = plan["new_packs"][0]
        first = self.build(pack)
        first_hash = pack["sha256"]
        second = self.build(pack, "payload-again.tar")
        self.assertEqual(first_hash, pack["sha256"])
        self.assertEqual(first.read_bytes(), second.read_bytes())
        with tarfile.open(first, "r:") as archive:
            self.assertEqual(archive.getnames(), ["research/財報實驗/季度 資料.pkl"])
            member = archive.getmembers()[0]
            self.assertTrue(member.isfile())
            self.assertEqual(member.mode, 0o640)
            self.assertEqual(archive.extractfile(member).read(), data)
        self.assertEqual(source.read_bytes(), data)
        self.assertEqual(source.stat().st_mtime_ns, before.st_mtime_ns)
        self.assertEqual(source.stat().st_mode, before.st_mode)

    def test_pack_refuses_source_changed_since_plan(self):
        source = self.write_source("experiment/data.npz", b"planned")
        pack = self.prepare()["new_packs"][0]
        source.write_bytes(b"new-longer-content")
        with self.assertRaisesRegex(ValueError, "Source changed after planning"):
            self.build(pack)
        self.assertEqual(source.read_bytes(), b"new-longer-content")
        self.assertNotIn("sha256", pack)

    def test_member_hash_detects_tampering_even_when_size_and_mtime_match(self):
        source = self.write_source("experiment/data.npz", b"planned")
        pack = self.prepare()["new_packs"][0]
        entry = pack["files"][0]
        source.write_bytes(b"altered")
        os.utime(source, ns=(entry["mtime_ns"], entry["mtime_ns"]))
        with self.assertRaisesRegex(ValueError, "Packed member mismatch"):
            self.build(pack)
        self.assertNotIn("sha256", pack)

    def test_future_snapshot_reuses_prior_tag_and_resumes_verified_remote_assets(self):
        self.write_source("experiment/old.npz", b"old evidence")
        old_plan = self.prepare("snapshot-old")
        old_pack = old_plan["new_packs"][0]
        self.build(old_pack)
        legacy_pack = copy.deepcopy(old_pack)
        legacy_pack.pop("release_tag")  # Catalog tag supplies compatibility fallback.
        self.save_previous_catalog("snapshot-old", [legacy_pack])
        self.write_source("experiment/new.npz", b"new evidence")
        plan = self.prepare("snapshot-new")
        self.assertEqual(plan["inherited_packs"][0]["release_tag"], "snapshot-old")
        self.assertEqual([f["path"] for p in plan["new_packs"] for f in p["files"]],
                         ["research/experiment/new.npz"])
        new_pack = plan["new_packs"][0]
        self.build(new_pack, "new-payload.tar")
        self.assertNotEqual(old_pack["name"], new_pack["name"])
        assets = {1: {old_pack["name"]: self.remote_asset(old_pack, 11)},
                  2: {new_pack["name"]: self.remote_asset(new_pack, 12)}}

        def release(repo, tag):
            self.assertEqual(repo, self.config["repository"])
            return {"id": 1 if tag == "snapshot-old" else 2,
                    "draft": tag == "snapshot-new"}

        with patch.object(publisher, "api", return_value={"private": True}), \
                patch.object(publisher, "find_release", side_effect=release), \
                patch.object(publisher, "release_assets", side_effect=lambda repo, rid: assets[rid]), \
                patch.object(publisher, "build_pack", side_effect=AssertionError("Must resume existing asset")), \
                contextlib.redirect_stdout(io.StringIO()):
            publisher.upload(plan, self.plan_path("snapshot-new"))
        catalog = json.loads((self.repo / "data/catalogs/snapshot-new.json").read_text())
        self.assertEqual([p["release_tag"] for p in catalog["packs"]],
                         ["snapshot-old", "snapshot-new"])
        self.assertEqual(len({p["name"] for p in catalog["packs"]}), 2)
        verification = json.loads((self.repo / "data/verification/snapshot-new.json").read_text())
        self.assertTrue(verification["all_asset_digests_match"])
        self.assertEqual(len(verification["assets"]), 2)
        self.assertEqual(json.loads((self.repo / "data/index.json").read_text())["latest"],
                         "snapshot-new")
        self.no_network.assert_not_called()

    def test_changed_historical_release_payload_is_refused_without_overwrite(self):
        source = self.write_source("experiment/data.npz", b"original evidence")
        old = self.prepare("snapshot-old")
        self.build(old["new_packs"][0])
        self.save_previous_catalog("snapshot-old", old["new_packs"])
        source.write_bytes(b"revised evidence")
        with self.assertRaisesRegex(ValueError, "Historical data changed"):
            self.prepare("snapshot-new")
        self.assertFalse(self.plan_path("snapshot-new").exists())
        self.assertEqual(source.read_bytes(), b"revised evidence")
        self.assertEqual(json.loads((self.repo / "data/index.json").read_text())["latest"],
                         "snapshot-old")

    def test_git_to_release_conflict_leaves_old_staged_and_new_source_intact(self):
        source = self.write_source("experiment/notes.txt", b"old note")
        self.prepare("snapshot-old")
        source.write_bytes(b"new result\n" * 30)
        with self.assertRaisesRegex(ValueError, "Git-to-Release transition"):
            self.prepare("snapshot-new")
        self.assertEqual((self.repo / "research/experiment/notes.txt").read_bytes(), b"old note")
        self.assertEqual(source.read_bytes(), b"new result\n" * 30)

    def test_remote_metadata_mismatch_refuses_snapshot_publication(self):
        self.write_source("experiment/data.npz", b"evidence")
        plan = self.prepare()
        pack = plan["new_packs"][0]
        self.build(pack)
        valid = self.remote_asset(pack)
        publisher.require_asset(valid, pack)
        for changes in ({"size": pack["size"] + 1}, {"digest": "sha256:" + "0" * 64},
                        {"digest": None}, {"state": "starter"}):
            with self.subTest(changes=changes):
                bad = dict(valid, **changes)
                with patch.object(publisher, "api", return_value={"private": True}), \
                        patch.object(publisher, "find_release", return_value={"id": 1, "draft": True}), \
                        patch.object(publisher, "release_assets", return_value={pack["name"]: bad}), \
                        self.assertRaisesRegex(ValueError, "Remote size/digest mismatch"):
                    publisher.upload(plan, self.plan_path(plan["tag"]))
                self.assertFalse((self.repo / "data/index.json").exists())
                self.assertFalse((self.repo / "data/catalogs" / (plan["tag"] + ".json")).exists())
        self.no_network.assert_not_called()

    def test_draft_release_lookup_uses_paginated_owner_visible_list(self):
        draft = {"id": 21, "tag_name": "snapshot-draft", "draft": True}
        pages = [[{"id": 1, "tag_name": "snapshot-old", "draft": False}], [draft]]
        with patch.object(publisher, "gh", return_value=json.dumps(pages)) as command:
            self.assertEqual(publisher.find_release("owner/repo", "snapshot-draft"), draft)
            self.assertEqual(command.call_args.args,
                             ("api", "--paginate", "--slurp", "repos/owner/repo/releases?per_page=100"))
        for invalid in ([], [[draft, draft]]):
            with self.subTest(pages=invalid), \
                    patch.object(publisher, "gh", return_value=json.dumps(invalid)), \
                    self.assertRaisesRegex(ValueError, "Expected one existing"):
                publisher.find_release("owner/repo", "snapshot-draft")


if __name__ == "__main__":
    unittest.main()
