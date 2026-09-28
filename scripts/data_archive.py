#!/usr/bin/env python3
"""List, fetch, and verify GitHub release research-data snapshots."""

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import secrets
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from urllib.error import HTTPError
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parent.parent
REPOSITORY = "Kevin-Shih-Kai-Jen/quant-learning-internship-2027"
CHUNK = 1024 * 1024
RESERVE = 256 * 1024 * 1024
DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def relative_parts(value):
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise ValueError("Invalid relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or PureWindowsPath(value).drive:
        raise ValueError("Absolute paths are forbidden: " + value)
    if any(part in ("", ".", "..") for part in value.split("/")):
        raise ValueError("Noncanonical or escaping path: " + value)
    return path.parts


def open_parent(root, parts, create=False):
    descriptor = os.open(root, DIRECTORY_FLAGS)
    try:
        for component in parts[:-1]:
            try:
                child = os.open(component, DIRECTORY_FLAGS, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(component, 0o755, dir_fd=descriptor)
                except FileExistsError:
                    pass
                child = os.open(component, DIRECTORY_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def open_regular(root, path):
    parts = relative_parts(path)
    parent = open_parent(root, parts)
    try:
        descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
    finally:
        os.close(parent)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ValueError("Expected regular file: " + path)
    return os.fdopen(descriptor, "rb")


def read_json(root, path):
    with open_regular(root, path) as source:
        return json.load(source)


def checked_hash(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Invalid SHA-256 value")


def checked_integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError("Invalid nonnegative integer: " + name)


def load_catalog(tag):
    index = read_json(REPO_ROOT, "data/index.json")
    if index.get("schema_version") != 1 or not isinstance(index.get("snapshots"), list):
        raise ValueError("Unsupported snapshot index")
    tag = tag or index.get("latest")
    if not isinstance(tag, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", tag):
        raise ValueError("Invalid snapshot tag")
    snapshots = [item for item in index["snapshots"] if item.get("tag") == tag]
    if len(snapshots) != 1 or snapshots[0].get("status") != "verified":
        raise ValueError("Snapshot is missing, duplicated, or not marked verified: " + tag)
    expected_path = "data/catalogs/" + tag + ".json"
    if snapshots[0].get("catalog") != expected_path:
        raise ValueError("Unexpected catalog path")
    catalog = read_json(REPO_ROOT, expected_path)
    if (catalog.get("schema_version") != 1 or catalog.get("repository") != REPOSITORY
            or catalog.get("tag") != tag or not isinstance(catalog.get("packs"), list)):
        raise ValueError("Unsupported or inconsistent catalog")
    names = set()
    paths = set()
    for pack in catalog["packs"]:
        name = pack.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*\.tar", name):
            raise ValueError("Invalid pack name")
        if name in names:
            raise ValueError("Duplicate pack name: " + name)
        names.add(name)
        release_tag = pack.get("release_tag", tag)
        if not isinstance(release_tag, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", release_tag):
            raise ValueError("Invalid pack release_tag")
        checked_integer(pack.get("size"), "pack size")
        checked_hash(pack.get("sha256"))
        if not isinstance(pack.get("files"), list) or not pack["files"]:
            raise ValueError("Pack must contain file entries: " + name)
        for entry in pack["files"]:
            parts = relative_parts(entry.get("path"))
            if len(parts) < 2 or parts[0] != "research":
                raise ValueError("Files must be under research/: " + entry["path"])
            if entry["path"] in paths:
                raise ValueError("Duplicate data path: " + entry["path"])
            paths.add(entry["path"])
            checked_hash(entry.get("sha256"))
            for field in ("size", "mode", "mtime_ns"):
                checked_integer(entry.get(field), field)
            if entry["mode"] > 0o7777:
                raise ValueError("Invalid file permissions")
    return catalog


def select_files(catalog, experiment):
    prefix = None
    if experiment:
        prefix = experiment.rstrip("/")
        relative_parts(prefix)
        if not prefix.startswith("research/"):
            prefix = "research/" + prefix
    selected = []
    for pack in catalog["packs"]:
        entries = [entry for entry in pack["files"]
                   if prefix is None or entry["path"] == prefix or entry["path"].startswith(prefix + "/")]
        if entries:
            selected.append((pack, entries))
    if not selected:
        raise ValueError("No archived files match the selection")
    return selected


def digest_stream(source, expected_size, output=None):
    digest = hashlib.sha256()
    size = 0
    while True:
        block = source.read(CHUNK)
        if not block:
            return size, digest.hexdigest()
        size += len(block)
        if size > expected_size:
            raise ValueError("File exceeds catalog size")
        digest.update(block)
        if output is not None:
            output.write(block)


def check_digest(result, entry):
    if result != (entry["size"], entry["sha256"]):
        raise ValueError("Size or SHA-256 mismatch: " + entry.get("path", entry.get("name", "file")))


def local_matches(destination, entry):
    try:
        source = open_regular(destination, entry["path"])
    except FileNotFoundError:
        return False
    with source:
        check_digest(digest_stream(source, entry["size"]), entry)
    return True


def validate_tar(path, pack):
    """Validate the whole pack before extracting any selected member."""
    expected = {entry["path"]: entry for entry in pack["files"]}
    seen = set()
    with tarfile.open(path, "r:") as archive:
        for member in archive:
            relative_parts(member.name)
            if not member.isfile() or member.issparse():
                raise ValueError("Only ordinary, nonsparse files are allowed: " + member.name)
            if member.name not in expected or member.name in seen:
                raise ValueError("Unexpected or duplicate tar member: " + member.name)
            entry = expected[member.name]
            if member.size != entry["size"]:
                raise ValueError("Tar member size mismatch: " + member.name)
            with archive.extractfile(member) as source:
                check_digest(digest_stream(source, entry["size"]), entry)
            seen.add(member.name)
    if seen != set(expected):
        raise ValueError("Pack is missing catalog members")


def install_member(destination, entry, source):
    parts = relative_parts(entry["path"])
    parent = open_parent(destination, parts, create=True)
    temporary = None
    try:
        try:
            info = os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            info = None
        if info is not None:
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("Destination is not a regular file: " + entry["path"])
            if local_matches(destination, entry):
                return "SKIP"
        space = os.fstatvfs(parent)
        if space.f_bavail * space.f_frsize < entry["size"] + RESERVE:
            raise OSError("Insufficient free space while restoring " + entry["path"])
        temporary = ".data-archive-" + secrets.token_hex(12) + ".tmp"
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o600, dir_fd=parent)
        with os.fdopen(descriptor, "wb") as output:
            check_digest(digest_stream(source, entry["size"], output), entry)
            output.flush()
            os.fchmod(output.fileno(), entry["mode"])
            os.utime(output.fileno(), ns=(entry["mtime_ns"], entry["mtime_ns"]))
            os.fsync(output.fileno())
        os.link(temporary, parts[-1], src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
        os.unlink(temporary, dir_fd=parent)
        temporary = None
        os.fsync(parent)
        return "FETCHED"
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary, dir_fd=parent)
            except FileNotFoundError:
                pass
        os.close(parent)


def download_pack(pack, tag, directory):
    """Try public assets without credentials; retain gh access for private releases."""
    tag = pack.get("release_tag", tag)
    url = f"https://github.com/{REPOSITORY}/releases/download/{tag}/{pack['name']}"
    request = Request(url, headers={"User-Agent": "quant-data-archive"})
    try:
        response = urlopen(request, timeout=60)
    except HTTPError as error:
        if error.code not in (401, 403, 404):
            raise
        error.close()
        subprocess.run(["gh", "release", "download", tag, "--repo", REPOSITORY,
                        "--pattern", pack["name"], "--dir", str(directory)], check=True)
        return
    with response, (directory / pack["name"]).open("xb") as output:
        check_digest(digest_stream(response, pack["size"], output), pack)


def fetch(catalog, selected, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for pack, entries in selected:
        missing = []
        for entry in entries:
            if local_matches(destination, entry):
                print("SKIP  " + entry["path"], flush=True)
            else:
                missing.append(entry)
        if not missing:
            continue
        required = pack["size"] + sum(entry["size"] for entry in missing) + RESERVE
        # The temporary pack and output must both fit, even when on one volume.
        if shutil.disk_usage(destination.parent).free < required:
            raise OSError(f"Insufficient free disk space: requires {required:,} bytes for {pack['name']}")
        with tempfile.TemporaryDirectory(prefix=".data-archive-", dir=destination.parent) as temporary:
            print("DOWNLOADING  " + pack["name"], flush=True)
            download_pack(pack, catalog["tag"], Path(temporary))
            with open_regular(Path(temporary), pack["name"]) as source:
                check_digest(digest_stream(source, pack["size"]), pack)
            asset = Path(temporary) / pack["name"]
            validate_tar(asset, pack)
            wanted = {entry["path"]: entry for entry in missing}
            with tarfile.open(asset, "r:") as archive:
                for member in archive:
                    if member.name in wanted:
                        with archive.extractfile(member) as source:
                            result = install_member(destination, wanted[member.name], source)
                        print(result + "  " + member.name, flush=True)


def verify(selected, destination):
    failures = 0
    for _, entries in selected:
        for entry in entries:
            try:
                if not local_matches(destination, entry):
                    raise FileNotFoundError("Selected file has not been fetched")
                print("VERIFIED  " + entry["path"], flush=True)
            except (OSError, ValueError) as error:
                failures += 1
                print("ERROR  " + entry["path"] + ": " + str(error), file=sys.stderr, flush=True)
    print(f"Verification completed with {failures} error(s).", flush=True)
    return 1 if failures else 0


def list_catalog(catalog):
    experiments = defaultdict(lambda: [0, 0, set()])
    for pack in catalog["packs"]:
        for entry in pack["files"]:
            experiment = entry["path"].split("/")[1]
            experiments[experiment][0] += 1
            experiments[experiment][1] += entry["size"]
            experiments[experiment][2].add(pack["name"])
    print("Repository: " + REPOSITORY)
    print("Snapshot: " + catalog["tag"])
    print("EXPERIMENT\tFILES\tORIGINAL_BYTES\tPACKS")
    for name, (count, size, packs) in sorted(experiments.items()):
        print(f"{name}\t{count}\t{size}\t{len(packs)}")
    print(f"{len(catalog['packs'])} packs; {sum(p['size'] for p in catalog['packs']):,} download bytes total.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("list", "fetch", "verify"):
        sub = commands.add_parser(command)
        sub.add_argument("--tag", help="verified snapshot tag (default: index latest)")
        if command != "list":
            choice = sub.add_mutually_exclusive_group(required=command == "fetch")
            choice.add_argument("--experiment", help="original experiment directory or research/ prefix")
            choice.add_argument("--all", action="store_true", help="select all catalog files")
            sub.add_argument("--dest", type=Path, help="destination root containing research/ (default: repository root)")
    args = parser.parse_args(argv)
    try:
        catalog = load_catalog(args.tag)
        if args.command == "list":
            list_catalog(catalog)
            return 0
        selected = select_files(catalog, args.experiment)
        destination = (args.dest or REPO_ROOT).expanduser().resolve()
        if args.command == "verify":
            return verify(selected, destination)
        fetch(catalog, selected, destination)
        print("Fetch completed. All selected files are verified; temporary packs were removed.")
        return 0
    except (OSError, ValueError, KeyError, TypeError, AttributeError, tarfile.TarError,
            subprocess.CalledProcessError) as error:
        print("ERROR: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
