#!/usr/bin/env python3
"""Restore and verify an original workspace from a versioned JPX snapshot.

Requires Python 3.9+ on macOS/Linux, with no third-party packages. Existing files
are accepted only when their size and SHA-256 match the manifest. New files are
verified in exclusive temporary files, then installed atomically without
overwriting another file. An interrupted restore can safely be run again.
"""

import argparse
import contextlib
import gzip
import hashlib
import json
import os
from pathlib import PurePosixPath
import re
import secrets
import stat
import sys
import tarfile


BLOCK = 1024 * 1024
MANIFEST = "SNAPSHOT_FILE_MANIFEST.json.gz"
DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
READ_FLAGS = os.O_RDONLY | os.O_NOFOLLOW


class RestoreError(Exception):
    pass


def relative_parts(value):
    """Reject ambiguous paths rather than silently normalizing them."""
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        raise RestoreError(f"Invalid relative path: {value!r}")
    parts = value.split("/")
    if any(part in ("", ".", "..") for part in parts) or ":" in parts[0]:
        raise RestoreError(f"Unsafe relative path: {value!r}")
    return parts


def nonnegative_int(value, label):
    if type(value) is not int or value < 0:
        raise RestoreError(f"{label} must be a nonnegative integer")
    return value


def digest_value(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise RestoreError("Invalid SHA-256 in manifest")
    return value.lower()


class Tree:
    """Access a tree through directory descriptors; never follow symlinks."""

    def __init__(self, path, create=False):
        absolute = os.path.abspath(os.fspath(path))
        self.fd = os.open("/", DIR_FLAGS)
        try:
            for part in absolute.split("/")[1:]:
                if not part:
                    continue
                if create:
                    try:
                        os.mkdir(part, mode=0o755, dir_fd=self.fd)
                    except FileExistsError:
                        pass
                next_fd = os.open(part, DIR_FLAGS, dir_fd=self.fd)
                os.close(self.fd)
                self.fd = next_fd
        except BaseException:
            os.close(self.fd)
            raise

    def close(self):
        os.close(self.fd)

    @contextlib.contextmanager
    def parent(self, path, create=False):
        parts = relative_parts(path)
        fd = os.dup(self.fd)
        try:
            for part in parts[:-1]:
                if create:
                    try:
                        os.mkdir(part, mode=0o755, dir_fd=fd)
                    except FileExistsError:
                        pass
                next_fd = os.open(part, DIR_FLAGS, dir_fd=fd)
                os.close(fd)
                fd = next_fd
            yield fd, parts[-1]
        finally:
            os.close(fd)

    @contextlib.contextmanager
    def read(self, path):
        with self.parent(path) as (fd, leaf):
            file_fd = os.open(leaf, READ_FLAGS | os.O_NONBLOCK, dir_fd=fd)
            try:
                if not stat.S_ISREG(os.fstat(file_fd).st_mode):
                    raise RestoreError(f"Not a regular file: {path}")
                stream = os.fdopen(file_fd, "rb")
            except BaseException:
                os.close(file_fd)
                raise
            with stream:
                yield stream


def load_manifest(archive):
    with archive.read(MANIFEST) as raw, gzip.GzipFile(fileobj=raw, mode="rb") as zipped:
        manifest = json.load(zipped)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise RestoreError("Unsupported manifest schema")
    rows = manifest.get("files")
    if not isinstance(rows, list):
        raise RestoreError("Manifest files must be a list")
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise RestoreError("Each manifest entry must be an object")
        relative_parts(row.get("path"))
        if row["path"] in seen:
            raise RestoreError(f"Duplicate output path: {row['path']}")
        seen.add(row["path"])
        nonnegative_int(row.get("size"), "size")
        row["sha256"] = digest_value(row.get("sha256"))
        nonnegative_int(row.get("mode"), "mode")
        nonnegative_int(row.get("mtime_ns"), "mtime_ns")
        storage = row.get("storage")
        if not isinstance(storage, dict):
            raise RestoreError("Missing storage object")
        kind = storage.get("kind")
        if kind in ("direct", "tar-gzip"):
            relative_parts(storage.get("path"))
            if kind == "tar-gzip":
                relative_parts(storage.get("member"))
        elif kind == "concat":
            parts = storage.get("parts")
            if not isinstance(parts, list) or not parts:
                raise RestoreError("Concatenated files require at least one part")
            for part in parts:
                if not isinstance(part, dict):
                    raise RestoreError("Invalid chunk entry")
                relative_parts(part.get("path"))
                nonnegative_int(part.get("size"), "chunk size")
                part["sha256"] = digest_value(part.get("sha256"))
            if sum(part["size"] for part in parts) != row["size"]:
                raise RestoreError(f"Chunk sizes do not sum to file size: {row['path']}")
        else:
            raise RestoreError(f"Unknown storage kind: {kind!r}")
    for path in seen:
        if any(str(parent) in seen for parent in PurePosixPath(path).parents if str(parent) != "."):
            raise RestoreError(f"A manifest file is also a parent directory: {path}")
    return rows


def stream_hash(stream):
    digest = hashlib.sha256()
    size = 0
    while True:
        data = stream.read(BLOCK)
        if not data:
            return size, digest.hexdigest()
        size += len(data)
        digest.update(data)


def existing_matches(dest, row):
    try:
        with dest.read(row["path"]) as stream:
            if os.fstat(stream.fileno()).st_size != row["size"]:
                raise RestoreError(f"Existing file has a different size: {row['path']}")
            size, digest = stream_hash(stream)
    except FileNotFoundError:
        return False
    if (size, digest) != (row["size"], row["sha256"]):
        raise RestoreError(f"Existing file has a different SHA-256: {row['path']}")
    return True


def copy_stream(stream, output, full_digest, expected_size, expected_digest=None):
    digest = hashlib.sha256()
    total = 0
    while True:
        data = stream.read(BLOCK)
        if not data:
            break
        total += len(data)
        if total > expected_size:
            raise RestoreError("Source exceeds its declared size")
        digest.update(data)
        full_digest.update(data)
        output.write(data)
    if total != expected_size:
        raise RestoreError("Source size does not match its declaration")
    if expected_digest is not None and digest.hexdigest() != expected_digest:
        raise RestoreError("Chunk SHA-256 does not match its declaration")
    return total


def install(dest, row, write_content):
    """No-clobber atomic install, including when another process creates a file."""
    with dest.parent(row["path"], create=True) as (parent_fd, leaf):
        temp = ".restore-" + secrets.token_hex(16) + ".tmp"
        file_fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                          0o600, dir_fd=parent_fd)
        try:
            with os.fdopen(file_fd, "wb") as output:
                digest = hashlib.sha256()
                size = write_content(output, digest)
                if size != row["size"] or digest.hexdigest() != row["sha256"]:
                    raise RestoreError(f"Restored content fails verification: {row['path']}")
                output.flush()
                # Ordinary permissions are preserved; privileged mode bits are never restored.
                os.fchmod(output.fileno(), row["mode"] & 0o777)
                os.utime(output.fileno(), ns=(row["mtime_ns"], row["mtime_ns"]))
                os.fsync(output.fileno())
            try:
                # rename() could replace a newly created target. link()+unlink()
                # publishes this verified inode atomically and refuses any target.
                os.link(temp, leaf, src_dir_fd=parent_fd, dst_dir_fd=parent_fd,
                        follow_symlinks=False)
            except FileExistsError:
                if not existing_matches(dest, row):
                    raise RestoreError(f"Destination changed during restore: {row['path']}")
        finally:
            os.unlink(temp, dir_fd=parent_fd)


def restore_plain(archive, dest, row):
    storage = row["storage"]

    def write_content(output, digest):
        if storage["kind"] == "direct":
            with archive.read(storage["path"]) as source:
                return copy_stream(source, output, digest, row["size"])
        total = 0
        for part in storage["parts"]:
            with archive.read(part["path"]) as source:
                total += copy_stream(source, output, digest, part["size"], part["sha256"])
        return total

    install(dest, row, write_content)


def restore_tar(archive, dest, package, rows):
    wanted = {}
    for row in rows:
        wanted.setdefault(row["storage"]["member"], []).append(row)
    seen = set()
    restored = 0
    with archive.read(package) as raw, tarfile.open(fileobj=raw, mode="r|gz") as tar:
        for member in tar:
            # Stream members; never extract paths or links using tarfile helpers.
            name = member.name.rstrip("/") if member.isdir() else member.name
            relative_parts(name)
            if name in seen:
                raise RestoreError(f"Duplicate tar member in {package}: {name}")
            seen.add(name)
            if not (member.isfile() or member.isdir()):
                raise RestoreError(f"Links/special files are forbidden in {package}: {name}")
            matches = wanted.pop(name, None)
            if matches is None:
                continue
            if not member.isfile() or len(matches) != 1:
                raise RestoreError(f"Expected one regular tar member: {name}")
            row = matches[0]
            if member.size != row["size"]:
                raise RestoreError(f"Tar member size mismatch: {name}")
            with tar.extractfile(member) as source:
                install(dest, row, lambda output, digest: copy_stream(
                    source, output, digest, row["size"]))
            restored += 1
    if wanted:
        raise RestoreError(f"Missing tar members in {package}: {', '.join(sorted(wanted)[:3])}")
    return restored


def run(archive_path, dest_path, verify_only=False):
    with contextlib.ExitStack() as stack:
        archive = Tree(archive_path)
        stack.callback(archive.close)
        rows = load_manifest(archive)
        dest = Tree(dest_path, create=not verify_only)
        stack.callback(dest.close)
        pending = []
        skipped = 0
        # Preflight every existing target before writing any content.
        for row in rows:
            if existing_matches(dest, row):
                skipped += 1
            elif verify_only:
                raise RestoreError(f"Missing destination file: {row['path']}")
            else:
                pending.append(row)
        restored = 0
        tar_groups = {}
        for row in pending:
            if row["storage"]["kind"] == "tar-gzip":
                tar_groups.setdefault(row["storage"]["path"], []).append(row)
            else:
                restore_plain(archive, dest, row)
                restored += 1
        for package, group in tar_groups.items():
            restored += restore_tar(archive, dest, package, group)
        return {"mode": "verify-only" if verify_only else "restore",
                "manifest_files": len(rows), "restored": restored,
                "existing_verified": skipped, "verified_bytes": sum(row["size"] for row in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", required=True, help="Directory containing the snapshot manifest")
    parser.add_argument("--dest", required=True, help="Original-workspace destination (symlinks forbidden)")
    parser.add_argument("--verify-only", action="store_true", help="Only hash existing destination files; write nothing")
    args = parser.parse_args()
    try:
        result = run(args.archive_root, args.dest, args.verify_only)
    except (RestoreError, OSError, ValueError, KeyError, TypeError, tarfile.TarError, EOFError) as exc:
        print(f"Restore failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
