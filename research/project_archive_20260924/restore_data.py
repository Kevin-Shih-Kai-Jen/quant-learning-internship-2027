#!/usr/bin/env python3
"""List, verify, or safely restore the project's lossless gzip archives."""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import secrets
import stat
import sys


ARCHIVE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ARCHIVE_ROOT.parent
MANIFEST = "COMPRESSED_DATA_MANIFEST.json"
CHUNK = 1024 * 1024
RESERVE = 256 * 1024 * 1024
DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def relative_parts(value):
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise ValueError("Invalid relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or PureWindowsPath(value).drive or ".." in path.parts:
        raise ValueError("Absolute paths and parent traversal are forbidden: " + value)
    if not path.parts or any(p in ("", ".", "..") for p in value.split("/")):
        raise ValueError("Path must use canonical relative components: " + value)
    return path.parts


def open_parent(root, parts, create=False):
    """Traverse with directory descriptors; never follow a symlink."""
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


def open_regular(root, relative):
    parts = relative_parts(relative)
    parent = open_parent(root, parts)
    try:
        descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
    finally:
        os.close(parent)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ValueError("Expected a regular file: " + relative)
    return os.fdopen(descriptor, "rb")


def load_entries():
    with open_regular(ARCHIVE_ROOT, MANIFEST) as stream:
        manifest = json.load(stream)
    if manifest.get("format") != "gzip" or not isinstance(manifest.get("files"), list):
        raise ValueError("Unsupported or incomplete archive manifest")
    seen = set()
    seen_archives = set()
    for entry in manifest["files"]:
        if not isinstance(entry, dict):
            raise ValueError("Invalid manifest entry")
        parts = relative_parts(entry.get("path"))
        archive_parts = relative_parts(entry.get("archive_path"))
        if parts[0] == ARCHIVE_ROOT.name:
            raise ValueError("Restoring over the recovery archive is forbidden")
        if archive_parts[0] != "compressed_data":
            raise ValueError("Archive payload must be inside compressed_data")
        if entry["path"] in seen or entry["archive_path"] in seen_archives:
            raise ValueError("Duplicate manifest path")
        seen.add(entry["path"])
        seen_archives.add(entry["archive_path"])
        for key in ("size", "archive_size", "mtime_ns", "mode"):
            if type(entry.get(key)) is not int or entry[key] < 0:
                raise ValueError("Invalid " + key + " for " + entry["path"])
        if entry["mode"] > 0o7777:
            raise ValueError("Invalid permission mode")
        for key in ("sha256", "archive_sha256"):
            if not isinstance(entry.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", entry[key]):
                raise ValueError("Invalid " + key + " for " + entry["path"])
    return manifest["files"]


def stream_digest(stream, output=None, expected_size=None):
    digest = hashlib.sha256()
    size = 0
    while True:
        block = stream.read(CHUNK)
        if not block:
            return size, digest.hexdigest()
        size += len(block)
        if expected_size is not None and size > expected_size:
            raise ValueError("Payload exceeds its recorded size")
        digest.update(block)
        if output is not None:
            output.write(block)


def validate_result(result, size, digest, label):
    if result != (size, digest):
        raise ValueError("Size or SHA-256 mismatch: " + label)


def verified_payload(entry, output=None):
    with open_regular(ARCHIVE_ROOT, entry["archive_path"]) as source:
        validate_result(stream_digest(source, expected_size=entry["archive_size"]),
                        entry["archive_size"], entry["archive_sha256"], entry["archive_path"])
        source.seek(0)
        with gzip.GzipFile(fileobj=source, mode="rb") as inflated:
            validate_result(stream_digest(inflated, output, entry["size"]),
                            entry["size"], entry["sha256"], entry["path"])


def matching_original(parent, name, entry):
    try:
        info = os.stat(name, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("Destination exists and is not a regular file: " + entry["path"])
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
    with os.fdopen(descriptor, "rb") as existing:
        if not stat.S_ISREG(os.fstat(existing.fileno()).st_mode):
            raise ValueError("Destination changed while checking")
        validate_result(stream_digest(existing, expected_size=entry["size"]),
                        entry["size"], entry["sha256"], "existing " + entry["path"])
    return True


def restore_entry(entry):
    parts = relative_parts(entry["path"])
    parent = open_parent(PROJECT_ROOT, parts, create=True)
    temporary = None
    try:
        if matching_original(parent, parts[-1], entry):
            return "SKIP (identical original exists)"
        disk = os.fstatvfs(parent)
        available = disk.f_bavail * disk.f_frsize
        if available < entry["size"] + RESERVE:
            raise OSError("Insufficient free disk space for " + entry["path"]
                          + "; requires original size plus 256 MiB reserve")
        temporary = ".restore-" + secrets.token_hex(12) + ".tmp"
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o600, dir_fd=parent)
        with os.fdopen(descriptor, "wb") as output:
            verified_payload(entry, output)
            output.flush()
            os.fchmod(output.fileno(), entry["mode"])
            os.utime(output.fileno(), ns=(entry["mtime_ns"], entry["mtime_ns"]))
            os.fsync(output.fileno())
        # link() fails if any destination (including a symlink) appeared meanwhile.
        os.link(temporary, parts[-1], src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
        os.unlink(temporary, dir_fd=parent)
        temporary = None
        os.fsync(parent)
        return "RESTORED"
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary, dir_fd=parent)
            except FileNotFoundError:
                pass
        os.close(parent)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--list", action="store_true", help="list archived originals (default)")
    action.add_argument("--verify", action="store_true", help="verify compressed and restored SHA-256 hashes")
    action.add_argument("--restore", action="store_true", help="restore originals while retaining gzip copies")
    parser.add_argument("--path", help="exact project-relative original path or directory prefix; requires --restore")
    args = parser.parse_args()
    if args.path and not args.restore:
        parser.error("--path requires --restore")
    entries = load_entries()
    if args.path:
        prefix = args.path.rstrip("/")
        relative_parts(prefix)
        entries = [entry for entry in entries if entry["path"] == prefix or entry["path"].startswith(prefix + "/")]
        if not entries:
            raise ValueError("No manifest paths match: " + prefix)
    total = sum(entry["size"] for entry in entries)
    compressed = sum(entry["archive_size"] for entry in entries)
    print(f"{len(entries)} files | original {total:,} bytes ({total / 1024**3:.3f} GiB)"
          f" | gzip {compressed:,} bytes ({compressed / 1024**3:.3f} GiB)", flush=True)
    failures = 0
    for entry in entries:
        try:
            if args.verify:
                verified_payload(entry)
                result = "VERIFIED"
            elif args.restore:
                result = restore_entry(entry)
            else:
                result = f"{entry['size']:,} bytes"
            print(result + "  " + entry["path"], flush=True)
        except (OSError, ValueError, EOFError) as error:
            failures += 1
            print("ERROR  " + entry["path"] + ": " + str(error), file=sys.stderr, flush=True)
            if args.restore:
                # Stop early to avoid repeated failures or consuming more disk unexpectedly.
                break
    if args.verify or args.restore:
        print(f"Completed with {failures} error(s). Compressed copies were retained.", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, EOFError, KeyError, TypeError) as error:
        print("ERROR: " + str(error), file=sys.stderr)
        sys.exit(1)
