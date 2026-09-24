#!/usr/bin/env python3
"""Execute the user-approved, manifest-bound lossless archival plan."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
MANIFEST = BASE / 'COMPRESSED_DATA_MANIFEST.json'
RESERVE = 512 * 1024 * 1024
CHUNK = 4 * 1024 * 1024

def digest(path, compressed=False):
    h = hashlib.sha256()
    size = 0
    opener = gzip.open if compressed else open
    with opener(path, 'rb') as stream:
        for block in iter(lambda: stream.read(CHUNK), b''):
            h.update(block)
            size += len(block)
    return h.hexdigest(), size

def save(value):
    temporary = MANIFEST.with_suffix('.json.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, MANIFEST)
    descriptor = os.open(BASE, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

def main():
    plan = json.loads((BASE / 'COMPRESSION_PLAN.json').read_text())
    original = {v['path']: v for v in json.loads((BASE / 'PROJECT_MANIFEST_SHA256.json').read_text())['files']}
    state = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {
        'format': 'gzip', 'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'authorization': 'User approved lossless compression replacing large originals after verification, 2026-09-24',
        'project_root': str(ROOT), 'files': []}
    done = {v['path']: v for v in state['files']}
    for index, candidate in enumerate(plan['candidates'], 1):
        rel = candidate['path']
        if rel in done and not (ROOT / rel).exists():
            continue
        if rel in done:
            raise RuntimeError('Previously archived source exists; inspect before continuing: ' + rel)
        path = Path(rel)
        if path.is_absolute() or '..' in path.parts or not rel.startswith('jpx_'):
            raise RuntimeError('Invalid planned path: ' + rel)
        source = ROOT / path
        if source.is_symlink() or not source.is_file() or not source.resolve().is_relative_to(ROOT):
            raise RuntimeError('Unsafe source: ' + rel)
        expected = original[rel]
        before = source.stat()
        if before.st_size != expected['size'] or before.st_mtime_ns != expected['mtime_ns']:
            raise RuntimeError('Source changed since inventory: ' + rel)
        archive = BASE / 'compressed_data' / (rel + '.gz')
        archive.parent.mkdir(parents=True, exist_ok=True)
        temporary = archive.with_name(archive.name + '.partial')
        if archive.exists() or temporary.exists():
            raise RuntimeError('Archive already exists: ' + str(archive))
        if shutil.disk_usage(BASE).free < before.st_size + RESERVE:
            raise RuntimeError('Insufficient free space for safe archival of: ' + rel)
        sha = hashlib.sha256()
        try:
            with source.open('rb') as src, temporary.open('xb') as target:
                with gzip.GzipFile(filename='', mode='wb', compresslevel=4, fileobj=target, mtime=0) as output:
                    for block in iter(lambda: src.read(CHUNK), b''):
                        if shutil.disk_usage(BASE).free < RESERVE:
                            raise RuntimeError('Disk reserve reached; source retained')
                        sha.update(block)
                        output.write(block)
                target.flush()
                os.fsync(target.fileno())
            restored_hash, restored_size = digest(temporary, compressed=True)
            after = source.stat()
            if sha.hexdigest() != expected['sha256'] or restored_hash != expected['sha256'] or restored_size != expected['size']:
                raise RuntimeError('Round-trip verification failed: ' + rel)
            if (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) != (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns):
                raise RuntimeError('Source changed during compression: ' + rel)
            archive_hash, archive_size = digest(temporary)
            if archive_size >= restored_size:
                temporary.unlink()
                print(json.dumps({'skipped_no_saving': rel}), flush=True)
                continue
            os.rename(temporary, archive)
            entry = dict(expected, archive_path=str(archive.relative_to(BASE)), archive_size=archive_size,
                         archive_sha256=archive_hash, original_retained=True, round_trip_verified=True)
            state['files'].append(entry)
            save(state)  # Durable recovery record before any removal.
            current = source.stat()
            if (current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns) != (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns):
                raise RuntimeError('Source changed before removal; both copies retained: ' + rel)
            source.unlink()
            entry['original_retained'] = False
            save(state)
            print(json.dumps({'completed': index, 'of': len(plan['candidates']), 'path': rel,
                              'original_bytes': restored_size, 'archive_bytes': archive_size,
                              'saved_bytes': restored_size - archive_size}), flush=True)
        except BaseException:
            if temporary.exists():
                temporary.unlink()
            raise
    state['completed_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    state['original_bytes'] = sum(v['size'] for v in state['files'])
    state['compressed_bytes'] = sum(v['archive_size'] for v in state['files'])
    state['saved_bytes'] = sum(v['size'] - v['archive_size'] for v in state['files'] if not v['original_retained'])
    save(state)
    print(json.dumps({k:v for k,v in state.items() if k != 'files'}), flush=True)

if __name__ == '__main__':
    main()
