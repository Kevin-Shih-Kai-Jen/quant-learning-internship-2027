#!/usr/bin/env python3
"""Stage readable research and upload verified, resumable data packs to a draft Release.

No original files are removed. Final git commit/push and release publication are
deliberate subsequent steps; see docs/DATA_GUIDE.md.
"""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
BLOCK = 4 * 1024 * 1024

def read_json(path):
    return json.loads(path.read_text())

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)

def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(BLOCK), b''):
            h.update(block)
    return h.hexdigest()

def gh(*args, capture=True):
    result = subprocess.run(['gh', *args], check=True, text=True,
                            stdout=subprocess.PIPE if capture else None)
    return result.stdout.strip() if capture else None

def api(endpoint):
    return json.loads(gh('api', endpoint))

def safe_record_path(value):
    p = Path(value)
    if p.is_absolute() or '..' in p.parts or not value.startswith('research/'):
        raise ValueError('Invalid research path: ' + value)
    return p

def classify(path, size, config):
    ext = path.suffix.lower()
    if ext in config['text_extensions'] and size <= config['text_max_bytes']:
        return 'git'
    if ext == '.csv' and size <= config['csv_git_max_bytes']:
        return 'git'
    if ext in {'.pdf', '.docx'} and size <= config['document_git_max_bytes']:
        return 'git'
    if ext in {'.png', '.jpg', '.jpeg', '.svg'} and size <= config['image_git_max_bytes']:
        return 'git'
    if path.name in {'.gitignore', '.gitkeep', 'Makefile', 'Dockerfile'}:
        return 'git'
    return 'release'

def previous_packs():
    index_path = ROOT / 'data/index.json'
    if not index_path.exists():
        return []
    index = read_json(index_path)
    latest = next(x for x in index['snapshots'] if x['tag'] == index['latest'])
    catalog = read_json(ROOT / latest['catalog'])
    result = []
    for pack in catalog['packs']:
        result.append(dict(pack, release_tag=pack.get('release_tag', catalog['tag'])))
    return result

def prepare(workspace, tag, config, plan_path):
    if plan_path.exists():
        existing = read_json(plan_path)
        if existing['workspace'] != str(workspace):
            raise ValueError('Existing plan belongs to another workspace')
        return existing
    if (ROOT / 'data/catalogs' / (tag + '.json')).exists():
        raise ValueError('Snapshot tag already finalized; use a new tag')
    inherited = previous_packs()
    existing_data = {f['path']: f for pack in inherited for f in pack['files']}
    records = []
    excluded = collections.Counter()
    grouped = collections.defaultdict(list)
    for directory, dirs, names in os.walk(workspace):
        kept = []
        for name in dirs:
            p = Path(directory) / name
            if name in config['excluded_directories'] or p.resolve() == ROOT:
                excluded['directory:' + name] += 1
            elif p.is_symlink():
                raise ValueError('Review symbolic link before uploading: ' + str(p))
            else:
                kept.append(name)
        dirs[:] = sorted(kept)
        for name in sorted(names):
            source = Path(directory) / name
            relative = source.relative_to(workspace)
            if name in config['excluded_filenames'] or name.endswith('.pyc'):
                excluded['file:' + name] += 1
                continue
            if name == '.env' or (name.startswith('.env.') and name != '.env.example'):
                raise ValueError('Potential credential file requires review: ' + str(relative))
            if source.is_symlink() or not source.is_file():
                raise ValueError('Non-regular file requires review: ' + str(relative))
            before = source.stat()
            storage = classify(source, before.st_size, config)
            record = {'path': 'research/' + relative.as_posix(), 'size': before.st_size,
                      'sha256': sha_file(source), 'mtime_ns': before.st_mtime_ns,
                      'mode': before.st_mode & 0o777, 'storage': storage}
            after = source.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError('File changed during planning: ' + str(relative))
            records.append(record)
            old = existing_data.get(record['path'])
            if old:
                if old['sha256'] != record['sha256']:
                    raise ValueError('Historical data changed; use a new experiment path: ' + record['path'])
                record['storage'] = 'release'
                continue
            if storage == 'git':
                destination = ROOT / record['path']
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.is_symlink():
                    raise ValueError('Symlink destination refused')
                if source.resolve() != destination.resolve():
                    shutil.copy2(source, destination)
                if sha_file(destination) != record['sha256']:
                    raise ValueError('Staged copy mismatch: ' + record['path'])
            else:
                previous_git_copy = ROOT / record['path']
                if previous_git_copy.exists() and sha_file(previous_git_copy) != record['sha256']:
                    raise ValueError('Git-to-Release transition requires a new experiment path or an explicit reviewed Git removal: ' + record['path'])
                if before.st_size > config['pack_max_bytes'] - 1024 * 1024:
                    raise ValueError('Single file exceeds pack policy; archive it into smaller parts first: ' + str(relative))
                components = relative.parts
                group = components[0] if len(components) > 1 else 'root-deliverables'
                if len(components) > 3 and components[1] == 'compressed_data':
                    group = 'compressed-' + components[2]
                grouped[group].append(record)
    packs = []
    for group, files in sorted(grouped.items()):
        current = []
        length = 0
        for record in files:
            estimated = ((record['size'] + 511) // 512) * 512 + 4096
            if current and length + estimated > config['pack_max_bytes'] - 10240:
                packs.append((group, current))
                current = []
                length = 0
            current.append(record)
            length += estimated
        if current:
            packs.append((group, current))
    entries = []
    for i, (group, files) in enumerate(packs, 1):
        slug = re.sub('[^A-Za-z0-9._-]', '-', group)[:85]
        entries.append({'name': f'{tag}-data-{i:03d}-{slug}.tar', 'release_tag': tag, 'files': files})
    plan = {'schema_version': 1, 'repository': config['repository'], 'tag': tag,
            'workspace': str(workspace), 'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'excluded': dict(excluded), 'git_files': [f for f in records if f['storage'] == 'git'],
            'inherited_packs': inherited, 'new_packs': entries}
    write_json(plan_path, plan)
    (plan_path.parent / (tag + '-git-paths.nul')).write_bytes(b'\0'.join(f['path'].encode() for f in plan['git_files']) + b'\0')
    write_json(ROOT / 'data/inventories' / (tag + '.json'), {
        'schema_version': 1, 'tag': tag, 'excluded': dict(excluded), 'files': records})
    print(json.dumps({'prepared_git_files': len(plan['git_files']), 'new_packs': len(entries),
                      'new_payload_bytes': sum(f['size'] for p in entries for f in p['files'])}), flush=True)
    return plan

def find_release(repo, tag):
    # GitHub's by-tag endpoint may omit unpublished drafts; list includes drafts
    # visible to the authenticated owner.
    pages = json.loads(gh('api', '--paginate', '--slurp', f'repos/{repo}/releases?per_page=100'))
    matches = [r for page in pages for r in page if r['tag_name'] == tag]
    if len(matches) != 1:
        raise ValueError('Expected one existing draft or published release for tag: ' + tag)
    return matches[0]

def release_assets(repo, release_id):
    pages = json.loads(gh('api', '--paginate', '--slurp', f'repos/{repo}/releases/{release_id}/assets?per_page=100'))
    return {a['name']: a for page in pages for a in page}

def require_asset(asset, pack):
    if asset.get('state') != 'uploaded' or asset['size'] != pack['size'] or asset.get('digest') != 'sha256:' + pack['sha256']:
        raise ValueError('Remote size/digest mismatch or incomplete asset: ' + pack['name'])

def build_pack(workspace, pack, target):
    need = sum(f['size'] + 4096 for f in pack['files'])
    if shutil.disk_usage(target.parent).free < need + 512 * 1024 * 1024:
        raise ValueError('Insufficient free space to stage one pack safely')
    with tarfile.open(target, 'w', format=tarfile.PAX_FORMAT) as archive:
        for record in pack['files']:
            rel = safe_record_path(record['path'])
            source = workspace / Path(*rel.parts[1:])
            if source.is_symlink() or not source.resolve().is_relative_to(workspace):
                raise ValueError('Unsafe source path: ' + str(source))
            st = source.stat()
            if st.st_size != record['size'] or st.st_mtime_ns != record['mtime_ns']:
                raise ValueError('Source changed after planning: ' + record['path'])
            info = tarfile.TarInfo(record['path'])
            info.size = record['size']
            info.mode = record['mode']
            info.mtime = record['mtime_ns'] // 1_000_000_000
            info.pax_headers = {}
            with source.open('rb') as stream:
                archive.addfile(info, stream)
    expected = {f['path']: f for f in pack['files']}
    with tarfile.open(target, 'r:') as archive:
        members = archive.getmembers()
        if len(members) != len(expected):
            raise ValueError('Archive membership mismatch')
        for member in members:
            record = expected[member.name]
            h = hashlib.sha256()
            stream = archive.extractfile(member)
            for block in iter(lambda: stream.read(BLOCK), b''):
                h.update(block)
            if h.hexdigest() != record['sha256'] or member.size != record['size']:
                raise ValueError('Packed member mismatch: ' + member.name)
    pack['size'] = target.stat().st_size
    pack['sha256'] = sha_file(target)

def upload(plan, plan_path):
    repo, tag = plan['repository'], plan['tag']
    meta = api('repos/' + repo)
    if not meta['private']:
        raise ValueError('This archive must remain private')
    release = find_release(repo, tag)
    if not release['draft']:
        raise ValueError('Only a draft release can receive this upload')
    known = release_assets(repo, release['id'])
    done = 0
    for i, pack in enumerate(plan['new_packs'], 1):
        remote = known.get(pack['name'])
        if remote and 'sha256' in pack:
            require_asset(remote, pack)
            pack['remote_id'] = remote['id']
            done += pack['size']
            print(json.dumps({'resumed_verified': i, 'of': len(plan['new_packs']), 'name': pack['name']}), flush=True)
            continue
        with tempfile.TemporaryDirectory(prefix='quant-release-') as directory:
            target = Path(directory) / pack['name']
            build_pack(Path(plan['workspace']), pack, target)
            write_json(plan_path, plan)
            if remote:
                require_asset(remote, pack)
            else:
                print(json.dumps({'uploading': i, 'of': len(plan['new_packs']), 'name': pack['name'], 'size': pack['size']}), flush=True)
                last_error = None
                for attempt in range(3):
                    try:
                        gh('release', 'upload', tag, str(target), '--repo', repo, capture=False)
                    except subprocess.CalledProcessError as error:
                        last_error = error
                    known = release_assets(repo, release['id'])
                    remote = known.get(pack['name'])
                    if remote:
                        require_asset(remote, pack)
                        break
                    if attempt < 2:
                        time.sleep(3 * (attempt + 1))
                if not remote:
                    raise last_error or ValueError('Missing remote asset')
            require_asset(remote, pack)
            pack['remote_id'] = remote['id']
            write_json(plan_path, plan)
            done += pack['size']
            print(json.dumps({'verified': i, 'of': len(plan['new_packs']), 'uploaded_verified_bytes': done,
                              'name': pack['name'], 'sha256': pack['sha256']}), flush=True)
    catalog = {'schema_version': 1, 'repository': repo, 'tag': tag,
               'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
               'packs': plan['inherited_packs'] + plan['new_packs']}
    catalog_path = ROOT / 'data/catalogs' / (tag + '.json')
    write_json(catalog_path, catalog)
    # Independently re-read GitHub asset metadata after all transfers.
    verification = []
    by_tag = {}
    for pack in catalog['packs']:
        ref = pack.get('release_tag', tag)
        if ref not in by_tag:
            other = find_release(repo, ref)
            by_tag[ref] = release_assets(repo, other['id'])
        remote = by_tag[ref][pack['name']]
        require_asset(remote, pack)
        verification.append({'name': pack['name'], 'release_tag': ref, 'size': pack['size'],
                             'sha256': pack['sha256'], 'github_digest': remote['digest'], 'asset_id': remote['id']})
    write_json(ROOT / 'data/verification' / (tag + '.json'), {
        'verified_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'status': 'remote-assets-verified',
        'repository_private': True, 'all_asset_digests_match': True, 'assets': verification,
        'note': 'GitHub server-reported SHA256 and sizes verified; draft must still be published after catalog commit.'})
    index_path = ROOT / 'data/index.json'
    index = read_json(index_path) if index_path.exists() else {'schema_version': 1, 'snapshots': []}
    index['snapshots'] = [x for x in index['snapshots'] if x['tag'] != tag]
    index['snapshots'].append({'tag': tag, 'catalog': str(catalog_path.relative_to(ROOT)), 'status': 'verified'})
    index['latest'] = tag
    write_json(index_path, index)
    print(json.dumps({'result': 'all_data_assets_verified', 'packs': len(catalog['packs']),
                      'bytes': sum(p['size'] for p in catalog['packs']),
                      'next': 'Commit and push catalogs, attach catalog and checksum assets, then publish draft against that commit.'}), flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', args.tag):
        parser.error('Tag must be a simple version/date label')
    workspace = args.workspace.resolve()
    if workspace == ROOT or not workspace.is_dir():
        parser.error('Workspace must be a separate source directory, not the repository root')
    config = read_json(ROOT / 'archive-config.json')
    plan_path = ROOT / 'data/upload-plans' / (args.tag + '.json')
    plan = prepare(workspace, args.tag, config, plan_path)
    if not args.prepare_only:
        upload(plan, plan_path)

if __name__ == '__main__':
    main()
