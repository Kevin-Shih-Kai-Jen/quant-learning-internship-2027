"""Check publication metadata and retrieve one complete NPZ member from each new pack."""
import argparse
import hashlib
import json
import subprocess
import tarfile
from concurrent.futures import ThreadPoolExecutor
from common import EXP, REPO, TMP, save, sha

REPOSITORY = 'Kevin-Shih-Kai-Jen/quant-learning-internship-2027'


def gh(*args):
    return subprocess.check_output(['gh', *args])


def api(path):
    return json.loads(gh('api', f'repos/{REPOSITORY}/{path}'))


def download(asset_id, start=None, end=None):
    args = ['gh', 'api', f'repos/{REPOSITORY}/releases/assets/{asset_id}',
            '-H', 'Accept: application/octet-stream']
    if start is not None:
        args += ['-H', f'Range: bytes={start}-{end}']
    for attempt in range(3):
        try:
            content = subprocess.check_output(args, timeout=60)
            if start is not None and len(content) != end-start+1:
                raise ValueError('Server did not return the requested byte range')
            return content
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError):
            if attempt == 2:
                raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    parser.add_argument('--git-commit', required=True)
    args = parser.parse_args()
    catalog_path = REPO/'data/catalogs'/f'{args.tag}.json'
    catalog = json.loads(catalog_path.read_text())
    release = api(f'releases/tags/{args.tag}')
    assert not release['draft']
    ref = api(f'git/ref/tags/{args.tag}')['object']
    if ref['type'] == 'tag':
        ref = api(f'git/tags/{ref["sha"]}')['object']
    assert ref['sha'] == args.git_commit
    assets = {a['name']:a for a in release['assets']}
    remote_catalog = download(assets[catalog_path.name]['id'])
    assert hashlib.sha256(remote_catalog).hexdigest() == sha(catalog_path)
    verified = []
    retrieved = []
    for pack in catalog['packs']:
        if pack.get('release_tag', args.tag) != args.tag:
            continue
        asset = assets[pack['name']]
        assert asset['state'] == 'uploaded' and asset['size'] == pack['size']
        assert asset['digest'] == 'sha256:' + pack['sha256']
        verified.append({'name':pack['name'], 'size':pack['size'], 'sha256':pack['sha256'], 'asset_id':asset['id']})
        offset = 0
        chosen = None
        for record in pack['files']:
            info = tarfile.TarInfo(record['path'])
            info.size, info.mode, info.mtime = record['size'], record['mode'], record['mtime_ns']//1_000_000_000
            header = info.tobuf(format=tarfile.PAX_FORMAT)
            if chosen is None and record['path'].endswith('.npz'):
                chosen = record, offset, header
            offset += len(header) + (record['size']+511)//512*512
        assert chosen is not None
        record, header_start, header = chosen
        assert download(asset['id'], header_start, header_start+len(header)-1) == header
        start = header_start+len(header)
        spans = [(x, min(x+256*1024, start+record['size'])-1) for x in range(start,start+record['size'],256*1024)]
        with ThreadPoolExecutor(max_workers=6) as pool:
            parts = list(pool.map(lambda span: download(asset['id'], *span), spans))
        content = b''.join(parts)
        assert len(content) == record['size'] and hashlib.sha256(content).hexdigest() == record['sha256']
        target = TMP/'published_retrieval'/record['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        retrieved.append({'path':record['path'], 'size':record['size'], 'sha256':record['sha256'], 'pack':pack['name'], 'byte_offset':start})
        print(json.dumps({'retrieved_and_verified':record['path'],'bytes':record['size']}),flush=True)
    receipt = {'tag':args.tag,'release_id':release['id'],'release_url':release['html_url'],
               'published':True,'tag_commit':args.git_commit,'catalog_download_sha256':sha(catalog_path),
               'new_pack_metadata':verified,'fully_retrieved_members':retrieved,
               'whole_packs_redownloaded':False,
               'verification_scope':'Every packed file was read and hashed during local pack creation; all new remote pack sizes and server SHA-256 digests match. One full NPZ member per new pack plus the catalog were downloaded and hashed after publication.'}
    save(EXP/'published_snapshot_verification.json',receipt)


if __name__ == '__main__':
    main()
