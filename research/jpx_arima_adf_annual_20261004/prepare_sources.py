"""Verify local source bytes against versioned inventory; verify draft pack separately."""
import argparse
import json
import shutil
import sys
from pathlib import Path
from common import EXP, REPO, TMP, SOURCE, INPUT, OUT, OLD, ORDERS, save, sha

INVENTORY = REPO / 'data/inventories/snapshot-2026-09-28-arima-learning.json'
PREFIX = 'research/jpx_arima_learning_20260928/originals/work/arima_grid/'


def local_sources():
    records = {f['path']: f for f in json.loads(INVENTORY.read_text())['files']}
    wanted = ['prices.npz', 'labels.pkl', 'baseline.pkl', 'prediction_keys.npz']
    wanted += [f'models/p{p}_q{q}/{name}' for p, q in ORDERS
               for name in ['parameters.csv.gz', 'predictions.npz']]
    checked = []
    INPUT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for name in wanted:
        record = records[PREFIX + name]
        path = SOURCE / name
        assert (path.stat().st_size, sha(path)) == (record['size'], record['sha256']), name
        # These source files belong to an older workspace: read-only; never clean them.
        checked.append({'name': name, 'local_path': str(path), 'size': record['size'],
                        'sha256': record['sha256'], 'versioned_inventory_match': True})
        if '/' not in name:
            target = INPUT / name
            if target.exists():
                assert sha(target) == record['sha256']
            else:
                shutil.copyfile(path, target)
    assert sha(OLD / 'calendar.csv') == records[PREFIX + 'calendar.csv']['sha256']
    shutil.copyfile(OLD / 'calendar.csv', INPUT / 'calendar.csv')
    receipt = {'source_commit': '6f6a57a515cd05f0676646479140d63e8ad07832',
               'source_tag': 'snapshot-2026-09-28-arima-learning', 'release_status': 'draft',
               'source_inventory_sha256': sha(INVENTORY), 'files': checked,
               'total_verified_bytes': sum(x['size'] for x in checked),
               'old_workspace_modified': False}
    save(EXP / 'source_local_verification.json', receipt)
    print(json.dumps({'local_files_verified': len(checked), 'bytes': receipt['total_verified_bytes']}))


def verify_pack():
    sys.path.insert(0, str(REPO / 'scripts'))
    import data_archive as archive
    source = json.loads((EXP / 'source_plan.json').read_text())
    record = source['reconstructed_packs'][3]
    pack = {'name': 'snapshot-2026-09-28-arima-learning-data-004-jpx_arima_learning_20260928.tar',
            'size': 739461120, 'sha256': '132d4d5269a23d681427054e40a9afa4deb289712793dec50197d534308dc7fb',
            'files': record['files']}
    path = TMP / pack['name']
    assert (path.stat().st_size, sha(path)) == (pack['size'], pack['sha256'])
    archive.validate_tar(path, pack)
    # Every TAR member validated against the Git-versioned per-file manifest.
    selected = [f for f in pack['files'] if f['path'].startswith(PREFIX)]
    receipt = {'asset_id': 594950920, 'release_id': 398077615,
               'source_tag': 'snapshot-2026-09-28-arima-learning', 'draft': True,
               'pack_size': pack['size'], 'pack_sha256': pack['sha256'],
               'all_tar_members_verified': len(pack['files']),
               'selected_grid_members': len(selected),
               'selected_grid_bytes': sum(f['size'] for f in selected),
               'tar_path': str(path), 'local_source_inventory_matches': True,
               'note': 'Read authenticated draft; this does not complete or publish the historical Release.'}
    save(EXP / 'source_remote_verification.json', receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['local', 'pack'])
    if parser.parse_args().mode == 'local': local_sources()
    else: verify_pack()
