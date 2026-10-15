"""Install prepared routing changes only after checking for concurrent user edits."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((HERE / 'manifest.json').read_text())
    live = Path(manifest['source_directory'])
    before, after = manifest['before'], manifest['after']
    # Complete all preconditions before making the first live edit.
    assert {p.name for p in live.glob('*.py')} == set(before), 'Live file list changed'
    for name, metadata in before.items():
        assert sha(live / name) == metadata['sha256'], f'Concurrent edit: {name}'
    for name in manifest['changed_files']:
        assert sha(HERE / 'after' / name) == after[name]['sha256'], f'Staged file changed: {name}'
    for name in manifest['changed_files']:
        (live / name).write_bytes((HERE / 'after' / name).read_bytes())
    for name, metadata in after.items():
        assert sha(live / name) == metadata['sha256'], f'Installed hash differs: {name}'
    print(json.dumps({'installed': manifest['changed_files'], 'verified_files': len(after)}, indent=2))


if __name__ == '__main__':
    main()
