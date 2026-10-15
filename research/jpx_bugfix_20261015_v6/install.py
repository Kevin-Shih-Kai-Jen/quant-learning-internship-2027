"""Install only reviewed files after verifying all source/test backups."""
from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT / 'before_manifest.json').read_text())
for name, item in manifest.items():
    path = Path(name)
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
        raise RuntimeError(f'File changed since backup; refusing installation: {path}')

changes = []
for source, destination in [
    (ROOT / 'after', Path('/Users/coolguy/Desktop/JPX/codes')),
    (ROOT / 'tests_after', Path('/Users/coolguy/Desktop/JPX/debug/JPX_debug_v3')),
]:
    for candidate in sorted(source.glob('*.py')):
        target = destination / candidate.name
        data = candidate.read_bytes()
        if str(target) not in manifest and target.exists():
            raise RuntimeError(f'Untracked destination exists: {target}')
        if not target.exists() or target.read_bytes() != data:
            changes.append((candidate, target))

readme = Path('/Users/coolguy/Desktop/JPX/debug/JPX_debug_v3/README_fixes_v6.md')
if readme.exists():
    raise RuntimeError(f'Destination exists: {readme}')
changes.append((ROOT / 'README.md', readme))

for source, target in changes:
    data = source.read_bytes()
    temp = target.with_name(target.name + '.jpx_v6_install_tmp')
    with temp.open('xb') as stream:
        stream.write(data)
    if target.exists():
        os.chmod(temp, target.stat().st_mode)
    os.replace(temp, target)
    assert target.read_bytes() == data

records = [{'path': str(target), 'source': str(source.relative_to(ROOT)),
            'bytes': target.stat().st_size,
            'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
           for source, target in changes]
(ROOT / 'installed_verification.json').write_text(json.dumps(records, indent=2) + '\n')
print('Installed and verified', len(records), 'files')
