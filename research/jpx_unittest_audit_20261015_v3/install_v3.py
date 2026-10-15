"""Install only new debug/v3 files; refuse source drift or existing destination."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/Users/coolguy/Desktop/JPX/codes')
DEST = SOURCE.parent / 'debug' / 'JPX_debug_v3'


def manifest(directory):
    return {p.name: {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in sorted(directory.glob('*.py'))}


expected = json.loads((ROOT / 'source_manifest.json').read_text())
if manifest(SOURCE) != expected:
    raise SystemExit('Source changed since the audit; stop and inspect before installing.')
if DEST.exists():
    raise SystemExit(f'Destination already exists; refusing to replace it: {DEST}')
shutil.copytree(ROOT / 'JPX_debug_v3', DEST)
verified = []
for p in sorted((ROOT / 'JPX_debug_v3').rglob('*')):
    if p.is_file():
        target = DEST / p.relative_to(ROOT / 'JPX_debug_v3')
        assert target.read_bytes() == p.read_bytes(), str(target)
        verified.append({'path': str(target), 'bytes': target.stat().st_size,
                         'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
assert manifest(SOURCE) == expected
(ROOT / 'installed_verification.json').write_text(json.dumps({
    'destination': str(DEST), 'source_unchanged': True,
    'verified_files': verified, 'existing_debug_files_modified': [],
}, ensure_ascii=False, indent=2) + '\n')
print(f'Installed {len(verified)} new files; all 9 source files unchanged: {DEST}')
