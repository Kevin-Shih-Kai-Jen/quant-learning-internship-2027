from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parent
CODES = Path('/Users/coolguy/Desktop/JPX/codes')
TESTS = CODES.parent / 'debug' / 'JPX_debug_v3'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


before = json.loads((ROOT / 'source_before.json').read_text())
for name, row in before.items():
    if digest(CODES / name) != row['sha256']:
        raise SystemExit(f'Source changed during preparation: {name}')

files = []
for name in ['features.py', 'JPX_deep_learning_models.py']:
    files.append((ROOT / 'after' / name, CODES / name))
for name in ['test_audit.py', 'test_regression_v1.py']:
    target = TESTS / name
    if target.read_bytes() != (ROOT / 'tests_before' / name).read_bytes():
        raise SystemExit(f'Test changed during preparation: {name}')
    files.append((ROOT / 'tests_after' / name, target))

for source, target in files:
    shutil.copy2(source, target)
    assert source.read_bytes() == target.read_bytes()

after = {p.name: {'bytes': p.stat().st_size, 'sha256': digest(p)}
         for p in sorted(CODES.glob('*.py'))}
for name in before:
    if name not in ['features.py', 'JPX_deep_learning_models.py']:
        assert before[name] == after[name], name

(ROOT / 'source_after.json').write_text(json.dumps(after, indent=2) + '\n')
(ROOT / 'installation.json').write_text(json.dumps({
    'changed_source_files': ['features.py', 'JPX_deep_learning_models.py'],
    'unchanged_source_files': [n for n in before if before[n] == after[n]],
    'updated_test_fixtures': ['test_audit.py', 'test_regression_v1.py'],
    'verified_files': [{'path': str(target), 'bytes': target.stat().st_size,
                        'sha256': digest(target)} for _, target in files],
}, indent=2) + '\n')
print('Installed 2 source edits and 2 matching test fixture updates; other source files unchanged.')
