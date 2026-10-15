from pathlib import Path
import ast
import hashlib
import json

ROOT = Path(__file__).resolve().parent
CODES = Path('/Users/coolguy/Desktop/JPX/codes')


def main():
    before = json.loads((ROOT / 'before_manifest.json').read_text())
    for entry in before:
        path = CODES / entry['path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise RuntimeError(f'Source changed since backup: {path}')

    target = CODES / 'features.py'
    content = (ROOT / 'after/features.py').read_bytes()
    ast.parse(content)
    target.write_bytes(content)

    verified = []
    for entry in json.loads((ROOT / 'after_manifest.json').read_text()):
        path = CODES / entry['path']
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry['sha256']:
            raise RuntimeError(f'Installed content mismatch: {path}')
        verified.append({'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest})
    result = {'changed_files': [str(target)], 'verified_sources': verified}
    (ROOT / 'installation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'installed': str(target), 'verified_source_files': len(verified)}))


if __name__ == '__main__':
    main()
