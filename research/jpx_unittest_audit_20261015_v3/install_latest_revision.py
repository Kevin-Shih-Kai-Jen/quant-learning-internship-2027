from pathlib import Path
import json, hashlib, shutil
root=Path(__file__).resolve().parent
record=json.loads((root/'installed_verification.json').read_text())
dest=Path(record['destination'])
changed=['README.md','REPORT.md','test_normalizer_reuse.py']+[str(p.relative_to(root/'JPX_debug_v3')) for p in (root/'JPX_debug_v3/results_1124').glob('*')]
for name in changed:
 target=dest/name
 old=next((r for r in record['verified_files'] if r['path']==str(target)),None)
 if old:assert hashlib.sha256(target.read_bytes()).hexdigest()==old['sha256']
 else:assert not target.exists()
 target.parent.mkdir(parents=True,exist_ok=True)
 shutil.copy2(root/'JPX_debug_v3'/name,target)
 row={'path':str(target),'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
 if old:old.update(row)
 else:record['verified_files'].append(row)
record['latest_frozen_source']='latest_revision_1124/source_snapshot'
record['note']='Source was edited by the user during audit; latest report is fixed to 11:24 snapshot. No production files were written by installer.'
(root/'installed_verification.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print('Installed final regression additions and report without modifying codes.')
