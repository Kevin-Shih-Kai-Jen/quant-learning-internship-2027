from pathlib import Path
import json, hashlib, shutil
root=Path(__file__).resolve().parent
record=json.loads((root/'installed_verification.json').read_text())
for name in ['README.md','REPORT.md']:
 target=Path(record['destination'])/name
 row=next(row for row in record['verified_files'] if row['path']==str(target))
 assert hashlib.sha256(target.read_bytes()).hexdigest()==row['sha256'],'Delivered doc changed outside this task'
 shutil.copy2(root/'JPX_debug_v3'/name,target)
 row['bytes']=target.stat().st_size
 row['sha256']=hashlib.sha256(target.read_bytes()).hexdigest()
(root/'installed_verification.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print('Refreshed the two delivered documents; tests and production code unchanged.')
