import collections, gzip, hashlib, json, os, re, shutil, tarfile, time
from pathlib import Path

SOURCE=Path('/Users/coolguy/Documents/Codex/2026-09-23/new-chat')
WORKSPACE=Path('/Users/coolguy/Documents/Codex/jpx-sync-staging-20260928/research')
ROOT=WORKSPACE/'jpx_arima_learning_20260928'
CHUNK=700*1024*1024
BLOCK=4*1024*1024
ROOT.mkdir(parents=True, exist_ok=False)

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(BLOCK), b''): h.update(b)
    return h.hexdigest()

files=[]; excluded=[]; jobgroups=collections.defaultdict(list)
secret=re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|sk-proj-[A-Za-z0-9_-]{30,}|AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)')
scanned=0
for directory, dirs, names in os.walk(SOURCE):
    for name in list(dirs):
        if name in {'__pycache__','mplcache','.git'}:
            excluded.append(str((Path(directory)/name).relative_to(SOURCE))+'/')
            dirs.remove(name)
    dirs.sort()
    for name in sorted(names):
        p=Path(directory)/name; rel=p.relative_to(SOURCE).as_posix()
        if name=='.DS_Store' or name.endswith('.pyc'):
            excluded.append(rel); continue
        if p.is_symlink() or not p.is_file(): raise ValueError('Nonregular source '+rel)
        if name.startswith('.env') or p.suffix in {'.pem','.key','.p12','.pfx'}: raise ValueError('Sensitive filename '+rel)
        before=p.stat(); digest=sha(p); after=p.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns): raise ValueError('Changed source '+rel)
        if p.suffix in {'.py','.json','.md','.log','.txt','.csv'}:
            scanned+=1
            with p.open('rb') as stream:
                tail=b''
                for b in iter(lambda:stream.read(BLOCK),b''):
                    if secret.search(tail+b): raise ValueError('Secret pattern requires review '+rel)
                    tail=b[-256:]
        record={'path':rel,'size':before.st_size,'sha256':digest,'mode':before.st_mode&0o777,'mtime_ns':before.st_mtime_ns}
        files.append(record)
        if '/jobs/' in rel:
            group=rel.split('/jobs/')[0]; jobgroups[group].append(record)
        elif before.st_size>CHUNK:
            parts=[]; base='packages/chunks/'+rel.replace('/','__')
            with p.open('rb') as f:
                n=0
                while True:
                    b=f.read(BLOCK)
                    if not b: break
                    n+=1; outrel=base+f'.part{n:03d}'; out=ROOT/outrel; out.parent.mkdir(parents=True,exist_ok=True)
                    h=hashlib.sha256(); size=0
                    with out.open('xb') as w:
                        while b:
                            w.write(b); h.update(b); size+=len(b)
                            if size==CHUNK: break
                            b=f.read(min(BLOCK,CHUNK-size))
                    parts.append({'path':outrel,'size':size,'sha256':h.hexdigest()})
            record['storage']={'kind':'concat','parts':parts}
        else:
            outrel='originals/'+rel; out=ROOT/outrel; out.parent.mkdir(parents=True,exist_ok=True)
            os.link(p,out)
            record['storage']={'kind':'direct','path':outrel}

for group, records in jobgroups.items():
    outrel='packages/'+group.replace('/','-')+'-jobs.tar.gz'; out=ROOT/outrel
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('xb') as raw, gzip.GzipFile(fileobj=raw,mode='wb',mtime=0,compresslevel=6) as gz, tarfile.open(fileobj=gz,mode='w|',format=tarfile.PAX_FORMAT) as tar:
        for r in records:
            p=SOURCE/r['path']; info=tarfile.TarInfo(r['path']); info.size=r['size']; info.mode=r['mode']; info.mtime=r['mtime_ns']//1000000000
            with p.open('rb') as f: tar.addfile(info,f)
            r['storage']={'kind':'tar-gzip','path':outrel,'member':r['path']}
    expected={r['path']:r for r in records}
    with tarfile.open(out,'r:gz') as tar:
        count=0
        for m in tar:
            f=tar.extractfile(m); h=hashlib.sha256()
            for b in iter(lambda:f.read(BLOCK),b''): h.update(b)
            assert m.size==expected[m.name]['size'] and h.hexdigest()==expected[m.name]['sha256']
            count+=1
        assert count==len(records)

manifest={'schema_version':1,'source':'JPX ARIMA learning workspace; archived 2026-09-28','files':files,'excluded':excluded}
with (ROOT/'SNAPSHOT_FILE_MANIFEST.json.gz').open('xb') as raw, gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as gz:
    gz.write((json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode())
summary={'schema_version':1,'tag':'snapshot-2026-09-28-arima-learning','original_files':len(files),'original_bytes':sum(r['size'] for r in files),'storage_counts':dict(collections.Counter(r['storage']['kind'] for r in files)),'excluded':excluded,'secret_pattern_scan_text_files':scanned,'secret_pattern_matches':0,'manifest_sha256':sha(ROOT/'SNAPSHOT_FILE_MANIFEST.json.gz'),'note':'Source bytes unchanged. Excluded only runtime caches. Split files rejoin byte-for-byte; job archives verified per member. This records archival integrity, not a new backtest.'}
(ROOT/'SNAPSHOT_SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
hist=WORKSPACE/'project_archive_20260924'; hist.mkdir(parents=True)
shutil.copy2('/Users/coolguy/Documents/Codex/jpx-sync-20260928/research/project_archive_20260924/COMPRESSED_DATA_MANIFEST.json',hist/'COMPRESSED_DATA_MANIFEST.json')
print(json.dumps(summary,ensure_ascii=False),flush=True)
