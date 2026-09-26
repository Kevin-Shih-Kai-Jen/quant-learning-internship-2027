"""Task-specific ranged transport; retains original archive validation/install.
No token or signed URL is logged or persisted. Credentials only authenticate the
repository's asset request; the signed asset download needs no Authorization.
"""
from pathlib import Path
import argparse,concurrent.futures,hashlib,importlib.util,json,subprocess,tarfile,time,urllib.request,urllib.error
R=Path(__file__).resolve().parent;REPO=R.parents[1]
spec=importlib.util.spec_from_file_location('archive',REPO/'scripts/data_archive.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,req,fp,code,msg,headers,newurl):return None

def signed_url(pack):
 token=subprocess.check_output(['gh','auth','token'],text=True).strip()
 req=urllib.request.Request(f'https://api.github.com/repos/{a.REPOSITORY}/releases/assets/{pack["remote_id"]}',headers={'Accept':'application/octet-stream','Authorization':'Bearer '+token,'User-Agent':'quant-research-archive','X-GitHub-Api-Version':'2022-11-28'})
 try:
  with urllib.request.build_opener(NoRedirect).open(req,timeout=60) as response:raise RuntimeError('Asset endpoint did not redirect')
 except urllib.error.HTTPError as e:
  if e.code not in (301,302,303,307,308):raise RuntimeError('Asset request HTTP '+str(e.code)) from None
  return e.headers['Location']

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dest',type=Path,required=True);args=ap.parse_args();dest=args.dest.resolve();catalog=a.load_catalog('snapshot-2026-09-25-v31-stock-mse');selected=[]
 for prefix in ['project_archive_20260924/compressed_data/jpx_v8_soft_rank_20260912/inputs.pkl.gz','jpx_v31_stock_specific_mse_20260925/shared_v7/predictions.npz','jpx_v31_stock_specific_mse_20260925/stock_specific/predictions.npz']:selected+=a.select_files(catalog,prefix)
 logs=[]
 for pack,entries in selected:
  if all(a.local_matches(dest,e) for e in entries):continue
  url=signed_url(pack);parts=dest/'transport'/pack['name'];parts.mkdir(parents=True,exist_ok=True);chunk=2*1024*1024;blocks=[(i,start,min(pack['size']-1,start+chunk-1)) for i,start in enumerate(range(0,pack['size'],chunk))];starttime=time.monotonic()
  def fetch(block):
   i,start,end=block;path=parts/f'{i:05d}.part'
   if path.exists() and path.stat().st_size==end-start+1:return path
   for attempt in range(4):
    try:
     req=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}','User-Agent':'quant-research-archive'})
     with urllib.request.urlopen(req,timeout=120) as response:
      if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{pack["size"]}':raise RuntimeError('Range not honored')
      data=response.read(end-start+2)
     if len(data)!=end-start+1:raise RuntimeError('Wrong range size')
     path.write_bytes(data);return path
    except Exception as e:
     if attempt==3:raise RuntimeError('Range transfer failed '+str(i)+' '+type(e).__name__) from None
     time.sleep(1+attempt)
  print('RANGED_DOWNLOAD',pack['name'],pack['size'],len(blocks),flush=True)
  with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
   for count,_ in enumerate(pool.map(fetch,blocks),1):
    if count%16==0 or count==len(blocks):print('PROGRESS',count,len(blocks),round(time.monotonic()-starttime,1),flush=True)
  tar=parts.with_suffix('.complete.tar');h=hashlib.sha256()
  with tar.open('wb') as out:
   for i,_,_ in blocks:data=(parts/f'{i:05d}.part').read_bytes();out.write(data);h.update(data)
  assert tar.stat().st_size==pack['size'] and h.hexdigest()==pack['sha256'];a.validate_tar(tar,pack)
  wanted={e['path']:e for e in entries}
  with tarfile.open(tar,'r:') as archive:
   for member in archive:
    if member.name in wanted:
     with archive.extractfile(member) as source:a.install_member(dest,wanted[member.name],source)
  assert a.verify([(pack,entries)],dest)==0
  logs.append(dict(pack=pack['name'],sha256=h.hexdigest(),size=pack['size'],selected=[e['path'] for e in entries],seconds=time.monotonic()-starttime))
 (R/'retrieval_verification.json').write_text(json.dumps(dict(passed=True,source_tag='snapshot-2026-09-25-v31-stock-mse',transport='range requests; original full TAR/member SHA256 validation and safe install',packs=logs),indent=2))
if __name__=='__main__':main()
