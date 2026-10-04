"""Retrieve only the archived price member via byte ranges; never claim whole-pack verification."""
import hashlib
import json
import subprocess
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from common import EXP, TMP, INPUT, SOURCE, save, sha


def fetch_range(start,stop):
    asset='repos/Kevin-Shih-Kai-Jen/quant-learning-internship-2027/releases/assets/594950920'
    for attempt in range(3):
        try:
            result=subprocess.run(['gh','api',asset,'-H','Accept: application/octet-stream',
                                   '-H',f'Range: bytes={start}-{stop}'],capture_output=True,timeout=60,check=True)
            if len(result.stdout)!=stop-start+1:raise ValueError(f'Unexpected range length: {len(result.stdout)}')
            return result.stdout
        except (subprocess.TimeoutExpired,subprocess.CalledProcessError,ValueError):
            if attempt==2:raise
    raise AssertionError('unreachable')


def run():
    plan=json.loads((EXP/'source_plan.json').read_text())
    files=plan['reconstructed_packs'][3]['files']
    offsets=[];offset=0
    for record in files:
        info=tarfile.TarInfo(record['path']);info.size=record['size'];info.mode=record['mode'];info.mtime=record['mtime_ns']//1_000_000_000
        header=info.tobuf(format=tarfile.PAX_FORMAT)
        offsets.append((record,offset+len(header),header))
        offset+=len(header)+(record['size']+511)//512*512
    first,first_offset,header=offsets[0]
    with (SOURCE/'labels.pkl').open('rb') as stream:expected_prefix=header+stream.read(8192-len(header))
    assert (TMP/'remote_pack4_first8192.bin').read_bytes()==expected_prefix
    record,start,header=next(row for row in offsets if row[0]['path'].endswith('/arima_grid/prices.npz'))
    chunk_size=256*1024
    ranges=[(x,min(x+chunk_size,start+record['size'])-1) for x in range(start,start+record['size'],chunk_size)]
    parts={};begun=time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending={pool.submit(fetch_range,a,b):(a,b) for a,b in ranges}
        for count,future in enumerate(as_completed(pending),1):
            a,b=pending[future];parts[a]=future.result()
            if count%4==0 or count==len(ranges):print(json.dumps({'price_ranges':count,'total':len(ranges),'seconds':round(time.monotonic()-begun,1)}),flush=True)
    content=b''.join(parts[a] for a,b in ranges)
    assert len(content)==record['size'] and hashlib.sha256(content).hexdigest()==record['sha256']
    assert record['sha256']==sha(INPUT/'prices.npz')
    target=TMP/'remote_verified_prices.npz';target.write_bytes(content)
    partial=TMP/'snapshot-2026-09-28-arima-learning-data-004-jpx_arima_learning_20260928.tar'
    receipt={'source_tag':'snapshot-2026-09-28-arima-learning','release_id':398077615,'asset_id':594950920,'draft':True,
             'pack_size':739461120,'github_pack_digest':'sha256:132d4d5269a23d681427054e40a9afa4deb289712793dec50197d534308dc7fb',
             'whole_pack_downloaded_or_verified':False,'reason':'Full download was too slow; stopped and preserved partial receipt.',
             'partial_full_download_bytes':partial.stat().st_size if partial.exists() else 0,
             'partial_full_download_sha256':sha(partial) if partial.exists() else None,
             'remote_first_header_and_label_prefix_matches':True,
             'fully_retrieved_member':record['path'],'member_size':record['size'],'member_sha256':record['sha256'],
             'byte_offset':start,'ranges':len(ranges),'download_seconds':time.monotonic()-begun,
             'matches_local_price_input':True,'all_other_local_inputs_verified_against_versioned_inventory':True}
    save(EXP/'source_remote_verification.json',receipt)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':run()
