"""Prepare a fresh run from a verified 2026-10-04 snapshot without the old draft Release.

Copy the code/plan into a NEW experiment directory under research/ before rerunning.
"""
import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--snapshot',type=Path,required=True,help='Restored research/jpx_arima_adf_annual_20261004 directory')
    parser.add_argument('--run-dir',type=Path,help='New empty temporary directory; default mkdtemp')
    args=parser.parse_args()
    exp=Path(__file__).resolve().parent
    snapshot=args.snapshot.resolve()
    runtime=exp/'runtime.json'
    if runtime.exists():raise ValueError('Existing runtime.json: use a NEW experiment/code directory; do not overwrite an active run.')
    run=args.run_dir.resolve() if args.run_dir else Path(tempfile.mkdtemp(prefix='jpx-arima-adf-replay-')).resolve()
    if args.run_dir:run.mkdir(parents=True,exist_ok=False)
    inputs=run/'inputs';inputs.mkdir()
    receipt=json.loads((exp/'source_local_verification.json').read_text())
    expected={r['name']:r for r in receipt['files']}
    checked=[]
    for name in ['prices.npz','labels.pkl','baseline.pkl','prediction_keys.npz']:
        path=snapshot/'shared_inputs'/name
        assert path.stat().st_size==expected[name]['size'] and sha(path)==expected[name]['sha256'],name
        shutil.copyfile(path,inputs/name);checked.append(name)
    shutil.copyfile(snapshot/'shared_inputs/calendar.csv',inputs/'calendar.csv')
    for p in range(1,6):
        for q in range(1,6):
            for name in ['predictions.npz','parameters.csv.gz']:
                key=f'models/p{p}_q{q}/{name}';path=snapshot/'fixed_d1'/f'p{p}_q{q}'/name
                assert path.stat().st_size==expected[key]['size'] and sha(path)==expected[key]['sha256'],key
                checked.append(key)
    config={'tmp':str(run),'repo':str(exp.parent.parent),'experiment':str(exp),
            'source_dir':str(snapshot/'shared_inputs'),'source_models_dir':str(snapshot/'fixed_d1')}
    runtime.write_text(json.dumps(config,indent=2))
    (exp/'replay_source_verification.json').write_text(json.dumps({'source_snapshot':str(snapshot),'checked_files':checked,'run_dir':str(run)},indent=2))
    print(json.dumps(config))


if __name__=='__main__':main()
