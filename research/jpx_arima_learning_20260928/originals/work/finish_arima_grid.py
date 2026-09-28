"""Wait for complete fitting checkpoints, then evaluate and package every model."""
from pathlib import Path
import os,time,json,subprocess,sys,sqlite3
ROOT=Path(__file__).resolve().parent;RUN=ROOT/'arima_grid'
last=0
while True:
    p=json.loads((RUN/'progress.json').read_text())
    if time.monotonic()-last>=60:
        print(json.dumps(p),flush=True);last=time.monotonic()
    if p.get('mode')=='full' and p['completed']==200000:
        c=sqlite3.connect(RUN/'fits.sqlite',timeout=60)
        assert c.execute('select count(*) from fits').fetchone()[0]==200000
        c.execute('PRAGMA wal_checkpoint(TRUNCATE)');c.close();break
    time.sleep(5)
env=os.environ.copy()
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:env[k]='1'
for script,log in [('evaluate_arima_grid.py','evaluation.log'),('report_arima_grid.py','report.log')]:
    print('START',script,flush=True)
    with (RUN/log).open('w') as output:
        result=subprocess.run([sys.executable,str(ROOT/script)],stdout=output,stderr=subprocess.STDOUT,env=env)
    if result.returncode:
        print('FAILED',script,log,flush=True);sys.exit(result.returncode)
    print('DONE',script,flush=True)
print('GRID_DELIVERY_READY',flush=True)
