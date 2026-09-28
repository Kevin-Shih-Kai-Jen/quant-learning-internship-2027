from pathlib import Path
import time,json,subprocess,sys,os,sqlite3
ROOT=Path(__file__).resolve().parent;RUN=ROOT/'arima_joint'
while True:
    p=json.loads((RUN/'progress.json').read_text())
    if p.get('mode')=='full' and p['completed_stock_year_orders']==16000:break
    time.sleep(5)
c=sqlite3.connect(RUN/'fits.sqlite',timeout=60);assert c.execute('select count(*) from fits').fetchone()[0]==16000;c.execute('pragma wal_checkpoint(TRUNCATE)');c.close()
env=os.environ.copy()
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:env[key]='1'
for script,log in [('evaluate_arima_joint.py','evaluation.log'),('report_arima_joint.py','report.log')]:
    print('START',script,flush=True)
    with (RUN/log).open('w') as out:r=subprocess.run([sys.executable,str(ROOT/script)],stdout=out,stderr=subprocess.STDOUT,env=env)
    if r.returncode:print('FAILED',script,log,flush=True);sys.exit(r.returncode)
    print('DONE',script,flush=True)
print('DELIVERY_READY',flush=True)
