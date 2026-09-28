"""Apply training-only residual ACF to two frozen ARIMA grid models."""
from pathlib import Path
import os,io,json,sqlite3,warnings,time,argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from run_arima_grid import ROOT,RUN as SOURCE,YEARS,forecast_states,sha,save

RUN=ROOT/'arima_acf';PS=[3,5];DATA=None;CON=None

def initialize():
    global DATA,CON
    with np.load(SOURCE/'prices.npz') as z:DATA={k:z[k] for k in z.files}
    CON=sqlite3.connect(f'file:{SOURCE}/fits.sqlite?mode=ro',uri=True)

def residual_acf(error):
    x=np.array(error,copy=True,dtype=float);finite=np.flatnonzero(np.isfinite(x));x[finite[:20]]=np.nan
    good=np.isfinite(x);n=int(good.sum());pairs=[int((good[k:]&good[:-k]).sum()) for k in range(1,21)]
    out={'n':n,'pairs':pairs,'mean':None,'acf':None,'acf_available':False,'lb10_pvalue':None}
    if n<100:return out,x
    mu=float(np.nanmean(x));center=np.where(good,x-mu,0.);den=float(center@center)
    out['mean']=mu
    if not np.isfinite(den) or den<=0:return out,x
    rho=np.array([float(center[k:]@center[:-k]/den) for k in range(1,21)])
    assert np.isfinite(rho).all() and (np.abs(rho)<=1+1e-12).all()
    out.update(acf=rho.tolist(),acf_available=min(pairs[:2])>=80)
    return out,x

def corrected_prices(base,params,p,scale,u1,u2):
    result=np.array(base,copy=True)
    result[:,0]+=scale*u1
    result[:,1]+=scale*((1+params[0]+params[p])*u1+u2)
    return result

def task(key):
    p,yi,ci=key;oi=(p-1)*5
    a0,b=CON.execute('select audit,payload from fits where oi=? and yi=? and ci=?',(oi,yi,ci)).fetchone()
    old=json.loads(a0);z=np.load(io.BytesIO(b));base=z['forecast_prices'];score=z['score'].copy();fb=z['fallback'].copy();params=z['params']
    pos=DATA['valid_positions'][DATA['years']==YEARS[yi]];end=int(pos[-1])+1;trainend=int(pos[0]);code=int(DATA['codes'][ci])
    info={'p':p,'year':YEARS[yi],'code':code,'source_status':old['status'],'train_end':str(DATA['dates'][trainend-1]),'first_signal':str(DATA['dates'][trainend]),'acf_available':False,'checks':0,'recursion_checks':0,'no_acf_days':0,'missing_residual_days':0,'corrected_invalid_days':0,'applied_days':0}
    assert old['training_slots']==trainend and old['train_last']<old['first_signal']
    pred=base.copy();err=np.full(len(pos),np.nan);u=np.zeros((len(pos),2));applied=np.zeros(len(pos),bool)
    if old['status']=='ok':
        y=(DATA['prices'][:end,ci]-old['anchor'])/old['scale']
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            f=ARIMA(y,order=(p,1,1),trend='n').filter(params,cov_type='none')
        rebuilt=forecast_states(f,pos)*old['scale']+old['anchor']
        np.testing.assert_allclose(rebuilt,base,rtol=2e-10,atol=2e-8)
        resid=np.asarray(f.resid).copy();resid[~np.isfinite(y)]=np.nan;resid[:int(f.loglikelihood_burn)]=np.nan
        diag,trim=residual_acf(resid[:trainend]);info.update(diag)
        finite=np.flatnonzero(np.isfinite(trim))
        if len(finite)>100 and np.isfinite(trim[finite[0]:]).all():
            info['lb10_pvalue']=float(acorr_ljungbox(trim[finite[0]:],lags=[10],model_df=p+1,return_df=True).lb_pvalue.iloc[0])
        err=resid[pos]
        if diag['acf_available']:
            rho=np.array(diag['acf'][:2]);applied=np.isfinite(err)&~fb
            u[applied]=err[applied,None]*rho
            pred=corrected_prices(base,params,p,old['scale'],u[:,0],u[:,1])
            valid=np.isfinite(pred).all(axis=1)&(pred>0).all(axis=1)&~fb
            newscore=np.zeros(len(pos));newscore[valid]=pred[valid,1]/pred[valid,0]-1
            valid &= np.isfinite(newscore);newscore[~valid]=0
            info['corrected_invalid_days']=int((~valid&~fb).sum());score=newscore;fb=~valid
            info['missing_residual_days']=int((~np.isfinite(err)&~z['fallback']).sum())
            info['applied_days']=int(applied.sum())
        else:info['no_acf_days']=int((~fb).sum())
        # Baseline identity and preservation are checked for every stock-year.
        np.testing.assert_allclose(corrected_prices(base,params,p,old['scale'],np.zeros(len(pos)),np.zeros(len(pos))),base,rtol=0,atol=0,equal_nan=True)
        keep=~applied;np.testing.assert_allclose(score[keep],z['score'][keep],rtol=0,atol=0)
        if code in [1301,1332,7203,9984] and diag['acf'] is not None:
            np.testing.assert_allclose(diag['acf'],acf(trim,nlags=20,fft=False,adjusted=False,missing='conservative')[1:],rtol=1e-10,atol=1e-12)
            for j in [0,len(pos)//2,len(pos)-1]:
                t=int(pos[j])
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    prefix=ARIMA(y[:t+1],order=(p,1,1),trend='n').filter(params,cov_type='none')
                    altered=y.copy();altered[t+1:]=7000+137*np.arange(len(y)-t-1)
                    alteredfit=ARIMA(altered,order=(p,1,1),trend='n').filter(params,cov_type='none')
                np.testing.assert_allclose(prefix.resid[-1],resid[t],rtol=1e-11,atol=1e-11,equal_nan=True)
                np.testing.assert_allclose(alteredfit.filter_results.predicted_state[:,t+1],prefix.filter_results.predicted_state[:,-1],rtol=1e-11,atol=1e-11)
                tr=np.asarray(prefix.resid)[:trainend].copy();tr[~np.isfinite(y[:trainend])]=np.nan;tr[:int(prefix.loglikelihood_burn)]=np.nan
                prefixdiag,_=residual_acf(tr);np.testing.assert_allclose(prefixdiag['acf'],diag['acf'],atol=1e-12)
                h=np.asarray(prefix.forecast(2))*old['scale']+old['anchor']
                corrected=corrected_prices(h[None,:],params,p,old['scale'],u[j:j+1,0],u[j:j+1,1])[0]
                np.testing.assert_allclose(corrected,pred[j],rtol=2e-10,atol=2e-8)
                # Independent equation recursion. Kalman initialization can differ
                # very slightly from finite-lag equations; only check settled histories.
                dy=np.diff(y[:t+1]);ar=params[:p];theta=params[p]
                if applied[j] and np.isfinite(dy[-p:]).all() and np.isfinite(resid[t]):
                    d1=float(ar@dy[-p:][::-1]+theta*resid[t]);d2=float(ar@np.r_[d1,dy[-(p-1):][::-1]])
                    raw=np.array([y[t]+d1,y[t]+d1+d2])*old['scale']+old['anchor']
                    if np.allclose(raw,h,rtol=1e-7,atol=1e-6):
                        d1+=u[j,0];d2=float(ar@np.r_[d1,dy[-(p-1):][::-1]]+theta*u[j,0]+u[j,1])
                        recursive=np.array([y[t]+d1,y[t]+d1+d2])*old['scale']+old['anchor']
                        np.testing.assert_allclose(recursive,pred[j],rtol=1e-7,atol=1e-6);info['recursion_checks']+=1
                info['checks']+=1
    assert np.isfinite(score).all()
    buf=io.BytesIO();np.savez_compressed(buf,score=score,fallback=fb,forecast_prices=pred,original_error=err,error_predictions=u,applied=applied)
    return key,info,buf.getvalue()

def main():
    args=argparse.ArgumentParser();args.add_argument('--workers',type=int,default=6);a=args.parse_args()
    RUN.mkdir(exist_ok=True)
    manifest={'runner_sha256':sha(__file__),'plan_sha256':sha(ROOT/'arima_acf_plan.md'),'source_results_sha256':sha(SOURCE/'results.json'),'formal_test_used':False}
    if (RUN/'manifest.json').exists():assert json.loads((RUN/'manifest.json').read_text())==manifest
    else:save(RUN/'manifest.json',manifest)
    con=sqlite3.connect(RUN/'corrections.sqlite');con.execute('pragma journal_mode=WAL');con.execute('create table if not exists corrections(p int,yi int,ci int,audit text,payload blob,primary key(p,yi,ci))')
    done=set(con.execute('select p,yi,ci from corrections'));todo=[(p,yi,ci) for p in PS for yi in range(4) for ci in range(2000) if (p,yi,ci) not in done]
    it=iter(todo);start=time.monotonic();count=len(done);last=start
    with ProcessPoolExecutor(max_workers=a.workers,initializer=initialize) as pool:
        pending={pool.submit(task,next(it)) for _ in range(min(3*a.workers,len(todo)))}
        while pending:
            ready,pending=wait(pending,timeout=1,return_when=FIRST_COMPLETED)
            for future in ready:
                key,info,payload=future.result();con.execute('insert into corrections values(?,?,?,?,?)',(*key,json.dumps(info,allow_nan=False),payload));count+=1
                if count%25==0:con.commit()
                try:pending.add(pool.submit(task,next(it)))
                except StopIteration:pass
            now=time.monotonic()
            if now-last>10 or not pending:
                con.commit();save(RUN/'progress.json',{'completed':count,'total':16000,'seconds':round(now-start,1)});print(count,'/16000',flush=True);last=now
    con.commit();con.execute('pragma wal_checkpoint(TRUNCATE)');con.close()
if __name__=='__main__':main()
