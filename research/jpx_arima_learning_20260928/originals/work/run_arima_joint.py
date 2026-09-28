"""Fixed-budget per-stock Target-MSE C/D trial, using matured training labels only."""
from pathlib import Path
import io,json,zlib,sqlite3,zipfile,os,time,warnings,argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from statsmodels.tsa.arima.model import ARIMA
from run_arima_grid import ROOT,RUN as SOURCE,YEARS,forecast_states,sha,save
RUN=ROOT/'arima_joint';PS=[3,5];DATA=None;CON=None
RAW=Path('/Users/coolguy/Desktop/JPX_data/modelling/JPX_data.zip')

def initialize():
    global DATA,CON
    with np.load(SOURCE/'prices.npz') as z:DATA={k:z[k] for k in z.files}
    CON=sqlite3.connect(f'file:{SOURCE}/fits.sqlite?mode=ro',uri=True)

def prepare():
    RUN.mkdir(exist_ok=True)
    prep=json.loads((SOURCE/'preparation.json').read_text())
    for name,h in prep['input_hashes'].items():assert sha(SOURCE/name)==h
    initialize();end=max(int(DATA['valid_positions'][DATA['years']==year][0]) for year in YEARS)
    member='JPX_data/raw/train_files/stock_prices.csv'
    with zipfile.ZipFile(RAW) as z:
        raw=pd.read_csv(z.open(member),usecols=['Date','SecuritiesCode','Target','SupervisionFlag'],parse_dates=['Date'])
    # Only matured training rows are cached. Validation labels are never sent to tasks.
    raw=raw.loc[raw.Date.lt(pd.Timestamp(DATA['dates'][end-2]))&~raw.SupervisionFlag].copy()
    wide=raw.pivot(index='Date',columns='SecuritiesCode',values='Target').reindex(index=pd.DatetimeIndex(DATA['dates'][:end-2]),columns=DATA['codes'])
    np.savez_compressed(RUN/'training_targets.npz',target=wide.to_numpy(),dates=DATA['dates'][:end-2],codes=DATA['codes'])
    manifest={'runner_sha256':sha(__file__),'plan_sha256':sha(ROOT/'arima_joint_plan.md'),'source_results_sha256':sha(SOURCE/'results.json'),'input_hashes':prep['input_hashes'],'training_targets_sha256':sha(RUN/'training_targets.npz'),'raw_member':member,'formal_test_used':False,'folds':[]}
    for year in YEARS:
        k=int(DATA['valid_positions'][DATA['years']==year][0]);manifest['folds'].append({'year':year,'price_train_end':str(DATA['dates'][k-1]),'last_allowed_target_origin':str(DATA['dates'][k-3]),'last_allowed_target_exit':str(DATA['dates'][k-1]),'first_validation_signal':str(DATA['dates'][k])})
    save(RUN/'manifest.json',manifest)

def acf2(result,y):
    e=np.asarray(result.resid).copy();e[~np.isfinite(y)]=np.nan;e[:int(result.loglikelihood_burn)]=np.nan
    trim=e.copy();valid=np.flatnonzero(np.isfinite(trim));trim[valid[:20]]=np.nan;good=np.isfinite(trim)
    n=int(good.sum());pairs=[int((good[h:]&good[:-h]).sum()) for h in [1,2]]
    if n<100 or min(pairs)<80:return e,np.zeros(2),False
    centered=np.where(good,trim-np.nanmean(trim),0.);den=float(centered@centered)
    if not np.isfinite(den) or den<=0:return e,np.zeros(2),False
    rho=np.array([centered[h:]@centered[:-h]/den for h in [1,2]])
    assert np.isfinite(rho).all() and (np.abs(rho)<=1+1e-12).all()
    return e,rho,True

def project(result,y,positions,params,p,anchor,scale,lam,rho=None):
    fc=forecast_states(result,positions)*scale+anchor
    e,r,available=acf2(result,y[:int(result.nobs)])
    if rho is None:rho=r
    u=np.where(np.isfinite(e[positions,None]),e[positions,None],0.)*rho*lam
    fc[:,0]+=scale*u[:,0];fc[:,1]+=scale*((1+params[0]+params[p])*u[:,0]+u[:,1])
    return fc,r,available

def evaluate_params(y,p,params,positions,anchor,scale,lam,rho):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');res=ARIMA(y,order=(p,1,1),trend='n').filter(params,cov_type='none')
    # Only the precomputed training rho is used for validation. project computes
    # residuals to read e[t], but does not replace rho with validation estimates.
    forecasts,_,_=project(res,y,positions,params,p,anchor,scale,lam,rho=rho)
    seen=np.isfinite(y);seen[:int(res.loglikelihood_burn)]=False;fv=res.filter_results.forecasts_error_cov[0,0,:]
    bad=np.maximum.accumulate(seen&(~np.isfinite(fv)|(fv<=0)))[positions]
    valid=np.isfinite(forecasts).all(axis=1)&(forecasts>0).all(axis=1)&~bad
    scores=np.zeros(len(positions));scores[valid]=forecasts[valid,1]/forecasts[valid,0]-1;valid &= np.isfinite(scores);scores[~valid]=0
    return scores,~valid,forecasts,res

def task(key,targets):
    p,yi,ci=key;oi=(p-1)*5;year=YEARS[yi];positions=DATA['valid_positions'][DATA['years']==year];cut=int(positions[0]);end=int(positions[-1])+1
    raw,payload=CON.execute('select audit,payload from fits where oi=? and yi=? and ci=?',(oi,yi,ci)).fetchone();old=json.loads(raw);base=np.load(io.BytesIO(payload));code=int(DATA['codes'][ci]);t0=time.monotonic()
    assert len(targets)==cut-2
    common={'p':p,'yi':yi,'code':code,'source_status':old['status'],'training_last_price':str(DATA['dates'][cut-1]),'max_target_origin':str(DATA['dates'][cut-3]),'max_target_exit':str(DATA['dates'][cut-1]),'first_validation_signal':str(DATA['dates'][cut]),'training_targets':0}
    assert common['max_target_exit']<common['first_validation_signal']
    if old['status']!='ok':return key,{**common,'C':{'status':'source_unavailable'},'D':{'status':'source_unavailable'},'seconds':time.monotonic()-t0}
    anchor=old['anchor'];scale=old['scale'];params0=base['params'];y=(DATA['prices'][:end,ci]-anchor)/scale;train=y[:cut]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');model=ARIMA(train,order=(p,1,1),trend='n');initial=model.filter(params0,cov_type='none')
    e,rho0,acfavailable=acf2(initial,train);validobs=np.flatnonzero(np.isfinite(e));originmask=np.isfinite(targets)&np.isfinite(train[:cut-2]);originmask[:int(validobs[20])+1 if len(validobs)>20 else cut-2]=False
    positions_train=np.flatnonzero(originmask);target=targets[positions_train];common['training_targets']=len(target)
    if len(target)<100:return key,{**common,'C':{'status':'insufficient_targets'},'D':{'status':'insufficient_targets'},'seconds':time.monotonic()-t0}
    assert np.max(positions_train)+2<cut
    v0=model.untransform_params(params0);sigma_unconstrained=v0[-1]
    def unpack(x):return model.transform_params(np.r_[x[:p+1],sigma_unconstrained])
    initialfc=forecast_states(initial,positions_train)*scale+anchor
    if not (np.isfinite(initialfc).all() and (initialfc>0).all()):return key,{**common,'C':{'status':'invalid_initial_training_forecasts'},'D':{'status':'invalid_initial_training_forecasts'},'seconds':time.monotonic()-t0}
    initial_mse=float(np.mean((initialfc[:,1]/initialfc[:,0]-1-target)**2));common['initial_mse']=initial_mse
    answer={**common}
    for variant in ['C','D']:
        joint=variant=='D';x0=np.r_[v0[:-1],np.zeros(2)] if joint else v0[:-1].copy();best={'loss':initial_mse*1e6,'x':x0.copy()};calls=0;invalid=0
        def objective(x):
            nonlocal calls,invalid
            calls+=1
            try:
                pars=unpack(x);lam=x[-2:] if joint else np.zeros(2)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore');f=model.filter(pars,cov_type='none')
                measured=np.isfinite(train);measured[:int(f.loglikelihood_burn)]=False;fv=f.filter_results.forecasts_error_cov[0,0,:]
                if not (np.isfinite(pars).all() and np.isfinite(fv[measured]).all() and (fv[measured]>0).all()):raise FloatingPointError()
                if joint:fc,_,_=project(f,train,positions_train,pars,p,anchor,scale,lam)
                else:fc=forecast_states(f,positions_train)*scale+anchor
                if not (np.isfinite(fc).all() and (fc>0).all()):raise FloatingPointError()
                pred=fc[:,1]/fc[:,0]-1;loss=float(np.mean((pred-target)**2)*1e6)
                if not np.isfinite(loss):raise FloatingPointError()
                if loss<best['loss']:best.update(loss=loss,x=np.array(x,copy=True))
                return loss
            except (ValueError,np.linalg.LinAlgError,FloatingPointError):invalid+=1;return 1e12
        np.testing.assert_allclose(objective(x0),initial_mse*1e6,rtol=1e-9,atol=1e-8)
        bounds=[(None,None)]*(p+1)+([(0,1),(0,1)] if joint else [])
        try:
            opt=minimize(objective,x0,method='L-BFGS-B',bounds=bounds,options={'maxiter':100,'maxfun':1500,'ftol':1e-9,'gtol':1e-5,'eps':1e-6,'maxls':20})
            converged=bool(opt.success);message=str(opt.message);iterations=int(opt.nit)
        except (ValueError,np.linalg.LinAlgError,FloatingPointError) as ex:converged=False;message=str(ex);iterations=-1
        pars=unpack(best['x']);lam=best['x'][-2:] if joint else np.zeros(2)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');fit=model.filter(pars,cov_type='none')
        _,rho,available=acf2(fit,train)
        assert best['loss']<=initial_mse*1e6+1e-9 and ((lam>=0)&(lam<=1)).all()
        assert np.abs(fit.arroots).min()>1-1e-8 and np.abs(fit.maroots).min()>1-1e-8
        checks=0;max_error=0.
        if code in [1301,1332,7203,9984]:
            score,fb,fc,full=evaluate_params(y,p,pars,positions,anchor,scale,lam,rho)
            original_check=forecast_states(ARIMA(y,order=(p,1,1),trend='n').filter(params0,cov_type='none'),positions)*scale+anchor
            np.testing.assert_allclose(original_check,base['forecast_prices'],rtol=2e-10,atol=2e-8)
            for j in [0,len(positions)//2,len(positions)-1]:
                t=int(positions[j]);prefix=y[:t+1]
                _,_,shortfc,short=evaluate_params(prefix,p,pars,np.array([t]),anchor,scale,lam,rho)
                np.testing.assert_allclose(shortfc[0],fc[j],rtol=2e-10,atol=2e-8)
                changed=y.copy();changed[t+1:]=9000+127*np.arange(len(y)-t-1)
                _,_,changedfc,changedfit=evaluate_params(changed,p,pars,np.array([t]),anchor,scale,lam,rho)
                np.testing.assert_allclose(changedfc[0],fc[j],rtol=2e-10,atol=2e-8)
                np.testing.assert_allclose(changedfit.resid[:t+1],full.resid[:t+1],rtol=1e-11,atol=1e-11,equal_nan=True)
                max_error=max(max_error,float(np.nanmax(np.abs(shortfc[0]-fc[j]))));checks+=1
        answer[variant]={'status':'optimized','converged':converged,'optimizer_message':message,'iterations':iterations,'function_calls':calls,'invalid_candidates':invalid,'initial_mse':initial_mse,'best_mse':best['loss']/1e6,'params':pars.tolist(),'lambda':lam.tolist(),'rho':rho.tolist(),'acf_available':available,'max_parameter_change':float(np.max(np.abs(pars-params0))),'min_ar_root':float(np.abs(fit.arroots).min()),'min_ma_root':float(np.abs(fit.maroots).min()),'causal_checks':checks,'max_causal_error':max_error}
    answer['seconds']=time.monotonic()-t0
    return key,answer

def run(workers=6,pilot=False):
    manifest=json.loads((RUN/'manifest.json').read_text());assert manifest['runner_sha256']==sha(__file__) and manifest['plan_sha256']==sha(ROOT/'arima_joint_plan.md')
    initialize()
    with np.load(RUN/'training_targets.npz') as z:target=z['target']
    con=sqlite3.connect(RUN/'fits.sqlite');con.execute('pragma journal_mode=WAL');con.execute('create table if not exists fits(p int,yi int,ci int,audit blob,primary key(p,yi,ci))');done=set(con.execute('select p,yi,ci from fits'))
    selected=[int(np.where(DATA['codes']==c)[0][0]) for c in [1301,1332,7203,9984]] if pilot else range(2000)
    todo=[(p,yi,ci) for ci in selected for yi in range(4) for p in PS if (p,yi,ci) not in done];it=iter(todo);count=len(done);started=time.monotonic();last=started
    def submit(pool,key):
        p,yi,ci=key;cut=int(DATA['valid_positions'][DATA['years']==YEARS[yi]][0]);return pool.submit(task,key,target[:cut-2,ci].copy())
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize) as pool:
        pending={submit(pool,next(it)) for _ in range(min(workers*3,len(todo)))}
        while pending:
            ready,pending=wait(pending,timeout=1,return_when=FIRST_COMPLETED)
            for f in ready:
                key,a=f.result();con.execute('insert into fits values(?,?,?,?)',(*key,zlib.compress(json.dumps(a,allow_nan=False).encode(),6)));count+=1
                if count%10==0:con.commit()
                try:pending.add(submit(pool,next(it)))
                except StopIteration:pass
            now=time.monotonic()
            if now-last>=10 or not pending:
                con.commit();progress={'completed_stock_year_orders':count,'total_stock_year_orders':len(done)+len(todo),'optimizer_fits':count*2,'seconds':round(now-started,1),'mode':'pilot' if pilot else 'full'};save(RUN/'progress.json',progress);print(json.dumps(progress),flush=True);last=now
    con.commit();con.execute('pragma wal_checkpoint(TRUNCATE)');con.close()
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','pilot','run']);a.add_argument('--workers',type=int,default=6);args=a.parse_args()
    if args.mode=='prepare':prepare()
    else:run(args.workers,args.mode=='pilot')
