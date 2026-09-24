from pathlib import Path
import importlib.util,sys,json,pickle,hashlib
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent;B=R.parent
V17=B/'jpx_v17_actual_forecast_revisions_20260915';V14=B/'jpx_v14_filtered_forecast_events_20260914'
sys.path.insert(0,str(V17))
spec=importlib.util.spec_from_file_location('prior_v17_runner',V17/'run.py');prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
FINS=['EPSActualQoQ','EPSActualGrowth','EPSExpectedGrowth','EPSForecastActualQoQ','EPSForecastActualYoY','EPSRevisionRelative']
PRICE=prior.NAMES;NAMES=PRICE+FINS
JOBS=['all_eps','price_mask','price_mask_step']
def save(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False))
def load():
    with (B/'jpx_v8_soft_rank_20260912/inputs.pkl').open('rb') as h:f,x,groups,cal=pickle.load(h)
    old=pd.read_pickle(V14/'financial_signal_features.pkl');new=pd.read_pickle(V17/'financial_signal_features.pkl')
    np.testing.assert_array_equal(old.EPSActualGrowth,new.EPSActualGrowth)
    z=np.column_stack([old[c].to_numpy() if c=='EPSExpectedGrowth' else new[c].to_numpy() for c in FINS])
    assert np.isfinite(z).all();np.testing.assert_array_equal(x,prior.make_x(f))
    return f,x,z,groups,cal
def main():
    f,x,z,groups,cal=load();full=np.column_stack([x,z]);eligible=(np.abs(z)<=100).all(axis=1)
    y=f.Target.to_numpy();code=f.SecuritiesCode.to_numpy();fallback=~np.isfinite(f[prior.INPUTS].to_numpy()).all(axis=1)
    yearmap=cal.set_index('Date').ValidationYear.to_dict();pending=[]
    state={n:dict(theta=np.zeros(18 if n=='all_eps' else 12),score=np.full(len(f),np.nan),rank=np.full(len(f),-1,np.int32),updates=[],history=[],daily=[]) for n in JOBS}
    max_grad=0.;max_step=0.
    for date,ids in groups.items():
        before={n:s['theta'].copy() for n,s in state.items()};due=[p for p in pending if p[1]<=date];pending=[p for p in pending if p[1]>date];assert len(due)<=1
        for sd,exitdate,ix in due:
            assert sd<exitdate<=date
            finite=np.isfinite(y[ix]);used=finite&eligible[ix];idx=ix[used];assert len(idx)>0
            xx=full[idx];px=x[idx];yy=y[idx]
            grams={'all':xx.T@xx/len(idx),'price':px.T@px/len(idx)}
            eig={k:float(np.linalg.eigvalsh(v)[-1]) for k,v in grams.items()};rates={k:1/(2*v) for k,v in eig.items()}
            for name,s in state.items():
                names=NAMES if name=='all_eps' else PRICE;X=xx if name=='all_eps' else px
                schedule='price' if name=='price_mask' else 'all';eta=rates[schedule]
                old=s['theta'];err=X@old-yy;grad=2*X.T@err/len(idx);new=old-eta*grad
                independent=2*(np.einsum('ni,nj->ij',X,X)/len(idx)@old-np.einsum('ni,n->i',X,yy)/len(idx))
                replay=old-eta*independent
                np.testing.assert_allclose(independent,grad,atol=3e-11,rtol=1e-9);np.testing.assert_allclose(replay,new,atol=3e-12,rtol=1e-9)
                lb=float(np.mean(err**2));la=float(np.mean((X@new-yy)**2));assert la<=lb+1e-13 and np.isfinite(new).all()
                max_grad=max(max_grad,float(np.max(np.abs(grad-independent))));max_step=max(max_step,float(np.max(np.abs(new-replay))))
                s['theta']=new;s['updates'].append(dict(Date=str(date.date()),SignalDate=str(sd.date()),ExitDate=str(exitdate.date()),KnownLabelStocks=int(finite.sum()),TrainingStocks=len(idx),ThresholdSkippedStocks=int((finite&~eligible[ix]).sum()),LearningRate=eta,LargestStepGramEigenvalue=eig[schedule],MSEBeforeStep=lb,MSEAfterStep=la,**{'Gradient_'+n:float(v) for n,v in zip(names,grad)}))
        for name,s in state.items():
            names=NAMES if name=='all_eps' else PRICE
            s['history'].append(dict(Date=str(date.date()),UpdatesToday=len(due),CumulativeUpdates=len(s['updates']),**{'Before_'+n:float(v) for n,v in zip(names,before[name])},**{'After_'+n:float(v) for n,v in zip(names,s['theta'])}))
        if date==pd.Timestamp('2020-10-01') or date>cal.Date.max():continue
        for name,s in state.items():
            X=full if name=='all_eps' else x;t=s['theta'];score=t[0]+X[ids,1:]@t[1:];assert np.isfinite(score).all()
            order=np.lexsort((code[ids],-score));rank=np.empty(len(ids),np.int32);rank[order]=np.arange(len(ids));s['score'][ids]=score;s['rank'][ids]=rank
            if date in yearmap:s['daily'].append(prior.daily_eval(date,ids,score,order,y,code,fallback,yearmap[date]))
        exits=f.loc[ids,'ExitDate'];assert exits.nunique()==1 and exits.notna().all();pending.append((date,exits.iloc[0],ids))
    assert not pending
    for name,s in state.items():
        out=R/name;out.mkdir(exist_ok=True);names=NAMES if name=='all_eps' else PRICE
        dd=pd.DataFrame(s['daily']);uu=pd.DataFrame(s['updates']);hh=pd.DataFrame(s['history'])
        assert len(dd)==953 and len(uu)==1199 and np.isfinite(s['score']).sum()==2326022
        dd.to_csv(out/'daily_metrics.csv',index=False);uu.to_csv(out/'training_updates.csv',index=False);hh.to_csv(out/'parameter_history.csv',index=False)
        np.savez_compressed(out/'predictions.npz',score=s['score'],rank=s['rank'])
        result=dict(variant=name,features=FINS if name=='all_eps' else [],parameter_count=len(names),sharpe=prior.sharp(dd.OfficialDailySpread),mean_rank_ic=float(dd.RankIC.mean()),mean_return_mse=float(dd.AllStockForecastMSE.mean()),mean_hard_rank_mse=float(dd.NormalizedHardRankMSE.mean()),training_stock_days=int(uu.TrainingStocks.sum()),threshold_skipped_stock_days=int(uu.ThresholdSkippedStocks.sum()),updates=len(uu),forecast_rows=int(np.isfinite(s['score']).sum()),days=len(dd),final_coefficients=dict(zip(names,map(float,s['theta']))),parameter_asof=hh.Date.iloc[-1])
        save(out/'results.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)
    save(R/'update_audit.json',dict(passed=True,gradient_checks=1199*3,max_gradient_error=max_grad,max_update_error=max_step,all_losses_nonincreasing_on_training_batch=True))
    sources=[V14/'financial_signal_features.pkl',V17/'financial_signal_features.pkl',B/'jpx_v8_soft_rank_20260912/inputs.pkl',R/'experiment_plan.md',R/'run.py']
    save(R/'manifest.json',dict(source_sha256={str(p.relative_to(B)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},eps_algorithm_unchanged=True,uses_external_filing_values=False))
if __name__=='__main__':main()
