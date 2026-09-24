from pathlib import Path
import sys, json, pickle, hashlib, importlib.util
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
V17=BASE/'jpx_v17_actual_forecast_revisions_20260915'
V16=BASE/'jpx_v16_single_financial_v7_20260915'
sys.path.insert(0,str(V17))
spec=importlib.util.spec_from_file_location('v17_runner',V17/'run.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
FINS=['EPSActualQoQ','EPSActualGrowth']
JOBS={'qoq':[0],'yoy':[1],'joint':[0,1]}
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))
def load():
    with (BASE/'jpx_v8_soft_rank_20260912/inputs.pkl').open('rb') as h:f,x,groups,cal=pickle.load(h)
    z=pd.read_pickle(V17/'financial_signal_features.pkl')[FINS].to_numpy()
    g=np.load(V16/'price_only/predictions.npz')['score']
    return f,x,groups,cal,z,g

def main():
    f,x,groups,cal,z,g=load();g.setflags(write=False)
    eligible=np.isfinite(z).all(axis=1)&(np.abs(z)<=100).all(axis=1)
    target=f.Target.to_numpy();codes=f.SecuritiesCode.to_numpy()
    fallback=~np.isfinite(f[prior.INPUTS].to_numpy()).all(axis=1)
    yearmap=cal.set_index('Date').ValidationYear.to_dict()
    reference=pd.read_csv(V17/'eps_actual_joint/training_updates.csv').set_index('SignalDate')
    state={k:dict(beta=np.zeros(len(cols)),score=np.full(len(f),np.nan),rank=np.full(len(f),-1,np.int32),updates=[],history=[],daily=[]) for k,cols in JOBS.items()}
    pending=[];stepchecks=0;maxgraderr=0.
    for date,ix in groups.items():
        before={k:s['beta'].copy() for k,s in state.items()}
        due=[p for p in pending if p[1]<=date];pending=[p for p in pending if p[1]>date]
        assert len(due)<=1
        for signal,exitdate,ids in due:
            used=np.isfinite(target[ids])&eligible[ids];idx=ids[used]
            assert signal<exitdate<=date and len(idx)>0 and np.isfinite(g[idx]).all()
            h=np.column_stack([x[idx],z[idx]])
            lam=np.linalg.eigvalsh(h.T@h/len(idx))[-1];eta=1/(2*lam)
            ref=reference.loc[str(signal.date())]
            np.testing.assert_allclose(eta,ref.LearningRate,rtol=1e-10,atol=1e-14)
            assert len(idx)==ref.TrainingStocks
            for k,cols in JOBS.items():
                s=state[k];F=z[idx][:,cols];old=s['beta'].copy();e=g[idx]+F@old-target[idx]
                grad=2*F.T@e/len(idx);new=old-eta*grad
                gram=np.einsum('ni,nj->ij',F,F)/len(idx)
                rhs=np.einsum('ni,n->i',F,target[idx]-g[idx])/len(idx)
                independent=2*(gram@old-rhs)
                np.testing.assert_allclose(grad,independent,atol=1e-12,rtol=1e-10)
                maxgraderr=max(maxgraderr,float(np.max(np.abs(grad-independent))))
                loss=float(e@e/len(idx));eafter=g[idx]+F@new-target[idx];after=float(eafter@eafter/len(idx))
                assert after<=loss+1e-14 and np.isfinite(new).all()
                s['beta']=new
                s['updates'].append(dict(Date=str(date.date()),SignalDate=str(signal.date()),ExitDate=str(exitdate.date()),TrainingStocks=len(idx),ThresholdSkippedStocks=int((np.isfinite(target[ids])&~eligible[ids]).sum()),LearningRate=float(eta),MSEBeforeStep=loss,MSEAfterStep=after,**{'Gradient_'+FINS[c]:float(v) for c,v in zip(cols,grad)}))
                stepchecks+=1
        for k,cols in JOBS.items():
            s=state[k];s['history'].append(dict(Date=str(date.date()),UpdatesToday=len(due),**{'Before_'+FINS[c]:float(v) for c,v in zip(cols,before[k])},**{'After_'+FINS[c]:float(v) for c,v in zip(cols,s['beta'])}))
        if date==pd.Timestamp('2020-10-01') or date>cal.Date.max():continue
        assert np.isfinite(g[ix]).all()
        for k,cols in JOBS.items():
            s=state[k];score=g[ix]+z[ix][:,cols]@s['beta'];assert np.isfinite(score).all()
            order=np.lexsort((codes[ix],-score));rank=np.empty(len(ix),np.int32);rank[order]=np.arange(len(ix))
            s['score'][ix]=score;s['rank'][ix]=rank
            if date in yearmap:s['daily'].append(prior.daily_eval(date,ix,score,order,target,codes,fallback,yearmap[date]))
        exits=f.loc[ix,'ExitDate'];assert exits.nunique()==1 and exits.notna().all()
        pending.append((date,exits.iloc[0],ix))
    assert not pending
    for k,cols in JOBS.items():
        s=state[k];out=ROOT/k;out.mkdir(exist_ok=True)
        d=pd.DataFrame(s['daily']);u=pd.DataFrame(s['updates'])
        assert len(d)==953 and len(u)==1199 and np.isfinite(s['score']).sum()==2326022
        assert u.TrainingStocks.sum()==2323958 and u.ThresholdSkippedStocks.sum()==1848
        d.to_csv(out/'daily_metrics.csv',index=False);u.to_csv(out/'training_updates.csv',index=False)
        pd.DataFrame(s['history']).to_csv(out/'parameter_history.csv',index=False)
        np.savez_compressed(out/'predictions.npz',score=s['score'],rank=s['rank'])
        result=dict(variant=k,features=[FINS[c] for c in cols],sharpe=prior.sharp(d.OfficialDailySpread),mean_rank_ic=float(d.RankIC.mean()),return_mse=float(d.AllStockForecastMSE.mean()),training_stock_days=int(u.TrainingStocks.sum()),skipped_stock_days=int(u.ThresholdSkippedStocks.sum()),updates=len(u),days=len(d),final_coefficients={FINS[c]:float(v) for c,v in zip(cols,s['beta'])})
        save(out/'results.json',result);print(json.dumps(result),flush=True)
    np.testing.assert_array_equal(g,np.load(V16/'price_only/predictions.npz')['score'])
    save(ROOT/'update_audit.json',dict(passed=True,checked_updates=stepchecks,max_gradient_error=maxgraderr,frozen_g_source_unchanged=True,learning_rate_matches_v17_joint=True))
    sources=[V16/'price_only/predictions.npz',V17/'financial_signal_features.pkl',BASE/'jpx_v8_soft_rank_20260912/inputs.pkl',ROOT/'run.py',ROOT/'experiment_plan.md']
    save(ROOT/'manifest.json',dict(source_sha256={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}))
if __name__=='__main__':main()
