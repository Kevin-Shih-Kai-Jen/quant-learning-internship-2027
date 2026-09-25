from pathlib import Path
import argparse,json,pickle,sys,hashlib,time
import numpy as np
import pandas as pd
from scipy.stats import rankdata
R=Path(__file__).resolve().parent;BASE=R.parent
sys.path.insert(0,str(BASE/'jpx_v7_daily_mse_20260912'))
from run_v7 import make_x,mse_step,FEATURES,INPUTS
NAMES=['alpha']+FEATURES;W=np.linspace(2.,1.,200)
def save(path,x):path.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def sha(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(task):
 path=task/'research/jpx_v8_soft_rank_20260912/inputs.pkl'
 with path.open('rb') as h:f,x,groups,cal=pickle.load(h)
 np.testing.assert_array_equal(x,make_x(f));assert np.isfinite(x).all();assert not f.duplicated(['SignalDate','SecuritiesCode']).any()
 return f,x,groups,cal

def evaluate(date,ids,score,order,y,year,fallback,counts):
 target=y[ids];known=np.isfinite(target);chosen=np.r_[order[:200],order[-200:][::-1]];missing=int((~known[chosen]).sum())
 spread=None if missing else float((target[order[:200]]@W-target[order[-200:][::-1]]@W)/W.mean())
 a=rankdata(score[known]);b=rankdata(target[known]);ic=float(np.corrcoef(a,b)[0,1]) if np.std(a)>0 and np.std(b)>0 else None
 return dict(Date=str(date.date()),ValidationYear=int(year),StocksRanked=len(ids),SelectedMissingTargets=missing,AllMissingTargets=int((~known).sum()),OfficialDailySpread=spread,IllustrativeGrossOneReturn=None if spread is None else spread/400,RankIC=ic,AllStockForecastMSE=float(np.mean((score[known]-target[known])**2)),MeanAbsoluteScore=float(np.mean(np.abs(score))),MaxAbsoluteScore=float(np.max(np.abs(score))),ZeroHistoryStocks=int((counts==0).sum()),Under12HistoryStocks=int((counts<12).sum()),SelectedUnder12HistoryStocks=int((counts[chosen]<12).sum()),FallbackStocks=int(fallback[ids].sum()))

def run(name,f,x,groups,cal):
 out=R/name;out.mkdir(exist_ok=True);individual=name=='stock_specific';codes=f.SecuritiesCode.to_numpy();unique=np.unique(codes);ci=np.searchsorted(unique,codes);n=len(f);theta=np.zeros((len(unique),12)) if individual else np.zeros(12)
 y=f.Target.to_numpy();score_all=np.full(n,np.nan);rank_all=np.full(n,-1,np.int32);forecast_counts=np.full(n,-1,np.int32);counts=np.zeros(len(unique),np.int32);usedmask=np.zeros(n,bool);release_pos=np.full(n,-1,np.int32)
 # Complete row-aligned state after each used training observation. No duplicate
 # before-state array: before is the previous saved state of the same stock.
 after_trace=np.full((n,12),np.nan) if individual else None
 eta_trace=np.full(n,np.nan) if individual else None;error_trace=np.full(n,np.nan) if individual else None
 fallback=~np.isfinite(f[INPUTS].to_numpy()).all(axis=1);ymap=cal.set_index('Date').ValidationYear.to_dict();end=cal.Date.max();pending=[];days=[];updates=[];shared_history=[];peak_after_loss=0.;max_projection_error=0.;start=time.monotonic()
 for di,(date,ids) in enumerate(groups.items()):
  due=[p for p in pending if p[1]<=date];pending=[p for p in pending if p[1]>date];assert len(due)<=1;before_global=theta.copy() if not individual else None
  for signal,exitdate,batch in due:
   known=np.isfinite(y[batch]);idx=batch[known];cc=ci[idx];a=x[idx];target=y[idx];assert len(np.unique(cc))==len(cc) and signal<exitdate<=date and not usedmask[idx].any()
   if individual:
    before=theta[cc].copy();err=np.einsum('ni,ni->n',a,before)-target;norm=np.einsum('ni,ni->n',a,a);eta=1/(2*norm);grad=2*err[:,None]*a;after=before-eta[:,None]*grad
    # Independent algebraic form; each stock sees only its own row.
    np.testing.assert_allclose(after,before-(err/norm)[:,None]*a,rtol=1e-11,atol=2e-13)
    err_after=np.einsum('ni,ni->n',a,after)-target;max_projection_error=max(max_projection_error,float(np.max(np.abs(err_after))));assert np.isfinite(after).all()
    np.testing.assert_allclose(err_after,0,atol=2e-9,rtol=0)
    theta[cc]=after;after_trace[idx]=after;eta_trace[idx]=eta;error_trace[idx]=err;lossbefore=float(np.mean(err**2));lossafter=float(np.mean(err_after**2))
    rates=dict(MinLearningRate=float(eta.min()),MedianLearningRate=float(np.median(eta)),MaxLearningRate=float(eta.max()))
   else:
    before=theta.copy();theta,grad,eta,lossbefore,lossafter,lam=mse_step(theta,a,target,np.ones(len(idx))/len(idx));rates=dict(MinLearningRate=eta,MedianLearningRate=eta,MaxLearningRate=eta,**{'Gradient_'+c:float(v) for c,v in zip(NAMES,grad)})
   assert lossafter<=lossbefore+1e-10*max(1,lossbefore);usedmask[idx]=True;release_pos[idx]=di;counts[cc]+=1;peak_after_loss=max(peak_after_loss,lossafter)
   updates.append(dict(Date=str(date.date()),SignalDate=str(signal.date()),ExitDate=str(exitdate.date()),TrainingStocks=len(idx),MissingTargetStocks=int((~known).sum()),MSEBeforeStep=lossbefore,MSEAfterStep=lossafter,**rates))
  if not individual:shared_history.append(dict(Date=str(date.date()),**{'Before_'+c:float(v) for c,v in zip(NAMES,before_global)},**{'After_'+c:float(v) for c,v in zip(NAMES,theta)}))
  if date==pd.Timestamp('2020-10-01') or date>end:continue
  if individual:score=np.einsum('ni,ni->n',x[ids],theta[ci[ids]])
  else:score=theta[0]+x[ids,1:]@theta[1:]
  assert np.isfinite(score).all();order=np.lexsort((codes[ids],-score));ranks=np.empty(len(ids),np.int32);ranks[order]=np.arange(len(ids));score_all[ids]=score;rank_all[ids]=ranks;forecast_counts[ids]=counts[ci[ids]]
  if date in ymap:days.append(evaluate(date,ids,score,order,y,ymap[date],fallback,counts[ci[ids]]))
  batch=ids[order];exits=f.loc[batch,'ExitDate'];assert exits.nunique()==1 and exits.notna().all();pending.append((date,exits.iloc[0],batch))
 assert not pending and len(updates)==1199 and len(days)==953 and usedmask.sum()==2325806 and np.isfinite(score_all).sum()==2326022
 pd.DataFrame(days).to_csv(out/'daily_metrics.csv',index=False);pd.DataFrame(updates).to_csv(out/'training_updates.csv',index=False)
 np.savez_compressed(out/'predictions.npz',score=score_all,rank=rank_all,own_known_updates=forecast_counts)
 if individual:
  idx=np.flatnonzero(usedmask);np.savez_compressed(out/'parameter_update_trace.npz',input_row=idx.astype(np.int32),release_day_index=release_pos[idx],after=after_trace[idx],learning_rate=eta_trace[idx],before_error=error_trace[idx])
  final=pd.DataFrame(theta,columns=NAMES);final.insert(0,'SecuritiesCode',unique);final.insert(1,'KnownUpdates',counts);final.to_csv(out/'final_stock_parameters.csv',index=False)
 else:pd.DataFrame(shared_history).to_csv(out/'parameter_history.csv',index=False);save(out/'final_parameters.json',dict(zip(NAMES,map(float,theta))))
 d=pd.DataFrame(days);result=dict(variant=name,sharpe=float(d.OfficialDailySpread.mean()/d.OfficialDailySpread.std(ddof=1)),mean_rank_ic=float(d.RankIC.mean()),validation_days=len(d),rank_ic_days=int(d.RankIC.notna().sum()),training_batch_dates=len(updates),training_stock_days=int(usedmask.sum()),parameter_vectors=len(unique) if individual else 1,parameters_per_vector=12,total_parameters=int(theta.size),stocks_in_input=len(unique),forecast_rows=int(np.isfinite(score_all).sum()),mse=float(d.AllStockForecastMSE.mean()),max_projection_error=max_projection_error,largest_post_training_mse=peak_after_loss,parameter_asof=str(list(groups)[-1].date()),test_evaluated=False,seconds=time.monotonic()-start)
 save(out/'results.json',result);print(json.dumps(result),flush=True)
 return result

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--task-dir',type=Path,required=True);args=ap.parse_args();task=args.task_dir.resolve();f,x,groups,cal=load(task)
 defaults=json.loads((R/'experiment_defaults_snapshot.json').read_text());assert not defaults['skip_training_for_extreme_values']
 for name in ['shared_v7','stock_specific']:run(name,f,x,groups,cal)
 with np.load(R/'shared_v7/predictions.npz') as a,np.load(task/'research/jpx_v23_g_thresholds_20260916/no_skip/predictions.npz') as b:
  np.testing.assert_allclose(a['score'],b['score'],rtol=1e-8,atol=3e-12);np.testing.assert_array_equal(a['rank'],b['rank'])
 sources=[BASE/'jpx_v7_daily_mse_20260912/run_v7.py',BASE/'jpx_v5_daily_returns_20260912/run_v5.py',BASE/'jpx_official_ranking_20260912/official_metric.py',R/'experiment_plan.md',R/'experiment_defaults_snapshot.json',R/'run.py']
 save(R/'manifest.json',dict(version='v31',source_snapshot='snapshot-2026-09-24',input_sha256=sha(task/'research/jpx_v8_soft_rank_20260912/inputs.pkl'),baseline_predictions_sha256=sha(task/'research/jpx_v23_g_thresholds_20260916/no_skip/predictions.npz'),source_sha256={str(p.relative_to(BASE)):sha(p) for p in sources},shared_v7_reproduced=True,features=NAMES,loss='MSE only; user cancelled MLE',no_extreme_filter=True,test_evaluated=False))
if __name__=='__main__':main()
