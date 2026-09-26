"""Annual expanding per-stock replay with fixed Gaussian variance.
All training evidence is retained by fold. No validation-time adaptation.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, math, pickle, sys, time
import numpy as np
import pandas as pd
from scipy.stats import rankdata
R=Path(__file__).resolve().parent;BASE=R.parent
sys.path.insert(0,str(BASE/'jpx_v7_daily_mse_20260912'))
from run_v7 import make_x, mse_step, INPUTS, NAMES
STAGES=['single','chunk5','chunk22','chunk60','calendar_year']
VARIANTS=['shared_online','stock_online','shared_frozen','stock_single_frozen','stock_chunk_frozen']
C=.5*np.log(2*np.pi);W=np.linspace(2.,1.,200)
def save(p,o):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(task):
 p=task/'research/jpx_v8_soft_rank_20260912/inputs.pkl'
 assert sha(p)=='6b0ff60bfb8eab20ec7c6e087e746dd6e0ccc38b9633d1a11d98477d84857bc6'
 with p.open('rb') as h:f,x,groups,cal=pickle.load(h)
 np.testing.assert_array_equal(x,make_x(f));assert np.isfinite(x).all();assert not f.duplicated(['SignalDate','SecuritiesCode']).any()
 return f,x,groups,cal

def blocks(n,years,stage):
 if stage<4:
  size=[1,5,22,60][stage]
  return [(s,min(n,s+size)) for s in range(0,n,size)]
 starts=np.r_[0,np.flatnonzero(np.diff(years)!=0)+1,n]
 return list(zip(starts[:-1],starts[1:])) if n else []

def fit_fold(f,x,cutoff,year,codes):
 out=R/f'fold_{year}';out.mkdir(exist_ok=True);target=f.Target.to_numpy();dates=f.SignalDate.to_numpy();exits=f.ExitDate.to_numpy()
 mask=(dates<np.datetime64(cutoff))&(exits<=np.datetime64(cutoff))&np.isfinite(target)&(dates!=np.datetime64('2020-10-01'))
 own=f.groupby('SecuritiesCode').indices;rows=[own[c][mask[own[c]]] for c in codes];counts=np.array([len(q) for q in rows],np.int32);offsets=np.r_[0,np.cumsum(counts)]
 train=np.concatenate(rows).astype(np.int32);ys=f.SignalDate.dt.year.to_numpy();total_blocks=0;total_steps=0
 for q in rows:
  yy=ys[q]
  for stage in range(5):
   b=blocks(len(q),yy,stage);total_blocks+=len(b);total_steps+=sum(math.isqrt(int(e-s)) for s,e in b)
 after=np.empty((total_steps,12));eta=np.empty(total_blocks);bs=np.empty(total_blocks,np.int32);be=np.empty(total_blocks,np.int32);bc=np.empty(total_blocks,np.int16);stage_code=np.empty(total_blocks,np.uint8);step_offsets=np.empty(total_blocks+1,np.int64)
 stage_theta=np.zeros((5,len(codes),12));diag=[];bi=0;ui=0;started=time.monotonic();peak_increase=0.
 for ci,q in enumerate(rows):
  a=x[q];y=target[q];theta=np.zeros(12);n=len(q);yy=ys[q]
  assert n==0 or ((f.loc[q,'ExitDate']<=cutoff).all() and (f.loc[q,'SignalDate']<cutoff).all())
  for stage in range(5):
   nsteps=0
   for s,e in blocks(n,yy,stage):
    aa=a[s:e];yyb=y[s:e];count=e-s;k=math.isqrt(int(count));step_offsets[bi]=ui;bs[bi]=offsets[ci]+s;be[bi]=offsets[ci]+e;bc[bi]=ci;stage_code[bi]=stage
    if count==1:
     v=aa[0];largest=float(v@v);lr=1/largest;err=float(v@theta-yyb[0]);theta=theta-lr*err*v;after[ui]=theta;ui+=1
    else:
     gram=aa.T@aa/count;cross=aa.T@yyb/count;largest=float(np.linalg.eigvalsh(gram)[-1]);assert largest>0;lr=1/largest
     loss0=float(np.mean((aa@theta-yyb)**2))
     for _ in range(k):theta=theta-lr*(gram@theta-cross);after[ui]=theta;ui+=1
     loss1=float(np.mean((aa@theta-yyb)**2));peak_increase=max(peak_increase,loss1-loss0);assert loss1<=loss0+1e-10*max(1,loss0)
    eta[bi]=lr;bi+=1;nsteps+=k
   stage_theta[stage,ci]=theta
   loss=float(np.mean((a@theta-y)**2)) if n else None
   diag.append(dict(ValidationYear=year,SecuritiesCode=int(codes[ci]),Stage=STAGES[stage],TrainRows=n,Updates=nsteps,HistoryMSE=loss,HistoryNLL=None if loss is None else C+.5*loss,CoefficientNorm=float(np.linalg.norm(theta))))
  if ci%500==0:print('FIT',year,ci,len(codes),'seconds',round(time.monotonic()-started,1),flush=True)
 assert bi==total_blocks and ui==total_steps and np.isfinite(after).all();step_offsets[-1]=ui
 np.savez_compressed(out/'training_trace.npz',train_rows=train,stock_offsets=offsets,block_stock=bc,block_stage=stage_code,block_start=bs,block_end=be,step_offsets=step_offsets,nll_learning_rate=eta,after=after)
 np.savez_compressed(out/'stage_parameters.npz',codes=codes,coefficients=stage_theta,train_counts=counts)
 pd.DataFrame(diag).to_csv(out/'training_stage_diagnostics.csv',index=False)
 # Same-pool shared v7 daily updates, replayed only up to this fold's cutoff.
 shared=np.zeros(12);shared_log=[]
 for date in sorted(f.loc[mask,'SignalDate'].unique()):
  q=np.flatnonzero(mask&(dates==date));a=x[q];y=target[q];before=shared.copy();shared,grad,rate,l0,l1,largest=mse_step(shared,a,y,np.full(len(q),1/len(q)))
  shared_log.append(dict(SignalDate=str(pd.Timestamp(date).date()),Rows=len(q),MSELearningRate=rate,MSEBefore=l0,MSEAfter=l1,**{'After_'+c:float(v) for c,v in zip(NAMES,shared)}))
 pd.DataFrame(shared_log).to_csv(out/'shared_training_updates.csv',index=False);save(out/'shared_parameters.json',dict(zip(NAMES,map(float,shared))))
 info=dict(validation_year=year,fit_at_signal_close=str(cutoff.date()),train_rows=int(len(train)),unique_training_stocks=int((counts>0).sum()),zero_history_stocks=int((counts==0).sum()),min_signal_date=str(f.loc[train,'SignalDate'].min().date()),max_signal_date=str(f.loc[train,'SignalDate'].max().date()),max_label_exit=str(f.loc[train,'ExitDate'].max().date()),training_blocks=bi,training_updates=ui,stages=STAGES,peak_block_mse_increase=peak_increase,seconds=time.monotonic()-started)
 save(out/'fit.json',info);print('FOLD_DONE',json.dumps(info),flush=True)
 return stage_theta,shared,counts,info

def metrics(f,score,cal,counts):
 codes=f.SecuritiesCode.to_numpy();y=f.Target.to_numpy();ranks=np.full(len(f),-1,np.int32);out=[];groups=f.groupby('SignalDate').indices
 for row in cal.itertuples():
  ids=groups[row.Date];s=score[ids];assert np.isfinite(s).all();order=np.lexsort((codes[ids],-s));ranks[ids[order]]=np.arange(len(ids));yy=y[ids];known=np.isfinite(yy);sel=np.r_[order[:200],order[-200:][::-1]];missing=int((~known[sel]).sum())
  spread=None if missing else float((yy[order[:200]]@W-yy[order[-200:][::-1]]@W)/W.mean());aa=rankdata(s[known]);bb=rankdata(yy[known]);ic=float(np.corrcoef(aa,bb)[0,1]) if np.std(aa)>0 and np.std(bb)>0 else None
  out.append(dict(Date=str(row.Date.date()),ValidationYear=int(row.ValidationYear),StocksRanked=len(ids),SelectedMissingTargets=missing,AllMissingTargets=int((~known).sum()),OfficialDailySpread=spread,RankIC=ic,MSE=float(np.mean((s[known]-yy[known])**2)),MeanAbsoluteScore=float(np.mean(np.abs(s))),MaxAbsoluteScore=float(np.max(np.abs(s))),ZeroTrainingHistoryStocks=int((counts[ids]==0).sum())))
 return pd.DataFrame(out),ranks

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--task-dir',type=Path,required=True);args=ap.parse_args();f,x,groups,cal=load(args.task_dir);codes=np.unique(f.SecuritiesCode);ci=np.searchsorted(codes,f.SecuritiesCode);scores={n:np.full(len(f),np.nan) for n in VARIANTS};history={n:np.zeros(len(f),np.int32) for n in VARIANTS};inputs={};folds=[]
 for variant,old in [('shared_online','shared_v7'),('stock_online','stock_specific')]:
  p=args.task_dir/'research/jpx_v31_stock_specific_mse_20260925'/old/'predictions.npz';inputs[variant]=sha(p)
  with np.load(p) as a:scores[variant]=a['score'];history[variant]=a['own_known_updates']
 for year,c in cal.groupby('ValidationYear',sort=True):
  cutoff=c.Date.min();pars,shared,counts,info=fit_fold(f,x,cutoff,int(year),codes);folds.append(info);val=f.SignalDate.isin(c.Date).to_numpy();ids=np.flatnonzero(val)
  scores['stock_single_frozen'][ids]=np.einsum('ni,ni->n',x[ids],pars[0,ci[ids]]);scores['stock_chunk_frozen'][ids]=np.einsum('ni,ni->n',x[ids],pars[-1,ci[ids]]);scores['shared_frozen'][ids]=x[ids]@shared
  for n in ['stock_single_frozen','stock_chunk_frozen','shared_frozen']:history[n][ids]=counts[ci[ids]]
  first=np.flatnonzero((f.SignalDate==cutoff).to_numpy())
  np.testing.assert_allclose(scores['stock_single_frozen'][first],scores['stock_online'][first],rtol=5e-8,atol=3e-10)
  np.testing.assert_allclose(scores['shared_frozen'][first],scores['shared_online'][first],rtol=5e-8,atol=3e-10)
 result={};valid=f.SignalDate.isin(cal.Date).to_numpy()
 for name in VARIANTS:
  out=R/name;out.mkdir(exist_ok=True);scores[name][~valid]=np.nan;d,rank=metrics(f,scores[name],cal,history[name]);d.to_csv(out/'daily_metrics.csv',index=False);np.savez_compressed(out/'predictions.npz',score=scores[name],rank=rank,training_history_counts=history[name])
  good=d.OfficialDailySpread.notna();sp=d.loc[good,'OfficialDailySpread'];result[name]=dict(variant=name,sharpe=float(sp.mean()/sp.std(ddof=1)),rank_ic=float(d.RankIC.mean()),mse=float(d.MSE.mean()),scored_days=int(good.sum()),rank_ic_days=int(d.RankIC.notna().sum()),validation_rows=int(valid.sum()),selected_missing_target_days=int((d.SelectedMissingTargets>0).sum()),zero_training_history_stock_days=int(d.ZeroTrainingHistoryStocks.sum()),validation_parameter_policy='daily_online' if name.endswith('_online') else 'frozen_for_whole_year');save(out/'results.json',result[name]);print('RESULT',json.dumps(result[name]),flush=True)
 assert len(cal)==953
 source_files=[R/'run.py',R/'config.json',R/'experiment_plan.md',R/'experiment_defaults_snapshot.json',BASE/'jpx_v7_daily_mse_20260912/run_v7.py',BASE/'jpx_v5_daily_returns_20260912/run_v5.py',BASE/'jpx_official_ranking_20260912/official_metric.py']
 save(R/'manifest.json',dict(version='v32',source_snapshot='snapshot-2026-09-25-v31-stock-mse',input_sha256='6b0ff60bfb8eab20ec7c6e087e746dd6e0ccc38b9633d1a11d98477d84857bc6',reference_prediction_sha256=inputs,source_sha256={str(p.relative_to(BASE)):sha(p) for p in source_files},folds=folds,test_evaluated=False,no_extreme_filter=True))
 save(R/'raw_results.json',result)
if __name__=='__main__':main()
