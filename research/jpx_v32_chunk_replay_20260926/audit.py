"""Independent residual-gradient/SVD replay and frozen-forecast checks."""
from pathlib import Path
import argparse,importlib.util,json,math,sys,time
import numpy as np
import pandas as pd
from scipy.stats import rankdata
R=Path(__file__).resolve().parent;spec=importlib.util.spec_from_file_location('v32',R/'run.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
sys.path.insert(0,str(r.BASE/'jpx_official_ranking_20260912'))
from official_metric import calc_spread_return_sharpe

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--task-dir',type=Path,required=True);args=ap.parse_args();f,x,groups,cal=r.load(args.task_dir);codes=np.unique(f.SecuritiesCode);code_idx=np.searchsorted(codes,f.SecuritiesCode);target=f.Target.to_numpy();valid=f.SignalDate.isin(cal.Date).to_numpy();folds=[];expected={n:np.full(len(f),np.nan) for n in ['shared_frozen','stock_single_frozen','stock_chunk_frozen']};expected_counts=np.zeros(len(f),np.int32)
 for year,calendar in cal.groupby('ValidationYear'):
  root=R/f'fold_{year}';cutoff=calendar.Date.min();tr=np.load(root/'training_trace.npz');par=np.load(root/'stage_parameters.npz');rows=tr['train_rows'];offset=tr['stock_offsets'];bs=tr['block_start'];be=tr['block_end'];bc=tr['block_stock'];st=tr['block_stage'];uo=tr['step_offsets'];rates=tr['nll_learning_rate'];after=tr['after'];refpars=par['coefficients'];counts=par['train_counts'];maxerr=0.;maxetaerr=0.;bi=0;ui=0;start=time.monotonic();diags=[]
  allowed=(f.SignalDate<cutoff)&(f.ExitDate<=cutoff)&f.Target.notna()&np.isfinite(target)&f.SignalDate.ne(pd.Timestamp('2020-10-01'))
  np.testing.assert_array_equal(np.sort(rows),np.flatnonzero(allowed.to_numpy()));assert len(np.unique(rows))==len(rows)
  for ci,code in enumerate(codes):
   own=rows[offset[ci]:offset[ci+1]];assert (f.loc[own,'SecuritiesCode']==code).all() and len(own)==counts[ci];assert np.all(np.diff(f.loc[own,'SignalDate'].to_numpy())>np.timedelta64(0,'D'))
   theta=np.zeros(12);n=len(own);yy=f.loc[own,'SignalDate'].dt.year.to_numpy();a=x[own];y=target[own]
   for stage in range(5):
    if stage==4:
     boundaries=np.r_[0,np.flatnonzero(yy[1:]!=yy[:-1])+1,n] if n else np.array([0])
     chunks=list(zip(boundaries[:-1],boundaries[1:]))
    else:
     size=(1,5,22,60)[stage];chunks=[(s,min(s+size,n)) for s in range(0,n,size)]
    expected_steps=[];first_ui=ui;phase_count=0
    for s,e in chunks:
     assert (bc[bi],st[bi],bs[bi],be[bi],uo[bi])==(ci,stage,offset[ci]+s,offset[ci]+e,ui)
     aa=a[s:e];yb=y[s:e];nn=e-s;k=int(np.floor(np.sqrt(nn)));assert k==math.isqrt(int(nn)) and uo[bi+1]-uo[bi]==k
     singular=np.linalg.norm(aa[0]) if nn==1 else np.linalg.svd(aa,compute_uv=False)[0];lr=nn/(singular*singular);maxetaerr=max(maxetaerr,abs(lr-rates[bi]));assert abs(lr-rates[bi])<=1e-11*abs(rates[bi])+1e-13
     loss0=np.mean((aa@theta-yb)**2)
     for _ in range(k):
      residual=aa@theta-yb;grad=aa.T@residual/nn;new=theta-lr*grad
      # The corresponding MSE step has twice the gradient and half the rate.
      equivalent=theta-(lr/2)*(2*grad);assert np.array_equal(new,equivalent)
      theta=new;expected_steps.append(theta.copy());ui+=1;phase_count+=1
     loss1=np.mean((aa@theta-yb)**2);assert loss1<=loss0+1e-9*max(1,float(loss0));bi+=1
    if expected_steps:
     ep=np.asarray(expected_steps);observed=after[first_ui:ui];maxerr=max(maxerr,float(np.max(np.abs(ep-observed))));np.testing.assert_allclose(ep,observed,rtol=2e-7,atol=2e-9)
    np.testing.assert_allclose(theta,refpars[stage,ci],rtol=2e-7,atol=2e-9)
    loss=float(np.mean((a@theta-y)**2)) if n else np.nan;diags.append([int(code),stage,n,phase_count,loss])
   if ci%500==0:print('AUDIT',year,ci,'seconds',round(time.monotonic()-start,1),flush=True)
  assert bi==len(bs) and ui==len(after);df=pd.read_csv(root/'training_stage_diagnostics.csv');d=np.array(diags);np.testing.assert_array_equal(df[['SecuritiesCode','TrainRows','Updates']].to_numpy(),d[:,[0,2,3]].astype(int));np.testing.assert_allclose(df.HistoryMSE.to_numpy(),d[:,4],rtol=1e-7,atol=1e-10,equal_nan=True)
  # Independently replay the shared daily model and verify every logged state.
  log=pd.read_csv(root/'shared_training_updates.csv');theta=np.zeros(12);shared_error=0.
  for q in log.itertuples():
   idx=np.flatnonzero((allowed&f.SignalDate.eq(pd.Timestamp(q.SignalDate))).to_numpy());a=x[idx];y=target[idx];assert len(idx)==q.Rows;sv=np.linalg.svd(a,compute_uv=False)[0];rate=len(idx)/(2*sv**2);np.testing.assert_allclose(rate,q.MSELearningRate,rtol=1e-10,atol=1e-12);theta-=rate*(2*a.T@(a@theta-y)/len(idx));state=np.array([getattr(q,'After_'+c) for c in r.NAMES]);shared_error=max(shared_error,float(np.max(np.abs(theta-state))));np.testing.assert_allclose(theta,state,rtol=2e-7,atol=1e-10)
  ids=np.flatnonzero(f.SignalDate.isin(calendar.Date).to_numpy());expected['shared_frozen'][ids]=x[ids]@theta
  for name,stage in [('stock_single_frozen',0),('stock_chunk_frozen',4)]:expected[name][ids]=np.sum(x[ids]*refpars[stage,code_idx[ids]],axis=1)
  expected_counts[ids]=counts[code_idx[ids]]
  folds.append(dict(validation_year=int(year),passed=True,train_rows=len(rows),blocks_verified=bi,all_updates_verified=ui,max_parameter_error=maxerr,max_learning_rate_error=maxetaerr,shared_max_parameter_error=shared_error,known_label_timing=True,stock_isolation=True,nll_mse_update_equivalence=True))
  print('FOLD_AUDIT',json.dumps(folds[-1]),flush=True);tr.close();par.close()
 models={}
 for name in r.VARIANTS:
  out=R/name;p=np.load(out/'predictions.npz');score=p['score'];rank=p['rank'];d=pd.read_csv(out/'daily_metrics.csv');result=json.loads((out/'results.json').read_text());assert np.isfinite(score[valid]).all() and np.isnan(score[~valid]).all()
  if name in expected:np.testing.assert_allclose(score,expected[name],rtol=1e-6,atol=5e-9,equal_nan=True);np.testing.assert_array_equal(p['training_history_counts'][valid],expected_counts[valid])
  else:
   old='shared_v7' if name=='shared_online' else 'stock_specific';original=np.load(args.task_dir/'research/jpx_v31_stock_specific_mse_20260925'/old/'predictions.npz');np.testing.assert_array_equal(score[valid],original['score'][valid]);np.testing.assert_array_equal(rank[valid],original['rank'][valid]);original.close()
  for q in d.itertuples():
   ids=groups[pd.Timestamp(q.Date)];s=score[ids];order=np.lexsort((f.loc[ids,'SecuritiesCode'].to_numpy(),-s));np.testing.assert_array_equal(rank[ids][order],np.arange(len(ids)));y=target[ids];known=np.isfinite(y);ar=rankdata(s[known]);br=rankdata(y[known]);ic=np.corrcoef(ar,br)[0,1] if np.std(ar)>0 and np.std(br)>0 else np.nan;np.testing.assert_allclose(ic,q.RankIC,atol=3e-14,rtol=1e-10,equal_nan=True);np.testing.assert_allclose(np.mean((s[known]-y[known])**2),q.MSE,atol=1e-11,rtol=1e-10)
   selected=np.r_[order[:200],order[-200:][::-1]];missing=int((~known[selected]).sum());assert missing==q.SelectedMissingTargets
   expected_spread=np.nan if missing else ((y[order[:200]]-y[order[-200:][::-1]])*r.W).sum()/r.W.mean();np.testing.assert_allclose(expected_spread,q.OfficialDailySpread,atol=5e-12,rtol=1e-10,equal_nan=True)
  scored=f.SignalDate.isin(pd.to_datetime(d.loc[d.OfficialDailySpread.notna(),'Date'])).to_numpy();official=calc_spread_return_sharpe(pd.DataFrame(dict(Date=f.loc[scored,'SignalDate'].to_numpy(),Target=target[scored],Rank=rank[scored])));np.testing.assert_allclose(official,result['sharpe'],atol=1e-13,rtol=0)
  models[name]=dict(passed=True,forecasts_checked=int(valid.sum()),official_sharpe=float(official),selected_missing_days=result['selected_missing_target_days']);r.save(out/'audit.json',models[name]);p.close()
 for rel,h in json.loads((R/'manifest.json').read_text())['source_sha256'].items():assert r.sha(r.BASE/rel)==h
 r.save(R/'audit.json',dict(passed=True,folds=folds,models=models,all_step_replay_verified=True,nll_mse_equivalence_verified=True,validation_parameters_frozen=True,known_label_timing_verified=True,no_extreme_filter=True,test_evaluated=False));print('ALL_AUDITS_PASSED',flush=True)
if __name__=='__main__':main()
