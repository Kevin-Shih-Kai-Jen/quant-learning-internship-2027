from pathlib import Path
import argparse,importlib.util,json,sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata
R=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('v31',R/'run.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
sys.path.insert(0,str(r.BASE/'jpx_official_ranking_20260912'))
from official_metric import calc_spread_return_sharpe

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--task-dir',type=Path,required=True);args=ap.parse_args();f,x,groups,cal=r.load(args.task_dir);dates=list(groups);daypos=pd.Index(dates).get_indexer(f.SignalDate);target=f.Target.to_numpy();codes=f.SecuritiesCode.to_numpy();reports={}
 for name in ['shared_v7','stock_specific']:
  out=R/name;result=json.loads((out/'results.json').read_text());d=pd.read_csv(out/'daily_metrics.csv').set_index('Date');u=pd.read_csv(out/'training_updates.csv');p=np.load(out/'predictions.npz');score=p['score'];rank=p['rank'];counts=p['own_known_updates'];checked=0;maxerror=0.
  if name=='stock_specific':
   with np.load(out/'parameter_update_trace.npz') as tr:
    ids=tr['input_row'];release=tr['release_day_index'];after=tr['after'];eta=tr['learning_rate'];error=tr['before_error']
   lookup=np.full(len(f),-1,int);lookup[ids]=np.arange(len(ids));assert len(ids)==2325806 and len(np.unique(ids))==len(ids)
   for code,rows in f.groupby('SecuritiesCode').indices.items():
    ti=lookup[rows];ti=ti[ti>=0];ti=ti[np.argsort(release[ti],kind='stable')];rr=ids[ti];assert (np.diff(release[ti])>0).all();assert np.isfinite(target[rr]).all()
    assert (f.loc[rr,'ExitDate'].to_numpy()<=np.array(dates,dtype='datetime64[ns]')[release[ti]]).all()
    assert (daypos[rr]<release[ti]).all()
    states=np.zeros((len(ti)+1,12))
    for j,k in enumerate(ti):
     row=ids[k];norm=float(x[row]@x[row]);err=float(x[row]@states[j]-target[row]);expected=states[j]-x[row]*(err/norm)
     states[j+1]=expected
    norms=np.einsum('ni,ni->n',x[rr],x[rr]);errors=np.einsum('ni,ni->n',x[rr],states[:-1])-target[rr]
    np.testing.assert_allclose(error[ti],errors,rtol=2e-9,atol=3e-10);np.testing.assert_allclose(eta[ti],1/(2*norms),rtol=2e-12,atol=1e-14);np.testing.assert_allclose(after[ti],states[1:],rtol=3e-8,atol=3e-10)
    at=np.searchsorted(release[ti],daypos[rows],side='right');pred=np.einsum('ni,ni->n',x[rows],states[at]);valid=np.isfinite(score[rows]);np.testing.assert_allclose(pred[valid],score[rows][valid],rtol=3e-8,atol=3e-9);np.testing.assert_array_equal(at[valid],counts[rows][valid]);maxerror=max(maxerror,float(np.max(np.abs(pred[valid]-score[rows][valid]))) if valid.any() else 0)
   training_checked=len(ids)
  else:
   hist=pd.read_csv(out/'parameter_history.csv').set_index('Date');bydate=u.set_index('Date');previous=np.zeros(12)
   for date,rows in groups.items():
    day=str(date.date());h=hist.loc[day];before=h[['Before_'+n for n in r.NAMES]].to_numpy(float);after=h[['After_'+n for n in r.NAMES]].to_numpy(float);np.testing.assert_allclose(before,previous,rtol=1e-9,atol=2e-15)
    if day in bydate.index:
     q=bydate.loc[day];batch=groups[pd.Timestamp(q.SignalDate)];batch=batch[np.isfinite(target[batch])];assert len(batch)==q.TrainingStocks and (f.loc[batch,'ExitDate']<=date).all()
     a=x[batch];err=a@before-target[batch];grad=2*np.einsum('ni,n->i',a,err)/len(batch);eta=1/(2*(np.linalg.svd(a,compute_uv=False)[0]**2/len(batch)))
     np.testing.assert_allclose(eta,q.MedianLearningRate,rtol=3e-10,atol=1e-12);np.testing.assert_allclose(before-eta*grad,after,rtol=3e-8,atol=3e-11)
    else:np.testing.assert_array_equal(before,after)
    valid=np.isfinite(score[rows]);pred=x[rows]@after;np.testing.assert_allclose(pred[valid],score[rows][valid],rtol=2e-8,atol=3e-9);maxerror=max(maxerror,float(np.max(np.abs(pred[valid]-score[rows][valid]))) if valid.any() else 0);previous=after
   training_checked=int(u.TrainingStocks.sum())
  for date,rows in groups.items():
   day=str(date.date())
   if date==pd.Timestamp('2020-10-01') or date>cal.Date.max():assert np.isnan(score[rows]).all();continue
   order=np.lexsort((codes[rows],-score[rows]));np.testing.assert_array_equal(np.argsort(order),rank[rows]);checked+=len(rows)
   if day in d.index:
    q=d.loc[day];y=target[rows];known=np.isfinite(y);a=rankdata(score[rows][known]);b=rankdata(y[known]);ic=np.corrcoef(a,b)[0,1] if np.std(a)>0 and np.std(b)>0 else np.nan
    np.testing.assert_allclose(ic,q.RankIC,rtol=1e-8,atol=2e-14,equal_nan=True)
    selected=np.r_[order[:200],order[-200:][::-1]];assert int((~known[selected]).sum())==q.SelectedMissingTargets
    if known[selected].all():spread=(y[order[:200]]@r.W-y[order[-200:][::-1]]@r.W)/r.W.mean();np.testing.assert_allclose(spread,q.OfficialDailySpread,rtol=1e-10,atol=3e-12)
  valid=f.SignalDate.isin(pd.to_datetime(d.index[d.OfficialDailySpread.notna()])).to_numpy();frame=pd.DataFrame(dict(Date=f.loc[valid,'SignalDate'].to_numpy(),Target=target[valid],Rank=rank[valid]));official=float(calc_spread_return_sharpe(frame));np.testing.assert_allclose(official,result['sharpe'],rtol=0,atol=1e-13)
  assert checked==2326022 and training_checked==2325806 and len(u)==1199
  reports[name]=dict(passed=True,forecasts_checked=checked,training_stock_days_checked=training_checked,official_sharpe=official,max_independent_prediction_error=maxerror,selected_missing_target_days=int((d.SelectedMissingTargets>0).sum()),independent_per_stock_replay=name=='stock_specific');r.save(out/'audit.json',reports[name]);print(name,json.dumps(reports[name]),flush=True)
 for rel,h in json.loads((R/'manifest.json').read_text())['source_sha256'].items():assert r.sha(r.BASE/rel)==h
 r.save(R/'audit.json',dict(passed=True,models=reports,stock_parameter_and_gradient_isolation_verified=True,known_label_timing_verified=True,same_full_forecast_universe=True,no_extreme_filter=True,test_evaluated=False))
if __name__=='__main__':main()
