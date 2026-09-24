from pathlib import Path
import importlib.util,sys,json,time,hashlib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
R=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('v26',R/'run.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
sys.path.insert(0,str(r.BASE/'jpx_official_ranking_20260912'))
from official_metric import calc_spread_return_sharpe
def main():
    f,x,z,groups,cal=r.joint_load();y=f.Target.to_numpy();code=f.SecuritiesCode.to_numpy();catalog=[dict(name='joint',reuse=False)];reports={}
    for item in catalog:
        name=item['name'];out=R/name
        while not (out/'results.json').exists():time.sleep(1)
        result=json.loads((out/'results.json').read_text());assert result['threshold_skipped_stock_days']==0 and result['training_stock_days']==2325806
        assert json.loads((out/'update_audit.json').read_text())['passed']
        names=r.NAMES+r.JOINT;X=np.column_stack([x,z[r.JOINT].to_numpy()]);hist=pd.read_csv(out/'parameter_history.csv').set_index('Date');u=pd.read_csv(out/'training_updates.csv').set_index('Date');dd=pd.read_csv(out/'daily_metrics.csv').set_index('Date')
        with np.load(out/'predictions.npz') as p:score=p['score'];rank=p['rank']
        previous=np.zeros(len(names));checked=0;maxerr=0.
        for date,ids in groups.items():
            day=str(date.date());h=hist.loc[day];before=h[['Before_'+n for n in names]].to_numpy(float);after=h[['After_'+n for n in names]].to_numpy(float)
            np.testing.assert_allclose(before,previous,atol=2e-15,rtol=1e-11)
            if day in u.index:
                q=u.loc[day];sd=pd.Timestamp(q.SignalDate);base=groups[sd];used=base[np.isfinite(y[base])]
                assert sd<pd.Timestamp(q.ExitDate)<=date and (f.loc[used,'ExitDate']<=date).all();assert len(used)==q.TrainingStocks==q.KnownLabelStocks and q.ThresholdSkippedStocks==0
                grad=q[['Gradient_'+n for n in names]].to_numpy(float);np.testing.assert_allclose(before-q.LearningRate*grad,after,atol=3e-12,rtol=1e-8)
            else:np.testing.assert_array_equal(before,after)
            previous=after
            if date==pd.Timestamp('2020-10-01') or date>cal.Date.max():assert np.isnan(score[ids]).all() and (rank[ids]==-1).all();continue
            replay=np.einsum('ni,i->n',X[ids],after);np.testing.assert_allclose(replay,score[ids],atol=3e-9,rtol=1e-9);maxerr=max(maxerr,float(np.max(np.abs(replay-score[ids]))))
            order=np.lexsort((code[ids],-score[ids]));np.testing.assert_array_equal(np.argsort(order),rank[ids]);checked+=len(ids)
            if day in dd.index:
                row=dd.loc[day];target=y[ids];known=np.isfinite(target);w=np.linspace(2.,1.,200);chosen=target[order]
                assert np.isfinite(chosen[:200]).all() and np.isfinite(chosen[-200:]).all()
                spread=float((chosen[:200]@w-chosen[-200:][::-1]@w)/w.mean());np.testing.assert_allclose(spread,row.OfficialDailySpread,atol=3e-12)
                if np.std(target[known])==0:assert pd.isna(row.RankIC)
                else:
                    ic=np.corrcoef(rankdata(score[ids][known]),rankdata(target[known]))[0,1];np.testing.assert_allclose(ic,row.RankIC,atol=2e-14)
                np.testing.assert_allclose(np.mean((score[ids][known]-target[known])**2),row.AllStockForecastMSE,atol=2e-12,rtol=1e-10)
        val=f.SignalDate.isin(cal.Date).to_numpy();frame=pd.DataFrame(dict(Date=f.loc[val,'SignalDate'].to_numpy(),Target=y[val],Rank=rank[val]));official=float(calc_spread_return_sharpe(frame));np.testing.assert_allclose(official,result['sharpe'],atol=1e-14)
        assert checked==2326022 and len(u)==1199 and len(dd)==953 and dd.RankIC.notna().sum()==952
        if item['reuse']:
            for filename in ['predictions.npz','parameter_history.csv','training_updates.csv','daily_metrics.csv']:assert hashlib.sha256((out/filename).read_bytes()).hexdigest()==hashlib.sha256((Path(item['source'])/filename).read_bytes()).hexdigest()
        reports[name]=dict(passed=True,forecasts_checked=checked,updates_checked=len(u),official_sharpe=official,max_forecast_error=maxerr,training_skipped=0,reused=item['reuse']);r.save(out/'audit.json',reports[name]);print(json.dumps({name:reports[name]}),flush=True)
    while not (R/'manifest.json').exists():time.sleep(1)
    for rel,h in json.loads((R/'manifest.json').read_text())['source_sha256'].items():assert hashlib.sha256((r.BASE/rel).read_bytes()).hexdigest()==h
    r.save(R/'audit.json',dict(passed=True,models=len(reports),variants=reports,all_training_rows_preserved=True,all_evaluation_pools_full=True,all_source_hashes_unchanged=True,training_gradient_and_step_checks_passed=True))
if __name__=='__main__':main()
