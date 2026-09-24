from pathlib import Path
import importlib.util,sys,json,hashlib
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('v21_runner',R/'run.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
sys.path.insert(0,str(r.B/'jpx_official_ranking_20260912'))
from official_metric import calc_spread_return_sharpe
def main():
    f,x,z,groups,cal=r.load();full=np.column_stack([x,z]);y=f.Target.to_numpy();code=f.SecuritiesCode.to_numpy()
    eligible=(np.abs(z)<=100).all(axis=1);mask=eligible&np.isfinite(y);reports={}
    for name in r.JOBS:
        out=R/name;names=r.NAMES if name=='all_eps' else r.PRICE;X=full if name=='all_eps' else x
        hist=pd.read_csv(out/'parameter_history.csv').set_index('Date');updates=pd.read_csv(out/'training_updates.csv').set_index('Date');dd=pd.read_csv(out/'daily_metrics.csv')
        with np.load(out/'predictions.npz') as archive:
            pred={key:archive[key] for key in ['score','rank']}
        previous=np.zeros(len(names));max_prediction_error=0.;checked=0
        for date,ids in groups.items():
            day=str(date.date());h=hist.loc[day];before=h[['Before_'+n for n in names]].to_numpy(float);after=h[['After_'+n for n in names]].to_numpy(float)
            np.testing.assert_allclose(before,previous,atol=2e-16,rtol=1e-12)
            if day in updates.index:
                u=updates.loc[day];sd=pd.Timestamp(u.SignalDate);base=groups[sd];ix=base[mask[base]]
                assert sd<pd.Timestamp(u.ExitDate)<=date and (f.loc[ix,'ExitDate']<=date).all()
                assert len(ix)==u.TrainingStocks and int((np.isfinite(y[base])&~eligible[base]).sum())==u.ThresholdSkippedStocks
                train=X[ix];step_design=x[ix] if name=='price_mask' else full[ix]
                gram=np.einsum('ni,nj->ij',step_design,step_design)/len(ix);eta=1/(2*np.linalg.eigvalsh(gram)[-1])
                err=np.einsum('ni,i->n',train,before)-y[ix];grad=2*np.mean(train*err[:,None],axis=0)
                np.testing.assert_allclose(eta,u.LearningRate,atol=2e-13,rtol=1e-10)
                np.testing.assert_allclose(grad,u[['Gradient_'+n for n in names]].to_numpy(float),atol=3e-11,rtol=1e-8)
                np.testing.assert_allclose(before-eta*grad,after,atol=3e-13,rtol=1e-8)
                np.testing.assert_allclose(np.mean(err**2),u.MSEBeforeStep,atol=3e-13,rtol=1e-10)
            else:np.testing.assert_array_equal(before,after)
            previous=after
            if date==pd.Timestamp('2020-10-01') or date>cal.Date.max():
                assert np.isnan(pred['score'][ids]).all() and (pred['rank'][ids]==-1).all();continue
            score=np.einsum('ni,i->n',X[ids],after)
            np.testing.assert_allclose(score,pred['score'][ids],atol=3e-10,rtol=1e-9)
            max_prediction_error=max(max_prediction_error,float(np.max(np.abs(score-pred['score'][ids]))))
            order=np.lexsort((code[ids],-pred['score'][ids]));np.testing.assert_array_equal(np.argsort(order),pred['rank'][ids]);checked+=len(ids)
        val=f.SignalDate.isin(cal.Date).to_numpy();frame=pd.DataFrame(dict(Date=f.loc[val,'SignalDate'].to_numpy(),Target=y[val],Rank=pred['rank'][val]))
        for day,g in frame.groupby('Date'):
            target=g.sort_values('Rank').Target.to_numpy();assert np.isfinite(target[:200]).all() and np.isfinite(target[-200:]).all()
            w=np.linspace(2,1,200);spread=(target[:200]@w-target[-200:][::-1]@w)/w.mean()
            np.testing.assert_allclose(spread,dd.loc[dd.Date.eq(str(day.date())),'OfficialDailySpread'].iloc[0],atol=3e-12)
        official=float(calc_spread_return_sharpe(frame));result=json.loads((out/'results.json').read_text());np.testing.assert_allclose(official,result['sharpe'],atol=1e-14)
        reports[name]=dict(forecasts_checked=checked,updates_checked=len(updates),official_sharpe=official,max_prediction_error=max_prediction_error)
        print(json.dumps({name:reports[name]}),flush=True)
    tables={n:pd.read_csv(R/n/'training_updates.csv') for n in r.JOBS}
    for col in ['Date','SignalDate','ExitDate','TrainingStocks','ThresholdSkippedStocks']:
        for n in r.JOBS[1:]:pd.testing.assert_series_equal(tables['all_eps'][col],tables[n][col])
    pd.testing.assert_series_equal(tables['all_eps'].LearningRate,tables['price_mask_step'].LearningRate)
    for rel,digest in json.loads((R/'manifest.json').read_text())['source_sha256'].items():assert hashlib.sha256((r.B/rel).read_bytes()).hexdigest()==digest
    r.save(R/'audit.json',dict(passed=True,variants=reports,all_six_features_unchanged_from_source_caches=True,shared_training_mask=True,main_and_step_control_rates_identical=True,source_hashes_unchanged=True))
    print(json.dumps(reports,indent=2))
if __name__=='__main__':main()
