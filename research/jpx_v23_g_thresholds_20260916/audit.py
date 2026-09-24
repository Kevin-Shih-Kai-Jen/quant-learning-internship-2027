from pathlib import Path
import importlib.util,sys,json,hashlib
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('v23_runner',R/'run.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
sys.path.insert(0,str(r.B/'jpx_official_ranking_20260912'))
from official_metric import calc_spread_return_sharpe
def main():
    f,x,z,groups,cal=r.load();full=x;y=f.Target.to_numpy();code=f.SecuritiesCode.to_numpy()
    reports={}
    for name in r.JOBS:
        out=R/name;names=r.NAMES;X=full
        threshold=r.THRESHOLDS[name];eligible=np.ones(len(f),bool) if threshold is None else (np.abs(z)<=threshold).all(axis=1);mask=eligible&np.isfinite(y)
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
                train=X[ix];step_design=full[ix]
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
            ids=groups[day];known=np.isfinite(y[ids]);scores=pred['score'][ids]
            daily=dd.loc[dd.Date.eq(str(day.date()))].iloc[0]
            if np.std(y[ids][known])==0:
                assert pd.isna(daily.RankIC)
            else:
                ic=pd.Series(scores[known]).rank(method='average').corr(pd.Series(y[ids][known]).rank(method='average'))
                np.testing.assert_allclose(ic,daily.RankIC,atol=2e-14)
            np.testing.assert_allclose(np.mean((scores[known]-y[ids][known])**2),daily.AllStockForecastMSE,rtol=1e-12)
            np.testing.assert_allclose(np.max(np.abs(scores)),daily.MaxAbsoluteScore,rtol=1e-12)
        official=float(calc_spread_return_sharpe(frame));result=json.loads((out/'results.json').read_text());np.testing.assert_allclose(official,result['sharpe'],atol=1e-14)
        reports[name]=dict(forecasts_checked=checked,updates_checked=len(updates),official_sharpe=official,max_prediction_error=max_prediction_error)
        print(json.dumps({name:reports[name]}),flush=True)
    with np.load(R/'skip100/predictions.npz') as now, np.load(r.B/'jpx_v21_all_eps_joint_20260916/price_mask/predictions.npz') as old:
        for key in ['score','rank']:np.testing.assert_array_equal(now[key],old[key])
    for filename in ['training_updates.csv','parameter_history.csv']:
        pd.testing.assert_frame_equal(pd.read_csv(R/'skip100'/filename),pd.read_csv(r.B/'jpx_v21_all_eps_joint_20260916/price_mask'/filename))
    with np.load(R/'no_skip/predictions.npz') as now, np.load(r.B/'jpx_v16_single_financial_v7_20260915/price_only/predictions.npz') as old:
        np.testing.assert_allclose(now['score'],old['score'],rtol=1e-10,atol=1e-12)
        np.testing.assert_array_equal(now['rank'],old['rank'])
    for name in r.JOBS:
        now=pd.read_csv(R/name/'training_updates.csv');old=pd.read_csv(r.B/'jpx_v22_eps_thresholds_20260916'/name/'training_updates.csv')
        for col in ['Date','SignalDate','ExitDate','TrainingStocks','ThresholdSkippedStocks']:
            pd.testing.assert_series_equal(now[col],old[col])
    for rel,digest in json.loads((R/'manifest.json').read_text())['source_sha256'].items():assert hashlib.sha256((r.B/rel).read_bytes()).hexdigest()==digest
    r.save(R/'audit.json',dict(passed=True,variants=reports,all_six_features_unchanged_from_source_caches=True,same_full_evaluation_universe=True,skip100_exactly_reproduces_v21_price_mask=True,no_skip_reproduces_v7=True,training_masks_match_v22=True,rank_ic_checked=True,source_hashes_unchanged=True))
    print(json.dumps(reports,indent=2))
if __name__=='__main__':main()
