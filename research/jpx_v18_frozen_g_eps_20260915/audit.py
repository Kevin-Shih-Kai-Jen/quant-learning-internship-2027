import importlib.util,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('v18_runner',ROOT/'run.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
import numpy as np
import pandas as pd
sys.path.insert(0,str(r.BASE/'jpx_official_ranking_20260912'))
from official_metric import calc_spread_return_sharpe

def main():
    f,x,groups,cal,z,g=r.load();target=f.Target.to_numpy();codes=f.SecuritiesCode.to_numpy()
    mask=(np.abs(z)<=100).all(axis=1)&np.isfinite(target)
    counts={};allframes=[]
    for name,cols in r.JOBS.items():
        out=ROOT/name;hist=pd.read_csv(out/'parameter_history.csv').set_index('Date');u=pd.read_csv(out/'training_updates.csv')
        pred=np.load(out/'predictions.npz');dd=pd.read_csv(out/'daily_metrics.csv');checked=0;maxerr=0.
        previous=np.zeros(len(cols))
        updates_by_date=u.set_index('Date')
        for date,ids in groups.items():
            day=str(date.date());row=hist.loc[day];before=row[['Before_'+r.FINS[c] for c in cols]].to_numpy(float);after=row[['After_'+r.FINS[c] for c in cols]].to_numpy(float)
            np.testing.assert_allclose(before,previous,atol=1e-16,rtol=1e-12)
            if day in updates_by_date.index:
                upd=updates_by_date.loc[day];signal=pd.Timestamp(upd.SignalDate);ix=groups[signal];ix=ix[mask[ix]]
                assert signal<pd.Timestamp(upd.ExitDate)<=date
                assert (f.loc[ix,'ExitDate']<=date).all()
                F=z[ix][:,cols];h=np.column_stack((x[ix],z[ix]));gram=h.T@h/len(ix)
                eta=1/(2*np.linalg.eigvalsh(gram)[-1])
                e=np.einsum('ni,i->n',F,before)+g[ix]-target[ix]
                grad=2*np.mean(F*e[:,None],axis=0);replay=before-eta*grad
                np.testing.assert_allclose(after,replay,atol=3e-15,rtol=1e-10)
                np.testing.assert_allclose(eta,upd.LearningRate,atol=1e-14,rtol=1e-10)
                assert len(ix)==upd.TrainingStocks
                np.testing.assert_allclose(np.mean(e**2),upd.MSEBeforeStep,atol=1e-14,rtol=1e-10)
            else:np.testing.assert_array_equal(before,after)
            previous=after
            if date==pd.Timestamp('2020-10-01') or date>cal.Date.max():
                assert np.isnan(pred['score'][ids]).all();continue
            score=g[ids]+np.einsum('ni,i->n',z[ids][:,cols],after)
            maxerr=max(maxerr,float(np.max(np.abs(score-pred['score'][ids]))))
            np.testing.assert_allclose(score,pred['score'][ids],atol=3e-13,rtol=1e-10)
            order=np.lexsort((codes[ids],-pred['score'][ids]));rank=np.argsort(order)
            np.testing.assert_array_equal(rank,pred['rank'][ids]);checked+=len(ids)
        validation=f.SignalDate.isin(cal.Date).to_numpy()
        frame=pd.DataFrame({'Date':f.loc[validation,'SignalDate'].to_numpy(),'Target':target[validation],'Rank':pred['rank'][validation]})
        for day,group in frame.groupby('Date'):
            ordered=group.sort_values('Rank');t=ordered.Target.to_numpy()
            assert np.isfinite(t[:200]).all() and np.isfinite(t[-200:]).all()
            spread=float(np.sum(t[:200]*np.linspace(2,1,200))-np.sum(t[-200:][::-1]*np.linspace(2,1,200)))/1.5
            actual=float(dd.loc[dd.Date.eq(str(day.date())),'OfficialDailySpread'].iloc[0])
            np.testing.assert_allclose(spread,actual,atol=1e-12)
        metric=float(calc_spread_return_sharpe(frame))
        np.testing.assert_allclose(metric,r.prior.sharp(dd.OfficialDailySpread),atol=1e-14)
        counts[name]=dict(forecasts_checked=checked,updates_checked=len(u),official_sharpe=metric,max_prediction_error=maxerr)
    frames=[pd.read_csv(ROOT/n/'training_updates.csv') for n in r.JOBS]
    for col in ['Date','SignalDate','TrainingStocks','ThresholdSkippedStocks','LearningRate']:
        for other in frames[1:]:pd.testing.assert_series_equal(frames[0][col],other[col])
    r.save(ROOT/'audit.json',dict(passed=True,variants=counts,shared_samples_and_learning_rates=True,frozen_g_point_in_time_source='v16/price_only/predictions.npz'))
    print(counts)
if __name__=='__main__':main()
