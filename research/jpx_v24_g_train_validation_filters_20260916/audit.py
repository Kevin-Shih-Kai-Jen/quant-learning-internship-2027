from pathlib import Path
import importlib.util,sys,json,hashlib
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('v24',R/'run.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
sys.path.insert(0,str(r.B/'jpx_official_ranking_20260912'))
from official_metric import calc_spread_return_sharpe
def main():
    f,x,z,groups,cal=r.v.load();y=f.Target.to_numpy();code=f.SecuritiesCode.to_numpy();sources={};hist={};reports={}
    for model in ['no_skip','skip100','skip10']:
        with np.load(r.V23/model/'predictions.npz') as p:sources[model]=p['score']
        hist[model]=pd.read_csv(r.V23/model/'parameter_history.csv').set_index('Date')
    for name,(model,threshold) in r.JOBS.items():
        out=R/name;dd=pd.read_csv(out/'daily_metrics.csv').set_index('Date');score=sources[model];frames=[];checked=0
        with np.load(out/'validation_selection.npz') as p:keep=p['eligible'];rank=p['rank']
        expected=f.SignalDate.isin(cal.Date).to_numpy().copy()
        if threshold is not None:expected &= (np.abs(z)<=threshold).all(axis=1)
        np.testing.assert_array_equal(keep,expected);assert (rank[~keep]==-1).all()
        for date in cal.Date:
            day=str(date.date());allids=groups[date];ids=allids[keep[allids]];assert len(ids)>=400
            theta=hist[model].loc[day,['After_'+n for n in r.v.NAMES]].to_numpy(float)
            np.testing.assert_allclose(np.einsum('ni,i->n',x[ids],theta),score[ids],atol=3e-10,rtol=1e-9)
            frame=pd.DataFrame({'Date':date,'SecuritiesCode':code[ids],'Target':y[ids],'Score':score[ids],'Rank':rank[ids]})
            sorted_frame=frame.sort_values(['Score','SecuritiesCode'],ascending=[False,True]);np.testing.assert_array_equal(sorted_frame.Rank,np.arange(len(ids)))
            ordered=sorted_frame.Target.to_numpy();w=np.linspace(2.,1.,200);assert np.isfinite(ordered[:200]).all() and np.isfinite(ordered[-200:]).all()
            spread=float(np.sum(ordered[:200]*w)/np.mean(w)-np.sum(ordered[-200:][::-1]*w)/np.mean(w));daily=dd.loc[day]
            np.testing.assert_allclose(spread,daily.OfficialDailySpread,atol=3e-12)
            known=frame.Target.notna();mse=float(((frame.loc[known,'Score']-frame.loc[known,'Target'])**2).mean());np.testing.assert_allclose(mse,daily.AllStockForecastMSE,rtol=1e-12)
            if frame.loc[known,'Target'].nunique()==1:assert pd.isna(daily.RankIC)
            else:
                ic=frame.loc[known,'Score'].rank().corr(frame.loc[known,'Target'].rank());np.testing.assert_allclose(ic,daily.RankIC,atol=2e-14)
            assert daily.StocksRanked==len(ids) and daily.ValidationExcludedStocks==len(allids)-len(ids)
            assert daily.NewLongEntries==daily.OldLongExcluded and daily.NewShortEntries==daily.OldShortExcluded
            assert set(sorted_frame.index[:200]).isdisjoint(set(sorted_frame.index[-200:]))
            frames.append(frame[['Date','Target','Rank']]);checked+=len(ids)
        official=float(calc_spread_return_sharpe(pd.concat(frames)));result=json.loads((out/'results.json').read_text());np.testing.assert_allclose(official,result['sharpe'],atol=1e-14)
        assert len(dd)==953 and dd.RankIC.notna().sum()==952 and dd.index[dd.RankIC.isna()].tolist()==['2020-09-29']
        reports[name]=dict(evaluated_stock_days=checked,days=953,rank_ic_days=952,sharpe=official,min_daily_pool=int(dd.StocksRanked.min()))
        print(json.dumps({name:reports[name]}),flush=True)
    now=pd.read_csv(R/'none_none/daily_metrics.csv');old=pd.read_csv(r.V23/'no_skip/daily_metrics.csv')
    for c in ['Date','StocksRanked','OfficialDailySpread','RankIC','AllStockForecastMSE']:pd.testing.assert_series_equal(now[c],old[c])
    for a,b in [('100_100','none_100'),('10_10','none_10')]:
        with np.load(R/a/'validation_selection.npz') as aa,np.load(R/b/'validation_selection.npz') as bb:np.testing.assert_array_equal(aa['eligible'],bb['eligible'])
    for rel,h in json.loads((R/'manifest.json').read_text())['source_sha256'].items():assert hashlib.sha256((r.B/rel).read_bytes()).hexdigest()==h
    r.save(R/'audit.json',dict(passed=True,variants=reports,predictions_match_point_in_time_parameters=True,no_skip_reproduces_v7=True,matched_evaluation_pools=True,all_days_at_least_400_stocks=True,source_hashes_unchanged=True,training_reused_from_audited_v23=True))
if __name__=='__main__':main()
