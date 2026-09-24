from pathlib import Path
import importlib.util,json,hashlib,sys
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent;B=R.parent;V23=B/'jpx_v23_g_thresholds_20260916'
spec=importlib.util.spec_from_file_location('v23',V23/'run.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
JOBS={'none_none':('no_skip',None),'100_100':('skip100',100.),'10_10':('skip10',10.),'none_100':('no_skip',100.),'none_10':('no_skip',10.)}
def save(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False))
def main():
    assert json.loads((V23/'audit.json').read_text())['passed']
    f,x,z,groups,cal=v.load();extreme=np.abs(z).max(axis=1);y=f.Target.to_numpy();codes=f.SecuritiesCode.to_numpy();fallback=~np.isfinite(f[v.prior.INPUTS].to_numpy()).all(axis=1)
    source={}
    for model in ['no_skip','skip100','skip10']:
        with np.load(V23/model/'predictions.npz') as p:source[model]={k:p[k] for k in ['score','rank']}
    reports={}
    for name,(model,threshold) in JOBS.items():
        out=R/name;out.mkdir(exist_ok=True);eligible=np.ones(len(f),bool) if threshold is None else extreme<=threshold
        ranking=np.full(len(f),-1,np.int32);keep=np.zeros(len(f),bool);daily=[]
        for row in cal.itertuples():
            date=row.Date;ids=groups[date];retained=ids[eligible[ids]];assert len(retained)>=400
            score=source[model]['score'][retained];assert np.isfinite(score).all()
            order=np.lexsort((codes[retained],-score));rank=np.empty(len(retained),np.int32);rank[order]=np.arange(len(retained));ranking[retained]=rank;keep[retained]=True
            d=v.prior.daily_eval(date,retained,score,order,y,codes,fallback,row.ValidationYear)
            oldorder=np.argsort(source[model]['rank'][ids]);oldlong=ids[oldorder[:200]];oldshort=ids[oldorder[-200:]]
            d.update(FullPoolStocks=len(ids),ValidationExcludedStocks=len(ids)-len(retained),ValidationExcludedPct=(len(ids)-len(retained))/len(ids)*100,OldLongExcluded=int((~eligible[oldlong]).sum()),OldShortExcluded=int((~eligible[oldshort]).sum()),NewLongEntries=len(set(retained[order[:200]])-set(oldlong)),NewShortEntries=len(set(retained[order[-200:]])-set(oldshort)))
            assert d['SelectedMissingTargets']==0
            daily.append(d)
        dd=pd.DataFrame(daily);dd.to_csv(out/'daily_metrics.csv',index=False)
        np.savez_compressed(out/'validation_selection.npz',eligible=keep,rank=ranking)
        train=json.loads((V23/model/'results.json').read_text())
        report=dict(variant=name,training_variant=model,validation_threshold=threshold,sharpe=v.prior.sharp(dd.OfficialDailySpread),rank_ic=float(dd.RankIC.mean()),rank_ic_days=int(dd.RankIC.notna().sum()),mean_mse=float(dd.AllStockForecastMSE.mean()),validation_days=len(dd),retained_stock_days=int(dd.StocksRanked.sum()),excluded_stock_days=int(dd.ValidationExcludedStocks.sum()),excluded_pct=float(dd.ValidationExcludedStocks.sum()/dd.FullPoolStocks.sum()*100),min_daily_stocks=int(dd.StocksRanked.min()),mean_daily_stocks=float(dd.StocksRanked.mean()),training_skipped_stock_days=train['threshold_skipped_stock_days'],old_long_excluded_total=int(dd.OldLongExcluded.sum()),old_short_excluded_total=int(dd.OldShortExcluded.sum()),source_predictions=str(V23/model/'predictions.npz'))
        save(out/'results.json',report);reports[name]=report;print(json.dumps(report,ensure_ascii=False),flush=True)
    paths=[V23/'audit.json',V23/'run.py',B/'jpx_v14_filtered_forecast_events_20260914/financial_signal_features.pkl',B/'jpx_v17_actual_forecast_revisions_20260915/financial_signal_features.pkl',B/'jpx_v8_soft_rank_20260912/inputs.pkl',R/'run.py',R/'experiment_plan.md']
    for model in ['no_skip','skip100','skip10']:paths += [V23/model/n for n in ['predictions.npz','parameter_history.csv','training_updates.csv','results.json']]
    save(R/'manifest.json',dict(source_sha256={str(p.relative_to(B)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},training_reused_unchanged=True,threshold_uses_only_signal_date_features=True))
    save(R/'run_results.json',reports)
if __name__=='__main__':main()
