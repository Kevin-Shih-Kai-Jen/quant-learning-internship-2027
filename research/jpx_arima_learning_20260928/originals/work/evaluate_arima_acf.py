"""Full-universe ranking comparison and residual diagnostics; never fits on Target."""
from pathlib import Path
import io,json,sqlite3
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests
from run_arima_acf import ROOT,RUN,SOURCE,PS,YEARS,sha,save
from evaluate_arima_grid import stats,calc_spread_return_sharpe

def daily_metrics(frame):
    frame=frame.sort_values(['Date','Score','SecuritiesCode'],ascending=[True,False,True]).copy()
    frame['Rank']=frame.groupby('Date').cumcount();daily=[];weights=np.linspace(2,1,200)
    for date,g in frame.groupby('Date',sort=True):
        assert g.SecuritiesCode.is_unique and np.array_equal(g.Rank,np.arange(len(g)))
        target=g.Target.to_numpy();known=np.isfinite(target);scores=g.Score.to_numpy()
        selected=np.r_[np.arange(200),np.arange(len(g)-1,len(g)-201,-1)]
        missing=int((~known[selected]).sum());r=rankdata(scores[known]);t=rankdata(target[known])
        ic=float(np.corrcoef(r,t)[0,1]) if np.ptp(r)>0 and np.ptp(t)>0 else np.nan
        spread=float((target[:200]@weights-target[-200:][::-1]@weights)/weights.mean()) if not missing else np.nan
        daily.append({'Date':date,'ValidationYear':int(g.ValidationYear.iloc[0]),'OfficialDailySpread':spread,'RankIC':ic,'SelectedMissingTargets':missing,'DistinctScores':g.Score.nunique(),'FallbackStocks':int(g.Fallback.sum()),'SelectedFallbackStocks':int(g.Fallback.to_numpy()[selected].sum())})
    d=pd.DataFrame(daily);valid=d.OfficialDailySpread.notna()
    official=float(calc_spread_return_sharpe(frame.loc[frame.Date.isin(d.loc[valid,'Date']),['Date','Rank','Target']]))
    assert abs(official-stats(d.loc[valid])['sharpe'])<1e-12
    return d,frame.sort_values(['Date','SecuritiesCode']).reset_index(drop=True)

def main():
    manifest=json.loads((RUN/'manifest.json').read_text());assert manifest['runner_sha256']==sha(ROOT/'run_arima_acf.py') and manifest['source_results_sha256']==sha(SOURCE/'results.json')
    con=sqlite3.connect(f'file:{RUN}/corrections.sqlite?mode=ro',uri=True);assert con.execute('select count(*) from corrections').fetchone()[0]==16000
    with np.load(SOURCE/'prices.npz') as z:data={k:z[k] for k in z.files}
    cal=pd.read_csv(SOURCE/'calendar.csv',parse_dates=['Date']);labels=pd.read_pickle(SOURCE/'labels.pkl').sort_values(['Date','SecuritiesCode']).reset_index(drop=True)
    with np.load(SOURCE/'prediction_keys.npz') as z:
        np.testing.assert_array_equal(z['date'],labels.Date.to_numpy(dtype='datetime64[D]'));np.testing.assert_array_equal(z['code'],labels.SecuritiesCode)
    dx=pd.Index(cal.Date).get_indexer(labels.Date);cx=pd.Index(data['codes']).get_indexer(labels.SecuritiesCode)
    assert dx.min()>=0 and cx.min()>=0
    baseline=pd.read_pickle(SOURCE/'baseline.pkl');assert len(baseline)==len(labels)
    b=labels.merge(baseline[['Date','SecuritiesCode','Rank']],on=['Date','SecuritiesCode'],validate='one_to_one')
    w=np.linspace(2,1,200);base_spread=[]
    for _,g in b.groupby('Date',sort=True):
        t=g.sort_values('Rank').Target.to_numpy();base_spread.append((t[:200]@w-t[-200:][::-1]@w)/w.mean())
    v7=pd.read_csv(SOURCE/'baseline_daily.csv',parse_dates=['Date']);np.testing.assert_allclose(base_spread,v7.OfficialDailySpread,rtol=1e-12,atol=1e-12)
    daily={};diags=[];residual_stats=[];overlaps={};audits=[]
    for p in PS:
        mat=np.zeros((len(cal),len(data['codes'])));fb=np.ones(mat.shape,bool);app=np.zeros(mat.shape,bool);nrecords=0
        for yi,ci,raw,payload in con.execute('select yi,ci,audit,payload from corrections where p=? order by yi,ci',(p,)):
            nrecords+=1;a=json.loads(raw);audits.append(a);z=np.load(io.BytesIO(payload));mask=cal.ValidationYear.eq(YEARS[yi]).to_numpy()
            assert a['train_end']<a['first_signal'];mat[mask,ci]=z['score'];fb[mask,ci]=z['fallback'];app[mask,ci]=z['applied']
            row={k:v for k,v in a.items() if k not in ['acf','pairs']}
            if a.get('acf') is not None:
                row.update({f'ACF{k}':v for k,v in enumerate(a['acf'],1)});row.update({f'Pairs{k}':v for k,v in enumerate(a['pairs'],1)})
            diags.append(row)
            for h in [1,2]:
                errors=z['original_error'][h:];pred=z['error_predictions'][:-h,h-1];valid=np.isfinite(errors)&z['applied'][:-h]
                residual_stats.append({'p':p,'Year':YEARS[yi],'h':h,'count':int(valid.sum()),'zero_sse':float((errors[valid]**2).sum()),'acf_sse':float(((errors[valid]-pred[valid])**2).sum())})
        assert nrecords==8000
        corrected=labels.copy();corrected['Score']=mat[dx,cx];corrected['Fallback']=fb[dx,cx];corrected['Applied']=app[dx,cx]
        d,corr=daily_metrics(corrected);key=f'ARIMA({p},1,1)+ACF';daily[key]=d;d.to_csv(RUN/f'p{p}_acf_daily.csv',index=False)
        basepred=np.load(SOURCE/'models'/f'p{p}_q1'/'predictions.npz');original=labels.copy();original['Score']=basepred['score'];original['Fallback']=basepred['fallback']
        db,base=daily_metrics(original);bk=f'ARIMA({p},1,1)';daily[bk]=db
        np.testing.assert_array_equal(base.Rank,basepred['rank'])
        reference=pd.read_csv(SOURCE/'models'/f'p{p}_q1'/'daily.csv')
        np.testing.assert_allclose(db.OfficialDailySpread,reference.OfficialDailySpread,rtol=1e-12,atol=1e-12,equal_nan=True)
        db.to_csv(RUN/f'p{p}_base_daily.csv',index=False)
        ranks=base.Rank.to_numpy();nr=corr.Rank.to_numpy();n=base.groupby('Date').Date.transform('size').to_numpy()
        ol=[]
        for date,indices in base.groupby('Date').indices.items():
            i=np.array(indices);ol.append({'Date':date,'Top200Overlap':float(((ranks[i]<200)&(nr[i]<200)).sum()/200),'Bottom200Overlap':float(((ranks[i]>=n[i]-200)&(nr[i]>=n[i]-200)).sum()/200),'RankCorrelation':float(np.corrcoef(ranks[i],nr[i])[0,1])})
        ol=pd.DataFrame(ol);ol.to_csv(RUN/f'p{p}_rank_changes.csv',index=False)
        overlaps[str(p)]={**ol.drop(columns='Date').mean().to_dict(),'ChangedRankFraction':float((ranks!=nr).mean()),'AppliedFraction':float(corr.Applied.mean()),'FallbackFraction':float(corr.Fallback.mean()),'SelectedFallbackStocks':int(d.SelectedFallbackStocks.sum())}
        np.savez_compressed(RUN/f'p{p}_predictions.npz',score=corr.Score.to_numpy(),rank=nr.astype(np.int16),fallback=corr.Fallback.to_numpy(),applied=corr.Applied.to_numpy())
        print('EVALUATED',p,flush=True)
    daily['v7']=v7;v7.to_csv(RUN/'v7_daily.csv',index=False)
    common=np.logical_and.reduce([d.OfficialDailySpread.notna().to_numpy() for d in daily.values()]);rows=[];annual=[]
    for name,d in daily.items():
        rows.append({'Model':name,**stats(d.loc[common]),'RawScoredDays':int(d.OfficialDailySpread.notna().sum())})
        for year in YEARS:annual.append({'Model':name,'Year':year,**stats(d.loc[common&cal.ValidationYear.eq(year).to_numpy()])})
    metrics=pd.DataFrame(rows);metrics.to_csv(RUN/'metrics.csv',index=False);pd.DataFrame(annual).to_csv(RUN/'annual.csv',index=False)
    dg=pd.DataFrame(diags)
    dg['LB10_BH_AdjustedP']=np.nan
    for _,g in dg.groupby(['p','year']):
        valid=g.lb10_pvalue.notna();ix=g.loc[valid].index
        if len(ix):dg.loc[ix,'LB10_BH_AdjustedP']=multipletests(g.loc[valid,'lb10_pvalue'],method='fdr_bh')[1]
    dg.to_csv(RUN/'residual_acf.csv.gz',index=False,compression='gzip')
    diagnostic=[]
    for (p,year),g in dg.groupby(['p','year']):
        good=g.loc[g.acf_available]
        diagnostic.append({'p':int(p),'Year':int(year),'StockYearRecords':len(g),'ACFAvailable':len(good),'MedianAbsACF1':float(good.ACF1.abs().median()),'MedianAbsACF2':float(good.ACF2.abs().median()),'LB10Available':int(g.lb10_pvalue.notna().sum()),'LB10RawPBelow005':int(g.lb10_pvalue.lt(.05).sum()),'LB10BHBelow005':int(g.LB10_BH_AdjustedP.lt(.05).sum()),'NoACFDays':int(g.no_acf_days.sum()),'MissingResidualDays':int(g.missing_residual_days.sum()),'NewInvalidForecastDays':int(g.corrected_invalid_days.sum())})
    pd.DataFrame(diagnostic).to_csv(RUN/'diagnostics_summary.csv',index=False)
    residual=pd.DataFrame(residual_stats).groupby(['p','Year','h'],as_index=False)[['count','zero_sse','acf_sse']].sum()
    residual['ZeroMSE']=residual.zero_sse/residual['count'];residual['ACFMSE']=residual.acf_sse/residual['count'];residual['MSEChange']=residual.ACFMSE-residual.ZeroMSE
    residual.to_csv(RUN/'future_residual_mse.csv',index=False)
    values=np.column_stack([daily[f'ARIMA({p},1,1){suffix}'].loc[common,'OfficialDailySpread'].to_numpy() for p in PS for suffix in ['', '+ACF']])
    observed=values.mean(axis=0)/values.std(axis=0,ddof=1);diff=observed[[1,3]]-observed[[0,2]];n=len(values);rng=np.random.default_rng(20260924);boot=[]
    for _ in range(2000):
        starts=rng.integers(0,n,size=int(np.ceil(n/20)));ix=((starts[:,None]+np.arange(20))%n).ravel()[:n];sample=values[ix];sh=sample.mean(axis=0)/sample.std(axis=0,ddof=1);boot.append(sh[[1,3]]-sh[[0,2]])
    boot=np.asarray(boot);lo,hi=np.quantile(boot,[.025,.975],axis=0);radius=float(np.quantile(np.max(np.abs(boot-diff),axis=1),.95))
    ci=[{'p':p,'SharpeChange':float(diff[i]),'Marginal95Low':float(lo[i]),'Marginal95High':float(hi[i]),'Simultaneous95Low':float(diff[i]-radius),'Simultaneous95High':float(diff[i]+radius)} for i,p in enumerate(PS)]
    pd.DataFrame(ci).to_csv(RUN/'comparisons.csv',index=False)
    result={'complete':True,'records':16000,'common_days':int(common.sum()),'excluded_dates':cal.loc[~common,'Date'].dt.strftime('%Y-%m-%d').tolist(),'metrics':rows,'annual':annual,'diagnostics':diagnostic,'overlaps':overlaps,'comparisons':ci,'causal_checks':sum(a['checks'] for a in audits),'independent_recursion_checks':sum(a['recursion_checks'] for a in audits),'baseline_forecast_checks':sum(a['source_status']=='ok' for a in audits),'all_official_metrics_verified':True,'formal_test_used':False,'annualized':False,'costs_included':False,'bootstrap':{'replicates':2000,'block_days':20,'seed':20260924,'simultaneous_radius':radius,'scope':'Two prespecified ACF versus base contrasts; does not correct earlier adaptive model selection'},'runner_sha256':sha(ROOT/'run_arima_acf.py'),'evaluator_sha256':sha(__file__),'plan_sha256':sha(ROOT/'arima_acf_plan.md')}
    save(RUN/'results.json',result);print(metrics.to_string(index=False),flush=True)
if __name__=='__main__':main()
