from pathlib import Path
import json,pickle
import numpy as np
import pandas as pd
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parent;V17=ROOT.parent;BASE=V17.parent;V14=BASE/'jpx_v14_filtered_forecast_events_20260914'
with (BASE/'jpx_v8_soft_rank_20260912/inputs.pkl').open('rb') as h:f,x,groups,calendar=pickle.load(h)
z=pd.read_pickle(V17/'financial_signal_features.pkl');ev=pd.read_pickle(V17/'new_financial_events.pkl');old=pd.read_pickle(V14/'financial_events.pkl')
valid=f.SignalDate.isin(calendar.Date).to_numpy();q=z.EPSActualQoQ.to_numpy();y=z.EPSActualGrowth.to_numpy();target=f.Target.to_numpy();eligible=(np.abs(q)<=100)&(np.abs(y)<=100)
def corr(a,b):return float(np.corrcoef(a,b)[0,1]) if len(a)>2 and np.std(a)>0 and np.std(b)>0 else None
def stats(a,b):return {'n':len(a),'pearson':corr(a,b),'spearman':corr(rankdata(a),rankdata(b))}
checks={};ix=valid&eligible;checks['daily_stock_rows_both_eligible']=stats(q[ix],y[ix]);ix&=(q!=0)&(y!=0);checks['daily_stock_rows_both_nonzero']=stats(q[ix],y[ix])
eq=ev[ev.Feature.eq('EPSActualQoQ')][['Code','SourceRow','AvailableDate','Value']]
ey=old[old.Kind.eq('A')&old.Metric.eq(2)][['Code','TriggerSourceRow','Values']].copy();ey['YoY']=[v[0] for v in ey.Values]
paired=eq.merge(ey,left_on=['Code','SourceRow'],right_on=['Code','TriggerSourceRow'],validate='one_to_one');paired=paired[paired.AvailableDate.between(calendar.Date.min(),calendar.Date.max()) & paired.Value.abs().le(100) & paired.YoY.abs().le(100)]
checks['paired_raw_announcement_events']=stats(paired.Value.to_numpy(),paired.YoY.to_numpy())
tr=pd.read_csv(V17/'eps_actual_joint/training_updates.csv');qt=pd.read_csv(V17/'eps_qoq_joint_mask/training_updates.csv');ratio=tr.LearningRate/qt.LearningRate
checks['joint_rate_vs_qoq_same_mask']={'median':float(ratio.median()),'min':float(ratio.min()),'fraction_below_half':float((ratio<.5).mean())}
hist=pd.read_csv(V17/'eps_actual_joint/parameter_history.csv');h=hist.set_index('Date');dail=[]
for date,idx in groups.items():
    if date not in set(calendar.Date):continue
    keep=eligible[idx]&np.isfinite(target[idx]);ii=idx[keep];a=q[ii];b=y[ii];pa=rankdata(a);pb=rankdata(b);ty=rankdata(target[ii]);a0=pa-pa.mean();b0=pb-pb.mean()
    resid=b0-(a0@b0/(a0@a0) if a0@a0>0 else 0)*a0
    theta=h.loc[str(date.date())];qa=float(theta.After_EPSActualQoQ)*a;ya=float(theta.After_EPSActualGrowth)*b
    dail.append({'Date':str(date.date()),'PredictorRankCorrelation':corr(pa,pb),'YoYRankResidualVsTargetRank':corr(resid,ty),'QoQRankIC':corr(pa,ty),'YoYRankIC':corr(pb,ty),'OpposingFinancialContributionsFraction':float(np.mean(qa*ya<0))})
pd.DataFrame(dail).to_csv(ROOT/'qoq_yoy_daily_diagnostics.csv',index=False)
checks['mean_daily_descriptive']={c:float(pd.DataFrame(dail)[c].mean()) for c in list(dail[0])[1:]}

# Reconstruct exact per-stock portfolio weights; split model-control return changes by recency.
market=pd.DatetimeIndex(groups).difference(pd.DatetimeIndex(['2020-10-01'])).sort_values();positions=market.searchsorted(f.SignalDate);code=f.SecuritiesCode.to_numpy();W=np.linspace(2.,1.,200);valset=set(calendar.Date);revision=[];contribs=[]
for feature,name in [('NetSalesRevisionRelative','sales_revision_relative'),('OperatingProfitRevisionRelative','profit_revision_relative'),('EPSRevisionRelative','eps_revision_relative')]:
    events=ev[ev.Feature.eq(feature)];age=np.full(len(f),99999,np.int32);isnegative=np.zeros(len(f),bool)
    lookup={c:g.sort_values(['AvailablePosition','EffectiveTimestamp','NewEventId']) for c,g in events.groupby('Code')}
    for c,idx in f.groupby('SecuritiesCode').indices.items():
        if c not in lookup:continue
        g=lookup[c];pos=g.AvailablePosition.to_numpy();k=np.searchsorted(pos,positions[idx],side='right')-1;known=k>=0;ii=idx[known];ki=k[known]
        age[ii]=positions[ii]-pos[ki];isnegative[ii]=g.BaseValue.to_numpy()[ki]<0
    model=np.load(V17/name/'predictions.npz');control=np.load(V17/(name+'_control')/'predictions.npz');mscore=model['score'];cscore=control['score'];mrank=model['rank'];crank=control['rank'];categories=['age_0','age_1_9','age_10_plus','never']
    totals={k:0. for k in categories};negcontrib=0.;dayschanged=0;selectedcounts=0;selectedrecent=0;selectednega=0;sumabs=0.
    for date,idx in groups.items():
        if date not in valset:continue
        n=len(idx);mw=np.zeros(n);cw=np.zeros(n)
        for rank,w in [(mrank[idx],mw),(crank[idx],cw)]:
            up=rank<200;dn=rank>=n-200;w[up]=.5*W[rank[up]]/W.sum();w[dn]=-.5*W[n-1-rank[dn]]/W.sum()
        delta=mw-cw;values=delta*np.nan_to_num(target[idx]);a=age[idx]
        cats=np.select([a==0,(a>=1)&(a<=9),(a>=10)&(a<99999)],['age_0','age_1_9','age_10_plus'],default='never')
        for k in categories:totals[k]+=float(values[cats==k].sum())
        negcontrib+=float(values[isnegative[idx]].sum());dayschanged+=int(np.any(delta!=0));selected= mw!=0;selectedcounts+=int(selected.sum());selectedrecent+=int((selected&(a<=9)).sum());selectednega+=int((selected&isnegative[idx]).sum());sumabs+=float(np.abs(delta).sum())
        contribs.append({'Variant':name,'Date':str(date.date()),'ReturnDifference':float(values.sum()),**{k:float(values[cats==k].sum()) for k in categories},'NegativeLastForecastBase':float(values[isnegative[idx]].sum())})
    md=pd.read_csv(V17/name/'daily_metrics.csv');cd=pd.read_csv(V17/(name+'_control')/'daily_metrics.csv');delta=md.IllustrativeGrossOneReturn-cd.IllustrativeGrossOneReturn
    np.testing.assert_allclose(sum(totals.values()),delta.sum(),atol=1e-12,rtol=0)
    active=valid&(age<=9);valid_events=events[events.AvailableDate.isin(calendar.Date)]
    revision.append({'Feature':feature,'Variant':name,'ValidationEvents':len(valid_events),'ValidationDatesWithNewEvent':int(valid_events.AvailableDate.nunique()),'ValidationEventStocks':int(valid_events.Code.nunique()),'FractionValidationRowsAge0to9':float(active.sum()/valid.sum()),'ValidationNegativeOldBaseEvents':int(valid_events.BaseValue.lt(0).sum()),'ValidationNegativeOldBaseFraction':float(valid_events.BaseValue.lt(0).mean()),'DaysPortfolioDifferentFromMatchedControl':dayschanged,'FractionSelectedStocksAge0to9':selectedrecent/selectedcounts,'MeanDailyReturnDeltaBP':float(delta.mean()*10000),'ModelDailyMeanBP':float(md.IllustrativeGrossOneReturn.mean()*10000),'ControlDailyMeanBP':float(cd.IllustrativeGrossOneReturn.mean()*10000),'ModelDailyStdBP':float(md.IllustrativeGrossOneReturn.std(ddof=1)*10000),'ControlDailyStdBP':float(cd.IllustrativeGrossOneReturn.std(ddof=1)*10000),'MeanDailyDeltaBPByLastEventAge':{k:v/len(calendar)*10000 for k,v in totals.items()},'NegativeLastForecastBaseMeanDeltaBP':negcontrib/len(calendar)*10000})
pd.DataFrame(contribs).to_csv(ROOT/'revision_return_contributions.csv',index=False)
checks['revisions']=revision
(ROOT/'diagnostics.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2,allow_nan=False))
print(json.dumps(checks,ensure_ascii=False,indent=2),flush=True)
