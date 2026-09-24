from pathlib import Path
import importlib.util,json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.signal import lfilter
R=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('v30run',R/'run.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
def csv(n,d):d.to_csv(R/n,index=False,encoding='utf-8-sig')
def corr(a,b,rank=False):
    if len(a)<3 or np.std(a)==0 or np.std(b)==0:return None
    return float(spearmanr(a,b).statistic if rank else np.corrcoef(a,b)[0,1])
def main():
    f,x,z,groups,cal=r.state_load();market=pd.DatetimeIndex(groups).difference(pd.DatetimeIndex(['2020-10-01'])).sort_values();pos=market.searchsorted(f.SignalDate);valid=f.SignalDate.isin(cal.Date).to_numpy();vR=z[r.RFEATURE].to_numpy();vF=z[r.FFEATURE].to_numpy()
    ev=pd.read_pickle(r.V14/'financial_events.pkl');re=ev[(ev.Metric==1)&(ev.Kind=='A')].copy();fe=pd.read_pickle(r.V27/'state_events.pkl');re['Value']=re.Values.map(lambda a:a[0]);re['ExpectedValue']=re.Values.map(lambda a:a[1]);np.testing.assert_allclose(re.Value,(re.Actual-re.Base)/re.Base.abs(),rtol=1e-12)
    rpos=np.full(len(f),-1,dtype=int);eventidx=np.full(len(f),-1,dtype=int);reconstruction=np.zeros(len(f));regroup={c:g.sort_values(['AvailablePosition','EffectiveTimestamp','EventId']) for c,g in re.groupby('Code')}
    for code,ids in f.groupby('SecuritiesCode').indices.items():
        if code not in regroup:continue
        g=regroup[code];ep=g.AvailablePosition.to_numpy(int);impulse=np.zeros(max(len(market),int(ep.max())+1));np.add.at(impulse,ep,g.Value.to_numpy());a=lfilter([1],[1,-np.exp(-1/9)],impulse);reconstruction[ids]=a[pos[ids]]
        at=np.searchsorted(ep,pos[ids],side='right')-1;known=at>=0;rpos[ids[known]]=ep[at[known]];eventidx[ids[known]]=g.index.to_numpy()[at[known]]
    np.testing.assert_allclose(reconstruction,vR,rtol=2e-10,atol=2e-10)
    with np.load(r.V27/'signal_event_mapping.npz') as p:fpos=p['event_position'];fidx=p['event_id']
    known=fidx>=0;f_replay=np.zeros(len(f));lookup=fe.set_index('StateEventId');f_replay[known]=lookup.loc[fidx[known],'NewYoY'].to_numpy()*np.exp(-(pos[known]-fpos[known])/9)
    np.testing.assert_allclose(f_replay,vF,rtol=1e-12,atol=1e-12)
    assert ((rpos<=pos)|(rpos<0)).all() and ((fpos<=pos)|(fpos<0)).all()
    rage=np.where(rpos>=0,pos-rpos,np.inf);fage=np.where(fpos>=0,pos-fpos,np.inf);near=np.minimum(rage,fage)<=2;both=(vR!=0)&(vF!=0)
    masks={'all_validation':valid,'near_update_0_to_2_days':valid&near,'other_days':valid&~near,'both_nonzero':valid&both,'both_nonzero_near_update':valid&both&near}
    correlations=[]
    for n,m in masks.items():correlations.append(dict(Sample=n,StockDays=int(m.sum()),Pearson=corr(vR[m],vF[m]),Spearman=corr(vR[m],vF[m],True),OppositeSignFraction=float(np.mean(vR[m]*vF[m]<0))))
    csv('feature_correlations.csv',pd.DataFrame(correlations))
    scales=[]
    for feature in z.columns:
        a=np.abs(z.loc[valid,feature].to_numpy());nonzero=a[a>0];scales.append(dict(Feature=feature,NonzeroStockDays=len(nonzero),MaxAbsolute=float(a.max()),NonzeroMedianAbsolute=float(np.median(nonzero)),NonzeroP99Absolute=float(np.quantile(nonzero,.99))))
    csv('feature_scales.csv',pd.DataFrame(scales))
    # Announcement matches preserve source row and target-quarter keys.
    timing=re[['EventId','TriggerSourceRow','Key','AvailablePosition']].merge(fe[['SourceRow','Key','Valid','StateEventId','Annual','Cumulative','KnownQuarters','NewForecast']],left_on='TriggerSourceRow',right_on='SourceRow',how='left',suffixes=('_Actual','_Forecast'))
    def end(k):return k[2]+pd.DateOffset(months=3*k[-1])-pd.Timedelta(days=1)
    timing['ActualQuarterEnd']=timing.Key_Actual.map(end);timing['ForecastQuarterEnd']=timing.Key_Forecast.map(lambda k:end(k) if isinstance(k,tuple) else pd.NaT)
    timing['Relation']=np.where(timing.ForecastQuarterEnd.isna(),'no_state_update',np.where(timing.ForecastQuarterEnd>timing.ActualQuarterEnd,'future_quarter',np.where(timing.ForecastQuarterEnd==timing.ActualQuarterEnd,'same_quarter','past_quarter')))
    csv('announcement_quarter_alignment.csv',timing.drop(columns=['Key_Actual','Key_Forecast']))
    matching=timing.Valid.eq(True);rows=timing[matching];np.testing.assert_allclose(rows.NewForecast,(rows.Annual-rows.Cumulative)/(4-rows.KnownQuarters),rtol=1e-12,atol=1e-5)
    with np.load(R/'g/predictions.npz') as p:gscore=p['score']
    contributions=[];gradients=[];coefficients=[];gdrift=[]
    for name,features in r.JOBS.items():
        if not features:continue
        hist=pd.read_csv(R/name/'parameter_history.csv').set_index('Date');updates=pd.read_csv(R/name/'training_updates.csv');frozen=name.startswith('frozen_');CR=np.zeros(len(f));CF=np.zeros(len(f));G=np.full(len(f),np.nan)
        for date,ids in groups.items():
            if date==pd.Timestamp('2020-10-01') or date>cal.Date.max():continue
            h=hist.loc[str(date.date())]
            if r.RFEATURE in features:CR[ids]=vR[ids]*h['After_'+r.RFEATURE]
            if r.FFEATURE in features:CF[ids]=vF[ids]*h['After_'+r.FFEATURE]
            G[ids]=gscore[ids] if frozen else x[ids]@h[['After_'+c for c in r.NAMES]].to_numpy(float)
        with np.load(R/name/'predictions.npz') as p:pred=p['score']
        scored=np.isfinite(pred);np.testing.assert_allclose(G[scored]+CR[scored]+CF[scored],pred[scored],rtol=1e-8,atol=2e-10)
        if frozen:np.testing.assert_array_equal(G[scored],gscore[scored])
        den=np.abs(CR)+np.abs(CF);nz=valid&(den>0);bothcontrib=valid&(CR!=0)&(CF!=0)
        cancellation=float(np.sum(den[valid]-np.abs(CR[valid]+CF[valid]))/np.sum(den[valid])) if np.sum(den[valid]) else 0
        contributions.append(dict(Variant=name,MeanAbsR=float(np.mean(np.abs(CR[valid]))),MeanAbsF=float(np.mean(np.abs(CF[valid]))),BothNonzeroStockDays=int(bothcontrib.sum()),OppositeContributionFraction=float(np.mean(CR[bothcontrib]*CF[bothcontrib]<0)) if bothcontrib.any() else None,AggregateCancellationFraction=cancellation))
        gdrift.append(dict(Variant=name,MeanAbsoluteGChange=float(np.mean(np.abs(G[valid]-gscore[valid]))),GScoreSpearmanVsPureG=corr(G[valid],gscore[valid],True)))
        for feature in features:
            beta=hist['After_'+feature].to_numpy();sg=np.sign(beta);sg=sg[sg!=0];grad=np.abs(updates['Gradient_'+feature].to_numpy());step=grad*updates.LearningRate.to_numpy()
            coefficients.append(dict(Variant=name,Feature=feature,FinalCoefficient=float(beta[-1]),SignFlips=int(np.sum(sg[1:]!=sg[:-1])),Min=float(beta.min()),Max=float(beta.max())))
            gradients.append(dict(Variant=name,Feature=feature,MedianAbsGradient=float(np.median(grad)),MaxAbsGradient=float(grad.max()),MedianAbsParameterUpdate=float(np.median(step)),MaxAbsParameterUpdate=float(step.max())))
    for n,d in [('prediction_contributions.csv',contributions),('financial_gradient_scales.csv',gradients),('coefficient_stability.csv',coefficients),('g_prediction_drift.csv',gdrift)]:csv(n,pd.DataFrame(d))
    diag=dict(passed=True,R_events=len(re),R_signal_reconstructed_all_rows=True,F_signal_reconstructed_all_rows=True,signal_timing_checked=True,valid_F_updates_matching_R_events=int(matching.sum()),all_R_event_relations=timing.Relation.value_counts().to_dict(),valid_F_event_relations=timing.loc[matching,'Relation'].value_counts().to_dict(),R_inherited_coverage_rule='v14 A event requires both actual growth and preannouncement expected growth finite',R_state='sum of decayed historical actual events',F_state='latest forecast state only, decayed since last state update',same_announcement_future_forecast_identity_checked=True,contribution_reconstruction_passed=True,frozen_g_exactly_matches_control=True)
    r.save(R/'diagnostics.json',diag);print(json.dumps(diag,ensure_ascii=False,indent=2));print(pd.DataFrame(correlations).to_string(index=False));print(pd.DataFrame(contributions).to_string(index=False));print(pd.DataFrame(gdrift).to_string(index=False));print(pd.DataFrame(scales).to_string(index=False));print(pd.DataFrame(coefficients).to_string(index=False));print(pd.DataFrame(gradients).to_string(index=False))
if __name__=='__main__':main()
