from pathlib import Path
import json,pickle
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent.parent
M=['NetSales','OperatingProfit','EPS']
events=pd.read_pickle(ROOT/'financial_events.pkl')
snap=pd.read_pickle(ROOT/'quarter_forecasts.pkl')
records=pd.read_pickle(ROOT/'financial_records.pkl').set_index('SourceRow')
with (ROOT.parent/'jpx_v8_soft_rank_20260912'/'inputs.pkl').open('rb') as h:
    f,x,groups,cal=pickle.load(h)
valid_dates=set(cal.Date)
ev=events[events.AvailableDate.isin(valid_dates)].copy()
flat=[]
for e in ev.itertuples(index=False):
    for i,v in enumerate(e.Values):
        typ=e.Kind if e.Kind!='A' else ['ActualGrowth','ExpectedGrowth'][i]
        flat.append({'EventId':e.EventId,'Date':e.AvailableDate,'Code':e.Code,'Metric':M[e.Metric],'Kind':typ,'Value':v,'Base':e.Base,'Expected':e.Expected,'Actual':e.Actual,'SourceRow':e.TriggerSourceRow})
flat=pd.DataFrame(flat)
flat['Abs']=flat.Value.abs()
tail=[]
for key,g in flat.groupby(['Metric','Kind']):
    a=g.Abs
    tail.append({'Metric':key[0],'Kind':key[1],'Count':len(g),'AbsMedian':a.median(),'AbsP95':a.quantile(.95),'AbsP99':a.quantile(.99),'AbsP999':a.quantile(.999),'AbsMax':a.max(),'AbsGT10':int((a>10).sum()),'AbsGT100':int((a>100).sum()),'AbsGT1000':int((a>1000).sum()),'Top1pctSquaredMagnitudeShare':float(g.nlargest(max(1,int(np.ceil(len(g)*.01))),'Abs').Value.pow(2).sum()/g.Value.pow(2).sum())})
pd.DataFrame(tail).to_csv(OUT/'event_tails_validation.csv',index=False)
flat.nlargest(50,'Abs').to_csv(OUT/'largest_raw_events_validation.csv',index=False)

# Match each new within-year quarter event to the final preceding-quarter
# forecast that was available strictly before this announcement.
sg={key:g.sort_values(['EffectiveTimestamp','SourceRow']) for key,g in snap.groupby('Key')}
mech=[]
for e in ev[ev.Kind.eq('Q') & ev.KnownQuarters.between(1,3)].itertuples(index=False):
    prevkey=e.Key[:-1]+(e.Key[-1]-1,)
    prior=sg.get(prevkey)
    if prior is None:continue
    prior=prior[prior.EffectiveTimestamp<e.EffectiveTimestamp]
    if prior.empty:continue
    p=prior.iloc[-1];m=e.Metric
    if not np.isclose(p.Values[m],e.Base,rtol=1e-12,atol=1e-12):continue
    annual_old=float(p.Annual[m]);annual_new=e.Annual
    if not np.isfinite(annual_old):continue
    latest_actual=e.Cumulative-float(p.Cumulative[m])
    unchanged=annual_old==annual_new
    expected_delta=(e.Base-latest_actual)/(4-int(e.KnownQuarters))
    mech.append({'EventId':e.EventId,'Date':e.AvailableDate,'Code':e.Code,'Metric':M[m],'KnownQuarters':int(e.KnownQuarters),'AnnualOld':annual_old,'AnnualNew':annual_new,'AnnualUnchanged':unchanged,'PreviousAverageForecast':e.Base,'NewAverageForecast':e.Expected,'JustReportedQuarterActual':latest_actual,'QoQGrowth':e.Values[0],'MechanicalDelta':expected_delta,'ActualDelta':e.Expected-e.Base,'SourceRow':e.TriggerSourceRow})
mech=pd.DataFrame(mech)
mech.to_csv(OUT/'quarter_rollforward.csv',index=False)
ms=[]
for metric,g in mech.groupby('Metric'):
    u=g[g.AnnualUnchanged]
    ms.append({'Metric':metric,'MatchedWithinYearQuarterEvents':len(g),'AnnualUnchanged':len(u),'AnnualUnchangedFraction':len(u)/len(g),'UnchangedAbsQoQMedian':u.QoQGrowth.abs().median(),'UnchangedAbsQoQP95':u.QoQGrowth.abs().quantile(.95),'UnchangedQoQAbsGT10pct':int((u.QoQGrowth.abs()>.1).sum()),'UnchangedFormulaMaxAbsError':(u.MechanicalDelta-u.ActualDelta).abs().max()})
pd.DataFrame(ms).to_csv(OUT/'quarter_rollforward_summary.csv',index=False)
mech[mech.AnnualUnchanged].assign(Abs=lambda z:z.QoQGrowth.abs()).nlargest(30,'Abs').to_csv(OUT/'largest_mechanical_rollforwards.csv',index=False)

discounts=[]
for variant in ['sgd_only','sgd_sqrt']:
    h=pd.read_csv(ROOT/variant/'parameter_history.csv',parse_dates=['Date'])
    h=h[h.Date.isin(valid_dates)]
    for metric in M:
        a=h['After_'+metric+'Discount'];delta=a.diff().abs()
        discounts.append({'Variant':variant,'Metric':metric,'ValidationDays':len(a),'Mean':a.mean(),'Median':a.median(),'DaysZero':int(a.eq(0).sum()),'DaysOne':int(a.eq(1).sum()),'DaysLT001':int(a.lt(.01).sum()),'DaysGT099':int(a.gt(.99).sum()),'DailyAbsChangeMedian':delta.median(),'DailyAbsChangeP95':delta.quantile(.95),'DaysChangeGT025':int(delta.gt(.25).sum()),'DaysChangeGT05':int(delta.gt(.5).sum()),'MaxAbsChange':delta.max()})
pd.DataFrame(discounts).to_csv(OUT/'discount_stability_validation.csv',index=False)

summary={'event_scalar_rows_validation':len(flat),'tail_abs_gt10':int(flat.Abs.gt(10).sum()),'tail_abs_gt100':int(flat.Abs.gt(100).sum()),'tail_abs_gt1000':int(flat.Abs.gt(1000).sum()),'mechanical_summary':ms,'discount_summary':discounts}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
