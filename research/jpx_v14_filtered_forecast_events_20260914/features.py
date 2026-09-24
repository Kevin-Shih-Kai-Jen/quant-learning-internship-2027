from pathlib import Path
import importlib.util,json,pickle
from collections import Counter
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
V8=ROOT.parent/'jpx_v8_soft_rank_20260912'
V11=ROOT.parent/'jpx_v11_persistent_finance_std22_20260913'
V12=ROOT.parent/'jpx_v12_financial_growth_20260913'
METRICS=['NetSales','OperatingProfit','EPS']
ACTUAL=['NetSales','OperatingProfit','EarningsPerShare']
FORECAST=['ForecastNetSales','ForecastOperatingProfit','ForecastEarningsPerShare']
PARTS=['ActualGrowth','ExpectedGrowth','ForecastQoQ','ForecastYoY','Revision']
FINS=[m+p for m in METRICS for p in PARTS]
VARIANTS=['sgd_only','sgd_sqrt']
TRAINING_LIMIT=100.

def training_mask(financial_values):
    values=np.asarray(financial_values,dtype=float)
    assert values.ndim==2 and values.shape[1]==15
    peak=np.max(np.abs(values),axis=1)
    return np.isfinite(values).all(axis=1)&(peak<=TRAINING_LIMIT),peak

def shift_year(key,n=-1):
    c,b,s,e,q=key
    return c,b,s+pd.DateOffset(years=n),e+pd.DateOffset(years=n),q

def previous_quarter(key):
    c,b,s,e,q=key
    return (c,b,s,e,q-1) if q>1 else (c,b,s-pd.DateOffset(years=1),e-pd.DateOffset(years=1),4)

def normal_year(start,end):
    return pd.notna(start) and pd.notna(end) and start+pd.DateOffset(years=1)-pd.Timedelta(days=1)==end

def grow(x,b):
    return (x-b)/abs(b) if np.isfinite(x) and np.isfinite(b) and b!=0 else np.nan

class Builder:
    def __init__(self):
        self.actual={};self.annual={};self.estimates={};self.frozen={};self.started=set()
        self.events=[];self.snapshots=[];self.outcomes=[];self.counts=Counter()

    def update(self,store,key,row,cols,patch=False):
        values,refs=store.get(key,(np.full(3,np.nan),[[],[],[]]))
        values=values.copy();refs=[list(x) for x in refs]
        for m,c in enumerate(cols):
            if not patch or getattr(row,c+'Present'):
                values[m]=getattr(row,c);refs[m]=[row.SourceRow]
        store[key]=(values,refs)

    def quarter(self,key):
        vals,refs=self.actual.get(key,(np.full(3,np.nan),[[],[],[]]))
        if key[-1]==1:return vals.copy(),[list(r) for r in refs]
        prev,pr=self.actual.get(key[:-1]+(key[-1]-1,),(np.full(3,np.nan),[[],[],[]]))
        return vals-prev,[list(set(a+b)) for a,b in zip(refs,pr)]

    def emit(self,row,key,m,kind,values,refs,detail):
        if not all(np.isfinite(x) for x in values):
            self.counts['unavailable_'+kind]+=1;return
        ev={'EventId':len(self.events),'Code':key[0],'Key':key,'Metric':m,'Kind':kind,
            'AvailablePosition':row.AvailablePosition,'AvailableDate':row.AvailableDate,
            'EffectiveTimestamp':row.EffectiveTimestamp,'TriggerSourceRow':row.SourceRow,
            'Values':list(map(float,values)),'References':sorted(set(refs)),**detail}
        self.events.append(ev);self.counts['events_'+kind]+=1

    def process(self,row):
        code=row.Code;typ=row.TypeOfDocument
        # Generic revision/correction rows do not identify their accounting
        # basis. A unique historical basis is NOT evidence for their basis.
        if typ in ['ForecastRevision','NumericalCorrection']:
            return 'unverified_accounting_basis_'+typ
        fs=row.CurrentFiscalYearStartDate;fe=row.CurrentFiscalYearEndDate
        if not normal_year(fs,fe):return 'nonstandard_fiscal_year'
        q={'1Q':1,'2Q':2,'3Q':3,'FY':4}.get(row.TypeOfCurrentPeriod)
        regular='FinancialStatements_' in typ
        correction=typ=='NumericalCorrection'
        ak=None;fk=None;first=False;oldannual=None
        if regular or correction:
            if q is None or row.CurrentPeriodEndDate!=fs+pd.DateOffset(months=3*q)-pd.Timedelta(days=1):return 'nonstandard_quarter_boundary'
            if correction:
                matches=[k for k in self.actual if k[0]==code and k[2:]==(fs,fe,q)]
                if len(matches)!=1:return 'correction_missing_or_ambiguous_original'
                ak=matches[0]
            else:ak=(code,row.Basis,fs,fe,q)
            first=ak not in self.actual
            if first:
                pre=self.estimates.get(ak)
                # Never take a forecast first published alongside the actual.
                if pre is not None and pre['time']<row.EffectiveTimestamp:
                    self.frozen[ak]=pre
                else:self.frozen[ak]={'v':np.full(3,np.nan),'refs':[[],[],[]],'time':pd.NaT}
            self.update(self.actual,ak,row,ACTUAL,correction)
            if first:
                rv,rr=self.quarter(ak);bv,br=self.quarter(shift_year(ak));pre=self.frozen[ak]
                for m in range(3):
                    self.emit(row,ak,m,'A',[grow(rv[m],bv[m]),grow(pre['v'][m],bv[m])],
                        rr[m]+br[m]+pre['refs'][m],{'Actual':rv[m],'Expected':pre['v'][m],'Base':bv[m],'ForecastTime':pre['time']})
            fk=(code,ak[1],fe+pd.Timedelta(days=1),fe+pd.DateOffset(years=1)) if q==4 else ak[:4]
            oldannual=self.annual.get(fk)
            if not correction or any(getattr(row,c+'Present') for c in FORECAST):self.update(self.annual,fk,row,FORECAST,True)
        else:
            if typ!='ForecastRevision':return 'unsupported_document'
            if row.TypeOfCurrentPeriod!='FY':return 'nonannual_revision'
            if not any(getattr(row,c+'Present') for c in FORECAST):return 'revision_no_selected_fields'
            bases={k[1] for k in self.annual if k[0]==code and k[2:]==(fs,fe)}
            bases|={k[1] for k in self.actual if k[0]==code and k[2:4]==(fs,fe)}
            if len(bases)!=1:return 'revision_missing_or_ambiguous_basis'
            fk=(code,next(iter(bases)),fs,fe);oldannual=self.annual.get(fk)
            self.update(self.annual,fk,row,FORECAST,True)
        # Recompute the affected fiscal year's residual estimate from current
        # known YTD, even when only a past actual was corrected.
        keys={fk}
        if correction and ak[:4]!=fk:keys.add(ak[:4])
        for target in keys:
            if target not in self.annual:continue
            ks=[k[-1] for k in self.actual if k[:4]==target]
            k=max(ks,default=0)
            if k>=4:continue
            qkey=target+(k+1,)
            cum,cr=self.actual[target+(k,)] if k else (np.zeros(3),[[],[],[]])
            av,ar=self.annual[target];nv=(av-cum)/(4-k)
            refs=[sorted(set(a+b)) for a,b in zip(ar,cr)]
            oldq=self.estimates.get(qkey)
            prev=self.frozen.get(previous_quarter(qkey),{'v':np.full(3,np.nan),'refs':[[],[],[]]})
            year=self.frozen.get(shift_year(qkey),{'v':np.full(3,np.nan),'refs':[[],[],[]]})
            baseline,brefs=self.quarter(shift_year(qkey))
            for m in range(3):
                prior_annual=oldannual[0][m] if target==fk and oldannual is not None else np.nan
                new_value=(target==fk and getattr(row,FORECAST[m]+'Present') and np.isfinite(av[m])
                           and (not np.isfinite(prior_annual) or av[m]!=prior_annual))
                if not new_value and np.isfinite(nv[m]) and (qkey,m) not in self.started:
                    self.counts['unchanged_or_absent_forecast_initial_signal_suppressed']+=1
                if new_value and np.isfinite(nv[m]) and (qkey,m) not in self.started:
                    self.started.add((qkey,m))
                    for kind,base in [('Q',prev),('Y',year)]:
                        self.emit(row,qkey,m,kind,[grow(nv[m],base['v'][m])],refs[m]+base['refs'][m],
                            {'Expected':nv[m],'Base':base['v'][m],'Annual':av[m],'OldAnnual':prior_annual,
                             'Cumulative':cum[m],'KnownQuarters':k,'HasNewForecastValue':True})
                if target==fk and oldannual is not None:
                    ov,orr=oldannual;oldres=(ov[m]-cum[m])/(4-k)
                    if np.isfinite(ov[m]) and np.isfinite(av[m]) and ov[m]!=av[m]:
                        delta=grow(nv[m],baseline[m])-grow(oldres,baseline[m])
                        self.emit(row,qkey,m,'U',[delta],refs[m]+orr[m]+brefs[m],
                            {'NewAnnual':av[m],'OldAnnual':ov[m],'Cumulative':cum[m],'KnownQuarters':k,
                             'NewExpected':nv[m],'OldExpected':oldres,'Base':baseline[m],'HasNewForecastValue':True})
            # The final preannouncement snapshot is frozen on the actual row,
            # but corrections can affect future expectations prospectively.
            estimate={'v':nv.copy(),'refs':refs,'time':row.EffectiveTimestamp}
            self.estimates[qkey]=estimate
            self.snapshots.append({'Key':qkey,'Values':nv.copy(),'Annual':av.copy(),'Cumulative':cum.copy(),
                'KnownQuarters':k,'References':refs,'EffectiveTimestamp':row.EffectiveTimestamp,'SourceRow':row.SourceRow})
        return 'processed'

    def run(self,records):
        # State dictionaries are per company to keep temporal matching bounded.
        for _,group in records.groupby('Code',sort=True):
            self.actual={};self.annual={};self.estimates={};self.frozen={};self.started=set()
            for row in group.sort_values(['EffectiveTimestamp','DisclosureNumber','SourceRow'],kind='stable').itertuples(index=False):
                reason=self.process(row);self.counts[reason]+=1
                self.outcomes.append({'SourceRow':row.SourceRow,'Disposition':reason})
        return pd.DataFrame(self.events),pd.DataFrame(self.snapshots),pd.DataFrame(self.outcomes)

def event_vector(e):
    z=np.zeros(15);start=5*e.Metric
    if e.Kind=='A':z[start:start+2]=e.Values
    else:z[start+{'Q':2,'Y':3,'U':4}[e.Kind]]=e.Values[0]
    return z

def map_signals(f,events,market):
    positions=market.searchsorted(f.SignalDate)
    z=np.zeros((len(f),15));counts=np.zeros(len(f),dtype=np.int32)
    evgroups={c:g.sort_values(['AvailablePosition','EffectiveTimestamp','EventId']) for c,g in events.groupby('Code')}
    for code,ix in f.groupby('SecuritiesCode').indices.items():
        if code not in evgroups:continue
        ev=list(evgroups[code].itertuples(index=False));j=0;state=np.zeros(15);last=0
        for i in ix:
            p=positions[i]
            while j<len(ev) and ev[j].AvailablePosition<=p:
                e=ev[j];state*=np.exp(-(e.AvailablePosition-last)/9);state+=event_vector(e)
                last=e.AvailablePosition;j+=1
            state*=np.exp(-(p-last)/9);last=p;z[i]=state;counts[i]=j
    result=pd.DataFrame(z,columns=FINS);result['KnownEventCount']=counts
    eligible,peak=training_mask(z);result['TrainingEligible']=eligible;result['TrainingFeatureMaxAbs']=peak
    return result

def build():
    with (V8/'inputs.pkl').open('rb') as h:f,_,groups,validation=pickle.load(h)
    market=pd.DatetimeIndex(groups).difference(pd.DatetimeIndex(['2020-10-01'])).sort_values()
    spec=importlib.util.spec_from_file_location('v12_source_features',V12/'features.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    records,stats=old.read_records(f.SecuritiesCode.unique(),market)
    records.to_pickle(ROOT/'financial_records.pkl')
    builder=Builder();events,snapshots,outcomes=builder.run(records)
    events.to_pickle(ROOT/'financial_events.pkl');snapshots.to_pickle(ROOT/'quarter_forecasts.pkl');outcomes.to_csv(ROOT/'source_dispositions.csv',index=False)
    signals=map_signals(f,events,market);signals.to_pickle(ROOT/'financial_signal_features.pkl')
    valid=f.SignalDate.isin(validation.Date)
    stats.update({'event_count':len(events),'quarter_snapshot_count':len(snapshots),'counts':dict(builder.counts),
        'signal_rows':len(f),'validation_rows':int(valid.sum()),'validation_any_financial_rows':int(signals.loc[valid,FINS].ne(0).any(axis=1).sum()),
        'training_threshold':TRAINING_LIMIT,'excluded_signal_rows':int((~signals.TrainingEligible).sum()),
        'excluded_validation_signal_rows':int((~signals.loc[valid,'TrainingEligible']).sum()),
        'ranges':{c:{'min':float(signals[c].min()),'max':float(signals[c].max()),'nonzero_validation_rows':int(signals.loc[valid,c].ne(0).sum())} for c in FINS}})
    (ROOT/'feature_summary.json').write_text(json.dumps(stats,indent=2));print(json.dumps(stats),flush=True)

def load():
    with (V8/'inputs.pkl').open('rb') as h:f,x,groups,calendar=pickle.load(h)
    std=pd.read_pickle(V11/'return_std22_features.pkl')
    x=x.copy();x[:,-2:]=std[['PR1Scaled22','VR1Scaled22']].fillna(0).to_numpy()
    fin=pd.read_pickle(ROOT/'financial_signal_features.pkl')
    return f,np.column_stack([x,fin[FINS].to_numpy()]),groups,calendar,fin

if __name__=='__main__':build()
