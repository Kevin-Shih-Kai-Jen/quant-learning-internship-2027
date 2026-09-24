from pathlib import Path
from collections import Counter
import json,pickle,zipfile,hashlib,io
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
V8=ROOT.parent/'jpx_v8_soft_rank_20260912'
V11=ROOT.parent/'jpx_v11_persistent_finance_std22_20260913'
FINS=['ActualNetSalesGrowth','ActualOperatingProfitGrowth','ActualEPS','ForecastNetSalesGrowth','ForecastOperatingProfitGrowth','ForecastEPS']
ACTUAL=['NetSales','OperatingProfit','EarningsPerShare']
FORECAST=['ForecastNetSales','ForecastOperatingProfit','ForecastEarningsPerShare']
VARIANTS=['six_financial','price_only']

def previous(key):
    code,basis,period,start,end,fyend=key
    return code,basis,period,start-pd.DateOffset(years=1),end-pd.DateOffset(years=1),fyend-pd.DateOffset(years=1)

def read_records(codes,calendar):
    with zipfile.ZipFile('/Users/coolguy/Desktop/JPX_data/modelling/JPX_data.zip') as z:content=z.read('JPX_data/raw/train_files/financials.csv')
    raw=pd.read_csv(io.BytesIO(content),dtype=str);raw['SourceRow']=np.arange(len(raw));raw['Code']=pd.to_numeric(raw.SecuritiesCode,errors='coerce')
    mask=raw.Code.isin(codes)&(raw.TypeOfDocument.str.contains('FinancialStatements',na=False)|raw.TypeOfDocument.isin(['NumericalCorrection','ForecastRevision']))
    r=raw.loc[mask].copy();r['Code']=r.Code.astype(int)
    for c in ACTUAL+FORECAST:
        r[c+'Present']=r[c].notna()&r[c].fillna('').str.strip().ne('')
        r[c]=pd.to_numeric(r[c],errors='coerce').where(lambda s:np.isfinite(s))
    for c in ['Date','DisclosedDate','CurrentFiscalYearStartDate','CurrentPeriodEndDate','CurrentFiscalYearEndDate']:r[c]=pd.to_datetime(r[c])
    r['EffectiveTimestamp']=r[['Date','DisclosedDate']].max(axis=1)+pd.to_timedelta(r.DisclosedTime)
    # First market close strictly AFTER publication, including at-close records.
    closes=calendar+pd.Timedelta(hours=15)
    pos=closes.searchsorted(r.EffectiveTimestamp,side='right')
    r['AvailablePosition']=pos;r['AvailableDate']=[calendar[i] if i<len(calendar) else pd.NaT for i in pos]
    r['Basis']=r.TypeOfDocument.str.replace(r'^(1Q|2Q|3Q|FY|OtherPeriod)FinancialStatements_','',regex=True)
    r=r.sort_values(['EffectiveTimestamp','DisclosureNumber','SourceRow'],kind='stable').reset_index(drop=True)
    return r,{'source_sha256':hashlib.sha256(content).hexdigest(),'selected_source_rows':len(r),'raw_rows':len(raw)}

def build_events(records):
    actual={};forecasts={};latest_actual={};latest_forecast={};events=[];counts=Counter();outcomes=[];last_signature={}
    def values_update(state,key,row,cols,patch):
        d=state.setdefault(key,{c:(np.nan,-1) for c in cols})
        for c in cols:
            if not patch or getattr(row,c+'Present'):d[c]=(getattr(row,c),row.SourceRow)
    def entry(state,key,c):return state.get(key,{}).get(c,(np.nan,-1))
    def snapshot(code,row):
        ak=latest_actual.get(code);fk=latest_forecast.get(code);prior=previous(ak) if ak else None
        fkbase=(fk[0],fk[1],'FY',fk[2]-pd.DateOffset(years=1),fk[3]-pd.DateOffset(years=1),fk[3]-pd.DateOffset(years=1)) if fk else None
        current=[entry(actual,ak,c) for c in ACTUAL]+[entry(forecasts,fk,c) for c in FORECAST]
        bases=[entry(actual,prior,c) for c in ACTUAL[:2]]+[(np.nan,-1)]+[entry(actual,fkbase,c) for c in ACTUAL[:2]]+[(np.nan,-1)]
        e={'EventId':len(events),'Code':code,'TriggerSourceRow':row.SourceRow,'AvailableDate':row.AvailableDate,
            'AvailablePosition':row.AvailablePosition,'EffectiveTimestamp':row.EffectiveTimestamp,
            'ActualKey':ak,'ActualPriorKey':prior,'ForecastKey':fk,'ForecastBaseKey':fkbase}
        signature=[]
        for k,name in enumerate(FINS):
            cv,ci=current[k];bv,bi=bases[k]
            if k in [2,5]:valid=np.isfinite(cv);v=cv if valid else 0.;reason='' if valid else 'missing_current'
            else:
                valid=np.isfinite(cv) and np.isfinite(bv) and bv!=0
                v=(cv-bv)/abs(bv) if valid else 0.
                reason='' if valid else 'missing_current' if not np.isfinite(cv) else 'missing_base' if not np.isfinite(bv) else 'zero_base'
            assert np.isfinite(v)
            e[name]=v;e[name+'Valid']=bool(valid);e[name+'Reason']=reason
            e[name+'CurrentValue']=cv;e[name+'BaseValue']=bv;e[name+'CurrentRow']=ci;e[name+'BaseRow']=bi
            signature.extend([v,bool(valid),ci,bi])
        signature.extend([ak,fk])
        if last_signature.get(code)!=tuple(signature):
            events.append(e);last_signature[code]=tuple(signature)
    for r in records.itertuples(index=False):
        code=r.Code;typ=r.TypeOfDocument;period=(r.TypeOfCurrentPeriod,r.CurrentFiscalYearStartDate,r.CurrentPeriodEndDate,r.CurrentFiscalYearEndDate)
        reason='';ak=None;changed=False
        if 'FinancialStatements_' in typ:
            ak=(code,r.Basis)+period;values_update(actual,ak,r,ACTUAL,False);changed=True
            old=latest_actual.get(code)
            if old is None or ak[4]>=old[4]:latest_actual[code]=ak
            counts['regular_statements']+=1
        elif typ=='NumericalCorrection':
            matches=[k for k in actual if k[0]==code and k[2:]==period]
            if len(matches)!=1:reason='correction_missing_or_ambiguous_original';counts[reason]+=1
            else:
                ak=matches[0];values_update(actual,ak,r,ACTUAL,True);changed=True;counts['correction_applied']+=1
        else:
            assert typ=='ForecastRevision'
            if r.TypeOfCurrentPeriod!='FY':reason='nonannual_forecast_revision_excluded';counts[reason]+=1
            elif not any(getattr(r,c+'Present') for c in FORECAST):reason='forecast_revision_without_selected_fields';counts[reason]+=1
            else:
                # Basis is not encoded on revision rows: only use a unique
                # basis already known for this target financial year.
                bases={k[1] for k in forecasts if k[0]==code and k[2:]==(r.CurrentFiscalYearStartDate,r.CurrentFiscalYearEndDate)}
                bases|={k[1] for k in actual if k[0]==code and k[3]==r.CurrentFiscalYearStartDate and k[5]==r.CurrentFiscalYearEndDate}
                if len(bases)!=1:reason='revision_missing_or_ambiguous_basis';counts[reason]+=1
                else:
                    fk=(code,next(iter(bases)),r.CurrentFiscalYearStartDate,r.CurrentFiscalYearEndDate)
                    values_update(forecasts,fk,r,FORECAST,True);changed=True
                    old=latest_forecast.get(code)
                    if old is None or fk[3]>=old[3]:latest_forecast[code]=fk
                    counts['annual_forecast_revision_applied']+=1
        if ak is not None:
            basis=ak[1]
            if r.TypeOfCurrentPeriod=='FY':
                fs=r.CurrentFiscalYearEndDate+pd.Timedelta(days=1);fe=r.CurrentFiscalYearEndDate+pd.DateOffset(years=1)
                rule='fy_statement_next_year'
            else:fs=r.CurrentFiscalYearStartDate;fe=r.CurrentFiscalYearEndDate;rule='interim_statement_current_year'
            fk=(code,basis,fs,fe)
            # Numerical correction only patches supplied fields. Regular
            # financial statements replace the forecast snapshot, including NA.
            if typ!='NumericalCorrection' or any(getattr(r,c+'Present') for c in FORECAST):
                values_update(forecasts,fk,r,FORECAST,typ=='NumericalCorrection')
                old=latest_forecast.get(code)
                if old is None or fk[3]>=old[3]:latest_forecast[code]=fk
                counts[rule]+=1
        if changed:snapshot(code,r)
        outcomes.append({'SourceRow':r.SourceRow,'TypeOfDocument':typ,'Disposition':reason or 'processed'})
    return pd.DataFrame(events),pd.DataFrame(outcomes),dict(counts)

def build():
    with (V8/'inputs.pkl').open('rb') as h:f,_,groups,validation=pickle.load(h)
    calendar=pd.DatetimeIndex(groups).difference(pd.DatetimeIndex(['2020-10-01'])).sort_values()
    records,stats=read_records(f.SecuritiesCode.unique(),calendar);records.to_pickle(ROOT/'financial_records.pkl')
    events,outcomes,counts=build_events(records);events.to_pickle(ROOT/'financial_events.pkl');outcomes.to_csv(ROOT/'source_dispositions.csv',index=False)
    n=len(f);ids=np.full(n,-1,dtype=np.int32);z=np.zeros((n,6));valid=np.zeros((n,6),dtype=bool)
    eventgroups={c:g.sort_values(['AvailableDate','EventId']) for c,g in events.loc[events.AvailableDate.notna()].groupby('Code')}
    for code,ix in f.groupby('SecuritiesCode').indices.items():
        if code not in eventgroups:continue
        g=eventgroups[code];loc=np.searchsorted(g.AvailableDate.to_numpy(),f.iloc[ix].SignalDate.to_numpy(),side='right')-1;has=loc>=0
        chosen=g.iloc[loc[has]];rows=ix[has];ids[rows]=chosen.EventId;z[rows]=chosen[FINS];valid[rows]=chosen[[c+'Valid' for c in FINS]]
    features=pd.DataFrame(z,columns=FINS);features['FinancialEventId']=ids
    for k,c in enumerate(FINS):features[c+'Valid']=valid[:,k]
    features.to_pickle(ROOT/'financial_signal_features.pkl')
    val=f.SignalDate.isin(validation.Date).to_numpy()
    stats|={'dispositions':counts,'event_count':len(events),'signal_rows':n,'validation_rows':int(val.sum()),
        'features':{c:{'min':float(z[:,k].min()),'max':float(z[:,k].max()),'validation_valid_rows':int(valid[val,k].sum()),
        'validation_nonzero_rows':int((z[val,k]!=0).sum()),'zero_base_events':int(events[c+'Reason'].eq('zero_base').sum())} for k,c in enumerate(FINS)}}
    (ROOT/'feature_summary.json').write_text(json.dumps(stats,indent=2));print(json.dumps(stats),flush=True)

def load(variant):
    assert variant in VARIANTS
    with (V8/'inputs.pkl').open('rb') as h:f,x,groups,calendar=pickle.load(h)
    std=pd.read_pickle(V11/'return_std22_features.pkl');f=f.copy();x=x.copy()
    for c in ['PR1','VR1']:f[c]=std[c+'Scaled22'].to_numpy()
    x[:,-2:]=f[['PR1','VR1']].fillna(0).to_numpy()
    fin=pd.read_pickle(ROOT/'financial_signal_features.pkl')
    if variant=='price_only':fin=fin.copy();fin[FINS]=0.
    f=pd.concat([f,fin],axis=1)
    return f,np.column_stack([x,fin[FINS].to_numpy()]),groups,calendar

if __name__=='__main__':build()
