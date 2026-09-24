from pathlib import Path
import json,pickle,zipfile,hashlib
from collections import Counter
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
V8=ROOT.parent/'jpx_v8_soft_rank_20260912'
ZIP=Path('/Users/coolguy/Desktop/JPX_data/modelling/JPX_data.zip')
MEMBER='JPX_data/raw/train_files/financials.csv'

def prior_key(key):
    code,basis,period,start,end,fyend=key
    return code,basis,period,start-pd.DateOffset(years=1),end-pd.DateOffset(years=1),fyend-pd.DateOffset(years=1)

def read_records(core_codes,calendar):
    with zipfile.ZipFile(ZIP) as archive:
        content=archive.read(MEMBER)
        import io
        raw=pd.read_csv(io.BytesIO(content),dtype=str)
    raw['SourceRow']=np.arange(len(raw));raw['Code']=pd.to_numeric(raw.SecuritiesCode,errors='coerce')
    keep=raw.Code.isin(core_codes)&(raw.TypeOfDocument.str.contains('FinancialStatements',na=False)|raw.TypeOfDocument.eq('NumericalCorrection'))
    f=raw.loc[keep].copy();f['Code']=f.Code.astype(int)
    for c in ['NetSales','OperatingProfit']:f[c]=pd.to_numeric(f[c],errors='coerce')
    for c in ['Date','DisclosedDate','CurrentFiscalYearStartDate','CurrentPeriodEndDate','CurrentFiscalYearEndDate']:f[c]=pd.to_datetime(f[c])
    f['EffectiveDate']=f[['Date','DisclosedDate']].max(axis=1)
    f['EffectiveTimestamp']=f.EffectiveDate+pd.to_timedelta(f.DisclosedTime)
    f['Basis']=f.TypeOfDocument.str.replace(r'^(1Q|2Q|3Q|FY|OtherPeriod)FinancialStatements_','',regex=True)
    # At or after 15:00 is first usable on a later session. A weekend/closure
    # naturally maps to the next available trading date.
    positions=calendar.searchsorted(f.EffectiveDate)
    same=np.zeros(len(f),dtype=bool);inside=positions<len(calendar)
    same[inside]=calendar.to_numpy()[positions[inside]]==f.EffectiveDate.to_numpy()[inside]
    late=f.DisclosedTime.ge('15:00:00').to_numpy()
    positions=positions+(same&late)
    f['AvailablePosition']=positions
    f['AvailableDate']=[calendar[p] if p<len(calendar) else pd.NaT for p in positions]
    f=f.sort_values(['EffectiveTimestamp','DisclosureNumber','SourceRow'],kind='stable')
    return f,{'raw_rows':len(raw),'raw_sha256':hashlib.sha256(content).hexdigest(),'core_financial_or_correction_rows':len(f),
        'core_forecast_revision_rows_excluded':int((raw.Code.isin(core_codes)&raw.TypeOfDocument.str.startswith('ForecastRevision',na=False)).sum()),
        'relevant_rows_with_date_disclosure_mismatch':int(f.Date.ne(f.DisclosedDate).sum())}

def make_events(records):
    states={};latest={};events=[];counts=Counter()
    def signature(key):
        if key is None:return None
        cur=states[key];prior=states.get(prior_key(key))
        def margin(record):
            if record is None or not np.isfinite([record.NetSales,record.OperatingProfit]).all() or record.NetSales<=0:return None
            return float(record.OperatingProfit/record.NetSales)
        return margin(cur),margin(prior)
    for record in records.itertuples(index=False):
        code=record.Code;period=(record.TypeOfCurrentPeriod,record.CurrentFiscalYearStartDate,record.CurrentPeriodEndDate,record.CurrentFiscalYearEndDate)
        correction=record.TypeOfDocument=='NumericalCorrection'
        counts['correction_rows' if correction else 'regular_statement_rows']+=1
        before_latest=latest.get(code);before_signature=signature(before_latest)
        if correction:
            keys=[key for key in states if key[0]==code and key[2:]==period]
            if len(keys)!=1:
                counts['correction_unmatched_or_ambiguous_basis']+=1;continue
            if not np.isfinite([record.NetSales,record.OperatingProfit]).all():
                counts['correction_without_both_numeric_components']+=1;continue
            key=keys[0];counts['correction_applied_to_known_period']+=1
        else:key=(code,record.Basis)+period
        states[key]=record
        if not correction:
            if before_latest is None or key[4]>=before_latest[4]:latest[code]=key
        current_key=latest.get(code)
        if current_key is None:continue
        touches_comparison=key==current_key or key==prior_key(current_key)
        if not touches_comparison:
            counts['historical_update_without_new_pulse']+=1;continue
        after_signature=signature(current_key)
        if correction and before_signature==after_signature:
            counts['correction_with_unchanged_margins_no_pulse']+=1;continue
        current=states[current_key];prior=states.get(prior_key(current_key))
        cm,pm=after_signature;available=cm is not None and pm is not None
        change=cm-pm if available else np.nan
        event={'EventId':len(events),'SecuritiesCode':code,'TriggerSourceRow':record.SourceRow,'TriggerType':record.TypeOfDocument,
            'TriggerDisclosedDate':record.DisclosedDate,'TriggerDisclosedTime':record.DisclosedTime,
            'EffectiveTimestamp':record.EffectiveTimestamp,'AvailableDate':record.AvailableDate,'AvailablePosition':record.AvailablePosition,
            'Basis':current_key[1],'TypeOfCurrentPeriod':current_key[2],'CurrentFiscalYearStartDate':current_key[3],
            'CurrentPeriodEndDate':current_key[4],'CurrentFiscalYearEndDate':current_key[5],
            'CurrentSourceRow':current.SourceRow,'CurrentNetSales':current.NetSales,'CurrentOperatingProfit':current.OperatingProfit,
            'CurrentSourceAvailableDate':current.AvailableDate,'CurrentMargin':cm,
            'PriorSourceRow':prior.SourceRow if prior is not None else -1,
            'PriorNetSales':prior.NetSales if prior is not None else np.nan,'PriorOperatingProfit':prior.OperatingProfit if prior is not None else np.nan,
            'PriorSourceAvailableDate':prior.AvailableDate if prior is not None else pd.NaT,'PriorMargin':pm,
            'MarginYoYChange':change,'ComparisonAvailable':available,
            'UnavailableReason':'' if available else 'no_prior_same_period' if prior is None else 'missing_component_or_nonpositive_sales'}
        events.append(event);counts['events_with_comparison' if available else 'events_without_comparison']+=1
    return pd.DataFrame(events),dict(counts)

def align_to_signals(f,events,calendar):
    n=len(f);event_ids=np.full(n,-1,dtype=np.int32);ages=np.full(n,-1,dtype=np.int16)
    decay=np.zeros(n);feature=np.zeros(n);status=np.zeros(n,dtype=np.int8)
    event_groups={code:g.sort_values(['AvailableDate','EventId'],kind='stable') for code,g in events.loc[events.AvailableDate.notna()].groupby('SecuritiesCode')}
    for code,ix in f.groupby('SecuritiesCode').indices.items():
        if code not in event_groups:continue
        g=event_groups[code];dates=f.iloc[ix].SignalDate.to_numpy()
        loc=np.searchsorted(g.AvailableDate.to_numpy(),dates,side='right')-1;has=loc>=0
        chosen=g.iloc[loc[has]];rows=ix[has]
        age=calendar.searchsorted(dates[has],side='right')-1-chosen.AvailablePosition.to_numpy()
        assert (age>=0).all()
        valid=chosen.ComparisonAvailable.to_numpy();active=age<10
        r=np.maximum(1.-age/10.,0.)
        event_ids[rows]=chosen.EventId.to_numpy();ages[rows]=age
        decay[rows]=r
        status[rows]=np.where(~valid,1,np.where(active,2,3))
        feature[rows]=np.where(valid&active,chosen.MarginYoYChange.fillna(0).to_numpy()*r,0.)
    assert np.isfinite(feature).all()
    return pd.DataFrame({'FinancialEventId':event_ids,'FinancialAge':ages,'DecayWeight':decay,'FinancialStatus':status,'FinancialFeature':feature})

def main():
    with (V8/'inputs.pkl').open('rb') as h:f,_,groups,validation=pickle.load(h)
    calendar=pd.DatetimeIndex(groups).difference(pd.DatetimeIndex(['2020-10-01'])).sort_values()
    records,raw_stats=read_records(f.SecuritiesCode.unique(),calendar)
    records.to_pickle(ROOT/'financial_records.pkl')
    events,event_stats=make_events(records)
    events.to_pickle(ROOT/'financial_events.pkl');events.to_csv(ROOT/'financial_events.csv',index=False)
    aligned=align_to_signals(f,events,calendar)
    aligned.to_pickle(ROOT/'financial_signal_features.pkl')
    summary={'source':str(ZIP),'member':MEMBER,**raw_stats,**event_stats,'signal_rows_in_cache':len(f),
        'status_codes':{'0':'no_announcement_yet','1':'comparison_unavailable','2':'valid_within_10_sessions','3':'valid_but_expired'},
        'active_valid_rows':int(aligned.FinancialStatus.eq(2).sum()),'nonzero_feature_rows':int(aligned.FinancialFeature.ne(0).sum()),
        'nonzero_feature_fraction':float(aligned.FinancialFeature.ne(0).mean()),
        'feature_min':float(aligned.FinancialFeature.min()),'feature_max':float(aligned.FinancialFeature.max()),
        'validation_rows':int(f.SignalDate.isin(validation.Date).sum()),
        'validation_nonzero_feature_rows':int((f.SignalDate.isin(validation.Date)&aligned.FinancialFeature.ne(0)).sum()),
        'extreme_events_abs_margin_change_over_one':int(events.MarginYoYChange.abs().gt(1).sum()),
        'prior_margin_negative_events':int(events.PriorMargin.lt(0).sum()),'prior_margin_zero_events':int(events.PriorMargin.eq(0).sum())}
    extreme=events.loc[events.ComparisonAvailable].assign(AbsChange=lambda x:x.MarginYoYChange.abs()).nlargest(20,'AbsChange')
    extreme.to_csv(ROOT/'largest_margin_changes.csv',index=False)
    (ROOT/'feature_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
