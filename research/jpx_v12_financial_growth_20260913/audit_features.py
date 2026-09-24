import json,pickle
from types import SimpleNamespace
import numpy as np
import pandas as pd
from features import ROOT,V8,V11,FINS,ACTUAL,FORECAST,load,build_events

def synthetic():
    rows=[]
    def row(typ,period,start,end,fyend,date,actual=None,forecast=None):
        d={'Code':1,'TypeOfDocument':typ,'TypeOfCurrentPeriod':period,'Basis':'Consolidated_JP',
            'CurrentFiscalYearStartDate':pd.Timestamp(start),'CurrentPeriodEndDate':pd.Timestamp(end),'CurrentFiscalYearEndDate':pd.Timestamp(fyend),
            'SourceRow':len(rows),'AvailableDate':pd.Timestamp(date),'AvailablePosition':len(rows),'EffectiveTimestamp':pd.Timestamp(date)}
        for cols,vals in [(ACTUAL,actual),(FORECAST,forecast)]:
            for k,c in enumerate(cols):d[c]=vals[k] if vals is not None else np.nan;d[c+'Present']=vals is not None
        rows.append(d)
    row('FYFinancialStatements_Consolidated_JP','FY','2016-01-01','2016-12-31','2016-12-31','2017-02-01',[100,-100,-2],[120,50,3])
    row('1QFinancialStatements_Consolidated_JP','1Q','2017-01-01','2017-03-31','2017-12-31','2017-05-01',[30,-10,-1],[130,60,4])
    row('ForecastRevision','2Q','2017-01-01','2017-06-30','2017-12-31','2017-06-01',None,[60,10,1])
    row('ForecastRevision','FY','2017-01-01','2017-12-31','2017-12-31','2017-07-01',None,[140,-60,-1])
    row('1QFinancialStatements_Consolidated_JP','1Q','2018-01-01','2018-03-31','2018-12-31','2018-05-01',[45,5,2],[170,70,5])
    e,o,c=build_events(pd.DataFrame(rows))
    assert e.iloc[0].ForecastNetSalesGrowth==.2 and e.iloc[0].ForecastOperatingProfitGrowth==1.5
    assert e.iloc[1].ForecastNetSalesGrowth==.3 and e.iloc[1].ForecastOperatingProfitGrowth==1.6
    assert 2 not in e.TriggerSourceRow.to_list()  # Half-year revision ignored.
    assert e.iloc[2].ForecastOperatingProfitGrowth==.4
    assert e.iloc[-1].ActualNetSalesGrowth==.5 and e.iloc[-1].ActualOperatingProfitGrowth==1.5
    assert e.iloc[-1].ActualEPS==2 and e.iloc[-1].ForecastEPS==5
    assert not e.iloc[-1].ForecastNetSalesGrowthValid  # No 2017 FY actual yet.
    # Fiscal-year shortening must override the old inferred later year end.
    shortened=pd.DataFrame(rows[:2]).copy()
    shortened.loc[1,'CurrentFiscalYearEndDate']=pd.Timestamp('2017-09-30')
    se,_,_=build_events(shortened)
    assert se.iloc[-1].ForecastKey[-1]==pd.Timestamp('2017-09-30')
    assert se.iloc[-1].ForecastEPS==4 and not se.iloc[-1].ForecastNetSalesGrowthValid
    return True

def main():
    assert synthetic()
    records=pd.read_pickle(ROOT/'financial_records.pkl').set_index('SourceRow');events=pd.read_pickle(ROOT/'financial_events.pkl')
    with (V8/'inputs.pkl').open('rb') as h:f,base_x,groups,cal=pickle.load(h)
    signals=pd.read_pickle(ROOT/'financial_signal_features.pkl')
    market=pd.DatetimeIndex(groups).difference(pd.DatetimeIndex(['2020-10-01'])).sort_values()
    # Independent explicit cutoff mapping checks every source row.
    for r in records.itertuples():
        eff=max(r.Date,r.DisclosedDate)+pd.to_timedelta(r.DisclosedTime)
        p=market.searchsorted(eff.normalize(),side='left')
        if p<len(market) and market[p]==eff.normalize() and eff.time()>=pd.Timestamp('15:00:00').time():p+=1
        assert p==r.AvailablePosition
        if p<len(market):assert market[p]==r.AvailableDate
        unix=pd.to_datetime(int(float(r.DisclosedUnixTime)),unit='s',utc=True).tz_convert('Asia/Tokyo').tz_localize(None)
        assert unix==r.DisclosedDate+pd.to_timedelta(r.DisclosedTime)
    max_error=0.;current_count=0;base_count=0;neg_to_pos=0;loss_shrink=0;zero_base=0
    for e in events.itertuples(index=False):
        trigger=records.loc[e.TriggerSourceRow];assert e.EffectiveTimestamp==trigger.EffectiveTimestamp
        ak=e.ActualKey;fk=e.ForecastKey
        if ak is not None:
            expected=ak[:3]+tuple(x-pd.DateOffset(years=1) for x in ak[3:]);assert expected==e.ActualPriorKey
        if fk is not None:
            start,end=fk[2]-pd.DateOffset(years=1),fk[3]-pd.DateOffset(years=1)
            assert e.ForecastBaseKey==(fk[0],fk[1],'FY',start,end,end)
        for k,name in enumerate(FINS):
            cv=getattr(e,name+'CurrentValue');bv=getattr(e,name+'BaseValue');ci=getattr(e,name+'CurrentRow');bi=getattr(e,name+'BaseRow')
            col=(ACTUAL+FORECAST)[k]
            if ci>=0:
                cur=records.loc[ci];assert cv==cur[col] or (pd.isna(cv) and pd.isna(cur[col]))
                assert cur.Code==e.Code and cur.EffectiveTimestamp<=e.EffectiveTimestamp;current_count+=1
                if k<3:
                    assert cur.TypeOfDocument!='ForecastRevision'
                    assert (cur.TypeOfCurrentPeriod,cur.CurrentFiscalYearStartDate,cur.CurrentPeriodEndDate,cur.CurrentFiscalYearEndDate)==ak[2:]
                elif cur.TypeOfDocument=='ForecastRevision':
                    assert cur.TypeOfCurrentPeriod=='FY' and (cur.CurrentFiscalYearStartDate,cur.CurrentFiscalYearEndDate)==fk[2:]
                elif cur.TypeOfCurrentPeriod=='FY':
                    assert (cur.CurrentFiscalYearEndDate+pd.Timedelta(days=1),cur.CurrentFiscalYearEndDate+pd.DateOffset(years=1))==fk[2:]
                else:assert (cur.CurrentFiscalYearStartDate,cur.CurrentFiscalYearEndDate)==fk[2:]
            if bi>=0:
                base=records.loc[bi];expectedkey=e.ActualPriorKey if k<3 else e.ForecastBaseKey
                assert (base.TypeOfCurrentPeriod,base.CurrentFiscalYearStartDate,base.CurrentPeriodEndDate,base.CurrentFiscalYearEndDate)==expectedkey[2:]
                assert base.Code==e.Code and base.EffectiveTimestamp<=e.EffectiveTimestamp and base.TypeOfDocument!='ForecastRevision'
                assert bv==base[ACTUAL[k%3]] or (pd.isna(bv) and pd.isna(base[ACTUAL[k%3]]));base_count+=1
            if k in [2,5]:valid=np.isfinite(cv);v=cv if valid else 0
            else:
                valid=np.isfinite(cv) and np.isfinite(bv) and bv!=0;v=(cv-bv)/abs(bv) if valid else 0
                if valid:neg_to_pos+=int(bv<0<cv);loss_shrink+=int(bv<cv<0)
                zero_base+=int(np.isfinite(cv) and bv==0)
            assert valid==getattr(e,name+'Valid');max_error=max(max_error,abs(v-getattr(e,name)))
    assert max_error==0
    ref=pd.merge_asof(f[['SignalDate','SecuritiesCode']].rename(columns={'SecuritiesCode':'Code'}).reset_index().sort_values('SignalDate'),
        events.loc[events.AvailableDate.notna()].sort_values(['AvailableDate','EventId']),by='Code',left_on='SignalDate',right_on='AvailableDate',direction='backward').sort_values('index')
    np.testing.assert_array_equal(signals.FinancialEventId,ref.EventId.fillna(-1).astype(int))
    for name in FINS:
        np.testing.assert_array_equal(signals[name],ref[name].fillna(0))
        np.testing.assert_array_equal(signals[name+'Valid'],ref[name+'Valid'].fillna(False).astype(bool))
    obs,x,_,_=load('six_financial');std=pd.read_pickle(V11/'return_std22_features.pkl')
    np.testing.assert_array_equal(x[:,:10],base_x[:,:10])
    np.testing.assert_array_equal(x[:,10:12],std[['PR1Scaled22','VR1Scaled22']].fillna(0))
    np.testing.assert_array_equal(x[:,12:],signals[FINS])
    assert x.shape[1]==18 and np.isfinite(x).all()
    # Real FY transition: 2017-04-03 2753 guidance is for FY2018, divided
    # by the simultaneously known FY2017 actual, not FY2016 or a 3Q value.
    ex=events.loc[events.Code.eq(2753)&events.EffectiveTimestamp.dt.date.eq(pd.Timestamp('2017-04-03').date())].iloc[-1]
    assert ex.ForecastKey[-1]==pd.Timestamp('2018-03-31')
    assert ex.ForecastNetSalesGrowthCurrentValue==32000000000 and ex.ForecastNetSalesGrowthBaseValue==30564000000
    example={'Code':2753,'AnnouncementDate':'2017-04-03','ForecastFiscalYearEnd':'2018-03-31','ActualBaseFiscalYearEnd':'2017-03-31',
        'ForecastNetSales':32000000000,'PriorFYActualNetSales':30564000000,'ForecastNetSalesGrowth':float(ex.ForecastNetSalesGrowth)}
    result={'passed':True,'source_rows_checked':len(records),'financial_events_checked':len(events),'signal_rows_checked':len(signals),
        'current_source_references_checked':current_count,'base_source_references_checked':base_count,'feature_arithmetic_max_error':max_error,
        'negative_base_to_positive_events':neg_to_pos,'negative_loss_shrink_events':loss_shrink,'zero_base_growth_cases':zero_base,
        'synthetic_period_and_signed_growth_checks':True,'real_forecast_year_transition':example,'same_price_volume_inputs_as_v11':True,
        'source_rows_known_at_event_time':True,'independent_asof_all_rows_passed':True}
    (ROOT/'feature_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
