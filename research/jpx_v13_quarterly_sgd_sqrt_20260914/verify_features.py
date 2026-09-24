import json,pickle
import numpy as np
import pandas as pd
from features import ROOT,V8,FINS,Builder,ACTUAL,FORECAST,event_vector,grow

def synthetic():
    rows=[]
    def add(year,k,date,actual,forecast,typ=None):
        fs=pd.Timestamp(year=year,month=1,day=1);fe=pd.Timestamp(year=year,month=12,day=31)
        row={'Code':1,'TypeOfDocument':typ or f'{"FY" if k==4 else str(k)+"Q"}FinancialStatements_Consolidated_JP',
            'Basis':'Consolidated_JP','TypeOfCurrentPeriod':'FY' if k==4 else str(k)+'Q',
            'CurrentFiscalYearStartDate':fs,'CurrentFiscalYearEndDate':fe,'CurrentPeriodEndDate':fs+pd.DateOffset(months=3*k)-pd.Timedelta(days=1),
            'SourceRow':len(rows),'DisclosureNumber':str(len(rows)),'EffectiveTimestamp':pd.Timestamp(date),'AvailableDate':pd.Timestamp(date),'AvailablePosition':len(rows)}
        for cols,val in [(ACTUAL,actual),(FORECAST,forecast)]:
            for m,col in enumerate(cols):row[col]=np.nan if val is None else val*[1,-1,.1][m];row[col+'Present']=val is not None
        rows.append(row)
    for k in range(1,4):add(2016,k,f'2016-{3*k+1:02d}-20',20*k,80)
    add(2016,4,'2017-01-20',80,100)
    add(2017,1,'2017-04-20',20,100)
    add(2017,2,'2017-07-20',40,100)
    add(2017,4,'2017-08-01',None,110,'ForecastRevision')
    add(2017,4,'2017-08-02',None,110,'ForecastRevision')
    add(2017,3,'2017-10-20',60,120)
    b=Builder();e,s,o=b.run(pd.DataFrame(rows))
    rev=e.loc[(e.TriggerSourceRow==6)&(e.Metric==0)&(e.Kind=='U')].iloc[0]
    assert rev.NewExpected==35 and rev.OldExpected==30 and rev.Base==20
    np.testing.assert_allclose(rev.Values,[.25])
    assert 7 not in e.TriggerSourceRow.to_list()
    act=e.loc[(e.TriggerSourceRow==8)&(e.Metric==0)&(e.Kind=='A')].iloc[0]
    assert act.Expected==35 and act.Actual==20 and act.Base==20
    # A release must use the old expectation; same-release guidance concerns Q4.
    assert act.ForecastTime==pd.Timestamp('2017-08-02')
    assert e.loc[(e.TriggerSourceRow==8)&(e.Kind=='U')].iloc[0].KnownQuarters==3
    # Future information cannot alter earlier events.
    early=Builder().run(pd.DataFrame(rows[:-1]))[0]
    assert e.iloc[:len(early)].Values.to_list()==early.Values.to_list()
    # Corrections patch known state; no re-emission of old actual surprise.
    corrected=[dict(x) for x in rows]
    row=dict(rows[5]);row.update(TypeOfDocument='NumericalCorrection',SourceRow=9,DisclosureNumber='9',
        EffectiveTimestamp=pd.Timestamp('2017-11-01'),AvailableDate=pd.Timestamp('2017-11-01'),AvailablePosition=9)
    row['NetSales']=42.;row['ForecastNetSalesPresent']=False;row['ForecastOperatingProfitPresent']=False;row['ForecastEarningsPerSharePresent']=False
    corrected.append(row);ce,_,_=Builder().run(pd.DataFrame(corrected))
    assert not ((ce.TriggerSourceRow==9)&(ce.Kind=='A')).any()
    assert grow(50,-100)==1.5 and grow(-60,-100)==.4 and np.isnan(grow(1,0))
    return {'remaining_quarter_example':{'old':30,'new':35,'fixed_base':20,'revision':.25},'synthetic_passed':True}

def main():
    result=synthetic()
    r=pd.read_pickle(ROOT/'financial_records.pkl').set_index('SourceRow')
    ev=pd.read_pickle(ROOT/'financial_events.pkl');snap=pd.read_pickle(ROOT/'quarter_forecasts.pkl')
    with (V8/'inputs.pkl').open('rb') as h:f,x,groups,cal=pickle.load(h)
    market=pd.DatetimeIndex(groups).difference(pd.DatetimeIndex(['2020-10-01'])).sort_values()
    # Independently verify source availability at a strict post-disclosure close.
    for row in r.itertuples():
        ts=max(row.Date,row.DisclosedDate)+pd.to_timedelta(row.DisclosedTime)
        day=market.searchsorted(ts.normalize())
        if day<len(market) and market[day]==ts.normalize() and ts.hour>=15:day+=1
        assert day==row.AvailablePosition
    time=r.EffectiveTimestamp.to_dict();code=r.Code.to_dict();position=r.AvailablePosition.to_dict()
    max_error=0.;refs=0
    for e in ev.itertuples(index=False):
        assert e.AvailablePosition==position[e.TriggerSourceRow]
        assert e.EffectiveTimestamp==time[e.TriggerSourceRow]
        for source in e.References:
            assert time[source]<=e.EffectiveTimestamp and code[source]==e.Code;refs+=1
        if e.Kind=='A':
            assert e.ForecastTime<e.EffectiveTimestamp
            expected=[grow(e.Actual,e.Base),grow(e.Expected,e.Base)]
        elif e.Kind in ['Q','Y']:expected=[grow(e.Expected,e.Base)]
        else:
            assert 0<=e.KnownQuarters<4
            assert e.NewExpected==(e.NewAnnual-e.Cumulative)/(4-e.KnownQuarters)
            assert e.OldExpected==(e.OldAnnual-e.Cumulative)/(4-e.KnownQuarters)
            expected=[grow(e.NewExpected,e.Base)-grow(e.OldExpected,e.Base)]
        max_error=max(max_error,float(np.max(abs(np.asarray(expected)-e.Values))))
    assert max_error==0
    for s in snap.itertuples(index=False):
        np.testing.assert_allclose(s.Values,(s.Annual-s.Cumulative)/(4-s.KnownQuarters),equal_nan=True)
    for kind in ['A','Q','Y']:
        subset=ev.loc[ev.Kind.eq(kind)]
        assert not subset.duplicated(['Key','Metric']).any()
    # Different algorithm: scatter event impulses to market dates, then run
    # a fixed one-session recurrence across EVERY market date for each company.
    actual=pd.read_pickle(ROOT/'financial_signal_features.pkl');maximum=0.
    evgroups={c:g for c,g in ev.groupby('Code')};checked=0
    for c,ix in f.groupby('SecuritiesCode').indices.items():
        impulses=np.zeros((len(market),15))
        if c in evgroups:
            for e in evgroups[c].itertuples(index=False):
                if e.AvailablePosition<len(market):impulses[e.AvailablePosition]+=event_vector(e)
        for p in range(1,len(market)):impulses[p]+=impulses[p-1]*np.exp(-1/9)
        positions=market.searchsorted(f.iloc[ix].SignalDate)
        expected=impulses[np.minimum(positions,len(market)-1)]
        observed=actual.iloc[ix][FINS].to_numpy()
        np.testing.assert_allclose(observed,expected,rtol=1e-11,atol=1e-10)
        maximum=max(maximum,float(abs(observed-expected).max()));checked+=len(ix)
    result.update(passed=True,source_rows=len(r),events=len(ev),reference_checks=refs,quarter_snapshots=len(snap),
        arithmetic_max_error=max_error,all_signal_rows_checked=checked,decay_recurrence_max_error=maximum,
        unique_actual_and_initial_forecast_events=True,strict_preannouncement_forecasts=True)
    (ROOT/'feature_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
