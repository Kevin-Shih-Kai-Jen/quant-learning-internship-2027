import pickle,json
import numpy as np
import pandas as pd
from financial_features import ROOT,V8

def main():
    with (V8/'inputs.pkl').open('rb') as h:f,_,groups,validation=pickle.load(h)
    cal=pd.DatetimeIndex(groups).difference(pd.DatetimeIndex(['2020-10-01'])).sort_values()
    records=pd.read_pickle(ROOT/'financial_records.pkl');events=pd.read_pickle(ROOT/'financial_events.pkl');saved=pd.read_pickle(ROOT/'financial_signal_features.pkl')
    cutoffs=(cal+pd.Timedelta(hours=15)).to_numpy(dtype='datetime64[ns]')
    expected_positions=np.searchsorted(cutoffs,records.EffectiveTimestamp.to_numpy(dtype='datetime64[ns]'),side='right')
    np.testing.assert_array_equal(expected_positions,records.AvailablePosition)
    # Verify that the original timestamp fields are JST, independently of the
    # availability mapping and the separate dataset Date.
    utc=pd.to_datetime(pd.to_numeric(records.DisclosedUnixTime),unit='s',utc=True)
    jst=utc.dt.tz_convert('Asia/Tokyo').dt.tz_localize(None)
    original=records.DisclosedDate+pd.to_timedelta(records.DisclosedTime)
    assert jst.eq(original).all()
    indexed=records.set_index('SourceRow')
    for e in events.itertuples(index=False):
        trigger=indexed.loc[e.TriggerSourceRow];current=indexed.loc[e.CurrentSourceRow]
        assert current.EffectiveTimestamp<=trigger.EffectiveTimestamp
        assert current.Code==e.SecuritiesCode
        assert current.CurrentPeriodEndDate==e.CurrentPeriodEndDate and current.CurrentFiscalYearStartDate==e.CurrentFiscalYearStartDate
        if e.PriorSourceRow<0:
            assert not e.ComparisonAvailable;continue
        prior=indexed.loc[e.PriorSourceRow]
        assert prior.Code==e.SecuritiesCode and prior.EffectiveTimestamp<=trigger.EffectiveTimestamp
        for column in ['CurrentFiscalYearStartDate','CurrentPeriodEndDate','CurrentFiscalYearEndDate']:
            assert prior[column]==current[column]-pd.DateOffset(years=1)
        assert prior.TypeOfCurrentPeriod==current.TypeOfCurrentPeriod
        if current.TypeOfDocument!='NumericalCorrection':assert current.Basis==e.Basis
        if prior.TypeOfDocument!='NumericalCorrection':assert prior.Basis==e.Basis
        valid=np.isfinite([current.NetSales,current.OperatingProfit,prior.NetSales,prior.OperatingProfit]).all() and current.NetSales>0 and prior.NetSales>0
        assert valid==e.ComparisonAvailable
        if valid:
            actual=current.OperatingProfit/current.NetSales-prior.OperatingProfit/prior.NetSales
            np.testing.assert_allclose(actual,e.MarginYoYChange,rtol=0,atol=1e-13)
    # Use an independent global as-of join to reproduce every per-stock feature.
    left=f[['SignalDate','SecuritiesCode']].copy();left['OriginalIndex']=np.arange(len(left));left.SignalDate=left.SignalDate.astype('datetime64[ns]')
    right=events.loc[events.AvailableDate.notna(),['SecuritiesCode','AvailableDate','AvailablePosition','EventId','ComparisonAvailable','MarginYoYChange']].copy()
    right.AvailableDate=right.AvailableDate.astype('datetime64[ns]')
    right=right.sort_values(['AvailableDate','EventId'],kind='stable').drop_duplicates(['SecuritiesCode','AvailableDate'],keep='last')
    joined=pd.merge_asof(left.sort_values('SignalDate',kind='stable'),right.sort_values('AvailableDate',kind='stable'),left_on='SignalDate',right_on='AvailableDate',by='SecuritiesCode',direction='backward').sort_values('OriginalIndex')
    has=joined.EventId.notna();valid=joined.ComparisonAvailable.eq(True)
    positions=np.searchsorted(cal.to_numpy(dtype='datetime64[ns]'),joined.SignalDate.to_numpy(),side='right')-1
    age=np.where(has,positions-joined.AvailablePosition.fillna(0).to_numpy(),-1)
    r=np.where(has,np.maximum(1-age/10,0),0)
    expected=np.where(valid&(age<10)&has,joined.MarginYoYChange.fillna(0)*r,0)
    np.testing.assert_array_equal(saved.FinancialEventId,joined.EventId.fillna(-1).astype(int))
    np.testing.assert_array_equal(saved.FinancialAge,age)
    np.testing.assert_allclose(saved.DecayWeight,r,atol=1e-15,rtol=0)
    np.testing.assert_allclose(saved.FinancialFeature,expected,atol=1e-13,rtol=0)
    assert saved.loc[saved.FinancialAge.ge(10),'FinancialFeature'].eq(0).all()
    example=joined.loc[joined.SecuritiesCode.eq(2753)&joined.SignalDate.eq(pd.Timestamp('2018-01-04'))].iloc[0]
    expected_example=2251000000/23555000000-2147000000/22761000000
    np.testing.assert_allclose(saved.iloc[int(example.OriginalIndex)].FinancialFeature,expected_example,atol=1e-14)
    result={'passed':True,'financial_events_audited':len(events),'all_signal_rows_independently_reproduced':len(saved),
        'source_announcement_timestamps_match_jst_unix_time':True,'strict_1500_cutoff_verified':True,'current_and_prior_components_known_asof_event':True,
        'same_fiscal_period_and_accounting_basis_verified':True,'all_features_expire_after_tenth_session':True,
        'example_2753_2018_01_04_financial_feature':float(expected_example)}
    (ROOT/'feature_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
