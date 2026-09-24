import json,pickle,hashlib
import numpy as np
import pandas as pd
from features import ROOT,V8,V9,V10,load,VARIANTS

def main():
    with (V8/'inputs.pkl').open('rb') as h:f,original_x,groups,validation=pickle.load(h)
    std=pd.read_pickle(ROOT/'return_std22_features.pkl');persistent=pd.read_pickle(ROOT/'financial_persistent_features.pkl')
    decay=pd.read_pickle(V9/'financial_signal_features.pkl');events=pd.read_pickle(V9/'financial_events.pkl')
    raw=pd.read_pickle(ROOT.parent/'jpx_v5_daily_returns_20260912/one_day_returns.pkl')
    calendar=pd.DatetimeIndex(groups).difference(pd.DatetimeIndex(['2020-10-01'])).sort_values()
    keys=['SignalDate','SecuritiesCode']
    # Full global as-of join is independent of the cached per-stock EventId map.
    source=events.loc[events.AvailableDate.notna()].sort_values(['AvailableDate','EventId'])
    ref=pd.merge_asof(f[keys].reset_index().sort_values('SignalDate'),source,
        left_on='SignalDate',right_on='AvailableDate',by='SecuritiesCode',direction='backward').sort_values('index')
    np.testing.assert_array_equal(persistent.FinancialEventId,ref.EventId.fillna(-1).astype(int))
    valid=ref.ComparisonAvailable.fillna(False).astype(bool);expected=ref.MarginYoYChange.where(valid,0.).to_numpy()
    np.testing.assert_array_equal(persistent.FinancialFeature,expected)
    assert (ref.loc[valid,'AvailableDate']<=ref.loc[valid,'SignalDate']).all()
    assert np.isfinite(persistent.FinancialFeature).all()
    np.testing.assert_array_equal(persistent.FinancialAge,decay.FinancialAge)
    np.testing.assert_array_equal(persistent.DecayWeight,ref.EventId.notna().astype(float))
    expired=valid.to_numpy()&persistent.FinancialAge.ge(10).to_numpy()
    assert (persistent.loc[expired,'FinancialStatus']==2).all()
    assert (decay.loc[expired,'FinancialFeature']==0).all()
    invalid=~valid.to_numpy();assert persistent.loc[invalid,'FinancialFeature'].eq(0).all()
    max_std_error=0.;max_scaled_error=0.;count=0;prefix_samples=0;examples=[]
    # Independently reconstruct each raw daily return from original price/volume
    # inputs, then compare every 22-session variance with centered sums.
    prices=pd.read_pickle(ROOT.parent/'jpx_stock_returns_20260910/features.pkl')
    prices=prices.loc[prices.SignalDate.isin(calendar)].sort_values(['SecuritiesCode','SignalDate'])
    raw=raw.loc[raw.SignalDate.isin(calendar)].sort_values(['SecuritiesCode','SignalDate'])
    prices=prices.set_index(keys).loc[pd.MultiIndex.from_frame(raw[keys])].reset_index()
    for col,level in [('PR1',prices.Close/prices.CumulativeFactor),('VR1',prices.Volume*prices.CumulativeFactor)]:
        prev=level.groupby(prices.SecuritiesCode).shift(1)
        derived=(level/prev.where(prev>0)-1).where(lambda s:np.isfinite(s))
        np.testing.assert_array_equal(derived.to_numpy(),raw[col].to_numpy())
    fgroups=f.groupby('SecuritiesCode').indices
    for code,g in raw.groupby('SecuritiesCode',sort=True):
        if code not in fgroups:continue
        g=g.set_index('SignalDate').reindex(calendar);ix=fgroups[code]
        dates=f.iloc[ix].SignalDate
        positions=calendar.get_indexer(dates.where(dates.ne(pd.Timestamp('2020-10-01')),pd.Timestamp('2020-09-30')))
        assert (positions>=0).all()
        for col in ['PR1','VR1']:
            values=g[col].to_numpy();mat=np.lib.stride_tricks.sliding_window_view(values,22)
            # Use a translation before centering; identical variance, different
            # arithmetic path from direct np.std used in production.
            shifted=mat-mat[:,:1]
            mean=np.sum(shifted,axis=1)/22
            var=np.sum((shifted-mean[:,None])**2,axis=1)/21
            expected_sd=np.r_[np.full(21,np.nan),np.sqrt(var)]
            sd=std.iloc[ix][col+'Std22'].to_numpy();es=expected_sd[positions]
            np.testing.assert_allclose(sd,es,atol=1e-11,rtol=1e-11,equal_nan=True)
            mask=np.isfinite(sd)&np.isfinite(es)
            if mask.any():max_std_error=max(max_std_error,float(max(abs(sd[mask]-es[mask]))))
            usable=np.isfinite(values)&np.isfinite(expected_sd)&(expected_sd>0)
            scaled=np.full(len(values),np.nan);scaled[usable]=values[usable]/expected_sd[usable]
            actual=std.iloc[ix][col+'Scaled22'].to_numpy();expected=scaled[positions]
            np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=1e-10,equal_nan=True)
            finite=np.isfinite(actual)&np.isfinite(expected)
            if finite.any():max_scaled_error=max(max_scaled_error,float(max(abs(actual[finite]-expected[finite]))))
            # Selected scalar windows check exactly 22 past/current sessions.
            for p in [22,len(calendar)//2,len(calendar)-1]:
                window=values[p-21:p+1]
                check=np.std(window,ddof=1)
                np.testing.assert_allclose(expected_sd[p],check,atol=1e-11,rtol=1e-11,equal_nan=True)
                prefix_samples+=1
            if code==1301 and col=='PR1':
                p=int(np.flatnonzero(np.isfinite(scaled))[0]);examples.append({'Code':code,'Date':str(calendar[p].date()),'WindowStart':str(calendar[p-21].date()),'RawReturn':float(values[p]),'Std22':float(expected_sd[p]),'ScaledReturn':float(scaled[p])})
            count+=len(ix)
    # Price/volume fallback and feature-only x are identical across training and
    # prediction builders; no other input is changed by standardization.
    import sys
    sys.path.insert(0,str(ROOT.parent/'jpx_v7_daily_mse_20260912'))
    from run_v7 import make_x
    for variant in VARIANTS:
        obs,x,_,_,fin=load(variant)
        np.testing.assert_array_equal(x,make_x(obs))
        np.testing.assert_array_equal(x[:,:-2],original_x[:,:-2])
        np.testing.assert_array_equal(obs.Target,f.Target)
        np.testing.assert_array_equal(obs.ExitDate,f.ExitDate)
        if variant=='persistent_raw':np.testing.assert_array_equal(x,original_x)
    for name in ['mse.cpp','mse.dylib','native_mse.py']:
        assert (ROOT/name).read_bytes()==(V10/name).read_bytes()
    result={'passed':True,'audited_financial_rows':len(f),'persistent_expired_feature_rows':int(expired.sum()),
        'audited_return_feature_cells':count,'scalar_past_window_checks':prefix_samples,'max_std22_error':max_std_error,
        'max_scaled_return_error':max_scaled_error,'examples':examples,'source_returns_rebuilt_from_adjusted_price_and_volume':True,
        'same_mse_sgd_kernel_as_v10':True,'same_nonreturn_features_and_labels':True,'source_v9_feature_audit_passed':json.loads((V9/'feature_audit.json').read_text())['passed']}
    (ROOT/'feature_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
