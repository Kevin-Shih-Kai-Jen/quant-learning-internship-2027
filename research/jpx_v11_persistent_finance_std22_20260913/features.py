from pathlib import Path
import json,pickle,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
V8=ROOT.parent/'jpx_v8_soft_rank_20260912'
V9=ROOT.parent/'jpx_v9_financial_decay_sgd_20260912'
V10=ROOT.parent/'jpx_v10_mse_sgd_financial_20260913'
VARIANTS=['persistent_raw','decay_std22','persistent_std22']

def build():
    with (V8/'inputs.pkl').open('rb') as h:f,x,groups,validation=pickle.load(h)
    raw=pd.read_pickle(ROOT.parent/'jpx_v5_daily_returns_20260912/one_day_returns.pkl')
    keys=['SignalDate','SecuritiesCode'];calendar=pd.DatetimeIndex(groups).difference(pd.DatetimeIndex(['2020-10-01'])).sort_values()
    raw=raw.loc[raw.SignalDate.isin(calendar)].copy()
    frames=[]
    for code,g in raw.groupby('SecuritiesCode',sort=True):
        g=g.set_index('SignalDate').reindex(calendar)
        frame=pd.DataFrame({'SignalDate':calendar,'SecuritiesCode':code})
        for col in ['PR1','VR1']:
            a=g[col].to_numpy();a=np.where(np.isfinite(a),a,np.nan)
            std=np.full(len(a),np.nan)
            windows=np.lib.stride_tricks.sliding_window_view(a,22)
            std[21:]=np.std(windows,axis=1,ddof=1)
            frame[col+'Std22']=std
            usable=np.isfinite(a)&np.isfinite(std)&(std>0)
            scaled=np.full(len(a),np.nan);scaled[usable]=a[usable]/std[usable]
            frame[col+'Scaled22']=scaled
        frames.append(frame)
    allstd=pd.concat(frames,ignore_index=True)
    # Closure dates are never forecast; store only past-known values there.
    closure=f.loc[f.SignalDate.eq(pd.Timestamp('2020-10-01')),keys].copy()
    prior=allstd.loc[allstd.SignalDate.eq(pd.Timestamp('2020-09-30'))].drop(columns='SignalDate')
    closure=closure.merge(prior,on='SecuritiesCode',how='left',validate='one_to_one')
    allstd=pd.concat([allstd,closure],ignore_index=True)
    standard=f[keys].merge(allstd,on=keys,how='left',validate='one_to_one').drop(columns=keys)
    assert len(standard)==len(f)
    standard.to_pickle(ROOT/'return_std22_features.pkl')
    finance=pd.read_pickle(V9/'financial_signal_features.pkl');events=pd.read_pickle(V9/'financial_events.pkl').set_index('EventId')
    persistent=finance.copy();has=finance.FinancialEventId.ge(0)
    vals=finance.loc[has,'FinancialEventId'].map(events.MarginYoYChange).fillna(0)
    valid=finance.loc[has,'FinancialEventId'].map(events.ComparisonAvailable).to_numpy()
    persistent['FinancialFeature']=0.;persistent.loc[has,'FinancialFeature']=np.where(valid,vals,0.)
    persistent['DecayWeight']=0.;persistent.loc[has,'DecayWeight']=1.
    persistent['FinancialStatus']=np.where(~has,0,np.where(finance.FinancialStatus.eq(1),1,2)).astype(np.int8)
    persistent.to_pickle(ROOT/'financial_persistent_features.pkl')
    stats={}
    val=f.SignalDate.isin(validation.Date)
    for col in ['PR1','VR1']:
        z=standard[col+'Scaled22'];sd=standard[col+'Std22']
        stats[col]={'all_rows':len(z),'finite_scaled_rows':int(z.notna().sum()),'validation_scaled_missing_rows':int((val&z.isna()).sum()),
            'validation_zero_std_rows':int((val&sd.eq(0)).sum()),'scaled_min':float(z.min()),'scaled_max':float(z.max()),
            'first_valid_date':str(f.loc[z.notna(),'SignalDate'].min().date())}
    summary={'return_standardization':stats,'validation_rows':int(val.sum()),
        'validation_nonzero_finance_decaying':int((val&finance.FinancialFeature.ne(0)).sum()),
        'validation_nonzero_finance_persistent':int((val&persistent.FinancialFeature.ne(0)).sum()),
        'std_window':22,'std_ddof':1,'includes_signal_day':True,'demeaned':False,'financial_event_source':str(V9/'financial_events.pkl')}
    (ROOT/'feature_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

def load(variant):
    assert variant in VARIANTS
    with (V8/'inputs.pkl').open('rb') as h:f,base_x,groups,calendar=pickle.load(h)
    finance=pd.read_pickle(V9/'financial_signal_features.pkl' if variant=='decay_std22' else ROOT/'financial_persistent_features.pkl')
    if variant.endswith('std22'):
        f=f.copy();std=pd.read_pickle(ROOT/'return_std22_features.pkl')
        for col in ['PR1','VR1']:f[col]=std[col+'Scaled22'].to_numpy()
        # Original x columns alpha + 11 economic features: PR1 and VR1 last.
        base_x=base_x.copy();base_x[:,-2:]=f[['PR1','VR1']].fillna(0).to_numpy()
    return f,base_x,groups,calendar,finance

if __name__=='__main__':build()
