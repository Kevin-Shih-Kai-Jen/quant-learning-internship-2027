from pathlib import Path
import json,statistics
import pandas as pd
import numpy as np
from feature_builder import RAW,FEATURES
ROOT=Path(__file__).resolve().parent
f=pd.read_pickle(ROOT/'return_t_features.pkl')
checks=0;zero_checks=0
for code,g in f.loc[f.SecuritiesCode.mod(53).eq(0)|f.SecuritiesCode.eq(7638)].groupby('SecuritiesCode'):
    a=g.loc[~g.SignalDate.eq('2020-10-01')].sort_values('SignalDate').reset_index(drop=True)
    positions=set(np.linspace(82,len(a)-1,20,dtype=int)) if len(a)>82 else set()
    if code==7638:positions.add(int(a.index[a.SignalDate.eq('2018-11-05')][0]))
    for j in sorted(positions):
        for col in RAW:
            vals=a.loc[j-22:j-1,col].to_numpy()
            if not np.isfinite(vals).all():
                assert pd.isna(a.loc[j,'Std22_'+col]);continue
            mean=statistics.mean(vals);std=statistics.stdev(vals)
            np.testing.assert_allclose(a.loc[j,'Mean22_'+col],mean,rtol=1e-12,atol=1e-16)
            np.testing.assert_allclose(a.loc[j,'Std22_'+col],std,rtol=1e-12,atol=1e-16)
            expected=(a.loc[j,col]-mean)/std if std>0 else np.nan
            np.testing.assert_allclose(a.loc[j,'T'+col],expected,rtol=1e-10,atol=1e-9,equal_nan=True)
            checks+=1;zero_checks+=int(std==0)
for k in (5,22,60):
    np.testing.assert_allclose(f[f'TPR{k}xTVR{k}'],f[f'TPR{k}']*f[f'TVR{k}'],equal_nan=True)
for col in RAW:
    zero=f['Std22_'+col].eq(0)
    assert f.loc[zero,'T'+col].isna().all()
    assert not f.loc[zero,'TEligible'].any()
assert f.loc[f.TEligible,'ReturnEligible'].all()
for _,g in f.groupby('SecuritiesCode'):
    g=g.sort_values('SignalDate').reset_index(drop=True)
    ix=g.index[g.SignalDate.eq('2020-10-01')]
    if len(ix) and ix[0]>0:
        i=ix[0]
        np.testing.assert_allclose(g.loc[i,FEATURES].astype(float),g.loc[i-1,FEATURES].astype(float),equal_nan=True)
orders=pd.read_csv(ROOT/'requested_orders.csv',parse_dates=['SignalDate','EntryDate','ExitDate'])
signals=pd.read_csv(ROOT.parent/'jpx_signed_band_20260910/market_signals.csv',parse_dates=['SignalDate','EntryDate','ExitDate'])
d=pd.read_csv(ROOT/'daily_returns.csv',parse_dates=['SignalDate','EntryDate','ExitDate'])
o=orders.merge(signals[['SignalDate','Regime']],on='SignalDate',validate='many_to_one')
long=np.where(o.Regime.eq(1),.7,np.where(o.Regime.eq(-1),.3,.5))
budget=np.where(o.SourceSide.eq('long'),long,1-long)
np.testing.assert_allclose(o.Weight.abs(),budget*o.SourceRank.map({1:.5,2:.3,3:.2}),atol=1e-14)
pd.testing.assert_frame_equal(d[['SignalDate','EntryDate','ExitDate','ValidationYear']],signals[['SignalDate','EntryDate','ExitDate','ValidationYear']],check_dtype=False)
fits=json.loads((ROOT/'model_fits.json').read_text())
for fit in fits:
    year=fit['validation_year'];cutoff=pd.Timestamp(fit['fit_asof'])
    mask=f.SignalDate.dt.year.eq(year-1)&f.ExitDate.le(cutoff)&f.Target.notna()&f.TEligible
    assert mask.sum()==fit['train_rows']
    assert f.loc[mask,'ExitDate'].max()<=cutoff
    v=f.loc[f.SignalDate.isin(signals.loc[signals.ValidationYear.eq(year),'SignalDate'])&f.TEligible]
    assert len(v)==fit['validation_rows']
    selected=orders.loc[orders.ValidationYear.eq(year)].merge(v,on=['SignalDate','SecuritiesCode'],suffixes=('','_feature'),validate='one_to_one')
    assert len(selected)==len(orders.loc[orders.ValidationYear.eq(year)])
    coeff=fit['coefficients']
    x=np.column_stack([np.ones(len(selected)),selected[list(coeff)[1:]].to_numpy()])
    np.testing.assert_allclose(selected.Prediction,x@np.array(list(coeff.values())),atol=1e-10,rtol=1e-9)
audit={'independent_mean_std_T_checks':checks,'zero_std_sample_checks':zero_checks,
    'all_zero_std_rows_excluded':True,'products_checked':True,'market_closure_features_carried':True,
    'previous_year_only_and_label_cutoffs_checked':True,'same_market_dates_and_budgets':True,
    'selected_predictions_recomputed':True,'all_passed':True}
(ROOT/'independent_audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(audit))
