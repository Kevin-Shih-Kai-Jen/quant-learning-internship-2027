from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'jpx_stock_returns_20260910'
RAW=['PR5','PR22','PR60','VR5','VR22','VR60','PR1']
FEATURES=['TPR5','TPR22','TPR60','TVR5','TVR22','TVR60','TPR5xTVR5','TPR22xTVR22','TPR60xTVR60','TPR1']
OLD=RAW[:6]+['PR5xVR5','PR22xVR22','PR60xVR60']

def prepare():
    f=pd.read_pickle(BASE/'features.pkl')
    p1=pd.read_pickle(ROOT.parent/'jpx_return_mean_20260911/price_return_1d.pkl')
    pd.testing.assert_frame_equal(f[['SecuritiesCode','SignalDate']],p1[['SecuritiesCode','SignalDate']])
    f['PR1']=p1.PR1
    closed=pd.to_datetime(json.loads((BASE/'feature_audit.json').read_text())['price']['market_wide_closures_excluded_from_lookbacks_only'])
    a=f.loc[~f.SignalDate.isin(closed)].copy()
    means=pd.DataFrame(np.nan,index=a.index,columns=RAW)
    stds=pd.DataFrame(np.nan,index=a.index,columns=RAW)
    for _,indices in a.groupby('SecuritiesCode').groups.items():
        vals=a.loc[indices,RAW].to_numpy(dtype=np.longdouble)
        if len(vals)<=22:continue
        windows=np.lib.stride_tricks.sliding_window_view(vals,22,axis=0)[:-1]
        means.loc[indices[22:],RAW]=np.mean(windows,axis=2,dtype=np.longdouble).astype(float)
        stds.loc[indices[22:],RAW]=np.std(windows,axis=2,ddof=1,dtype=np.longdouble).astype(float)
    audit={}
    for col in RAW:
        mean=means[col];std=stds[col]
        f['T'+col]=(a[col]-mean)/std.where(std.gt(0))
        f['Mean22_'+col]=mean;f['Std22_'+col]=std
        finite=f['T'+col].replace([np.inf,-np.inf],np.nan).dropna().abs()
        audit[col]={'zero_std_rows':int(std.eq(0).sum()),'missing_std_rows':int(std.isna().sum()),
            'min_positive_std':float(std.loc[std.gt(0)].min()),
            'absolute_T_quantiles':{str(q):float(x) for q,x in finite.quantile([.5,.99,.999,1]).items()}}
    cm=f.SignalDate.isin(closed)
    columns=['T'+c for c in RAW]+['Mean22_'+c for c in RAW]+['Std22_'+c for c in RAW]
    f.loc[cm,columns]=f.groupby('SecuritiesCode')[columns].shift(1).loc[cm]
    for k in (5,22,60):f[f'TPR{k}xTVR{k}']=f[f'TPR{k}']*f[f'TVR{k}']
    f['TEligible']=np.isfinite(f[FEATURES]).all(axis=1)&f.Close.gt(0)&f.Volume.gt(0)
    f.loc[cm,'TEligible']=f.groupby('SecuritiesCode').TEligible.shift(1,fill_value=False).loc[cm]
    keep=['SecuritiesCode','SignalDate','PreviousSignalDate','EntryDate','ExitDate','Target','TEligible','ReturnEligible']+FEATURES+OLD+['PR1']+['Mean22_'+c for c in RAW]+['Std22_'+c for c in RAW]
    f=f[keep]
    f.to_pickle(ROOT/'return_t_features.pkl')
    (ROOT/'feature_audit.json').write_text(json.dumps({'inputs':audit,'rows':len(f),'eligible':int(f.TEligible.sum()),
        'features':FEATURES,'window':22,'excludes_current':True,'std_ddof':1,'zero_std_policy':'undefined; omit candidate','clipping':None},indent=2,allow_nan=False))
    print('Return T features prepared',json.dumps(audit),flush=True)

if __name__=='__main__':prepare()
