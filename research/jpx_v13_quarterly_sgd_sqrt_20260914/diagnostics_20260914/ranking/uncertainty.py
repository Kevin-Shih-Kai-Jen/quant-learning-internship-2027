"""Exploratory paired circular moving-block bootstrap of fixed daily outputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
OUT=Path(__file__).resolve().parent
d=pd.read_csv(OUT/'daily_metrics.csv')
a=d[d.Variant.eq('sgd_only')].sort_values('Date').reset_index(drop=True)
b=d[d.Variant.eq('sgd_sqrt')].sort_values('Date').reset_index(drop=True)
assert (a.Date==b.Date).all()
n=len(a);rng=np.random.default_rng(20260914);length=20;reps=5000
indices=(rng.integers(0,n,size=(reps,int(np.ceil(n/length)),1))+np.arange(length)[None,None,:])%n
indices=indices.reshape(reps,-1)[:,:n]
out={'method':'Paired circular moving-block bootstrap; resample same date blocks for both variants','block_length':length,'replicates':reps,'seed':20260914,'warning':'Exploratory uncertainty conditional on this reused validation sample; not an untouched holdout and not proof of stationarity.'}
for col in ['RankIC','LongShortReturn']:
    x=a[col].to_numpy();y=b[col].to_numpy()
    for name,v in [('sgd_only',x),('sgd_sqrt',y),('sqrt_minus_only',y-x)]:
        means=np.nanmean(v[indices],axis=1)
        out[col+'_'+name]={'mean':float(np.nanmean(v)),'finite_days':int(np.isfinite(v).sum()),'95pct_percentile_CI':np.quantile(means,[.025,.975]).tolist()}
sx=x[indices].mean(axis=1)/x[indices].std(axis=1,ddof=1)
sy=y[indices].mean(axis=1)/y[indices].std(axis=1,ddof=1)
out['Sharpe_sqrt_minus_only']={'difference':float(y.mean()/y.std(ddof=1)-x.mean()/x.std(ddof=1)),'95pct_percentile_CI':np.quantile(sy-sx,[.025,.975]).tolist()}
(OUT/'uncertainty.json').write_text(json.dumps(out,indent=2,ensure_ascii=False))
diffs=pd.DataFrame({'Date':a.Date,'ValidationYear':a.ValidationYear,'LSDifference':b.LongShortReturn-a.LongShortReturn})
byyear=diffs.groupby('ValidationYear').agg(Days=('Date','size'),MeanDailyLSDifference=('LSDifference','mean'),SumLSDifference=('LSDifference','sum'))
byyear['ContributionToTotalMeanDifference']=byyear.SumLSDifference/n
byyear['FractionOfNetMeanDifference']=byyear.SumLSDifference/byyear.SumLSDifference.sum()
byyear.to_csv(OUT/'annual_difference_contribution.csv')
print(json.dumps(out,indent=2));print(byyear.to_string())
