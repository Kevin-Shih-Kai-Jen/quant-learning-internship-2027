from pathlib import Path
import numpy as np,pandas as pd,json
ROOT=Path(__file__).resolve().parent;V=ROOT.parent
pairs=[('eps_joint_minus_qoq_same_mask','eps_actual_joint','eps_qoq_joint_mask'),('sales_revision_minus_control','sales_revision_relative','sales_revision_relative_control'),('profit_revision_minus_control','profit_revision_relative','profit_revision_relative_control'),('eps_revision_minus_control','eps_revision_relative','eps_revision_relative_control')]
rng=np.random.default_rng(20260915);B=4000;L=20;n=953;starts=rng.integers(0,n-L+1,size=(B,(n+L-1)//L));indices=(starts[:,:,None]+np.arange(L)[None,None,:]).reshape(B,-1)[:,:n]
rows=[]
for label,a,b in pairs:
    ad=pd.read_csv(V/a/'daily_metrics.csv');bd=pd.read_csv(V/b/'daily_metrics.csv');assert ad.Date.equals(bd.Date)
    x=ad.OfficialDailySpread.to_numpy();y=bd.OfficialDailySpread.to_numpy();assert np.isfinite(x).all() and np.isfinite(y).all()
    observed=float(x.mean()/x.std(ddof=1)-y.mean()/y.std(ddof=1));xx=x[indices];yy=y[indices];delta=xx.mean(axis=1)/xx.std(axis=1,ddof=1)-yy.mean(axis=1)/yy.std(axis=1,ddof=1);lo,hi=np.quantile(delta,[.025,.975])
    rows.append({'Comparison':label,'ObservedDeltaSharpe':observed,'Lower95':float(lo),'Upper95':float(hi),'Replicates':B,'BlockTradingDays':L,'ExploratoryNotMultipleTestingAdjusted':True})
pd.DataFrame(rows).to_csv(ROOT/'paired_block_bootstrap.csv',index=False)
(ROOT/'bootstrap.json').write_text(json.dumps(rows,indent=2))
print(pd.DataFrame(rows).to_string(index=False))
