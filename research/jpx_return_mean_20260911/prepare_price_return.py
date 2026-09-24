"""Prepare the explicitly requested one-session price return without changing the active model."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'jpx_stock_returns_20260910'
f=pd.read_pickle(BASE/'features.pkl')
audit=json.loads((BASE/'feature_audit.json').read_text())
closed=pd.to_datetime(audit['price']['market_wide_closures_excluded_from_lookbacks_only'])
a=f.loc[~f.SignalDate.isin(closed),['SecuritiesCode','SignalDate','Close','CumulativeFactor']].copy()
a['Price']=a.Close/a.CumulativeFactor
prior=a.groupby('SecuritiesCode').Price.shift(1)
a['PR1']=a.Price/prior.where(prior>0)-1
out=f[['SecuritiesCode','SignalDate']].copy()
out['PR1']=a.PR1
cm=out.SignalDate.isin(closed)
out.loc[cm,'PR1']=out.groupby('SecuritiesCode').PR1.shift(1).loc[cm]
check=a.groupby('SecuritiesCode').agg(first=('PR1','first'))
assert np.isfinite(out.PR1.dropna()).all()
out.to_pickle(ROOT/'price_return_1d.pkl')
info={'formula':'adjusted_price[t] / adjusted_price[t-1] - 1',
      'market_calendar':'same as existing validated features; market-wide closure carries prior feature',
      'rows':len(out),'finite_rows':int(out.PR1.notna().sum()),'active_model_changed':False}
(ROOT/'price_return_preparation.json').write_text(json.dumps(info,indent=2))
print(json.dumps(info))
