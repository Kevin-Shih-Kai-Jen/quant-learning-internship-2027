"""Validate completed annual outputs while other annual fits continue; never tune."""
from pathlib import Path
import sys,json,warnings
import numpy as np
import pandas as pd
from run_arima515 import RUN,BASE,daily_metrics,stats

year=int(sys.argv[1])
data=np.load(RUN/'prices.npz');codes=data['codes']
cal=pd.read_csv(RUN/'calendar.csv',parse_dates=['Date'])
dates=cal.loc[cal.ValidationYear.eq(year),'Date'].to_numpy()
paths=[RUN/'jobs'/f'{year}_{code}.json' for code in codes]
if not all(p.exists() for p in paths):
    print('WAIT',year,sum(p.exists() for p in paths),len(paths));sys.exit(0)
pieces=[]
for code in codes:
    p=np.load(RUN/'jobs'/f'{year}_{code}.npz')
    pieces.append(pd.DataFrame({'Date':dates,'SecuritiesCode':code,'Score':p['score'],'Fallback':p['fallback']}))
labels=pd.read_pickle(RUN/'labels.pkl')
frame=labels.loc[labels.ValidationYear.eq(year)].merge(pd.concat(pieces),on=['Date','SecuritiesCode'],validate='one_to_one')
frame=frame.sort_values(['Date','Score','SecuritiesCode'],ascending=[True,False,True])
frame['Rank']=frame.groupby('Date').cumcount()
daily=daily_metrics(frame,'Score','Rank')
baseline=pd.read_pickle(RUN/'baseline.pkl').merge(labels.loc[labels.ValidationYear.eq(year)],on=['Date','SecuritiesCode'],validate='one_to_one')
base=daily_metrics(baseline,'g','Rank')
valid=daily.SelectedMissingTargets.eq(0)&base.SelectedMissingTargets.eq(0)
sys.path.insert(0,str(BASE/'jpx_official_ranking_20260912'))
from official_metric import calc_spread_return_sharpe
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    check=calc_spread_return_sharpe(frame.loc[frame.Date.isin(daily.loc[valid,'Date']),['Date','Rank','Target']])
np.testing.assert_allclose(check,stats(daily.loc[valid])['sharpe'],atol=1e-12,rtol=0)
result={'year':year,'arima':stats(daily.loc[valid]),'v7':stats(base.loc[valid]),
        'fallback_fraction':float(frame.Fallback.mean()),'official_check_passed':True}
(RUN/f'partial_{year}.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
