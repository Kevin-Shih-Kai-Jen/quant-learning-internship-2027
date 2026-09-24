"""Independent checks of signed order statistics, raw-bound smoothing, and stored wealth paths."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

root=Path(__file__).resolve().parent
r=json.loads((root/'results.json').read_text())
s=pd.read_csv(root/'market_signals.csv',parse_dates=['SignalDate','EntryDate','ExitDate'])
checks=[]
last_u,last_l=.001,-.001
for a in r['annual_bounds']:
    y=a['validation_year'];cutoff=pd.Timestamp(a['asof'])
    h=s[(s.SignalDate<cutoff)&(s.ExitDate<=cutoff)]
    assert len(h)==a['historical_rows']
    if len(h):
        p=sorted(h.loc[h.MarketPrediction>0,'MarketPrediction'].tolist())
        n=sorted(h.loc[h.MarketPrediction<0,'MarketPrediction'].tolist(),reverse=True)
        # Integer ceiling avoids relying on floating-point multiplication by .1.
        kp=(len(p)+9)//10;kn=(len(n)+9)//10
        u,l=p[kp-1],n[kn-1]
        assert kp==a['positive_selected_count'] and kn==a['negative_selected_count']
        np.testing.assert_allclose([u,l],[a['raw_upper'],a['raw_lower']],rtol=0,atol=1e-15)
        np.testing.assert_allclose([last_u,last_l],[a['previous_raw_upper'],a['previous_raw_lower']],rtol=0,atol=1e-15)
        used_u,used_l=.6*u+.4*last_u,.6*l+.4*last_l
        last_u,last_l=u,l
    else:
        used_u,used_l=.001,-.001
    np.testing.assert_allclose([used_u,used_l],[a['used_upper'],a['used_lower']],rtol=0,atol=1e-15)
    v=s[s.ValidationYear==y]
    expected=np.where(v.MarketPrediction>used_u,1,np.where(v.MarketPrediction<used_l,-1,0))
    np.testing.assert_array_equal(expected,v.Regime)
    checks.append({'year':y,'history_timing':True,'signed_counts_and_order_statistics':True,
        'uses_previous_raw_not_smoothed':True,'boundary_and_regime_assignment':True})

def independent_metrics(ret):
    ret=np.asarray(ret,dtype=float);wealth=np.r_[1.,np.cumprod(1+ret)]
    return {'cumulative_return':wealth[-1]-1,'annualized_return_252':wealth[-1]**(252/len(ret))-1,
            'max_drawdown':np.min(wealth/np.maximum.accumulate(wealth)-1)}

for name,a in r['scenarios'].items():
    d=pd.read_csv(root/name/'daily_returns.csv')
    for col,key in [('StrategyReturn','strategy'),('IndexReturn','benchmark')]:
        m=independent_metrics(d[col])
        for k,value in m.items():
            np.testing.assert_allclose(value,a[key][k],rtol=0,atol=1e-12)
    if name=='signed_smoothed':
        for y,v in d.groupby('ValidationYear'):
            actual=next(x for x in r['annual'] if x['year']==y)
            p=independent_metrics(v.StrategyReturn);b=independent_metrics(v.IndexReturn)
            assert bool(p['cumulative_return']>b['cumulative_return'] and abs(p['max_drawdown'])<abs(b['max_drawdown']))==actual['both_conditions_met']
out={'threshold_checks':checks,'independent_performance_checks':True,'annual_conditions_checked':True,
     'total_neutral_days':int(s.Regime.eq(0).sum()),'total_days':len(s),'passed':True}
(root/'independent_audit.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
