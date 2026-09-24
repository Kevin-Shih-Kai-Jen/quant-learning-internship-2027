import sys,json,pickle
import numpy as np
import pandas as pd
from features import ROOT,V8,VARIANTS
from native import LIB,initial,dp

def run(variant,labels):
    out=ROOT/variant
    assert (out/'results.json').exists()
    ranks=pd.concat([pd.read_csv(out/f'ranks_{year}.csv.gz',parse_dates=['Date']) for year in range(2018,2022)])
    valid=ranks.merge(labels,on=['Date','SecuritiesCode'],validate='one_to_one');rows=[]
    for date,day in valid.groupby('Date',sort=True):
        day=day.loc[np.isfinite(day.Target)&np.isfinite(day.g)];n=len(day)
        truth=np.ascontiguousarray(day.Target.rank(ascending=False,method='average').to_numpy())
        xx=np.zeros((n,27));xx[:,1]=day.g.to_numpy();t=initial();t[1]=1.;soft=np.zeros(n)
        loss=LIB.evaluate_rank(n,dp(xx),dp(truth),dp(t),-1,-1,1.,None,dp(soft),8)
        independent=float(np.mean(((soft-truth)/(n-1))**2))
        np.testing.assert_allclose(loss,independent,atol=1e-12)
        flat=float(np.mean((((n+1)/2-truth)/(n-1))**2))
        rows.append({'Date':date,'ScoredStocks':n,'SoftRankLoss':loss,'FlatScoreSoftRankLoss':flat,'ScoreRange':float(day.g.max()-day.g.min())})
    daily=pd.DataFrame(rows);daily.to_csv(out/'daily_soft_rank_loss.csv',index=False)
    result={'days':len(daily),'mean_daily_soft_rank_loss':float(daily.SoftRankLoss.mean()),
        'mean_daily_flat_score_soft_rank_loss':float(daily.FlatScoreSoftRankLoss.mean()),
        'beat_flat_score_loss_days':int((daily.SoftRankLoss<daily.FlatScoreSoftRankLoss).sum()),
        'flat_score_is_a_non_trading_diagnostic':True,'temperature':1.,'universe':'all finite-label validation stocks, no training threshold filter'}
    (out/'score_diagnostics.json').write_text(json.dumps(result,indent=2));print(variant,result,flush=True)

if __name__=='__main__':
    with (V8/'inputs.pkl').open('rb') as h:f,*_=pickle.load(h)
    labels=f[['SignalDate','SecuritiesCode','Target']].rename(columns={'SignalDate':'Date'})
    for variant in sys.argv[1:] or VARIANTS:run(variant,labels)
