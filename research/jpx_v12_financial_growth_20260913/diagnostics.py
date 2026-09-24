import json,pickle
import numpy as np
import pandas as pd
from features import ROOT,V8,FINS

def main():
    out=ROOT/'six_financial';summary=json.loads((out/'results.json').read_text())
    noeta=0;eps_sufficient=0;norm_sufficient=0;eps_dominates=0;count=0
    cols=['Status','GradNormSquared','LossBefore','ActualEPSGradient','ForecastEPSGradient','ActualEPSInput','ForecastEPSInput']
    for g in pd.read_csv(out/'batch_trace.csv.gz',usecols=cols,chunksize=200000,float_precision='round_trip'):
        mask=g.Status.eq(2);noeta+=int(mask.sum());count+=len(g)
        eps_sufficient+=int((mask&(g.ActualEPSInput**2+g.ForecastEPSInput**2>999900)).sum())
        norm=g.GradNormSquared/(4*g.LossBefore.where(g.LossBefore.ne(0)))
        norm_sufficient+=int((mask&norm.gt(999900)).sum())
        eps_dominates+=int(((g.GradNormSquared>0)&(g.ActualEPSGradient**2+g.ForecastEPSGradient**2>=.9*g.GradNormSquared)).sum())
    assert noeta==summary['no_acceptable_eta_skips']
    daily=pd.read_csv(out/'daily_spread_returns.csv',parse_dates=['Date']);worst=daily.nlargest(3,'AllStockForecastMSE');days=set(worst.Date)
    hist=pd.read_csv(out/'parameter_history.csv',parse_dates=['Date'],float_precision='round_trip').set_index('Date')
    with (V8/'inputs.pkl').open('rb') as h:f,_,_,_=pickle.load(h)
    labels=f[['SignalDate','SecuritiesCode','Target']].rename(columns={'SignalDate':'Date'})
    examples=[]
    for year in sorted({int(daily.loc[daily.Date.eq(d),'ValidationYear'].iloc[0]) for d in days}):
        g=pd.read_csv(out/f'ranks_{year}.csv.gz',parse_dates=['Date'],float_precision='round_trip')
        g=g.loc[g.Date.isin(days)].merge(labels,on=['Date','SecuritiesCode'],validate='one_to_one');g['SquaredError']=(g.g-g.Target)**2
        for d,z in g.groupby('Date'):
            z=z.nlargest(5,'SquaredError').copy();t=hist.loc[d]
            for c in FINS:z[c+'Contribution']=z[c]*t['After_'+c]
            z['EPSContribution']=z.ActualEPSContribution+z.ForecastEPSContribution
            examples.append(z[['Date','SecuritiesCode','g','Target','SquaredError','FinancialContribution','EPSContribution']+FINS+[c+'Contribution' for c in FINS]])
    examples=pd.concat(examples,ignore_index=True).sort_values('SquaredError',ascending=False);examples.to_csv(ROOT/'largest_prediction_errors.csv',index=False)
    result={'training_batches':count,'no_acceptable_eta_batches':noeta,'no_eta_with_norm_squared_above_armijo_limit':norm_sufficient,
        'no_eta_where_eps_inputs_alone_exceed_armijo_norm_limit':eps_sufficient,'eps_gradient_at_least_90pct_of_squared_norm_batches':eps_dominates,
        'eps_gradient_dominance_fraction':eps_dominates/count,'armijo_min_eta_norm_squared_limit':999900,
        'interpretation':'For a single-observation linear squared loss, Armijo requires eta*||x||^2 <= 1-c (c=1e-4). EPS dominance is a scale diagnostic, not predictive importance or causal performance attribution.',
        'worst_example':{k:(str(v.date()) if k=='Date' else int(v) if k=='SecuritiesCode' else float(v)) for k,v in examples.iloc[0][['Date','SecuritiesCode','g','Target','FinancialContribution','EPSContribution']].items()}}
    (ROOT/'diagnostics.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
