"""Post-hoc diagnostic only: fixed v13 predictions, no model training."""
from pathlib import Path
import json, pickle
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent.parent
with (ROOT.parent/'jpx_v8_soft_rank_20260912/inputs.pkl').open('rb') as f:
    base,_,_,_=pickle.load(f)
labels=base[['SignalDate','SecuritiesCode','Target']].rename(columns={'SignalDate':'Date'})
labels['Date']=labels.Date.astype(str)
del base

def corr(a,b,rank=False):
    if rank:
        a=pd.Series(a).rank(method='average').to_numpy()
        b=pd.Series(b).rank(method='average').to_numpy()
    if np.std(a)==0 or np.std(b)==0:return np.nan
    return np.corrcoef(a,b)[0,1]

rows=[]; sel={}; pred={}
for v in ['sgd_only','sgd_sqrt']:
    r=pd.concat([pd.read_csv(ROOT/v/f'ranks_{y}.csv.gz') for y in range(2018,2022)],ignore_index=True)
    r=r.merge(labels,on=['Date','SecuritiesCode'],validate='one_to_one')
    s=pd.read_csv(ROOT/v/'selected_200_each_side.csv.gz')
    sel[v]=s
    pred[v]=r
    for date,g in r.groupby('Date',sort=True):
        g=g[np.isfinite(g.Target)&np.isfinite(g.g)]
        y=g.Target.to_numpy();p=g.g.to_numpy()
        f=g.FinancialContribution.to_numpy();pv=p-f
        row={'Date':date,'ValidationYear':int(g.ValidationYear.iloc[0]),'Variant':v,'N':len(g),
             'RankIC':corr(p,y,True),'PearsonIC':corr(p,y),
             'PricePartRankIC':corr(pv,y,True),'FinancialPartRankIC':corr(f,y,True),
             'PredictionMean':p.mean(),'PredictionStd':p.std(ddof=0),'TargetMean':y.mean(),'TargetStd':y.std(ddof=0),
             'MSE':np.mean((p-y)**2),'ZeroMSE':np.mean(y**2),
             'CrossSectionMeanBiasSquared':(p.mean()-y.mean())**2,
             'CenteredMSE':np.mean(((p-p.mean())-(y-y.mean()))**2),
             'PredictionVariance':p.var(),'TargetVariance':y.var(),'PredictionTargetCovariance':np.mean((p-p.mean())*(y-y.mean())),
             'FinancialAbsShare':np.mean(np.abs(f))/(np.mean(np.abs(f))+np.mean(np.abs(pv))),
             'MeanAbsoluteFinancial':np.mean(np.abs(f)),'MeanAbsolutePrice':np.mean(np.abs(pv)),
             'MaxAbsPrediction':np.abs(p).max()}
        q=s[s.Date.eq(date)]
        for side in ['long','short']:
            z=q[q.Side.eq(side)]
            row[side.title()+'WeightedReturn']=float((z.Target*z.WithinSideWeight).sum())
            row[side.title()+'EWReturn']=float(z.Target.mean())
            row[side.title()+'AbsFinancial']=float(z.FinancialContribution.abs().mean())
            row[side.title()+'MeanPrediction']=float(z.g.mean())
        row['LongShortReturn']=row['LongWeightedReturn']-row['ShortWeightedReturn']
        row['LongExcessUniverse']=row['LongWeightedReturn']-row['TargetMean']
        row['ShortExcessUniverse']=row['ShortWeightedReturn']-row['TargetMean']
        rows.append(row)
    print('processed',v,len(r),flush=True)
d=pd.DataFrame(rows);d.to_csv(OUT/'daily_metrics.csv',index=False)
over=[]
for date,a in pred['sgd_only'].groupby('Date',sort=True):
    b=pred['sgd_sqrt'].loc[lambda x:x.Date.eq(date)]
    z=a.merge(b,on=['Date','SecuritiesCode'],validate='one_to_one',suffixes=('_only','_sqrt'))
    row={'Date':date,'ValidationYear':int(a.ValidationYear.iloc[0]),'PredictionRankCorrelation':corr(z.g_only,z.g_sqrt,True),
         'PredictionPearsonCorrelation':corr(z.g_only,z.g_sqrt),'N':len(z)}
    sets={}
    for v in ['sgd_only','sgd_sqrt']:
        for side in ['long','short']:
            sets[(v,side)]=set(sel[v].loc[sel[v].Date.eq(date)&sel[v].Side.eq(side),'SecuritiesCode'])
    for side in ['long','short']:
        row[side.title()+'OverlapFraction']=len(sets[('sgd_only',side)]&sets[('sgd_sqrt',side)])/200
    row['OnlyLongSqrtShortFraction']=len(sets[('sgd_only','long')]&sets[('sgd_sqrt','short')])/200
    row['OnlyShortSqrtLongFraction']=len(sets[('sgd_only','short')]&sets[('sgd_sqrt','long')])/200
    over.append(row)
o=pd.DataFrame(over);o.to_csv(OUT/'daily_overlap.csv',index=False)

summ=[]
for variant,vd in d.groupby('Variant'):
    for yr,g in [('all',vd)]+list(vd.groupby('ValidationYear')):
        x={'Variant':variant,'Period':str(yr),'Days':len(g)}
        for c in ['RankIC','PearsonIC','PricePartRankIC','FinancialPartRankIC','PredictionMean','PredictionStd','TargetMean','TargetStd','MSE','ZeroMSE','CrossSectionMeanBiasSquared','CenteredMSE','PredictionVariance','TargetVariance','PredictionTargetCovariance','FinancialAbsShare','MeanAbsoluteFinancial','MeanAbsolutePrice','LongWeightedReturn','ShortWeightedReturn','LongShortReturn','LongExcessUniverse','ShortExcessUniverse']:
            x[c]=float(g[c].mean())
        x['RankICFiniteDays']=int(g.RankIC.notna().sum())
        x['RankICPositiveFraction']=float((g.RankIC.dropna()>0).mean())
        x['LongShortPositiveFraction']=float((g.LongShortReturn>0).mean())
        x['Sharpe']=float(g.LongShortReturn.mean()/g.LongShortReturn.std(ddof=1))
        x['MedianDailyMSE']=float(g.MSE.median())
        x['MedianDailyPredictionStd']=float(g.PredictionStd.median())
        x['FractionMSEBelowZeroPrediction']=float((g.MSE<g.ZeroMSE).mean())
        x['Top10MSEDaysFraction']=float(g.MSE.nlargest(10).sum()/g.MSE.sum())
        x['MSEOverZero']=x['MSE']/x['ZeroMSE']
        summ.append(x)
su=pd.DataFrame(summ);su.to_csv(OUT/'summary.csv',index=False)
os=[]
for yr,g in [('all',o)]+list(o.groupby('ValidationYear')):
    os.append({'Period':str(yr),'Days':len(g),**{c:float(g[c].mean()) for c in o.columns if c not in ['Date','ValidationYear','N']}})
pd.DataFrame(os).to_csv(OUT/'overlap_summary.csv',index=False)

joined=d[d.Variant.eq('sgd_only')].merge(d[d.Variant.eq('sgd_sqrt')],on='Date',suffixes=('_only','_sqrt'),validate='one_to_one')
paired={}
for metric in ['RankIC','PearsonIC','LongShortReturn','MSE']:
    diff=joined[f'{metric}_sqrt']-joined[f'{metric}_only']
    paired[metric]={'MeanDifference':float(diff.mean()),'MedianDifference':float(diff.median()),'SqrtHigherFraction':float((diff>0).mean()),'DailyDifferenceStd':float(diff.std(ddof=1))}
paired['MSEDecomposition']={}
for c in ['CrossSectionMeanBiasSquared','PredictionVariance','TargetVariance','PredictionTargetCovariance']:
    diff=joined[f'{c}_sqrt'].mean()-joined[f'{c}_only'].mean()
    paired['MSEDecomposition'][c]=float(diff)
paired['overlap']=os[0]
paired['notes']=[
    'All diagnostics use already-saved validation forecasts. No training or parameter selection.',
    'Returns are decimal. LongShortReturn is long weighted return minus short weighted return; half of it is illustrative gross-one 50/50 portfolio return.',
    'Spearman IC uses average tied ranks of g and Target, excluding missing/non-finite Target. Comparisons use the same stock universe.',
    'Mean IC omits undefined correlations: all stocks have Target=0 on 2020-09-29, hence 952 finite IC days out of 953 validation days.',
    'Financial and price-part IC are fixed-prediction decomposition, not separately trained ablations and not causal attribution.',
    'MSE = cross-sectional mean bias squared + prediction variance + target variance - 2 prediction-target covariance, then averaged equally across days.',
    'Overlap is intersection size / 200 for each selected side; random set overlap would be about 200/N but does not account for common covariates.',
]
(OUT/'paired_summary.json').write_text(json.dumps(paired,indent=2,ensure_ascii=False))
print(su.to_string(index=False),flush=True)
print(json.dumps(paired,indent=2),flush=True)
