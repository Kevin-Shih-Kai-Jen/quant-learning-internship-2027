import json
import pandas as pd
from features import ROOT,V14,load,VARIANTS
from run import ranking_metrics

def main():
    f,_,_,_,_=load();labels=f[['SignalDate','SecuritiesCode','Target']].rename(columns={'SignalDate':'Date'});summary={}
    for variant in VARIANTS:
        ranks=pd.concat([pd.read_csv(V14/variant/f'ranks_{year}.csv.gz',parse_dates=['Date']) for year in range(2018,2022)])
        result=ranking_metrics(ranks.merge(labels,on=['Date','SecuritiesCode'],validate='one_to_one'))
        result.to_csv(ROOT/f'v14_{variant}_ranking_metrics.csv',index=False)
        old=json.loads((V14/variant/'results.json').read_text())
        summary[variant]={'sharpe':old['validation']['official_style_unannualized_sharpe'],
            'rank_ic':float(result.RankIC.mean()),'normalized_hard_rank_mse':float(result.NormalizedHardRankMSE.mean())}
    (ROOT/'v14_comparison_metrics.json').write_text(json.dumps(summary,indent=2))
    print(summary)
if __name__=='__main__':main()
