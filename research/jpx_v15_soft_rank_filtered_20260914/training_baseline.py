import json
import numpy as np
import pandas as pd
from features import ROOT,VARIANTS

def main():
    updates=pd.read_csv(ROOT/'sgd_only/training_updates.csv');flat=[]
    for date in updates.SignalDate:
        with np.load(ROOT/'sgd_only/traces'/f'{date[:10]}.npz') as z:
            ranks=z['truth_rank'];n=len(ranks)
            flat.append(float(np.mean((((n+1)/2-ranks)/(n-1))**2)))
    baseline=pd.DataFrame({'SignalDate':updates.SignalDate,'FlatScoreRankLoss':flat})
    baseline.to_csv(ROOT/'training_flat_score_baseline.csv',index=False);summary={}
    for variant in VARIANTS:
        data=pd.read_csv(ROOT/variant/'training_updates.csv').merge(baseline,on='SignalDate',validate='one_to_one')
        summary[variant]={'mean_flat_training_loss':float(data.FlatScoreRankLoss.mean()),
            'mean_loss_after_sgd':float(data.LossAfterSGD.mean()),'mean_loss_after_full':float(data.LossAfterFull.mean()),
            'after_sgd_beats_flat_days':int((data.LossAfterSGD<data.FlatScoreRankLoss).sum()),
            'after_full_beats_flat_days':int((data.LossAfterFull<data.FlatScoreRankLoss).sum()),'days':len(data)}
    (ROOT/'training_flat_score_baseline.json').write_text(json.dumps(summary,indent=2));print(summary)
if __name__=='__main__':main()
