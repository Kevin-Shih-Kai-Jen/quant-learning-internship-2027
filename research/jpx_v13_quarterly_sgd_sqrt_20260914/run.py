import sys,json,time,math
import numpy as np
import pandas as pd
from features import ROOT,METRICS,VARIANTS,load,V11
from native import initial,coefficients,train
sys.path.insert(0,str(ROOT.parent/'jpx_v7_daily_mse_20260912'))
from run_v7 import FEATURES,INPUTS,evaluate
NAMES=['alpha']+FEATURES+[m+'Beta'+j for m in METRICS for j in ['Actual','QoQ','YoY','Revision']]+[m+'Discount' for m in METRICS]

def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))

def run(variant,f,x,groups,calendar,fin):
    out=ROOT/variant;out.mkdir(exist_ok=True);(out/'traces').mkdir(exist_ok=True)
    start=time.monotonic();theta=initial();pending=[];states=[];updates=[];rankings=[]
    years=calendar.set_index('Date').ValidationYear.to_dict();end=calendar.Date.max()
    std=pd.read_pickle(V11/'return_std22_features.pkl')
    fallback=(~np.isfinite(f[INPUTS[:-2]]).all(axis=1)).to_numpy()|(~np.isfinite(std[['PR1Scaled22','VR1Scaled22']]).all(axis=1)).to_numpy()
    for date,indices in groups.items():
        daybefore=theta.copy();due=[p for p in pending if p['exit']<=date];pending=[p for p in pending if p['exit']>date]
        assert len(due)<=1
        for batch in due:
            assert batch['date']<batch['exit']<=date
            idx=batch['ix'];known=np.isfinite(f.loc[idx,'Target']);ix=idx[known];y=f.loc[ix,'Target'].to_numpy();n=len(ix)
            order=np.random.default_rng(20260912+int(batch['date'].strftime('%Y%m%d'))).permutation(n).astype(np.int32)
            full=math.isqrt(n) if variant=='sgd_sqrt' else 0;before=theta.copy()
            theta,tr,snap,losses=train(x[ix],y,theta,order,full)
            flat=tr.reshape(-1,10);accepted=tr[:,:,8]==0;fa=flat[:,8]==0
            assert (flat[fa,3]<flat[fa,2]).all()
            assert (flat[fa,3]<=flat[fa,2]-1e-4*flat[fa,9]+1e-12).all()
            assert losses[2]<=losses[1]+1e-12*max(1,losses[1])
            np.savez_compressed(out/'traces'/f'{batch["date"].date()}.npz',indices=ix,order=order,
                theta_before=before,theta_after=theta,trace=tr,snapshots=snap,losses=losses)
            updates.append({'Date':date,'SignalDate':batch['date'],'ExitDate':batch['exit'],'KnownLabelStocks':n,'MissingTargets':int((~known).sum()),
                'SGDAttempts':n,'SGDAccepted':int(accepted[:n].any(axis=1).sum()),'SGDNoEta':int((tr[:n,:,8]==2).sum()),
                'FullAttempts':full,'FullAccepted':int(accepted[n:].any(axis=1).sum()),'FullNoEta':int((tr[n:,:,8]==2).sum()),
                'LinearAccepted':int(accepted[:,0].sum()),'DiscountAccepted':int(accepted[:,1].sum()),
                'CandidateEvaluations':int(tr[:,:,7].sum()),'LossBeforeSGD':losses[0],'LossAfterSGD':losses[1],'LossAfterFull':losses[2],
                **{'Before_'+c:float(v) for c,v in zip(NAMES,before)},**{'After_'+c:float(v) for c,v in zip(NAMES,theta)}})
        states.append({'Date':date,'NewLabelDays':len(due),'CumulativeLabelDays':len(updates),
            **{'Before_'+c:float(v) for c,v in zip(NAMES,daybefore)},**{'After_'+c:float(v) for c,v in zip(NAMES,theta)}})
        if date==pd.Timestamp('2020-10-01') or date>end:continue
        # Prediction path receives only feature matrix and identifiers, no label.
        effective=coefficients(theta);scores=x[indices]@effective
        ranks=f.loc[indices,['SignalDate','SecuritiesCode']].rename(columns={'SignalDate':'Date'}).copy()
        ranks['g']=scores;ranks['FinancialContribution']=x[indices,12:]@effective[12:]
        ranks['KnownEventCount']=fin.iloc[indices].KnownEventCount.to_numpy();ranks['Fallback']=fallback[indices]
        ranks=ranks.sort_values(['g','SecuritiesCode'],ascending=[False,True],kind='stable');ranks['Rank']=np.arange(len(ranks))
        ranks['ValidationYear']=years.get(date,0);assert np.isfinite(scores).all();rankings.append(ranks)
        exits=f.loc[indices,'ExitDate'];assert exits.nunique()==1 and exits.notna().all()
        pending.append({'date':date,'exit':exits.iloc[0],'ix':indices})
        if len(rankings)%100==0:print(variant,date.date(),'label_days',len(updates),'seconds',round(time.monotonic()-start,1),flush=True)
    assert not pending
    ranks=pd.concat(rankings,ignore_index=True)
    for yr,g in ranks.groupby('ValidationYear'):g.to_csv(out/(f'ranks_{yr}.csv.gz' if yr else 'ranks_warmup.csv.gz'),index=False,compression='gzip')
    states=pd.DataFrame(states);states.to_csv(out/'parameter_history.csv',index=False)
    u=pd.DataFrame(updates);u.to_csv(out/'training_updates.csv',index=False)
    validation=ranks.loc[ranks.ValidationYear.ne(0)];labels=f[['SignalDate','SecuritiesCode','Target']]
    daily,selected,total=evaluate(validation,labels)
    scored=validation.merge(labels.rename(columns={'SignalDate':'Date'}),on=['Date','SecuritiesCode'],validate='one_to_one')
    scored['SquaredError']=(scored.g-scored.Target)**2
    metrics=scored.groupby('Date').agg(AllStockForecastMSE=('SquaredError','mean'),MaxAbsolutePrediction=('g',lambda s:s.abs().max())).reset_index()
    daily=daily.merge(calendar,on='Date',validate='one_to_one').merge(metrics,on='Date',validate='one_to_one')
    daily.to_csv(out/'daily_spread_returns.csv',index=False);selected.to_csv(out/'selected_200_each_side.csv.gz',index=False,compression='gzip')
    scored.nlargest(30,'SquaredError').to_csv(out/'largest_prediction_errors.csv',index=False)
    result={'version':'v13','variant':variant,'validation':total,'mean_daily_forecast_mse':float(daily.AllStockForecastMSE.mean()),
        'training_days':len(u),'forecast_days':ranks.Date.nunique(),'forecast_rows':len(ranks),
        **{c:int(u[c].sum()) for c in ['KnownLabelStocks','SGDAttempts','SGDAccepted','SGDNoEta','FullAttempts','FullAccepted','FullNoEta','CandidateEvaluations']},
        'sgd_increased_day_loss_days':int((u.LossAfterSGD>u.LossBeforeSGD+1e-12).sum()),
        'final_increased_day_loss_days':int((u.LossAfterFull>u.LossBeforeSGD+1e-12).sum()),
        'mean_training_loss_before':float(u.LossBeforeSGD.mean()),'mean_training_loss_after_sgd':float(u.LossAfterSGD.mean()),
        'mean_training_loss_after_full':float(u.LossAfterFull.mean()),'full_improved_days':int((u.LossAfterFull<u.LossAfterSGD).sum()),
        'final_parameters':dict(zip(NAMES,map(float,theta))),'seconds':time.monotonic()-start,
        'last_forecast_date':str(end.date()),'parameter_asof':str(states.Date.iloc[-1].date()),'internal_checks_passed':True}
    save(out/'results.json',result);print('DONE',json.dumps(result),flush=True)
    return result

def main():
    for name in ['preflight.json','feature_audit.json']:assert json.loads((ROOT/name).read_text())['passed']
    f,x,groups,cal,fin=load()
    for variant in (sys.argv[1:] or VARIANTS):run(variant,f,x,groups,cal,fin)
if __name__=='__main__':main()
