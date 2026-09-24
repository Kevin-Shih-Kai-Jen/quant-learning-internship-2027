import sys,json,pickle,gzip,time
import numpy as np
import pandas as pd
from features import ROOT,FINS,VARIANTS,load
from native_mse import train
sys.path.insert(0,str(ROOT.parent/'jpx_v7_daily_mse_20260912'))
from run_v7 import INPUTS,FEATURES,make_x,evaluate,save
NAMES=['alpha']+FEATURES+FINS
TRACE_NAMES=['BatchIndex','BatchSize','LossBefore','LossAfter','Eta','GradInf','GradNormSquared','Candidates','Status','HitUpperEta']+[c+'Gradient' for c in FINS]+[c+'Input' for c in FINS]+[c+'Before' for c in FINS]

def predict(obs,t):
    assert 'Target' not in obs
    x=np.column_stack([make_x(obs),obs[FINS].to_numpy()]);scores=x@t
    r=obs[['SignalDate','SecuritiesCode','FinancialEventId']+FINS+[c+'Valid' for c in FINS]].rename(columns={'SignalDate':'Date'}).copy()
    r['g']=scores;r['FinancialContribution']=x[:,12:]@t[12:]
    r['TFallback']=~np.isfinite(obs[INPUTS[:-2]]).all(axis=1);r['ReturnFallback']=~np.isfinite(obs[INPUTS[-2:]]).all(axis=1)
    r['FinancialFallback']=~obs[[c+'Valid' for c in FINS]].all(axis=1);r['Fallback']=r.TFallback|r.ReturnFallback
    r=r.sort_values(['Date','g','SecuritiesCode'],ascending=[True,False,True],kind='stable');r['Rank']=r.groupby('Date').cumcount()
    assert np.isfinite(r.g).all();return r

def main(variant):
    assert json.loads((ROOT/'preflight.json').read_text())['passed'] and json.loads((ROOT/'feature_audit.json').read_text())['passed']
    f,xall,groups,calendar=load(variant);out=ROOT/variant;out.mkdir(exist_ok=True);start=time.monotonic()
    year=calendar.set_index('Date').ValidationYear.to_dict();end=calendar.Date.max()
    t=np.zeros(18);pending=[];states=[];updates=[];snaps=[];rankings=[]
    with gzip.open(out/'batch_trace.csv.gz','wt') as handle:
        handle.write('SignalDate,UpdateDate,'+','.join(TRACE_NAMES)+'\n')
        for date,idx in groups.items():
            beforeday=t.copy();due=[b for b in pending if b['exit']<=date];pending=[b for b in pending if b['exit']>date];assert len(due)<=1
            for batch in due:
                assert batch['date']<batch['exit']<=date
                ix=batch['ix'];known=np.isfinite(f.loc[ix,'Target']);ix=ix[known];truth=f.loc[ix,'Target'].to_numpy();n=len(ix)
                order=np.random.default_rng(20260912+int(batch['date'].strftime('%Y%m%d'))).permutation(n).astype(np.int32)
                z=xall[ix[order],12:];sample={0,n//2,n-1}
                for k in range(6):sample.add(int(np.argmax(abs(z[:,k]))))
                before=t.copy();t,tr,ids,ss,losses=train(xall[ix],truth,t,order,1,snapshot_batches=sorted(sample))
                accept=tr[:,8]==0;fg=tr[:,10:16];fb=tr[:,22:28];afterfin=np.vstack([fb[1:],t[12:]])
                np.testing.assert_array_equal(tr[:,16:22],z)
                np.testing.assert_allclose(afterfin,fb-tr[:,4,None]*fg,atol=1e-13,rtol=1e-12)
                assert (fg[z==0]==0).all();np.testing.assert_array_equal(afterfin[z==0],fb[z==0])
                assert (tr[accept,3]<tr[accept,2]).all() and (tr[accept,3]<=tr[accept,2]-1e-4*tr[accept,4]*tr[accept,6]+1e-12).all()
                prefix=f'{batch["date"].date()},{date.date()},'
                handle.writelines(prefix+','.join(format(v,'.17g') for v in row)+'\n' for row in tr)
                snaps.append({'SignalDate':batch['date'],'UpdateDate':date,'indices':ix,'order':order,'snapshot_batches':ids,'theta_before_gradient':ss,'snapshot_trace':tr[ids]})
                updates.append({'Date':date,'SignalDate':batch['date'],'ExitDate':batch['exit'],'ForecastStocks':len(known),'KnownLabelStocks':n,
                    'MissingTargetStocks':int((~known).sum()),'Batches':n,'AcceptedUpdates':int(accept.sum()),'SmallGradientSkips':int((tr[:,8]==1).sum()),
                    'NoAcceptableEtaSkips':int((tr[:,8]==2).sum()),'CandidateEvaluations':int(tr[:,7].sum()),'FinancialGradientDominatesBatches':int(((tr[:,6]>0)&(np.sum(fg**2,axis=1)>=.9*tr[:,6])).sum()),
                    'EPSGradientDominatesBatches':int(((tr[:,6]>0)&((fg[:,2]**2+fg[:,5]**2)>=.9*tr[:,6])).sum()),'FullDayLossBefore':losses[0],'FullDayLossAfter':losses[1],
                    **{c+'NonzeroInputStocks':int((z[:,k]!=0).sum()) for k,c in enumerate(FINS)},
                    **{'Before_'+c:float(v) for c,v in zip(NAMES,before)},**{'After_'+c:float(v) for c,v in zip(NAMES,t)}})
            states.append({'Date':date,'NewLabelDays':len(due),'CumulativeLabelDays':len(updates),**{'Before_'+c:float(v) for c,v in zip(NAMES,beforeday)},**{'After_'+c:float(v) for c,v in zip(NAMES,t)}})
            if date==pd.Timestamp('2020-10-01') or date>end:continue
            obs=f.loc[idx,['SignalDate','SecuritiesCode','FinancialEventId']+INPUTS+FINS+[c+'Valid' for c in FINS]]
            rank=predict(obs,t);rank['ValidationYear']=year.get(date,0);rankings.append(rank)
            exits=f.loc[idx,'ExitDate'];assert exits.nunique()==1 and exits.notna().all();pending.append({'date':date,'exit':exits.iloc[0],'ix':idx})
            if len(rankings)%100==0:print(variant,date.date(),'label_days',len(updates),'seconds',round(time.monotonic()-start,1),flush=True)
    assert not pending
    ranks=pd.concat(rankings,ignore_index=True)
    for y,g in ranks.groupby('ValidationYear'):g.to_csv(out/(f'ranks_{y}.csv.gz' if y else 'ranks_warmup.csv.gz'),index=False,compression='gzip')
    pd.DataFrame(states).to_csv(out/'parameter_history.csv',index=False);u=pd.DataFrame(updates);u.to_csv(out/'training_updates.csv',index=False)
    with gzip.open(out/'gradient_snapshots.pkl.gz','wb') as h:pickle.dump(snaps,h,protocol=5)
    valid=ranks.loc[ranks.ValidationYear.ne(0)];labels=f[['SignalDate','SecuritiesCode','Target']]
    daily,selected,total=evaluate(valid,labels)
    scored=valid.merge(labels.rename(columns={'SignalDate':'Date'}),on=['Date','SecuritiesCode'],validate='one_to_one');scored['SquaredError']=(scored.g-scored.Target)**2
    metrics=scored.groupby('Date').agg(AllStockForecastMSE=('SquaredError','mean'),FinancialFallbackStocks=('FinancialFallback','sum')).reset_index()
    daily=daily.merge(calendar,on='Date',validate='one_to_one').merge(metrics,on='Date',validate='one_to_one')
    daily.to_csv(out/'daily_spread_returns.csv',index=False);selected.to_csv(out/'selected_200_each_side.csv.gz',index=False,compression='gzip')
    result={'version':'v12','variant':variant,'validation':total,'label_days':len(updates),'training_stocks':int(u.KnownLabelStocks.sum()),
        **{k:int(u[c].sum()) for k,c in {'batches':'Batches','accepted_updates':'AcceptedUpdates','small_gradient_skips':'SmallGradientSkips','no_acceptable_eta_skips':'NoAcceptableEtaSkips','candidate_evaluations':'CandidateEvaluations','financial_gradient_dominates_batches':'FinancialGradientDominatesBatches','eps_gradient_dominates_batches':'EPSGradientDominatesBatches'}.items()},
        'mean_daily_forecast_mse':float(daily.AllStockForecastMSE.mean()),'full_day_loss_increased_days':int((u.FullDayLossAfter>u.FullDayLossBefore+1e-14).sum()),
        'final_coefficients':dict(zip(NAMES,map(float,t))),'last_forecast_date':str(end.date()),'final_parameter_asof':str(states[-1]['Date'].date()),'seconds':time.monotonic()-start,'internal_checks_passed':True}
    save(out/'results.json',result);print('DONE',json.dumps(result),flush=True)
if __name__=='__main__':main(sys.argv[1])
