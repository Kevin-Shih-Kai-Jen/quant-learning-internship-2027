import sys,json,pickle,gzip,time,hashlib
import numpy as np
import pandas as pd
from financial_features import ROOT,V8
from native_v9 import train
sys.path.insert(0,str(V8))
from run_v8 import INPUTS,FEATURES,make_x,rank_metrics,evaluate,save
NAMES=['alpha']+FEATURES+['FinancialMarginYoYDecay']
TRACE_NAMES=['BatchIndex','BatchSize','LossBefore','LossAfter','Eta','GradInf','GradNormSquared','Candidates','Status','HitUpperEta','BetaGradient','OwnFinancialFeature','BetaBefore']

def predict(observations,theta):
    assert 'Target' not in observations
    x=np.column_stack([make_x(observations),observations.FinancialFeature.to_numpy()])
    scores=x@theta;assert np.isfinite(scores).all()
    result=observations[['SignalDate','SecuritiesCode','FinancialFeature','FinancialEventId','FinancialAge','DecayWeight','FinancialStatus']].rename(columns={'SignalDate':'Date'}).copy()
    result['g']=scores
    result['TFallback']=~np.isfinite(observations[INPUTS[:-2]]).all(axis=1)
    result['ReturnFallback']=~np.isfinite(observations[INPUTS[-2:]]).all(axis=1)
    result['Fallback']=result.TFallback|result.ReturnFallback
    result['FinancialContribution']=theta[-1]*observations.FinancialFeature
    result=result.sort_values(['Date','g','SecuritiesCode'],ascending=[True,False,True],kind='stable')
    result['Rank']=result.groupby('Date').cumcount()
    return result

def main():
    assert json.loads((ROOT/'preflight.json').read_text())['passed']
    assert json.loads((ROOT/'feature_audit.json').read_text())['passed']
    out=ROOT/'sgd';out.mkdir(exist_ok=True);start=time.monotonic()
    with (V8/'inputs.pkl').open('rb') as handle:f,base_x,groups,calendar=pickle.load(handle)
    finance=pd.read_pickle(ROOT/'financial_signal_features.pkl');assert len(finance)==len(f)
    f=pd.concat([f,finance],axis=1);xall=np.column_stack([base_x,finance.FinancialFeature.to_numpy()]);del base_x
    year_map=calendar.set_index('Date').ValidationYear.to_dict();end=calendar.Date.max()
    theta=np.zeros(13);pending=[];states=[];updates=[];snapshots=[];rankings=[]
    with gzip.open(out/'batch_trace.csv.gz','wt') as handle:
        handle.write('SignalDate,UpdateDate,'+','.join(TRACE_NAMES)+'\n')
        for date,idx in groups.items():
            before_day=theta.copy();due=[b for b in pending if b['exit']<=date];pending=[b for b in pending if b['exit']>date]
            assert len(due)<=1
            for batch in due:
                assert batch['date']<batch['exit']<=date
                ix=batch['indices'];known=np.isfinite(f.loc[ix,'Target'].to_numpy());ix=ix[known]
                truth=f.loc[ix,'Target'].rank(ascending=False,method='average').to_numpy();n=len(ix)
                order=np.random.default_rng(20260912+int(batch['date'].strftime('%Y%m%d'))).permutation(n).astype(np.int32)
                ordered_z=xall[ix[order],-1];snap={0,n//2,n-1,int(np.argmax(abs(ordered_z)))}
                for mask in [ordered_z==0,ordered_z!=0]:
                    if mask.any():snap.add(int(np.flatnonzero(mask)[0]))
                before=theta.copy();theta,tr,snapshot_ids,ss,losses=train(xall[ix],truth,theta,order,1,snapshot_batches=sorted(snap))
                accepted=tr[:,8]==0;own_zero=tr[:,11]==0
                assert len(tr)==n and (tr[:,1]==1).all() and theta[0]==0
                np.testing.assert_array_equal(tr[:,11],ordered_z)
                assert (tr[accepted,3]<tr[accepted,2]).all()
                assert (tr[accepted,3]<=tr[accepted,2]-1e-4*tr[accepted,4]*tr[accepted,6]+1e-14).all()
                assert (tr[~accepted,4]==0).all()
                next_beta=tr[:,12]-tr[:,4]*tr[:,10]
                actual_next_beta=np.r_[tr[1:,12],theta[-1]]
                np.testing.assert_allclose(next_beta,actual_next_beta,atol=1e-14,rtol=1e-13)
                if (ordered_z==0).all():assert (tr[:,10]==0).all() and theta[-1]==before[-1]
                prefix=f'{batch["date"].date()},{date.date()},'
                handle.writelines(prefix+','.join(format(v,'.17g') for v in row)+'\n' for row in tr)
                snapshots.append({'SignalDate':batch['date'],'UpdateDate':date,'indices':ix,'order':order,'snapshot_batches':snapshot_ids,
                    'theta_before_gradient':ss,'snapshot_trace':tr[snapshot_ids]})
                updates.append({'Date':date,'SignalDate':batch['date'],'ExitDate':batch['exit'],'ForecastStocks':len(known),
                    'KnownLabelStocks':n,'MissingTargetStocks':int((~known).sum()),'NonzeroFinancialFeatureStocks':int((ordered_z!=0).sum()),
                    'Batches':len(tr),'AcceptedUpdates':int(accepted.sum()),'SmallGradientSkips':int((tr[:,8]==1).sum()),
                    'NoAcceptableEtaSkips':int((tr[:,8]==2).sum()),'CandidateEvaluations':int(tr[:,7].sum()),'HitUpperEtaBatches':int(tr[:,9].sum()),
                    'OwnZeroFeatureNonzeroBetaGradientBatches':int((own_zero&(abs(tr[:,10])>1e-12)).sum()),
                    'OwnZeroFeatureActualBetaChanges':int((own_zero&(actual_next_beta!=tr[:,12])).sum()),
                    'FinancialGradientDominatesBatches':int(((tr[:,6]>0)&(tr[:,10]**2>=.9*tr[:,6])).sum()),
                    'EtaMinAccepted':float(tr[accepted,4].min()) if accepted.any() else 0.,'EtaMaxAccepted':float(tr[accepted,4].max()) if accepted.any() else 0.,
                    'FullDayLossBefore':losses[0],'FullDayLossAfter':losses[1],
                    **{f'Before_{name}':v for name,v in zip(NAMES,before)},**{f'After_{name}':v for name,v in zip(NAMES,theta)}})
            states.append({'Date':date,'NewLabelDays':len(due),'CumulativeLabelDays':len(updates),
                **{f'Before_{name}':v for name,v in zip(NAMES,before_day)},**{f'After_{name}':v for name,v in zip(NAMES,theta)}})
            if date==pd.Timestamp('2020-10-01') or date>end:continue
            columns=['SignalDate','SecuritiesCode']+INPUTS+list(finance.columns)
            rank=predict(f.loc[idx,columns],theta);rank['ValidationYear']=year_map.get(date,0);rankings.append(rank)
            exits=f.loc[idx,'ExitDate'];assert exits.nunique()==1 and exits.notna().all()
            pending.append({'date':date,'exit':exits.iloc[0],'indices':idx})
            if len(rankings)%100==0:
                pd.DataFrame(states).to_csv(out/'parameter_history.partial.csv',index=False)
                print('v9_sgd',date.date(),'label_days',len(updates),'beta',float(theta[-1]),'seconds',round(time.monotonic()-start,1),flush=True)
    assert not pending
    ranks=pd.concat(rankings,ignore_index=True)
    for year,g in ranks.groupby('ValidationYear'):
        g.to_csv(out/(f'ranks_{year}.csv.gz' if year else 'ranks_warmup.csv.gz'),index=False,compression='gzip')
    pd.DataFrame(states).to_csv(out/'parameter_history.csv',index=False)
    u=pd.DataFrame(updates);u.to_csv(out/'training_updates.csv',index=False)
    with gzip.open(out/'gradient_snapshots.pkl.gz','wb') as handle:pickle.dump(snapshots,handle,protocol=5)
    valid=ranks.loc[ranks.ValidationYear.ne(0)];labels=f[['SignalDate','SecuritiesCode','Target']]
    daily,selected,total=evaluate(valid,labels)
    daily=daily.merge(calendar,on='Date',validate='one_to_one').merge(rank_metrics(valid,labels),on='Date',validate='one_to_one')
    daily.to_csv(out/'daily_spread_returns.csv',index=False);selected.to_csv(out/'selected_200_each_side.csv.gz',index=False,compression='gzip')
    totals={key:int(u[column].sum()) for key,column in {'batches':'Batches','accepted_updates':'AcceptedUpdates','small_gradient_skips':'SmallGradientSkips',
        'no_acceptable_eta_skips':'NoAcceptableEtaSkips','candidate_evaluations':'CandidateEvaluations','hit_upper_eta_batches':'HitUpperEtaBatches',
        'own_zero_feature_nonzero_beta_gradient_batches':'OwnZeroFeatureNonzeroBetaGradientBatches','own_zero_feature_actual_beta_changes':'OwnZeroFeatureActualBetaChanges',
        'financial_gradient_dominates_batches':'FinancialGradientDominatesBatches'}.items()}
    result={'version':'v9','variant':'sgd','validation':total,'label_days':len(updates),'training_stocks':int(u.KnownLabelStocks.sum()),**totals,
        'full_day_loss_increased_days':int((u.FullDayLossAfter>u.FullDayLossBefore+1e-14).sum()),
        'mean_rank_ic':float(daily.RankIC.mean()),'mean_normalized_hard_rank_mse':float(daily.NormalizedHardRankMSE.mean()),
        'final_coefficients':dict(zip(NAMES,map(float,theta))),'last_forecast_date':str(end.date()),'final_parameter_asof':str(states[-1]['Date'].date()),
        'seconds':time.monotonic()-start,'internal_checks_passed':True}
    save(out/'results.json',result)
    dailies={'v9_sgd':daily,'v8_sgd':pd.read_csv(V8/'sgd/daily_spread_returns.csv',parse_dates=['Date']),
        'v7_equal':pd.read_csv(ROOT.parent/'jpx_v7_daily_mse_20260912/v7_equal/daily_spread_returns.csv',parse_dates=['Date'])}
    common=pd.Index(calendar.Date)
    for d in dailies.values():common=common.intersection(d.loc[d.SelectedMissingTargets.eq(0),'Date'])
    comparison={};annual=[]
    for name,d in dailies.items():
        values=d.loc[d.Date.isin(common),'OfficialDailySpread']
        comparison[name]={'common_days':len(values),'official_unannualized_sharpe':float(values.mean()/values.std(ddof=1))}
    for year,g in calendar.groupby('ValidationYear'):
        dates=common.intersection(g.Date);record={'validation_year':int(year),'days':len(dates)}
        for name,d in dailies.items():
            values=d.loc[d.Date.isin(dates),'OfficialDailySpread'];record[name]=float(values.mean()/values.std(ddof=1))
        annual.append(record)
    combined=calendar.copy()
    for name,d in dailies.items():
        cols=['OfficialDailySpread','SelectedMissingTargets']
        combined=combined.merge(d[['Date']+cols].rename(columns={c:f'{name}_{c}' for c in cols}),on='Date',validate='one_to_one')
    combined['CommonScorableDate']=combined.Date.isin(common);combined.to_csv(ROOT/'daily_comparison.csv',index=False)
    save(ROOT/'results.json',{'version':'v9','model':result,'comparison':comparison,'annual':annual,'validation_days':len(calendar),'common_scorable_days':len(common),
        'new_feature':'r * (current operating margin - previous-year same-period operating margin)','tau':1.,'decay_sessions':10,'costs_included':False,'formal_test_used':False})
    paths=[ROOT/'experiment_plan.md',ROOT/'financial_features.py',ROOT/'native_v9.py',ROOT/'soft_rank.cpp',ROOT/'soft_rank.dylib',ROOT/'run_v9.py',ROOT/'preflight.py',ROOT/'audit_features.py',V8/'inputs.pkl']
    save(ROOT/'source_hashes.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    print('DONE',json.dumps(result),flush=True);print('COMPARISON',json.dumps(comparison),flush=True)

if __name__=='__main__':main()
