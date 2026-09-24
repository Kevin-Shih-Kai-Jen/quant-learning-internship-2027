from pathlib import Path
import sys,json,zipfile,hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
V5=ROOT.parent/'jpx_v5_daily_returns_20260912'
sys.path.insert(0,str(V5))
from run_v5 import predict_rank_v5,FEATURES,INPUTS,BASE_FEATURES,evaluate
def save(name,obj):
    (ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))

def day_loss_gradient(theta,a,b,c):
    v=a@theta
    q=float(theta@v-2*b@theta+c)
    assert q>-1e-10
    loss=float(np.sqrt(max(q,0.)))
    grad=(v-b)/loss if loss>1e-14 else np.zeros_like(theta)
    return loss,grad

def day_step(theta,a,b,c,rate):
    loss,grad=day_loss_gradient(theta,a,b,c)
    norm2=float(grad@grad)
    if norm2<=1e-28:
        return theta.copy(),rate,loss,loss,0,False
    trial=rate*1.5
    for bt in range(60):
        candidate=theta-trial*grad
        candidate_loss,_=day_loss_gradient(candidate,a,b,c)
        if candidate_loss<=loss-1e-4*trial*norm2:
            return candidate,trial,loss,candidate_loss,bt,True
        trial*=.5
    # Numerical stall is reported; never claim a rejected step was applied.
    return theta.copy(),rate,loss,loss,60,False

def replay(theta,days,rate):
    count=0;backtracks=0;accepted=0;stalled=0;largest_change=0.
    for a,b,c in days:
        theta,newrate,before,after,bt,moved=day_step(theta,a,b,c,rate)
        assert after<=before+1e-14
        rate=newrate;count+=1;backtracks+=bt;accepted+=moved;stalled+=bt==60
        largest_change=max(largest_change,before-after)
    return theta,rate,{'ReplayedDays':count,'AcceptedSteps':int(accepted),'Backtracks':backtracks,
        'StalledSteps':int(stalled),'LargestBatchLossDecrease':largest_change,'LastBatchLossBefore':before,
        'LastBatchLossAfter':after,'LastRate':float(rate)}

def frame_x(frame):
    z=frame[INPUTS].replace([np.inf,-np.inf],np.nan).fillna(0).copy()
    for k in [5,22,60]:z[f'P{k}xV{k}']=z[f'T{k}']*z[f'V{k}']
    return np.column_stack([np.ones(len(z)),z[FEATURES].to_numpy()])

def daily_forecast_error(holdings):
    h=holdings.assign(SquaredError=(holdings.g-holdings.Target)**2)
    out=h.groupby('Date').agg(Stocks=('Target','count'),MSE=('SquaredError','mean'))
    out.loc[out.Stocks.ne(400),'MSE']=np.nan
    out['RMSE']=np.sqrt(out.MSE)
    return out.reset_index()

def main():
    f=pd.read_pickle(ROOT.parent/'jpx_stock_returns_20260910/features.pkl')
    ret=pd.read_pickle(V5/'one_day_returns.pkl')
    f=f.merge(ret,on=['SignalDate','SecuritiesCode'],validate='one_to_one')
    with zipfile.ZipFile('/Users/coolguy/Desktop/JPX_data/modelling/JPX_data.zip') as z:
        raw=pd.read_csv(z.open('JPX_data/raw/train_files/stock_prices.csv'),usecols=['Date','SecuritiesCode','SupervisionFlag'],parse_dates=['Date']).rename(columns={'Date':'SignalDate'})
    f=f.merge(raw,on=['SignalDate','SecuritiesCode'],validate='one_to_one')
    f=f.loc[~f.SupervisionFlag].sort_values(['SignalDate','SecuritiesCode']).reset_index(drop=True)
    calendar=pd.read_csv(V5/'daily_spread_returns.csv',parse_dates=['Date'])[['Date','ValidationYear']]
    score_year=calendar.set_index('Date').ValidationYear.to_dict()
    end_signal=calendar.Date.max();closed=pd.Timestamp('2020-10-01')
    groups={d:idx.to_numpy() for d,idx in f.groupby('SignalDate').groups.items()}
    xall=frame_x(f);lookup=f.set_index(['SignalDate','SecuritiesCode']).index
    theta=np.zeros(len(FEATURES)+1);rate=.1
    days=[];pending=[];rankings=[];cohorts=[];updates=[];states=[];releases=[]
    for date,indices in groups.items():
        if date==closed:continue
        before_update=theta.copy();added=0
        remain=[]
        for batch in pending:
            if batch['exit']>date:remain.append(batch);continue
            idx=batch['indices'];label=f.loc[idx,'Target'].to_numpy()
            assert batch['date']<batch['exit']<=date
            available=np.isfinite(label).all()
            releases.append({'SignalDate':batch['date'],'ExitDate':batch['exit'],'AddedAsOf':date,'Included':bool(available),'MissingTargets':int((~np.isfinite(label)).sum())})
            if not available:continue
            x=xall[idx];assert len(x)==400
            days.append((x.T@x/400,x.T@label/400,float(label@label/400)));added+=1
        pending=remain
        if added:
            rate_before=rate
            theta,rate,info=replay(theta,days,rate)
            updates.append({'Date':date,'AddedDays':added,'TrainingDays':len(days),'TrainingRows':len(days)*400,'RateBefore':rate_before,**info})
        state={'Date':date,'TrainingDays':len(days),'AddedDays':added,
            **{f'Before_{n}':float(t) for n,t in zip(['alpha']+FEATURES,before_update)},
            **{f'After_{n}':float(t) for n,t in zip(['alpha']+FEATURES,theta)}}
        states.append(state)
        if date>end_signal:continue
        obs=f.loc[indices,['SignalDate','SecuritiesCode']+INPUTS]
        rank=predict_rank_v5(obs,theta)
        rank['ValidationYear']=score_year.get(date,0)
        if date in score_year:rankings.append(rank)
        take=pd.concat([rank.head(200).assign(Side='long'),rank.tail(200).iloc[::-1].assign(Side='short')])
        assert len(take)==400 and take.SecuritiesCode.nunique()==400
        take['SideRank']=np.tile(np.arange(1,201),2)
        keys=pd.MultiIndex.from_arrays([take.Date,take.SecuritiesCode],names=['SignalDate','SecuritiesCode'])
        idx=lookup.get_indexer(keys);assert (idx>=0).all()
        exits=f.loc[idx,'ExitDate'];assert exits.nunique()==1 and exits.notna().all()
        take['ExitDate']=exits.iloc[0];take['SourceIndex']=idx;cohorts.append(take)
        pending.append({'date':date,'exit':exits.iloc[0],'indices':idx})
        if date.month==12 and date.day>=27 or len(states)%250==0:
            print('PROGRESS',str(date.date()),'history days',len(days),'loss',updates[-1]['LastBatchLossAfter'] if updates else None,flush=True)
    assert not pending
    ranked=pd.concat(rankings,ignore_index=True)
    cohort=pd.concat(cohorts,ignore_index=True)
    # Persist all forecasts without future Target before the retrospective scorer sees labels.
    for year,g in ranked.groupby('ValidationYear'):g.to_csv(ROOT/f'ranks_{year}.csv.gz',index=False,compression='gzip')
    cohort.to_csv(ROOT/'all_selected_forecasts.csv.gz',index=False,compression='gzip')
    pd.DataFrame(states).to_csv(ROOT/'parameter_history.csv',index=False)
    pd.DataFrame(updates).to_csv(ROOT/'training_updates.csv',index=False)
    pd.DataFrame(releases).to_csv(ROOT/'label_releases.csv',index=False)
    labels=f[['SignalDate','SecuritiesCode','Target']]
    daily,selected,total=evaluate(ranked,labels)
    daily=daily.merge(calendar,on='Date',validate='one_to_one')
    daily=daily.merge(daily_forecast_error(selected).rename(columns={'MSE':'ForecastMSE','RMSE':'ForecastRMSE'}).drop(columns='Stocks'),on='Date',validate='one_to_one')
    daily.to_csv(ROOT/'daily_spread_returns.csv',index=False)
    selected.to_csv(ROOT/'selected_200_each_side.csv.gz',index=False,compression='gzip')
    olddaily=pd.read_csv(V5/'daily_spread_returns.csv',parse_dates=['Date'])
    oldselected=pd.read_csv(V5/'selected_200_each_side.csv.gz',parse_dates=['Date'])
    common=pd.Index(daily.loc[daily.SelectedMissingTargets.eq(0),'Date']).intersection(olddaily.loc[olddaily.SelectedMissingTargets.eq(0),'Date'])
    annual=[]
    for year in sorted(score_year.values()):
        if any(y['validation_year']==year for y in annual):continue
        years=calendar.loc[calendar.ValidationYear.eq(year),'Date'];dates=common.intersection(years)
        rec={'validation_year':year,'common_days':len(dates)}
        for version,ds,hs in [('v6',daily,selected),('v5',olddaily,oldselected)]:
            s=ds.loc[ds.Date.isin(dates),'OfficialDailySpread']
            rec[version]={'official_unannualized_sharpe':float(s.mean()/s.std(ddof=1))}
        annual.append(rec)
    comparisons={}
    for version,ds,hs in [('v6',daily,selected),('v5',olddaily,oldselected)]:
        s=ds.loc[ds.Date.isin(common),'OfficialDailySpread']
        comparisons[version]={'official_unannualized_sharpe':float(s.mean()/s.std(ddof=1))}
    forecast_comparison=daily_forecast_error(selected).rename(columns={'MSE':'v6_MSE','RMSE':'v6_RMSE'}).drop(columns='Stocks').merge(daily_forecast_error(oldselected).rename(columns={'MSE':'v5_MSE','RMSE':'v5_RMSE'}).drop(columns='Stocks'),on='Date',validate='one_to_one')
    forecast_comparison.to_csv(ROOT/'daily_forecast_loss_comparison.csv',index=False)
    result={'version':'v6','features':FEATURES,'loss':'each historical day separately: sqrt(mean of squared errors across its 400 selected stocks); no across-date averaging',
        'optimizer':'one chronological replay epoch over the expanding historical window per label-release date; one daily-batch RMSE gradient step per historical day with Armijo backtracking; one zero initialization',
        'raw_feature_standardization':False,'formal_test_used':False,'costs_included':False,
        'validation_total':total,'common_scored_days':len(common),'comparison':comparisons,'annual':annual,
        'warmup_start':str(min(groups).date()),'validation_start':str(calendar.Date.min().date()),'last_forecast_date':str(end_signal.date()),
        'final_parameter_asof':str(states[-1]['Date'].date()),'training_days':len(days),'training_rows':400*len(days),
        'excluded_incomplete_label_cohorts':sum(not z['Included'] for z in releases),'update_days':len(updates),
        'total_daily_batch_visits':sum(u['ReplayedDays'] for u in updates),
        'accepted_gradient_steps':sum(u['AcceptedSteps'] for u in updates),'stalled_steps':sum(u['StalledSteps'] for u in updates),
        'final_last_batch_rmse_after_step':updates[-1]['LastBatchLossAfter'],
        'final_coefficients':dict(zip(['alpha']+FEATURES,map(float,theta))),'internal_checks_passed':True}
    save('results.json',result)
    save('source_hashes.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'experiment_plan.md',ROOT/'run_v6.py',V5/'run_v5.py',V5.parent/'jpx_official_ranking_20260912/official_metric.py']})
    print('TOTAL',json.dumps(result),flush=True)

if __name__=='__main__':main()
