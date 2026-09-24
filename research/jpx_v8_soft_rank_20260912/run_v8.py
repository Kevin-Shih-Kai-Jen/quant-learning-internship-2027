from pathlib import Path
import sys,json,zipfile,pickle,gzip,time,argparse,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'jpx_v7_daily_mse_20260912'))
from run_v7 import make_x,NAMES,INPUTS,FEATURES,V5,predict_rank_v5,evaluate,save
from native import train
VARIANTS=['daily','sgd','mini128']
TRACE_NAMES=['BatchIndex','BatchSize','LossBefore','LossAfter','Eta','GradInf','GradNormSquared','Candidates','Status','HitUpperEta']

def prepare():
    f=pd.read_pickle(ROOT.parent/'jpx_stock_returns_20260910/features.pkl')
    f=f[['SignalDate','SecuritiesCode','ExitDate','Target']+INPUTS[:-2]]
    f=f.merge(pd.read_pickle(V5/'one_day_returns.pkl'),on=['SignalDate','SecuritiesCode'],validate='one_to_one')
    with zipfile.ZipFile('/Users/coolguy/Desktop/JPX_data/modelling/JPX_data.zip') as z:
        raw=pd.read_csv(z.open('JPX_data/raw/train_files/stock_prices.csv'),usecols=['Date','SecuritiesCode','SupervisionFlag'],parse_dates=['Date']).rename(columns={'Date':'SignalDate'})
    f=f.merge(raw,on=['SignalDate','SecuritiesCode'],validate='one_to_one')
    f=f.loc[~f.SupervisionFlag].sort_values(['SignalDate','SecuritiesCode']).reset_index(drop=True)
    calendar=pd.read_csv(V5/'daily_spread_returns.csv',parse_dates=['Date'])[['Date','ValidationYear']]
    groups={d:idx.to_numpy() for d,idx in f.groupby('SignalDate').groups.items()}
    with (ROOT/'inputs.pkl').open('wb') as handle:pickle.dump((f,make_x(f),groups,calendar),handle,protocol=5)
    print('PREPARED',len(f),len(groups),len(calendar),flush=True)

def rank_metrics(ranks,labels):
    z=ranks.merge(labels.rename(columns={'SignalDate':'Date'}),on=['Date','SecuritiesCode'],validate='one_to_one')
    rows=[]
    for date,g in z.groupby('Date',sort=True):
        g=g.loc[g.Target.notna()].copy();n=len(g)
        true=g.Target.rank(ascending=False,method='average')
        # Official tie breaking remains score descending, code ascending. Re-rank
        # the observed-label subset only for like-for-like rank-error evaluation.
        predicted=g.Rank.rank(ascending=True,method='first')
        score_rank=g.g.rank(ascending=False,method='average')
        ic=float(true.corr(score_rank)) if true.std()>0 and score_rank.std()>0 else np.nan
        rows.append({'Date':date,'KnownLabelStocksForRankMetric':n,'HardRankMSE':float(np.mean((true-predicted)**2)),
            'NormalizedHardRankMSE':float(np.mean(((true-predicted)/(n-1))**2)),'RankIC':ic})
    return pd.DataFrame(rows)

def run(variant):
    out=ROOT/variant;out.mkdir(exist_ok=True);start=time.monotonic()
    with (ROOT/'inputs.pkl').open('rb') as h:f,xall,groups,calendar=pickle.load(h)
    year_map=calendar.set_index('Date').ValidationYear.to_dict();end=calendar.Date.max()
    theta=np.zeros(12);pending=[];states=[];updates=[];snapshots=[];rankings=[]
    with gzip.open(out/'batch_trace.csv.gz','wt') as tracefile:
        tracefile.write('SignalDate,UpdateDate,'+','.join(TRACE_NAMES)+'\n')
        for date,idx in groups.items():
            before_day=theta.copy();due=[b for b in pending if b['exit']<=date]
            pending=[b for b in pending if b['exit']>date]
            assert len(due)<=1
            for batch in due:
                assert batch['date']<batch['exit']<=date
                ix=batch['indices'];known=np.isfinite(f.loc[ix,'Target'].to_numpy());ix=ix[known]
                truth=f.loc[ix,'Target'].rank(ascending=False,method='average').to_numpy()
                n=len(ix);assert n>1
                order=np.random.default_rng(20260912+int(batch['date'].strftime('%Y%m%d'))).permutation(n).astype(np.int32)
                size=n if variant=='daily' else 1 if variant=='sgd' else 128
                before=theta.copy();theta,tr,snap,ss,losses=train(xall[ix],truth,theta,order,size)
                assert int(tr[:,1].sum())==n and theta[0]==0
                accepted=tr[:,8]==0
                assert (tr[accepted,3]<tr[accepted,2]).all()
                assert (tr[accepted,3]<=tr[accepted,2]-1e-4*tr[accepted,4]*tr[accepted,6]+1e-14).all()
                assert (tr[~accepted,4]==0).all()
                prefix=f'{batch["date"].date()},{date.date()},'
                tracefile.writelines(prefix+','.join(format(v,'.17g') for v in row)+'\n' for row in tr)
                snapshots.append({'SignalDate':batch['date'],'UpdateDate':date,'indices':ix,'order':order,'batch_size':size,
                    'snapshot_batches':snap,'theta_before_gradient':ss,'snapshot_trace':tr[snap]})
                updates.append({'Date':date,'SignalDate':batch['date'],'ExitDate':batch['exit'],'ForecastStocks':len(known),
                    'KnownLabelStocks':n,'MissingTargetStocks':int((~known).sum()),'Batches':len(tr),'AcceptedUpdates':int(accepted.sum()),
                    'SmallGradientSkips':int((tr[:,8]==1).sum()),'NoAcceptableEtaSkips':int((tr[:,8]==2).sum()),
                    'CandidateEvaluations':int(tr[:,7].sum()),'HitUpperEtaBatches':int(tr[:,9].sum()),
                    'EtaMinAccepted':float(tr[accepted,4].min()) if accepted.any() else 0.,'EtaMaxAccepted':float(tr[accepted,4].max()) if accepted.any() else 0.,
                    'LastBatchSize':int(tr[-1,1]),'FullDayLossBefore':losses[0],'FullDayLossAfter':losses[1],
                    **{f'Before_{name}':v for name,v in zip(NAMES,before)},**{f'After_{name}':v for name,v in zip(NAMES,theta)}})
            states.append({'Date':date,'NewLabelDays':len(due),'CumulativeLabelDays':len(updates),
                **{f'Before_{name}':v for name,v in zip(NAMES,before_day)},**{f'After_{name}':v for name,v in zip(NAMES,theta)}})
            if date==pd.Timestamp('2020-10-01') or date>end:continue
            rank=predict_rank_v5(f.loc[idx,['SignalDate','SecuritiesCode']+INPUTS],theta)
            rank['ValidationYear']=year_map.get(date,0);rankings.append(rank)
            exits=f.loc[idx,'ExitDate'];assert exits.nunique()==1 and exits.notna().all()
            pending.append({'date':date,'exit':exits.iloc[0],'indices':idx})
            if len(rankings)%100==0:
                pd.DataFrame(states).to_csv(out/'parameter_history.partial.csv',index=False)
                print(variant,date.date(),'label_days',len(updates),'seconds',round(time.monotonic()-start,1),flush=True)
    assert not pending
    ranks=pd.concat(rankings,ignore_index=True)
    for year,g in ranks.groupby('ValidationYear'):
        g.to_csv(out/(f'ranks_{year}.csv.gz' if year else 'ranks_warmup.csv.gz'),index=False,compression='gzip')
    pd.DataFrame(states).to_csv(out/'parameter_history.csv',index=False)
    pd.DataFrame(updates).to_csv(out/'training_updates.csv',index=False)
    with gzip.open(out/'gradient_snapshots.pkl.gz','wb') as h:pickle.dump(snapshots,h,protocol=5)
    valid=ranks.loc[ranks.ValidationYear.ne(0)];labels=f[['SignalDate','SecuritiesCode','Target']]
    daily,selected,total=evaluate(valid,labels)
    daily=daily.merge(calendar,on='Date',validate='one_to_one').merge(rank_metrics(valid,labels),on='Date',validate='one_to_one')
    daily.to_csv(out/'daily_spread_returns.csv',index=False)
    selected.to_csv(out/'selected_200_each_side.csv.gz',index=False,compression='gzip')
    result={'variant':variant,'validation':total,'label_days':len(updates),'training_stocks':sum(u['KnownLabelStocks'] for u in updates),
        'batches':sum(u['Batches'] for u in updates),'accepted_updates':sum(u['AcceptedUpdates'] for u in updates),
        'small_gradient_skips':sum(u['SmallGradientSkips'] for u in updates),'no_acceptable_eta_skips':sum(u['NoAcceptableEtaSkips'] for u in updates),
        'hit_upper_eta_batches':sum(u['HitUpperEtaBatches'] for u in updates),'candidate_evaluations':sum(u['CandidateEvaluations'] for u in updates),
        'full_day_loss_increased_days':int(sum(u['FullDayLossAfter']>u['FullDayLossBefore']+1e-14 for u in updates)),
        'mean_rank_ic':float(daily.RankIC.mean()),'mean_normalized_hard_rank_mse':float(daily.NormalizedHardRankMSE.mean()),
        'final_coefficients':dict(zip(NAMES,map(float,theta))),'last_forecast_date':str(end.date()),'final_parameter_asof':str(states[-1]['Date'].date()),
        'seconds':time.monotonic()-start,'internal_checks_passed':True}
    save(out/'results.json',result);print('DONE',json.dumps(result),flush=True)

def finalize_saved(variant):
    # Recover the JSON summary from complete, already-scored CSV artifacts.
    # No training state is changed and no model is fit again.
    out=ROOT/variant;u=pd.read_csv(out/'training_updates.csv');d=pd.read_csv(out/'daily_spread_returns.csv')
    h=pd.read_csv(out/'parameter_history.csv');valid=d.loc[d.SelectedMissingTargets.eq(0)];s=valid.OfficialDailySpread
    score=float(s.mean()/s.std(ddof=1))
    total={'official_style_unannualized_sharpe':score,'annualized_equivalent_sqrt252':float(score*np.sqrt(252)),
        'signal_days':len(d),'scored_days':len(valid),'unscorable_selected_missing_label_days':int(d.SelectedMissingTargets.gt(0).sum()),
        'ranked_rows':int(d.StocksRanked.sum()),'stocks_per_day_min':int(d.StocksRanked.min()),'stocks_per_day_max':int(d.StocksRanked.max()),
        'fallback_rows':int(d.FallbackStocks.sum()),'selected_fallback_rows':int(d.SelectedFallbackStocks.sum()),
        'unavailable_target_rows':int(d.AllMissingTargets.sum()),'missing_selected_target_rows':int(d.SelectedMissingTargets.sum()),
        'mean_official_spread':float(s.mean()),'sample_std_official_spread':float(s.std(ddof=1)),
        'mean_illustrative_gross_one_return':float(valid.IllustrativeGrossOneReturn.mean())}
    result={'variant':variant,'validation':total,'label_days':len(u),'training_stocks':int(u.KnownLabelStocks.sum()),
        'batches':int(u.Batches.sum()),'accepted_updates':int(u.AcceptedUpdates.sum()),'small_gradient_skips':int(u.SmallGradientSkips.sum()),
        'no_acceptable_eta_skips':int(u.NoAcceptableEtaSkips.sum()),'hit_upper_eta_batches':int(u.HitUpperEtaBatches.sum()),
        'candidate_evaluations':int(u.CandidateEvaluations.sum()),'full_day_loss_increased_days':int((u.FullDayLossAfter>u.FullDayLossBefore+1e-14).sum()),
        'mean_rank_ic':float(d.RankIC.mean()),'mean_normalized_hard_rank_mse':float(d.NormalizedHardRankMSE.mean()),
        'final_coefficients':{n:float(h.iloc[-1]['After_'+n]) for n in NAMES},'last_forecast_date':str(d.Date.max()),
        'final_parameter_asof':str(h.Date.iloc[-1]),'internal_checks_passed':True,'summary_recovered_from_completed_scored_csv':True}
    save(out/'results.json',result);print('FINALIZED',variant,score,flush=True)

def compare():
    dailies={v:pd.read_csv(ROOT/v/'daily_spread_returns.csv',parse_dates=['Date']) for v in VARIANTS}
    for v,path in {'v7_equal':ROOT.parent/'jpx_v7_daily_mse_20260912/v7_equal','v7_jpx':ROOT.parent/'jpx_v7_daily_mse_20260912/v7_jpx','v5':V5}.items():
        dailies[v]=pd.read_csv(path/'daily_spread_returns.csv',parse_dates=['Date'])
    calendar=dailies['daily'][['Date','ValidationYear']];common=pd.Index(calendar.Date)
    for d in dailies.values():common=common.intersection(d.loc[d.SelectedMissingTargets.eq(0),'Date'])
    comparison={};annual=[]
    for v,d in dailies.items():
        s=d.loc[d.Date.isin(common),'OfficialDailySpread']
        comparison[v]={'common_days':len(s),'official_unannualized_sharpe':float(s.mean()/s.std(ddof=1)),'mean_daily_spread':float(s.mean()),'daily_spread_std':float(s.std(ddof=1))}
    for year,g in calendar.groupby('ValidationYear'):
        dates=common.intersection(g.Date);rec={'validation_year':int(year),'days':len(dates)}
        for v,d in dailies.items():
            s=d.loc[d.Date.isin(dates),'OfficialDailySpread'];rec[v]=float(s.mean()/s.std(ddof=1))
        annual.append(rec)
    combined=calendar.copy()
    for v,d in dailies.items():
        cols=['OfficialDailySpread','SelectedMissingTargets']+(['RankIC','NormalizedHardRankMSE'] if v in VARIANTS else [])
        combined=combined.merge(d[['Date']+cols].rename(columns={c:f'{v}_{c}' for c in cols}),on='Date',validate='one_to_one')
    combined['CommonScorableDate']=combined.Date.isin(common);combined.to_csv(ROOT/'daily_comparison.csv',index=False)
    summary={'version':'v8','designated_main_variant':'daily','comparison':comparison,'annual':annual,
        'variants':{v:json.loads((ROOT/v/'results.json').read_text()) for v in VARIANTS},'features':FEATURES,'tau':1.,
        'loss':'mean(((soft_rank_i - true_rank_i)/(N-1))**2), same stock i, no square root',
        'costs_included':False,'formal_test_used':False,'validation_dates':len(calendar),'common_scorable_dates':len(common)}
    save(ROOT/'results.json',summary)
    paths=[ROOT/'experiment_plan.md',ROOT/'run_v8.py',ROOT/'native.py',ROOT/'soft_rank.cpp',ROOT/'soft_rank.dylib',ROOT/'preflight.py',V5/'run_v5.py',V5/'one_day_returns.pkl',ROOT.parent/'jpx_v7_daily_mse_20260912/run_v7.py',ROOT.parent/'jpx_official_ranking_20260912/official_metric.py']
    save(ROOT/'source_hashes.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','compare','finalize']+VARIANTS);a=p.parse_args().action
    if a=='prepare':prepare()
    elif a=='compare':compare()
    elif a=='finalize':
        for variant in VARIANTS:finalize_saved(variant)
    else:run(a)
