from pathlib import Path
import sys,json,zipfile,hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
V5=ROOT.parent/'jpx_v5_daily_returns_20260912'
sys.path.insert(0,str(V5))
from run_v5 import predict_rank_v5,FEATURES,INPUTS,evaluate
NAMES=['alpha']+FEATURES
VARIANTS=['v7_equal','v7_jpx']

def save(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))

def make_x(frame):
    v=frame[INPUTS].replace([np.inf,-np.inf],np.nan).fillna(0).copy()
    for k in [5,22,60]:v[f'P{k}xV{k}']=v[f'T{k}']*v[f'V{k}']
    return np.column_stack([np.ones(len(v)),v[FEATURES].to_numpy()])

def raw_weights(n,variant):
    assert n>=400
    if variant=='v7_equal':return np.ones(n)
    assert variant=='v7_jpx'
    w=np.zeros(n);w[:200]=np.linspace(2.,1.,200);w[-200:]=np.linspace(1.,2.,200)
    return w

def mse_step(theta,x,y,weights):
    assert np.isfinite(x).all() and np.isfinite(y).all()
    assert (weights>=0).all() and abs(weights.sum()-1)<1e-12
    err=x@theta-y
    gram=x.T@(weights[:,None]*x)
    gradient=2*x.T@(weights*err)
    largest=float(np.linalg.eigvalsh(gram)[-1]);assert largest>0
    rate=1/(2*largest)
    new=theta-rate*gradient
    before=float(weights@err**2);after=float(weights@(x@new-y)**2)
    assert after<=before+1e-12*max(1,before)
    return new,gradient,rate,before,after,largest

def daily_losses(ranks,labels):
    merged=ranks.merge(labels.rename(columns={'SignalDate':'Date'}),on=['Date','SecuritiesCode'],validate='one_to_one')
    merged['SE']=(merged.g-merged.Target)**2
    rows=[]
    for date,g in merged.groupby('Date',sort=True):
        known=g.Target.notna();valid=g.loc[known]
        pos=valid.TrainingRawWeight.gt(0)
        rows.append({'Date':date,'AllStockForecastMSE':float(valid.SE.mean()),
            'TrainingWeightForecastMSE':float(np.average(valid.loc[pos,'SE'],weights=valid.loc[pos,'TrainingRawWeight'])),
            'MissingTargetStocks':int((~known).sum())})
    return pd.DataFrame(rows)

def run_variant(variant,f,xall,groups,lookup,calendar):
    out=ROOT/variant;out.mkdir(exist_ok=True)
    year_map=calendar.set_index('Date').ValidationYear.to_dict()
    end_signal=calendar.Date.max();closed=pd.Timestamp('2020-10-01')
    theta=np.zeros(12);pending=[];rankings=[];states=[];updates=[]
    for date,indices in groups.items():
        before_day=theta.copy();due=[b for b in pending if b['exit']<=date]
        pending=[b for b in pending if b['exit']>date]
        assert len(due)<=1
        for batch in due:
            assert batch['date']<batch['exit']<=date
            idx=batch['indices'];y=f.loc[idx,'Target'].to_numpy();raw=batch['weights']
            known=np.isfinite(y);used=known&(raw>0)
            assert used.any()
            weights=raw[used]/raw[used].sum();x=xall[idx[used]]
            forecast_loss=float(weights@(batch['forecast'][used]-y[used])**2)
            theta,grad,rate,loss_before,loss_after,largest=mse_step(theta,x,y[used],weights)
            updates.append({'Date':date,'SignalDate':batch['date'],'ExitDate':batch['exit'],
                'ForecastStocks':len(idx),'KnownLabelStocks':int(known.sum()),'PositiveWeightStocks':int(used.sum()),
                'MissingTargetStocks':int((~known).sum()),'MissingPositiveWeightStocks':int(((~known)&(raw>0)).sum()),
                'TrainingWeightSumBeforeNormalization':float(raw[used].sum()),'ForecastMSE':forecast_loss,
                'MSEBeforeStep':loss_before,'MSEAfterStep':loss_after,'LearningRate':rate,'LargestGramEigenvalue':largest,
                **{f'Gradient_{n}':float(v) for n,v in zip(NAMES,grad)}})
        states.append({'Date':date,'UpdatesToday':len(due),'CumulativeUpdates':len(updates),
            **{f'Before_{n}':float(v) for n,v in zip(NAMES,before_day)},
            **{f'After_{n}':float(v) for n,v in zip(NAMES,theta)}})
        if date==closed or date>end_signal:continue
        obs=f.loc[indices,['SignalDate','SecuritiesCode']+INPUTS]
        rank=predict_rank_v5(obs,theta)
        rank['ValidationYear']=year_map.get(date,0)
        weights=raw_weights(len(rank),variant)
        rank['TrainingRawWeight']=weights
        rankings.append(rank)
        keys=pd.MultiIndex.from_arrays([rank.Date,rank.SecuritiesCode],names=['SignalDate','SecuritiesCode'])
        idx=lookup.get_indexer(keys);assert (idx>=0).all()
        exits=f.loc[idx,'ExitDate'];assert exits.nunique()==1 and exits.notna().all()
        pending.append({'date':date,'exit':exits.iloc[0],'indices':idx,'weights':weights,'forecast':rank.g.to_numpy().copy()})
    assert not pending
    ranks=pd.concat(rankings,ignore_index=True)
    for year,g in ranks.groupby('ValidationYear'):
        g.to_csv(out/(f'ranks_{year}.csv.gz' if year else 'ranks_warmup.csv.gz'),index=False,compression='gzip')
    pd.DataFrame(states).to_csv(out/'parameter_history.csv',index=False)
    pd.DataFrame(updates).to_csv(out/'training_updates.csv',index=False)
    valid=ranks.loc[ranks.ValidationYear.ne(0)]
    labels=f[['SignalDate','SecuritiesCode','Target']]
    daily,selected,total=evaluate(valid,labels)
    daily=daily.merge(calendar,on='Date',validate='one_to_one').merge(daily_losses(valid,labels),on='Date',validate='one_to_one')
    daily.to_csv(out/'daily_spread_returns.csv',index=False)
    selected.to_csv(out/'selected_200_each_side.csv.gz',index=False,compression='gzip')
    result={'variant':variant,'validation':total,'daily_updates':len(updates),'initial_parameter':'zeros once, 2017-01-04',
        'no_history_replay':True,'one_step_per_date':True,'training_loss':'ordinary daily MSE, no square root or across-date average',
        'training_weight_rule':'all observed-label stocks equal' if variant=='v7_equal' else 'own prediction-time extreme 200 on each side, raw positive magnitudes 2 to 1, zero outside, normalize total',
        'total_positive_weight_training_rows':sum(u['PositiveWeightStocks'] for u in updates),
        'dates_with_missing_positive_weight_targets':sum(u['MissingPositiveWeightStocks']>0 for u in updates),
        'missing_positive_weight_targets':sum(u['MissingPositiveWeightStocks'] for u in updates),
        'final_parameter_asof':str(states[-1]['Date'].date()),'last_forecast_date':str(end_signal.date()),
        'final_coefficients':dict(zip(NAMES,map(float,theta))),'all_internal_checks_passed':True}
    save(out/'results.json',result)
    print('DONE',variant,json.dumps(result),flush=True)
    return result,daily

def main():
    f=pd.read_pickle(ROOT.parent/'jpx_stock_returns_20260910/features.pkl')
    f=f[['SignalDate','SecuritiesCode','ExitDate','Target']+INPUTS[:-2]]
    f=f.merge(pd.read_pickle(V5/'one_day_returns.pkl'),on=['SignalDate','SecuritiesCode'],validate='one_to_one')
    with zipfile.ZipFile('/Users/coolguy/Desktop/JPX_data/modelling/JPX_data.zip') as z:
        raw=pd.read_csv(z.open('JPX_data/raw/train_files/stock_prices.csv'),usecols=['Date','SecuritiesCode','SupervisionFlag'],parse_dates=['Date']).rename(columns={'Date':'SignalDate'})
    f=f.merge(raw,on=['SignalDate','SecuritiesCode'],validate='one_to_one')
    f=f.loc[~f.SupervisionFlag].sort_values(['SignalDate','SecuritiesCode']).reset_index(drop=True)
    calendar=pd.read_csv(V5/'daily_spread_returns.csv',parse_dates=['Date'])[['Date','ValidationYear']]
    groups={d:idx.to_numpy() for d,idx in f.groupby('SignalDate').groups.items()}
    lookup=f.set_index(['SignalDate','SecuritiesCode']).index;xall=make_x(f)
    results={};dailies={}
    for variant in VARIANTS:results[variant],dailies[variant]=run_variant(variant,f,xall,groups,lookup,calendar)
    dailies['v5']=pd.read_csv(V5/'daily_spread_returns.csv',parse_dates=['Date'])
    common=pd.Index(calendar.Date)
    for d in dailies.values():common=common.intersection(d.loc[d.SelectedMissingTargets.eq(0),'Date'])
    comparison={};annual=[]
    for variant,d in dailies.items():
        s=d.loc[d.Date.isin(common),'OfficialDailySpread']
        comparison[variant]={'common_days':len(s),'official_unannualized_sharpe':float(s.mean()/s.std(ddof=1))}
    for year,g in calendar.groupby('ValidationYear'):
        dates=common.intersection(g.Date);record={'validation_year':int(year),'days':len(dates)}
        for variant,d in dailies.items():
            s=d.loc[d.Date.isin(dates),'OfficialDailySpread'];record[variant]=float(s.mean()/s.std(ddof=1))
        annual.append(record)
    daily_comparison=calendar.copy()
    for variant,d in dailies.items():
        columns=['OfficialDailySpread','SelectedMissingTargets']+(['AllStockForecastMSE','TrainingWeightForecastMSE'] if variant!='v5' else [])
        daily_comparison=daily_comparison.merge(d[['Date']+columns].rename(columns={c:f'{variant}_{c}' for c in columns}),on='Date',validate='one_to_one')
    daily_comparison.to_csv(ROOT/'daily_comparison.csv',index=False)
    summary={'main_variant':'v7_equal','reference_variant':'v7_jpx','features':FEATURES,'step_size_rule':'1/(2*largest eigenvalue of daily weighted X transpose X)',
        'common_days':len(common),'comparison':comparison,'annual':annual,'variants':results,'costs_included':False,'formal_test_used':False,
        'official_metric_source':'https://www.kaggle.com/code/smeitoma/jpx-competition-metric-definition'}
    save(ROOT/'results.json',summary)
    save(ROOT/'source_hashes.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'experiment_plan.md',ROOT/'run_v7.py',V5/'run_v5.py',V5/'one_day_returns.pkl',V5.parent/'jpx_official_ranking_20260912/official_metric.py']})
    print('COMPARISON',json.dumps(comparison),flush=True)

if __name__=='__main__':main()
