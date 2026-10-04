"""Build full candidate evidence, causal annual selectors and comparable metrics."""
import io
import json
import shutil
import sqlite3
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from common import EXP, REPO, SOURCE, SOURCE_MODELS, OLD, INPUT, OUT, CONFIG, ORDERS, YEARS, save, sha, hashes

sys.path.insert(0,str(REPO/'research/jpx_official_ranking_20260912'))
from official_metric import calc_spread_return_sharpe


def stats(daily):
    s=daily.OfficialDailySpread.dropna()
    sd=float(s.std(ddof=1))
    return {'days':len(daily),'scored_days':len(s),
            'sharpe':float(s.mean()/sd) if len(s)>1 and sd>0 else None,
            'mean_daily_rank_ic':float(daily.RankIC.mean()) if daily.RankIC.notna().any() else None,
            'rank_ic_days':int(daily.RankIC.notna().sum()),
            'mean_spread':float(s.mean()) if len(s) else None,
            'std_spread':sd if np.isfinite(sd) else None,
            'mean_daily_target_mse':float(daily.ForecastMSE.mean()) if 'ForecastMSE' in daily else None}


def choose_order(dailies, cal, cutoff, year):
    """No returns whose labels have not matured at the selection cutoff are eligible."""
    history=(cal.ValidationYear.to_numpy()<year)&(cal.ExitDate.to_numpy()<=np.datetime64(cutoff))
    common=history.copy()
    for daily in dailies: common &= np.isfinite(daily.OfficialDailySpread.to_numpy())
    scored=[]
    if not common.any():
        oi=ORDERS.index(tuple(CONFIG['initial_pq']))
        return oi, {'rule':'initial_pq_no_prior_oos','eligible_days':0,'candidate_metrics':[]}
    for oi,daily in enumerate(dailies):
        m=stats(daily.loc[common]);p,q=ORDERS[oi]
        scored.append({'oi':oi,'p':p,'q':q,**m})
    def key(row):
        sr=row['sharpe'] if row['sharpe'] is not None else -np.inf
        ic=row['mean_daily_rank_ic'] if row['mean_daily_rank_ic'] is not None else -np.inf
        return (-sr,-ic,row['p']+row['q'],row['p'],row['q'])
    selected=min(scored,key=key)
    return selected['oi'],{'rule':'past_matured_oos_sharpe_then_ic','eligible_days':int(common.sum()),
                          'last_eligible_signal':str(cal.loc[common,'Date'].max().date()),
                          'max_exit_date':str(cal.loc[common,'ExitDate'].max().date()),
                          'unmatured_prior_fold_days':int(((cal.ValidationYear.to_numpy()<year)&~history).sum()),
                          'candidate_metrics':scored}


def load_context():
    labels=pd.read_pickle(INPUT/'labels.pkl').sort_values(['Date','SecuritiesCode']).reset_index(drop=True)
    cal=pd.read_csv(INPUT/'calendar.csv',parse_dates=['Date','EntryDate','ExitDate'])
    with np.load(INPUT/'prices.npz') as z: prices={k:z[k] for k in z.files}
    with np.load(INPUT/'prediction_keys.npz') as z:
        np.testing.assert_array_equal(labels.Date.to_numpy(dtype='datetime64[D]'),z['date'])
        np.testing.assert_array_equal(labels.SecuritiesCode.to_numpy(),z['code'])
        np.testing.assert_allclose(labels.Target.to_numpy(),z['target'],rtol=0,atol=0,equal_nan=True)
    date_ix=pd.Index(cal.Date).get_indexer(labels.Date)
    code_ix=pd.Index(prices['codes']).get_indexer(labels.SecuritiesCode)
    assert (date_ix>=0).all() and (code_ix>=0).all() and len(labels)==1864363 and len(cal)==953
    boundaries=np.r_[0,np.flatnonzero(np.diff(date_ix))+1,len(labels)]
    assert len(boundaries)==954
    target=labels.Target.to_numpy()
    target_ranks=[]
    for lo,hi in zip(boundaries[:-1],boundaries[1:]):
        t=target[lo:hi];known=np.isfinite(t);target_ranks.append(rankdata(t[known]))
    return {'labels':labels,'cal':cal,'prices':prices,'date_ix':date_ix,'code_ix':code_ix,
            'boundaries':boundaries,'target':target,'target_ranks':target_ranks,
            'row_codes':labels.SecuritiesCode.to_numpy(),'row_years':labels.ValidationYear.to_numpy()}


def metrics(score,fallback,ctx,save_weights=False):
    assert np.isfinite(score).all()
    weights=np.linspace(2.,1.,200)
    ranks=np.empty(len(score),dtype=np.int16)
    hold=np.zeros((len(ctx['cal']),len(ctx['prices']['codes'])),dtype=np.float32) if save_weights else None
    daily=[]
    for day,(lo,hi) in enumerate(zip(ctx['boundaries'][:-1],ctx['boundaries'][1:])):
        s=score[lo:hi];t=ctx['target'][lo:hi];fb=fallback[lo:hi]
        ix=np.lexsort((ctx['row_codes'][lo:hi],-s))
        ranks[lo+ix]=np.arange(hi-lo,dtype=np.int16)
        selected=np.r_[ix[:200],ix[-200:][::-1]]
        missing=int((~np.isfinite(t[selected])).sum())
        spread=(t[ix[:200]]@weights-t[ix[-200:][::-1]]@weights)/weights.mean() if not missing else np.nan
        known=np.isfinite(t)
        a=rankdata(s[known]);b=ctx['target_ranks'][day]
        ic=float(np.corrcoef(a,b)[0,1]) if np.ptp(a)>0 and np.ptp(b)>0 else np.nan
        daily.append({'Date':ctx['cal'].Date.iloc[day],'ValidationYear':int(ctx['cal'].ValidationYear.iloc[day]),
            'OfficialDailySpread':spread,'RankIC':ic,'ForecastMSE':float(np.mean((s[known]-t[known])**2)),
            'StocksRanked':hi-lo,'SelectedMissingTargets':missing,'FallbackStocks':int(fb.sum()),
            'SelectedFallbackStocks':int(fb[selected].sum()),'DistinctScores':len(np.unique(s))})
        if save_weights:
            stock_ix=ctx['code_ix'][lo:hi]
            hold[day,stock_ix[ix[:200]]]=weights/weights.sum()
            hold[day,stock_ix[ix[-200:][::-1]]]=-weights/weights.sum()
    daily=pd.DataFrame(daily)
    return daily,ranks,hold


def official_check(daily,ranks,ctx):
    eligible=daily.SelectedMissingTargets.eq(0)
    frame=ctx['labels'][['Date','Target']].copy();frame['Rank']=ranks
    mask=eligible.to_numpy()[ctx['date_ix']]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result=float(calc_spread_return_sharpe(frame.loc[mask,['Date','Rank','Target']]))
    error=abs(result-stats(daily.loc[eligible])['sharpe'])
    assert error<1e-12,error
    return error


def build_candidates(ctx):
    con=sqlite3.connect(f'file:{OUT}/new_fits.sqlite?mode=ro',uri=True)
    expected=json.loads((EXP/'adf_summary.json').read_text())['new_fits']
    assert con.execute('SELECT COUNT(*) FROM fits').fetchone()[0]==expected
    decisions=json.loads((OUT/'adf_decisions.json').read_text())['records']
    d_by_year=np.full((len(YEARS),len(ctx['prices']['codes'])),-1,dtype=np.int8)
    for r in decisions:
        if r['chosen_d'] is not None:d_by_year[YEARS.index(r['year']),r['ci']]=r['chosen_d']
    all_summaries=[]
    for oi,(p,q) in enumerate(ORDERS):
        model=f'p{p}_q{q}'
        baseline_dest=OUT/'fixed_d1'/model;baseline_dest.mkdir(parents=True,exist_ok=True)
        adf_dest=OUT/'adf'/model;adf_dest.mkdir(parents=True,exist_ok=True)
        if (adf_dest/'summary.json').exists():
            all_summaries.append(json.loads((adf_dest/'summary.json').read_text()));continue
        with np.load(SOURCE_MODELS/model/'predictions.npz') as z:
            oldscore=z['score'];oldfb=z['fallback'];oldranks=z['rank']
        # Retain the independently versioned source predictions for full reproduction.
        shutil.copyfile(SOURCE_MODELS/model/'predictions.npz',baseline_dest/'predictions.npz')
        shutil.copyfile(SOURCE_MODELS/model/'parameters.csv.gz',baseline_dest/'parameters.csv.gz')
        olddaily=pd.read_csv(OLD/'models'/model/'daily.csv',parse_dates=['Date'])
        np.testing.assert_array_equal(olddaily.Date,ctx['cal'].Date)
        diff=(oldscore-ctx['target'])**2
        known=np.isfinite(ctx['target'])
        olddaily['ForecastMSE']=np.add.reduceat(np.where(known,diff,0.),ctx['boundaries'][:-1])/np.add.reduceat(known.astype(int),ctx['boundaries'][:-1])
        olddaily.to_csv(baseline_dest/'daily.csv',index=False)
        # Every source rank is checked on deterministic dates, not only winning models.
        for day in [0,245,486,728,952]:
            lo,hi=ctx['boundaries'][day:day+2]
            ix=np.lexsort((ctx['row_codes'][lo:hi],-oldscore[lo:hi]))
            np.testing.assert_array_equal(oldranks[lo:hi][ix],np.arange(hi-lo))
        matrix=np.zeros((len(ctx['cal']),len(ctx['prices']['codes'])))
        fbmatrix=np.ones(matrix.shape,dtype=bool)
        matrix[ctx['date_ix'],ctx['code_ix']]=oldscore
        fbmatrix[ctx['date_ix'],ctx['code_ix']]=oldfb
        for yi,year in enumerate(YEARS):
            rows=np.flatnonzero(ctx['cal'].ValidationYear.to_numpy()==year)
            missing=np.flatnonzero(d_by_year[yi]<0)
            matrix[np.ix_(rows,missing)]=0.;fbmatrix[np.ix_(rows,missing)]=True
        fit_statuses={};parameter_rows=[]
        for yi,ci,status,audit,payload in con.execute('SELECT yi,ci,status,audit,payload FROM fits WHERE oi=? ORDER BY yi,ci',(oi,)):
            a=json.loads(audit)
            assert a['d']==d_by_year[yi,ci] and a['train_last']<a['first_signal']
            pred=np.load(io.BytesIO(payload))
            rows=np.flatnonzero(ctx['cal'].ValidationYear.to_numpy()==YEARS[yi])
            matrix[rows,ci]=pred['score'];fbmatrix[rows,ci]=pred['fallback']
            fit_statuses[status]=fit_statuses.get(status,0)+1
            parameter_rows.append({'year':YEARS[yi],'code':int(ctx['prices']['codes'][ci]),'d':a['d'],
                                   'status':status,**a.get('params',{})})
        score=matrix[ctx['date_ix'],ctx['code_ix']];fallback=fbmatrix[ctx['date_ix'],ctx['code_ix']]
        chosen_d=np.array([d_by_year[YEARS.index(int(y)),int(ci)] for y,ci in zip(ctx['row_years'],ctx['code_ix'])],dtype=np.int8)
        np.testing.assert_array_equal(score[chosen_d==1],oldscore[chosen_d==1])
        np.testing.assert_array_equal(fallback[chosen_d==1],oldfb[chosen_d==1])
        assert (score[chosen_d<0]==0).all() and fallback[chosen_d<0].all()
        daily,ranks,_=metrics(score,fallback,ctx)
        error=official_check(daily,ranks,ctx)
        np.savez_compressed(adf_dest/'predictions.npz',score=score,rank=ranks,fallback=fallback)
        daily.to_csv(adf_dest/'daily.csv',index=False)
        pd.DataFrame(parameter_rows).to_csv(adf_dest/'new_parameters.csv.gz',index=False,compression='gzip')
        summary={'p':p,'q':q,'raw_metrics':stats(daily),'new_fit_statuses':fit_statuses,
                 'official_metric_error':error,'d1_reuse_bitwise_equal':True,
                 'fallback_stock_days':int(fallback.sum()),'fallback_fraction':float(fallback.mean()),
                 'selected_fallback_stock_days':int(daily.SelectedFallbackStocks.sum()),
                 'constant_score_days':int(daily.DistinctScores.eq(1).sum())}
        save(adf_dest/'summary.json',summary);all_summaries.append(summary)
        print(json.dumps({'evaluated':model,'adf_sharpe':summary['raw_metrics']['sharpe'],'d1_sharpe':stats(olddaily)['sharpe']}),flush=True)
    con.close();save(OUT/'candidate_summaries.json',all_summaries)


def selector(family,ctx):
    dailies=[pd.read_csv(OUT/family/f'p{p}_q{q}'/'daily.csv',parse_dates=['Date']) for p,q in ORDERS]
    score=np.zeros(len(ctx['labels']));fallback=np.ones(len(score),dtype=bool)
    choices=[]
    for year in YEARS:
        first=int(ctx['prices']['valid_positions'][ctx['prices']['years']==year][0])
        cutoff=ctx['prices']['dates'][first-1]
        oi,info=choose_order(dailies,ctx['cal'],cutoff,year)
        p,q=ORDERS[oi]
        if info['eligible_days']:
            assert np.datetime64(info['max_exit_date'])<=cutoff
        choices.append({'family':family,'year':year,'cutoff':str(cutoff),'p':p,'q':q,**info})
        with np.load(OUT/family/f'p{p}_q{q}'/'predictions.npz') as z:
            rowmask=ctx['row_years']==year
            score[rowmask]=z['score'][rowmask];fallback[rowmask]=z['fallback'][rowmask]
        # Each whole realized year must equal that candidate's causal daily path.
    daily,ranks,hold=metrics(score,fallback,ctx,True)
    error=official_check(daily,ranks,ctx)
    dest=OUT/'selectors'/family;dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/'predictions.npz',score=score,rank=ranks,fallback=fallback)
    np.savez_compressed(dest/'holdings.npz',weights=hold,dates=ctx['cal'].Date.to_numpy(dtype='datetime64[D]'),codes=ctx['prices']['codes'])
    daily.to_csv(dest/'daily.csv',index=False)
    save(dest/'choices.json',choices)
    save(dest/'summary.json',{'raw_metrics':stats(daily),'official_metric_error':error})
    return daily,choices,hold


def baseline(ctx):
    frame=pd.read_pickle(INPUT/'baseline.pkl').sort_values(['Date','SecuritiesCode']).reset_index(drop=True)
    pd.testing.assert_frame_equal(frame[['Date','SecuritiesCode']],ctx['labels'][['Date','SecuritiesCode']])
    score=frame.g.to_numpy();fallback=np.zeros(len(score),dtype=bool)
    daily,ranks,hold=metrics(score,fallback,ctx,True)
    np.testing.assert_array_equal(ranks,frame.Rank.to_numpy())
    olddaily=pd.read_csv(OLD/'baseline_daily.csv',parse_dates=['Date'])
    np.testing.assert_allclose(daily.OfficialDailySpread,olddaily.OfficialDailySpread,rtol=1e-12,atol=1e-12)
    error=official_check(daily,ranks,ctx)
    dest=OUT/'baseline';dest.mkdir(exist_ok=True)
    np.savez_compressed(dest/'predictions.npz',score=score,rank=ranks)
    np.savez_compressed(dest/'holdings.npz',weights=hold,dates=ctx['cal'].Date.to_numpy(dtype='datetime64[D]'),codes=ctx['prices']['codes'])
    daily.to_csv(dest/'daily.csv',index=False)
    return daily,hold,error


def matched_controls(choices,ctx):
    decisions=json.loads((OUT/'adf_decisions.json').read_text())['records']
    missing={(r['year'],r['ci']) for r in decisions if r['chosen_d'] is None}
    availability_fallback=np.array([(int(y),int(ci)) in missing for y,ci in zip(ctx['row_years'],ctx['code_ix'])])
    scores=np.zeros(len(ctx['labels']));fallback=np.ones(len(scores),dtype=bool)
    for choice in choices:
        p,q,year=choice['p'],choice['q'],choice['year']
        with np.load(OUT/'fixed_d1'/f'p{p}_q{q}'/'predictions.npz') as z:
            mask=ctx['row_years']==year;scores[mask]=z['score'][mask];fallback[mask]=z['fallback'][mask]
    paths={}
    for apply_mask,name in [(False,'d1_at_adf_orders'),(True,'d1_at_adf_orders_matched_adf_coverage')]:
        s=scores.copy();fb=fallback.copy()
        if apply_mask:s[availability_fallback]=0.;fb[availability_fallback]=True
        daily,ranks,_=metrics(s,fb,ctx)
        error=official_check(daily,ranks,ctx)
        dest=OUT/'selectors'/name;dest.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest/'predictions.npz',score=s,rank=ranks,fallback=fb)
        daily.to_csv(dest/'daily.csv',index=False)
        save(dest/'summary.json',{'raw_metrics':stats(daily),'official_metric_error':error,
                                 'diagnostic_only':True,'chosen_orders_inherited_from_adf_selector':True})
        paths[name]=daily
    return paths


def bootstrap(values,reference_index):
    rng=np.random.default_rng(CONFIG['seed']);n=len(values)
    observed=values.mean(axis=0)/values.std(axis=0,ddof=1)
    diff=observed-observed[reference_index]
    boot=[]
    for _ in range(CONFIG['bootstrap_replicates']):
        starts=rng.integers(0,n,size=int(np.ceil(n/CONFIG['bootstrap_block_days'])))
        ix=((starts[:,None]+np.arange(CONFIG['bootstrap_block_days']))%n).ravel()[:n]
        sample=values[ix];sr=sample.mean(axis=0)/sample.std(axis=0,ddof=1)
        boot.append(sr-sr[reference_index])
    boot=np.asarray(boot);lo,hi=np.quantile(boot,[.025,.975],axis=0)
    radius=float(np.quantile(np.max(np.abs(boot-diff),axis=1),.95))
    return [{'difference':float(diff[i]),'marginal_95':[float(lo[i]),float(hi[i])],
             'simultaneous_95':[float(diff[i]-radius),float(diff[i]+radius)]} for i in range(values.shape[1])]


def summarize(ctx,adf_path,d1_path,base,holdings,choices,controls):
    families={family:[pd.read_csv(OUT/family/f'p{p}_q{q}'/'daily.csv',parse_dates=['Date']) for p,q in ORDERS]
              for family in ['fixed_d1','adf']}
    common=base.OfficialDailySpread.notna().to_numpy()
    for daily in families['fixed_d1']+families['adf']+[adf_path,d1_path]+list(controls.values()):common &= daily.OfficialDailySpread.notna().to_numpy()
    common_dates=ctx['cal'].loc[common].copy();common_dates.to_csv(OUT/'common_calendar.csv',index=False)
    excluded=ctx['cal'].loc[~common].copy();excluded.to_csv(OUT/'excluded_dates.csv',index=False)
    leader=[];annual=[]
    for family in families:
        values=np.column_stack([d.loc[common,'OfficialDailySpread'].to_numpy() for d in families[family]]+[base.loc[common,'OfficialDailySpread'].to_numpy()])
        intervals=bootstrap(values,len(ORDERS))
        for oi,(p,q) in enumerate(ORDERS):
            daily=families[family][oi]
            leader.append({'family':family,'p':p,'q':q,**stats(daily.loc[common]),
                           'sharpe_minus_v7':intervals[oi]['difference'],
                           'simultaneous95_low':intervals[oi]['simultaneous_95'][0],
                           'simultaneous95_high':intervals[oi]['simultaneous_95'][1]})
            for year in YEARS:
                mask=common&(ctx['cal'].ValidationYear.to_numpy()==year)
                annual.append({'family':family,'p':p,'q':q,'year':year,**stats(daily.loc[mask])})
    leaderboard=pd.DataFrame(leader).sort_values(['family','sharpe','mean_daily_rank_ic'],ascending=[True,False,False])
    leaderboard.to_csv(OUT/'leaderboard.csv',index=False);pd.DataFrame(annual).to_csv(OUT/'annual_candidates.csv',index=False)
    paths={'v7_equal':base,'fixed_d1_annual_selector':d1_path,'adf_annual_selector':adf_path,
           'fixed_311_posthoc_reference':families['fixed_d1'][ORDERS.index((3,1))]}
    paths.update(controls)
    path_rows=[]
    for name,daily in paths.items():
        path_rows.append({'strategy':name,'period':'all',**stats(daily.loc[common])})
        path_rows.append({'strategy':name,'period':'2019-2021',**stats(daily.loc[common&(ctx['cal'].ValidationYear.to_numpy()>=2019)])})
        for year in YEARS:
            path_rows.append({'strategy':name,'period':str(year),**stats(daily.loc[common&(ctx['cal'].ValidationYear.to_numpy()==year)])})
    pd.DataFrame(path_rows).to_csv(OUT/'strategy_metrics.csv',index=False)
    names=list(paths);values=np.column_stack([paths[n].loc[common,'OfficialDailySpread'].to_numpy() for n in names])
    vs_v7=bootstrap(values,0);vs_d1=bootstrap(values,1)
    vs_matched=bootstrap(values,names.index('d1_at_adf_orders_matched_adf_coverage'))
    contrasts={name:{'versus_v7':vs_v7[i],'versus_fixed_d1_annual':vs_d1[i],
                     'versus_matched_order_and_coverage_d1':vs_matched[i]} for i,name in enumerate(names)}
    costs=[]
    for name,hold in holdings.items():
        # Unit long + unit short notionals. This target-weight turnover ignores price drift.
        turnover=np.abs(np.diff(np.vstack([np.zeros((1,hold.shape[1])),hold]),axis=0)).sum(axis=1)
        turnover[-1]+=np.abs(hold[-1]).sum()  # close final open position in the last P&L interval
        gross=paths[name].OfficialDailySpread.to_numpy()/200.
        for bps in [0,5,10,20]:
            net=gross-bps*1e-4*turnover
            finite=common&np.isfinite(net)
            costs.append({'strategy':name,'cost_bps_per_traded_notional':bps,
                          'net_sharpe':float(np.mean(net[finite])/np.std(net[finite],ddof=1)),
                          'mean_unit_long_short_return':float(np.mean(net[finite])),
                          'mean_l1_target_weight_turnover':float(np.mean(turnover[finite])),
                          'mean_gross_break_even_bps':float(np.mean(gross[finite])/np.mean(turnover[finite])*1e4),
                          'ignores_price_drift_borrow_fees_and_execution_limits':True})
        pd.DataFrame({'Date':ctx['cal'].Date,'GrossUnitLongShortReturn':gross,'L1TargetWeightTurnover':turnover}).to_csv(OUT/'selectors'/f'{name}_turnover.csv',index=False)
    pd.DataFrame(costs).to_csv(OUT/'cost_sensitivity.csv',index=False)
    result={'complete':True,'specification_hashes':hashes(),'common_days':int(common.sum()),
            'excluded_dates':excluded.Date.dt.strftime('%Y-%m-%d').tolist(),
            'all_strategy_metrics':path_rows,'annual_choices':choices,'contrasts':contrasts,
            'cost_sensitivity':costs,'formal_test_used':False,'baseline_changed':False,
            'bootstrap':{'block_days':20,'replicates':2000,'seed':CONFIG['seed'],
                         'limitation':'Resamples realized selected paths; does not refit or reselect. Repeated validation development, not new holdout evidence.'}}
    save(OUT/'results.json',result)
    return result


def main():
    ctx=load_context()
    inputs=OUT/'shared_inputs';inputs.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(INPUT/'prices.npz',inputs/'prices.npz')
    shutil.copyfile(INPUT/'prediction_keys.npz',inputs/'prediction_keys.npz')
    shutil.copyfile(INPUT/'calendar.csv',inputs/'calendar.csv')
    shutil.copyfile(INPUT/'labels.pkl',inputs/'labels.pkl')
    shutil.copyfile(INPUT/'baseline.pkl',inputs/'baseline.pkl')
    build_candidates(ctx)
    d1,choices1,h1=selector('fixed_d1',ctx)
    adf,choicesa,ha=selector('adf',ctx)
    base,hb,error=baseline(ctx)
    controls=matched_controls(choicesa,ctx)
    result=summarize(ctx,adf,d1,base,{'v7_equal':hb,'fixed_d1_annual_selector':h1,'adf_annual_selector':ha},choices1+choicesa,controls)
    save(EXP/'evaluation_summary.json',{'common_days':result['common_days'],'strategies':[r for r in result['all_strategy_metrics'] if r['period']=='all'],
                                    'baseline_metric_error':error,'formal_test_used':False})
    print(json.dumps(json.loads((EXP/'evaluation_summary.json').read_text())),flush=True)


if __name__=='__main__':main()
