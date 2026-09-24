from pathlib import Path
import sys,json,zipfile,hashlib
import numpy as np
import pandas as pd
from official_metric import calc_spread_return_sharpe
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'jpx_stock_returns_20260910'
sys.path.insert(0,str(BASE/'source'))
from jpx_common import gd_fit
FEATURES=['T5','T22','T60','V5','V22','V60','P5xV5','P22xV22','P60xV60']
BASE_FEATURES=FEATURES[:6]
W=np.linspace(2.,1.,200)

def save(name,obj):(ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))

def predict_rank(observations,theta):
    # Strict feature-only interface: labels must be separated before prediction.
    assert 'Target' not in observations
    v=observations[['SignalDate','SecuritiesCode']+BASE_FEATURES].copy()
    v['Fallback']=~np.isfinite(v[BASE_FEATURES]).all(axis=1)
    for c in BASE_FEATURES:v[c]=v[c].where(np.isfinite(v[c]),0.)
    for k in [5,22,60]:v[f'P{k}xV{k}']=v[f'T{k}']*v[f'V{k}']
    v['g']=theta[0]+v[FEATURES].to_numpy()@theta[1:]
    v=v.sort_values(['SignalDate','g','SecuritiesCode'],ascending=[True,False,True],kind='stable')
    v['Rank']=v.groupby('SignalDate').cumcount()
    return v[['SignalDate','SecuritiesCode','g','Rank','Fallback']].rename(columns={'SignalDate':'Date'})

def evaluate(ranks,labels):
    scored=ranks.merge(labels.rename(columns={'SignalDate':'Date'}),on=['Date','SecuritiesCode'],validate='one_to_one')
    days=[];holdings=[]
    for date,g in scored.groupby('Date',sort=True):
        ordered=g.sort_values('Rank');n=len(ordered)
        assert n>=400 and ordered['Rank'].tolist()==list(range(n))
        up=ordered.head(200).copy();down=ordered.tail(200).iloc[::-1].copy()
        missing=int(up.Target.isna().sum()+down.Target.isna().sum())
        for side,frame in [('long',up),('short',down)]:
            frame['Side']=side;frame['SideRank']=np.arange(1,201);frame['RawRankWeight']=W
            frame['WithinSideWeight']=W/W.sum();frame['IllustrativeSigned50_50Weight']=(1 if side=='long' else -1)*.5*W/W.sum()
            holdings.append(frame)
        if missing:spread=None;long=None;short=None
        else:
            long=float(up.Target.to_numpy()@W/W.mean());short=float(down.Target.to_numpy()@W/W.mean());spread=long-short
        days.append({'Date':date,'StocksRanked':n,'FallbackStocks':int(g.Fallback.sum()),'SelectedFallbackStocks':int(up.Fallback.sum()+down.Fallback.sum()),
            'SelectedMissingTargets':missing,'AllMissingTargets':int(g.Target.isna().sum()),'LongWeightedScore':long,'ShortWeightedScore':short,
            'OfficialDailySpread':spread,'IllustrativeGrossOneReturn':None if spread is None else spread/400})
    daily=pd.DataFrame(days);selected=pd.concat(holdings,ignore_index=True)
    valid=daily.loc[daily.SelectedMissingTargets.eq(0)];covered=scored.loc[scored.Date.isin(valid.Date)]
    manual=float(valid.OfficialDailySpread.mean()/valid.OfficialDailySpread.std(ddof=1))
    exact=float(calc_spread_return_sharpe(covered[['Date','Rank','Target']]))
    np.testing.assert_allclose(exact,manual,rtol=0,atol=1e-13)
    normalized=float(valid.IllustrativeGrossOneReturn.mean()/valid.IllustrativeGrossOneReturn.std(ddof=1))
    np.testing.assert_allclose(normalized,exact,atol=1e-13,rtol=0)
    return daily,selected,{'official_style_unannualized_sharpe':exact,'annualized_equivalent_sqrt252':exact*np.sqrt(252),
        'signal_days':len(daily),'scored_days':len(valid),'unscorable_selected_missing_label_days':int(daily.SelectedMissingTargets.gt(0).sum()),
        'ranked_rows':len(scored),'stocks_per_day_min':int(daily.StocksRanked.min()),'stocks_per_day_max':int(daily.StocksRanked.max()),
        'fallback_rows':int(scored.Fallback.sum()),'selected_fallback_rows':int(selected.Fallback.sum()),'unavailable_target_rows':int(scored.Target.isna().sum()),
        'missing_selected_target_rows':int(selected.Target.isna().sum()),'mean_official_spread':float(valid.OfficialDailySpread.mean()),
        'sample_std_official_spread':float(valid.OfficialDailySpread.std(ddof=1)),
        'mean_illustrative_gross_one_return':float(valid.IllustrativeGrossOneReturn.mean()),'exact_official_metric_difference':abs(exact-manual)}

def main():
    f=pd.read_pickle(BASE/'features.pkl')
    intervals=pd.read_csv(ROOT.parent/'jpx_signed_band_20260910/market_signals.csv',usecols=['SignalDate','EntryDate','ExitDate','ValidationYear'],parse_dates=['SignalDate','EntryDate','ExitDate'])
    with zipfile.ZipFile('/Users/coolguy/Desktop/JPX_data/modelling/JPX-market-regime-experiment.zip') as z:refs=json.loads(z.read('jpx_hold_same_results/results.json'))['model']
    with zipfile.ZipFile('/Users/coolguy/Desktop/JPX_data/modelling/JPX_data.zip') as z:
        raw=pd.read_csv(z.open('JPX_data/raw/train_files/stock_prices.csv'),usecols=['Date','SecuritiesCode','SupervisionFlag','Target'],parse_dates=['Date'])
    raw=raw.rename(columns={'Date':'SignalDate'})
    metadata=raw[['SignalDate','SecuritiesCode','SupervisionFlag']]
    results=[];full_results=[];fits=[];ranked=[];full_ranked=[];daily_all=[];hold_all=[]
    for year,ref in zip([2018,2019,2020,2021],refs):
        dates=intervals.loc[intervals.ValidationYear.eq(year),'SignalDate'];cutoff=dates.min()
        tr=f.loc[f.SignalDate.dt.year.eq(year-1)&f.ExitDate.le(cutoff)&f.Target.notna()&f.Eligible]
        assert len(tr)==ref['train_rows'] and tr.ExitDate.max()<=cutoff
        x=np.column_stack([np.ones(len(tr)),tr[FEATURES].to_numpy()]);theta,fit=gd_fit(x,tr.Target.to_numpy())
        np.testing.assert_allclose(theta,np.array(list(ref['coefficients'].values())),atol=1e-9,rtol=0)
        fits.append({'validation_year':year,'fit_asof':str(cutoff.date()),'training_year':year-1,'train_rows':len(tr),
            'last_training_label_exit':str(tr.ExitDate.max().date()),'coefficients':dict(zip(['alpha']+FEATURES,map(float,theta))),'fit':fit})
        obs=f.loc[f.SignalDate.isin(dates),['SignalDate','SecuritiesCode']+BASE_FEATURES].merge(metadata,on=['SignalDate','SecuritiesCode'],validate='one_to_one')
        labels=raw.loc[raw.SignalDate.isin(dates),['SignalDate','SecuritiesCode','Target']]
        # Persist predictions without any held-out label before invoking the scorer.
        full=predict_rank(obs,theta);full['ValidationYear']=year
        active=obs.loc[~obs.SupervisionFlag];r=predict_rank(active,theta);r['ValidationYear']=year
        r.to_csv(ROOT/f'ranks_{year}.csv.gz',index=False,compression='gzip')
        full.to_csv(ROOT/f'full_core_ranks_{year}.csv.gz',index=False,compression='gzip')
        daily,holdings,res=evaluate(r,labels);daily['ValidationYear']=year;holdings['ValidationYear']=year
        _,_,fullres=evaluate(full,labels)
        res.update(validation_year=year,supervision_rows_excluded=int(obs.SupervisionFlag.sum()));fullres['validation_year']=year
        results.append(res);full_results.append(fullres);ranked.append(r);full_ranked.append(full);daily_all.append(daily);hold_all.append(holdings)
        print('YEAR',json.dumps(res),flush=True)
    combined=pd.concat(ranked,ignore_index=True);alllabels=raw.loc[raw.SignalDate.isin(intervals.SignalDate),['SignalDate','SecuritiesCode','Target']]
    daily,holdings,total=evaluate(combined,alllabels)
    daily=daily.merge(intervals[['SignalDate','ValidationYear']].rename(columns={'SignalDate':'Date'}),on='Date',validate='one_to_one')
    assert holdings.ValidationYear.eq(holdings.Date.map(intervals.set_index('SignalDate').ValidationYear)).all()
    _,_,fulltotal=evaluate(pd.concat(full_ranked,ignore_index=True),alllabels)
    daily.to_csv(ROOT/'daily_spread_returns.csv',index=False);holdings.to_csv(ROOT/'selected_200_each_side.csv.gz',index=False,compression='gzip')
    # File for the inference/submission interface contains no g or Target, original stock row order.
    sample_date=combined.Date.max();sample=combined.loc[combined.Date.eq(sample_date)].sort_values('SecuritiesCode')
    sample[['Date','SecuritiesCode','Rank']].to_csv(ROOT/'historical_submission_example.csv',index=False)
    save('model_fits.json',fits)
    result={'model':'original level price/volume T and same-window products, no Ridge, previous-year train',
        'total':total,'annual':results,'full_core_universe_sensitivity':{'total':fulltotal,'annual':full_results},
        'official_metric_source':'https://www.kaggle.com/code/smeitoma/jpx-competition-metric-definition',
        'official_metric_version':1,'weights':{'portfolio_size_each_side':200,'toprank_weight_ratio':2,'sum_raw_side_weights':float(W.sum()),'mean_raw_side_weights':float(W.mean()),
            'first_within_side_weight':float(W[0]/W.sum()),'last_within_side_weight':float(W[-1]/W.sum())},
        'no_regime':True,'no_sign_transfer':True,'no_softmax':True,'costs_included':False,'formal_test_used':False,
        'validation_targets_not_used_in_predictions':True,'historical_universe_proxy':'core rows, excluding current SupervisionFlag; actual test requires sample_prediction rows',
        'all_internal_checks_passed':True}
    save('results.json',result)
    save('source_hashes.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'official_metric.json',ROOT/'official_metric.ipynb',ROOT/'official_metric.py',ROOT/'run_experiment.py',ROOT/'experiment_plan.md',BASE/'source/jpx_common.py']})
    print('TOTAL',json.dumps(total),flush=True);print('FULL_CORE',json.dumps(fulltotal),flush=True)

if __name__=='__main__':main()
