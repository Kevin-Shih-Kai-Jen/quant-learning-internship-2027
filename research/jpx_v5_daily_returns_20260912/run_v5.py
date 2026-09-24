from pathlib import Path
import sys,json,zipfile,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
V4=ROOT.parent/'jpx_official_ranking_20260912'
BASE=ROOT.parent/'jpx_stock_returns_20260910'
sys.path.insert(0,str(V4))
from run_experiment import evaluate,predict_rank as predict_rank_v4,FEATURES as OLD,BASE_FEATURES
FEATURES=OLD+['PR1','VR1']
INPUTS=BASE_FEATURES+['PR1','VR1']

def save(name,obj):(ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))

def build_returns(f):
    closed=pd.to_datetime(json.loads((BASE/'feature_audit.json').read_text())['price']['market_wide_closures_excluded_from_lookbacks_only'])
    active=f.loc[~f.SignalDate.isin(closed),['SecuritiesCode','SignalDate','Close','Volume','CumulativeFactor']].copy()
    active['AP']=active.Close/active.CumulativeFactor;active['AV']=active.Volume*active.CumulativeFactor
    for c,src in [('PR1','AP'),('VR1','AV')]:
        prev=active.groupby('SecuritiesCode')[src].shift(1)
        f[c]=active[src]/prev.where(prev>0)-1
        f[c]=f[c].where(np.isfinite(f[c]))
        cm=f.SignalDate.isin(closed);f.loc[cm,c]=f.groupby('SecuritiesCode')[c].shift(1).loc[cm]
    f[['SignalDate','SecuritiesCode','PR1','VR1']].to_pickle(ROOT/'one_day_returns.pkl')
    return f

def predict_rank_v5(observations,theta):
    assert 'Target' not in observations
    v=observations[['SignalDate','SecuritiesCode']+INPUTS].copy()
    v['TFallback']=~np.isfinite(v[BASE_FEATURES]).all(axis=1)
    v['ReturnFallback']=~np.isfinite(v[['PR1','VR1']]).all(axis=1)
    v['Fallback']=v.TFallback|v.ReturnFallback
    for c in INPUTS:v[c]=v[c].where(np.isfinite(v[c]),0.)
    for k in [5,22,60]:v[f'P{k}xV{k}']=v[f'T{k}']*v[f'V{k}']
    v['g']=theta[0]+v[FEATURES].to_numpy()@theta[1:]
    v=v.sort_values(['SignalDate','g','SecuritiesCode'],ascending=[True,False,True],kind='stable')
    v['Rank']=v.groupby('SignalDate').cumcount()
    return v[['SignalDate','SecuritiesCode','g','Rank','Fallback','TFallback','ReturnFallback']].rename(columns={'SignalDate':'Date'})

def fit_raw(tr):
    x=np.column_stack([np.ones(len(tr)),tr[FEATURES].to_numpy()]);y=tr.Target.to_numpy()
    assert np.isfinite(x).all() and np.isfinite(y).all()
    theta,residual,rank,singular=np.linalg.lstsq(x,y,rcond=None)
    assert rank==x.shape[1]
    q,r=np.linalg.qr(x,mode='reduced');theta_qr=np.linalg.solve(r,q.T@y)
    np.testing.assert_allclose(theta,theta_qr,atol=1e-10,rtol=1e-8)
    pred=x@theta;err=pred-y;gradient=x.T@err/len(y)
    assert abs(gradient).max()<1e-10
    return theta,{'method':'direct raw-feature least squares; independent QR check; no input standardization or regularization',
        'train_mse':float(np.mean(err**2)),'matrix_rank':int(rank),'design_condition_number':float(singular[0]/singular[-1]),
        'qr_max_coefficient_difference':float(abs(theta-theta_qr).max()),'normal_residual_max':float(abs(gradient).max())}

def main():
    f=build_returns(pd.read_pickle(BASE/'features.pkl'))
    oldfits=json.loads((V4/'model_fits.json').read_text())
    olddaily=pd.read_csv(V4/'daily_spread_returns.csv',parse_dates=['Date'])
    with zipfile.ZipFile('/Users/coolguy/Desktop/JPX_data/modelling/JPX_data.zip') as z:
        raw=pd.read_csv(z.open('JPX_data/raw/train_files/stock_prices.csv'),usecols=['Date','SecuritiesCode','SupervisionFlag','Target'],parse_dates=['Date']).rename(columns={'Date':'SignalDate'})
    ranks=[];v4ranks=[];fits=[];annual=[];reproductions=[]
    for ref in oldfits:
        year=ref['validation_year'];cutoff=pd.Timestamp(ref['fit_asof'])
        tr=f.loc[f.SignalDate.dt.year.eq(year-1)&f.ExitDate.le(cutoff)&f.Target.notna()&f.Eligible]
        assert len(tr)==ref['train_rows'] and np.isfinite(tr[INPUTS]).all().all()
        assert tr.ExitDate.max()<=cutoff
        theta,fit=fit_raw(tr)
        dates=olddaily.loc[olddaily.ValidationYear.eq(year),'Date']
        observations=f.loc[f.SignalDate.isin(dates),['SignalDate','SecuritiesCode']+INPUTS].merge(raw[['SignalDate','SecuritiesCode','SupervisionFlag']],on=['SignalDate','SecuritiesCode'],validate='one_to_one')
        observations=observations.loc[~observations.SupervisionFlag].drop(columns='SupervisionFlag')
        old=predict_rank_v4(observations,np.array(list(ref['coefficients'].values())));old['ValidationYear']=year
        stored=pd.read_csv(V4/f'ranks_{year}.csv.gz',parse_dates=['Date'])
        keys=['Date','SecuritiesCode'];check=old.merge(stored,on=keys,suffixes=('_replay','_saved'),validate='one_to_one')
        assert len(check)==len(old)==len(stored) and check.Rank_replay.eq(check.Rank_saved).all()
        np.testing.assert_allclose(check.g_replay,check.g_saved,atol=1e-12,rtol=0)
        new=predict_rank_v5(observations,theta);new['ValidationYear']=year
        assert len(new)==len(old)
        new.to_csv(ROOT/f'ranks_{year}.csv.gz',index=False,compression='gzip')
        labels=raw.loc[raw.SignalDate.isin(dates),['SignalDate','SecuritiesCode','Target']]
        daily,selected,score=evaluate(new,labels)
        od,_,os=evaluate(old,labels)
        np.testing.assert_allclose(od.OfficialDailySpread,olddaily.loc[olddaily.ValidationYear.eq(year),'OfficialDailySpread'],atol=1e-12,rtol=0)
        annual.append({'validation_year':year,'v5':score,'v4':os})
        fitrecord={'validation_year':year,'training_year':year-1,'fit_asof':str(cutoff.date()),'train_rows':len(tr),'v4_train_rows':ref['train_rows'],
            'last_training_label_exit':str(tr.ExitDate.max().date()),'coefficients':dict(zip(['alpha']+FEATURES,map(float,theta))),'fit':fit,
            'training_return_ranges':{c:{'min':float(tr[c].min()),'max':float(tr[c].max())} for c in ['PR1','VR1']}}
        fits.append(fitrecord);ranks.append(new);v4ranks.append(old)
        reproductions.append({'year':year,'rows':len(check),'all_ranks_identical':True,'g_max_error':float(abs(check.g_replay-check.g_saved).max())})
        print('YEAR',year,'v4',os['official_style_unannualized_sharpe'],'v5',score['official_style_unannualized_sharpe'],'missing selected targets',score['missing_selected_target_rows'],flush=True)
    combined=pd.concat(ranks,ignore_index=True);oldcombined=pd.concat(v4ranks,ignore_index=True)
    labels=raw.loc[raw.SignalDate.isin(olddaily.Date),['SignalDate','SecuritiesCode','Target']]
    daily,selected,total=evaluate(combined,labels);od,_,oldtotal=evaluate(oldcombined,labels)
    common_dates=daily.loc[daily.SelectedMissingTargets.eq(0),'Date']
    common_dates=pd.Index(common_dates).intersection(od.loc[od.SelectedMissingTargets.eq(0),'Date'])
    common_v5=float(daily.loc[daily.Date.isin(common_dates),'OfficialDailySpread'].mean()/daily.loc[daily.Date.isin(common_dates),'OfficialDailySpread'].std(ddof=1))
    common_v4=float(od.loc[od.Date.isin(common_dates),'OfficialDailySpread'].mean()/od.loc[od.Date.isin(common_dates),'OfficialDailySpread'].std(ddof=1))
    daily=daily.merge(olddaily[['Date','ValidationYear']],on='Date',validate='one_to_one')
    daily.to_csv(ROOT/'daily_spread_returns.csv',index=False);selected.to_csv(ROOT/'selected_200_each_side.csv.gz',index=False,compression='gzip')
    examples=selected.merge(f[['SignalDate','SecuritiesCode','PR1','VR1']].rename(columns={'SignalDate':'Date'}),on=['Date','SecuritiesCode'],validate='one_to_one')
    examples[['Date','SecuritiesCode','Side','SideRank','g','PR1','VR1','TFallback','ReturnFallback']].sort_values('VR1',ascending=False).head(25).to_csv(ROOT/'largest_selected_volume_returns.csv',index=False)
    feature_audit={'rows':len(f),'raw_return_ranges':{c:{'min':float(f[c].min()),'max':float(f[c].max()),'undefined_rows':int(f[c].isna().sum())} for c in ['PR1','VR1']},
        'validation_return_fallback_rows':int(combined.ReturnFallback.sum()),'selected_return_fallback_rows':int(selected.ReturnFallback.sum()),
        'validation_t_fallback_rows':int(combined.TFallback.sum()),'selected_t_fallback_rows':int(selected.TFallback.sum()),
        'training_samples_unchanged_in_all_years':True,'no_return_standardization_or_clipping':True}
    save('feature_audit.json',feature_audit);save('model_fits.json',fits);save('v4_reproduction.json',reproductions)
    result={'version':'v5','baseline_version':'v4','features':FEATURES,'v5':total,'v4':oldtotal,'annual':annual,
        'same_scored_dates':{'days':len(common_dates),'v5_official_sharpe':common_v5,'v4_official_sharpe':common_v4},
        'v4_directory':str(V4),'official_metric_source':'https://www.kaggle.com/code/smeitoma/jpx-competition-metric-definition',
        'no_regime':True,'no_ridge':True,'no_softmax':True,'costs_included':False,'formal_test_used':False,'all_internal_checks_passed':True}
    save('results.json',result)
    save('source_hashes.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'experiment_plan.md',ROOT/'run_v5.py',V4/'official_metric.py',V4/'run_experiment.py',V4/'model_fits.json']})
    print('TOTAL',json.dumps(result),flush=True)

if __name__=='__main__':main()
