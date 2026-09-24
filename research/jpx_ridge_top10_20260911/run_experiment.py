from pathlib import Path
import sys,json,zipfile,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
MODEL=ROOT.parent/'jpx_return_t_20260911'
PREV=ROOT.parent/'jpx_return_t_top10_20260911'
BASE=ROOT.parent/'jpx_stock_returns_20260910'
MARKET=ROOT.parent/'jpx_signed_band_20260910/market_signals.csv'
sys.path.insert(0,str(BASE/'source'));sys.path.insert(0,str(PREV))
from jpx_common import performance
from jpx_execution import execute_hold,ORDER_COLUMNS
from allocation import allocate_slots
LAMBDAS=[0.,.001,.01,.1,1.,10.,100.]

def save(name,x): (ROOT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))

def sufficient(x,y):
    xm=x.mean(axis=0);ym=float(y.mean());xc=x-xm;yc=y-ym
    return xm,ym,xc.T@xc/len(y),xc.T@yc/len(y)

def ridge(stats,penalty):
    xm,ym,gram,cross=stats
    beta=np.linalg.solve(gram+penalty*np.eye(len(cross)),cross)
    return np.r_[ym-xm@beta,beta]

def predict(x,theta):return theta[0]+x@theta[1:]

def choose_lambda(train,cols,outer_cutoff):
    dates=pd.DatetimeIndex(train.SignalDate.unique()).sort_values()
    bounds=[len(dates)//2,3*len(dates)//4,len(dates)]
    records=[];sse={a:0. for a in LAMBDAS};nrows=0
    for j in range(2):
        valdates=dates[bounds[j]:bounds[j+1]];cutoff=valdates.min()
        tr=train.loc[train.SignalDate.lt(cutoff)&train.ExitDate.le(cutoff)]
        va=train.loc[train.SignalDate.isin(valdates)]
        assert tr.ExitDate.max()<=cutoff and tr.SignalDate.max()<va.SignalDate.min()
        assert va.ExitDate.max()<=outer_cutoff
        stats=sufficient(tr[cols].to_numpy(),tr.Target.to_numpy())
        x=va[cols].to_numpy();y=va.Target.to_numpy();scores=[]
        for a in LAMBDAS:
            theta=ridge(stats,a);error=float(np.sum((predict(x,theta)-y)**2))
            sse[a]+=error
            scores.append({'lambda':a,'mse':error/len(y),'squared_error_sum':error,'coefficient_l2':float(np.linalg.norm(theta[1:]))})
        nrows+=len(va)
        records.append({'fold':j+1,'fit_asof':str(cutoff.date()),'train_rows':len(tr),
            'training_first_signal':str(tr.SignalDate.min().date()),'training_last_signal':str(tr.SignalDate.max().date()),
            'last_training_label_exit':str(tr.ExitDate.max().date()),'validation_rows':len(va),
            'validation_first_signal':str(va.SignalDate.min().date()),'validation_last_signal':str(va.SignalDate.max().date()),
            'last_validation_label_exit':str(va.ExitDate.max().date()),'scores':scores})
    aggregate=[{'lambda':a,'mse':sse[a]/nrows,'validation_rows':nrows} for a in LAMBDAS]
    chosen=min(aggregate,key=lambda r:(r['mse'],r['lambda']))['lambda']
    return chosen,{'folds':records,'aggregate_scores':aggregate,'chosen_lambda':chosen,
        'at_upper_grid_boundary':chosen==max(LAMBDAS)}

def fit_check(train,cols,chosen,reference):
    x=train[cols].to_numpy();y=train.Target.to_numpy();stats=sufficient(x,y)
    theta=ridge(stats,chosen);ols=ridge(stats,0.)
    np.testing.assert_allclose(ols,np.array(list(reference.values())),atol=1e-10,rtol=1e-7)
    xm,ym,gram,cross=stats
    xc=x-xm;yc=y-ym
    # MSE + lambda ||beta||^2 is equivalent to this augmented least-squares problem.
    augmented=np.vstack([xc/np.sqrt(len(y)),np.sqrt(chosen)*np.eye(len(cols))])
    target=np.r_[yc/np.sqrt(len(y)),np.zeros(len(cols))]
    beta_check=np.linalg.lstsq(augmented,target,rcond=None)[0]
    np.testing.assert_allclose(theta[1:],beta_check,atol=1e-10,rtol=1e-7)
    gradient=gram@theta[1:]-cross+chosen*theta[1:]
    assert np.max(np.abs(gradient))<1e-10
    norm=float(np.linalg.norm(theta[1:]));oldnorm=float(np.linalg.norm(ols[1:]))
    assert norm<=oldnorm+1e-12
    norms=[float(np.linalg.norm(ridge(stats,a)[1:])) for a in LAMBDAS]
    assert (np.diff(norms)<=1e-12).all()
    mse=float(np.mean((predict(x,theta)-y)**2));olsmse=float(np.mean((predict(x,ols)-y)**2))
    assert mse>=olsmse-1e-14
    return theta,ols,{'train_mse':mse,'ols_train_mse':olsmse,'penalty':chosen*norm**2,'penalized_loss':mse+chosen*norm**2,
        'coefficient_l2':norm,'ols_coefficient_l2':oldnorm,'coefficient_l2_ratio':norm/oldnorm,
        'coefficient_max_abs':float(np.max(np.abs(theta[1:]))),'ols_coefficient_max_abs':float(np.max(np.abs(ols[1:]))),
        'stationarity_max_abs':float(np.max(np.abs(gradient))),
        'augmented_lstsq_max_beta_difference':float(np.max(np.abs(theta[1:]-beta_check))),
        'ols_original_coefficient_max_difference':float(np.max(np.abs(ols-np.array(list(reference.values()))))),
        'norm_path':list(zip(LAMBDAS,norms))}

def main():
    f=pd.read_pickle(MODEL/'return_t_features.pkl')
    priorfits=json.loads((MODEL/'model_fits.json').read_text())
    signals=pd.read_csv(MARKET,parse_dates=['SignalDate','EntryDate','ExitDate'])
    olddaily=pd.read_csv(PREV/'daily_returns.csv',parse_dates=['SignalDate','EntryDate','ExitDate'])
    with zipfile.ZipFile('/Users/coolguy/Desktop/JPX_data/modelling/JPX-market-regime-experiment.zip') as z:
        benchmark=pd.read_csv(z.open('jpx_inputs/nikkei_benchmark.csv'),parse_dates=['Date']).set_index('Date').Close
    fits=[];tuning=[];orders=[];intervals=[]
    for reference in priorfits:
        year=reference['validation_year'];cols=list(reference['coefficients'])[1:]
        s=signals.loc[signals.ValidationYear.eq(year)];cutoff=s.SignalDate.min()
        train=f.loc[f.SignalDate.dt.year.eq(year-1)&f.ExitDate.le(cutoff)&f.Target.notna()&f.TEligible]
        assert len(train)==reference['train_rows'] and train.ExitDate.max()<=cutoff
        penalty,tune=choose_lambda(train,cols,cutoff)
        tune['outer_validation_year']=year;tune['outer_fit_asof']=str(cutoff.date());tuning.append(tune)
        theta,ols,fit=fit_check(train,cols,penalty,reference['coefficients'])
        v=f.loc[f.SignalDate.isin(s.SignalDate)&f.TEligible].copy()
        assert len(v)==reference['validation_rows']
        x=v[cols].to_numpy();v['Prediction']=predict(x,theta);op=predict(x,ols)
        v['RawPrediction']=v.Prediction;v['PreviousRawPrediction']=v.Prediction
        v['LagFallback']=False;v['U']=1.;v['ValidationYear']=year
        dailyic=[]
        for date,day in v.groupby('SignalDate'):
            row=s.loc[s.SignalDate.eq(date)].iloc[0]
            long=.7 if row.Regime==1 else .3 if row.Regime==-1 else .5
            allocated=allocate_slots(day,np.array([np.full(10,long/10),np.full(10,(1-long)/10)]))
            orders.append(allocated[ORDER_COLUMNS])
            valid=day.loc[day.Target.notna()]
            ic=valid.Prediction.rank().corr(valid.Target.rank()) if valid.Target.nunique()>1 and valid.Prediction.nunique()>1 else np.nan
            dailyic.append(ic)
            intervals.append({**row[['SignalDate','EntryDate','ExitDate','ValidationYear']].to_dict(),
                'DailyRankIC':ic,'RawDailyRankIC':ic,'V2DailyRankIC':np.nan,'EligibleStocks':len(day)})
        labeled=v.Target.notna();target=v.loc[labeled,'Target']
        fits.append({'validation_year':year,'fit_asof':str(cutoff.date()),'lambda':penalty,'train_rows':len(train),
            'last_training_label_exit':str(train.ExitDate.max().date()),'validation_rows':len(v),
            'coefficients':dict(zip(['alpha']+cols,map(float,theta))),'fit':fit,
            'validation_mse':float(np.mean((v.loc[labeled,'Prediction']-target)**2)),
            'ols_validation_mse':float(np.mean((op[labeled]-target)**2)),
            'zero_validation_mse':float(np.mean(target**2)),'mean_daily_rank_ic':float(np.nanmean(dailyic)),
            'prediction_abs_quantiles':{str(q):float(n) for q,n in v.Prediction.abs().quantile([.5,.99,1]).items()},
            'ols_prediction_abs_quantiles':{str(q):float(n) for q,n in pd.Series(op).abs().quantile([.5,.99,1]).items()}})
        print('Fitted',year,'lambda',penalty,'coefficient norm ratio',fit['coefficient_l2_ratio'],flush=True)
    save('model_fits.json',fits);save('lambda_selection.json',tuning)
    selected=pd.concat(orders,ignore_index=True);timeline=pd.DataFrame(intervals).sort_values('EntryDate')
    selected.to_csv(ROOT/'requested_orders.csv',index=False)
    pd.testing.assert_frame_equal(timeline[['SignalDate','EntryDate','ExitDate','ValidationYear','EligibleStocks']].reset_index(drop=True),olddaily[['SignalDate','EntryDate','ExitDate','ValidationYear','EligibleStocks']].reset_index(drop=True),check_dtype=False)
    bars=pd.read_pickle(BASE/'bars.pkl')
    bars=bars.loc[bars.SecuritiesCode.isin(selected.SecuritiesCode.unique())&bars.Date.between(timeline.EntryDate.min(),timeline.ExitDate.max())]
    trades,daily,events=execute_hold(selected,bars,timeline,benchmark)
    np.testing.assert_allclose(daily.StrategyReturn,events.groupby('IntervalEntryDate').Contribution.sum().reindex(daily.EntryDate).fillna(0),atol=1e-12,rtol=0)
    np.testing.assert_allclose(daily.IndexReturn,olddaily.IndexReturn,atol=1e-12,rtol=0)
    assert daily.CashWeight.ge(-1e-12).all() and not trades.duplicated(['EntryDate','SecuritiesCode']).any()
    assert trades.loc[trades.Executed,'ExecutedWeight'].abs().le(.07+1e-12).all()
    o=selected.merge(signals[['SignalDate','Regime']],on='SignalDate',validate='many_to_one')
    long=np.where(o.Regime.eq(1),.7,np.where(o.Regime.eq(-1),.3,.5))
    np.testing.assert_allclose(o.Weight.abs(),np.where(o.SourceSide.eq('long'),long,1-long)/10,atol=1e-14,rtol=0)
    for name,frame in [('selected_trades',trades),('daily_returns',daily),('position_events',events)]:frame.to_csv(ROOT/(name+'.csv'),index=False)
    prior=json.loads((PREV/'results.json').read_text())
    result={'ridge_top10':performance(daily.StrategyReturn),'ols_top10':performance(olddaily.StrategyReturn),'benchmark':performance(daily.IndexReturn),
        'annual':[{'year':int(y),'ridge':performance(g.StrategyReturn),'ols':performance(olddaily.loc[olddaily.ValidationYear.eq(y),'StrategyReturn'])} for y,g in daily.groupby('ValidationYear')],
        'lambda_grid':LAMBDAS,'chosen_lambdas':{str(fit['validation_year']):fit['lambda'] for fit in fits},
        'lambda_selection':'two chronological inner folds within previous-year training; minimum pooled inner-validation MSE',
        'mean_daily_rank_ic':float(daily.DailyRankIC.mean()),'ols_mean_daily_rank_ic':float(olddaily.DailyRankIC.mean()),
        'fits':fits,'market_signals_sha256':hashlib.sha256(MARKET.read_bytes()).hexdigest(),
        'plan_sha256':hashlib.sha256((ROOT/'experiment_plan.md').read_bytes()).hexdigest(),
        'features_and_portfolio_rules_unchanged':True,'test_used':False,'costs':0,'all_checks_passed':True}
    assert result['market_signals_sha256']==prior['market_signals_sha256']
    save('results.json',result)
    print('RESULT',json.dumps({k:v for k,v in result.items() if k!='fits'},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
