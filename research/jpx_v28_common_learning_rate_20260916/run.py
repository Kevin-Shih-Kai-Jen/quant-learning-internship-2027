from pathlib import Path
import sys,json,pickle,hashlib,time,shutil
import numpy as np
import pandas as pd
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parent; BASE=ROOT.parent; V7=BASE/'jpx_v7_daily_mse_20260912';V14=BASE/'jpx_v14_filtered_forecast_events_20260914'
sys.path.insert(0,str(V7))
from run_v7 import make_x,INPUTS,FEATURES,mse_step
V27=BASE/'jpx_v27_profit_forecast_state_20260916'
COMMON=None
NAMES=['alpha']+FEATURES
FINS=[m+p for m in ['NetSales','OperatingProfit','EPS'] for p in ['ActualGrowth','ExpectedGrowth','ForecastQoQ','ForecastYoY','Revision']]
W=np.linspace(2.,1.,200)
def save(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False))
def sharp(s):return float(s.mean()/s.std(ddof=1))
def load():
    with (BASE/'jpx_v8_soft_rank_20260912/inputs.pkl').open('rb') as h:f,x,groups,calendar=pickle.load(h)
    z=pd.read_pickle(V14/'financial_signal_features.pkl')
    np.testing.assert_array_equal(x,make_x(f))
    assert len(z)==len(f) and np.isfinite(z[FINS].to_numpy()).all()
    np.testing.assert_array_equal(z.TrainingEligible.to_numpy(),np.max(np.abs(z[FINS].to_numpy()),axis=1)<=100)
    return f,x,z,groups,calendar

def daily_eval(date,ix,score,order,y,code,fallback,year):
    target=y[ix];known=np.isfinite(target);top=order[:200];bottom=order[-200:][::-1]
    chosen=np.r_[top,bottom];miss=int((~known[chosen]).sum())
    long=float(target[top]@W/W.mean()) if not miss else None
    short=float(target[bottom]@W/W.mean()) if not miss else None
    a=rankdata(-score[known],method='average');b=rankdata(-target[known],method='average')
    ic=float(np.corrcoef(a,b)[0,1]) if np.std(a)>0 and np.std(b)>0 else None
    spread=long-short if not miss else None
    return {'Date':str(date.date()),'ValidationYear':int(year),'StocksRanked':len(ix),'SelectedMissingTargets':miss,'AllMissingTargets':int((~known).sum()),'FallbackStocks':int(fallback[ix].sum()),'SelectedFallbackStocks':int(fallback[ix[chosen]].sum()),'OfficialDailySpread':spread,'IllustrativeGrossOneReturn':spread/400 if spread is not None else None,'RankIC':ic,'NormalizedHardRankMSE':float(np.mean(((a-b)/(known.sum()-1))**2)),'AllStockForecastMSE':float(np.mean((score[known]-target[known])**2)),'MaxAbsoluteScore':float(np.abs(score).max())}

def run_one(name,features,f,x,z,groups,calendar):
    out=ROOT/name;out.mkdir(exist_ok=True);t0=time.monotonic()
    zz=z[features].to_numpy()
    xx=np.column_stack([x,zz])
    names=NAMES+features
    npar=len(names);theta=np.zeros(npar);scores=np.full(len(f),np.nan);ranks=np.full(len(f),-1,dtype=np.int32)
    eligibility=np.ones(len(f),bool)
    target=f.Target.to_numpy();codes=f.SecuritiesCode.to_numpy();fallback=~np.isfinite(f[INPUTS].to_numpy()).all(axis=1)
    yearmap=calendar.set_index('Date').ValidationYear.to_dict();end=calendar.Date.max();closed=pd.Timestamp('2020-10-01')
    pending=[];updates=[];hist=[];days=[];max_grad=0.;max_step=0.;max_loss=0.;max_rate=0.;predictions=0;trainrows=0;skiprows=0
    for date,ix in groups.items():
        beforeday=theta.copy();due=[b for b in pending if b[1]<=date];pending=[b for b in pending if b[1]>date]
        assert len(due)<=1
        for sd,exitdate,ids in due:
            assert sd<exitdate<=date
            finite=np.isfinite(target[ids]);used=finite&eligibility[ids];idx=ids[used]
            trainx=xx[idx];yy=target[idx];weights=np.ones(len(idx))/len(idx);old=theta.copy()
            err=trainx@old-yy
            grad=2*trainx.T@(weights*err)
            lam=float(np.linalg.eigvalsh(trainx.T@(weights[:,None]*trainx))[-1])
            eta=float(COMMON.loc[str(date.date()),'LearningRate'])
            assert eta<=1/(2*lam)*(1+1e-10)
            theta=old-eta*grad
            lb=float(weights@err**2);la=float(weights@(trainx@theta-yy)**2)
            assert la<=lb+1e-12*max(1,lb)
            # Independently assemble normal-equation gradient and spectral step.
            gram=np.einsum('ni,nj->ij',trainx,trainx)/len(idx)
            rhs=np.einsum('ni,n->i',trainx,yy)/len(idx)
            gg=2*(gram@old-rhs);lr=float(COMMON.loc[str(date.date()),'LearningRate']);replay=old-lr*gg
            ge=float(np.max(np.abs(gg-grad)));se=float(np.max(np.abs(replay-theta)));re=abs(lr-eta)
            np.testing.assert_allclose(gg,grad,atol=2e-10,rtol=2e-9)
            np.testing.assert_allclose(replay,theta,atol=2e-11,rtol=2e-9)
            np.testing.assert_allclose(lr,eta,atol=2e-13,rtol=2e-10)
            e0=np.einsum('ni,i->n',trainx,old)-yy;e1=np.einsum('ni,i->n',trainx,theta)-yy
            np.testing.assert_allclose([np.mean(e0*e0),np.mean(e1*e1)],[lb,la],atol=2e-12,rtol=2e-10)
            max_grad=max(max_grad,ge);max_step=max(max_step,se);max_rate=max(max_rate,re);max_loss=max(max_loss,abs(np.mean(e1*e1)-la))
            trainrows+=len(idx);skip=int((finite&~eligibility[ids]).sum());skiprows+=skip
            updates.append({'Date':str(date.date()),'SignalDate':str(sd.date()),'ExitDate':str(exitdate.date()),'KnownLabelStocks':int(finite.sum()),'TrainingStocks':len(idx),'ThresholdSkippedStocks':skip,'MSEBeforeStep':lb,'MSEAfterStep':la,'LearningRate':eta,'LargestGramEigenvalue':lam,**{f'Gradient_{n}':float(v) for n,v in zip(names,grad)}})
        hist.append({'Date':str(date.date()),'UpdatesToday':len(due),'CumulativeUpdates':len(updates),**{f'Before_{n}':float(v) for n,v in zip(names,beforeday)},**{f'After_{n}':float(v) for n,v in zip(names,theta)}})
        if date==closed or date>end:continue
        score=theta[0]+xx[ix,1:]@theta[1:];assert np.isfinite(score).all()
        order=np.lexsort((codes[ix],-score));rank=np.empty(len(ix),np.int32);rank[order]=np.arange(len(ix))
        scores[ix]=score;ranks[ix]=rank;predictions+=len(ix)
        if date in yearmap:days.append(daily_eval(date,ix,score,order,target,codes,fallback,yearmap[date]))
        ids=ix[order];exitvals=f.loc[ids,'ExitDate'];assert exitvals.nunique()==1 and exitvals.notna().all()
        pending.append((date,exitvals.iloc[0],ids))
    assert not pending and len(updates)==1199 and len(days)==953 and predictions==2326022
    df=pd.DataFrame(days);df.to_csv(out/'daily_metrics.csv',index=False)
    pd.DataFrame(updates).to_csv(out/'training_updates.csv',index=False);pd.DataFrame(hist).to_csv(out/'parameter_history.csv',index=False)
    np.savez_compressed(out/'predictions.npz',score=scores,rank=ranks)
    rr={'name':name,'features':features,'includes_feature':bool(features),'parameter_count':npar,'sharpe':sharp(df.OfficialDailySpread.dropna()),'scored_days':int(df.OfficialDailySpread.notna().sum()),'mean_rank_ic':float(df.RankIC.mean()),'rank_ic_days':int(df.RankIC.notna().sum()),'mean_hard_rank_mse':float(df.NormalizedHardRankMSE.mean()),'mean_return_mse':float(df.AllStockForecastMSE.mean()),'training_updates':len(updates),'training_stock_days':trainrows,'threshold_skipped_stock_days':skiprows,'forecast_rows':predictions,'final_coefficients':dict(zip(names,map(float,theta))),'parameter_asof':hist[-1]['Date'],'seconds':time.monotonic()-t0}
    save(out/'results.json',rr);save(out/'update_audit.json',{'passed':True,'checked_updates':len(updates),'max_gradient_abs_error':max_grad,'max_parameter_step_abs_error':max_step,'max_loss_abs_error':max_loss,'max_rate_abs_error':max_rate})
    print(json.dumps(rr,ensure_ascii=False),flush=True)
    return rr

JOBS={'g_common':[],'overwrite':['ProfitLatestYoY'],'change_old':['ProfitYoYChange','ProfitPreviousYoY']}
def state_load():
    f,x,_,groups,calendar=load();z=pd.read_pickle(V27/'financial_signal_features.pkl');assert np.isfinite(z.to_numpy()).all()
    return f,x,z,groups,calendar

def prepare_common(f,x,z,groups):
    # Dates only are reused. Rates are recomputed from the already-matured batch,
    # without consulting returns, coefficients, losses, Sharpe, or future batches.
    timing=pd.read_csv(V27/'overwrite/training_updates.csv')
    target=f.Target.to_numpy();rows=[]
    designs={name:np.column_stack([x,z[features].to_numpy()]) for name,features in JOBS.items()}
    for q in timing.itertuples():
        ids=groups[pd.Timestamp(q.SignalDate)];ids=ids[np.isfinite(target[ids])]
        assert pd.Timestamp(q.SignalDate)<pd.Timestamp(q.ExitDate)<=pd.Timestamp(q.Date)
        assert (f.loc[ids,'ExitDate']<=pd.Timestamp(q.Date)).all()
        rates={}
        for name,X in designs.items():
            a=X[ids];lam=float(np.linalg.eigvalsh(a.T@a/len(ids))[-1]);rates[name]=1/(2*lam)
        eta=min(rates['overwrite'],rates['change_old'])
        assert eta<=rates['g_common']*(1+1e-10)
        rows.append(dict(Date=q.Date,SignalDate=q.SignalDate,ExitDate=q.ExitDate,TrainingStocks=len(ids),LearningRate=eta,**{n+'_OwnLearningRate':v for n,v in rates.items()}))
    schedule=pd.DataFrame(rows);schedule.to_csv(ROOT/'common_learning_rates.csv',index=False)
    return schedule.set_index('Date')

def main():
    global COMMON
    defaults=json.loads((BASE/'JPX-experiment-defaults.json').read_text());assert not defaults['skip_training_for_extreme_values'] and not defaults['skip_validation_for_extreme_values']
    assert json.loads((V27/'feature_audit.json').read_text())['passed']
    f,x,z,groups,calendar=state_load();COMMON=prepare_common(f,x,z,groups)
    for name,features in JOBS.items():run_one(name,features,f,x,z,groups,calendar)
    allu={n:pd.read_csv(ROOT/n/'training_updates.csv') for n in JOBS}
    for n,u in allu.items():
        for c in ['Date','SignalDate','ExitDate','TrainingStocks','ThresholdSkippedStocks','LearningRate']:
            pd.testing.assert_series_equal(u[c],allu['g_common'][c],check_exact=True)
    paths=[BASE/'JPX-experiment-defaults.json',BASE/'jpx_v8_soft_rank_20260912/inputs.pkl',V27/'financial_signal_features.pkl',V27/'feature_audit.json',V27/'feature_manifest.json',V27/'run.py',V27/'features.py',ROOT/'run.py',ROOT/'experiment_plan.md',ROOT/'common_learning_rates.csv']
    for rel in ['jpx_v23_g_thresholds_20260916/no_skip','jpx_v27_profit_forecast_state_20260916/overwrite','jpx_v27_profit_forecast_state_20260916/change_old']:
        paths += [BASE/rel/name for name in ['daily_metrics.csv','results.json','training_updates.csv']]
    save(ROOT/'manifest.json',dict(source_sha256={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},training_threshold=None,validation_threshold=None,models=JOBS,learning_rate='same per-day scalar: min(A own spectral rate, B own spectral rate); no outcome-based tuning'))
if __name__=='__main__':main()
