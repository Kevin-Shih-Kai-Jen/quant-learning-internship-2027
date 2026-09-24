from pathlib import Path
import sys,json,pickle,hashlib,time,shutil
import numpy as np
import pandas as pd
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parent; BASE=ROOT.parent; V7=BASE/'jpx_v7_daily_mse_20260912';V14=BASE/'jpx_v14_filtered_forecast_events_20260914'
sys.path.insert(0,str(V7))
from run_v7 import make_x,INPUTS,FEATURES,mse_step
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
            theta,grad,eta,lb,la,lam=mse_step(old,trainx,yy,weights)
            # Independently assemble normal-equation gradient and spectral step.
            gram=np.einsum('ni,nj->ij',trainx,trainx)/len(idx)
            rhs=np.einsum('ni,n->i',trainx,yy)/len(idx)
            gg=2*(gram@old-rhs);lr=1/(2*np.linalg.eigvalsh(gram)[-1]);replay=old-lr*gg
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
    rr={'name':name,'features':features,'includes_feature':True,'parameter_count':npar,'sharpe':sharp(df.OfficialDailySpread.dropna()),'scored_days':int(df.OfficialDailySpread.notna().sum()),'mean_rank_ic':float(df.RankIC.mean()),'rank_ic_days':int(df.RankIC.notna().sum()),'mean_hard_rank_mse':float(df.NormalizedHardRankMSE.mean()),'mean_return_mse':float(df.AllStockForecastMSE.mean()),'training_updates':len(updates),'training_stock_days':trainrows,'threshold_skipped_stock_days':skiprows,'forecast_rows':predictions,'final_coefficients':dict(zip(names,map(float,theta))),'parameter_asof':hist[-1]['Date'],'seconds':time.monotonic()-t0}
    save(out/'results.json',rr);save(out/'update_audit.json',{'passed':True,'checked_updates':len(updates),'max_gradient_abs_error':max_grad,'max_parameter_step_abs_error':max_step,'max_loss_abs_error':max_loss,'max_rate_abs_error':max_rate})
    print(json.dumps(rr,ensure_ascii=False),flush=True)
    return rr

JOINT=['OperatingProfitForecastYoY','OperatingProfitRevisionRelative']
def joint_load():
    f,x,old,groups,calendar=load();new=pd.read_pickle(BASE/'jpx_v17_actual_forecast_revisions_20260915/financial_signal_features.pkl')
    z=pd.DataFrame({JOINT[0]:old[JOINT[0]],JOINT[1]:new[JOINT[1]]});assert np.isfinite(z.to_numpy()).all()
    return f,x,z,groups,calendar

def main():
    defaults=json.loads((BASE/'JPX-experiment-defaults.json').read_text());assert not defaults['skip_training_for_extreme_values'] and not defaults['skip_validation_for_extreme_values']
    f,x,z,groups,calendar=joint_load();run_one('joint',JOINT,f,x,z,groups,calendar)
    sources=[BASE/'JPX-experiment-defaults.json',BASE/'jpx_v8_soft_rank_20260912/inputs.pkl',V14/'financial_signal_features.pkl',BASE/'jpx_v17_actual_forecast_revisions_20260915/financial_signal_features.pkl',ROOT/'run.py',ROOT/'experiment_plan.md']
    for rel in ['jpx_v23_g_thresholds_20260916/no_skip','jpx_v25_single_features_unfiltered_20260916/v16_feature_09','jpx_v25_single_features_unfiltered_20260916/v17_profit_revision_relative']:
        sources += [BASE/rel/name for name in ['results.json','daily_metrics.csv','training_updates.csv']]
    save(ROOT/'manifest.json',dict(source_sha256={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},training_threshold=None,validation_threshold=None,features=JOINT,feature_formulas_unchanged=True))
if __name__=='__main__':main()
