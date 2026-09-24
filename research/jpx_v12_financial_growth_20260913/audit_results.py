import json,gzip,pickle,sys
import numpy as np
import pandas as pd
from features import ROOT,FINS,load
from run_v12 import NAMES,save
from preflight import independent_step

def main(variant):
    out=ROOT/variant;f,xall,groups,calendar=load(variant)
    hist=pd.read_csv(out/'parameter_history.csv',parse_dates=['Date'],float_precision='round_trip').set_index('Date')
    u=pd.read_csv(out/'training_updates.csv',parse_dates=['Date','SignalDate','ExitDate'],float_precision='round_trip')
    bcols=['Before_'+n for n in NAMES];acols=['After_'+n for n in NAMES]
    np.testing.assert_array_equal(hist.iloc[0][bcols],np.zeros(18));np.testing.assert_array_equal(hist[bcols].iloc[1:],hist[acols].iloc[:-1])
    np.testing.assert_array_equal(hist.loc[hist.NewLabelDays.eq(0),bcols],hist.loc[hist.NewLabelDays.eq(0),acols])
    assert len(u)==1199 and u.SignalDate.is_unique and (u.SignalDate<u.ExitDate).all() and u.Date.eq(u.ExitDate).all()
    if variant=='price_only':assert hist[[f'After_{n}' for n in FINS]].eq(0).all().all()
    umap=u.set_index('SignalDate');batchcount=0;datecount=0;zeros=np.zeros(6,dtype=np.int64);etahist={}
    max_grad=0.;max_loss=0.;max_state=0.;max_daymse=0.;max_rounding=0.;pending=None;dominance=[]
    gradcols=[n+'Gradient' for n in FINS];inputcols=[n+'Input' for n in FINS];finbeforecols=[n+'Before' for n in FINS]
    def audit_day(g):
        nonlocal batchcount,datecount,max_grad,max_loss,max_state,max_daymse,max_rounding
        date=pd.Timestamp(g.SignalDate.iloc[0]);meta=umap.loc[date];ix=groups[date];known=np.isfinite(f.loc[ix,'Target']);ix=ix[known];n=len(ix)
        assert n==meta.Batches==meta.KnownLabelStocks==len(g) and len(known)==meta.ForecastStocks
        assert f.loc[ix,'ExitDate'].eq(meta.ExitDate).all() and g.UpdateDate.eq(str(meta.Date.date())).all()
        order=np.random.default_rng(20260912+int(date.strftime('%Y%m%d'))).permutation(n)
        xx=xall[ix[order]];yy=f.loc[ix[order],'Target'].to_numpy();z=xx[:,12:]
        np.testing.assert_array_equal(g.BatchIndex,np.arange(n));np.testing.assert_array_equal(g[inputcols],z)
        assert g.BatchSize.eq(1).all();accepted=g.Status.eq(0)
        assert g.loc[accepted,'LossAfter'].lt(g.loc[accepted,'LossBefore']).all()
        assert (g.LossBefore-g.LossAfter-1e-4*g.Eta*g.GradNormSquared).ge(-1e-10).all()
        assert g.loc[~accepted,'Eta'].eq(0).all() and g.loc[g.Status.eq(2),'Candidates'].eq(46).all()
        assert g.loc[g.Status.eq(1),'GradInf'].le(1e-10).all()
        grads=g[gradcols].to_numpy();bf=g[finbeforecols].to_numpy();eta=g.Eta.to_numpy()
        dominance.append({'SignalDate':date,'FinancialGradientDominatesBatches':int(((g.GradNormSquared.to_numpy()>0)&(np.sum(grads**2,axis=1)>=.9*g.GradNormSquared.to_numpy())).sum()),'EPSGradientDominatesBatches':int(((g.GradNormSquared.to_numpy()>0)&((grads[:,2]**2+grads[:,5]**2)>=.9*g.GradNormSquared.to_numpy())).sum())})
        post=hist.loc[meta.Date,acols].to_numpy(dtype=float);pre=hist.loc[meta.Date,bcols].to_numpy(dtype=float)
        np.testing.assert_array_equal(pre,meta[bcols].to_numpy(dtype=float));np.testing.assert_array_equal(post,meta[acols].to_numpy(dtype=float))
        nextfin=np.vstack([bf[1:],post[12:]]);diff=nextfin-(bf-eta[:,None]*grads)
        max_rounding=max(max_rounding,float(np.max(abs(diff))))
        np.testing.assert_allclose(nextfin,bf-eta[:,None]*grads,atol=1e-12,rtol=1e-10)
        assert (grads[z==0]==0).all();np.testing.assert_array_equal(nextfin[z==0],bf[z==0]);zeros[:]+=(z==0).sum(axis=0)
        np.testing.assert_allclose(g.GradNormSquared,4*g.LossBefore*np.sum(xx**2,axis=1),atol=1e-7,rtol=1e-10)
        np.testing.assert_allclose(grads**2,4*g.LossBefore.to_numpy()[:,None]*z**2,atol=1e-7,rtol=1e-10)
        # Independent NumPy sequential replay of EVERY applied SGD step.
        # The trace eta is taken as recorded; its full search is separately
        # replayed on every stored gradient snapshot below.
        state=pre.copy();calcgrad=np.empty((n,6));calcloss=np.empty(n);afterloss=np.empty(n)
        for j in range(n):
            err=xx[j]@state-yy[j];gradient=2*err*xx[j]
            calcgrad[j]=gradient[12:];calcloss[j]=err*err
            state=state-eta[j]*gradient
            afterloss[j]=(xx[j]@state-yy[j])**2
        np.testing.assert_allclose(calcgrad,grads,atol=2e-7,rtol=1e-7)
        np.testing.assert_allclose(calcloss,g.LossBefore,atol=1e-10,rtol=1e-8)
        np.testing.assert_allclose(afterloss,g.LossAfter,atol=1e-10,rtol=1e-8)
        np.testing.assert_allclose(state,post,atol=1e-11,rtol=1e-8)
        max_grad=max(max_grad,float(np.max(abs(calcgrad-grads))));max_loss=max(max_loss,float(np.max(abs(calcloss-g.LossBefore.to_numpy()))))
        max_state=max(max_state,float(np.max(abs(state-post))))
        losses=[np.mean((xx@t-yy)**2) for t in [pre,post]];refloss=[meta.FullDayLossBefore,meta.FullDayLossAfter]
        np.testing.assert_allclose(losses,refloss,atol=1e-10,rtol=1e-10);max_daymse=max(max_daymse,float(max(abs(np.array(losses)-refloss))))
        for eta,count in g.loc[accepted,'Eta'].value_counts().items():
            key=format(eta,'.8g');etahist[key]=etahist.get(key,0)+int(count)
        batchcount+=n;datecount+=1
        if datecount%300==0:print(variant,'full sequential applied-step replay',datecount,'days',flush=True)
    for chunk in pd.read_csv(out/'batch_trace.csv.gz',chunksize=100000,float_precision='round_trip'):
        if pending is not None:chunk=pd.concat([pending,chunk],ignore_index=True)
        last=chunk.SignalDate.iloc[-1];pending=chunk.loc[chunk.SignalDate.eq(last)].copy()
        for _,g in chunk.loc[chunk.SignalDate.ne(last)].groupby('SignalDate',sort=False):audit_day(g.reset_index(drop=True))
    if pending is not None:audit_day(pending.reset_index(drop=True))
    assert batchcount==int(u.KnownLabelStocks.sum()) and datecount==len(u)
    with gzip.open(out/'gradient_snapshots.pkl.gz','rb') as h:snaps=pickle.load(h)
    searches=0;max_samplegrad=0.;fdcount=0;max_fd=0.
    for si,s in enumerate(snaps):
        ix=s['indices'];expected=groups[s['SignalDate']];expected=expected[np.isfinite(f.loc[expected,'Target'])]
        np.testing.assert_array_equal(ix,expected)
        expectedorder=np.random.default_rng(20260912+int(s['SignalDate'].strftime('%Y%m%d'))).permutation(len(ix));np.testing.assert_array_equal(s['order'],expectedorder)
        for b,state,tr in zip(s['snapshot_batches'],s['theta_before_gradient'],s['snapshot_trace']):
            j=ix[s['order'][b]];xx=xall[j];y=f.loc[j,'Target'];theta=state[:18];gradient=state[18:]
            new,g,trace=independent_step(xx,y,theta)
            np.testing.assert_allclose(g,gradient,atol=1e-7,rtol=1e-9);np.testing.assert_allclose(trace,tr[2:],atol=1e-7,rtol=1e-8)
            assert trace[5]==tr[7] and trace[6]==tr[8];max_samplegrad=max(max_samplegrad,float(max(abs(g-gradient))));searches+=1
            if b==0 and si in [0,213,214,len(snaps)//2,len(snaps)-1]:
                finite=[]
                for k in range(18):
                    step=1e-5/max(1.,abs(xx[k]));d=np.eye(18)[k]*step
                    finite.append(((xx@(theta+d)-y)**2-(xx@(theta-d)-y)**2)/(2*step))
                np.testing.assert_allclose(g,finite,atol=1e-5,rtol=1e-5);max_fd=max(max_fd,float(max(abs(g-finite))));fdcount+=1
    print(variant,'all applied steps and snapshot candidate searches passed',flush=True)
    daily=pd.read_csv(out/'daily_spread_returns.csv',parse_dates=['Date'],float_precision='round_trip').set_index('Date')
    count=0;dates=[];max_score=0.;max_spread=0.;max_mse=0.
    for year in ['warmup',2018,2019,2020,2021]:
        ranks=pd.read_csv(out/f'ranks_{year}.csv.gz',parse_dates=['Date'],float_precision='round_trip')
        for date,g in ranks.groupby('Date',sort=True):
            dates.append(date);ix=groups[date];t=hist.loc[date,acols].to_numpy(dtype=float)
            score=xall[ix]@t;codes=f.loc[ix,'SecuritiesCode'].to_numpy();order=np.lexsort((codes,-score));g=g.sort_values('Rank').reset_index(drop=True)
            np.testing.assert_array_equal(g.SecuritiesCode,codes[order]);np.testing.assert_array_equal(g.Rank,np.arange(len(ix)))
            np.testing.assert_allclose(g.g,score[order],atol=1e-11,rtol=1e-10);max_score=max(max_score,float(max(abs(g.g-score[order]))))
            np.testing.assert_array_equal(g[FINS],f.loc[ix[order],FINS]);np.testing.assert_array_equal(g.FinancialEventId,f.loc[ix[order],'FinancialEventId'])
            np.testing.assert_allclose(g.FinancialContribution,xall[ix[order],12:]@t[12:],atol=1e-11,rtol=1e-10)
            count+=len(g)
            if date not in daily.index:continue
            y=f.loc[ix[order],'Target'].to_numpy();w=np.linspace(2.,1.,200)
            missing=int(np.isnan(np.r_[y[:200],y[-200:]]).sum());assert missing==daily.loc[date,'SelectedMissingTargets']
            if not missing:
                spread=float(y[:200]@w/w.mean()-y[-200:][::-1]@w/w.mean());max_spread=max(max_spread,abs(spread-daily.loc[date,'OfficialDailySpread']))
            mse=float(np.nanmean((g.g.to_numpy()-y)**2));max_mse=max(max_mse,abs(mse-daily.loc[date,'AllStockForecastMSE']))
    assert len(dates)==len(set(dates))==1199 and set(daily.index)==set(calendar.Date) and pd.Timestamp('2020-10-01') not in dates
    assert max_spread<1e-10 and max_mse<1e-10
    result={'passed':True,'variant':variant,'all_applied_sgd_steps_independently_replayed':batchcount,'training_days_replayed':datecount,
        'zero_financial_input_coefficient_unchanged_counts':dict(zip(FINS,map(int,zeros))),'max_sequential_gradient_error':max_grad,
        'max_sequential_loss_error':max_loss,'max_sequential_final_state_error':max_state,'max_daily_training_mse_error':max_daymse,
        'max_financial_coefficient_step_rounding_error':max_rounding,'snapshot_full_candidate_searches_replayed':searches,
        'max_snapshot_gradient_error':max_samplegrad,'finite_difference_batches':fdcount,'max_finite_difference_error':max_fd,
        'forecast_rows_reproduced':count,'forecast_dates_reproduced':len(dates),'max_forecast_score_error':max_score,
        'max_official_spread_error':max_spread,'max_forecast_mse_error':max_mse,'accepted_eta_counts':etahist,
        'no_future_target_updates':True,'financial_features_frozen_at_signal_date':True,'feature_audit_passed':True}
    dom=pd.DataFrame(dominance).set_index('SignalDate')
    changed=0
    for c in ['FinancialGradientDominatesBatches','EPSGradientDominatesBatches']:
        correct=u.SignalDate.map(dom[c]);changed+=int((u[c]!=correct).sum());u[c]=correct
    result['zero_gradient_dominance_counter_corrections']=changed
    if changed:
        u.to_csv(out/'training_updates.csv',index=False)
        stored=json.loads((out/'results.json').read_text())
        stored['financial_gradient_dominates_batches']=int(u.FinancialGradientDominatesBatches.sum())
        stored['eps_gradient_dominates_batches']=int(u.EPSGradientDominatesBatches.sum())
        stored['diagnostic_dominance_counts_exclude_zero_norm']=True
        save(out/'results.json',stored)
    save(out/'audit.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main(sys.argv[1])
