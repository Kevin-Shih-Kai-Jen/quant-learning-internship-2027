import json,pickle,gzip,sys
import numpy as np
import pandas as pd
from run_v11 import ROOT,V8,V9,NAMES,save
from features import load
from preflight import independent_step

def main(variant):
    out=ROOT/variant
    f,base_x,groups,calendar,finance=load(variant)
    xall=np.column_stack([base_x,finance.FinancialFeature.to_numpy()])
    lookup=f.set_index(['SignalDate','SecuritiesCode']).index
    hist=pd.read_csv(out/'parameter_history.csv',parse_dates=['Date'],float_precision='round_trip').set_index('Date')
    updates=pd.read_csv(out/'training_updates.csv',parse_dates=['Date','SignalDate','ExitDate'],float_precision='round_trip')
    beforecols=['Before_'+n for n in NAMES];aftercols=['After_'+n for n in NAMES]
    np.testing.assert_array_equal(hist.iloc[0][beforecols],np.zeros(13))
    np.testing.assert_array_equal(hist[beforecols].iloc[1:],hist[aftercols].iloc[:-1])
    np.testing.assert_array_equal(hist.loc[hist.NewLabelDays.eq(0),beforecols],hist.loc[hist.NewLabelDays.eq(0),aftercols])
    assert hist.After_alpha.ne(0).any()
    assert len(updates)==1199 and updates.SignalDate.is_unique
    assert (updates.SignalDate<updates.ExitDate).all() and updates.ExitDate.eq(updates.Date).all()
    day_loss_error=0.
    for u in updates.itertuples(index=False):
        source=f.loc[groups[u.SignalDate]];ix=source.index[np.isfinite(source.Target)]
        assert u.KnownLabelStocks==u.Batches==len(ix) and u.ForecastStocks==len(source)
        assert source.ExitDate.eq(u.ExitDate).all()
        assert u.NonzeroFinancialFeatureStocks==int((xall[ix,-1]!=0).sum())
        pre=np.array([getattr(u,c) for c in beforecols]);post=np.array([getattr(u,c) for c in aftercols])
        np.testing.assert_array_equal(post,hist.loc[u.Date,aftercols])
        np.testing.assert_array_equal(pre,hist.loc[u.Date,beforecols])
        losses=[np.mean((xall[ix]@t-f.loc[ix,'Target'].to_numpy())**2) for t in [pre,post]]
        ref=[u.FullDayLossBefore,u.FullDayLossAfter]
        np.testing.assert_allclose(losses,ref,atol=1e-12,rtol=1e-11)
        day_loss_error=max(day_loss_error,float(max(abs(np.array(losses)-ref))))
    batches=0;last_beta=0.;max_beta_error=0.;eta_hist={};perday={};ownzero=0;zeroskips=0
    update_map=updates.set_index('SignalDate').Date.to_dict()
    for c in pd.read_csv(out/'batch_trace.csv.gz',chunksize=100000,float_precision='round_trip'):
        accepted=c.Status.eq(0);batches+=len(c)
        assert c.BatchSize.eq(1).all() and c.loc[accepted,'LossAfter'].lt(c.loc[accepted,'LossBefore']).all()
        assert (c.LossBefore-c.LossAfter-1e-4*c.Eta*c.GradNormSquared).ge(-1e-12).all()
        assert c.loc[~accepted,'Eta'].eq(0).all()
        assert c.loc[c.Status.eq(1),'GradInf'].le(1e-10).all()
        assert c.loc[c.Status.eq(2),'Candidates'].eq(46).all()
        beta_next=c.BetaBefore-c.Eta*c.BetaGradient
        err=abs(c.BetaBefore.iloc[0]-last_beta);max_beta_error=max(max_beta_error,float(err));assert err<1e-12
        errs=abs(beta_next.iloc[:-1].to_numpy()-c.BetaBefore.iloc[1:].to_numpy())
        max_beta_error=max(max_beta_error,float(max(errs)))
        np.testing.assert_allclose(beta_next.iloc[:-1],c.BetaBefore.iloc[1:],atol=1e-14,rtol=1e-12)
        zero=c.OwnFinancialFeature.eq(0);ownzero+=int(zero.sum())
        assert c.loc[zero,'BetaGradient'].eq(0).all()
        np.testing.assert_array_equal(beta_next[zero],c.loc[zero,'BetaBefore'])
        internal_zero=zero.iloc[:-1].to_numpy()
        np.testing.assert_array_equal(c.BetaBefore.iloc[1:].to_numpy()[internal_zero],c.BetaBefore.iloc[:-1].to_numpy()[internal_zero])
        last_beta=float(beta_next.iloc[-1])
        for eta,n in c.loc[accepted,'Eta'].value_counts().items():
            key=format(eta,'.8g');eta_hist[key]=eta_hist.get(key,0)+int(n)
        for signal,g in c.groupby('SignalDate',sort=False):
            date=pd.Timestamp(signal);ix=groups[date];ix=ix[np.isfinite(f.loc[ix,'Target'])]
            order=np.random.default_rng(20260912+int(date.strftime('%Y%m%d'))).permutation(len(ix))
            bb=g.BatchIndex.to_numpy(dtype=int);prev=perday.get(signal,0)
            np.testing.assert_array_equal(bb,np.arange(prev,prev+len(g)));perday[signal]=prev+len(g)
            xx=xall[ix[order[bb]]]
            np.testing.assert_array_equal(g.OwnFinancialFeature,xx[:,-1])
            # For single-observation MSE, ||gradient||² = 4 * loss * ||x||².
            np.testing.assert_allclose(g.GradNormSquared,4*g.LossBefore*np.einsum('ij,ij->i',xx,xx),atol=1e-10,rtol=1e-11)
            np.testing.assert_allclose(g.BetaGradient**2,4*g.LossBefore*xx[:,-1]**2,atol=1e-10,rtol=1e-11)
            assert g.UpdateDate.eq(str(update_map[date].date())).all()
    assert batches==int(updates.KnownLabelStocks.sum())
    assert abs(last_beta-hist.iloc[-1][aftercols[-1]])<1e-12
    assert updates.OwnZeroFeatureNonzeroBetaGradientBatches.eq(0).all() and updates.OwnZeroFeatureActualBetaChanges.eq(0).all()
    for u in updates.itertuples(index=False):assert perday[str(u.SignalDate.date())]==u.KnownLabelStocks
    print(variant,'all batch traces and daily MSE audited',flush=True)
    with gzip.open(out/'gradient_snapshots.pkl.gz','rb') as h:snaps=pickle.load(h)
    replayed=0;fdcount=0;max_grad=0.;max_fd=0.;examples=[]
    for si,s in enumerate(snaps):
        ix=groups[s['SignalDate']];ix=ix[np.isfinite(f.loc[ix,'Target'])]
        np.testing.assert_array_equal(ix,s['indices'])
        order=np.random.default_rng(20260912+int(s['SignalDate'].strftime('%Y%m%d'))).permutation(len(ix))
        np.testing.assert_array_equal(order,s['order'])
        for b,state,tr in zip(s['snapshot_batches'],s['theta_before_gradient'],s['snapshot_trace']):
            j=ix[order[b]];x=xall[j];y=f.loc[j,'Target'];t=state[:13];gradient=state[13:]
            new,g,ref=independent_step(x,y,t)
            max_grad=max(max_grad,float(max(abs(g-gradient))))
            np.testing.assert_allclose(gradient,g,atol=1e-10,rtol=1e-10)
            np.testing.assert_allclose(tr[2:],ref,atol=1e-10,rtol=1e-9)
            assert tr[7]==ref[5] and tr[8]==ref[6]
            if b==0:np.testing.assert_array_equal(t,updates.iloc[si][beforecols].to_numpy(dtype=float))
            if b==len(ix)-1:np.testing.assert_allclose(new,updates.iloc[si][aftercols].to_numpy(dtype=float),atol=1e-12,rtol=1e-10)
            if b==0 and si in [0,213,214,len(snaps)//2,len(snaps)-1]:
                fd=[]
                for k in range(13):
                    h=1e-5/max(1.,abs(x[k]));d=np.eye(13)[k]*h
                    fd.append(((x@(t+d)-y)**2-(x@(t-d)-y)**2)/(2*h))
                max_fd=max(max_fd,float(max(abs(g-fd))));np.testing.assert_allclose(g,fd,atol=1e-7,rtol=1e-5);fdcount+=1
            if x[-1]==0 and len(examples)<2 and t[-1]!=0:
                examples.append({'SignalDate':str(s['SignalDate'].date()),'Code':int(f.loc[j,'SecuritiesCode']),'BatchIndex':int(b),'BetaBefore':float(t[-1]),'BetaAfter':float(new[-1]),'BetaGradient':float(g[-1])})
            replayed+=1
    print(variant,'independent gradients and eta replayed',replayed,flush=True)
    daily=pd.read_csv(out/'daily_spread_returns.csv',parse_dates=['Date'],float_precision='round_trip').set_index('Date')
    dates=[];count=0;max_score=0.;max_spread=0.;max_forecast_mse=0.
    for name in ['warmup',2018,2019,2020,2021]:
        ranks=pd.read_csv(out/f'ranks_{name}.csv.gz',parse_dates=['Date'],float_precision='round_trip')
        for date,g in ranks.groupby('Date',sort=True):
            dates.append(date);ix=groups[date];t=hist.loc[date,aftercols].to_numpy(dtype=float)
            score=xall[ix]@t;codes=f.loc[ix,'SecuritiesCode'].to_numpy();order=np.lexsort((codes,-score))
            g=g.sort_values('Rank').reset_index(drop=True)
            np.testing.assert_array_equal(g.SecuritiesCode,codes[order]);np.testing.assert_array_equal(g.Rank,np.arange(len(ix)))
            np.testing.assert_allclose(g.g,score[order],atol=1e-12,rtol=1e-11)
            max_score=max(max_score,float(max(abs(g.g-score[order]))));count+=len(g)
            for col in finance.columns:np.testing.assert_array_equal(g[col],finance.loc[ix[order],col])
            np.testing.assert_allclose(g.FinancialContribution,t[-1]*xall[ix[order],-1],atol=1e-13,rtol=1e-12)
            if date not in daily.index:continue
            y=f.loc[ix[order],'Target'].to_numpy();w=np.linspace(2.,1.,200)
            missing=int(np.isnan(np.r_[y[:200],y[-200:]]).sum());assert missing==daily.loc[date,'SelectedMissingTargets']
            if not missing:
                spread=float(y[:200]@w/w.mean()-y[-200:][::-1]@w/w.mean())
                max_spread=max(max_spread,abs(spread-daily.loc[date,'OfficialDailySpread']))
            fmse=float(np.nanmean((g.g.to_numpy()-y)**2));max_forecast_mse=max(max_forecast_mse,abs(fmse-daily.loc[date,'AllStockForecastMSE']))
    assert len(dates)==len(set(dates))==1199 and pd.Timestamp('2020-10-01') not in dates
    assert set(daily.index)==set(calendar.Date) and max_spread<1e-10 and max_forecast_mse<1e-10
    result={'passed':True,'variant':variant,'audited_batches':batches,'own_zero_feature_batches_beta_unchanged':ownzero,
        'all_batch_mse_gradient_norm_identities_verified':True,'daily_training_mse_max_error':day_loss_error,
        'max_beta_step_rounding_error':max_beta_error,'independent_snapshot_batch_replays':replayed,'max_analytic_gradient_error':max_grad,
        'finite_difference_real_batches':fdcount,'max_finite_difference_error':max_fd,'zero_feature_beta_examples':examples,
        'forecast_dates_reproduced':len(dates),'forecast_rows_reproduced':count,'max_forecast_score_error':max_score,
        'max_official_spread_error':max_spread,'max_forecast_mse_error':max_forecast_mse,'accepted_eta_counts':eta_hist,
        'no_future_label_updates':True,'training_financial_features_frozen_at_signal_date':True,
        'source_feature_audit_passed':json.loads((ROOT/'feature_audit.json').read_text())['passed']}
    save(out/'audit.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main(sys.argv[1])
