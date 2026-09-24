import json,pickle,gzip,importlib.util
import numpy as np
import pandas as pd
from financial_features import ROOT,V8
from run_v9 import NAMES,INPUTS,predict,save

spec=importlib.util.spec_from_file_location('v9_independent_preflight',ROOT/'preflight.py')
mathcheck=importlib.util.module_from_spec(spec);spec.loader.exec_module(mathcheck)
independent=mathcheck.independent;independent_step=mathcheck.independent_step

def main():
    out=ROOT/'sgd'
    with (V8/'inputs.pkl').open('rb') as h:f,base_x,groups,calendar=pickle.load(h)
    finance=pd.read_pickle(ROOT/'financial_signal_features.pkl');f=pd.concat([f,finance],axis=1)
    xall=np.column_stack([base_x,finance.FinancialFeature.to_numpy()]);lookup=f.set_index(['SignalDate','SecuritiesCode']).index
    history=pd.read_csv(out/'parameter_history.csv',parse_dates=['Date'],float_precision='round_trip').set_index('Date')
    updates=pd.read_csv(out/'training_updates.csv',parse_dates=['Date','SignalDate','ExitDate'],float_precision='round_trip')
    assert len(updates)==1199 and updates.SignalDate.is_unique
    assert (updates.SignalDate<updates.ExitDate).all() and (updates.ExitDate==updates.Date).all()
    np.testing.assert_array_equal(history.iloc[0][['Before_'+n for n in NAMES]].to_numpy(),np.zeros(13))
    np.testing.assert_allclose(history[['Before_'+n for n in NAMES]].iloc[1:].to_numpy(),history[['After_'+n for n in NAMES]].iloc[:-1].to_numpy(),atol=1e-14,rtol=0)
    assert history.After_alpha.eq(0).all()
    for u in updates.itertuples(index=False):
        source=f.loc[groups[u.SignalDate]];known=source.Target.notna();n=int(known.sum())
        assert u.Batches==u.KnownLabelStocks==n and u.ForecastStocks==len(source)
        assert source.ExitDate.eq(u.ExitDate).all()
        assert u.NonzeroFinancialFeatureStocks==int(source.loc[known,'FinancialFeature'].ne(0).sum())
        np.testing.assert_allclose(np.array([getattr(u,'After_'+name) for name in NAMES]),history.loc[u.Date,['After_'+name for name in NAMES]].to_numpy(),atol=1e-14,rtol=0)
        if u.NonzeroFinancialFeatureStocks==0:assert getattr(u,'Before_'+NAMES[-1])==getattr(u,'After_'+NAMES[-1])
    baseline=pd.read_csv(V8/'sgd/parameter_history.csv',parse_dates=['Date'],float_precision='round_trip').set_index('Date')
    first_finance_update=updates.loc[updates.NonzeroFinancialFeatureStocks.gt(0),'Date'].min()
    early=history.index[history.index<first_finance_update]
    np.testing.assert_allclose(history.loc[early,['After_'+n for n in NAMES[:-1]]],baseline.loc[early,['After_'+n for n in NAMES[:-1]]],atol=1e-12,rtol=1e-10)
    batch_count=0;eta_hist={};perday={};last_beta=0.;max_beta_step_error=0.;zero_own_nonzero_grad=0
    update_map=updates.set_index('SignalDate').Date.to_dict()
    for chunk in pd.read_csv(out/'batch_trace.csv.gz',chunksize=100000,float_precision='round_trip'):
        accepted=chunk.Status.eq(0);batch_count+=len(chunk)
        assert chunk.BatchSize.eq(1).all() and chunk.loc[accepted,'LossAfter'].lt(chunk.loc[accepted,'LossBefore']).all()
        margin=chunk.LossBefore-chunk.LossAfter-1e-4*chunk.Eta*chunk.GradNormSquared
        assert margin.ge(-1e-14).all() and chunk.loc[~accepted,'Eta'].eq(0).all()
        assert chunk.loc[chunk.Status.eq(1),'GradInf'].le(1e-10).all()
        assert chunk.loc[chunk.Status.eq(2),'Candidates'].eq(46).all()
        expected_beta=chunk.BetaBefore-chunk.Eta*chunk.BetaGradient
        error=abs(chunk.BetaBefore.iloc[0]-last_beta);max_beta_step_error=max(max_beta_step_error,float(error))
        assert error<1e-12
        inner=abs(expected_beta.iloc[:-1].to_numpy()-chunk.BetaBefore.iloc[1:].to_numpy())
        max_beta_step_error=max(max_beta_step_error,float(inner.max()) if len(inner) else 0.)
        np.testing.assert_allclose(expected_beta.iloc[:-1],chunk.BetaBefore.iloc[1:],atol=1e-14,rtol=1e-13)
        last_beta=float(expected_beta.iloc[-1])
        zero_own_nonzero_grad+=int((chunk.OwnFinancialFeature.eq(0)&chunk.BetaGradient.abs().gt(1e-12)).sum())
        for eta,count in chunk.loc[accepted,'Eta'].value_counts().items():
            key=format(eta,'.8g');eta_hist[key]=eta_hist.get(key,0)+int(count)
        for signal,g in chunk.groupby('SignalDate',sort=False):
            date=pd.Timestamp(signal);ix=groups[date];ix=ix[f.loc[ix,'Target'].notna()]
            order=np.random.default_rng(20260912+int(date.strftime('%Y%m%d'))).permutation(len(ix))
            batches=g.BatchIndex.to_numpy(dtype=int);previous=perday.get(signal,0)
            np.testing.assert_array_equal(batches,np.arange(previous,previous+len(g)));perday[signal]=previous+len(g)
            np.testing.assert_array_equal(g.OwnFinancialFeature,xall[ix[order[batches]],-1])
            assert g.UpdateDate.eq(str(update_map[date].date())).all()
            if (xall[ix,-1]==0).all():assert g.BetaGradient.eq(0).all()
    assert batch_count==int(updates.KnownLabelStocks.sum())
    assert zero_own_nonzero_grad==int(updates.OwnZeroFeatureNonzeroBetaGradientBatches.sum())
    assert abs(last_beta-history.iloc[-1]['After_'+NAMES[-1]])<1e-12
    for u in updates.itertuples(index=False):assert perday[str(u.SignalDate.date())]==u.KnownLabelStocks
    with gzip.open(out/'gradient_snapshots.pkl.gz','rb') as handle:snaps=pickle.load(handle)
    first_nonzero=int(np.flatnonzero(updates.NonzeroFinancialFeatureStocks.gt(0))[0])
    extreme_date=f.iloc[int(np.argmax(abs(finance.FinancialFeature)))].SignalDate
    sample=sorted(set([0,first_nonzero,len(snaps)//2,len(snaps)-1,next(i for i,s in enumerate(snaps) if s['UpdateDate']==pd.Timestamp('2020-10-01')),
        next(i for i,s in enumerate(snaps) if s['SignalDate']==extreme_date)]))
    replayed=0;fd_count=0;max_gradient_error=0.;max_fd_error=0.;zero_own_gradient_examples=[]
    for si in sample:
        s=snaps[si];ix=s['indices'];x=xall[ix];y=f.loc[ix,'Target'].rank(ascending=False,method='average').to_numpy()
        expected_ix=groups[s['SignalDate']];expected_ix=expected_ix[f.loc[expected_ix,'Target'].notna()]
        np.testing.assert_array_equal(ix,expected_ix)
        expected_order=np.random.default_rng(20260912+int(s['SignalDate'].strftime('%Y%m%d'))).permutation(len(ix))
        np.testing.assert_array_equal(s['order'],expected_order)
        for b,state,tr in zip(s['snapshot_batches'],s['theta_before_gradient'],s['snapshot_trace']):
            theta=state[:13];gradient=state[13:];ids=s['order'][b:b+1]
            new,eta,old,after,ref,nt,status=independent_step(x,y,theta,ids)
            max_gradient_error=max(max_gradient_error,float(abs(gradient-ref).max()))
            np.testing.assert_allclose(gradient,ref,atol=1e-10,rtol=1e-8)
            np.testing.assert_allclose(tr[[2,3,4]],[old,after,eta],atol=1e-10,rtol=1e-8)
            assert tr[7]==nt and tr[8]==status
            if b==0:np.testing.assert_allclose(theta,updates.iloc[si][['Before_'+n for n in NAMES]].to_numpy(dtype=float),atol=1e-13,rtol=0)
            if b==len(ix)-1:np.testing.assert_allclose(new,updates.iloc[si][['After_'+n for n in NAMES]].to_numpy(dtype=float),atol=1e-10,rtol=1e-8)
            if x[ids[0],-1]==0 and abs(ref[-1])>1e-10:
                zero_own_gradient_examples.append({'SignalDate':str(s['SignalDate'].date()),'BatchIndex':int(b),'SecuritiesCode':int(f.loc[ix[ids[0]],'SecuritiesCode']),'OwnFeature':0.,'BetaGradient':float(ref[-1]),'Eta':eta})
            if b==s['snapshot_batches'][0]:
                finite=[]
                for k in range(13):
                    h=1e-6/max(1.,float(abs(x[:,k]).max()));step=np.eye(13)[k]*h
                    finite.append((independent(x,y,theta+step,ids)[0]-independent(x,y,theta-step,ids)[0])/(2*h))
                max_fd_error=max(max_fd_error,float(abs(gradient-finite).max()))
                np.testing.assert_allclose(gradient,finite,atol=2e-6,rtol=5e-4);fd_count+=1
            replayed+=1
    count=0;dates=[];max_spread_error=0.;max_score_error=0.
    daily=pd.read_csv(out/'daily_spread_returns.csv',parse_dates=['Date'],float_precision='round_trip').set_index('Date')
    for path in [out/'ranks_warmup.csv.gz']+[out/f'ranks_{year}.csv.gz' for year in [2018,2019,2020,2021]]:
        ranks=pd.read_csv(path,parse_dates=['Date'],float_precision='round_trip')
        for date,g in ranks.groupby('Date',sort=True):
            dates.append(date);ix=groups[date];theta=history.loc[date,['After_'+n for n in NAMES]].to_numpy(dtype=float)
            columns=['SignalDate','SecuritiesCode']+INPUTS+list(finance.columns)
            reproduced=predict(f.loc[ix,columns],theta).reset_index(drop=True);g=g.sort_values('Rank').reset_index(drop=True)
            np.testing.assert_array_equal(g.SecuritiesCode,reproduced.SecuritiesCode)
            np.testing.assert_array_equal(g.Rank,np.arange(len(ix)))
            np.testing.assert_allclose(g.g,reproduced.g,atol=1e-10,rtol=1e-10)
            np.testing.assert_array_equal(g.FinancialFeature,reproduced.FinancialFeature)
            np.testing.assert_allclose(g.FinancialContribution,reproduced.FinancialContribution,atol=1e-13,rtol=1e-12)
            max_score_error=max(max_score_error,float(abs(g.g-reproduced.g).max()));count+=len(g)
            if date not in daily.index:continue
            keys=pd.MultiIndex.from_arrays([g.Date,g.SecuritiesCode]);j=lookup.get_indexer(keys);assert (j>=0).all()
            y=f.loc[j,'Target'].to_numpy();w=np.linspace(2.,1.,200);selected=np.r_[y[:200],y[-200:]]
            missing=int(np.isnan(selected).sum());assert missing==daily.loc[date,'SelectedMissingTargets']
            if missing:assert pd.isna(daily.loc[date,'OfficialDailySpread'])
            else:
                spread=float(y[:200]@w/w.mean()-y[-200:][::-1]@w/w.mean())
                max_spread_error=max(max_spread_error,abs(spread-daily.loc[date,'OfficialDailySpread']))
    assert len(dates)==1199 and len(set(dates))==1199 and pd.Timestamp('2020-10-01') not in dates
    assert set(daily.index)==set(calendar.Date) and max_spread_error<1e-10
    result={'passed':True,'label_days':len(updates),'audited_batches':batch_count,'forecast_rows_reproduced':count,'forecast_dates_reproduced':len(dates),
        'max_forecast_score_error':max_score_error,'max_official_spread_error':max_spread_error,'max_beta_step_rounding_error':max_beta_step_error,
        'independent_real_batch_replays':replayed,'finite_difference_real_batches':fd_count,'max_analytic_gradient_error':max_gradient_error,'max_finite_difference_error':max_fd_error,
        'own_zero_input_nonzero_beta_gradient_batches':zero_own_nonzero_grad,'zero_own_gradient_examples':zero_own_gradient_examples,
        'accepted_eta_counts':eta_hist,'initial_baseline_state_reproduced_dates':len(early),'first_financial_feature_training_date':str(first_finance_update.date()),
        'all_zero_days_beta_unchanged':True,'training_uses_frozen_signal_date_financial_features':True,'no_future_label_updates':True,
        'feature_audit':json.loads((ROOT/'feature_audit.json').read_text()),'preflight':json.loads((ROOT/'preflight.json').read_text())}
    save(ROOT/'audit.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
