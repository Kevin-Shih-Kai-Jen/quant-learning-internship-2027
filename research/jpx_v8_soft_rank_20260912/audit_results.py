from pathlib import Path
import json,pickle,gzip,sys
import numpy as np
import pandas as pd
from run_v8 import ROOT,VARIANTS,NAMES,INPUTS,predict_rank_v5,save
from preflight import independent,independent_step

def main():
    with (ROOT/'inputs.pkl').open('rb') as h:f,xall,groups,calendar=pickle.load(h)
    lookup=f.set_index(['SignalDate','SecuritiesCode']).index;reports={}
    for variant in VARIANTS:
        out=ROOT/variant
        history=pd.read_csv(out/'parameter_history.csv',parse_dates=['Date'],float_precision='round_trip').set_index('Date')
        updates=pd.read_csv(out/'training_updates.csv',parse_dates=['Date','SignalDate','ExitDate'],float_precision='round_trip')
        assert len(updates)==1199 and updates.SignalDate.is_unique
        assert (updates.SignalDate<updates.ExitDate).all() and (updates.ExitDate==updates.Date).all()
        assert pd.Timestamp('2020-10-01') in updates.Date.values
        np.testing.assert_array_equal(history.iloc[0][['Before_'+n for n in NAMES]].to_numpy(),np.zeros(12))
        np.testing.assert_allclose(history[['Before_'+n for n in NAMES]].iloc[1:].to_numpy(),history[['After_'+n for n in NAMES]].iloc[:-1].to_numpy(),atol=1e-14,rtol=0)
        assert (history.Before_alpha==0).all() and (history.After_alpha==0).all()
        for u in updates.itertuples(index=False):
            source=f.loc[groups[u.SignalDate]];n=int(source.Target.notna().sum())
            assert u.KnownLabelStocks==n and u.ForecastStocks==len(source)
            assert source.ExitDate.eq(u.ExitDate).all()
            expected=1 if variant=='daily' else n if variant=='sgd' else (n+127)//128
            assert u.Batches==expected
            assert u.LastBatchSize==(n if variant=='daily' else 1 if variant=='sgd' else (n-1)%128+1)
            np.testing.assert_allclose(np.array([getattr(u,'After_'+name) for name in NAMES]),history.loc[u.Date,['After_'+name for name in NAMES]].to_numpy(),atol=1e-14,rtol=0)
        batch_count=accepted_count=skipped_count=0;eta_hist={};min_margin=1.
        perday={}
        for chunk in pd.read_csv(out/'batch_trace.csv.gz',chunksize=100000,float_precision='round_trip'):
            a=chunk.Status.eq(0);batch_count+=len(chunk);accepted_count+=int(a.sum());skipped_count+=int((~a).sum())
            assert chunk[a].LossAfter.lt(chunk[a].LossBefore).all()
            margin=chunk.LossBefore-chunk.LossAfter-1e-4*chunk.Eta*chunk.GradNormSquared
            assert (margin>=-1e-14).all();min_margin=min(min_margin,float(margin.min()))
            assert chunk.loc[~a,'Eta'].eq(0).all()
            assert chunk.loc[chunk.Status.eq(1),'GradInf'].le(1e-10).all()
            assert chunk.loc[chunk.Status.eq(2),'Candidates'].eq(46).all()
            for eta,count in chunk.loc[a,'Eta'].value_counts().items():
                key=format(eta,'.8g');eta_hist[key]=eta_hist.get(key,0)+int(count)
            for date,g in chunk.groupby('SignalDate',sort=False):
                p=perday.setdefault(date,{'count':0,'stocks':0,'last':-1})
                assert np.array_equal(g.BatchIndex.to_numpy(),np.arange(p['last']+1,p['last']+1+len(g)))
                p['count']+=len(g);p['stocks']+=int(g.BatchSize.sum());p['last']=int(g.BatchIndex.iloc[-1])
        for u in updates.itertuples(index=False):
            p=perday[str(u.SignalDate.date())];assert p['count']==u.Batches and p['stocks']==u.KnownLabelStocks
        # Independently replay saved first/middle/last batches from five dates,
        # including closure-label release and both ends of the entire history.
        with gzip.open(out/'gradient_snapshots.pkl.gz','rb') as h:snaps=pickle.load(h)
        sample_indices=sorted(set([0,1,len(snaps)//2,len(snaps)-1,next(i for i,s in enumerate(snaps) if s['UpdateDate']==pd.Timestamp('2020-10-01'))]))
        checked=0;max_grad_error=0.;max_finite_error=0.
        for si in sample_indices:
            s=snaps[si];ix=s['indices'];x=xall[ix];y=f.loc[ix,'Target'].rank(ascending=False,method='average').to_numpy()
            np.testing.assert_array_equal(ix,groups[s['SignalDate']][f.loc[groups[s['SignalDate']],'Target'].notna()])
            expected_order=np.random.default_rng(20260912+int(s['SignalDate'].strftime('%Y%m%d'))).permutation(len(ix))
            np.testing.assert_array_equal(s['order'],expected_order)
            for b,ss,tr in zip(s['snapshot_batches'],s['theta_before_gradient'],s['snapshot_trace']):
                theta=ss[:12];grad=ss[12:];ids=s['order'][b*s['batch_size']:(b+1)*s['batch_size']]
                new,eta,old,after,refgrad,candidates,status=independent_step(x,y,theta,ids)
                max_grad_error=max(max_grad_error,float(abs(grad-refgrad).max()))
                np.testing.assert_allclose(grad,refgrad,atol=1e-10,rtol=1e-8)
                np.testing.assert_allclose(tr[[2,3,4]],[old,after,eta],atol=1e-10,rtol=1e-8)
                assert tr[7]==candidates and tr[8]==status
                if b==0:
                    np.testing.assert_allclose(theta,updates.iloc[si][['Before_'+n for n in NAMES]].to_numpy(dtype=float),atol=1e-13,rtol=0)
                if b==int(updates.iloc[si].Batches)-1:
                    np.testing.assert_allclose(new,updates.iloc[si][['After_'+n for n in NAMES]].to_numpy(dtype=float),atol=1e-10,rtol=1e-8)
                # One real batch per selected date: independent central differences.
                if b==s['snapshot_batches'][0]:
                    finite=[]
                    for k in range(12):
                        h=1e-6/max(1.,float(abs(x[:,k]).max()));step=np.eye(12)[k]*h
                        finite.append((independent(x,y,theta+step,ids)[0]-independent(x,y,theta-step,ids)[0])/(2*h))
                    max_finite_error=max(max_finite_error,float(abs(grad-finite).max()))
                    np.testing.assert_allclose(grad,finite,atol=2e-6,rtol=5e-4)
                checked+=1
        # Every saved forecast is rebuilt with its own as-of parameter vector,
        # using a feature-only function; every official spread is recalculated.
        forecasts=0;spread_errors=[];all_dates=[]
        daily=pd.read_csv(out/'daily_spread_returns.csv',parse_dates=['Date'],float_precision='round_trip').set_index('Date')
        for file in [out/'ranks_warmup.csv.gz']+[out/f'ranks_{y}.csv.gz' for y in [2018,2019,2020,2021]]:
            ranks=pd.read_csv(file,parse_dates=['Date'],float_precision='round_trip')
            for date,g in ranks.groupby('Date',sort=True):
                all_dates.append(date);idx=groups[date];theta=history.loc[date,['After_'+n for n in NAMES]].to_numpy(dtype=float)
                recreated=predict_rank_v5(f.loc[idx,['SignalDate','SecuritiesCode']+INPUTS],theta)
                g=g.sort_values('Rank').reset_index(drop=True);recreated=recreated.reset_index(drop=True)
                np.testing.assert_array_equal(g.SecuritiesCode,recreated.SecuritiesCode)
                np.testing.assert_array_equal(g.Rank,np.arange(len(idx)))
                np.testing.assert_allclose(g.g,recreated.g,atol=1e-10,rtol=1e-10)
                forecasts+=len(g)
                if date not in daily.index:continue
                ix=lookup.get_indexer(pd.MultiIndex.from_arrays([g.Date,g.SecuritiesCode]));assert (ix>=0).all()
                y=f.loc[ix,'Target'].to_numpy();w=np.linspace(2.,1.,200)
                selected=np.r_[y[:200],y[-200:][::-1]]
                missing=int(np.isnan(selected).sum());assert missing==daily.loc[date,'SelectedMissingTargets']
                if not missing:
                    spread=float(y[:200]@w/w.mean()-y[-200:][::-1]@w/w.mean())
                    spread_errors.append(abs(spread-daily.loc[date,'OfficialDailySpread']))
                else:assert pd.isna(daily.loc[date,'OfficialDailySpread'])
        assert len(all_dates)==1199 and len(set(all_dates))==1199 and pd.Timestamp('2020-10-01') not in all_dates
        assert set(daily.index)==set(calendar.Date) and max(spread_errors)<1e-10
        reports[variant]={'passed':True,'label_days':len(updates),'audited_batches':batch_count,'accepted_updates':accepted_count,'skipped_updates':skipped_count,
            'independent_real_batch_replays':checked,'finite_difference_real_batches':len(sample_indices),'max_analytic_gradient_error':max_grad_error,
            'max_finite_difference_gradient_error':max_finite_error,'forecast_rows_reproduced':forecasts,'forecast_dates_reproduced':len(all_dates),
            'max_official_daily_spread_difference':max(spread_errors),'accepted_eta_counts':eta_hist,'minimum_armijo_margin':min_margin,
            'all_batch_sizes_and_full_coverage_verified':True,'no_future_label_updates':True,'cross_year_parameter_continuity':True}
        save(out/'audit.json',reports[variant])
        print('AUDITED',variant,json.dumps(reports[variant]),flush=True)
    save(ROOT/'audit.json',{'all_checks_passed':True,'variants':reports,'preflight':json.loads((ROOT/'preflight.json').read_text())})

if __name__=='__main__':main()
