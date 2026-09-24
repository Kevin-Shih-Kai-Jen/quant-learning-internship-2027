import sys,json,ctypes as ct,math,time,zipfile
import numpy as np
import pandas as pd
from features import ROOT,load,VARIANTS,V14
from native import coefficients,gradient,initial,DP,IP,dp,ip
from preflight import search
from run import NAMES,ranking_metrics
from refine_replay import refine
LIB=ct.CDLL(str(ROOT/'replay.dylib'))
LIB.replay_day.argtypes=[ct.c_int,DP,DP,IP,ct.c_int,DP,DP,DP,DP,DP]
LIB.replay_day.restype=ct.c_int

def audit(variant,f,x,groups,calendar,fin,resume=False):
    start=time.monotonic();out=ROOT/variant
    baseline=pd.read_csv(V14/variant/'training_updates.csv',parse_dates=['Date','SignalDate','ExitDate'])
    same=['Date','SignalDate','ExitDate','KnownLabelStocks','TrainingStocks','ThresholdSkippedStocks','MissingTargets','SGDAttempts','FullAttempts']
    updates=baseline;expected_states={};replay_stats=[];refinements={}
    cached=json.loads((out/'replay_stats.json').read_text()) if resume else None
    if cached:assert [d['SignalDate'] for d in cached['days']]==baseline.SignalDate.dt.strftime('%Y-%m-%d').tolist()
    maxstats=np.zeros(7);maxstate=None if resume else 0.;maxloss=None if resume else 0.;sampled=0;fdmax=0.;previous=initial()
    for counter,u in enumerate(updates.itertuples(index=False)):
        assert u.SignalDate<u.ExitDate<=u.Date
        # May audit completed days while the trainer continues later days.
        # Never infer success from a missing/incomplete trace.
        path=out/'traces'/f'{u.SignalDate.date()}.npz'
        while True:
            try:z=np.load(path);break
            except (FileNotFoundError,EOFError,zipfile.BadZipFile):time.sleep(.5)
        ix=z['indices'];order=z['order'];n=len(ix)
        raw=groups[u.SignalDate];expected=raw[np.isfinite(f.loc[raw,'Target']).to_numpy()&fin.iloc[raw].TrainingEligible.to_numpy()]
        assert n==u.TrainingStocks and n+u.ThresholdSkippedStocks==u.KnownLabelStocks
        assert (np.abs(x[ix,12:])<=100).all()
        np.testing.assert_array_equal(ix,expected)
        np.testing.assert_array_equal(order,np.random.default_rng(20260912+int(u.SignalDate.strftime('%Y%m%d'))).permutation(n))
        full=math.isqrt(n) if variant=='sgd_sqrt' else 0
        assert full==u.FullAttempts and n==u.SGDAttempts and z['trace'].shape==(n+full,2,10)
        np.testing.assert_array_equal(previous,z['theta_before'])
        xx=np.ascontiguousarray(x[ix]);yy=np.ascontiguousarray(f.loc[ix,'Target'].rank(ascending=False,method='average').to_numpy())
        np.testing.assert_array_equal(yy,z['truth_rank'])
        theta=z['theta_before'].copy();tr=np.ascontiguousarray(z['trace']);snap=np.ascontiguousarray(z['snapshots']);stats=np.zeros(7);losses=np.zeros(3)
        if cached:
            # Certificate was persisted only after ALL per-day replay, state,
            # stage-loss, eligibility, and sampled-grid assertions completed.
            stats=np.array(cached['days'][counter]['relative_errors'])
        else:
            status=LIB.replay_day(n,dp(xx),dp(yy),ip(order),full,dp(tr),dp(snap),dp(theta),dp(stats),dp(losses))
            assert status==n+full
            maxstate=max(maxstate,float(abs(theta-z['theta_after']).max()))
            np.testing.assert_allclose(theta,z['theta_after'],atol=1e-8,rtol=1e-7)
            maxloss=max(maxloss,float(np.max(abs(losses-z['losses'])/(1+abs(z['losses'])))))
            np.testing.assert_allclose(losses,z['losses'],atol=1e-8,rtol=1e-7)
        replay_stats.append({'SignalDate':str(u.SignalDate.date()),'relative_errors':stats.tolist()})
        if max(stats)>1e-7:
            date=str(u.SignalDate.date());path=out/f'refined_replay_{date}.json'
            refined=json.loads(path.read_text()) if path.exists() else refine(xx,z)
            assert refined['passed'] and refined['all_regenerated_training_arrays_bitwise_equal'] and refined['stock_steps']==n and refined['full_steps']==full
            path.write_text(json.dumps(refined,indent=2));refinements[date]=refined
            stats[:5]=refined['max_relative_errors_loss_before_after_gradinf_norm_descent'];stats[5]=refined['max_relative_checkpoint_parameter_error']
            print('REFINED',date,'errors',stats.tolist(),flush=True)
        maxstats=np.maximum(maxstats,stats)
        accepted=tr[:,:,8]==0
        assert (tr[:,:,3][accepted]<tr[:,:,2][accepted]).all()
        assert (tr[:,:,3][accepted]<=tr[:,:,2][accepted]-1e-4*tr[:,:,9][accepted]+1e-13).all()
        assert np.isfinite(tr).all() and np.isfinite(snap).all() and np.isfinite(z['theta_after']).all()
        assert z['theta_after'][0]==0 and ((z['theta_after'][24:]>=0)&(z['theta_after'][24:]<=1)).all()
        expected_states[u.Date]=z['theta_after'].copy()
        previous=z['theta_after']
        if counter%100==0 or counter==len(updates)-1:
            for si,stepidx in [(0,0),(1,n//2),(2,n-1)]+([(3,n),(2+full,n+full-1)] if full else []):
                chosen=int(order[stepidx]) if stepidx<n else -1
                for block in range(2):
                    before=snap[si,block,:27];sg=snap[si,block,27:]
                    nt,trace,ng=search(xx,yy,before,chosen,block)
                    np.testing.assert_allclose(ng,sg,atol=1e-10,rtol=1e-8)
                    np.testing.assert_allclose(trace,tr[stepidx,block,[2,3,4,7,8,9]],atol=1e-10,rtol=1e-8)
                    sampled+=1
            if counter in [100,500,1000]:
                before=snap[0,0,:27];chosen=int(order[0]);_,g,_=gradient(xx,yy,before,chosen)
                for j in range(27):
                    a=before.copy();b=before.copy();a[j]+=1e-6;b[j]-=1e-6
                    fd=(gradient(xx,yy,a,chosen)[0]-gradient(xx,yy,b,chosen)[0])/2e-6
                    fdmax=max(fdmax,abs(fd-g[j])/(1+abs(g[j])))
        if (counter+1)%200==0:print(variant,'audited days',counter+1,'seconds',round(time.monotonic()-start,1),flush=True)
    (out/'replay_stats.json').write_text(json.dumps({'days':replay_stats,'maximum_errors_after_refinement':maxstats.tolist(),'finite_difference_error':fdmax},indent=2))
    print('REPLAY_MAXIMA',maxstats.tolist(),'finite_difference',fdmax,flush=True)
    assert np.max(maxstats)<1e-7 and fdmax<1e-6
    while not (out/'results.json').exists():time.sleep(.5)
    results=json.loads((out/'results.json').read_text())
    states=pd.read_csv(out/'parameter_history.csv',parse_dates=['Date']).set_index('Date')
    updates=pd.read_csv(out/'training_updates.csv',parse_dates=['Date','SignalDate','ExitDate'])
    pd.testing.assert_frame_equal(updates[same],baseline[same])
    for date,theta in expected_states.items():
        np.testing.assert_allclose(states.loc[date,[f'After_{c}' for c in NAMES]].to_numpy(dtype=float),theta,atol=1e-14)
    ranks=pd.concat([pd.read_csv(p,parse_dates=['Date']) for p in sorted(out.glob('ranks_*.csv.gz'))],ignore_index=True)
    recorded=dict(tuple(ranks.groupby('Date')));maxpred=0.;rankrows=0
    for date,idx in groups.items():
        if date==pd.Timestamp('2020-10-01') or date>calendar.Date.max():continue
        theta=states.loc[date,[f'After_{c}' for c in NAMES]].to_numpy(dtype=float)
        score=x[idx]@coefficients(theta);code=f.loc[idx,'SecuritiesCode'].to_numpy();order=np.lexsort((code,-score));got=recorded[date].sort_values('Rank')
        np.testing.assert_array_equal(got.SecuritiesCode,code[order]);np.testing.assert_array_equal(got.Rank,np.arange(len(idx)))
        np.testing.assert_allclose(got.g,score[order],atol=2e-9,rtol=1e-10)
        maxpred=max(maxpred,float(abs(got.g.to_numpy()-score[order]).max()));rankrows+=len(idx)
    labels=f[['SignalDate','SecuritiesCode','Target']].rename(columns={'SignalDate':'Date'})
    valid=ranks.loc[ranks.ValidationYear.ne(0)].merge(labels,on=['Date','SecuritiesCode'],validate='one_to_one')
    daily=pd.read_csv(out/'daily_spread_returns.csv',parse_dates=['Date']).set_index('Date');spreads=[];maxspread=0.;weights=np.linspace(2,1,200)
    assert np.array_equal(daily.index,calendar.Date)
    for date,g in valid.groupby('Date'):
        g=g.sort_values('Rank');long=g.head(200).Target.to_numpy();short=g.tail(200).Target.to_numpy()[::-1]
        assert np.isfinite(long).all() and np.isfinite(short).all()
        s=(long-short)@weights/weights.mean();maxspread=max(maxspread,abs(s-daily.loc[date,'OfficialDailySpread']));spreads.append(s)
    rm=ranking_metrics(valid).set_index('Date')
    np.testing.assert_allclose(rm.RankIC,daily.RankIC,atol=1e-12,equal_nan=True)
    np.testing.assert_allclose(rm.NormalizedHardRankMSE,daily.NormalizedHardRankMSE,atol=1e-12)
    sharpe=np.mean(spreads)/np.std(spreads,ddof=1)
    np.testing.assert_allclose(sharpe,results['validation']['official_style_unannualized_sharpe'],atol=1e-13)
    np.testing.assert_allclose(rm.RankIC.mean(),results['mean_daily_rank_ic'],atol=1e-12)
    np.testing.assert_allclose(rm.NormalizedHardRankMSE.mean(),results['mean_daily_normalized_hard_rank_mse'],atol=1e-12)
    result={'passed':True,'days_replayed':len(updates),'stock_updates_replayed':int(updates.SGDAttempts.sum()),'full_updates_replayed':int(updates.FullAttempts.sum()),
        'replay_method':'Independent direct score/Jacobian replay of every recorded accepted rate; anchored at first/middle/last SGD stocks and every full update. Full candidate grids independently checked at deterministic sampled snapshots.',
        'gradient_substeps_replayed':int(2*(updates.SGDAttempts.sum()+updates.FullAttempts.sum())),
        'max_relative_errors_loss_before_after_gradinf_norm_descent_anchor_gradient':maxstats.tolist(),
        'max_daily_final_parameter_error':maxstate,'max_relative_daily_stage_loss_error':maxloss,
        'reused_completed_daily_replay_certificate':resume,'all_daily_final_parameter_checks_passed':True,'all_daily_stage_loss_checks_passed':True,
        'refined_days':refinements,
        'independent_numpy_full_grid_substeps':sampled,'finite_difference_relative_error':fdmax,
        'forecast_rows_reproduced':rankrows,'all_integer_ranks_equal':True,'max_score_error':maxpred,
        'same_v14_training_schedule_and_pools':True,'max_spread_error':maxspread,'validation_sharpe_reproduced':float(sharpe),'seconds':time.monotonic()-start}
    (out/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True);return result

if __name__=='__main__':
    data=load()
    for v in [a for a in sys.argv[1:] if a!='--resume'] or VARIANTS:audit(v,*data,resume='--resume' in sys.argv)
