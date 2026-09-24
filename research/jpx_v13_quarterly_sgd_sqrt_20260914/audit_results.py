import sys,json,ctypes as ct,math,time
import numpy as np
import pandas as pd
from features import ROOT,load,VARIANTS
from native import coefficients,gradient,initial,DP,IP,dp,ip,train
from preflight import step
from run import NAMES
LIB=ct.CDLL(str(ROOT/'replay.dylib'))
LIB.replay.argtypes=[ct.c_int,DP,DP,IP,ct.c_int,DP,DP,DP,DP];LIB.replay.restype=ct.c_int
LIB.replay_segment.argtypes=[ct.c_int,DP,DP,IP,ct.c_int,ct.c_int,DP,DP,DP];LIB.replay_segment.restype=ct.c_int

def audit(variant,f,x,groups,calendar,fin):
    start=time.monotonic();out=ROOT/variant
    states=pd.read_csv(out/'parameter_history.csv',parse_dates=['Date']).set_index('Date')
    updates=pd.read_csv(out/'training_updates.csv',parse_dates=['Date','SignalDate','ExitDate'])
    results=json.loads((out/'results.json').read_text());fullrows=0;stockrows=0;maxstats=np.zeros(5);maxstate=0.;maxloss=0.;sampled=0;fdmax=0.
    previous=initial()
    for counter,u in enumerate(updates.itertuples(index=False)):
        assert u.SignalDate<u.ExitDate<=u.Date
        z=np.load(out/'traces'/f'{u.SignalDate.date()}.npz');ix=z['indices'];order=z['order'];n=len(ix)
        raw=groups[u.SignalDate];expected=raw[np.isfinite(f.loc[raw,'Target'])]
        np.testing.assert_array_equal(ix,expected)
        np.testing.assert_array_equal(order,np.random.default_rng(20260912+int(u.SignalDate.strftime('%Y%m%d'))).permutation(n))
        full=math.isqrt(n) if variant=='sgd_sqrt' else 0
        assert full==u.FullAttempts and n==u.SGDAttempts and z['trace'].shape==(n+full,2,10)
        np.testing.assert_allclose(previous,z['theta_before'],atol=1e-14)
        xx=np.ascontiguousarray(x[ix]);yy=np.ascontiguousarray(f.loc[ix,'Target'].to_numpy());theta=z['theta_before'].copy()
        tr=np.ascontiguousarray(z['trace']);stats=np.zeros(5)
        # Recreate checkpoints without altering the numerical training path.
        # Demand BITWISE agreement with every saved trace and final parameter.
        regenerated,rt,rs,rl,checkpoints=train(xx,yy,theta,order,full,capture=True)
        np.testing.assert_array_equal(regenerated,z['theta_after']);np.testing.assert_array_equal(rt,tr)
        np.testing.assert_array_equal(rs,z['snapshots']);np.testing.assert_array_equal(rl,z['losses'])
        starts=list(range(0,n,8))+list(range(n,n+full));ends=starts[1:]+[n+full]
        for p,(begin,end) in enumerate(zip(starts,ends)):
            theta=checkpoints[p].copy()
            status=LIB.replay_segment(n,dp(xx),dp(yy),ip(order),begin,end,dp(tr),dp(theta),dp(stats));assert status==end-begin
            expected_theta=checkpoints[p+1] if p+1<len(starts) else z['theta_after']
            maxstats=np.maximum(maxstats,stats);maxstate=max(maxstate,float(abs(theta-expected_theta).max()))
            np.testing.assert_allclose(theta,expected_theta,atol=1e-9,rtol=1e-8)
        mid=checkpoints[len(range(0,n,8))] if full else z['theta_after']
        losses=np.array([gradient(xx,yy,v)[0] for v in [z['theta_before'],mid,z['theta_after']]])
        maxloss=max(maxloss,float(np.max(abs(losses-z['losses'])/(1+abs(z['losses'])))))
        np.testing.assert_allclose(losses,z['losses'],atol=2e-8,rtol=2e-7)
        st=states.loc[u.Date,[f'After_{c}' for c in NAMES]].to_numpy(dtype=float)
        np.testing.assert_allclose(st,z['theta_after'],atol=1e-14)
        previous=z['theta_after'];stockrows+=n;fullrows+=full
        # Full grid independently replayed in NumPy at first/middle/last stocks
        # and first/last full-day passes on a deterministic set of 13 dates.
        if counter%100==0 or counter==len(updates)-1:
            for si,stepidx in [(0,0),(1,n//2),(2,n-1)]+([(3,n),(2+full,n+full-1)] if full else []):
                ids=order[stepidx:stepidx+1] if stepidx<n else np.arange(n)
                for block in range(2):
                    before=z['snapshots'][si,block,:27];sg=z['snapshots'][si,block,27:]
                    _,ng=gradient(xx[ids],yy[ids],before)
                    if block==0:ng[24:]=0
                    else:ng[:24]=0
                    np.testing.assert_allclose(ng,sg,atol=1e-9,rtol=1e-9)
                    nt,eta,count=step(xx[ids],yy[ids],before,block)
                    np.testing.assert_allclose(eta,tr[stepidx,block,4],atol=1e-14)
                    assert count==tr[stepidx,block,7]
                    sampled+=1
            if counter in [100,500,1000]:
                before=z['snapshots'][0,0,:27];ids=order[:1];_,g=gradient(xx[ids],yy[ids],before)
                for j in range(27):
                    a=before.copy();b=before.copy();a[j]+=1e-6;b[j]-=1e-6
                    fd=(gradient(xx[ids],yy[ids],a)[0]-gradient(xx[ids],yy[ids],b)[0])/2e-6
                    fdmax=max(fdmax,abs(fd-g[j])/(1+abs(g[j])))
        if (counter+1)%300==0:print(variant,'replayed days',counter+1,'seconds',round(time.monotonic()-start,1),flush=True)
    assert np.max(maxstats)<2e-6 and fdmax<1e-6
    # Reproduce EVERY saved prediction and integer rank from the daily state.
    ranks=pd.concat([pd.read_csv(p,parse_dates=['Date']) for p in sorted(out.glob('ranks_*.csv.gz'))],ignore_index=True)
    recorded={date:g for date,g in ranks.groupby('Date')};maxpred=0.;rankrows=0
    for date,idx in groups.items():
        if date==pd.Timestamp('2020-10-01') or date>calendar.Date.max():continue
        theta=states.loc[date,[f'After_{c}' for c in NAMES]].to_numpy(dtype=float)
        score=x[idx]@coefficients(theta);code=f.loc[idx,'SecuritiesCode'].to_numpy();order=np.lexsort((code,-score))
        got=recorded[date].sort_values('Rank')
        np.testing.assert_array_equal(got.SecuritiesCode,code[order]);np.testing.assert_array_equal(got.Rank,np.arange(len(idx)))
        np.testing.assert_allclose(got.g,score[order],atol=2e-10,rtol=1e-10)
        maxpred=max(maxpred,float(abs(got.g.to_numpy()-score[order]).max()));rankrows+=len(idx)
    labels=f[['SignalDate','SecuritiesCode','Target']].rename(columns={'SignalDate':'Date'})
    valid=ranks.loc[ranks.ValidationYear.ne(0)].merge(labels,on=['Date','SecuritiesCode'],validate='one_to_one')
    daily=pd.read_csv(out/'daily_spread_returns.csv',parse_dates=['Date']).set_index('Date');spreads=[];mses=[];maxspread=0.
    weights=np.linspace(2,1,200)
    for date,g in valid.groupby('Date'):
        g=g.sort_values('Rank');long=g.head(200).Target.to_numpy();short=g.tail(200).Target.to_numpy()[::-1]
        assert np.isfinite(long).all() and np.isfinite(short).all()
        s=(long-short)@weights/weights.mean();m=float(((g.g-g.Target)**2).mean())
        maxspread=max(maxspread,abs(s-daily.loc[date,'OfficialDailySpread']));spreads.append(s);mses.append(m)
        np.testing.assert_allclose(m,daily.loc[date,'AllStockForecastMSE'],atol=1e-9,rtol=1e-10)
    sharpe=np.mean(spreads)/np.std(spreads,ddof=1)
    np.testing.assert_allclose(sharpe,results['validation']['official_style_unannualized_sharpe'],atol=1e-13)
    np.testing.assert_allclose(np.mean(mses),results['mean_daily_forecast_mse'],atol=1e-10)
    result={'passed':True,'days_replayed':len(updates),'stock_updates_replayed':stockrows,'full_updates_replayed':fullrows,
        'replay_method':'independent Jacobian replay, anchored every 8 stocks and each full pass; checkpoints regenerate bitwise-identical full traces',
        'gradient_substeps_replayed':2*(stockrows+fullrows),'max_relative_errors_loss_before_after_gradinf_norm_descent':maxstats.tolist(),
        'max_daily_final_parameter_error':maxstate,'max_relative_daily_stage_loss_error':maxloss,
        'independent_numpy_full_grid_substeps':sampled,'finite_difference_relative_error':fdmax,
        'forecast_rows_reproduced':rankrows,'all_integer_ranks_equal':True,'max_prediction_error':maxpred,
        'max_spread_error':maxspread,'validation_sharpe_reproduced':float(sharpe),'seconds':time.monotonic()-start}
    (out/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    return result

if __name__=='__main__':
    data=load()
    for v in sys.argv[1:] or VARIANTS:audit(v,*data)
