"""Fixed-market random-PC controls. Generate every state/rank before reading Target."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import platform
import time
import zipfile
import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'jpx_pca_score_ab_20260928' / 'run_ab.py'
spec = importlib.util.spec_from_file_location('original_ab', SOURCE)
ab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ab)
SEED, PATHS = 20260929, 1000
NAMES = ['B_RANDOM', 'B_RANDOM_NORM']


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def matrix_rank(scores):
    # Codes are ascending, hence stable sort exactly implements code tie-break.
    order = np.argsort(-scores, axis=1, kind='stable')
    ranks = np.empty(order.shape, dtype=np.uint16)
    np.put_along_axis(ranks, order, np.arange(scores.shape[1], dtype=np.uint16)[None], axis=1)
    return ranks


def score_matrices(g, keep, v, slopes, dmax, kmax):
    q = v - v.mean(axis=0)
    den = np.sum(q*q, axis=0)
    assert np.all(den > 1e-24), 'Degenerate demeaned PC: stop rather than alter candidate set'
    delta = q * slopes
    delta_norm = q * (np.where(slopes < 0, -1., 1.) * dmax / np.sqrt(den))
    delta_norm[:, kmax] = delta[:, kmax]
    score = np.broadcast_to(g, (2, len(slopes), len(g))).copy()
    score[0][:, keep] -= delta.T
    score[1][:, keep] -= delta_norm.T
    return score, delta, delta_norm


def generate(args):
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    (out / 'states').mkdir()
    ranks_path = args.input / 'research/jpx_v7_daily_mse_20260912/v7_equal/ranks_2018.csv.gz'
    assert ab.sha(ranks_path) == ab.RANK_SHA[2018]
    frame = pd.read_csv(ranks_path, usecols=['Date', 'SecuritiesCode', 'g', 'Rank', 'ValidationYear'],
                        parse_dates=['Date'], float_precision='round_trip')
    assert frame.Date.nunique() == 245 and frame.ValidationYear.eq(2018).all()
    with zipfile.ZipFile(args.zip) as z:
        member = ab.raw_member(z)
        with z.open(member) as f:
            raw_sha = hashlib.file_digest(f, 'sha256').hexdigest()
        assert raw_sha == ab.RAW_SHA
        with z.open(member) as f:
            price = pd.read_csv(f, usecols=['Date', 'SecuritiesCode', 'Close', 'Volume',
                                           'AdjustmentFactor', 'SupervisionFlag'], parse_dates=['Date'])
    price = price.loc[price.Date.le(frame.Date.max())].copy()
    panel, closures = ab.build_returns(price)
    groups = {d: np.sort(g.SecuritiesCode.to_numpy()) for d, g in price.loc[~price.SupervisionFlag].groupby('Date')}
    rng = np.random.default_rng(SEED)
    windows, hashes = [], []
    start = time.time()
    max_norm_error, max_score_error = 0., 0.
    for day, (date, group) in enumerate(frame.groupby('Date', sort=True)):
        group = group.sort_values('SecuritiesCode')
        codes, g = group.SecuritiesCode.to_numpy(), group.g.to_numpy()
        np.testing.assert_array_equal(codes, groups[date])
        arank, aw = ab.rank_weights(g, codes)
        np.testing.assert_array_equal(arank, group.Rank.to_numpy())
        keep, x, v, eigenvalues, info = ab.pca_asof(panel, date, codes)
        kmax = int(np.argmax(eigenvalues * (aw[keep] @ v)**2))
        q = v - v.mean(axis=0)
        den = np.sum(q*q, axis=0)
        assert np.all(den > 1e-24)
        slopes = (q.T @ (g[keep] - g[keep].mean())) / den
        bmax_score, bmax_slope, _ = ab.ablate(g[keep], v[:, kmax])
        slopes[kmax] = bmax_slope
        dmax = float(np.linalg.norm(q[:, kmax] * bmax_slope))
        scores, delta, delta_norm = score_matrices(g, keep, v, slopes, dmax, kmax)
        np.testing.assert_allclose(scores[0, kmax, keep], bmax_score, atol=1e-14, rtol=1e-12)
        rank = np.stack([matrix_rank(scores[i]) for i in range(2)])
        direct_bmax = g.copy()
        direct_bmax[keep] = bmax_score
        brank, _ = ab.rank_weights(direct_bmax, codes)
        np.testing.assert_array_equal(rank[0, kmax], brank)
        np.testing.assert_array_equal(rank[1, kmax], brank)
        normerr = float(np.max(np.abs(np.linalg.norm(delta_norm, axis=0) - dmax)))
        max_norm_error = max(max_norm_error, normerr)
        assert normerr < 1e-12
        if day in [0, 122, 244]:
            for k in sorted(set([0, len(slopes)-1, kmax])):
                independent, _, _ = ab.ablate(g[keep], v[:, k])
                max_score_error = max(max_score_error, float(np.max(np.abs(scores[0, k, keep] - independent))))
                np.testing.assert_allclose(scores[0, k, keep], independent, atol=1e-14, rtol=1e-12)
                full = g.copy(); full[keep] = independent
                rr, _ = ab.rank_weights(full, codes)
                np.testing.assert_array_equal(rank[0, k], rr)
        choices = rng.integers(0, len(slopes), size=PATHS, dtype=np.int32)
        path = out / 'states' / f'{day:03d}_{date.date()}.npz'
        ab.atomic_npz(path, date=np.array(str(date.date())), codes=codes.astype(np.int32),
                      g=g, keep=keep, v=v, eigenvalues=eigenvalues, slopes=slopes,
                      dmax=np.array(dmax), kmax=np.array(kmax), ranks=rank,
                      baseline_rank=arank, choices=choices)
        hashes.append({'path':str(path.relative_to(out)), 'size':path.stat().st_size, 'sha256':ab.sha(path)})
        info.update(Date=str(date.date()), day=day, universe=len(g), max_PC=kmax+1, dmax=dmax)
        windows.append(info)
        if day % 20 == 0 or day == 244:
            print(json.dumps({'stage':'generate','day':day+1,'total':245,'elapsed_s':round(time.time()-start,1)}),flush=True)
    pd.DataFrame(windows).to_csv(out / 'windows.csv', index=False)
    audit = {'stage':'generation_complete', 'days':245, 'paths_per_control':PATHS, 'seed':SEED,
             'rng':'numpy.random.default_rng / PCG64', 'plan_sha256':ab.sha(ROOT/'experiment_plan.md'),
             'source_code_sha256':ab.sha(SOURCE), 'experiment_code_sha256':ab.sha(Path(__file__)),
             'ranks_sha256':ab.sha(ranks_path), 'raw_member_sha256':raw_sha,
             'zip_sha256':ab.sha(args.zip), 'raw_member':member,
             'first_date':str(frame.Date.min().date()), 'last_date':str(frame.Date.max().date()),
             'Target_read_in_generation':False, 'test_used':False,
             'original_A_ranks_reproduced':True,'max_norm_matching_error':max_norm_error,
             'independent_ablation_max_error':max_score_error, 'market_closures':closures,
             'states':hashes,'environment':{'python':platform.python_version(), 'numpy':np.__version__,
                                          'pandas':pd.__version__, 'scipy':scipy.__version__}}
    save(out/'generation_audit.json', audit)


def metrics(ranks, scores, target, arank):
    K, N = ranks.shape
    order = np.argsort(ranks, axis=1)
    long, short = order[:, :200], order[:, -200:][:, ::-1]
    missing = (~np.isfinite(target[long])).sum(axis=1) + (~np.isfinite(target[short])).sum(axis=1)
    spread = (target[long] @ ab.W - target[short] @ ab.W) / ab.W.mean()
    spread[missing > 0] = np.nan
    valid = np.isfinite(target)
    yr = rankdata(target[valid]); yc = yr - yr.mean()
    xr = rankdata(scores[:, valid], axis=1); xc = xr - xr.mean(axis=1, keepdims=True)
    denom = np.sqrt(np.sum(xc*xc, axis=1) * (yc @ yc))
    ic = np.divide(xc @ yc, denom, out=np.full(K,np.nan), where=denom>0)
    w = np.zeros((K,N)); aw = np.zeros(N)
    np.put_along_axis(w, long, np.broadcast_to(.5*ab.W/ab.W.sum(),long.shape),axis=1)
    np.put_along_axis(w, short,np.broadcast_to(-.5*ab.W/ab.W.sum(),short.shape),axis=1)
    aorder=np.argsort(arank)
    aw[aorder[:200]]=.5*ab.W/ab.W.sum();aw[aorder[-200:][::-1]]=-.5*ab.W/ab.W.sum()
    return {'spread':spread, 'missing':missing, 'RankIC':ic,
            'weight_L1_from_A':np.abs(w-aw).sum(axis=1),
            'replaced_long':np.sum((ranks<200)&(arank>=200),axis=1),
            'replaced_short':np.sum((ranks>=N-200)&(arank<N-200),axis=1)}


def summarize(x):
    return {'days':len(x), 'mean_spread':float(np.mean(x)), 'sd_spread':float(np.std(x,ddof=1)),
            'Sharpe':float(np.mean(x)/np.std(x,ddof=1))}


def evaluate(args):
    out=args.out
    audit=json.loads((out/'generation_audit.json').read_text())
    assert audit['Target_read_in_generation'] is False and len(audit['states'])==245
    with zipfile.ZipFile(args.zip) as z:
        with z.open(ab.raw_member(z)) as f:
            labels=pd.read_csv(f,usecols=['Date','SecuritiesCode','Target'],parse_dates=['Date'])
    label_groups={d:g.set_index('SecuritiesCode').Target for d,g in labels.groupby('Date')}
    spread=np.empty((2,PATHS,245)); ic=np.empty_like(spread); l1=np.empty_like(spread)
    replacement=np.empty_like(spread); removal=np.empty_like(spread)
    daily=[]; candidates=[]; choice_record=[]; all_arrays=[]
    common=np.ones(245,dtype=bool); start=time.time(); missing_candidates=0
    original=pd.read_csv(ROOT.parent/'jpx_pca_score_ab_20260928/results_v2/daily_scores.csv')
    original=original.loc[original.ValidationYear.eq(2018)].set_index(['Date','variant'])
    max_reference_error=0.; direct_checks=0
    for day, entry in enumerate(audit['states']):
        path=out/entry['path']; assert ab.sha(path)==entry['sha256']
        with np.load(path,allow_pickle=False) as s:
            date=str(s['date']); codes=s['codes']; g=s['g']; keep=s['keep']; v=s['v']
            slopes=s['slopes']; dmax=float(s['dmax']); kmax=int(s['kmax']); ranks=s['ranks']
            arank=s['baseline_rank']; choices=s['choices']
            scores,delta,delta_norm=score_matrices(g,keep,v,slopes,dmax,kmax)
            target=label_groups[pd.Timestamp(date)].reindex(codes).to_numpy()
            a_spread,a_missing=ab.spread_from_ranks(arank,target)
            a_ic=ab.spearman(g,target)
            arms=[metrics(ranks[i],scores[i],target,arank) for i in range(2)]
            b_spread=arms[0]['spread'][kmax]; b_ic=arms[0]['RankIC'][kmax]
            for name,val in [('A',a_spread),('B_MAX',b_spread)]:
                reference=float(original.loc[(date,name),'OfficialDailySpread'])
                max_reference_error=max(max_reference_error,abs(val-reference))
                np.testing.assert_allclose(val,reference,atol=1e-10,rtol=1e-10)
            common[day]=a_missing==0 and np.isfinite(b_spread)
            for arm,m in enumerate(arms):
                missing_candidates+=int(np.sum(m['missing']>0))
                common[day]&=bool(np.all(m['missing'][choices]==0))
                spread[arm,:,day]=m['spread'][choices]
                ic[arm,:,day]=m['RankIC'][choices]
                l1[arm,:,day]=m['weight_L1_from_A'][choices]
                replacement[arm,:,day]=m['replaced_long'][choices]+m['replaced_short'][choices]
                removal[arm,:,day]=np.linalg.norm(delta if arm==0 else delta_norm,axis=0)[choices]
                for k in range(len(slopes)):
                    candidates.append({'Date':date,'control':NAMES[arm],'PC':k+1,
                                       **{name:float(values[k]) for name,values in m.items()},
                                       'removed_score_L2':float(np.linalg.norm((delta if arm==0 else delta_norm)[:,k]))})
                if day in [0,122,244]:
                    for k in sorted(set([0,len(slopes)-1,kmax,int(choices[0])])):
                        independent,missing=ab.spread_from_ranks(ranks[arm,k],target)
                        assert missing==m['missing'][k]
                        np.testing.assert_allclose(independent,m['spread'][k],atol=1e-12,rtol=1e-12)
                        np.testing.assert_allclose(ab.spearman(scores[arm,k],target),m['RankIC'][k],atol=1e-12,rtol=1e-12)
                        direct_checks+=1
            daily.append({'Date':date,'A_spread':a_spread,'B_MAX_spread':b_spread,'A_RankIC':a_ic,
                          'B_MAX_RankIC':b_ic,'B_MAX_weight_L1_from_A':arms[0]['weight_L1_from_A'][kmax],
                          'B_MAX_replaced_total':arms[0]['replaced_long'][kmax]+arms[0]['replaced_short'][kmax],
                          'B_MAX_removed_score_L2':dmax})
            choice_record.append(choices)
        if day%40==0 or day==244:
            print(json.dumps({'stage':'evaluate','day':day+1,'elapsed_s':round(time.time()-start,1)}),flush=True)
    d=pd.DataFrame(daily); d['common']=common
    d.to_csv(out/'daily_baselines.csv',index=False)
    pd.DataFrame(candidates).to_csv(out/'candidate_daily.csv.gz',index=False)
    ab.atomic_npz(out/'path_traces.npz', spread=spread,RankIC=ic,weight_L1_from_A=l1,
                  replaced_total=replacement,removed_score_L2=removal,
                  choices=np.asarray(choice_record),dates=d.Date.to_numpy(dtype='U10'),common=common)
    a=d.loc[common,'A_spread'].to_numpy(); b=d.loc[common,'B_MAX_spread'].to_numpy()
    base={'A':summarize(a),'B_MAX':summarize(b)}
    for name in base:
        base[name]['RankIC']=float(d.loc[common,name+'_RankIC'].mean())
    controls={}; path_tables=[]
    for arm,name in enumerate(NAMES):
        s=spread[arm][:,common]; mu=s.mean(axis=1); sd=s.std(axis=1,ddof=1); sr=mu/sd
        table=pd.DataFrame({'path':np.arange(PATHS),'Sharpe':sr,'delta_Sharpe_vs_A':sr-base['A']['Sharpe'],
                            'mean_spread':mu,'sd_spread':sd,'RankIC':np.nanmean(ic[arm][:,common],axis=1),
                            'mean_weight_L1_from_A':l1[arm][:,common].mean(axis=1),
                            'mean_replaced_total':replacement[arm][:,common].mean(axis=1),
                            'mean_removed_score_L2':removal[arm][:,common].mean(axis=1)})
        table.to_csv(out/(name+'_paths.csv'),index=False)
        path_tables.append(table)
        controls[name]={'paths':PATHS,'Sharpe_quantiles':{str(q):float(np.quantile(sr,q)) for q in [0,.025,.25,.5,.75,.975,1]},
                        'B_MAX_greater_count':int(np.sum(base['B_MAX']['Sharpe']>sr)),
                        'B_MAX_equal_count':int(np.sum(base['B_MAX']['Sharpe']==sr)),
                        'greater_than_A_count':int(np.sum(sr>base['A']['Sharpe'])),
                        'mean_path_mean_spread':float(mu.mean()),'mean_path_sd_spread':float(sd.mean()),
                        'mean_path_RankIC':float(table.RankIC.mean()),
                        'mean_weight_L1_from_A':float(table.mean_weight_L1_from_A.mean()),
                        'mean_replaced_total':float(table.mean_replaced_total.mean()),
                        'mean_removed_score_L2':float(table.mean_removed_score_L2.mean())}
    result={'scope':'Known 2018 validation; fixed-market random direction controls, no fresh holdout',
            'original_days':245,'common_days':int(common.sum()),'excluded_dates':d.loc[~common,'Date'].tolist(),
            'seed':SEED,'paths_per_control':PATHS,'baseline':base,'controls':controls,
            'B_MAX_diagnostics':{k:float(d.loc[common,k].mean()) for k in d.columns if k.startswith('B_MAX_') and k not in ['B_MAX_spread','B_MAX_RankIC']},
            'max_daily_reference_error':max_reference_error,'independent_metric_checks':direct_checks,
            'missing_candidate_days':missing_candidates,'test_used':False,'costs_included':False,
            'intervals_are_confidence_intervals':False,'formal_p_value_computed':False}
    save(out/'results.json',result)
    save(out/'evaluation_audit.json',{'generation_audit_sha256':ab.sha(out/'generation_audit.json'),
                                    'states_verified':len(audit['states']),'max_daily_reference_error':max_reference_error,
                                    'independent_metric_checks':direct_checks,'source_dates_preserved':True,
                                    'test_used':False,'generation_completed_before_Target_read':True})
    make_plots(out,d,spread,common,path_tables,base)
    print(json.dumps(result,ensure_ascii=False),flush=True)


def make_plots(out,d,spread,common,tables,base):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for arm,ax in enumerate(axs):
        ax.hist(tables[arm].Sharpe,bins=35,color=['#4477aa','#66a182'][arm],alpha=.85)
        ax.axvline(base['A']['Sharpe'],color='#555555',linestyle='--',label='A')
        ax.axvline(base['B_MAX']['Sharpe'],color='#bb4422',label='B_MAX')
        ax.set(title=NAMES[arm],xlabel='Full-year official Sharpe (not annualized)',ylabel='Random paths')
        ax.legend()
    fig.suptitle('2018 validation: 1,000 random PC paths per control')
    fig.savefig(out/'sharpe_distribution.png',dpi=150);plt.close(fig)
    dates=pd.to_datetime(d.loc[common,'Date']); a=d.loc[common,'A_spread'].to_numpy()
    fig,ax=plt.subplots(figsize=(10,4.5),constrained_layout=True)
    ax.plot(dates,np.cumsum(d.loc[common,'B_MAX_spread'].to_numpy()-a),color='#bb4422',label='B_MAX minus A')
    for arm,color in enumerate(['#4477aa','#66a182']):
        c=np.cumsum(spread[arm][:,common]-a,axis=1)
        lo,med,hi=np.quantile(c,[.025,.5,.975],axis=0)
        ax.fill_between(dates,lo,hi,color=color,alpha=.15)
        ax.plot(dates,med,color=color,label=NAMES[arm]+' minus A (median)')
    ax.axhline(0,color='#888888',linewidth=.7)
    ax.set(title='Original chronology; bands show random-direction variation',ylabel='Cumulative official spread difference (not account return)')
    ax.legend(fontsize=8)
    fig.savefig(out/'cumulative_spread_difference.png',dpi=150);plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True);p.add_argument('--zip',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--stage',choices=['generate','evaluate','all'],default='all')
    args=p.parse_args()
    with threadpool_limits(limits=2):
        if args.stage in ['generate','all']:generate(args)
        if args.stage in ['evaluate','all']:evaluate(args)


if __name__=='__main__':main()
