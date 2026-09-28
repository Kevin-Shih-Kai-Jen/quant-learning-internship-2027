"""Evaluate A/B/C/D on common full-universe dates and save compact evidence."""
import io,json,zlib,sqlite3,warnings
from pathlib import Path
import numpy as np
import pandas as pd
from run_arima_joint import ROOT,RUN,SOURCE,PS,YEARS,sha,save,evaluate_params
from evaluate_arima_acf import daily_metrics
from evaluate_arima_grid import stats

def main():
    manifest=json.loads((RUN/'manifest.json').read_text());assert manifest['runner_sha256']==sha(ROOT/'run_arima_joint.py') and manifest['plan_sha256']==sha(ROOT/'arima_joint_plan.md')
    con=sqlite3.connect(f'file:{RUN}/fits.sqlite?mode=ro',uri=True);assert con.execute('select count(*) from fits').fetchone()[0]==16000
    sourcecon=sqlite3.connect(f'file:{SOURCE}/fits.sqlite?mode=ro',uri=True)
    with np.load(SOURCE/'prices.npz') as z:data={k:z[k] for k in z.files}
    labels=pd.read_pickle(SOURCE/'labels.pkl').sort_values(['Date','SecuritiesCode']).reset_index(drop=True);cal=pd.read_csv(SOURCE/'calendar.csv',parse_dates=['Date'])
    dx=pd.Index(cal.Date).get_indexer(labels.Date);cx=pd.Index(data['codes']).get_indexer(labels.SecuritiesCode);assert dx.min()>=0 and cx.min()>=0
    metrics_daily={};predictions={};training=[];validation_audit=[];status_summary=[]
    for p in PS:
        for group in ['A','B']:
            path=SOURCE/'models'/f'p{p}_q1'/'predictions.npz' if group=='A' else ROOT/'arima_acf'/f'p{p}_predictions.npz'
            with np.load(path) as z:score=z['score'];fallback=z['fallback'];rank=z['rank']
            f=labels.copy();f['Score']=score;f['Fallback']=fallback;d,ranked=daily_metrics(f);np.testing.assert_array_equal(ranked.Rank,rank)
            ref=SOURCE/'models'/f'p{p}_q1'/'daily.csv' if group=='A' else ROOT/'arima_acf'/f'p{p}_acf_daily.csv'
            np.testing.assert_allclose(d.OfficialDailySpread,pd.read_csv(ref).OfficialDailySpread,rtol=1e-12,atol=1e-12,equal_nan=True)
            metrics_daily[(p,group)]=d;predictions[(p,group)]=score
        mats={k:np.zeros((len(cal),len(data['codes']))) for k in ['C','D']};fallbacks={k:np.ones(mats[k].shape,bool) for k in mats}
        for yi,ci,packed in con.execute('select yi,ci,audit from fits where p=? order by yi,ci',(p,)):
            a=json.loads(zlib.decompress(packed));oldraw,payload=sourcecon.execute('select audit,payload from fits where oi=? and yi=? and ci=?',((p-1)*5,yi,ci)).fetchone();old=json.loads(oldraw);base=np.load(io.BytesIO(payload))
            mask=cal.ValidationYear.eq(YEARS[yi]).to_numpy();pos=data['valid_positions'][data['years']==YEARS[yi]];cut=int(pos[0]);end=int(pos[-1])+1
            assert a['max_target_exit']<a['first_validation_signal'] and a['max_target_exit']==str(data['dates'][cut-1])
            assert a['max_target_origin']==str(data['dates'][cut-3])
            for group in ['C','D']:
                v=a[group];scores=base['score'].copy();fb=base['fallback'].copy()
                record={'p':p,'Year':YEARS[yi],'SecuritiesCode':a['code'],'Group':group,'training_targets':a['training_targets'],**v}
                for field in ['params','lambda','rho']:record.pop(field,None)
                if v['status']=='optimized':
                    assert v['best_mse']<=v['initial_mse']+1e-15
                    y=(data['prices'][:end,ci]-old['anchor'])/old['scale'];params=np.asarray(v['params']);lam=np.asarray(v['lambda']);rho=np.asarray(v['rho'])
                    scores,fb,fc,res=evaluate_params(y,p,params,pos,old['anchor'],old['scale'],lam,rho)
                    assert np.isfinite(scores).all() and (np.abs(rho)<=1+1e-12).all() and ((lam>=0)&(lam<=1)).all()
                    record.update(lambda1=float(lam[0]),lambda2=float(lam[1]),rho1=float(rho[0]),rho2=float(rho[1]))
                    record.update({f'parameter_{i}':float(x) for i,x in enumerate(params)})
                    validation_audit.append({'p':p,'Year':YEARS[yi],'code':a['code'],'Group':group,'fallback_days':int(fb.sum()),'changed_fallback_days':int((fb!=base['fallback']).sum())})
                mats[group][mask,ci]=scores;fallbacks[group][mask,ci]=fb;training.append(record)
        for group in ['C','D']:
            f=labels.copy();f['Score']=mats[group][dx,cx];f['Fallback']=fallbacks[group][dx,cx];d,ranked=daily_metrics(f);metrics_daily[(p,group)]=d;predictions[(p,group)]=f.Score.to_numpy()
            np.savez_compressed(RUN/f'p{p}_{group}_predictions.npz',score=ranked.Score.to_numpy(),rank=ranked.Rank.to_numpy(dtype=np.int16),fallback=ranked.Fallback.to_numpy())
            status_summary.append({'p':p,'Group':group,'FallbackFraction':float(f.Fallback.mean()),'SelectedFallbackStocks':int(d.SelectedFallbackStocks.sum())})
            print('EVALUATED',p,group,flush=True)
    v7=pd.read_csv(SOURCE/'baseline_daily.csv',parse_dates=['Date']);metrics_daily[(0,'v7')]=v7
    common=np.logical_and.reduce([d.OfficialDailySpread.notna().to_numpy() for d in metrics_daily.values()]);rows=[];annual=[]
    target=labels.Target.to_numpy();key_common=common[dx]
    for (p,group),d in metrics_daily.items():
        d.to_csv(RUN/f'p{p}_{group}_daily.csv',index=False)
        mse=None
        if p:
            pred=predictions[(p,group)];known=np.isfinite(target)&key_common;mse=float(np.mean((target[known]-pred[known])**2))
        rows.append({'p':p,'Group':group,**stats(d.loc[common]),'TargetMSE':mse,'RawScoredDays':int(d.OfficialDailySpread.notna().sum())})
        for year in YEARS:
            mask=common&cal.ValidationYear.eq(year).to_numpy();mse_y=None
            if p:
                known=np.isfinite(target)&mask[dx];mse_y=float(np.mean((target[known]-pred[known])**2))
            annual.append({'p':p,'Group':group,'Year':year,**stats(d.loc[mask]),'TargetMSE':mse_y})
    pd.DataFrame(rows).to_csv(RUN/'metrics.csv',index=False);pd.DataFrame(annual).to_csv(RUN/'annual.csv',index=False)
    tr=pd.DataFrame(training);tr.to_csv(RUN/'training_audit.csv.gz',index=False,compression='gzip');pd.DataFrame(validation_audit).to_csv(RUN/'validation_audit.csv.gz',index=False,compression='gzip')
    training_summary=[]
    for (p,group),g in tr.groupby(['p','Group']):
        trained=g.loc[g.status.eq('optimized')];weight=trained.training_targets.to_numpy();initial=float(np.average(trained.initial_mse,weights=weight));final=float(np.average(trained.best_mse,weights=weight))
        training_summary.append({'p':int(p),'Group':group,'records':len(g),'optimized':len(trained),'converged':int(trained.converged.eq(True).sum()),'statuses':g.status.value_counts().to_dict(),'initial_mse_weighted':initial,'final_mse_weighted':final,'training_mse_relative_change':final/initial-1,'median_function_calls':float(trained.function_calls.median()),'median_lambda1':float(trained.lambda1.median()),'median_lambda2':float(trained.lambda2.median()),'lambda1_at_zero':int(trained.lambda1.le(1e-6).sum()),'lambda2_at_zero':int(trained.lambda2.le(1e-6).sum()),'lambda1_at_one':int(trained.lambda1.ge(1-1e-6).sum()),'lambda2_at_one':int(trained.lambda2.ge(1-1e-6).sum()),'causal_checks':int(trained.causal_checks.sum()),'median_parameter_change':float(trained.max_parameter_change.median())})
    pairs=[(p,new,old) for p in PS for new,old in [('C','A'),('D','B'),('D','C')]]
    keys=[(p,g) for p in PS for g in ['A','B','C','D']];values=np.column_stack([metrics_daily[k].loc[common,'OfficialDailySpread'].to_numpy() for k in keys]);index={k:i for i,k in enumerate(keys)}
    obs=values.mean(axis=0)/values.std(axis=0,ddof=1);diff=np.array([obs[index[(p,new)]]-obs[index[(p,old)]] for p,new,old in pairs]);rng=np.random.default_rng(20260924);n=len(values);boot=[]
    for _ in range(2000):
        starts=rng.integers(0,n,size=int(np.ceil(n/20)));ix=((starts[:,None]+np.arange(20))%n).ravel()[:n];sample=values[ix];sh=sample.mean(axis=0)/sample.std(axis=0,ddof=1);boot.append([sh[index[(p,new)]]-sh[index[(p,old)]] for p,new,old in pairs])
    boot=np.asarray(boot);lo,hi=np.quantile(boot,[.025,.975],axis=0);radius=float(np.quantile(np.max(np.abs(boot-diff),axis=1),.95))
    contrasts=[{'p':p,'Contrast':f'{new}-{old}','SharpeDifference':float(diff[i]),'Marginal95Low':float(lo[i]),'Marginal95High':float(hi[i]),'Simultaneous95Low':float(diff[i]-radius),'Simultaneous95High':float(diff[i]+radius)} for i,(p,new,old) in enumerate(pairs)]
    pd.DataFrame(contrasts).to_csv(RUN/'contrasts.csv',index=False)
    pivot=tr.loc[tr.status.eq('optimized')].pivot(index=['p','Year','SecuritiesCode'],columns='Group',values='best_mse').dropna();local_d_worse=int((pivot.D>pivot.C+1e-12).sum())
    result={'complete':True,'stock_year_orders':16000,'optimizer_jobs':32000,'common_days':int(common.sum()),'excluded_dates':cal.loc[~common,'Date'].dt.strftime('%Y-%m-%d').tolist(),'metrics':rows,'annual':annual,'training_summary':training_summary,'prediction_summary':status_summary,'contrasts':contrasts,'D_training_loss_greater_than_C_count':local_d_worse,'paired_optimized_records':len(pivot),'all_official_metric_checks_passed':True,'formal_test_used':False,'annualized':False,'costs_included':False,'causal_checks':sum(x['causal_checks'] for x in training_summary),'bootstrap':{'replicates':2000,'block_days':20,'seed':20260924,'simultaneous_radius':radius,'scope':'6 fixed contrasts only, not past adaptive research'},'runner_sha256':sha(ROOT/'run_arima_joint.py'),'evaluator_sha256':sha(__file__),'plan_sha256':sha(ROOT/'arima_joint_plan.md')}
    save(RUN/'results.json',result);print(pd.DataFrame(rows).to_string(index=False),flush=True)
if __name__=='__main__':main()
