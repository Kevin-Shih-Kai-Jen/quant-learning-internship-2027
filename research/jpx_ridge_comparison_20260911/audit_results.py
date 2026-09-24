from pathlib import Path
import json,gc
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent

def main():
    result=json.loads((ROOT/'results.json').read_text());assert result['complete'] and len(result['variants'])==10
    audits=[]
    for family,path,elig in [('level_t',ROOT.parent/'jpx_stock_returns_20260910/features.pkl','Eligible'),('return_t',ROOT.parent/'jpx_return_t_20260911/return_t_features.pkl','TEligible')]:
        f=pd.read_pickle(path);fits=json.loads((ROOT/family/'model_fits.json').read_text())
        tunes=json.loads((ROOT/family/'lambda_selection.json').read_text())
        selected=pd.read_csv(ROOT/family/'ranked_orders_50_50.csv',parse_dates=['SignalDate'])
        for fit,tune in zip(fits,tunes):
            year=fit['validation_year'];cols=list(fit['coefficients'])[1:];cutoff=pd.Timestamp(fit['fit_asof'])
            tr=f.loc[f.SignalDate.dt.year.eq(year-1)&f.ExitDate.le(cutoff)&f.Target.notna()&f[elig]]
            assert len(tr)==fit['train_rows'] and tr.ExitDate.max()<=cutoff
            totals={a:0. for a in result['lambda_grid']};n=0;max_score_error=0.
            for fold in tune['folds']:
                start=pd.Timestamp(fold['validation_first_signal']);end=pd.Timestamp(fold['validation_last_signal'])
                a=tr.loc[tr.SignalDate.lt(start)&tr.ExitDate.le(start)];v=tr.loc[tr.SignalDate.between(start,end)]
                assert len(a)==fold['train_rows'] and len(v)==fold['validation_rows']
                assert a.ExitDate.max()<=start and v.ExitDate.max()<=cutoff
                # Independent SVD of the centered design, avoiding the production normal-equation solve.
                x=a[cols].to_numpy();y=a.Target.to_numpy();xm=x.mean(axis=0);ym=y.mean()
                u,s,vt=np.linalg.svd((x-xm)/np.sqrt(len(x)),full_matrices=False)
                rhs=u.T@((y-ym)/np.sqrt(len(x)))
                for score in fold['scores']:
                    lam=score['lambda'];beta=vt.T@((s/(s*s+lam))*rhs)
                    pred=ym+(v[cols].to_numpy()-xm)@beta
                    sse=float(np.sum((pred-v.Target.to_numpy())**2));totals[lam]+=sse
                    err=abs(sse/len(v)-score['mse']);max_score_error=max(max_score_error,err)
                    assert err<1e-13
                n+=len(v)
            chosen=min(totals,key=lambda a:(totals[a]/n,a));assert chosen==fit['lambda']==tune['chosen_lambda']
            sel=selected.loc[selected.ValidationYear.eq(year)]
            joined=sel.merge(f[['SignalDate','SecuritiesCode']+cols],on=['SignalDate','SecuritiesCode'],validate='one_to_one')
            theta=np.array(list(fit['coefficients'].values()))
            np.testing.assert_allclose(joined.Prediction,theta[0]+joined[cols].to_numpy()@theta[1:],atol=1e-12,rtol=0)
            audits.append({'family':family,'year':year,'selected_rows':len(sel),'lambda':chosen,'independent_svd_max_inner_mse_error':max_score_error})
        del f,tr,x,u;gc.collect()
    ports=[]
    sensitivity=json.loads((ROOT/'return_t_calendar_sensitivity.json').read_text())
    extra=[{'key':'return_t_calendar_corrected_'+label,'ridge':sensitivity[label],'close_only':False} for label in ['ols','ridge']]
    for item in result['variants']+extra:
        out=ROOT/item['key'];d=pd.read_csv(out/'daily_returns.csv');t=pd.read_csv(out/'selected_trades.csv');o=pd.read_csv(out/'requested_orders.csv');e=pd.read_csv(out/'position_events.csv')
        assert len(d)==953 and not o.duplicated(['SignalDate','SecuritiesCode']).any()
        assert (o.Weight*o.Prediction>0).all()
        assert o.groupby('SignalDate').Weight.apply(lambda x:x.abs().sum()).le(1+1e-12).all()
        executed=t.loc[t.Executed].copy()
        scale=executed.EntryClose*executed.ExitCumulativeFactor/executed.EntryCumulativeFactor
        target=scale*(1+executed.Prediction)
        touch=target.between(executed.ExitLow,executed.ExitHigh)&(not item['close_only'])
        fill=np.where(touch,target,executed.ExitClose)
        np.testing.assert_allclose(executed.ExitPrice,fill,atol=1e-8,rtol=1e-12)
        pnl=np.sign(executed.RequestedWeight)*executed.EntryNotional*(fill/scale-1)
        np.testing.assert_allclose(executed.RealizedPnl,pnl,atol=1e-10,rtol=1e-10)
        np.testing.assert_allclose(d.Pnl,e.groupby('IntervalEntryDate').Pnl.sum().reindex(d.EntryDate).fillna(0),atol=1e-10,rtol=0)
        wealth=(1+d.StrategyReturn).cumprod();peak=np.maximum.accumulate(np.r_[1.,wealth])[1:]
        dd=float((wealth/peak-1).min());ret=float(wealth.iloc[-1]-1)
        assert abs(dd-item['ridge']['max_drawdown'])<1e-12 and abs(ret-item['ridge']['cumulative_return'])<1e-12
        if 'ols' in item:assert item['drawdown_lower']==(abs(dd)<abs(item['ols']['max_drawdown']))
        if item['key'].startswith('return_t_calendar_corrected_'):
            mapped=o.merge(d[['SignalDate','EntryDate','ExitDate']],on='SignalDate',suffixes=('_order','_calendar'),validate='many_to_one')
            assert mapped.EntryDate_order.eq(mapped.EntryDate_calendar).all() and mapped.ExitDate_order.eq(mapped.ExitDate_calendar).all()
        assert executed.ExitDate.max()<=d.ExitDate.max()
        ports.append({'key':item['key'],'intervals':len(d),'executed_trades':len(executed),'close_only':item['close_only'],'drawdown_recomputed':dd})
    report={'all_checks_passed':True,'independent_model_checks':audits,'independent_execution_checks':ports,'scope':'Independent SVD inner lambda scores, selected predictions, raw fills, PnL and compounded drawdown; 10 paired baseline reproductions in main runner.'}
    (ROOT/'independent_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps(report),flush=True)

if __name__=='__main__':main()
