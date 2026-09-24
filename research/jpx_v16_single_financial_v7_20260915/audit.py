from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from run import ROOT,BASE,V7,FINS,NAMES,load,save,W
sys.path.insert(0,str(BASE/'jpx_official_ranking_20260912'))
from official_metric import calc_spread_return_sharpe

def main():
    f,x,z,groups,calendar=load();codes=f.SecuritiesCode.to_numpy();target=f.Target.to_numpy();yearmap=calendar.set_index('Date').ValidationYear.to_dict()
    idxmap=f.set_index(['SignalDate','SecuritiesCode']).index;old=np.load(ROOT/'price_only/predictions.npz');max_old_score=0.;nold=0
    for path in sorted((V7/'v7_equal').glob('ranks_*.csv.gz')):
        oldr=pd.read_csv(path,parse_dates=['Date']);idx=idxmap.get_indexer(pd.MultiIndex.from_arrays([oldr.Date,oldr.SecuritiesCode]));assert (idx>=0).all()
        np.testing.assert_array_equal(old['rank'][idx],oldr.Rank.to_numpy())
        np.testing.assert_allclose(old['score'][idx],oldr.g.to_numpy(),atol=1e-12,rtol=0)
        max_old_score=max(max_old_score,float(np.abs(old['score'][idx]-oldr.g.to_numpy()).max()));nold+=len(idx)
    assert nold==2326022
    od=pd.read_csv(V7/'v7_equal/daily_spread_returns.csv');nd=pd.read_csv(ROOT/'price_only/daily_metrics.csv')
    assert od.Date.tolist()==nd.Date.tolist();np.testing.assert_allclose(od.OfficialDailySpread,nd.OfficialDailySpread,atol=1e-12,rtol=0)
    save(ROOT/'baseline_reproduction.json',{'passed':True,'all_prediction_and_rank_rows':nold,'max_saved_score_difference':max_old_score,'all_953_daily_spreads_match':True,'sharpe':float(nd.OfficialDailySpread.mean()/nd.OfficialDailySpread.std(ddof=1))})
    del old
    totals=[]
    for out in [ROOT/'price_only']+[ROOT/f'{s}_{j:02d}' for j in range(1,16) for s in ['feature','control']]:
        rr=json.loads((out/'results.json').read_text());feature=rr['feature'];add=rr['includes_feature']
        xx=np.column_stack([x,z[feature].to_numpy()]) if add else x;names=NAMES+([feature] if add else [])
        pred=np.load(out/'predictions.npz');score=pred['score'];rank=pred['rank'];hist=pd.read_csv(out/'parameter_history.csv').set_index('Date');daily=pd.read_csv(out/'daily_metrics.csv').set_index('Date')
        maxerr=0.;n=0;official=[];selected_missing=0
        for date,ix in groups.items():
            if date>calendar.Date.max() or date==pd.Timestamp('2020-10-01'):
                assert (rank[ix]==-1).all() and np.isnan(score[ix]).all();continue
            theta=hist.loc[str(date.date()),[f'After_{v}' for v in names]].to_numpy(float)
            replay=theta[0]+np.einsum('ij,j->i',xx[ix,1:],theta[1:]);np.testing.assert_allclose(replay,score[ix],atol=1e-12,rtol=1e-10)
            n+=len(ix);maxerr=max(maxerr,float(np.abs(replay-score[ix]).max()))
            oo=np.lexsort((codes[ix],-score[ix]));expected=np.empty(len(ix),np.int32);expected[oo]=np.arange(len(ix));np.testing.assert_array_equal(expected,rank[ix])
            if date not in yearmap:continue
            ii=ix[oo];missing=int((~np.isfinite(target[np.r_[ii[:200],ii[-200:]]])).sum());selected_missing+=missing
            row=daily.loc[str(date.date())];assert missing==row.SelectedMissingTargets
            if missing:assert pd.isna(row.OfficialDailySpread);continue
            # Independent matrix form of official function, then full original metric below.
            side=(target[ii[:200]]*W).sum()-(target[ii[-200:][::-1]]*W).sum()
            np.testing.assert_allclose(side/W.mean(),row.OfficialDailySpread,atol=2e-12,rtol=1e-10)
            official.append(pd.DataFrame({'Date':str(date.date()),'Rank':rank[ix],'Target':target[ix]}))
        exact=float(calc_spread_return_sharpe(pd.concat(official,ignore_index=True)))
        np.testing.assert_allclose(exact,rr['sharpe'],atol=2e-13,rtol=0)
        updates=pd.read_csv(out/'training_updates.csv');assert len(updates)==1199 and (updates.MSEAfterStep<=updates.MSEBeforeStep+1e-12).all()
        assert pd.to_datetime(updates.ExitDate).le(pd.to_datetime(updates.Date)).all()
        assert pd.to_datetime(updates.SignalDate).lt(pd.to_datetime(updates.ExitDate)).all()
        if feature:
            other=ROOT/(out.name.replace('feature_','control_') if add else out.name.replace('control_','feature_'))
            ctrl=pd.read_csv(other/'training_updates.csv')
            for col in ['SignalDate','Date','ExitDate','KnownLabelStocks','TrainingStocks','ThresholdSkippedStocks']:assert updates[col].equals(ctrl[col]),col
            expectedcounts=[]
            for row in updates.itertuples():
                ix=groups[pd.Timestamp(row.SignalDate)];known=np.isfinite(target[ix]);ok=z[feature].to_numpy()[ix]
                expectedcounts.append(int((known&(np.abs(ok)>100)).sum()))
            np.testing.assert_array_equal(expectedcounts,updates.ThresholdSkippedStocks)
        audit={'passed':True,'all_forecast_rows':n,'all_integer_ranks_match':True,'max_independent_score_error':maxerr,'official_sharpe_recalculated':exact,'selected_missing_targets':selected_missing,'all_1199_update_numerics_checked_in_update_audit':True,'matched_control_schedule_and_mask_counts_checked':bool(feature)}
        save(out/'audit.json',audit);totals.append({'name':out.name,**audit});print('AUDITED',out.name,exact,flush=True)
    save(ROOT/'audit.json',{'passed':True,'models':totals,'baseline_reproduced':True})
if __name__=='__main__':main()
