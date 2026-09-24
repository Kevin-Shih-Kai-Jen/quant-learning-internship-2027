import json,sys
import numpy as np
import pandas as pd
from run import ROOT,BASE,V7,NAMES,load,save,specs
sys.path.insert(0,str(BASE/'jpx_official_ranking_20260912'))
from official_metric import calc_spread_return_sharpe

def main():
    f,x,z,groups,calendar=load();codes=f.SecuritiesCode.to_numpy();target=f.Target.to_numpy();yearmap=calendar.set_index('Date').ValidationYear.to_dict();w=np.linspace(2,1,200)
    assert json.loads((ROOT/'feature_audit.json').read_text())['passed'];audits=[]
    for name,features,mask_features in specs():
        out=ROOT/name;rr=json.loads((out/'results.json').read_text());xx=np.column_stack([x,z[features].to_numpy()]) if features else x;names=NAMES+features
        pred=np.load(out/'predictions.npz');score=pred['score'];rank=pred['rank'];history=pd.read_csv(out/'parameter_history.csv').set_index('Date');daily=pd.read_csv(out/'daily_metrics.csv').set_index('Date');peak=z[mask_features].abs().max(axis=1).to_numpy()
        count=0;maxerr=0.;official=[]
        for date,ix in groups.items():
            if date>calendar.Date.max() or date==pd.Timestamp('2020-10-01'):
                assert (rank[ix]==-1).all() and np.isnan(score[ix]).all();continue
            theta=history.loc[str(date.date()),[f'After_{v}' for v in names]].to_numpy(float);replay=theta[0]+np.einsum('ij,j->i',xx[ix,1:],theta[1:])
            np.testing.assert_allclose(replay,score[ix],atol=1e-12,rtol=1e-10);count+=len(ix);maxerr=max(maxerr,float(np.max(np.abs(replay-score[ix]))))
            oo=np.lexsort((codes[ix],-score[ix]));expected=np.empty(len(ix),np.int32);expected[oo]=np.arange(len(ix));np.testing.assert_array_equal(expected,rank[ix])
            if date not in yearmap:continue
            ii=ix[oo];missing=int((~np.isfinite(target[np.r_[ii[:200],ii[-200:]]])).sum());row=daily.loc[str(date.date())];assert missing==row.SelectedMissingTargets
            if missing:assert pd.isna(row.OfficialDailySpread);continue
            spread=((target[ii[:200]]*w).sum()-(target[ii[-200:][::-1]]*w).sum())/w.mean()
            np.testing.assert_allclose(spread,row.OfficialDailySpread,atol=2e-12,rtol=1e-10)
            official.append(pd.DataFrame({'Date':str(date.date()),'Rank':rank[ix],'Target':target[ix]}))
        exact=float(calc_spread_return_sharpe(pd.concat(official,ignore_index=True)));np.testing.assert_allclose(exact,rr['sharpe'],atol=2e-13,rtol=0)
        tr=pd.read_csv(out/'training_updates.csv');assert len(tr)==1199
        for row in tr.itertuples():
            assert pd.Timestamp(row.SignalDate)<pd.Timestamp(row.ExitDate)<=pd.Timestamp(row.Date)
            ix=groups[pd.Timestamp(row.SignalDate)];known=np.isfinite(target[ix]);excluded=known&(peak[ix]>100)
            assert int(excluded.sum())==row.ThresholdSkippedStocks and int((known&~excluded).sum())==row.TrainingStocks
        a={'passed':True,'all_forecast_rows':count,'all_integer_ranks_checked':True,'all_953_daily_returns_checked':True,'official_sharpe_recalculated':exact,'max_independent_score_abs_error':maxerr,'all_1199_updates_independently_checked_in_update_audit':True,'all_threshold_masks_and_training_timing_checked':True}
        save(out/'audit.json',a);audits.append({'name':name,**a});print('AUDITED',name,exact,flush=True)
    save(ROOT/'audit.json',{'passed':True,'model_count':len(audits),'models':audits,'feature_audit_passed':True})
if __name__=='__main__':main()
