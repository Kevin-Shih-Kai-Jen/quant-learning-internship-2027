from pathlib import Path
import json
import numpy as np
import pandas as pd
from official_metric import calc_spread_return_sharpe
from run_experiment import predict_rank,BASE_FEATURES,FEATURES
ROOT=Path(__file__).resolve().parent

def main():
    # Independent analytic score; the official numerator is not a NAV return.
    toy=pd.concat([pd.DataFrame({'Date':pd.Timestamp('2024-01-01')+pd.Timedelta(days=j),
        'Rank':np.arange(500),'Target':np.r_[np.full(200,r),np.zeros(300)]}) for j,r in enumerate([.01,.02,-.005])])
    expected=np.array([2.,4.,-1.])
    np.testing.assert_allclose(calc_spread_return_sharpe(toy),expected.mean()/expected.std(ddof=1),atol=1e-13)
    observations=pd.DataFrame({'SignalDate':pd.Timestamp('2024-01-01'),'SecuritiesCode':np.arange(500),**{c:np.zeros(500) for c in BASE_FEATURES}})
    observations['T5']=np.arange(500,dtype=float);observations.loc[0,'T5']=np.nan
    theta=np.r_[1.,1.,np.zeros(8)]
    pred=predict_rank(observations,theta)
    assert pred.g.gt(0).all() and len(pred.tail(200))==200
    assert pred.iloc[0].SecuritiesCode==499 and pred.iloc[-1].SecuritiesCode==0
    scaled=predict_rank(observations,theta*2)
    assert pred.SecuritiesCode.tolist()==scaled.SecuritiesCode.tolist()
    try:predict_rank(observations.assign(Target=999.),theta)
    except AssertionError:pass
    else:raise AssertionError('Predictor accepted a held-out Target column')
    r=json.loads((ROOT/'results.json').read_text());fits=json.loads((ROOT/'model_fits.json').read_text())
    f=pd.read_pickle(ROOT.parent/'jpx_stock_returns_20260910/features.pkl')
    selected=pd.read_csv(ROOT/'selected_200_each_side.csv.gz',parse_dates=['Date'])
    daily=pd.read_csv(ROOT/'daily_spread_returns.csv',parse_dates=['Date'])
    checks=[];manual=[];g_error=0.
    for fit in fits:
        y=fit['validation_year'];a=pd.read_csv(ROOT/f'ranks_{y}.csv.gz',parse_dates=['Date'])
        assert 'Target' not in a and not a.duplicated(['Date','SecuritiesCode']).any()
        for date,g in a.groupby('Date'):
            ordered=g.sort_values(['g','SecuritiesCode'],ascending=[False,True])
            assert ordered.Rank.tolist()==list(range(len(g)))
            h=selected.loc[selected.Date.eq(date)]
            top=h.loc[h.Side.eq('long')].sort_values('SideRank');bottom=h.loc[h.Side.eq('short')].sort_values('SideRank')
            assert len(top)==len(bottom)==200
            assert top.SecuritiesCode.tolist()==ordered.head(200).SecuritiesCode.tolist()
            assert bottom.SecuritiesCode.tolist()==ordered.tail(200).iloc[::-1].SecuritiesCode.tolist()
            weights=2-np.arange(200)/199
            np.testing.assert_allclose(top.RawRankWeight,weights,atol=1e-15,rtol=0)
            np.testing.assert_allclose(bottom.RawRankWeight,weights,atol=1e-15,rtol=0)
            assert not top.Target.isna().any() and not bottom.Target.isna().any()
            spread=200*(np.average(top.Target,weights=weights)-np.average(bottom.Target,weights=weights))
            manual.append({'Date':date,'Spread':spread})
        # Independently reconstruct all selected predictions from raw saved T features.
        h=selected.loc[selected.ValidationYear.eq(y)]
        joined=h[['Date','SecuritiesCode','g','Fallback']].merge(f.rename(columns={'SignalDate':'Date'})[['Date','SecuritiesCode']+BASE_FEATURES],on=['Date','SecuritiesCode'],validate='one_to_one')
        np.testing.assert_array_equal(joined.Fallback,~np.isfinite(joined[BASE_FEATURES]).all(axis=1))
        b=joined[BASE_FEATURES].replace([np.inf,-np.inf],np.nan).fillna(0)
        for k in [5,22,60]:b[f'P{k}xV{k}']=b[f'T{k}']*b[f'V{k}']
        theta=np.array(list(fit['coefficients'].values()));expected=theta[0]+b[FEATURES].to_numpy()@theta[1:]
        error=float(np.max(np.abs(joined.g-expected)));g_error=max(g_error,error)
        assert error<1e-12
        cutoff=pd.Timestamp(fit['fit_asof'])
        tr=f.loc[f.SignalDate.dt.year.eq(y-1)&f.ExitDate.le(cutoff)&f.Target.notna()&f.Eligible]
        assert len(tr)==fit['train_rows'] and tr.ExitDate.max()<=cutoff
        checks.append({'year':y,'ranked_rows':len(a),'selected_rows':len(h),'training_rows':len(tr),'g_max_abs_error':error})
    independent=pd.DataFrame(manual).sort_values('Date')
    joined=daily.merge(independent,on='Date',validate='one_to_one')
    np.testing.assert_allclose(joined.OfficialDailySpread,joined.Spread,atol=1e-13,rtol=0)
    score=float(joined.Spread.mean()/joined.Spread.std(ddof=1))
    np.testing.assert_allclose(score,r['total']['official_style_unannualized_sharpe'],atol=1e-13,rtol=0)
    result={'all_checks_passed':True,'analytic_official_scale_and_ddof_checked':True,'positive_g_still_has_bottom_200_shorts':True,
        'positive_rescaling_preserves_ranks':True,'predictor_rejects_target_input':True,'all_daily_ranks_and_200_each_side_checked':True,
        'missing_selected_targets':int(selected.Target.isna().sum()),'selected_prediction_max_error':g_error,'years':checks,
        'independent_score':score,'annualization_not_applied_to_official_score':True}
    (ROOT/'independent_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
