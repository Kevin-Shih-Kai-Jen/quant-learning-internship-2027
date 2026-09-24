from pathlib import Path
import json,sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from run_v6 import FEATURES,INPUTS

def independent_x(frame):
    cols={c:np.where(np.isfinite(frame[c]),frame[c],0.) for c in INPUTS}
    for k in [5,22,60]:cols[f'P{k}xV{k}']=cols[f'T{k}']*cols[f'V{k}']
    return np.column_stack([np.ones(len(frame))]+[cols[c] for c in FEATURES])

def independent_replay(theta,cohorts,rate):
    # Direct 400-row residuals, independent of the production quadratic statistics.
    theta=theta.copy()
    for x,y in cohorts:
        err=x@theta-y;loss=float(np.linalg.norm(err)/20)
        grad=x.T@err/(400*loss) if loss>1e-14 else np.zeros(12)
        norm2=float(grad@grad)
        if norm2<=1e-28:continue
        trial=rate*1.5
        for _ in range(60):
            candidate=theta-trial*grad
            newloss=float(np.linalg.norm(x@candidate-y)/20)
            if newloss<=loss-.0001*trial*norm2:
                theta=candidate;rate=trial;break
            trial*=.5
    return theta,rate

def main():
    result=json.loads((ROOT/'results.json').read_text())
    states=pd.read_csv(ROOT/'parameter_history.csv',parse_dates=['Date'])
    updates=pd.read_csv(ROOT/'training_updates.csv',parse_dates=['Date'])
    releases=pd.read_csv(ROOT/'label_releases.csv',parse_dates=['SignalDate','ExitDate','AddedAsOf'])
    allselected=pd.read_csv(ROOT/'all_selected_forecasts.csv.gz',parse_dates=['Date','ExitDate'])
    selected=pd.read_csv(ROOT/'selected_200_each_side.csv.gz',parse_dates=['Date'])
    daily=pd.read_csv(ROOT/'daily_spread_returns.csv',parse_dates=['Date'])
    names=['alpha']+FEATURES
    before=states[[f'Before_{c}' for c in names]].to_numpy()
    after=states[[f'After_{c}' for c in names]].to_numpy()
    np.testing.assert_array_equal(before[0],np.zeros(12))
    np.testing.assert_array_equal(before[1:],after[:-1])
    assert (states.TrainingDays.diff().dropna()>=0).all()
    np.testing.assert_array_equal(updates.ReplayedDays,updates.TrainingDays)
    np.testing.assert_allclose(updates.RateBefore.iloc[1:],updates.LastRate.iloc[:-1],atol=1e-18,rtol=1e-12)
    assert (releases.SignalDate<releases.ExitDate).all() and (releases.ExitDate<=releases.AddedAsOf).all()
    included=releases.loc[releases.Included].sort_values('SignalDate')
    for u in updates.itertuples():
        history=included.loc[included.AddedAsOf.le(u.Date)]
        assert len(history)==u.TrainingDays==u.ReplayedDays
        assert history.AddedAsOf.eq(u.Date).sum()==u.AddedDays
    f=pd.read_pickle(ROOT.parent/'jpx_stock_returns_20260910/features.pkl')
    f=f.merge(pd.read_pickle(ROOT.parent/'jpx_v5_daily_returns_20260912/one_day_returns.pkl'),on=['SignalDate','SecuritiesCode'],validate='one_to_one').rename(columns={'SignalDate':'Date'})
    allselected=allselected.merge(f[['Date','SecuritiesCode','Target']+INPUTS],on=['Date','SecuritiesCode'],validate='one_to_one')
    assert allselected.groupby('Date').size().eq(400).all()
    assert not allselected.duplicated(['Date','SecuritiesCode']).any()
    x=independent_x(allselected)
    model=states.set_index('Date')[[f'After_{c}' for c in names]].reindex(allselected.Date).to_numpy()
    pred=np.einsum('ij,ij->i',x,model)
    max_error=float(abs(pred-allselected.g.to_numpy()).max())
    assert max_error<1e-11
    cohorts={date:(x[g.index],g.Target.to_numpy()) for date,g in allselected.groupby('Date',sort=True)}
    for r in releases.itertuples():
        assert r.MissingTargets==int((~np.isfinite(cohorts[r.SignalDate][1])).sum())
        assert r.Included==(r.MissingTargets==0)
    # Recreate the entire expanding replay at first update, year transitions, and final date.
    chosen=[updates.Date.iloc[0],updates.Date.iloc[-1]]
    chosen+=updates.groupby(updates.Date.dt.year).Date.first().tolist()
    replay_checks=[]
    for date in sorted(set(chosen)):
        u=updates.loc[updates.Date.eq(date)].iloc[0]
        state=states.loc[states.Date.eq(date)].iloc[0]
        initial=state[[f'Before_{c}' for c in names]].to_numpy(dtype=float)
        expected=state[[f'After_{c}' for c in names]].to_numpy(dtype=float)
        history=included.loc[included.AddedAsOf.le(date),'SignalDate']
        actual,rate=independent_replay(initial,[cohorts[d] for d in history],float(u.RateBefore))
        err=float(abs(actual-expected).max())
        np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=1e-8)
        replay_checks.append({'date':str(date.date()),'historical_day_batches_replayed':len(history),'parameter_max_error':err})
    independent_spreads=[]
    for year in [2018,2019,2020,2021]:
        ranks=pd.read_csv(ROOT/f'ranks_{year}.csv.gz',parse_dates=['Date'])
        old=pd.read_csv(ROOT.parent/f'jpx_v5_daily_returns_20260912/ranks_{year}.csv.gz',usecols=['Date','SecuritiesCode'],parse_dates=['Date'])
        assert 'Target' not in ranks
        assert set(map(tuple,ranks[['Date','SecuritiesCode']].to_numpy()))==set(map(tuple,old.to_numpy()))
        for date,g in ranks.groupby('Date'):
            g=g.sort_values(['g','SecuritiesCode'],ascending=[False,True])
            assert g.Rank.tolist()==list(range(len(g)))
            h=selected.loc[selected.Date.eq(date)]
            up=h.loc[h.Side.eq('long')].sort_values('SideRank')
            down=h.loc[h.Side.eq('short')].sort_values('SideRank')
            assert len(up)==len(down)==200
            assert up.SecuritiesCode.tolist()==g.head(200).SecuritiesCode.tolist()
            assert down.SecuritiesCode.tolist()==g.tail(200).iloc[::-1].SecuritiesCode.tolist()
            weights=2-np.arange(200)/199
            np.testing.assert_allclose(up.RawRankWeight,weights,atol=1e-14)
            np.testing.assert_allclose(down.RawRankWeight,weights,atol=1e-14)
            if h.Target.isna().any():spread=np.nan;rmse=np.nan
            else:
                spread=200*(np.average(up.Target,weights=weights)-np.average(down.Target,weights=weights))
                rmse=float(np.linalg.norm(h.g-h.Target)/20)
            independent_spreads.append({'Date':date,'Spread':spread,'DailyRMSE':rmse})
    independent=pd.DataFrame(independent_spreads)
    joined=daily.merge(independent,on='Date',validate='one_to_one')
    np.testing.assert_allclose(joined.Spread,joined.OfficialDailySpread,atol=1e-12,equal_nan=True)
    np.testing.assert_allclose(joined.DailyRMSE,joined.ForecastRMSE,atol=1e-12,equal_nan=True)
    score=float(joined.Spread.mean()/joined.Spread.std(ddof=1))
    np.testing.assert_allclose(score,result['validation_total']['official_style_unannualized_sharpe'],atol=1e-13)
    audit={'all_checks_passed':True,'one_zero_initialization_and_no_parameter_reset':True,
        'every_expansion_revisits_all_available_historical_day_batches':True,
        'release_times_and_incomplete_label_exclusions_checked':True,'each_batch_is_exactly_400':True,
        'independent_replay_uses_each_days_direct_rmse_without_date_averaging':replay_checks,
        'forecast_parameter_max_error':max_error,'all_validation_universes_match_v5':True,
        'all_unique_ranks_and_both_200_stock_sides_checked':True,'daily_rmse_and_jpx_spreads_independently_checked':True,
        'independent_official_sharpe':score,'total_daily_batch_visits':int(updates.ReplayedDays.sum()),
        'selected_missing_targets':int(selected.Target.isna().sum())}
    (ROOT/'independent_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit))

if __name__=='__main__':main()
