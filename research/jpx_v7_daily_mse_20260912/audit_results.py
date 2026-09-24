from pathlib import Path
import sys,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from run_v7 import NAMES,FEATURES,INPUTS

def build_x(g):
    v={c:np.where(np.isfinite(g[c]),g[c],0.) for c in INPUTS}
    for k in [5,22,60]:v[f'P{k}xV{k}']=v[f'T{k}']*v[f'V{k}']
    return np.column_stack([np.ones(len(g))]+[v[c] for c in FEATURES])

def main():
    f=pd.read_pickle(ROOT.parent/'jpx_stock_returns_20260910/features.pkl')[['SignalDate','SecuritiesCode','ExitDate','Target']+INPUTS[:-2]]
    f=f.merge(pd.read_pickle(ROOT.parent/'jpx_v5_daily_returns_20260912/one_day_returns.pkl'),on=['SignalDate','SecuritiesCode'],validate='one_to_one').rename(columns={'SignalDate':'Date'})
    summary=json.loads((ROOT/'results.json').read_text());audits={}
    for variant in ['v7_equal','v7_jpx']:
        out=ROOT/variant
        r=pd.concat([pd.read_csv(out/name,parse_dates=['Date']) for name in ['ranks_warmup.csv.gz']+[f'ranks_{y}.csv.gz' for y in [2018,2019,2020,2021]]],ignore_index=True)
        assert 'Target' not in r and not r.duplicated(['Date','SecuritiesCode']).any()
        assert not r.Date.eq(pd.Timestamp('2020-10-01')).any()
        states=pd.read_csv(out/'parameter_history.csv',parse_dates=['Date']).set_index('Date')
        updates=pd.read_csv(out/'training_updates.csv',parse_dates=['Date','SignalDate','ExitDate'])
        selected=pd.read_csv(out/'selected_200_each_side.csv.gz',parse_dates=['Date'])
        daily=pd.read_csv(out/'daily_spread_returns.csv',parse_dates=['Date'])
        assert updates.Date.is_unique and updates.SignalDate.is_unique
        assert (states.UpdatesToday<=1).all()
        np.testing.assert_array_equal(states.CumulativeUpdates,states.UpdatesToday.cumsum())
        before=states[[f'Before_{n}' for n in NAMES]].to_numpy()
        after=states[[f'After_{n}' for n in NAMES]].to_numpy()
        np.testing.assert_array_equal(before[0],np.zeros(12))
        np.testing.assert_array_equal(before[1:],after[:-1])
        no_update=states.UpdatesToday.eq(0).to_numpy()
        np.testing.assert_array_equal(before[no_update],after[no_update])
        assert set(updates.SignalDate)==set(r.Date.unique())
        assert (updates.SignalDate<updates.ExitDate).all() and (updates.ExitDate<=updates.Date).all()
        assert updates.ExitDate.eq(updates.Date).all()
        joined=r.merge(f,on=['Date','SecuritiesCode'],validate='one_to_one')
        allgroups={d:g.sort_values('Rank') for d,g in joined.groupby('Date',sort=True)}
        max_forecast_error=0.;max_parameter_error=0.;max_gradient_error=0.
        for date,g in allgroups.items():
            assert g.Rank.tolist()==list(range(len(g)))
            ordered=g.sort_values(['g','SecuritiesCode'],ascending=[False,True])
            assert ordered.SecuritiesCode.tolist()==g.SecuritiesCode.tolist()
            x=build_x(g);theta=states.loc[date,[f'After_{n}' for n in NAMES]].to_numpy(dtype=float)
            error=float(abs(x@theta-g.g.to_numpy()).max());max_forecast_error=max(max_forecast_error,error)
            assert error<1e-10
            if variant=='v7_equal':expected=np.ones(len(g))
            else:expected=np.r_[np.linspace(2,1,200),np.zeros(len(g)-400),np.linspace(1,2,200)]
            np.testing.assert_allclose(g.TrainingRawWeight,expected,atol=1e-14)
        for u in updates.itertuples():
            g=allgroups[u.SignalDate];x=build_x(g);y=g.Target.to_numpy();w=g.TrainingRawWeight.to_numpy()
            assert g.ExitDate.eq(u.ExitDate).all()
            keep=np.isfinite(y)&(w>0)
            assert int(keep.sum())==u.PositiveWeightStocks
            assert int((~np.isfinite(y)).sum())==u.MissingTargetStocks
            assert int(((~np.isfinite(y))&(w>0)).sum())==u.MissingPositiveWeightStocks
            x=x[keep];y=y[keep];w=w[keep]/w[keep].sum()
            theta=states.loc[u.Date,[f'Before_{n}' for n in NAMES]].to_numpy(dtype=float)
            expected_after=states.loc[u.Date,[f'After_{n}' for n in NAMES]].to_numpy(dtype=float)
            err=x@theta-y
            # Independent row-by-row sums and a singular-value learning-rate check.
            grad=2*np.sum((w*err)[:,None]*x,axis=0)
            singular=np.linalg.svd(np.sqrt(w)[:,None]*x,compute_uv=False)
            rate=1/(2*singular[0]**2)
            predicted_after=theta-rate*grad
            pe=float(abs(predicted_after-expected_after).max());max_parameter_error=max(max_parameter_error,pe)
            np.testing.assert_allclose(predicted_after,expected_after,atol=1e-11,rtol=1e-9)
            saved_grad=np.array([getattr(u,f'Gradient_{n}') for n in NAMES])
            ge=float(abs(grad-saved_grad).max());max_gradient_error=max(max_gradient_error,ge)
            np.testing.assert_allclose(grad,saved_grad,atol=1e-9,rtol=1e-9)
            np.testing.assert_allclose(rate,u.LearningRate,atol=1e-13,rtol=1e-10)
            np.testing.assert_allclose(w@err**2,u.MSEBeforeStep,atol=1e-11,rtol=1e-10)
            np.testing.assert_allclose(w@(x@expected_after-y)**2,u.MSEAfterStep,atol=1e-11,rtol=1e-10)
            np.testing.assert_allclose(w@(g.g.to_numpy()[keep]-y)**2,u.ForecastMSE,atol=1e-11,rtol=1e-10)
        spreads=[]
        for date,h in selected.groupby('Date'):
            g=allgroups[date]
            up=h.loc[h.Side.eq('long')].sort_values('SideRank');down=h.loc[h.Side.eq('short')].sort_values('SideRank')
            assert len(up)==len(down)==200
            assert up.SecuritiesCode.tolist()==g.head(200).SecuritiesCode.tolist()
            assert down.SecuritiesCode.tolist()==g.tail(200).iloc[::-1].SecuritiesCode.tolist()
            weights=2-np.arange(200)/199
            np.testing.assert_allclose(up.RawRankWeight,weights,atol=1e-14)
            np.testing.assert_allclose(down.RawRankWeight,weights,atol=1e-14)
            spread=np.nan if h.Target.isna().any() else 200*(np.average(up.Target,weights=weights)-np.average(down.Target,weights=weights))
            spreads.append({'Date':date,'Spread':spread})
        check=daily.merge(pd.DataFrame(spreads),on='Date',validate='one_to_one')
        np.testing.assert_allclose(check.Spread,check.OfficialDailySpread,atol=1e-11,equal_nan=True)
        score=float(check.Spread.mean()/check.Spread.std(ddof=1))
        np.testing.assert_allclose(score,summary['variants'][variant]['validation']['official_style_unannualized_sharpe'],atol=1e-12)
        for year in [2018,2019,2020,2021]:
            orig=pd.read_csv(ROOT.parent/f'jpx_v5_daily_returns_20260912/ranks_{year}.csv.gz',usecols=['Date','SecuritiesCode'],parse_dates=['Date'])
            current=r.loc[r.ValidationYear.eq(year),['Date','SecuritiesCode']]
            assert set(map(tuple,orig.to_numpy()))==set(map(tuple,current.to_numpy()))
        audits[variant]={'all_checks_passed':True,'every_training_day_used_once':len(updates),
            'one_parameter_update_per_date_verified':True,'zero_initialization_and_continuity_verified':True,
            'all_forecast_ranks_and_frozen_training_weights_verified':True,'no_future_labels_used':True,
            'all_daily_mse_gradients_and_single_updates_independently_recomputed':True,
            'learning_rates_crosschecked_with_singular_values':True,'parameter_max_error':max_parameter_error,
            'gradient_max_error':max_gradient_error,'forecast_max_error':max_forecast_error,
            'official_sharpe_independent':score,'all_validation_universes_equal_v5':True}
        print(variant,json.dumps(audits[variant]),flush=True)
    save={'all_checks_passed':True,'variants':audits}
    (ROOT/'independent_audit.json').write_text(json.dumps(save,indent=2))

if __name__=='__main__':main()
