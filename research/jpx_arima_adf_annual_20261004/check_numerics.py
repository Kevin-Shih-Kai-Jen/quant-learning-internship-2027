"""Adversarial time-boundary tests and independent stock forecast checks."""
import io
import json
import sqlite3
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from common import EXP, INPUT, OUT, YEARS, ORDERS, save
from prepare_adf import longest_segment, decide_d
from evaluate import choose_order


def selection_time_test():
    cal=pd.DataFrame({'Date':pd.to_datetime(['2018-12-24','2018-12-25','2018-12-27','2019-01-07']),
                      'ExitDate':pd.to_datetime(['2018-12-26','2018-12-27','2019-01-04','2019-01-09']),
                      'ValidationYear':[2018,2018,2018,2019]})
    dailies=[]
    for oi in range(len(ORDERS)):
        dailies.append(pd.DataFrame({'OfficialDailySpread':[1.,2.,3.,4.] if oi==0 else [1.,-1.,0.,0.],
                                     'RankIC':[0.,0.,0.,0.]}))
    oi,info=choose_order(dailies,cal,np.datetime64('2018-12-27'),2019)
    assert oi==0 and info['eligible_days']==2 and info['unmatured_prior_fold_days']==1
    for daily in dailies:
        daily.loc[[2,3],'OfficialDailySpread']=[1e12,-1e12]
        daily.loc[[2,3],'RankIC']=[1.,1.]
    other,after=choose_order(dailies,cal,np.datetime64('2018-12-27'),2019)
    assert other==oi and after==info
    # A fully tied history resolves by complexity, then p, then q, not candidate order.
    tied=[dailies[0].copy() for _ in ORDERS]
    assert choose_order(tied,cal,np.datetime64('2018-12-27'),2019)[0]==0
    assert choose_order(dailies,cal,np.datetime64('2017-12-28'),2018)[0]==0
    return {'mature_labels_only':True,'unmatured_and_future_return_perturbations_invariant':True,
            'no_history_initialization':True,'deterministic_tie_break':True}


def adf_checks():
    with np.load(INPUT/'prices.npz') as z:data={k:z[k] for k in z.files}
    records=json.loads((OUT/'adf_decisions.json').read_text())['records']
    chosen=[]
    for year in YEARS:
        for d in [0,1,2,None]:
            rows=[r for r in records if r['year']==year and r['chosen_d']==d]
            chosen.extend(rows[:2])
    max_stat_diff=0.
    for r in chosen:
        end=r['train_end_exclusive'];ci=r['ci']
        history=data['prices'][:end,ci]
        changed=data['prices'][:,ci].copy();changed[end:]=1e7+np.arange(len(changed)-end)
        recomputed=decide_d(changed[:end],data['dates'][:end],r['code'],r['year'])
        assert recomputed['chosen_d']==r['chosen_d'] and recomputed['tests']==r['tests']
        start,stop=longest_segment(history)
        assert (start,stop)==(r['segment_start_index'],r['segment_end_exclusive'])
        for test in r['tests']:
            if test['status']!='ok':continue
            series=np.diff(history[start:stop],n=test['d']) if test['d'] else history[start:stop]
            raw=adfuller(series,regression='c',autolag='AIC')
            delta=abs(raw[0]-test['statistic']);max_stat_diff=max(max_stat_diff,delta)
            assert bool(raw[0]<raw[4]['5%'])==test['reject_unit_root_5pct']
            assert raw[2]==test['lag']
    # Missing calendar slots never get compressed into fake adjacent observations.
    a=np.array([1.,2.,np.nan,3.,4.]);assert longest_segment(a)==(3,5)
    return {'stock_years':len(chosen),'future_price_invariance':True,
            'critical_value_decision_matches_unscaled_adf':True,'max_statistic_roundoff':max_stat_diff,
            'missing_gap_and_newest_tie_test':True}


def forecast_checks():
    with np.load(INPUT/'prices.npz') as z:data={k:z[k] for k in z.files}
    con=sqlite3.connect(f'file:{OUT}/new_fits.sqlite?mode=ro',uri=True)
    all_rows=con.execute("SELECT oi,yi,ci,audit,payload FROM fits WHERE status='ok' ORDER BY oi,yi,ci").fetchall()
    # Samples chosen by stock/order keys and d, independent of validation results.
    selected=[]
    for yi,year in enumerate(YEARS):
        for d in [0,2]:
            for p,q in [(1,1),(3,1),(5,5)]:
                options=[r for r in all_rows if r[0]==ORDERS.index((p,q)) and r[1]==yi and json.loads(r[3])['d']==d]
                if options:selected.append(options[0])
    checks=[]
    for oi,yi,ci,encoded,payload in selected:
        a=json.loads(encoded);z=np.load(io.BytesIO(payload));p,q=ORDERS[oi]
        positions=data['valid_positions'][data['years']==YEARS[yi]]
        y=(data['prices'][:int(positions[-1])+1,ci]-a['anchor'])/a['scale']
        for j in [0,len(positions)//2,len(positions)-1]:
            pos=int(positions[j])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                prefix=ARIMA(y[:pos+1],order=(p,a['d'],q),trend=a['trend']).filter(z['params'],cov_type='none')
                prices=np.asarray(prefix.forecast(2))*a['scale']+a['anchor']
                changed=y.copy();changed[pos+1:]=-100000.-np.arange(len(y)-pos-1)
                perturbed=ARIMA(changed,order=(p,a['d'],q),trend=a['trend']).filter(z['params'],cov_type='none')
            np.testing.assert_allclose(prices,z['forecast_prices'][j],rtol=2e-10,atol=2e-8)
            np.testing.assert_allclose(prefix.filter_results.predicted_state[:,-1],
                                       perturbed.filter_results.predicted_state[:,pos+1],rtol=1e-11,atol=1e-10)
            fv=prefix.filter_results.forecasts_error_cov[0,0,:]
            observed=np.isfinite(y[:pos+1]);observed[:int(prefix.loglikelihood_burn)]=False
            bad=bool((observed&(~np.isfinite(fv)|(fv<=0))).any()) or not(np.isfinite(prices).all() and (prices>0).all())
            score=0. if bad else prices[1]/prices[0]-1.
            bad=bad or not np.isfinite(score)
            if bad:score=0.
            assert bool(z['fallback'][j])==bad
            np.testing.assert_allclose(score,z['score'][j],rtol=1e-10,atol=1e-12)
            checks.append({'code':a['code'],'year':a['year'],'p':p,'d':a['d'],'q':q,
                           'date':str(data['dates'][pos]),'price_error':float(np.max(np.abs(prices-z['forecast_prices'][j])))})
    con.close()
    return {'checks':checks,'count':len(checks),'max_price_error':max(c['price_error'] for c in checks),
            'direct_forecast_and_future_perturbation_passed':True}


if __name__=='__main__':
    result={'selection_time_tests':selection_time_test(),'adf_tests':adf_checks(),'forecast_tests':forecast_checks(),'passed':True}
    save(EXP/'numerical_checks.json',result)
    print(json.dumps({k:v if k!='forecast_tests' else {a:b for a,b in v.items() if a!='checks'} for k,v in result.items()}))
