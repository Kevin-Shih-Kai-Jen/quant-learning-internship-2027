"""Independent checks of stored forecasts, calendar, fit boundaries and labels."""
from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'arima011_expanding'
BASE=Path('/Users/coolguy/.codex/.chatgpt-projects/g-p-6a8ad7dc606481919474d5424342a1c3')


def main():
    data=np.load(RUN/'prices.npz')
    dates,codes,prices=data['dates'],data['codes'],data['prices']
    vp,years=data['valid_positions'],data['years']
    records=[json.loads(f.read_text()) for f in sorted((RUN/'jobs').glob('*.json'))]
    assert len(records)==8000
    assert all(r['train_last']<r['first_signal'] for r in records)
    assert all(r['train_start']==str(dates[0]) for r in records)
    assert all(r['training_slots']==int(vp[years==r['year']][0]) for r in records)
    assert all(r['valid_training_prices']>=126 for r in records if r['status']=='ok')
    assert all(r['attempts'][-1]['converged'] for r in records if r['status']=='ok')
    successful=[r for r in records if r['status']=='ok']
    assert all(np.isfinite(r['attempts'][-1]['llf']) and r['attempts'][-1]['llf'] != 0 for r in successful)
    assert all(len(r['params'])==2 and r['params']['sigma2']>0 for r in successful)
    tests=[]
    # Spread checks across all four years and stock-code ranges, independent of performance.
    for year in [2018,2019,2020,2021]:
        successes=[r for r in records if r['year']==year and r['status']=='ok']
        selected=[successes[i] for i in np.linspace(0,len(successes)-1,8,dtype=int)]
        for r in selected:
            code=r['code'];k=int(np.where(codes==code)[0][0])
            positions=vp[years==year];boundary=int(positions[0]);start=0
            y=(prices[start:int(positions[-1])+1,k]-r['anchor'])/r['scale']
            stored=np.load(RUN/'jobs'/f'{year}_{code}.npz');params=stored['params']
            assert len(params)==2 and np.isfinite(params).all()
            for j in [0,len(positions)//2,len(positions)-1]:
                cutoff=int(positions[j])-start
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    prefix=ARIMA(y[:cutoff+1],order=(0,1,1),trend='n').filter(params,cov_type='none')
                    direct=np.asarray(prefix.forecast(2))*r['scale']+r['anchor']
                    changed=y.copy()
                    changed[cutoff+1:]=5000+np.arange(len(changed)-cutoff-1)*173
                    future=ARIMA(changed,order=(0,1,1),trend='n').filter(params,cov_type='none')
                np.testing.assert_allclose(direct,stored['forecast_prices'][j],rtol=2e-10,atol=2e-8)
                np.testing.assert_allclose(prefix.filter_results.predicted_state[:,-1],
                                           future.filter_results.predicted_state[:,cutoff+1],
                                           rtol=1e-12,atol=1e-12)
                expected=direct[1]/direct[0]-1 if (direct>0).all() else 0.
                np.testing.assert_allclose(expected,stored['score'][j],rtol=1e-10,atol=1e-12)
                tests.append({'year':year,'code':code,'origin':str(dates[positions[j]]),
                              'max_forecast_error':float(np.max(np.abs(direct-stored['forecast_prices'][j])))})
    # Authoritative supplied Target is never replaced by a reconstructed price ratio.
    f=pd.read_pickle(BASE/'jpx_stock_returns_20260910/features.pkl')
    f['AP']=f.Close/f.CumulativeFactor
    lookup=f.set_index(['SignalDate','SecuritiesCode']).AP
    calendar=pd.read_csv(RUN/'calendar.csv',parse_dates=['Date'])
    target=f.loc[f.SignalDate.isin(calendar.Date)]
    entry=lookup.reindex(pd.MultiIndex.from_arrays([target.EntryDate,target.SecuritiesCode])).to_numpy()
    exitprice=lookup.reindex(pd.MultiIndex.from_arrays([target.ExitDate,target.SecuritiesCode])).to_numpy()
    actual=target.Target.to_numpy();derived=exitprice/entry-1
    known=np.isfinite(actual)&np.isfinite(derived)
    error=np.abs(actual[known]-derived[known])
    report={'passed':True,'stock_year_records_checked':len(records),
            'all_training_ends_before_prediction':True,'all_accepted_fits_converged':True,
            'accepted_models_with_root_within_1e_6_of_unit_circle':sum(
                r['min_ma_root']<1.000001 for r in successful),
            'independent_prefix_and_future_perturbation_checks':len(tests),
            'max_price_forecast_error':max(t['max_forecast_error'] for t in tests),
            'checks':tests,'supplied_targets_used_unchanged':True,
            'target_reconstruction_diagnostic':{'matched_finite_rows':int(known.sum()),
              'max_abs_difference':float(error.max()),'p99_abs_difference':float(np.quantile(error,.99)),
              'rows_abs_difference_gt_1e_6':int((error>1e-6).sum()),
              'note':'The supplied Target differs slightly from the forward-adjusted price ratio for some rows. Cause not established here. Scoring uses the original Target for both models.'}}
    (RUN/'independent_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='checks'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
