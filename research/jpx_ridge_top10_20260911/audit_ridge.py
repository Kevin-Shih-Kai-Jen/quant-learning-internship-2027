from pathlib import Path
import json
import numpy as np
import pandas as pd
from run_experiment import ridge,sufficient,LAMBDAS
ROOT=Path(__file__).resolve().parent
# Analytic one-feature example confirms MSE normalization, lambda units and
# unpenalized intercept independently of the JPX data or augmented solver.
x=np.array([[-1.],[1.]]);y=np.array([3.,7.])
np.testing.assert_allclose(ridge(sufficient(x,y),0.),[5.,2.],atol=1e-12)
np.testing.assert_allclose(ridge(sufficient(x,y),1.),[5.,1.],atol=1e-12)
np.testing.assert_allclose(ridge(sufficient(x,y),3.),[5.,.5],atol=1e-12)
np.testing.assert_allclose(ridge(sufficient(x*2,y),1.),[5.,.8],atol=1e-12)
f=pd.read_pickle(ROOT.parent/'jpx_return_t_20260911/return_t_features.pkl')
fits=json.loads((ROOT/'model_fits.json').read_text())
tuning=json.loads((ROOT/'lambda_selection.json').read_text())
for fit,tune in zip(fits,tuning):
    year=fit['validation_year'];cutoff=pd.Timestamp(fit['fit_asof'])
    outer=f.loc[f.SignalDate.dt.year.eq(year-1)&f.ExitDate.le(cutoff)&f.Target.notna()&f.TEligible]
    assert len(outer)==fit['train_rows']
    weighted={a:0. for a in LAMBDAS};n=0
    for fold in tune['folds']:
        asof=pd.Timestamp(fold['fit_asof'])
        training=outer.loc[outer.SignalDate.lt(asof)&outer.ExitDate.le(asof)]
        valid=outer.loc[outer.SignalDate.between(pd.Timestamp(fold['validation_first_signal']),pd.Timestamp(fold['validation_last_signal']))]
        assert len(training)==fold['train_rows'] and len(valid)==fold['validation_rows']
        assert training.ExitDate.max()<=asof and training.SignalDate.max()<valid.SignalDate.min()
        assert valid.ExitDate.max()<=cutoff
        assert set(training.SignalDate).isdisjoint(valid.SignalDate)
        for score in fold['scores']:weighted[score['lambda']]+=score['mse']*len(valid)
        n+=len(valid)
    expected=min(LAMBDAS,key=lambda a:(weighted[a]/n,a))
    assert expected==fit['lambda']==tune['chosen_lambda']
    for score in tune['aggregate_scores']:
        np.testing.assert_allclose(score['mse'],weighted[score['lambda']]/n,atol=1e-16,rtol=1e-12)
    assert fit['fit']['coefficient_l2']<=fit['fit']['ols_coefficient_l2']+1e-12
result={'analytic_lambda_mse_units_checked':True,'intercept_not_penalized':True,
    'penalty_is_on_existing_feature_units':True,'inner_folds_have_no_future_labels':True,
    'same_date_stocks_kept_together':True,'lambda_choice_recomputed_from_inner_scores':True,
    'outer_validation_not_used_for_lambda':True,'all_years_at_upper_grid_boundary':all(f['lambda']==max(LAMBDAS) for f in fits),
    'all_passed':True}
(ROOT/'independent_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
