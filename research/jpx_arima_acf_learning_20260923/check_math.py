"""核對遞迴梯度與未來資料隔離；合成測試資料不是回測績效。"""
import json
from pathlib import Path

import numpy as np

from model_math import Parameters, forecast, path_and_gradient, target_mse


def main():
    p = Parameters(a=.1, phi=.4, theta=.2, b=.3, c=.05)
    rng = np.random.default_rng(20260923)
    x = 100 + np.cumsum(rng.normal(0, .5, 150))
    init = dict(initial_delta=0., initial_residual=0.)
    y = np.full(len(x), np.nan)
    y[:-2] = x[2:] / x[1:-1] - 1
    mature = np.arange(len(x)) < 100
    loss, analytic = target_mse(x, y, mature, p, **init)
    numerical = []
    for j in range(5):
        plus, minus = p.array(), p.array()
        plus[j] += 1e-6
        minus[j] -= 1e-6
        high = target_mse(x, y, mature, Parameters(*plus), **init)[0]
        low = target_mse(x, y, mature, Parameters(*minus), **init)[0]
        numerical.append((high-low)/2e-6)
    np.testing.assert_allclose(analytic, numerical, rtol=1e-6, atol=1e-11)
    pred, grad, residual = path_and_gradient(x, p, **init)
    changed = x.copy()
    changed[105:] += 12
    pred2, grad2, _ = path_and_gradient(changed, p, **init)
    np.testing.assert_array_equal(pred[:104], pred2[:104])
    np.testing.assert_array_equal(grad[:104], grad2[:104])
    changed_y = y.copy()
    changed_y[100:] = .99
    loss2, g2 = target_mse(x, changed_y, mature, p, **init)
    assert loss == loss2
    np.testing.assert_array_equal(analytic, g2)
    for j in [3, 50, 120]:
        single = forecast(x[j], x[j]-x[j-1], residual[j-1], p)
        np.testing.assert_allclose(single['predicted_target'], pred[j-1], atol=1e-15)
    base = Parameters(.1, .4, .2, 0., 0.)
    f = forecast(100., 2., 1., base)
    assert f['e1'] == f['e2'] == 0
    np.testing.assert_allclose(f['z1'], 1.1)
    np.testing.assert_allclose(f['z2'], .54)
    np.testing.assert_allclose(f['predicted_target'], .54/101.1)
    corrected = forecast(100., 2., 1., Parameters(.1,.4,.2,.3,0.))
    np.testing.assert_allclose(corrected['z1'], 1.4)
    np.testing.assert_allclose(corrected['z2'], .81)
    np.testing.assert_allclose(corrected['predicted_target'], .81/101.4)
    result = {'status':'passed', 'synthetic_math_checks_only':True,
              'max_gradient_error':float(np.max(abs(analytic-numerical))),
              'future_price_invariance':True, 'unmatured_target_invariance':True,
              'zero_residual_ar_reduces_to_arima_111':True,
              'formal_backtest_run':False}
    Path(__file__).with_name('math_checks.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
