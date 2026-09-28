# JPX ARIMA(0,1,1): pre-run specification

User requests removing AR and using one MA lag, to examine whether forecast errors are useful. Retain no drift, expanding training, annual validation and official t+1 to t+2 ranking horizon.

## Structural finding recorded BEFORE performance is measured
For price differences x[t] = epsilon[t] + theta * epsilon[t-1], forecasts from t are E[x[t+1]|t] = theta * estimated epsilon[t], E[x[t+2]|t] = 0. Thus predicted price at t+2 equals predicted price at t+1. The existing ratio-of-price-point-forecasts score is identically zero. This is not a claim that the exact expectation of the random price ratio must be zero; the experiment uses the same point-forecast ratio proxy as prior runs.

- Verify the flat second-step price forecast on real data rather than interpret arbitrary tie-break Sharpe as MA forecasting power.
- All stocks independently fit ARIMA(0,1,1), trend='n', Gaussian MLE; invertible MA; same numerical normalization and optimization as previous run (500, then 1000 iterations if unconverged). Minimum 126 observed training prices.
- Expanding start 2017-01-04, annual refit before each existing ValidationYear first signal; no daily coefficient refit. Within-year updates only use observed prices through each forecast date. Missing observations retained.
- Score = forecast adjusted P[t+2|t] / forecast adjusted P[t+1|t] - 1. Save raw computed scores; after verifying flat horizon, use the analytical exact zero to prevent floating point roundoff from generating rankings. This is not outcome-based clipping.
- Fixed zero-score fallback for insufficient history, failed fit, invalid numerical fit or nonpositive forecast. Full stock universe and original Target retained. No finite extreme returns clipped.
- Same 953 dates and 1,864,363 stock-days, 2018..2021 ValidationYear folds; 2021 is incomplete. Top/bottom 200 with official weights 2 to 1. Ties by SecuritiesCode ascending. Check ranks equal an explicit all-zero-score code-order control.
- Score Rank IC is undefined on all-constant scores: report null, never silently convert to zero. Official Sharpe may be finite but measures only the code-order tie-break portfolio. Do not select or promote the model using it.
- One-step predicted return vs today's observed price is stored only as a numerical diagnostic showing MA affects the first forecast. No off-horizon performance search, no additional model order or drift tried.
- Verify prefix forecasts, perturb future inputs, training boundaries, expanding window, valid full forward-filter innovation variances and official metric implementation.
- Formal test untouched, previous models unmodified. This experiment cannot by itself refute all shock/innovation predictability hypotheses.
