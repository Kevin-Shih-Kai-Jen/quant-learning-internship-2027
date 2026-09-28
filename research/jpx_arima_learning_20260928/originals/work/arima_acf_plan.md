# ARIMA(3,1,1) and ARIMA(5,1,1): residual ACF correction

User confirmed: preserve existing ARIMA coefficients; only add residual ACF correction. Do not retrain on Target MSE, tune correction strength, average the two models, or use formal test. This is an exploratory follow-up after selecting models on reused validation.

Reuse the completed grid fits, training-only scale/anchor, causal price cache, universe, Target, annual folds, expanding start, and original fallback rules. Annual coefficients remain fixed within a fold. No expensive re-estimation is needed.

Residual is the original ARIMA one-step price innovation, in its normalized price units, from forward filtering only. It is not the JPX Target error, squared error, smoothed error, or the error of the corrected model. Missing price observations have missing residuals. For estimating ACF discard the first 20 observed training residuals; retain the calendar and require at least 100 remaining residuals and 80 valid pairs at each of lags 1 and 2. Compute mean-centered, unadjusted conservative ACF: sum of available lagged centered products / sum of available centered squares. Set missing centered terms to zero, never compress missing dates. ACF lags 1..20 are reported; only lags 1 and 2 affect predictions. ACF is estimated using training dates strictly before first validation signal and is fixed for that year. No significance gating or threshold tuning.

Retain the baseline's zero-mean innovation assumption. Fixed correction strength = 1, no extra residual intercept: u1 = rho1 * e[t], u2 = rho2 * e[t]. These are separate direct-lag projections using the latest observed original residual, not rho1 squared and not a multivariate regression. The stationary equal-variance interpretation is an approximation. ACF itself is not a complete forecasting model.

For q=1, propagate hypothetical future residuals through the original differenced ARIMA dynamics:
 corrected P1 = base P1 + scale*u1
 corrected P2 = base P2 + scale*((1 + phi1 + theta1)*u1 + u2)
 score = corrected P2 / corrected P1 - 1.
This includes the effect of tomorrow's correction on the following price and the MA(1) term, without adding today's MA residual again. Original forward states remain unmodified as actual observations arrive.

If training ACF unavailable or current residual missing, retain the original baseline forecast. If the original baseline is invalid retain its zero-score fallback; if corrected prices are invalid/nonpositive use zero score and record separately. Preserve all stocks and finite extreme values. No future realization is used to forecast earlier dates.

Evaluate 4 variants (two originals and their ACF corrections) plus v7 on their common scorable dates; official top/bottom 200, 2-to-1 weights, descending score and stock-code ascending ties. Report full and annual Sharpe and Rank IC, changes in ranking, fallback, diagnostics. Preserve all output. Paired 20-day circular-block bootstrap, 2000 draws, seed 20260924, ACF-minus-corresponding-base differences; approximate simultaneous interval across the two ACF comparisons only, not correction of prior 25-model selection. No annualization or costs.

Audit original stored forecasts/scores, independent recursive propagation on deterministic samples with full finite AR histories, zero-correction identity, prefix/modified-future invariance of residuals and forecasts, training-only ACF calculation, sample ACF agreement with statsmodels, complete unique ranks, official metric independent reproduction. Failed mathematical assertions are fatal. Do not silently change a specification based on results.
