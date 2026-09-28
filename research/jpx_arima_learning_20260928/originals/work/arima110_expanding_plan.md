# JPX ARIMA(1,1,0) expanding-window experiment

User requests AR(1,1), forecast relative to original price, high/low 200 rankings, expanding training and annual validation. Interpret AR(1,1) as ARIMA(1,1,0), no drift. Scope: standalone exploratory strategy, not replacement of v7.

- Fit each stock using all available dates before each existing ValidationYear boundary, beginning 2017. No rolling 252-day cutoff. Minimum 126 observed training prices; leading/middle missing observations retained.
- Annual parameter refit; within-year coefficients fixed, daily forward state updated only with observed prices through signal day. Expanding training does not mean daily parameter estimation.
- User confirmed official tomorrow-to-day-after horizon. Score = forecast adjusted P[t+2|t] / forecast adjusted P[t+1|t] - 1. Both forecasts use information through t only. Second step recursively uses first predicted state, never actual future price. This point-forecast ratio is a return proxy, not the exact expectation of the return ratio.
- Same baseline calendar of 953 signal days, 2017-12-29 to 2021-12-01, existing ValidationYear 2018..2021; final year is incomplete. No fabricated remainder of 2021. Same 1,864,363 stock-days, official Target unchanged.
- Daily unique ranks, descending score, tie by code; official top/bottom 200 and weights 2 to 1. No clipping finite extremes. Fixed zero-score fallback for insufficient history, failed fit or invalid/nonpositive forecast prices. Missing current observations are handled by the forward state filter.
- Same causal price adjustments, training-only numerical normalization, stationary AR constraint, MLE solver and convergence checks as prior run. No MA parameter; missing MA roots are not a failure.
- Sharpe primary, Rank IC secondary; unannualized and before costs. v7 is existing historical reference; it is not refit in this experiment. Prior ARIMA515 uses rolling 252 days, so neither comparison isolates a single design change. Formal test untouched. Reused validation is exploratory.
- Checks: training dates strictly earlier than first forecast; expanding start retained across years; prefix/future perturbation audit; independent forecast-score recomputation; official metric equivalence; full forward innovation variances; same universe.

## Durable user preference for subsequent experiments
Use expanding training windows and yearly validation unless explicitly revised by user. Do not silently carry forward the previous 252-day window. State coefficient refit cadence separately. When models use future-return labels, exclude labels not yet realized by training cutoff.
