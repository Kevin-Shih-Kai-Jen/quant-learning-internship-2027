# Ridge across the ten previously listed variants

Change stock fitting only: MSE + lambda * sum(beta_j^2), intercept unpenalized, existing feature units unchanged. Annual training uses the previous year, with all training label exits known by the first validation signal. Market models, frozen market signals, thresholds, candidate universe, top-three 50/30/20 within-side allocation, wrong-sign transfers and execution rules remain unchanged.

Use the previously established lambda grid [0, .001, .01, .1, 1, 10, 100]. Choose on pooled MSE of two chronological inner folds within the training year (first half to next quarter; first three quarters to final quarter), purging unavailable labels. No selection by outer-period returns or drawdown; do not expand the grid after seeing results. Price/volume level T and return T are separate feature families; their coefficients and lambda selection are shared across allocation variants with identical training data.

Compare six target-price exit variants and four close-only variants from the last comparison. Reproduce old orders and all ten baseline daily-return series before interpreting Ridge. Close-only variants use the original execution engine's close_only flag and independent capital accounting, not a weighted shortcut. Frozen original market g is used for original dynamic variants; frozen expanding g/signed regimes for their respective variants. Fixed 70/30 is included only for its previously listed close-only variant.

Report paired cumulative return, drawdown, Sharpe, annual results, selected lambdas and realized exposure. Same 953 historical intervals, no costs, no fresh holdout. Keep the active baseline unchanged: these are requested trials.
