# 前五個未加 Ridge 策略的成本情境

Select the five highest cumulative-return variants from the ten previously listed OLS strategies, using the corrected Return-T performance when ranking. Selected: original dynamic close-only, fixed30/70 close-only, fixed50/50 close-only, original dynamic target exit, fixed30/70 target exit. This is ex-post development-data ranking, not independent model selection.

Freeze original model predictions, original market regimes, actual-calendar order dates, top-three side allocation, sign transfers, target/close exit schedules and original no-volume carry policy. Reconstruct all five zero-cost trade ledgers using the verified original engine and reproduce all 953 daily returns. Cost engine replays the immutable gross execution schedule and resizes each next interval's notionals using remaining equity, including reserved capital and entry-cost budget. It does not change limit-fill probability or assume historical quote data that do not exist.

Before cost results, fix these illustrative scenarios (all proportional costs apply on BOTH entry and exit value, 1 bp = 0.01%):

| Scenario | Commission/fees each leg (bp) | Spread cost each leg (bp) | Extra slippage each leg (bp) | Annual stock borrow rate |
|---|---:|---:|---:|---:|
| low, optimistic | 1 | 1 | 1 | 1% |
| middle | 5 | 2.5 | 2.5 | 3% |
| high stress | 10 | 5 | 5 | 10% |

Rates are hypothetical stress assumptions, not measured 2018–2021 quotes, historical borrow rates or a particular broker invoice. A spread cost of 2.5 bp each leg corresponds to 5 bp full bid-ask spread if both legs cross half-spread. Extra slippage excludes this spread amount to avoid double counting. Both modeled as explicit cost debits against the unchanged theoretical fill prices (execution-cost overlay). Entry cost rate c reserves capital by new_base = (equity - carried_notional)/(1+c*sum_executable_slot_weights); each slot gets new_base * abs(weight). Exit costs use split-adjusted exit notional, not original entry notional.

Borrow fee proxy uses annual rate / 365 * entry notional of shorts * actual calendar holding days, including weekends and zero-volume carry days. Accrue over each modeled holding interval, not per trade count. This is an entry-notional trade-date approximation, not an exact settled-collateral broker calculation. No short-proceeds interest or cash interest assumed.

No fee is charged to unavailable/cash orders. Cost components remain separate. Zero-cost identity and a second independent reconciliation of trade costs/PnL/compounding are required. Include standalone commission-only and incremental spread/slippage/borrow scenarios, plus a numerical break-even all-in per-leg cost with 0% and 3% annual borrow. Break-even is a historical diagnostic, not a selected trading parameter.

Without specified broker, account size, base currency, tax residence, historic bid/ask/depth and lending inventory, do not fabricate minimum commissions, actual market impact, 100-share lot rounding, taxes, FX conversion, borrow rejection, margin liquidation or dividends. Explain these exclusions and the current idealized entry/limit-fill assumptions explicitly. No leverage financing charged because gross stock notional is capped at equity; this does not prove an actual broker would allow every short.

Official current references checked 2026-09-11: IBKR Japan commissions (tier1 0.05% min JPY80; fixed0.08% minJPY80; account/market-volume dependent), stock-borrow cost/availability, JPX100-share unit standardization on2018-10-01. Current fee references motivate explanations only; they are not historical rate evidence or personalized brokerage assumptions.
