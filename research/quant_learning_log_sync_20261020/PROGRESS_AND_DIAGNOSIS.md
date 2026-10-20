# Quant Learning & Internship 2027 — 2026-10-20 v2.46 delta

As-of 2026-10-20 18:58 Asia/Taipei. Evidence-qualified **daily learning and research status**, not new trading alpha or final-test data.

## Verified project anchors
- Quant main **f8ebf9db13bf62497f4e04e2accf20937055994d** remains last publicly observable commit (2026-10-13); latest JPX portfolio commit **b128fd5c26e4a3a4d57cadb919c7582cd502eb5d** (2026-10-17). The completed 2026-10-17 CNN-LSTM 8-fold validation and separate 2026-10-18 Ensemble 8-fold historical development evaluations remain prior results, not current-run scores.
- Kaggle **joint No-ReLU/No-Pooling** variant reportedly submitted on 2026-10-20 (lookback 42, Conv k=3, 8 folds × 20 epochs). Initial GPU checks were reported in a recent conversation but **no first-fold results, model checkpoint or full-run completion** are verifiable from checked sources. A submission/queue is neither success nor failed alpha evidence. A joint intervention does not isolate ReLU from pooling; a fair component attribution would need ReLU × Pool 2×2 matched Date×Ticker/seed/label/metric controls.
- PRISM-VQ paper arXiv:2605.13407 by Kim & Song (GRU, cross-asset attention, VQ discrete latent factors, structure-conditioned temporal MoE, dynamic factor loadings) was studied at the formula-map level on 2026-10-20. **No JPX PRISM implementation, no independent no-hint derivation, no measured performance.** Source: https://arxiv.org/abs/2605.13407 .

## v2 mastery, review and research decisions
- New narrow mastery: PRISM-VQ architecture/factor structure = **Learned / unverified**. No existing Assisted/Independent/Research-ready ability promoted or downgraded based only on code artifacts or paper explanations.
- 2026-10-19 D21/D7/D2 due reviews -> overdue/unverified; 2026-10-20 D21 Markov, D7 OLS, D2 Ensemble tie/ReLU -> due/unverified, not passed. New paper/factorial-ablation review: **D2 2026-10-22; D7 2026-10-27; D21 2026-11-10**. Preserve D21 > D7 > D2.
- Research Log: new queued two-factor architecture ablation and paper-derived hypothesis. Formal Failed Experiments: **none newly verified**. No Project 0 signal or fitted weights transfer; only fair-ablation and factor as-of-data-audit ideas.
- Untouched final test remains closed; `v7_equal` remains the official baseline.

## Daily Learning Diagnosis (new evidence only)
1. **Research / Transfer:** ReLU and Pooling simultaneously changed; not a one-variable attribution or yet verified Kaggle score. Next session: no-hint 2×2 design fixing Date×Ticker, seeds, target availability, zero-signal tie controls and costs.
2. **Derivation / AI-dependency:** PRISM-VQ paper formulas were requested/explained, but no unseen self-derived tensor/factor equation, point-in-time prior or 200/200 JPX scoring adaptation was demonstrated. Next session: no-hint reconstruct `ŷ_i=α_i+β_p,iᵀ f_p+β_l,iᵀ f_l,i` and GRU→attention→VQ→MoE shape/timing contracts.

Teaching: paper branch notation -> no-hint reconstruction -> JPX as-of protocol; coding remains test-first; Markov mainline unchanged.

This new **v2.46** note preserves 2026-10-19 Library v2.45 evidence and does not rewrite older research/Release data. Portable control-state ZIP SHA-256: `b701282a5cd3f65fe7adca64aba6d4311757acd32081e8ef3364ee3524e5d6f1`. File-manifest/readback status is maintained separately; ZIP verification is not a proof of Kaggle execution.
