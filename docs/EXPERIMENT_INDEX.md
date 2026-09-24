# 實驗與資料下載索引

這份索引由首次快照的 [上傳計畫](../data/upload-plans/snapshot-2026-09-24.json) 彙整，標籤為 `snapshot-2026-09-24`。**首次上傳仍以驗證紀錄確認完成；本表列出待保存／已分包的內容，不表示全部資產已上傳。** 狀態請查 [快照入口](../data/index.json) 與 [遠端驗證紀錄](../data/verification/snapshot-2026-09-24.json)。

本次計畫共有 **64 個頂層區域（含根目錄）**；Git 可讀檔案 **1,918 檔 · 93.70 MiB**，Release 資料 **8,288 檔 · 10.25 GiB**。每列對應一個原專案頂層目錄；報告保持原始內容，正式基準仍 v7，v30 是最新完成實驗，ARIMA 支線未正式回測。

## 下載方式

在儲存庫根目錄，以表中的完整前綴替換 `PREFIX`：

```sh
python3 scripts/data_archive.py fetch --experiment 'PREFIX' --tag snapshot-2026-09-24
python3 scripts/data_archive.py verify --experiment 'PREFIX' --tag snapshot-2026-09-24
```

若最後一欄有 gzip 前綴，**也要另外執行一次 fetch／verify**。主目錄 Release 為 0 檔時，可跳過主前綴下載；例如 v19 的可讀檔案已在 Git，另取回最後一欄的 gzip 即可取得本列封存資料。這些 gzip 是原 `.pkl`／`.csv` 的無損封存；下載後仍可能需用 `research/project_archive_20260924/restore_data.py` 還原。完整步驟見 [資料指南](DATA_GUIDE.md)。實驗引用的共同資料不一定在本列，請依程式輸入和 [相依查核](../research/project_archive_20260924/PRESERVATION_NOTES.md) 一起取回。

數量按上傳計畫的逐檔清單算，不是 TAR 分包數；大小為檔案內容位元組換算（1 MiB=1,048,576 bytes），不含 TAR 標頭、暫存或解壓所需空間。同一包可能含其他檔案，因此實際下載流量可能較大。Git／主目錄 Release 兩欄每檔只計一次；最後一欄是 gzip 的實驗歸屬對照，**已包含在 `project_archive_20260924` 主目錄數字中，不能再加總**。0 檔表示該區域本身無 Release 檔案，不代表其運行完全沒有外部依賴。

## v4–v30 主線

| 實驗／區域與可讀入口 | Git 檔案 | 主目錄 Release 資料 | 主目錄下載前綴 | 另外取回的原 gzip 封存 |
|---|---:|---:|---|---|
| v4 官方排名評分<br>[報告](../research/jpx_official_ranking_20260912/JPX-official-ranking-report.md) · [計畫](../research/jpx_official_ranking_20260912/experiment_plan.md) | 18 檔 · 237.70 KiB | 10 檔 · 64.34 MiB | `jpx_official_ranking_20260912` | — |
| v5 一日價量變動率<br>[報告](../research/jpx_v5_daily_returns_20260912/JPX-v4-v5-comparison.md) · [計畫](../research/jpx_v5_daily_returns_20260912/experiment_plan.md) | 16 檔 · 156.00 KiB | 5 檔 · 38.94 MiB | `jpx_v5_daily_returns_20260912` | 1 檔 · 27.82 MiB<br>`project_archive_20260924/compressed_data/jpx_v5_daily_returns_20260912` |
| v6 RMSE 歷史重訪<br>[報告](../research/jpx_v6_expanding_daily_gd_20260912/JPX-v6-expanding-daily-RMSE-report.md) · [計畫](../research/jpx_v6_expanding_daily_gd_20260912/experiment_plan.md) | 16 檔 · 486.45 KiB | 7 檔 · 49.76 MiB | `jpx_v6_expanding_daily_gd_20260912` | — |
| v7 每日 MSE 正式基準<br>[報告](../research/jpx_v7_daily_mse_20260912/JPX-v7-daily-MSE-comparison.md) · [計畫](../research/jpx_v7_daily_mse_20260912/experiment_plan.md) | 19 檔 · 742.45 KiB | 16 檔 · 100.67 MiB | `jpx_v7_daily_mse_20260912` | — |
| v8 Soft rank<br>[報告](../research/jpx_v8_soft_rank_20260912/JPX-v8-same-stock-rank-loss-report.md) · [計畫](../research/jpx_v8_soft_rank_20260912/experiment_plan.md) | 31 檔 · 824.25 KiB | 31 檔 · 263.50 MiB | `jpx_v8_soft_rank_20260912` | 1 檔 · 319.22 MiB<br>`project_archive_20260924/compressed_data/jpx_v8_soft_rank_20260912` |
| v9 財報線性衰減<br>[報告](../research/jpx_v9_financial_decay_sgd_20260912/JPX-v9-financial-decay-SGD-report.md) · [計畫](../research/jpx_v9_financial_decay_sgd_20260912/experiment_plan.md) | 26 檔 · 338.60 KiB | 11 檔 · 214.66 MiB | `jpx_v9_financial_decay_sgd_20260912` | 4 檔 · 14.53 MiB<br>`project_archive_20260924/compressed_data/jpx_v9_financial_decay_sgd_20260912` |
| v10 MSE＋SGD<br>[報告](../research/jpx_v10_mse_sgd_financial_20260913/JPX-v10-MSE-SGD-financial-report.md) · [計畫](../research/jpx_v10_mse_sgd_financial_20260913/experiment_plan.md) | 27 檔 · 608.71 KiB | 21 檔 · 368.64 MiB | `jpx_v10_mse_sgd_financial_20260913` | — |
| v11 持續財報／22 日標準差<br>[報告](../research/jpx_v11_persistent_finance_std22_20260913/JPX-v11-persistent-finance-std22-report.md) · [計畫](../research/jpx_v11_persistent_finance_std22_20260913/experiment_plan.md) | 42 檔 · 991.82 KiB | 31 檔 · 716.49 MiB | `jpx_v11_persistent_finance_std22_20260913` | 2 檔 · 67.82 MiB<br>`project_archive_20260924/compressed_data/jpx_v11_persistent_finance_std22_20260913` |
| v12 六財報特徵<br>[報告](../research/jpx_v12_financial_growth_20260913/JPX-v12-six-financial-features-report.md) · [計畫](../research/jpx_v12_financial_growth_20260913/experiment_plan.md) | 55 檔 · 854.52 KiB | 39 檔 · 1.50 GiB | `jpx_v12_financial_growth_20260913` | 10 檔 · 28.35 MiB<br>`project_archive_20260924/compressed_data/jpx_v12_financial_growth_20260913` |
| v13 季度 Forecast／分塊 SGD<br>[報告](../research/jpx_v13_quarterly_sgd_sqrt_20260914/JPX-v13-quarterly-SGD-sqrt-report.md) · [計畫](../research/jpx_v13_quarterly_sgd_sqrt_20260914/experiment_plan.md) · [已知問題](../research/jpx_v13_quarterly_sgd_sqrt_20260914/KNOWN_ISSUES.md) | 84 檔 · 1.21 MiB | 2,503 檔 · 537.04 MiB | `jpx_v13_quarterly_sgd_sqrt_20260914` | 10 檔 · 205.73 MiB<br>`project_archive_20260924/compressed_data/jpx_v13_quarterly_sgd_sqrt_20260914` |
| v14 事件／訓練篩選<br>[報告](../research/jpx_v14_filtered_forecast_events_20260914/JPX-v14-filtered-forecast-report.md) · [計畫](../research/jpx_v14_filtered_forecast_events_20260914/experiment_plan.md) | 39 檔 · 1.03 MiB | 2,412 檔 · 570.09 MiB | `jpx_v14_filtered_forecast_events_20260914` | 9 檔 · 208.60 MiB<br>`project_archive_20260924/compressed_data/jpx_v14_filtered_forecast_events_20260914` |
| v15 Soft rank 財報對照<br>[報告](../research/jpx_v15_soft_rank_filtered_20260914/JPX-v15-soft-rank-report.md) · [計畫](../research/jpx_v15_soft_rank_filtered_20260914/experiment_plan.md) | 63 檔 · 1.45 MiB | 2,415 檔 · 553.62 MiB | `jpx_v15_soft_rank_filtered_20260914` | 4 檔 · 1.23 MiB<br>`project_archive_20260924/compressed_data/jpx_v15_soft_rank_filtered_20260914` |
| v16 單財報成分<br>[報告](../research/jpx_v16_single_financial_v7_20260915/JPX-v16-single-financial-report.md) · [計畫](../research/jpx_v16_single_financial_v7_20260915/experiment_plan.md) | 142 檔 · 4.55 MiB | 93 檔 · 684.14 MiB | `jpx_v16_single_financial_v7_20260915` | 1 檔 · 1.74 MiB<br>`project_archive_20260924/compressed_data/jpx_v16_single_financial_v7_20260915` |
| v17 財報修正公式<br>[報告](../research/jpx_v17_actual_forecast_revisions_20260915/JPX-v17-revised-financial-report.md) · [計畫](../research/jpx_v17_actual_forecast_revisions_20260915/experiment_plan.md) | 100 檔 · 2.52 MiB | 49 檔 · 353.71 MiB | `jpx_v17_actual_forecast_revisions_20260915` | 5 檔 · 173.13 MiB<br>`project_archive_20260924/compressed_data/jpx_v17_actual_forecast_revisions_20260915` |
| v18 固定 g／EPS<br>[報告](../research/jpx_v18_frozen_g_eps_20260915/JPX-v18-frozen-g-report.md) · [計畫](../research/jpx_v18_frozen_g_eps_20260915/experiment_plan.md) | 30 檔 · 1.14 MiB | 4 檔 · 63.46 MiB | `jpx_v18_frozen_g_eps_20260915` | — |
| v19 QoQ 預測 YoY<br>[報告](../research/jpx_v19_qoq_predict_yoy_20260915/JPX-v19-qoq-predict-yoy-report.md) | 21 檔 · 528.62 KiB | 0 檔 | `jpx_v19_qoq_predict_yoy_20260915` | 4 檔 · 8.53 MiB<br>`project_archive_20260924/compressed_data/jpx_v19_qoq_predict_yoy_20260915` |
| v20 EPS 資料查核<br>[報告](../research/jpx_v20_eps_data_audit_20260915/JPX-v20-eps-data-audit-report.md) | 28 檔 · 301.36 KiB | 1 檔 · 9.97 KiB | `jpx_v20_eps_data_audit_20260915` | — |
| v21 全部 EPS 共同訓練<br>[報告](../research/jpx_v21_all_eps_joint_20260916/JPX-v21-all-eps-joint-report.md) · [計畫](../research/jpx_v21_all_eps_joint_20260916/experiment_plan.md) | 26 檔 · 488.61 KiB | 10 檔 · 67.10 MiB | `jpx_v21_all_eps_joint_20260916` | — |
| v22 EPS 門檻<br>[報告](../research/jpx_v22_eps_thresholds_20260916/JPX-v22-threshold-report.md) · [計畫](../research/jpx_v22_eps_thresholds_20260916/experiment_plan.md) | 27 檔 · 677.87 KiB | 10 檔 · 68.06 MiB | `jpx_v22_eps_thresholds_20260916` | — |
| v23 純 g 訓練門檻<br>[報告](../research/jpx_v23_g_thresholds_20260916/JPX-v23-g-threshold-report.md) · [計畫](../research/jpx_v23_g_thresholds_20260916/experiment_plan.md) | 27 檔 · 673.27 KiB | 9 檔 · 66.11 MiB | `jpx_v23_g_thresholds_20260916` | 1 檔 · 489.98 KiB<br>`project_archive_20260924/compressed_data/jpx_v23_g_thresholds_20260916` |
| v24 訓練／驗證同步篩選<br>[報告](../research/jpx_v24_g_train_validation_filters_20260916/JPX-v24-train-validation-filter-report.md) · [計畫](../research/jpx_v24_g_train_validation_filters_20260916/experiment_plan.md) | 30 檔 · 927.50 KiB | 5 檔 · 16.64 MiB | `jpx_v24_g_train_validation_filters_20260916` | 1 檔 · 497.38 KiB<br>`project_archive_20260924/compressed_data/jpx_v24_g_train_validation_filters_20260916` |
| v25 單項無篩選重做<br>[報告](../research/jpx_v25_single_features_unfiltered_20260916/JPX-v25-unfiltered-single-features-report.md) · [計畫](../research/jpx_v25_single_features_unfiltered_20260916/experiment_plan.md) | 108 檔 · 3.16 MiB | 63 檔 · 464.20 MiB | `jpx_v25_single_features_unfiltered_20260916` | 1 檔 · 1.24 MiB<br>`project_archive_20260924/compressed_data/jpx_v25_single_features_unfiltered_20260916` |
| v26 營業利益 Forecast 修正<br>[報告](../research/jpx_v26_profit_forecast_revision_20260916/JPX-v26-profit-forecast-revision-report.md) · [計畫](../research/jpx_v26_profit_forecast_revision_20260916/experiment_plan.md) | 23 檔 · 195.02 KiB | 4 檔 · 22.80 MiB | `jpx_v26_profit_forecast_revision_20260916` | — |
| v27 Forecast 最新狀態<br>[報告](../research/jpx_v27_profit_forecast_state_20260916/JPX-v27-forecast-state-report.md) · [計畫](../research/jpx_v27_profit_forecast_state_20260916/experiment_plan.md) | 34 檔 · 450.20 KiB | 8 檔 · 45.49 MiB | `jpx_v27_profit_forecast_state_20260916` | 3 檔 · 26.60 MiB<br>`project_archive_20260924/compressed_data/jpx_v27_profit_forecast_state_20260916` |
| v28 共同學習率<br>[報告](../research/jpx_v28_common_learning_rate_20260916/JPX-v28-common-learning-rate-report.md) · [計畫](../research/jpx_v28_common_learning_rate_20260916/experiment_plan.md) | 32 檔 · 643.20 KiB | 10 檔 · 67.20 MiB | `jpx_v28_common_learning_rate_20260916` | — |
| v29 學習率倍率<br>[報告](../research/jpx_v29_learning_rate_sweep_20260916/JPX-v29-learning-rate-sweep-report.md) · [計畫](../research/jpx_v29_learning_rate_sweep_20260916/experiment_plan.md) | 83 檔 · 2.37 MiB | 45 檔 · 331.52 MiB | `jpx_v29_learning_rate_sweep_20260916` | 1 檔 · 949.78 KiB<br>`project_archive_20260924/compressed_data/jpx_v29_learning_rate_sweep_20260916` |
| v30 實際／Forecast 七組<br>[報告](../research/jpx_v30_profit_actual_forecast_20260916/JPX-v30-profit-actual-forecast-report.md) · [計畫](../research/jpx_v30_profit_actual_forecast_20260916/experiment_plan.md) | 65 檔 · 1.97 MiB | 15 檔 · 151.34 MiB | `jpx_v30_profit_actual_forecast_20260916` | 2 檔 · 1.04 MiB<br>`project_archive_20260924/compressed_data/jpx_v30_profit_actual_forecast_20260916` |

## 早期策略與 ARIMA 支線

| 實驗／區域與可讀入口 | Git 檔案 | 主目錄 Release 資料 | 主目錄下載前綴 | 另外取回的原 gzip 封存 |
|---|---:|---:|---|---|
| 累積訓練與門檻選擇<br>[報告](../research/jpx_expanding_threshold_20260910/JPX-expanding-threshold-report.md) · [計畫](../research/jpx_expanding_threshold_20260910/experiment_plan.md) | 40 檔 · 2.79 MiB | 11 檔 · 6.39 MiB | `jpx_expanding_threshold_20260910` | 5 檔 · 2.30 MiB<br>`project_archive_20260924/compressed_data/jpx_expanding_threshold_20260910` |
| ±10% 中性帶<br>[報告](../research/jpx_neutral_10pct_20260910/JPX-neutral-10pct-report.md) | 18 檔 · 486.24 KiB | 5 檔 · 2.32 MiB | `jpx_neutral_10pct_20260910` | 3 檔 · 2.17 MiB<br>`project_archive_20260924/compressed_data/jpx_neutral_10pct_20260910` |
| 正負門檻平滑<br>[報告](../research/jpx_signed_band_20260910/JPX-signed-band-report.md) · [計畫](../research/jpx_signed_band_20260910/experiment_plan.md) | 17 檔 · 410.08 KiB | 7 檔 · 3.19 MiB | `jpx_signed_band_20260910` | 4 檔 · 3.04 MiB<br>`project_archive_20260924/compressed_data/jpx_signed_band_20260910` |
| 原始 return 特徵<br>[報告](../research/jpx_stock_returns_20260910/JPX-stock-returns-report.md) · [計畫](../research/jpx_stock_returns_20260910/experiment_plan.md) | 16 檔 · 332.78 KiB | 2 檔 · 1.53 MiB | `jpx_stock_returns_20260910` | 3 檔 · 436.31 MiB<br>`project_archive_20260924/compressed_data/jpx_stock_returns_20260910` |
| 交易成本敏感度<br>[報告](../research/jpx_costs_top5_20260911/JPX-top5-cost-report.md) · [計畫](../research/jpx_costs_top5_20260911/experiment_plan.md) | 27 檔 · 2.34 MiB | 30 檔 · 8.76 MiB | `jpx_costs_top5_20260911` | 40 檔 · 46.69 MiB<br>`project_archive_20260924/compressed_data/jpx_costs_top5_20260911` |
| 早期模型比較<br>[報告](../research/jpx_model_comparison_20260911/JPX-model-comparison.md) | 3 檔 · 37.35 KiB | 0 檔 | `jpx_model_comparison_20260911` | — |
| return／均值比值<br>[報告](../research/jpx_return_mean_20260911/JPX-return-mean-report.md) · [計畫](../research/jpx_return_mean_20260911/experiment_plan.md) | 12 檔 · 58.79 KiB | 3 檔 · 1.80 MiB | `jpx_return_mean_20260911` | 3 檔 · 419.25 MiB<br>`project_archive_20260924/compressed_data/jpx_return_mean_20260911` |
| return T 特徵<br>[報告](../research/jpx_return_t_20260911/JPX-return-T-report.md) · [計畫](../research/jpx_return_t_20260911/experiment_plan.md) | 12 檔 · 62.33 KiB | 3 檔 · 1.79 MiB | `jpx_return_t_20260911` | 2 檔 · 549.62 MiB<br>`project_archive_20260924/compressed_data/jpx_return_t_20260911` |
| 前十名等權<br>[報告](../research/jpx_return_t_top10_20260911/JPX-top10-equal-report.md) · [計畫](../research/jpx_return_t_top10_20260911/experiment_plan.md) | 9 檔 · 32.30 KiB | 1 檔 · 279.50 KiB | `jpx_return_t_top10_20260911` | 3 檔 · 3.77 MiB<br>`project_archive_20260924/compressed_data/jpx_return_t_top10_20260911` |
| 十組 Ridge 對照<br>[報告](../research/jpx_ridge_comparison_20260911/JPX-Ridge-comparison-report.md) · [計畫](../research/jpx_ridge_comparison_20260911/experiment_plan.md) | 39 檔 · 2.84 MiB | 30 檔 · 22.17 MiB | `jpx_ridge_comparison_20260911` | 12 檔 · 9.35 MiB<br>`project_archive_20260924/compressed_data/jpx_ridge_comparison_20260911` |
| 前十名 Ridge<br>[報告](../research/jpx_ridge_top10_20260911/JPX-Ridge-report.md) · [計畫](../research/jpx_ridge_top10_20260911/experiment_plan.md) | 11 檔 · 71.44 KiB | 1 檔 · 282.68 KiB | `jpx_ridge_top10_20260911` | 3 檔 · 3.71 MiB<br>`project_archive_20260924/compressed_data/jpx_ridge_top10_20260911` |
| 個股 expanding<br>[報告](../research/jpx_stock_expanding_20260911/JPX-stock-expanding-report.md) · [計畫](../research/jpx_stock_expanding_20260911/experiment_plan.md) | 8 檔 · 293.61 KiB | 2 檔 · 1.54 MiB | `jpx_stock_expanding_20260911` | 1 檔 · 806.96 KiB<br>`project_archive_20260924/compressed_data/jpx_stock_expanding_20260911` |
| ARIMA／ACF（未正式回測）<br>[未完成計畫](../research/jpx_arima_acf_learning_20260923/experiment_plan.md) | 7 檔 · 12.45 KiB | 0 檔 | `jpx_arima_acf_learning_20260923` | — |

## 文書、網站、素材與保存紀錄

| 實驗／區域與可讀入口 | Git 檔案 | 主目錄 Release 資料 | 主目錄下載前綴 | 另外取回的原 gzip 封存 |
|---|---:|---:|---|---|
| 原專案根目錄<br>[JPX 現況](../research/JPX-current-baseline.md) · [大銀研究](../research/大銀微系統完整研究報告_2026-08-26.md) · [企劃書](../research/機器人與自動化股票研究企劃書.docx) | 12 檔 · 195.05 KiB | 2 檔 · 5.88 MiB | `大銀微系統_產品與應用產業_外行人版.pptx`<br>`大銀微系統_產品與應用產業_外行人版.pptx.inspect.ndjson` | — |
| 圖表素材 3nItKd<br>[圖表資料快照](../research/.chart-data-3nItKd/chart-data-snapshot.json) | 1 檔 · 1.23 KiB | 1 檔 · 5.60 MiB | `.chart-data-3nItKd` | — |
| 圖表素材 85FQ0i<br>[圖表資料快照](../research/.chart-data-85FQ0i/chart-data-snapshot.json) | 1 檔 · 1.23 KiB | 1 檔 · 5.60 MiB | `.chart-data-85FQ0i` | — |
| 圖表素材 8fpw1a<br>[圖表資料快照](../research/.chart-data-8fpw1a/chart-data-snapshot.json) | 1 檔 · 1.23 KiB | 1 檔 · 5.91 MiB | `.chart-data-8fpw1a` | — |
| 圖表素材 A7NyGy<br>[圖表資料快照](../research/.chart-data-A7NyGy/chart-data-snapshot.json) | 1 檔 · 1.23 KiB | 1 檔 · 5.91 MiB | `.chart-data-A7NyGy` | — |
| 圖表素材 aIjGXv<br>[圖表資料快照](../research/.chart-data-aIjGXv/chart-data-snapshot.json) | 1 檔 · 1.23 KiB | 1 檔 · 5.91 MiB | `.chart-data-aIjGXv` | — |
| 圖表素材 bb22bt<br>[圖表資料快照](../research/.chart-data-bb22bt/chart-data-snapshot.json) | 1 檔 · 1.23 KiB | 1 檔 · 5.60 MiB | `.chart-data-bb22bt` | — |
| 其他產品規劃<br>[排班產品規劃](../research/docs/ai-scheduling-app-product-plan.md) | 1 檔 · 48.37 KiB | 0 檔 | `docs` | — |
| 大銀申請網站<br>[網頁程式](../research/hiwin-application-site/app/page.tsx) · [套件設定](../research/hiwin-application-site/package.json) | 65 檔 · 1.46 MiB | 4 檔 · 542 B | `hiwin-application-site` | — |
| 最終成品／交付包<br>[v14 報告](../research/output/jpx_mobile_data_20260914/JPX-v14-filtered-forecast-report.md) · [v15 報告](../research/output/jpx_mobile_data_20260914/JPX-v15-soft-rank-report.md) · [成品目錄](../research/output/) | 49 檔 · 3.70 MiB | 29 檔 · 73.28 MiB | `output` | — |
| 簡報建置、模板與素材<br>[版面／建置紀錄](../research/ppt_build/) · [字型修正核對](../research/ppt_build/hiwin_industry_research_2026q2/validation-receipt-mac-fontfix.json) | 215 檔 · 17.90 MiB | 63 檔 · 68.92 MiB | `ppt_build` | — |
| 原無損封存與資料來源<br>[續接記憶](../research/project_archive_20260924/MEMORY_BRIEF.md) · [封存總覽](../research/project_archive_20260924/README.md) | 20 檔 · 8.72 MiB | 144 檔 · 2.74 GiB | `project_archive_20260924` | 本列主目錄已含 139 檔 gzip；其實驗歸屬另列於上方各實驗對照。 |
| 企劃書 QA<br>[企劃書 PDF](../research/qa_robotics_proposal/機器人與自動化股票研究企劃書.pdf) · [逐頁 QA](../research/qa_robotics_proposal/) | 17 檔 · 1.16 MiB | 1 檔 · 241.12 KiB | `qa_robotics_proposal` | — |
| 企劃書 QA final<br>[企劃書 PDF](../research/qa_robotics_proposal_final/機器人與自動化股票研究企劃書.pdf) · [逐頁 QA](../research/qa_robotics_proposal_final/) | 4 檔 · 1.90 MiB | 15 檔 · 6.13 MiB | `qa_robotics_proposal_final` | — |
| 企劃書 QA final v2<br>[企劃書 PDF](../research/qa_robotics_proposal_final_v2/機器人與自動化股票研究企劃書.pdf) · [逐頁 QA](../research/qa_robotics_proposal_final_v2/) | 4 檔 · 1.92 MiB | 15 檔 · 6.10 MiB | `qa_robotics_proposal_final_v2` | — |
| 企劃書 QA final v3<br>[企劃書 PDF](../research/qa_robotics_proposal_final_v3/機器人與自動化股票研究企劃書.pdf) · [逐頁 QA](../research/qa_robotics_proposal_final_v3/) | 4 檔 · 1.93 MiB | 15 檔 · 6.08 MiB | `qa_robotics_proposal_final_v3` | — |
| 企劃書 QA final v4<br>[企劃書 PDF](../research/qa_robotics_proposal_final_v4/機器人與自動化股票研究企劃書.pdf) · [逐頁 QA](../research/qa_robotics_proposal_final_v4/) | 4 檔 · 1.88 MiB | 15 檔 · 6.10 MiB | `qa_robotics_proposal_final_v4` | — |
| 企劃書 QA final v5<br>[企劃書 PDF](../research/qa_robotics_proposal_final_v5/機器人與自動化股票研究企劃書.pdf) · [逐頁 QA](../research/qa_robotics_proposal_final_v5/) | 4 檔 · 1.86 MiB | 15 檔 · 6.10 MiB | `qa_robotics_proposal_final_v5` | — |
| 企劃書 QA final v6<br>[企劃書 PDF](../research/qa_robotics_proposal_final_v6/機器人與自動化股票研究企劃書.pdf) · [逐頁 QA](../research/qa_robotics_proposal_final_v6/) | 3 檔 · 1.83 MiB | 15 檔 · 6.10 MiB | `qa_robotics_proposal_final_v6` | — |
| 企劃書 QA v2<br>[企劃書 PDF](../research/qa_robotics_proposal_v2/機器人與自動化股票研究企劃書.pdf) · [逐頁 QA](../research/qa_robotics_proposal_v2/) | 17 檔 · 1.14 MiB | 1 檔 · 249.50 KiB | `qa_robotics_proposal_v2` | — |
| 企劃書 QA v3<br>[企劃書 PDF](../research/qa_robotics_proposal_v3/機器人與自動化股票研究企劃書.pdf) · [逐頁 QA](../research/qa_robotics_proposal_v3/) | 17 檔 · 1.14 MiB | 1 檔 · 249.50 KiB | `qa_robotics_proposal_v3` | — |
| 企劃書 QA v4<br>[企劃書 PDF](../research/qa_robotics_proposal_v4/機器人與自動化股票研究企劃書.pdf) · [逐頁 QA](../research/qa_robotics_proposal_v4/) | 4 檔 · 1.98 MiB | 16 檔 · 6.10 MiB | `qa_robotics_proposal_v4` | — |
| 排班工具<br>[工具說明](../research/scheduling-app/README.md) | 4 檔 · 90.84 KiB | 0 檔 | `scheduling-app` | — |
| 歷史 PDF／圖表查核<br>[PDF／圖表查核](../research/tmp/) | 37 檔 · 5.82 MiB | 9 檔 · 2.80 MiB | `tmp` | — |

## 使用時要留意

- 每份可讀報告與計畫可直接由 GitHub 開啟；較大的簡報、PDF、原生函式庫、圖表或資料，依表中前綴從 Release 取回，不能以 GitHub 目錄畫面判斷它們遺失。
- 原封存資料中還有 `project_archive_20260924/external_sources`：JPX 原始資料 ZIP、早期 regime 實驗 ZIP 及簡報模板。只需要來源時可用此完整前綴，不必下載整個封存區。
- `project_archive_20260924` 的前綴會包含所有大型 gzip；磁碟有限時，改用各實驗列的 gzip 前綴。v30 等模型仍有跨版本依賴，不會因它是最新版就變成獨立套件。
- `.chart-data-*`、`qa_*` 與 `tmp` 保留的是歷史素材／核對產物，不能僅按名稱當成垃圾；不同版本的成品與修正證據均分開定位。
- 原電腦絕對路徑、套件版本、macOS 函式庫與字型可能需另做適配。本表只負責找資料，沒有重跑回測或驗證跨平台重建。
- 空的 `sources/` 不會出現在檔案表；唯讀規則仍適用。排除項與對話／瀏覽器資料範圍見 [上傳範圍](UPLOAD_SCOPE.md)。
