# 研究地圖

先用這份索引定位，再讀原始報告與資料。`research/` 保持原專案相對路徑；大型檔案與二進位成品可能須透過 [資料指南](DATA_GUIDE.md) 從 Release 取回。下列路徑是研究定位，不表示 clone 後每個檔案都已在本機。

## JPX 主線

| 範圍 | 問題與進度 | 原檔入口 |
|---|---|---|
| 現行狀態 | 正式 v7、最新 v32、選模規則與候選區分 | [研究摘要](../research/JPX-current-baseline.md)、[預設](../research/JPX-experiment-defaults.json)、[版本登記](../research/jpx_model_versions.json) |
| 全部沿革 | 早期策略、v4–v30、失敗／修正及參數摘要 | [實驗歷程](../research/project_archive_20260924/EXPERIMENT_HISTORY.md) |
| 早期策略 | 市場 regime、中性帶、return T、配置及 Ridge | [模型比較](../research/jpx_model_comparison_20260911/JPX-model-comparison.md) |
| 交易成本 | 毛利是否能承擔換手與借券費 | [成本試算](../research/jpx_costs_top5_20260911/JPX-top5-cost-report.md) |
| v4–v7 | 官方評分、單日價量特徵、RMSE 重訪到每日 MSE | [正式 v7 報告](../research/jpx_v7_daily_mse_20260912/JPX-v7-daily-MSE-comparison.md) |
| v8–v15 | Soft rank、SGD、財報訊號與季度 Forecast | [v13 報告](../research/jpx_v13_quarterly_sgd_sqrt_20260914/JPX-v13-quarterly-SGD-sqrt-report.md)、[v15 報告](../research/jpx_v15_soft_rank_filtered_20260914/JPX-v15-soft-rank-report.md) |
| v16–v25 | 單一成分、EPS、固定 g、資料查核與極端值門檻 | [v20 資料查核](../research/jpx_v20_eps_data_audit_20260915/JPX-v20-eps-data-audit-report.md)、[v25 無篩選重做](../research/jpx_v25_single_features_unfiltered_20260916/JPX-v25-unfiltered-single-features-report.md) |
| v26–v30 | 營業利益修正、最新 Forecast 狀態、共同步長與實際值 | [v29 步長比較](../research/jpx_v29_learning_rate_sweep_20260916/JPX-v29-learning-rate-sweep-report.md)、[v30 七組對照](../research/jpx_v30_profit_actual_forecast_20260916/JPX-v30-profit-actual-forecast-report.md) |
| ARIMA 學習 | 後續七輪已回測：25 組網格、殘差 ACF 與 Target MSE 共同訓練；test 保留 | [9/28 完整研究](../research/jpx_arima_learning_20260928/README.md)、[9/23 舊計畫（歷史）](../research/jpx_arima_acf_learning_20260923/experiment_plan.md) |

## 不能省略的研究證據

| 類別 | 位置與用途 |
|---|---|
| 財年修正前 | `research/jpx_v12_financial_growth_20260913/pre_period_fix/`：保留初次錯配期間的運算，現行報告已用修正後結果。 |
| 溢位失敗版 | `research/jpx_v13_quarterly_sgd_sqrt_20260914/joint_step_overflow/`：保留暖身溢位的原設定與程式。 |
| 財報口徑問題 | [v13 已知問題](../research/jpx_v13_quarterly_sgd_sqrt_20260914/KNOWN_ISSUES.md)、[完整診斷](../research/jpx_v13_quarterly_sgd_sqrt_20260914/diagnostics_20260914/JPX-v13-diagnostic-report.md)。 |
| 已替代修正公式 | `research/jpx_v17_actual_forecast_revisions_20260915/superseded_absolute_revision/`：保留公式演變。 |
| 逐步過程 | 各實驗的 `training_updates`、`parameter_history`、NPZ traces／predictions、逐年 ranks、selected stocks 及財報事件。摘要和最終係數不能取代。 |
| 數據來源 | `research/project_archive_20260924/external_sources/`：JPX 原始資料 ZIP、早期 regime 實驗 ZIP、簡報模板；內容需按 catalog 下載。 |
| 相依與環境 | [保存注意事項](../research/project_archive_20260924/PRESERVATION_NOTES.md)：共同特徵、絕對路徑、套件與平台限制。 |

## 產業研究與最終成品

報告、成品、建置程式與素材都需保留。部分舊文書／簡報 helper 路徑已失效；即使有原始碼，也不能假定隨時可以重建相同版面。二進位成品的確切收錄與下載位置，以 catalog 為準。

| 主題 | 原專案路徑（位於 `research/` 下） |
|---|---|
| 大銀微系統完整文字研究 | `大銀微系統完整研究報告_2026-08-26.md` |
| 產品／應用入門簡報 | `大銀微系統_產品與應用產業_外行人版.pptx` |
| 機器人與自動化研究企劃 | `機器人與自動化股票研究企劃書.docx`；成品 PDF 與逐頁 QA 位於 `qa_robotics_proposal_final_v6/`。 |
| 機器人研究簡報 | `output/大銀微系統_機器人與人形機器人_完整研究簡報_2026_Mac相容版.pptx`，另保留既有較短版。 |
| 產業連結／營收拆解簡報 | `output/大銀微系統_產業連結與營收拆解_研究簡報_2026Q2_Mac字型修正版.pptx`，原版另存。 |
| JPX 行動閱讀 PDF／交付包 | `output/pdf/`、`output/JPX-*.zip`、`output/jpx_mobile_data_20260914/`；屬交付形式，不能取代完整實驗目錄。 |
| 建置、圖表、素材與 QA | `ppt_build/`、各實驗目錄、`qa_*`、`tmp/pdfs/`；是否收錄各中間檔以 catalog 為準，不自行刪除唯一成品或修正版。 |

## 網站與其他專案

- `research/hiwin-application-site/`：網站程式、public 素材、設定與套件鎖檔。套件可安裝不代表服務端部署、帳號及遠端 CDN 已備份。
- [排班工具](../research/scheduling-app/README.md) 與 [產品規劃](../research/docs/ai-scheduling-app-product-plan.md)：瀏覽器 `localStorage` 的實際使用者資料不在專案檔案中，除非另行匯出，不能算已保存。
- `research/sources/`：唯讀同步參考材料；快照只記錄當時實際存在的內容。

## 如何判斷已存好

研究數值的歷史 audit、9/24 本機 gzip 完整性檢查，以及 GitHub Release 上傳驗證是三種不同證據。先看 [快照索引](../data/index.json)、對應 catalog 和 `data/verification/<tag>.json`；不要把其中一種當成全部完成。此儲存庫也不等於未匯出對話的完整備份。

2026-09-25 新增：[v31 每檔獨立參數 MSE](../research/jpx_v31_stock_specific_mse_20260925/JPX-v31-stock-specific-mse-report.md)。正式基準仍 v7；最新進度以本導讀及版本登記為準，9/24 歷史摘要原樣保留。

2026-09-26 新增：[v32 逐檔歷史分塊複習](../research/jpx_v32_chunk_replay_20260926/JPX-v32-chunk-replay-report.md)。每fold訓練一次，validation整年固定；正式基準維持v7。

2026-09-28 新增：[PCA 市場與持倉風險診斷](../research/jpx_pca_risk_20260928/JPX_PCA_report.md)，以及 [PCA 分數消融 A/B](../research/jpx_pca_score_ab_20260928/JPX_PCA_AB_report.md)。A/B 沿用相同 JPX 排名配重；953 日官方未年化 Sharpe：A +0.00767887、每期 PC6 消融 +0.00459482、每期最大估計風險方向消融 +0.02620535。兩個差異的探索區間皆含 0，正式基準維持 v7；原報告同一 PC6 另有單日核對，不能當成長期證據。大型逐筆資料 Release 尚待發布，詳見該實驗 sync_status.json；不能因報告已在 Git 就視為全部封存完成。


2026-09-29 新增：[2018 隨機 PC 對照](../research/jpx_pca_random_2018_20260929/REPORT.md)。原 245 日時序、普通與等幅隨機方向各 1,000 條路徑；B_MAX Sharpe 0.075797，高於兩組所有路徑（各組最高 0.009089／0.025079）。等幅對照平均每日相對 A 換入 51.10 檔，B_MAX 51.42 檔；全年累積 spread 差的 69.5% 集中在 2018 年 4 月。這是已知 validation 的機制證據，不是新 holdout 或正式 p 值；test 未使用、正式基準未更換。程式、小型結果及逐路徑年度摘要保存於 Git；247 個大型逐筆檔共 1,254,437,351 bytes 尚待版本化 Release，詳見該實驗 sync_status.json。

2026-09-29 新增：[2018 年 4 月每日最大 PC 持倉風險占比分布](../research/jpx_pca_april_distribution_20260929/REPORT.md)。原模型 A、PCA 涵蓋持倉範圍內的估計變異：平均 32.22%，中位數 30.19%，20 日中 10 日嚴格超過使用者提出的 30% 門檻；範圍 8.05%–73.56%。只做既有狀態描述，未比較其他月份、未套用切換策略或開啟 test；程式、每日 CSV、摘要與圖表保存在該目錄。來源大型 NPZ 的 Release 待發布狀態仍沿原記錄。
