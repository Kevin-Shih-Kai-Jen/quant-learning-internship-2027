# JPX 實驗歷程與續接索引

整理日期：2026-09-24。這份摘要根據本地報告、計畫與設定製作，濃縮研究記憶，不取代逐筆資料、程式、參數或稽核紀錄。本次依使用者授權，將大型 CSV／PKL 無損封存至 `compressed_data/<原相對路徑>.gz`；每檔解壓內容經 SHA-256 與原檔一致核對後，才移除原位置副本，以釋放空間。資料的完整內容及研究過程保留，報告、程式及既有 NPZ／CSV.GZ 維持原狀；大型資料不一定仍能從原路徑直接讀取。重跑前請先讀 [還原指南](RESTORE_GUIDE.md)，按需恢復。

下文「核對通過」指原報告記載的實驗核對結果，本次沒有重新執行大型回測；封存的雜湊核對另見還原指南與封存紀錄。所有連結均相對於本文件，歷史報告內的原始資料路徑可能需先還原。

## 續接時先讀這些

- [現行研究摘要](../JPX-current-baseline.md)、[現行預設](../JPX-experiment-defaults.json)、[正式基準設定](../jpx_active_baseline.json)、[版本登記](../jpx_model_versions.json)。版本登記最新完成實驗是 v30；另有 9/23 ARIMA 學習支線，尚未完成正式回測。
- 正式基準仍為 **v7_equal**。v5 曾有較高的歷史分數，但使用者選擇 v7 的每日更新規則；後來的實驗候選不能自行視為正式基準。
- 最新登記的 validation 候選為 **v30/frozen_r**，Sharpe **+0.00435804**、Rank IC **−0.00099064**，低於原 v7 Sharpe **+0.00767887**；尚未做 test。
- 2026-09-16 最終規則：訓練與驗證均不依財報極端值排除股票日、不裁切；驗證預設完整股票池。**Sharpe 第一、Rank IC 第二；Rank IC 不必為正；MSE 只作診斷、不選模型，但普通 MSE 仍是既有訓練 loss。** 使用全部既有 validation 選候選，test 留待最後總體評估。保留缺值、Target 到期與財報事件有效性處理；EPS 算法維持舊定義。
- 9/16 前曾採「Sharpe 高於 g 且 Rank IC>0」判準，僅是當時判準；v29 已明確替換，續接不能沿用舊硬門檻。

## 基準、資料與評分口徑

v7 為全股票共用線性模型，11 個特徵加截距：`T5,T22,T60,V5,V22,V60,P5xV5,P22xV22,P60xV60,PR1,VR1`。PR1/VR1 為一日價量變動率原值，不標準化、不截尾；價格／量使用截至前一列的 AdjustmentFactor 累積因子調整。2017-01-04 零初始化一次，2017 年暖身，跨年接續。當某訊號日 Target 在 ExitDate 收盤到期後，用當前參數重算該日誤差，全體已知標籤股票等權普通 MSE，只更新一次，不重訪歷史日。缺 Target 只排該筆訓練誤差，不填零；缺輸入依既有規則補零並記錄。`A=XᵀWX`、`η=1/(2λmax(A))`，W 歸一化。總計 1,199 次更新、2,325,806 筆有效訓練誤差；末次參數截至 2021-12-03，未回填早期預測。

2018–2021 共 953 個既有歷史驗證日，最後訊號日 2021-12-01。核心股票池排除當日 SupervisionFlag；每日預測排序，多空各 200 檔，每側最極端排名權重 2，線性降至 1。官方 daily spread 是兩側加權 Target 和之差除以平均權重 1.5；Sharpe 為其平均／樣本標準差，**不年化**。Daily spread 不是帳戶報酬，不能直接複利。以下 v4–v30 表中的 Sharpe 均採此尺度，除特別說明外未扣成本、未使用新保留 test，且 validation 已反覆用於開發。

v7_jpx 參照版只在原訊號時排名前後各 200 檔使用正的 2→1 誤差權重，其餘零，權重等 Target 到期也不重選；Sharpe **−0.04018310**。其訓練 loss 是研究自訂，不能稱為官方指定 loss。詳見 [v7 報告](../jpx_v7_daily_mse_20260912/JPX-v7-daily-MSE-comparison.md) 與 [事前計畫](../jpx_v7_daily_mse_20260912/experiment_plan.md)。

## 9/10–9/11：早期交易策略與成本研究

這一階段用逐筆持倉模擬引擎、目標價觸及／收盤退出、停牌延後退出與錯號資金轉移；每側前三名通常按 50/30/20 分配。其累積帳戶報酬及年化 Sharpe **不可與後來官方未年化評分混為同一數字**。同期間為 2018-01-04 進場至 2021-12-03 退出，共 953 區間。

| 研究 | 關鍵設定與保留結論 | 原報告 |
|---|---|---|
| 原始市場 regime／v3 | 自建樣本市值加權價格指數，不是官方日經或期貨；g 正負切換初始 70/30、30/70。動態累積 +54.06%、回撤 22.22%；固定 50/50 為 +27.43%、25.51%；日經 +19.24%、31.80%。市場 g 四年 MSE 均比預測零差，交易改善不等於預測已準確。 | [原始報告](../jpx_model_comparison_20260911/original-market-regime-report.md) |
| ±10% 中性 | 953 日全中性，等同 v3 固定初始 50/50；+27.43%。原 ZIP 重播一致，這不是一般中性門檻失效的證明。 | [報告](../jpx_neutral_10pct_20260910/JPX-neutral-10pct-report.md)、[ZIP 重查](../jpx_neutral_10pct_20260910/zip_verification/JPX-ZIP-recheck-report.md) |
| Expanding 門檻 | 歷史預測絕對值分位數 25/50/75%，先內部前半校準、後半評分。2019 選 75%，驗證 +10.05%、回撤 21.53%，未勝日經；2020/2021 沒合格候選，未自行填現金或拼接全期結果。 | [報告](../jpx_expanding_threshold_20260910/JPX-expanding-threshold-report.md) |
| 正負各 10%＋60/40 平滑 | 歷史正負訊號分別取近零 10%；當期原始邊界 0.6＋去年原始邊界 0.4，初始 ±0.1%；+34.78%、回撤 23.06%，相同 expanding 無中性為 +24.35%、29.34%。全期勝日經、逐年只 2018 同時達標。 | [報告](../jpx_signed_band_20260910/JPX-signed-band-report.md) |
| 原始 return | 5/22/60 日價量 return 及交乘，上一年訓練；−6.56%、回撤 41.00%。改個股 expanding 後 −41.47%、47.52%。 | [return](../jpx_stock_returns_20260910/JPX-stock-returns-report.md)、[expanding](../jpx_stock_expanding_20260911/JPX-stock-expanding-report.md) |
| return／前 22 日平均−1 | 加原始 PR1；−52.77%、回撤 77.58%。7638 的精確零分母被浮點誤差誤認非零，已依原規則修正；真實非零小分母保留，仍有百萬級特徵。 | [報告](../jpx_return_mean_20260911/JPX-return-mean-report.md) |
| return T／前十等權 | 前 22 日不含當日之 z-score（ddof=1），含 1 日價格 T：+47.39%、回撤 36.09%，未同時勝日經；改各側前十等權後 −20.52%、32.12%。比較索引註明使用修正休市日問題後數字。 | [return T](../jpx_return_t_20260911/JPX-return-T-report.md)、[前十](../jpx_return_t_top10_20260911/JPX-top10-equal-report.md) |
| Ridge | `mean(error²)+λΣβ²`，截距不罰；網格 0/0.001/0.01/0.1/1/10/100，由訓練年內時間切分 MSE 選。前十版四年均 λ=100，−36.93%、回撤 58.34%；另 10 組成對比較回撤全部增加。係數變小不保證持倉風險下降。 | [前十 Ridge](../jpx_ridge_top10_20260911/JPX-Ridge-report.md)、[十組](../jpx_ridge_comparison_20260911/JPX-Ridge-comparison-report.md) |
| 成本敏感度 | 前五個無 Ridge 策略按已見歷史排序，不是獨立選策。原動態／收盤退出毛報酬 +88.11%，低／中／高成本變 +4.45%／−73.38%／−96.48%。每邊總費率 3/10/20bp，借券年率 1/3/10%；尚非實盤成交模擬。 | [成本報告](../jpx_costs_top5_20260911/JPX-top5-cost-report.md) |

完整橫向索引見 [模型比較](../jpx_model_comparison_20260911/JPX-model-comparison.md)。多項設定同時變動的比較不能作單一因素因果結論。

## 9/12–9/16：v4–v30 研究主線

| 日期／版本 | 變更與結果（Sharpe 未年化） | 詳細證據 |
|---|---|---|
| 9/12 v4 | 9 個價量水準 T 特徵、上一年重估；改採官方多空各 200 名評分：**+0.01749933**。移除 regime、錯號移轉、目標價出場等交易規則。 | [報告](../jpx_official_ranking_20260912/JPX-official-ranking-report.md) |
| 9/12 v5 | 加原值 PR1/VR1，全部 12 參數重估：**+0.04674195**，四年均優於 v4。這是歷史參考，不是目前正式基準。 | [報告](../jpx_v5_daily_returns_20260912/JPX-v4-v5-comparison.md) |
| 9/12 v6 | 連續參數、每次擴窗重訪全部已成熟歷史；各日已固定 400 檔 RMSE 一步，共 717,290 次更新；**−0.06252334**。後被 v7 的「不重訪、MSE」規則取代。 | [報告](../jpx_v6_expanding_daily_gd_20260912/JPX-v6-expanding-daily-RMSE-report.md) |
| 9/12 v7 | 每日全部已知標籤股票等權 MSE 一次更新，**+0.00767887**；排名加權參照 **−0.04018310**。使用者指定等權版為正式基準。 | [報告](../jpx_v7_daily_mse_20260912/JPX-v7-daily-MSE-comparison.md) |
| 9/12 v8 | 同檔真實名次對 sigmoid soft rank 平方差，τ=1；整日／逐檔 SGD／mini128：**−0.03527185／−0.03164664／−0.04360392**。 | [報告](../jpx_v8_soft_rank_20260912/JPX-v8-same-stock-rank-loss-report.md) |
| 9/12 v9 | v8 SGD 加營業利益率同比差，10 交易日線性由 1→0.1，第 11 日歸零；**−0.01953833**，較 v8 SGD 好但低於 v7。 | [報告](../jpx_v9_financial_decay_sgd_20260912/JPX-v9-financial-decay-SGD-report.md) |
| 9/13 v10 | 改普通報酬 MSE＋逐檔 SGD；無財報／財報衰減：**−0.04311345／−0.03949089**；財報版日均 MSE 0.00498207 高於對照 0.00455938。 | [報告](../jpx_v10_mse_sgd_financial_20260913/JPX-v10-MSE-SGD-financial-report.md) |
| 9/13 v11 | 2×2：v10／僅持續財報／僅 PR1、VR1 除自身 22 日 std／兩者：**−0.03949089／−0.03930989／−0.02327178／−0.02676471**。持續財報令 MSE 上升。 | [報告](../jpx_v11_persistent_finance_std22_20260913/JPX-v11-persistent-finance-std22-report.md) |
| 9/13 v12 | 六財報：實際營收／營業利益 YoY、EPS 原值、全年 Forecast 營收／營業利益對去年 FY 成長、Forecast EPS 原值；18 參數、PR1/VR1 標準化；**+0.00887445**，價量對照 −0.02808045，日均 MSE **0.54387164**。財年縮短錯配已修正並重跑；EPS 主導梯度尺度不代表預測重要性。會計口徑推定另有待追查風險。 | [報告](../jpx_v12_financial_growth_20260913/JPX-v12-six-financial-features-report.md) |
| 9/14 v13 | 全年 Forecast 減已知 YTD 後分剩餘季度；驚喜、預測 QoQ/YoY、修正皆 exp(−a/9)，27 參數。分塊 SGD／再加每日 ⌊√N⌋ 次整體更新：**−0.01685636／−0.04218165**，MSE 0.03820259／0.00159566。原聯合更新暖身溢位，保留失敗版後改分塊。**後確認口徑錯配，不能作乾淨資料結論。** | [報告](../jpx_v13_quarterly_sgd_sqrt_20260914/JPX-v13-quarterly-SGD-sqrt-report.md)、[已知問題](../jpx_v13_quarterly_sgd_sqrt_20260914/KNOWN_ISSUES.md) |
| 9/14 v14 | 訓練排任一財報成分絕對值>100（各跳 2,294 股票日）；全年 Forecast 未變不新建訊號；無法確認口徑的修正不套用。SGD／混合：**−0.01760090／−0.02987275**，MSE 0.02110705／0.00133558。三項規則同改，不歸因單項。 | [報告](../jpx_v14_filtered_forecast_events_20260914/JPX-v14-filtered-forecast-report.md) |
| 9/14 v15 | 沿 v14 財報與 100 倍訓練篩選，改 soft rank（τ=1）；SGD／混合：**−0.00621273／−0.01945739**，Rank IC −0.00099868／−0.00320747。 | [報告](../jpx_v15_soft_rank_filtered_20260914/JPX-v15-soft-rank-report.md) |
| 9/15 v16 | 回 v7 每日一次 MSE、PR1/VR1 原值，逐一加 15 財報成分，exp(−a/9)，各有同篩選股票日價量對照；10 欄高於 v7、15 欄正 Sharpe。最佳營業利益實際 YoY **+0.01169366**。此輪有門檻，後由 v25 無篩選重做。 | [報告](../jpx_v16_single_financial_v7_20260915/JPX-v16-single-financial-report.md) |
| 9/15 v17 | EPS 預測對已知實際；三種修正改 `(new−old)/old`（保留分母正負）；EPS 實際 QoQ＋YoY 聯合 **+0.00781016**。負基期時正負號不可直接當利多／利空。 | [報告](../jpx_v17_actual_forecast_revisions_20260915/JPX-v17-revised-financial-report.md)、[判讀診斷](../jpx_v17_actual_forecast_revisions_20260915/interpretation_diagnostics/JPX-feature-interpretation.md) |
| 9/15 v18 | 固定每個訊號時點 v7 g，只學 EPS QoQ／YoY／聯合係數，共同樣本及步長：**+0.00658367／+0.00646778／+0.00627586**，均低於 v7。 | [報告](../jpx_v18_frozen_g_eps_20260915/JPX-v18-frozen-g-report.md) |
| 9/15 v19 | 財報預測支線：2018 暖身、2019–2021 expanding OLS，EPS YoY=a+B×QoQ。21,993 事件 MSE 比歷史均值高 1.135%；兩比率≤100 的 21,813 事件低 1.978%。不是股價排名模型。 | [報告](../jpx_v19_qoq_predict_yoy_20260915/JPX-v19-qoq-predict-yoy-report.md) |
| 9/15 v20 | 20 筆 EPS 原始欄位／時點核對，確認拆季語意及來源差異；未修資料、未重訓。舊 EPS 結果僅代表舊資料定義，v7 無 EPS 不受此項影響。 | [資料稽核](../jpx_v20_eps_data_audit_20260915/JPX-v20-eps-data-audit-report.md) |
| 9/16 v21 | 使用者要求保留 EPS 算法，g＋六 EPS 共同學 18 參數；**−0.00023157**，同樣本 g **+0.00564407**，再同步長 g **+0.00460263**。 | [報告](../jpx_v21_all_eps_joint_20260916/JPX-v21-all-eps-joint-report.md) |
| 9/16 v22 | g＋六 EPS，訓練門檻 100／10／不跳過，驗證完整池；**−0.00023157／+0.00303035／−0.04662785**；日均 MSE 0.000773331／0.000999315／0.000643657。探索差值區間均跨 0，未選新門檻。 | [報告](../jpx_v22_eps_thresholds_20260916/JPX-v22-threshold-report.md) |
| 9/16 v23 | 純 g，EPS 只作訓練篩選，不跳過／100／10：**+0.00767887／+0.00564407／+0.00538562**。原 v7 與 v21 price_mask 精確重現；沒有可靠改善證據。 | [報告](../jpx_v23_g_thresholds_20260916/JPX-v23-g-threshold-report.md) |
| 9/16 v24 | 純 g 訓練＋驗證同步篩選，不跳過／100／10：**+0.00767887／+0.00630339／+0.00427927**；驗證排 2,756／29,429 股票日。不同池不可直接當完整池能力改善，探索區間仍跨 0。 | [報告](../jpx_v24_g_train_validation_filters_20260916/JPX-v24-train-validation-filter-report.md) |
| 9/16 v25 | 依新規則不排極端值：21 單指標中重訓 18、核對沿用 3；現行 16 項、歷史已替代 5 項分列。現行 7 項 Sharpe 高於 v7、全部平均 Rank IC 負；六 EPS Sharpe 均負。最佳營業利益實際 YoY **+0.01450931**、Rank IC −0.00016194。 | [報告](../jpx_v25_single_features_unfiltered_20260916/JPX-v25-unfiltered-single-features-report.md) |
| 9/16 v26 | g＋營業利益 Forecast YoY＋相對修正共同 14 參數、不篩選；**+0.01151621**，Rank IC −0.00019437；預測年比單項 +0.01265711，修正單項 +0.00773022。此修正是獨立衰減事件，不是最新值覆蓋。 | [報告](../jpx_v26_profit_forecast_revision_20260916/JPX-v26-profit-forecast-revision-report.md) |
| 9/16 v27 | 最新 Forecast 年比狀態：A=`g+β×new`；B=`g+r1(new−old)+r2×old`，共同去年同季 Forecast 基期、最新覆蓋、exp(−a/9)。A／B：**+0.01112196／+0.00193527**，Rank IC −0.00072297／−0.00136374。換季同口徑重算 old，保留 4,608 次非零修正；差值區間跨 0。 | [報告](../jpx_v27_profit_forecast_state_20260916/JPX-v27-forecast-state-report.md) |
| 9/16 v28 | 同日 A、B、g 共用 `min(ηA,ηB)`。g／A／B：**+0.00081145／+0.00106306／+0.00222359**。A/B 次序反轉，顯示對步長排程敏感；差值區間跨 0。 | [報告](../jpx_v28_common_learning_rate_20260916/JPX-v28-common-learning-rate-report.md) |
| 9/16 v29 | 改選擇規則為 Sharpe 優先、Rank IC 次、不設正值硬門檻。共同步長倍率 0.25/0.5/1/1.5/1.9 × g/A/B，共 15 組；各模型最高 Sharpe 均為 1 倍。候選 B/1 **+0.00222359**、Rank IC −0.00137060，仍低於 v7；1 倍精確重現 v28。test 未使用。 | [報告](../jpx_v29_learning_rate_sweep_20260916/JPX-v29-learning-rate-sweep-report.md)、[計畫](../jpx_v29_learning_rate_sweep_20260916/experiment_plan.md) |
| 9/16 v30 | R=既有實際營業利益 YoY 累加事件；F=v27 最新 Forecast YoY 狀態。七組同日共用步長；原排程 94 日超完整模型穩定上限，全部改 `min(原步長,完整模型譜步長)`。共同 g／g+R／g+F／g+R+F：**+0.00269547／+0.00331220／+0.00384279／+0.00400258**；固定本輪訊號時 g 加 R／F／R+F：**+0.00435804／+0.00315320／+0.00411408**。候選 frozen_r；配對區間均跨 0，test 未使用。 | [報告](../jpx_v30_profit_actual_forecast_20260916/JPX-v30-profit-actual-forecast-report.md)、[計畫](../jpx_v30_profit_actual_forecast_20260916/experiment_plan.md) |

v30 另保存重要診斷：全 validation R/F Pearson=0.005161、Spearman=0.099486；係數正負切換未增加。來源全期間同公告 21,135 筆有效 Forecast 指向實際值之後季度，不能當同季 surprise。R 仍限既有 v14 事件有效性（當時實際年增和事前預期成長均可算），不得推廣為所有實際財報值。固定 g 必須是原訊號日已保存預測，不是期末參數回填。

## 必須保留的修正、失敗與未解問題

1. **數值核對不等於資料語意正確。** v13 的 1808／2018-03-15／SourceRow 22861 把個別 Forecast 配到合併實績，產生錯誤負季度預測及極端成長率。原 v13 沒有修正重訓；歷史值只供診斷。v12 亦曾用同樣唯一口徑推定規則，全面影響未確認。v14 改排無法確認口徑事件，不應聲稱已把全部舊結果修好。見 [v13 診斷](../jpx_v13_quarterly_sgd_sqrt_20260914/diagnostics_20260914/JPX-v13-diagnostic-report.md)。
2. **v12 財年縮短修正已完成，與上述口徑問題是兩件事。** 初次運算 74 事件保留錯誤年度截止日，影響 3,675 股票日；現報告採修正重跑數字，初步版留在 `jpx_v12_financial_growth_20260913/pre_period_fix/`。
3. **v13 溢位版是失敗紀錄。** 同時更新所有參數在暖身溢位，留在 `jpx_v13_quarterly_sgd_sqrt_20260914/joint_step_overflow/`；後改先影響係數、再折價參數之分塊法，不是按 validation Sharpe 挑優。
4. **EPS 尚未全面修復。** v20 確認 4617 累計 EPS 相減與官方單季不同；4572 原 JPX CSV −62.50 與 2021-08-06 公司公告 −62.56 不符，原因未定。2884 的 2022 更正不能倒填 2021 預測；極端比率亦可能是真實低基期。20 案抽查不能推估全集錯誤率。使用者後續仍要求不改 EPS 算法，因此 v21 以後也不能自動稱為「已修復 EPS」。
5. **低 MSE、縮小係數、負係數或低相關都不是交易因果證明。** v13 的 MSE 大幅降低而 Sharpe 變差；v22 無篩選 MSE 最低而 Sharpe 最差。v17/v26–v30 配對診斷是探索，多數區間跨 0；不能直接斷言市場不信任修正、某訊號穩定負貢獻，或已通過新資料驗證。
6. **保存每輪實驗當時規則。** v14–v24 的門檻、早期 Ridge 的 MSE 選參數以及 v25 當時雙指標門檻都屬歷史規格；新預設不應改寫舊報告。v29/v30 已各保存 `experiment_defaults_snapshot.json`。

## 9/23：ARIMA＋殘差 ACF 學習支線（未正式回測）

來源：[事前計畫](../jpx_arima_acf_learning_20260923/experiment_plan.md)、[資料稽核](../jpx_arima_acf_learning_20260923/data_audit.json)、[數學核對](../jpx_arima_acf_learning_20260923/math_checks.json)。使用者已指定 X=收盤價、d=1、ACF 用於預測殘差；未知選擇先詢問。待確認逐檔／共用參數、殘差截距 c、每日到期 Target GD／252 日窗重訓，正式訓練不得自行代選。還需明確決定原始／調整 Close 與缺價處理。

已推導 `z[t]=X[t]−X[t−1]`、`z[t+1]=a+φz[t]+θe[t]+e[t+1]`、`e[t+1]=c+be[t]+u[t+1]`；用未來 u 條件均值零遞迴兩步，`predicted_target[t]=zhat2/Xhat1`。差分不是百分比報酬；ACF 不是已知的新殘差；b=c=0 時退化 ARIMA(1,1,1)，帶 AR(1) 殘差的整體形式可寫為有限制 ARIMA(2,1,1)。價格先預測再取比值是 plug-in，不保證等於報酬條件期望。

已查訓練檔 2017-01-04–2021-12-03，共 2,332,531 筆、2,000 檔，Close 缺值 7,608；test 未讀取。以既有因子調整與前值填補重建報酬，2,328,293 可比筆中 57,161 筆與官方 Target 差>1e−8，最大 0.0019778113；原因未定，原 Target 保留。核心程式拒絕缺失價格，初始狀態由呼叫端指定。合成資料梯度中央差分最大差 3.25e−15、因果時序與退化案例核對通過，**不是 JPX 績效**。

## 保存與重現時的最小證據鏈

每項實驗至少保留事前 plan、執行／稽核程式、設定或 manifest、來源資料及雜湊、財報事件／特徵、逐日預測與排序、參數軌跡／training_updates、完整 results、獨立 audit 與原報告；失敗、修正前及診斷目錄也屬研究證據。這些內容可由原檔或無損封存保存，不用本摘要取代大型資料。具體檔名依各實驗原報告定位；遇到已封存的原路徑，先按 [還原指南](RESTORE_GUIDE.md) 恢復所需檔案。原始資料或程式內若有本機絕對路徑，搬家後還需另做路徑適配；不能只上傳 Markdown 就宣稱完整可重現。
