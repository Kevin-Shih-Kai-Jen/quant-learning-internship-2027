# 目前研究模型：v7 等權每日 MSE

2026-09-12 依使用者最新確認，使用每天全部股票的普通平均 MSE，在 Target 已實現後只做一次 gradient descent。參數只初始化一次，跨日期與年度接續；不開根號、不重訪舊日期。這取代先前 v6 的歷史重訪與 RMSE 設定。

十一個 v5 價量特徵保留，一日價量變動率維持原值。主模型 v7_equal 使用所有已知 Target 股票等權；參照版 v7_jpx 依預測當時固定的多空各 200 檔官方 2→1 權重計算 MSE，其餘權重為零，兩側誤差權重均為正。

相同 953 個驗證日，JPX 未年化 Sharpe：主模型 0.00767887，參照版 −0.04018310。每版共 1,199 次單次日更新，全部梯度與參數已獨立核對。此為未扣成本的歷史逐日驗證，不是新保留 test 或排行榜成績。

[完整比較報告](/Users/coolguy/.codex/.chatgpt-projects/g-p-6a8ad7dc606481919474d5424342a1c3/jpx_v7_daily_mse_20260912/JPX-v7-daily-MSE-comparison.md)。舊模型結果完整保留。

## 最新完成的 v8 比較

2026-09-12 已完成使用同一檔真實名次與 soft rank 差平方的三組試驗。相同 953 日的 JPX 未年化 Sharpe：每天整批 −0.03527185、逐檔 −0.03164664、每 128 檔 −0.04360392，皆低於 v7 等權。v8 保存為完整可查核的實驗，現有基準仍為 v7 等權。

[完整 v8 報告](/Users/coolguy/.codex/.chatgpt-projects/g-p-6a8ad7dc606481919474d5424342a1c3/jpx_v8_soft_rank_20260912/JPX-v8-same-stock-rank-loss-report.md)。

## 最新完成的 v9 財報特徵比較

2026-09-12 完成 v8 SGD 加入「r × 營業利益率同比差值」的新特徵，r 在 10 個交易日內由 1 降到 0.1，第 11 日歸零。相同 953 個驗證日，JPX 未年化 Sharpe 由原 SGD 的 −0.03164664 改善至 −0.01953833，但仍低於 v7 等權的 0.00767887。財報特徵、梯度抽查與全部排名核對通過。v9 保留為實驗，現有基準仍為 v7 等權。

[完整 v9 報告](/Users/coolguy/.codex/.chatgpt-projects/g-p-6a8ad7dc606481919474d5424342a1c3/jpx_v9_financial_decay_sgd_20260912/JPX-v9-financial-decay-SGD-report.md)。

## 最新完成的 v10 普通 MSE＋SGD＋財報衰減

2026-09-13 依要求移除 soft rank，改用 v7 普通報酬平方誤差，結合逐檔 SGD 與 v9 財報衰減。相同 953 日，JPX 未年化 Sharpe：無財報 SGD −0.04311345、財報 SGD −0.03949089。財報特徵略改善 Sharpe，但兩組皆為負；財報版平均每日預測 MSE 較高（0.00498207，對照 0.00455938）。兩組逐筆紀錄、保存梯度樣本及全部排名獨立核對通過。v10 完整保存為實驗，現有基準仍為 v7 等權。

[完整 v10 報告](/Users/coolguy/.codex/.chatgpt-projects/g-p-6a8ad7dc606481919474d5424342a1c3/jpx_v10_mse_sgd_financial_20260913/JPX-v10-MSE-SGD-financial-report.md)。

## 最新完成的 v11 財報持續與 return 標準化比較

2026-09-13 完成 2×2 特徵比較。相同 953 日的 JPX 未年化 Sharpe：原 v10 −0.03949089、只取消財報衰減 −0.03930989、只將 PR1／VR1 各自除以 22 日標準差 −0.02327178、兩者都改 −0.02676471。只標準化版為這四組最佳，但仍為負值；不衰減財報使預測 MSE 明顯升高。全部特徵、更新紀錄與排名核對通過。v11 完整保存為實驗，現有基準仍為 v7 等權。

[完整 v11 報告](/Users/coolguy/.codex/.chatgpt-projects/g-p-6a8ad7dc606481919474d5424342a1c3/jpx_v11_persistent_finance_std22_20260913/JPX-v11-persistent-finance-std22-report.md)。

## 最新完成的 v12 六項財報模型

2026-09-13 依使用者確認，將實際營收／營業利益同比成長、實際 EPS 原值、全年 Forecast 營收／營業利益相對去年完整 FY 實際值的成長、Forecast EPS 原值，加入標準化 PR1／VR1 的價量模型；共18個共同學習參數。財年縮短處理已修正並重跑。相同953日的JPX未年化Sharpe：新模型 +0.00887445、純價量對照 −0.02808045；但平均每日報酬MSE為0.54387164，遠高於對照0.00449157。約93.83%的batch由EPS占梯度平方範數至少90%；這是尺度診斷，不是預測重要性。全量逐步更新、特徵與排名核對通過。v12保存為完整實驗；現有基準仍為v7等權。

[完整 v12 報告](/Users/coolguy/.codex/.chatgpt-projects/g-p-6a8ad7dc606481919474d5424342a1c3/jpx_v12_financial_growth_20260913/JPX-v12-six-financial-features-report.md)。

## 最新完成的 v13 季度預期與每日整體更新

2026-09-14 完成季度財報預期模型：全年 Forecast 扣除已公布累計實績後分配至剩餘季度，加入實際驚喜、Forecast 季增／年增與預期修正，所有事件依 exp(−a/9) 衰減。兩組共用分塊 SGD（先影響係數、再折價參數）；混合組每天額外做 ⌊√N⌋ 次當日平均 MSE 更新。原同時更新全部參數在暖身期數值溢位，已保存並改用分塊法，沒有按驗證績效調整。相同953日：單獨 SGD 未年化 Sharpe −0.01685636、MSE 0.03820259；混合組 Sharpe −0.04218165、MSE 0.00159566。MSE降低95.82%，選股排序表現則變差。兩組所有更新、特徵與排名核對通過。現有基準仍為v7等權。

[完整 v13 報告](/Users/coolguy/.codex/.chatgpt-projects/g-p-6a8ad7dc606481919474d5424342a1c3/jpx_v13_quarterly_sgd_sqrt_20260914/JPX-v13-quarterly-SGD-sqrt-report.md)。

## v13 資料語意診斷更新

2026-09-14 已確認1808的2018-03-15 ForecastRevision將個別預測配到合併實績。既有數值稽核不能排除此類口徑問題，v13結果需先修正資料再重新評估。另已發現近零EPS基期造成極端成長率。本次只做診斷，未重訓。[完整診斷](/Users/coolguy/.codex/.chatgpt-projects/g-p-6a8ad7dc606481919474d5424342a1c3/jpx_v13_quarterly_sgd_sqrt_20260914/diagnostics_20260914/JPX-v13-diagnostic-report.md)。

## 最新完成的 v14 訓練篩選與新 Forecast 事件

2026-09-14 將任一財報成長成分絕對值>100的股票日從SGD和整體loss排除，N按通過篩選的已知標籤股票計算；兩組各跳過2294個訓練股票日。全年Forecast不變時當次新增Forecast訊號為零，舊事件仍自然衰減。無法確認會計口徑的ForecastRevision與NumericalCorrection不套用，避免v13已確認的錯配。相同完整953日股票池：單獨SGD Sharpe −0.01760090、MSE 0.02110705；混合組Sharpe −0.02987275、MSE 0.00133558。兩組特徵、篩選、更新和排名核對通過。這輪同時改三項規則，與v13差異不應歸因單一因素。正式基準仍為v7。

[完整 v14 報告](/Users/coolguy/.codex/.chatgpt-projects/g-p-6a8ad7dc606481919474d5424342a1c3/jpx_v14_filtered_forecast_events_20260914/JPX-v14-filtered-forecast-report.md)。

## 最新完成的 v15 Soft rank 對照

2026-09-14 沿用 v14 財報特徵與 100 倍訓練篩選，只改為 sigmoid soft rank loss。相同 953 日的未年化 Sharpe：單獨 SGD -0.00621273；SGD＋每日 √N 次整體更新 -0.01945739。兩組完整更新紀錄與排名核對通過。這是既有歷史期間的實驗，未扣成本，也尚未使用新保留測試集。正式基準仍為 v7。

[完整 v15 報告](/Users/coolguy/.codex/.chatgpt-projects/g-p-6a8ad7dc606481919474d5424342a1c3/jpx_v15_soft_rank_filtered_20260914/JPX-v15-soft-rank-report.md)。

## 最新完成的 v16 單一財報成分對照

2026-09-15 回到 v7 每日一次普通 MSE、PR1/VR1 原值，保留財報 exp(-a/9)，逐一加入15欄財報成分，另設15組相同篩選股票日的純價量對照。相同953日，基準Sharpe +0.00767887；10欄高於v7，15欄大於0。最佳單欄為營業利益／實際 YoY 成長，Sharpe +0.01169366。全部更新與排名稽核通過。正式基準仍為v7。

[完整 v16 報告](/Users/coolguy/.codex/.chatgpt-projects/g-p-6a8ad7dc606481919474d5424342a1c3/jpx_v16_single_financial_v7_20260915/JPX-v16-single-financial-report.md)。

## 最新完成的 v17 修訂財報比較

2026-09-15 將EPS預測改與已知實際值比較，營收／營業利益／EPS三項修正改成(新單季預測-舊單季預測)/舊單季預測（帶正負號），並驗證EPS實際季增＋年增。共16組模型與對照通過稽核；相同953日，聯合EPS模型Sharpe +0.00781016，v7為+0.00767887。正式基準仍為v7。

[完整 v17 報告](/Users/coolguy/.codex/.chatgpt-projects/g-p-6a8ad7dc606481919474d5424342a1c3/jpx_v17_actual_forecast_revisions_20260915/JPX-v17-revised-financial-report.md)。

## 最新完成的 v18 固定 g 比較

固定原 v7 各訊號日預測，只訓練 EPS 季增／年增係數。三組共同樣本及步長，953 日 Sharpe：季增 +0.00658367、年增 +0.00646778、聯合 +0.00627586，皆低於 v7 +0.00767887。正式基準仍為 v7。

[完整 v18 報告](/Users/coolguy/.codex/.chatgpt-projects/g-p-6a8ad7dc606481919474d5424342a1c3/jpx_v18_frozen_g_eps_20260915/JPX-v18-frozen-g-report.md)。

## 最新完成的 v19 季增預測年增

2018暖身後，2019-2021 expanding OLS：EPS年增 = a + B × 季增。完整21,993個事件 MSE相對歷史平均高1.135%；兩比率均不超過100倍的21,813事件 MSE低1.978%。此為財報間的預測實驗，未改股價模型，正式基準仍為v7。

[完整 v19 報告](/Users/coolguy/.codex/.chatgpt-projects/g-p-6a8ad7dc606481919474d5424342a1c3/jpx_v19_qoq_predict_yoy_20260915/JPX-v19-qoq-predict-yoy-report.md)。

## 最新完成的 v20 EPS 資料核對

20筆EPS抽查：原JPX來源欄位／季度／時點核對通過；已確認累計EPS拆季與官方單季數字的重大差異（4617），以及原CSV與公司原公告EPS不一致（4572）。另有晚發財報更正與真實低基期案例。尚未修復資料或重訓，舊EPS結果只代表舊資料定義；正式純價量v7保持原樣。

[完整資料核對報告](/Users/coolguy/.codex/.chatgpt-projects/g-p-6a8ad7dc606481919474d5424342a1c3/jpx_v20_eps_data_audit_20260915/JPX-v20-eps-data-audit-report.md)。

## 最新完成的 v21 全EPS共同訓練

2026-09-16依使用者要求，維持目前EPS算法，把價量g與六項EPS係數一起訓練。953日Sharpe -0.00023157，原v7 +0.00767887；同樣本價量對照 +0.00564407，同樣本同主模型步長對照 +0.00460263。獨立核對通過，正式基準維持v7。

[完整 v21 報告](/Users/coolguy/.codex/.chatgpt-projects/g-p-6a8ad7dc606481919474d5424342a1c3/jpx_v21_all_eps_joint_20260916/JPX-v21-all-eps-joint-report.md)。

## 最新完成的 v22 EPS極端值門檻比較

2026-09-16：沿用g＋六項EPS共同訓練與原EPS算法，僅比較訓練門檻100倍、10倍、不跳過；完整預測股票池一致。Sharpe分別為−0.00023157、+0.00303035、−0.04662785；日均預測MSE為0.000773331、0.000999315、0.000643657。100倍組精確重現v21；所有更新、預測、排名與官方指標核對通過。三組Sharpe差值探索區間皆跨0，未選定新門檻。正式基準仍v7；完整結果見jpx_v22_eps_thresholds_20260916/JPX-v22-threshold-report.md。

## 最新完成的 v23 純g訓練篩選對照

2026-09-16：純v7價量g、12參數共同訓練，EPS僅作訓練股票日篩選。不跳過、100倍、10倍Sharpe分別+0.00767887、+0.00564407、+0.00538562；平均Rank IC分別−0.00073389、−0.00069463、−0.00070168。三組差值探索區間皆跨0，沒有可靠改善證據。原v7不跳過重現通過；100倍精確重現v21 price_mask；所有更新預測與指標核對通過。正式基準維持v7。完整結果見jpx_v23_g_thresholds_20260916/JPX-v23-g-threshold-report.md。

## 最新完成的 v24 純g訓練驗證同步篩選

2026-09-16：純g沿用v23訓練，驗證也以相同六項EPS門檻篩選後重新排名、多空各200檔。訓練／驗證皆不跳過、100倍、10倍Sharpe分別+0.00767887、+0.00630339、+0.00427927；平均Rank IC為−0.00073389、−0.00065814、−0.00065144。100倍與10倍驗證排除2,756與29,429股票日。額外原v7訓練不篩選、驗證100倍／10倍對照Sharpe+0.00800069／+0.00442733。差值探索區間皆跨0；全部核對通過。不同門檻驗證股票池不同，不將結果直接視為完整JPX股票池模型能力改善。正式基準維持v7。完整報告：jpx_v24_g_train_validation_filters_20260916/JPX-v24-train-validation-filter-report.md。

## 後續實驗預設：不排除極端值（2026-09-16使用者指定）

所有後續訓練取消依財報特徵大小跳過股票日的規則；不裁切數值。驗證預設完整股票池，主指標Sharpe與Rank IC，MSE僅診斷。保留既有缺值、到期Target與財報事件有效性處理，不改EPS算法。歷史有門檻的結果保留供對照，新實驗需讀取JPX-experiment-defaults.json。正式基準仍v7。

## 最新完成的 v25 單指標不排除重做

2026-09-16依使用者指定，後續訓練不跳過極端值。v16／v17共21個單指標版本中，18項曾排除樣本者重訓、3項未排除者核對沿用。現行16項、已替代歷史對照5項分開呈現。現行7項Sharpe高於v7，但全部平均Rank IC仍負，沒有同時滿足Sharpe高於g與Rank IC>0者。現行六項EPS Sharpe均轉負；最高現行Sharpe為營業利益實際年增+0.01450931，Rank IC−0.00016194，尚未通過雙指標標準。全部核對通過，正式基準仍v7。新預設見JPX-experiment-defaults.json；完整報告：jpx_v25_single_features_unfiltered_20260916/JPX-v25-unfiltered-single-features-report.md。

## 最新完成的 v26 營業利益預測年比＋修正

2026-09-16：g＋OperatingProfitForecastYoY＋OperatingProfitRevisionRelative共同訓練14參數，訓練／驗證不排除極端值。組合Sharpe+0.01151621、Rank IC−0.00019437；純g+0.00767887、預測年比單項+0.01265711、修正單項+0.00773022。組合Sharpe不是最高，Rank IC是四組最高但仍負；各配對差值區間皆跨0。此次修正是獨立加權衰減事件，並非新年比覆蓋舊年比狀態。全部核對通過，正式基準維持v7。完整報告：jpx_v26_profit_forecast_revision_20260916/JPX-v26-profit-forecast-revision-report.md。

## 最新完成的 v27 預測狀態覆蓋與差額＋舊值

2026-09-16：比較A=g+beta×new與B=g+r1×(new-old)+r2×old；new、old為相同去年同季Forecast基期下的營業利益預測年比。新狀態覆蓋，按exp(-a/9)衰減，價量與財報共同訓練，不排除極端值。A Sharpe +0.01112196、Rank IC −0.00072297；B Sharpe +0.00193527、Rank IC −0.00136374。純g Sharpe +0.00767887。A未穩定逐期改善；各配對差值95%區間皆跨0，兩者未通過雙指標門檻。換季將舊全年Forecast扣同一最新YTD實績、除相同剩餘季數；保留4,608次非零修正。資料時序、全部更新、預測與指標核對通過。正式基準維持v7。完整報告：jpx_v27_profit_forecast_state_20260916/JPX-v27-forecast-state-report.md。

## 最新完成的 v28 共同learning rate對照

2026-09-16：依使用者要求，同一更新日A覆蓋、B差額＋舊值及純g使用完全相同learning rate，取A/B已到期批次各自原步長之較小值；可逐日變動，不按績效選值。沿用v27狀態、exp(-a/9)、共同訓練及不排除極端值。共同步長純g Sharpe +0.00081145、Rank IC −0.00138883；A +0.00106306、−0.00146921；B +0.00222359、−0.00137060。相較原各自步長A +0.01112196、B +0.00193527，排序反轉，顯示結果對步長排程敏感；各配對差值95%區間仍跨0，兩者未通過雙指標門檻。1199次更新的步長、樣本、全部預測與官方指標核對通過。正式基準維持v7。完整報告：jpx_v28_common_learning_rate_20260916/JPX-v28-common-learning-rate-report.md。

## 最新完成的 v29 learning rate倍率validation比較

2026-09-16使用者更新選擇目標：Sharpe第一、Rank IC第二，MSE不參與選擇，Rank IC負值不作硬性淘汰；全部既有validation選候選，test之後才總體驗證。新規則已寫入JPX-experiment-defaults.json與版本登記，v29保存設定快照。

比較共同每日步長乘0.25、0.5、1、1.5、1.9，各跑純g、A覆蓋、B差額＋舊值，共15組。953日中三模型最高Sharpe皆為1倍：純g +0.00081145、A +0.00106306、B +0.00222359；B的Rank IC −0.00137060。最高候選為B／1倍，但仍低於原v7 Sharpe +0.00767887。1.5倍Rank IC較好而Sharpe轉負，按Sharpe優先不選。全部15組1199次更新、完整股票日與預測核對通過，1倍三組逐筆精確重現v28，極端值排除0筆。test未讀取或評估，正式基準維持v7。完整報告：jpx_v29_learning_rate_sweep_20260916/JPX-v29-learning-rate-sweep-report.md。

## 最新完成的 v30 營業利益實際與Forecast七組對照

2026-09-16：R=既有營業利益實際年增事件累加訊號，F=v27最新Forecast年比狀態，維持exp(-a/9)、完整股票池及不排除極端值。新增R後原步長94日超出完整模型穩定上限，七組統一改min(原步長,完整模型譜步長)。同共同排程純g Sharpe +0.00269547；共同g+R +0.00331220、g+F +0.00384279、g+R+F +0.00400258；固定本輪g後+R +0.00435804、+F +0.00315320、+R+F +0.00411408。共同合併高於兩單項，固定g合併略低於R；所有所列配對區間跨0，不確認穩定負貢獻。

診斷全validation R/F Pearson 0.005161、Spearman 0.099486，聯合係數正負切換未增加；來源全期間同公告21,135筆有效Forecast均指向實際值之後的季度，不能當作同季surprise。兩欄全部訊號重建、七組1199次更新、原訊號日固定g、預測排名與官方指標核對通過。Sharpe優先、Rank IC次之，MSE不選模型；test未使用，正式v7保持。完整報告：jpx_v30_profit_actual_forecast_20260916/JPX-v30-profit-actual-forecast-report.md。
