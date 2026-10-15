# JPX unittest v3｜最後一次稽核：2026-10-15 11:24（台北）

**104 項：81 通過、19 項 assertion failure、4 項 error、0 跳過。23 個未通過的方法對應 17 組問題：7 組主流程問題、10 組條件性／邊界問題。** 一項方法包含兩個失敗子案例，因此 unittest 原始輸出顯示 failures=20, errors=4。

你在測試期間持續修改 normalize_data。本報告固定對應 11:24 保存的來源快照；之後的編輯請重跑測試，不要把舊結果套用到新的程式。最新封存來源位於 latest_revision_1124/source_snapshot，雜湊位於同目錄的 source_manifest.json。完整結果位於 results_1124/test_results.json。

我們只新增測試與報告，沒有修改 codes 的模型或資料處理邏輯。使用者在本輪自行修改的 features.py 已分版保存。

## 已解除的問題

先前 std: None 是必填參數，現已改成 std: float | None = None，所以 B16「呼叫漏傳 std」在最後一次測試已通過，不再列為目前問題。feature_columns 也已正確傳入存檔函式，模型歸位至 models.py 後的 forward 測試通過。

## 主流程優先順序

| 編號 | 問題 | 影響 |
|---|---|---|
| B01 | 呼叫使用 epochs，函式接受 epoch | 主流程／P1 |
| B02 | epoch 沒有控制迴圈 | 主流程／P1 |
| B03 | 先用全期間標準化，再切 train/validation | 主流程／P1 |
| B04 | checkpoint 寫入固定 0、1，遺失訓練標準化參數 | 主流程／P1 |
| B05 | use_loaded_model 讀空路徑，且丟棄載入結果 | 主流程／P2 |
| B06 | model_path=str 是型別物件，不是路徑預設值 | 主流程／P2 |
| B07 | 把預測值加總印成收益百分比 | 主流程／P2 |

## 全部可重現問題

### B01｜呼叫使用 epochs，函式接受 epoch（主流程／P1）

來源：`JPX_deep_learning_models.py:73`。model_1 與 model_2（第 124 行）都在 train 呼叫報 TypeError，訓練尚未開始。

對應測試：

- `test_audit.TrainingContractTests.test_real_model1_call_matches_train_signature`
- `test_audit.TrainingContractTests.test_real_model2_call_matches_train_signature`

### B02｜epoch 沒有控制迴圈（主流程／P1）

來源：`deep_learning.py:44`。4 個樣本、batch=2、epoch=20 應更新 40 次，實際只有 2 次；epoch=0 也仍改變模型參數。

對應測試：

- `test_audit.TrainingContractTests.test_epoch_twenty_runs_twenty_complete_passes`
- `test_audit.TrainingContractTests.test_zero_epoch_cannot_update_parameters`

### B03｜先用全期間標準化，再切 train/validation（主流程／P1）

來源：`JPX_deep_learning_models.py:34`。只改驗證期資料，訓練特徵也改變。合成 Close 訓練值 [11,21] 應轉為 [-1,1]，實際為 [-1.341641,-0.447214]；驗證 [31,41] 應為 [3,5]，實際為 [0.447214,1.341641]。

對應測試：

- `test_audit.PreprocessingAuditTests.test_validation_values_cannot_change_training_features`
- `test_audit.PreprocessingAuditTests.test_training_features_use_training_statistics`
- `test_audit.PreprocessingAuditTests.test_validation_uses_same_training_transform`

### B04｜checkpoint 寫入固定 0、1，遺失訓練標準化參數（主流程／P1）

來源：`JPX_deep_learning_models.py:141`。六個特徵卻儲存 scalar mean=0、std=1，不能用來重現訓練前處理。common_process 也未回傳擬合參數。feature_columns 已補齊，對應測試通過，這次沒有把它列為缺失。

對應測試：

- `test_audit.CheckpointAuditTests.test_model2_checkpoint_retains_one_scale_per_feature`

### B05｜use_loaded_model 讀空路徑，且丟棄載入結果（主流程／P2）

來源：`main.py:8`。helper 固定傳入空字串，實際讀取會找不到檔案；即使替換讀檔為成功，helper 仍回傳 None，模型與前處理資料沒有提供給後續使用。

對應測試：

- `test_audit.CheckpointAuditTests.test_use_loaded_model_passes_a_nonempty_checkpoint_path`
- `test_audit.CheckpointAuditTests.test_use_loaded_model_returns_the_loaded_result_for_use`

### B06｜model_path=str 是型別物件，不是路徑預設值（主流程／P2）

來源：`deep_learning.py:146`。省略 model_path 時，Path 收到 str 型別物件而報 TypeError。應要求呼叫者給路徑，或使用真正的有效預設。

對應測試：

- `test_audit.CheckpointAuditTests.test_load_model_path_is_required_or_has_a_real_path_default`

### B07｜把預測值加總印成收益百分比（主流程／P2）

來源：`JPX_deep_learning_models.py:132`。所有預測加同一常數，排名與持倉應不變，輸出的 Total earnings 卻由每 fold 6 變為 46。這是模型分數加總，不是實際 Target 配權後的收益；加上 % 也未建立正確單位。

對應測試：

- `test_audit.ReportingAuditTests.test_reported_earnings_are_invariant_to_prediction_level_shift`

### B08｜常數欄位的回傳 std 無法直接重播轉換（邊界／P2）

來源：`features.py:61`。ExpectedDividend 全為 0 時，normalize_data 回傳的轉換值有限，但 std=0；用回傳的 (x-mean)/std 重播產生 NaN。這個問題在把 std 當除數重播時觸發；應保存實際安全 scale 或明確處理零變異。

對應測試：

- `test_audit.PreprocessingAuditTests.test_returned_scaling_metadata_reproduces_constant_column`

### B09｜重複的股票日期仍被當作有效列（輸入防護／P2）

來源：`features.py:16`。檢查函式已找出重複 stock/date，但 get_valid_rows 丟棄該結果；合成重複列仍進入序列，可能重複計算同一股票日。測試接受拒絕輸入或明確去重，不指定保留哪一筆。

對應測試：

- `test_audit.PreprocessingAuditTests.test_duplicate_stock_dates_do_not_survive_valid_rows`

### B10｜最後一列的非正調整因子漏檢（輸入防護／P2）

來源：`data_processor.py:179`。只檢查 shift 後的累積因子，最後一列 AdjustmentFactor=0 或 -0.5 均被保留。測試不判定拆股生效日，只驗證現有非正因子禁用規則的缺口。

對應測試：

- `test_audit.PreprocessingAuditTests.test_final_row_nonpositive_factor_is_rejected_or_repaired`

### B11｜time_sequence=0 或負數沒有拒絕（輸入防護／P2）

來源：`features.py:70`。產生空／不一致的序列及錯位標籤；window=0 的第一個標籤會索引最後一列。這些輸入應在建立序列前明確拒絕。

對應測試：

- `test_audit.PreprocessingAuditTests.test_sequence_window_zero_is_rejected`
- `test_audit.PreprocessingAuditTests.test_sequence_negative_window_is_rejected`

### B12｜缺欄位訊息把存在的欄位也列為缺失（錯誤訊息／P3）

來源：`features.py:34`。要求 Close 與 DoesNotExist 時，訊息顯示整份清單；list comprehension 應加入 n，現在卻加入 features_col_name。

對應測試：

- `test_audit.PreprocessingAuditTests.test_missing_column_message_names_only_missing_columns`

### B13｜NaN 預測仍得到有限的績效分數（輸入防護／P2）

來源：`evaluation.py:54`。401 檔、3 日的合成資料中放入一個 NaN 預測，overall_performance 仍回傳有限分數。失效的模型輸出應被明確標示，不能照常被排序配置持倉。

對應測試：

- `test_audit.ReportingAuditTests.test_nonfinite_predictions_cannot_silently_produce_a_valid_score`

### B14｜缺失持倉 Target 被 sum 靜默略過（輸入防護／P2）

來源：`evaluation.py:91`。Weight=[0.5,-0.5]、Target=[NaN,0.1] 仍算出有限收益 -0.05。此為直接呼叫評估函式時的缺口；目前 common_process 會排除缺失 Target，因此不宣稱正常管線一定觸發。

對應測試：

- `test_audit.ReportingAuditTests.test_missing_target_cannot_be_silently_treated_as_zero_earning`

### B15｜float32 將很小的日收益變動壓成零（數值邊界／P3）

來源：`evaluation.py:107`。日收益 [0.01,0.0100000001,0.0100000002] 在 float64 有非零變異，轉 float32 後被誤判標準差為零並報錯。這是精度邊界，未證實發生在真實 JPX 結果。

對應測試：

- `test_audit.ReportingAuditTests.test_small_but_nonzero_daily_variance_is_not_rounded_to_zero`

### B17｜傳入非零 std 的分支沒有回傳結果（新增分支／P2）

來源：`features.py:44`。以 std=2.0、mean=10.0 呼叫，函式回傳 None；此分支只有複製 DataFrame，尚未做轉換或 return。這是使用可選 std 分支時觸發，目前 common_process 仍使用預設分支。

對應測試：

- `test_normalizer_reuse.NormalizerReuseTests.test_nonzero_scalar_std_returns_the_transformed_frame`

### B18｜if not std 不能判斷六個特徵的標準差向量（參數重用／P2）

來源：`features.py:55`。把 normalize_data 自己回傳的六組 mean/std 用於重播，if not std 引發 NumPy truth value is ambiguous。測試已先把常數欄位尺度替換為 1，單獨確認向量處理問題。若這個新分支要重用訓練的逐特徵參數，就需要支援向量；若只支援純量，則仍缺少這條必要的逐特徵重播流程。

對應測試：

- `test_normalizer_reuse.NormalizerReuseTests.test_fitted_featurewise_metadata_can_be_reused`

## 通過的測試及觀察

62 項既有測試的副本已更新模型位置、分數 dict 與建構測試的資料替身，全部通過。新增 42 項中 19 通過、23 未通過。原本 v1/v2 沒有被修改。測試涵蓋張量轉換、資料列篩選、股票與日期排序、建窗和 Target 對齊、訓練／評估 batch、no-grad、預測串接、排名配權、Sharpe、Rank IC，以及真正 PyTorch 序列化的模型／metadata 往返。

- 42 日卷積搭配 kernel=42，LSTM 收到 1 個時間步：記錄為架構選擇。
- expanding folds 重用同一模型及 optimizer：記錄為暖啟動選擇。
- 驗證集分開建 42 日窗會失去每檔前 41 列：是否應帶入訓練尾段歷史需先定義評分範圍。
- main 目前選擇訓練入口；early stopping 尚未實作。未把這些選擇或未新增功能另算 bugs。

## 如何解讀與重跑

```bash
python3.12 -B /Users/coolguy/Desktop/JPX/debug/JPX_debug_v3/run_tests.py
```

上述指令測目前的 codes。要精確重播這份報告，從存檔目錄執行 `python3.12 -B JPX_debug_v3/run_tests.py --codes latest_revision_1124/source_snapshot`。可以在最後加上表列的完整測試名稱，單獨重現。

環境為 Python 3.12.2、PyTorch 2.11.0、NumPy 2.0.0、pandas 2.2.2、SciPy 1.17.1、scikit-learn 1.8.0。只使用 CPU 小型合成資料，未讀市場 CSV、未跑完整訓練，序列化只在記憶體進行。後段存檔／報表測試替換訓練，避免 epochs 介面錯誤遮蔽它們；真實訓練呼叫另有 smoke tests。

標準化下游測試保留對先前必填 std 的相容替身；最後快照已使用可選 std，因此最後一次執行並不需要或啟動該替身。新參數重用測試直接呼叫真正 normalize_data。

本次未量測行覆蓋率，也不能保證找出所有 bugs。未驗證真實拆股／除息與 ExpectedDividend 口徑、Target 可知時間、完整市場池、同分排名政策、MPS/CUDA、長時間資源用量或投資績效。合成無效輸入的失敗不表示真實 CSV 一定含有那些值。

## 版本紀錄

首輪 101 項、必填 std 版本的 102 項，以及這次 104 項結果均保留。102 項版本已獨立保存於 Git commit dee9551fb66ccef385bfba03a169e5b78d3fc946；最新修訂是新增提交，沒有刪除或覆蓋舊 Git 歷史。
