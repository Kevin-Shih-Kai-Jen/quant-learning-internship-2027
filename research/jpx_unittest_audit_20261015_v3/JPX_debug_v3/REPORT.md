# JPX unittest 稽核報告｜2026-10-15

對象：`/Users/coolguy/Desktop/JPX/codes` 的九個 Python 檔案。此報告以 `source_manifest.json` 的內容雜湊識別版本；新增測試與報告，沒有修改這九個檔案。

## 實際結果

**102 個測試方法：80 通過、18 assertion failure、4 error、0 skipped。** 22 個失敗方法歸納為 16 組問題：8 組主流程問題，加上 8 組條件性或邊界問題。B10 含兩個失敗子案例，所以 unittest 顯示 `failures=19, errors=4`；不要把子案例數或測試數當成獨立 bug 數。

環境：Python 3.12.2、PyTorch 2.11.0、NumPy 2.0.0、pandas 2.2.2、SciPy 1.17.1、scikit-learn 1.8.0。CPU、合成資料。未讀取真實 CSV，也未執行完整歷史訓練。

## 優先處理的主流程問題

| 編號 | 優先級 | 問題 | 來源位置 |
|---|---|---|---|
| B01 | P1 | 呼叫使用 epochs，函式接受 epoch | `JPX_deep_learning_models.py:73` |
| B02 | P1 | epoch 沒有控制迴圈 | `deep_learning.py:44` |
| B03 | P1 | 先用全期間標準化，再切 train/validation | `JPX_deep_learning_models.py:34` |
| B04 | P1 | checkpoint 寫入固定 0、1，遺失訓練標準化參數 | `JPX_deep_learning_models.py:141` |
| B05 | P2 | use_loaded_model 讀空路徑，且丟棄載入結果 | `main.py:8` |
| B06 | P2 | model_path=str 是型別物件，不是路徑預設值 | `deep_learning.py:146` |
| B07 | P2 | 把預測值加總印成收益百分比 | `JPX_deep_learning_models.py:132` |
| B16 | P1 | normalize_data 新增必填 std，但呼叫端未提供 | `features.py:47` |

## 逐項證據與重現測試

下列測試都保留紅燈，沒有使用 expectedFailure。P1 先處理訓練／資料洩漏；P2 是輸出、存取或條件性正確性問題；P3 是較低優先的診斷／精度邊界。

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

### B16｜normalize_data 新增必填 std，但呼叫端未提供（主流程／P1）

來源：`features.py:47`。執行期間使用者新增 std: None；這是型別註記，不是預設值。common_process 仍只傳 df 與欄位，因此報 missing required positional argument: std。本測試使用真實呼叫，沒有加參數替身。

對應測試：

- `test_audit.PreprocessingAuditTests.test_common_process_call_matches_normalize_signature`

### B08｜常數欄位的回傳 std 無法直接重播轉換（邊界／P2）

來源：`features.py:59`。ExpectedDividend 全為 0 時，normalize_data 回傳的轉換值有限，但 std=0；用回傳的 (x-mean)/std 重播產生 NaN。這個問題在把 std 當除數重播時觸發；應保存實際安全 scale 或明確處理零變異。

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

來源：`features.py:64`。產生空／不一致的序列及錯位標籤；window=0 的第一個標籤會索引最後一列。這些輸入應在建立序列前明確拒絕。

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

## 通過的檢查代表什麼

調整介面後的 62 項舊測試全部通過；40 項新增測試中 18 項通過、22 項未通過。通過內容包括各股票獨立建窗、排序與最後一列 Target 對齊、所有 batch 預測串接、樣本加權 loss、eval/no-grad、權重正負與總和、日均 Spearman、現行 Sharpe 公式，以及真正 PyTorch 序列化的參數與 metadata 往返。

`feature_columns` 已出現在 model_2 的存檔呼叫，測試通過；先前缺少該參數的問題已不存在。模型搬到 models.py 後也能正常建構與 forward。

## 設計觀察，未列為 bug

- 42 日輸入配 kernel_size=42，LSTM 收到的時間長度為 1。是否希望它學多個時間步，需要先決定架構。
- 每個 fold 沿用同一個模型與 optimizer；這是暖啟動選擇，不能直接認定錯誤。
- 驗證資料分開建 42 日窗，每檔前 41 列不評分；60 列合成資料只有 19 筆預測。是否應借用訓練尾段歷史，要先決定評估範圍。
- main 目前呼叫 model_2 進行訓練；沒有自動切換至已存模型。未把此入口選擇另外算一個 bug。
- 目前未實作 early stopping；本次沒有把尚未加入的功能當作程式錯誤。

## 測試修訂與限制

原版 62 項測試在首輪來源得到 55 通過、2 失敗、5 error。5 個 error 是舊模型位置，2 個失敗是舊 list 回傳假設；這 7 項不能當成目前程式的 bugs。v3 的副本更新為 models.CNN_LSTM 與 dict 分數，並將模型建構測試的資料替身改放在 common_process 邊界：原本使用 2024 年資料，會被 model_1 的 2017–2021 範圍排光。原始 v1/v2 檔案未修改。

安裝前偵測到使用者修改 features.py，新增必填 std: None。修改前的來源、報告與 101 項結果保存在 before_concurrent_edit；最新來源改為 102 項測試。B16 用真實 common_process 呼叫重現；其他標準化測試在函式已有必填 std 時明確傳入 std=None，前處理下游測試也僅補入該參數，以免 B16 遮蔽 B03、B08，並未修改待測程式。

第一輪開發結果保留於 development_results_first_run；其中包含上述 2024 年測試 fixture 錯誤，最終判定請以 results 與本報告為準。後段存檔／報表測試刻意替換訓練，避免 B01 遮蔽其他錯誤；真實 train 呼叫由獨立 smoke tests 驗證。序列化使用記憶體，沒有覆蓋任何現有 checkpoint。

本次沒有量測行覆蓋率，也沒有證明所有 bugs 都已找出。尚未驗證真正拆股／除息生效日與 ExpectedDividend 口徑、Target 到期時間、完整 JPX 市場池與停牌缺口、同分排名政策、MPS/CUDA、訓練資源耗用及真實績效。輸入防護案例的失敗，不代表你的真實資料已含那些無效值。

## 重跑

```bash
python3.12 -B /Users/coolguy/Desktop/JPX/debug/JPX_debug_v3/run_tests.py
```

只看單一問題時，在後面加上完整測試名稱，例如 `test_audit.TrainingContractTests.test_epoch_twenty_runs_twenty_complete_passes`。每次修正後應以新的結果目錄保存紀錄；本次的失敗版來源保留於 source_snapshot。
