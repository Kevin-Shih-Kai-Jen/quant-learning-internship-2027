# JPX bug fixes — 2026-10-15 v6

本版本一次修正先前已確認的程式問題，並將 unittest 更新為目前函式介面。原始程式及舊測試完整保存在 `before/`、`tests_before/`，最終程式及測試在 `after/`、`tests_after/`。

## 修正內容

- `deep_learning.py`：移除未使用且引用未定義 FEAUTRES 的重複 common_process 和 return 後的死碼；修正 Tensor 型別註記；明確拒絕非正整數 epochs。保留使用者已修正的 range(epochs) 與必要 model_path 參數。
- `evaluation.py`：修正會造成模組匯入失敗的 np.array[float]；回傳型別改為 dict[str, float]；統一 earnings 欄位；保留新版 daily_earning 介面，使用 float64 避免極小的非零變異被截斷；拒絕非有限預測、缺少日期或非有限權重／Target；驗證預測維度、數量和視窗長度。
- `features.py`：拒絕重複股票日期及非正整數視窗；缺欄位訊息只列真正缺少的欄位。
- `data_processor.py`：直接檢查每列 AdjustmentFactor，避免最後一列的非正值被 shifted cumulative product 漏掉；拒絕無效的累乘係數。
- `main.py`：use_loaded_model(model_path) 接收明確的路徑，並回傳模型、mean、std、欄位及訓練日期。
- `JPX_deep_learning_models.py`：將累加的每日報酬正確格式化為百分比，標明是每日報酬總和。這不是複利報酬或經交易成本調整後的實際帳戶績效。

模型架構、Adam、L1Loss、20 epochs、八個資料切分、跨 fold 沿用模型的設計，以及只用訓練資料估計標準化參數的流程均保留。

## 測試更新與結果

- 舊的 epoch= 呼叫改為 epochs=；載入模型測試傳入明確路徑並檢查回傳。
- 測試新版 daily_earning，不重新加入已移除的 one_day_earning。
- 更新 earnings 回傳契約與測試替身，移除過時的 normalize_data 介面相容轉接。
- 新增六個回歸測試，涵蓋收益百分比、逐日多空加總、非法值和非法 epochs／視窗。
- 目前候選版本 110/110 通過，無跳過、失敗或錯誤。實際安裝後結果另存 `installed_results/`。
- 修正前四個測試模組皆因 evaluation.py 型別註記而無法匯入；這是四個匯入錯誤，不能解讀成只測了四項功能。

測試只使用小型合成資料和 CPU，含真實 PyTorch 訓練及記憶體內序列化。未讀取市場 CSV、未執行完整回測、未測試 MPS/CUDA，不能保證所有可能 bug 均已排除。

## 執行測試

```sh
python3.12 -B /Users/coolguy/Desktop/JPX/debug/JPX_debug_v3/run_tests.py
```

重現本封存版本：從本目錄執行以下指令。

```sh
python3.12 -B tests_after/run_tests.py --codes after
```

## 模型路徑

save_model 會在目前工作目錄的 checkpoints 子目錄存檔，並印出完整路徑。以程式印出的 `Model is saved: ...` 後方路徑為準，再傳給 use_loaded_model。若在 `/Users/coolguy/Desktop/JPX` 執行訓練，檔案名稱為 CNN_LSTM，範例是：

```python
from main import use_loaded_model

model, mean, std, columns, start, end = use_loaded_model(
    "/Users/coolguy/Desktop/JPX/checkpoints/CNN_LSTM"
)
```

這只是路徑範例，本次未執行完整訓練或建立正式 checkpoint。若從 codes 目錄執行，則位於 codes/checkpoints/CNN_LSTM。

## 保存與清理

小型程式、原始版本及測試紀錄另存 GitHub 新版本分支，保留既有研究和失敗紀錄。未取回大型 Release 資料，未刪除任何使用者資料或既有檔案。同步後逐檔比對遠端內容、容量和 SHA-256，詳見 sync_manifest.json、sync_status.json。
