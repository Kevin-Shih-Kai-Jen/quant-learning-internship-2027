# JPX unittest v3

這版測試直接讀取目前的 `JPX/codes/`。新增 40 項稽核，並把原本 62 項測試的副本更新為目前介面，共 **102 項**。原本的 v1、v2 和統一入口保持原樣。

## 執行

```bash
python3.12 -B /Users/coolguy/Desktop/JPX/debug/JPX_debug_v3/run_tests.py
```

只看訓練問題：

```bash
python3.12 -B /Users/coolguy/Desktop/JPX/debug/JPX_debug_v3/run_tests.py test_audit.TrainingContractTests
```

保存新結果時可加上 `--report-dir /你選擇的新結果目錄`。測試會印出實際 Python 與待測路徑，也會記錄模組路徑、版本、每個來源檔的 SHA-256，以及執行前後是否改變。退出碼 1 代表發現失敗或錯誤，不代表測試程式沒有執行。

從 GitHub 取回時，請明確指定封存程式：

```bash
python3.12 -B JPX_debug_v3/run_tests.py --codes source_snapshot --report-dir replay_results
```

## 本次結果

**102 項：80 通過、18 項 assertion failure、4 項執行 error、0 跳過。** 其中一項測試的兩個子案例都失敗，因此 unittest 原始輸出會顯示 `failures=19, errors=4`；這是 22 個失敗的測試方法，不是 23 個獨立問題，也不是 22 個獨立 bugs。

22 個測試方法對應 16 組問題：8 組主流程問題，8 組在特定輸入、重播方式或數值條件下出現的邊界問題。完整證據、來源位置及對應測試見 [REPORT.md](REPORT.md)；原始結果見 [results/test_results.txt](results/test_results.txt) 和 [results/test_results.json](results/test_results.json)。

新增檔案只有測試與說明；`codes/` 九個程式的內容與邏輯均未更動。測試沒有把已知錯誤標成 `expectedFailure`，因此修正程式前保留紅燈是正常的。

## 測試範圍

- 合成資料、CPU、小模型；不讀取真正的股價 CSV，也不啟動完整訓練。
- 序列／標籤／預測對齊、資料篩選、訓練與評估迴圈、排名權重、Sharpe、Rank IC、存取模型，以及跨模組呼叫。
- 存取測試使用真正的 PyTorch 序列化與反序列化，暫存在記憶體；沒有建立或覆蓋你的 checkpoint。
- 後段流程測試會明確替換訓練及評分，避免前段的 `epochs` 錯誤遮蔽存檔或報表問題；它們不等於完整回測。
- `DesignObservationTests` 是現況紀錄，不代表這些模型設計已經被認定為 bugs。

尚未涵蓋真實市場資料語意、拆股生效日、Target 可知時間、MPS/CUDA、長時間訓練、所有異常輸入與完整投資績效。測試通過不能證明沒有其他 bugs。

執行期間使用者新增 `normalize_data(..., std: None)`，新版另有真實呼叫測試捕捉其缺參數錯誤。其他標準化測試僅補傳 `std=None` 以隔離下游問題；這個替身行為已在程式與報告中註明。修改前的 101 項結果另行保留。
