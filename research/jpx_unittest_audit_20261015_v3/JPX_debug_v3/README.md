# JPX unittest v3

最後測試版本：2026-10-15 11:24（台北）。**104 項：81 通過、19 項 assertion failure、4 項 error。** 23 個未通過方法對應 17 組問題：7 組主流程、10 組條件性／邊界問題。完整清單與證據見 [REPORT.md](REPORT.md)，原始輸出見 [results_1124/test_results.txt](results_1124/test_results.txt)。

執行目前的 codes：

```bash
python3.12 -B /Users/coolguy/Desktop/JPX/debug/JPX_debug_v3/run_tests.py
```

只跑某一組：

```bash
python3.12 -B /Users/coolguy/Desktop/JPX/debug/JPX_debug_v3/run_tests.py test_audit.TrainingContractTests
python3.12 -B /Users/coolguy/Desktop/JPX/debug/JPX_debug_v3/run_tests.py test_normalizer_reuse.NormalizerReuseTests
```

使用 `--report-dir /新結果目錄` 可保存紀錄；用 `--codes /指定程式目錄` 可重播封存快照。執行器會印出實際來源、記錄依賴版本與 SHA-256，且拒絕把零個測試算成成功。退出碼 1 表示測試未全數通過。

原本 v1/v2 保留不動。v3 是 62 項更新介面的回歸測試加上 42 項新增測試；沒有修改 production code，沒有使用 expectedFailure 隱藏問題。使用 CPU 合成資料與記憶體序列化，不讀 CSV、不覆蓋 checkpoint、不啟動完整歷史訓練。

你在執行期間修改了 normalize_data；目前 std 預設 None 已解除原本缺參數問題，但新分支的回傳與逐特徵參數重用仍有失敗。報告只代表上述固定快照。舊 102 項結果仍在 results/；最後結果在 results_1124/。

測試不能保證涵蓋所有 bugs。真實資料語意、Target 到期、GPU 和長時間訓練等未驗證範圍已在完整報告列出。
