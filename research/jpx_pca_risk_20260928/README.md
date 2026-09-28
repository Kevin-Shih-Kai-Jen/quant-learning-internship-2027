# JPX PCA 風險診斷（2026-09-28）

先閱讀 [中文報告](JPX_PCA_report.md)。下載 [互動 HTML](JPX_PCA_report.html) 後直接用瀏覽器開啟，無需網路。最終數值在 `final_results/`。

原始 ZIP 完整。本次第一個解壓暫存副本不足原始長度，已拒用；最終直接從 ZIP 讀取 CSV，雜湊、全部 2,332,531 列、OHLCV 與累積調整因子核對通過。不是使用者原始資料損壞。

正式基準維持 v7_equal。使用 2021-12-01 當日保存權重；這是固定持倉歷史共變異數分析，不是每日調倉績效回測或樣本外結果。完整持倉版取共同可觀測 160 日，不能當成完整 252 日估計。

## 重跑

需要 Python、NumPy、pandas、matplotlib。實際版本記於 `final_results/results.json`。

1. 使用儲存庫 `scripts/data_archive.py fetch/verify`，tag `snapshot-2026-09-26-v32-chunk-replay`，取回：
   - `jpx_v7_daily_mse_20260912/v7_equal`。
   - `project_archive_20260924/compressed_data/jpx_stock_returns_20260910/bars.pkl.gz`。
2. 依 `docs/DATA_GUIDE.md` 使用 `restore_data.py` 還原 bars.pkl，核對 gzip 與原檔 SHA-256。
3. 將這些檔案保留在專屬 INPUT/archive/research 下，stock_list.csv 放在 INPUT 下。`--zip` 指向使用者提供的 raw.zip，內部路徑是 `raw/train_files/stock_prices.csv`。不要用不完整的解壓副本替代。

```sh
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python run_analysis.py \
  --input /absolute/INPUT \
  --zip /absolute/raw.zip \
  --out /absolute/NEW_OUTPUT
python -m unittest test_jpx_pca.py -v
```

程式拒絕覆寫輸出目錄及不符合預期雜湊的資料。SVD 用全部非零方向計算風險；詳細股票解讀輸出前 10 個方向的兩端主要 loading。沒有以截斷 PCA 近似全部持倉變異數。

`build_report.py` 讀取本目錄 `final_results` 產生 Markdown、HTML 與 SVG。更換輸入結果目錄時，另開版本並修改 builder 的 OUT；不要覆蓋已保存結果。

## 檔案

- `run_analysis.py`：來源核對、報酬矩陣、兩種樣本、所有風險分解。
- `jpx_pca.py`、`test_jpx_pca.py`：數學核心與 8 項單元測試。
- `experiment_plan.md`：固定設定與缺值處理。
- `final_results/results.json`：完整特徵值、曝險、風險占比、主要 loading、產業描述及數值核對。
- `*_components.csv`：所有可識別方向的市場占比及持倉曝險／風險。
- `*_dates.csv`、`*_universe.csv`、`excluded_252.csv`、`omitted_joint_dates.csv`：樣本選擇。
- `holdings_20211201.csv`：實際 400 個資本權重，來源可追溯至原封存。
- `verification.json`：驗證範圍、數值核對與本次暫存副本問題的更正。

`results/` 是首次輸入檢查時留下的核對記錄；`results_v2/` 是相同數值、只輸出前 5 個方向 loading 的中間版本。`final_results/` 補齊前 10 個方向以解釋風險最高的 PC6，估計方式與數值沒有改動。這些歷程保留，最終解讀以 final_results 為準。

本研究未建立新的大型資料；重跑所需原始資料沿用既有 Release。沒有改變 alpha 模型、持倉、舊實驗、保留 test 或儲存庫可見性。
