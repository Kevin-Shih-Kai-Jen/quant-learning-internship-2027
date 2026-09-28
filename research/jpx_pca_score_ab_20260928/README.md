# JPX PCA 分數消融（2026-09-28）

先讀 [中文報告](JPX_PCA_AB_report.md)。[HTML](JPX_PCA_AB_report.html) 可下載後直接用瀏覽器閱讀，圖表已內嵌；內文資料連結需保持目錄結構或在 GitHub 開啟。

原預測 A 與 B 都使用原 JPX 多空各 200 檔、側內 2→1 配重。B_PC6 每期移除第六方向；B_MAX 每期移除原持倉被 PCA 涵蓋部分的最大估計風險方向。另核對原報告 2021-12-01 的同一 PC6，僅有一日評分。

最終輸出是 `results_v2/`，原 `results/` 是第一輪生成後 2019 NPZ 工作檔寫入不完整的失敗版，已拒用並保留。沒有原始資料損壞。這是 953 日既有 validation 上的探索診斷，未使用保留 test，不替換正式 v7_equal。

## 重跑

依 repo `docs/DATA_GUIDE.md` 按需取回並驗證 `jpx_v7_daily_mse_20260912/v7_equal`，明確指定 catalog tag `snapshot-2026-09-26-v32-chunk-replay`；實體來源資產在 `snapshot-2026-09-24`。原訓練行情 ZIP 使用 `raw/train_files/stock_prices.csv`，亦支援此路徑之前有其他目錄前綴的歷史封存。只讀 train，不讀保留 test。

環境版本詳見 `results_v2/generation_audit.json`。主要依賴 NumPy、pandas、SciPy；報告另需 matplotlib、markdown-it-py 3.0.0。

```sh
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python -m unittest test_ab.py -v
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python run_ab.py \
  --input /absolute/TASK_INPUT --zip /absolute/raw.zip \
  --out /absolute/NEW_OUTPUT --stage all
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python exact_snapshot.py \
  --input /absolute/TASK_INPUT --zip /absolute/raw.zip \
  --out /absolute/NEW_SNAPSHOT_OUTPUT
```

`run_ab.py` 拒絕覆寫輸出。`generate` 階段不讀 Target，保存全部預測後才執行 `evaluate`；若評分中斷，可用 `--stage evaluate` 讀同一份已驗證預測，不必重新產生。`build_report.py` 讀本目錄的最終輸出；新實驗請另開版本並調整輸入位置。

NPZ 使用 `allow_pickle=False` 讀取；含日期、股票代碼、三組分數／rank、兩個選中方向的逐股 loading、PCA 涵蓋旗標、全部非零特徵值與三組持倉曝險。未在當日排名股池的 rank 是 65535；未估計 loading 是 NaN，不能視為零風險。

保存與 GitHub 同步狀態見 [sync_status.json](sync_status.json)。大型檔案需 Release，不能因能讀報告就聲稱全部逐筆紀錄已保存上遠端。發布未完成前保留本次全部輸出與下載資料。
