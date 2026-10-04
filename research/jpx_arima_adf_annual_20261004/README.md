# 5% ADF 差分與年度選階

本輪由使用者於 2026-10-04 明確授權實測：每檔股票以 ADF 5% critical value 選 d，所有股票共用依歷史成熟樣本外績效選出的年度 p,q。正式 test 保留，正式基準不變。

- [實驗計畫](experiment_plan.md)、[設定](config.json)
- [績效報告](REPORT.md)、[策略摘要](strategy_results.json)
- [缺價覆蓋機制對照計畫](coverage_diagnostic_plan.md)
- [數值與時間順序核對](numerical_checks.json)、[估計失敗紀錄](fit_audit_summary.json)
- [ADF 區段限制](adf_segment_diagnostics.json)
- [來源本機核對](source_local_verification.json)、[來源遠端取回核對](source_remote_verification.json)
- [同步狀態](sync_status.json)、[實際清理紀錄](cleanup_receipt.json)

## 重現

來源 9/28 ARIMA Release 本次查得仍是 draft。新版本會保存這輪需要的來源價格、原 Target／日曆、固定 d=1 預測和係數，以及新模型全部過程。取得本輪新快照後，不需要依賴旧草稿才能重現本輪。

使用根 repo 的 `scripts/data_archive.py`，指定 `snapshot-2026-10-04-arima-adf-annual` 及前綴 `jpx_arima_adf_annual_20261004`，在新的專屬暫存目錄 fetch／verify。實際發布狀態必須以 `sync_status.json` 和根 `data/index.json` 為準。

不要覆寫本次歷史結果。將此目錄的程式、設定、計畫及小型核對文件複製到 `research/` 下另一個新的重現目錄，然後執行：

```sh
python3 configure_from_snapshot.py --snapshot /verified-download/research/jpx_arima_adf_annual_20261004
```

這會檢查 54 個來源檔的大小／SHA-256，建立新的暫存 runtime（`runtime.json` 不入 Git），保留唯讀快照；不讀取正式 test。接著在該新程式目錄執行：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 prepare_adf.py --workers 6
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 fit_adf_grid.py --pilot --workers 6
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 fit_adf_grid.py --workers 6
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 check_numerics.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 evaluate.py
MPLCONFIGDIR=/tmp/jpx-arima-mpl python3 build_report.py
```

程式依同一 Git checkout 中的旧官方 metric、既有 d=1 每日結果及日曆做獨立對照。依賴版本見 `environment.json`。部分運算可能受平台、BLAS 及最佳化數值差異影響；本次没有完成跨平台重建。

原始估計使用的 common／ADF／fit runner 保存於 `execution_sources/`，`execution_code_hashes.json` 記錄該階段雜湊。後續 common 僅增加可移植來源路徑覆寫，預設值和實際估計方法不變；最終交付程式另存最終雜湊。

## 大型證據內容

- `shared_inputs/`：原價格、原 Target keys、日曆、原 labels／v7 pickle。
- `fixed_d1/`：25 組原預測、名次、後備與係數；完整來源可重取。
- `adf/`：25 組每檔 ADF 差分後的全期預測、名次、後備、每日指標與新係數。
- `new_fits.sqlite`：11,275 次新估計的參數、兩步價格預測、逐年狀態與失敗日志。
- `adf_decisions.json`、CSV：8,000 個股票年度的全部檢定、ACF／PACF 與差分選擇。
- `selectors/`：逐年選模明細、逐日預測、持倉權重、兩條匹配對照及換手。

所有變更採新版本；不刪原外部研究工作區、同步唯讀材料或其他任務的檔案。清理只處理本輪專屬暫存目錄，且必須在新成果同步、遠端大小／SHA-256 及取回核對完成後。
