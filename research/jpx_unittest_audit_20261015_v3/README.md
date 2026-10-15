# 最後修訂：2026-10-15 11:24

最新結果以 [latest_revision_1124/REPORT.md](latest_revision_1124/REPORT.md) 為準：104 項，81 通過、19 項 assertion failure、4 項 error，17 組問題。來源與雜湊同在 latest_revision_1124；原 102 項來源與結果保留於以下原紀錄，且已保存於 commit dee9551fb66ccef385bfba03a169e5b78d3fc946。重跑目前程式請看 [測試入口說明](JPX_debug_v3/README.md)。

---

# JPX unittest audit v3 — 2026-10-15

執行入口、完整問題清單與重跑方式見 [JPX_debug_v3/README.md](JPX_debug_v3/README.md) 與 [報告](JPX_debug_v3/REPORT.md)。

本次新增 40 項 unittest，加上更新介面的 62 項回歸測試，共 102 項；80 通過、18 項 assertion failure、4 項 error。22 個失敗方法對應 16 組問題：8 組主流程與 8 組邊界／輸入防護問題。

## 保存內容

- `JPX_debug_v3/`：已安裝到 Desktop 的新測試、說明與執行結果。
- `source_snapshot/`、`source_manifest.json`：最後一次受測來源的 9 個原始碼檔案及逐檔容量／SHA-256。
- `results/`：最後一次工作區執行。`installed_results/`：從 `/private/tmp` 直接執行已安裝入口、使用預設來源路徑的驗證；結果完全一致。
- `before_concurrent_edit/`：使用者在執行中新增 `std: None` 之前的來源、101 項結果與當時報告；保留而不混入最終結論。
- `baseline_results/`：原本 v1/v2 的 62 項結果，原有介面錯位已標示，不能誤算成新 bugs。
- `development_results_first_run/`：測試開發第一輪結果，包含後來修正的 fixture 日期問題。
- `issue_catalog.json`：所有失敗方法與根因群組的完整對照。
- `install_v3.py`、`installed_verification.json`：僅新增 debug/v3 的安裝與逐檔比對；既有 codes、debug/v1/v2 皆未被我們修改。
- `sync_manifest.json`、`sync_status.json`：雲端保存範圍及遠端取回校驗紀錄（完成同步後寫入）。

`legacy_tests/` 是本機讀取舊版測試用的參考副本，沒有新增內容，也不屬於本輪同步範圍；原始測試仍在 Desktop。v3 的已更新測試副本已完整保存，取回與重跑不依賴 legacy_tests。

## 同步與資料範圍

新版本分支：`codex/jpx-unittest-audit-20261015-v3`；目錄：`research/jpx_unittest_audit_20261015_v3`。沒有下載市場資料或 Release，沒有使用 holdout test；所有新成果都是小型程式、合成測試結果與文字報告，無須建立大型資料 Release。

本次沒有刪除市場資料、來源檔或其他任務檔案。測試序列化資料只存在記憶體；本機保留輕量測試、版本快照與報告。
