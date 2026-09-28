# JPX ARIMA 學習與回測紀錄

本目錄保存七輪 ARIMA 研究：從價格差分與歷史誤差預測，到 25 組階數比較、殘差 ACF 修正及 Target MSE 共同訓練。使用者提出研究方向，助理負責程式實作、回測與核對；研究紀錄保留原結果、修正及失敗資訊，未自動取代原 v7 模型。

這是 **2026-09-28 歸檔版本**，不是重新回測日期。

- 歸檔路徑：`research/jpx_arima_learning_20260928`
- 取回版本：[`snapshot-2026-09-28-arima-learning`](https://github.com/Kevin-Shih-Kai-Jen/quant-learning-internship-2027/releases/tag/snapshot-2026-09-28-arima-learning)
- 原始報告保留不改；`reports/` 為修正相對圖片連結的閱讀複本。
- 大型資料從版本 Release 按需取回，依索引核對 SHA-256。

## 七輪實驗

所有表中 Sharpe 均未年化、未扣交易成本。同期 v7_equal Sharpe 為 **+0.007679**。

| 輪次 | 實驗 | 主要結果與判讀 |
|---|---|---|
| 1 | [ARIMA(5,1,5)，rolling 252 日](reports/JPX-ARIMA515-report.md) | Sharpe **−0.019735**。首輪探索，不能直接推論整個 ARIMA 家族有效或無效。 |
| 2 | [ARIMA(1,1,0)，expanding](reports/JPX-ARIMA110-expanding-report.md) | Sharpe **−0.035775**。對齊 JPX「明天→後天」目標，未優於同期 v7。 |
| 3 | [ARIMA(0,1,1)，expanding](reports/JPX-ARIMA011-expanding-report.md) | 無漂移設定下，兩步價格點預測相同，953 日分數全部為零；Rank IC 未定義。代碼同分排序對照 Sharpe **−0.028491**，不是模型預測能力。 |
| 4 | [ARIMA(0,1,2)，expanding](reports/JPX-ARIMA012-expanding-report.md) | 可產生不同排名分數；Sharpe **+0.003571**，仍低於 v7。非零訊號不等於有用訊號。 |
| 5 | [ARIMA(r,1,s) 全部 25 組](reports/JPX-ARIMA-grid-report.md) | r、s 各取 1～5，共 200,000 次股票年度估計。第一名 **(3,1,1)：+0.075442**；第二名 **(5,1,1)：+0.060424**。固定網格的近似同時區間未確認任何一組穩定勝過 v7。 |
| 6 | [固定原 ARIMA，事後加入殘差 ACF](reports/JPX-ARIMA-ACF-report.md) | (3,1,1)、(5,1,1) 修正後 Sharpe 分別 **+0.069266、+0.058801**，整段績效均未改善；差異區間跨零。 |
| 7 | [Target MSE 與 ACF 共同訓練](reports/JPX-ARIMA-joint-report.md) | 保留既有 A、B，只新增訓練 C、D。更換訓練目標與共同訓練均未改善整段 Sharpe；詳見下表。 |

首輪 (5,1,5) 採 **rolling 252 日**；第二輪起依使用者要求採 **expanding window、逐年驗證**。網格內的 (5,1,5) 是另行估計的 expanding 版本，Sharpe 為 **−0.047762**，不可與首輪結果混為同一版本。

## 最新四組對照

| 組別 | 方法 | ARIMA(3,1,1) Sharpe | ARIMA(5,1,1) Sharpe |
|---|---|---:|---:|
| A | 原 Gaussian 最大概似 ARIMA，無 ACF 修正 | **+0.075442** | **+0.060424** |
| B | 固定 A 的參數，加入殘差 ACF，修正強度固定 1 | +0.069266 | +0.058801 |
| C | 從 A 出發，以 Target MSE 最佳化 AR／MA 係數，無 ACF | +0.043554 | +0.001237 |
| D | 與 C 相同目標，同時調整 AR／MA 係數及 ACF 修正強度 | +0.013702 | −0.001954 |

A、B 沿用已完成的預測，未重訓；兩個 ARIMA 階數分開比較，沒有合成投資組合。

**D−C 是加入共同訓練 ACF 的主要對照**：兩者損失函數相同，都是原始 JPX Target 與預測報酬的 MSE。ACF 是目前候選 ARIMA 殘差的統計量，不是另一套獨立自由係數；D 額外學習 λ₁、λ₂，範圍為 0～1。

D 的整段 Sharpe 低於 C，且兩項 D−C 的近似同時區間仍包含零。訓練 MSE 改善沒有延續到驗證資料，不能以訓練損失下降宣稱模型更有效。

## 資料與模型相依關係

原始 JPX `train_files/stock_prices.csv` 與既有參考專案，提供因果調整價格、原始 Target、v7 基準及官方評分實作。四個早期單模型實驗使用這些共同來源，各自保留結果與稽核。

後續相依鏈為：

```text
ARIMA(0,1,2) 的共同輸入快取
    └─ 25 組 expanding ARIMA 網格
         ├─ A：原始 (3,1,1)、(5,1,1) 參數與預測
         ├─ B：沿用 A，計算訓練殘差 ACF 後修正
         └─ C、D：從 A 初始化，以已成熟訓練 Target 最佳化
                    └─ 四組比較沿用 A、B，加入新 C、D 結果
```

網格重用的是相同資料準備結果，並非把 MA(2) 的預測當成新模型輸入。重跑 ACF／共同訓練需要前輪網格參數與共同資料；只取得最新共同訓練 ZIP 並不足以獨立重建所有實驗。程式保留歷史本機路徑，搬到其他電腦時須依資料索引調整來源位置。

## 評估規則與限制

- 共同驗證資料為 **953 個訊號日、1,864,363 個股票日**，ValidationYear 2018～2021；2021 是截至 2021-12-01 的部分年度。這批驗證資料已反覆用於選模，不能當成新的獨立確認證據。
- **正式 test 未讀取、未評分，繼續保留。** 要確認可重複的預測價值，應先固定研究決策，再使用未參與選模的資料。
- 訊號在 t 收盤後產生，分數為 `P̂[t+2|t] / P̂[t+1|t] − 1`，對齊明天收盤到後天收盤。這是點預測比值代理，不宣稱等於隨機報酬比值的精確條件期望。
- Expanding 實驗每年重新估計係數；年內只隨已發生價格更新 forward states。含 Target 的訓練只使用截止當時已實現的標籤。
- 沿用 JPX 完整唯一排名、同分按股票代碼、前後各 200 檔及 2→1 權重；不自行選擇投入金額。失敗或非法預測依事前規則後備，不刪除股票或隱藏失敗紀錄。
- ACF 修正未改善本次整段結果，不代表所有殘差修正方式都無效；MA／殘差也不等於已辨認的新聞衝擊，研究未提供 shock 的因果證明。
- C、D 為固定預算的局部最佳化，沒有全域最優保證。D 的 (5,1,1) 約 **7.7%** 最佳化工作未回報收斂；15,397 筆 C、D 配對中，有 **1,103 筆** D 的訓練損失仍高於 C。這限制了對共同訓練方法的結論。
- Bootstrap 區間保留時間區塊，網格／共同訓練的同時區間只處理各自固定比較集合，未校正全部歷史探索；也不是未來年度 Sharpe 的預測區間。所有績效未納入交易成本與成交限制。

目前保留原模型與全部研究候選，不因驗證集名次自動替換 v7。

## 完整保存與取回

完整原始工作區保存在 `originals/` 及 `packages/`：四組大量零散的 `jobs/` 以無損 tar.gz 保存，兩個大於 700 MiB 的檔案分片保存，其餘依資料索引分配到 Git 或 Release。全部七份既有 ZIP、早期未對齊試跑、數值修正與失敗紀錄均保留。

`SNAPSHOT_FILE_MANIFEST.json.gz` 記錄逐一原始檔案的保存位置、重建對應與校驗值。`restore_original_workspace.py` 依 manifest 將檔案重建到本次專屬的新目錄；不要還原到仍在使用的工作區或覆蓋其他版本。

取回時，先取得此 tag 的 Git 可讀檔案，再使用根 repo 的 `data_archive.py` 工具：

從儲存庫根目錄執行（Python 3.9+、macOS／Linux）：

```sh
ARIMA_TASK=$(python3 -c 'import pathlib,tempfile; print(pathlib.Path(tempfile.mkdtemp(prefix="jpx-arima-")).resolve())')
mkdir -p "$ARIMA_TASK/research"
cp -R research/jpx_arima_learning_20260928 "$ARIMA_TASK/research/"
python3 scripts/data_archive.py fetch --experiment jpx_arima_learning_20260928 --tag snapshot-2026-09-28-arima-learning --dest "$ARIMA_TASK"
python3 scripts/data_archive.py verify --experiment jpx_arima_learning_20260928 --tag snapshot-2026-09-28-arima-learning --dest "$ARIMA_TASK"
python3 "$ARIMA_TASK/research/jpx_arima_learning_20260928/restore_original_workspace.py" --archive-root "$ARIMA_TASK/research/jpx_arima_learning_20260928" --dest "$ARIMA_TASK/original-workspace"
python3 "$ARIMA_TASK/research/jpx_arima_learning_20260928/restore_original_workspace.py" --archive-root "$ARIMA_TASK/research/jpx_arima_learning_20260928" --dest "$ARIMA_TASK/original-workspace" --verify-only
```

完整下載與重建會同時保留封存與原檔，預留至少 14 GB 可用空間；若只閱讀結論，無須執行這些指令。`--archive-root` 與 `--dest` 請使用解析後不含符號連結的實際路徑。

Release 取回內容須與 Git 中的可讀檔案一起複製到相同目錄結構，保留 `originals/`、`packages/`、manifest 與還原工具之間的相對位置，再執行 `restore_original_workspace.py`。下載完成後依索引核對 SHA-256；重建後依逐檔 manifest 再核對，不可只確認檔名或 ZIP 存在。

本快照保存本聊天室工作區；外部原始 JPX 市場資料及原參考專案仍依根 repo 資料索引中的對應版本取回。閱讀成果可先看本 README 與 `reports/`，需要重跑或深入稽核時才下載大型資料。

原始市場資料 ZIP 的既有版本是 `snapshot-2026-09-24`，前綴為 `project_archive_20260924/external_sources`；參考專案的各實驗依根目錄 catalog 取回。此新快照延續已驗證的歷史資料目錄，不重新上傳同一套外部原始資料。
