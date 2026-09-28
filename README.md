# Quant Learning & Internship 2027

量化研究、實驗過程與實習準備的公開保存庫（2026-09-28 依使用者指示改為公開）。以 **JPX 股票預測研究**為主線，同時保留產業研究、簡報、文件與網站專案。

這裡把閱讀材料與大型資料分開保存：**Git 管理程式、報告、設定與索引；版本化 GitHub Releases 保存大型資料及二進位產物。** 單獨下載程式碼 ZIP 不等於取得完整研究資料。

**雲端是歷史資料的主保存位置，本機採按需取用。** 每次任務把必要資料下載到專屬臨時目錄；新成果／變更先成功同步並核對遠端內容，再清理本次可重取的下載副本及暫存。本機長期保留輕量導讀、工具與索引。此規則不授權刪除未同步資料、唯讀 `sources/`、其他專案或共享使用中的檔案，詳見 [資料指南](docs/DATA_GUIDE.md)。

## 從這裡開始

| 你想做什麼 | 入口 |
|---|---|
| 用最少內容接續研究 | [最小續接記憶](research/project_archive_20260924/MEMORY_BRIEF.md) |
| 了解每次嘗試、結果與修正 | [完整實驗歷程](research/project_archive_20260924/EXPERIMENT_HISTORY.md) |
| 找到特定研究、成品或資料夾 | [研究地圖](docs/RESEARCH_MAP.md) |
| 查每個實驗的資料量與精確下載前綴 | [實驗與資料下載索引](docs/EXPERIMENT_INDEX.md) |
| 下載、驗證、還原實驗資料 | [資料指南](docs/DATA_GUIDE.md) |
| 讓下一個對話從 GitHub 接手 | [可直接貼上的續接指令](docs/FUTURE_CHAT_PROMPT.md) |
| 查看專案內所有聊天室的必讀規則 | [共用指令](docs/PROJECT_INSTRUCTIONS.md) · [已設定紀錄](docs/PROJECT_SETUP.md) |
| 查看本機清理結果與保留範圍 | [2026-09-24 清理紀錄](docs/LOCAL_CLEANUP.md) |
| 查看協作與保存規則 | [AGENTS.md](AGENTS.md) |
| 公開下載入口或新聊天室無法連線 | [存取說明](docs/ACCESS.md) |
| 確認某次快照包含什麼 | [資料索引](data/index.json) · [完整檔案盤點](data/inventories/snapshot-2026-09-24.json) · [Release 資料目錄](data/catalogs/snapshot-2026-09-24.json) |
| 查上傳與完整性核對證據 | [首次快照驗證紀錄](data/verification/snapshot-2026-09-24.json) |

## 研究目前在哪裡

**2026-09-28 新增：[ARIMA 七輪研究與完整資料](research/jpx_arima_learning_20260928/README.md)。** 已完成 25 組階數比較、殘差 ACF 修正與 Target MSE 共同訓練；舊 9/23 計畫中的「尚未回測」只代表當時進度。新成果獨立歸檔，保留失敗／修正與全部逐筆資料，正式基準維持 v7，test 未用。v31、v32 進度另見 [研究地圖](docs/RESEARCH_MAP.md)；下表 v30 是早期保存時的摘要。

| 狀態 | 內容 |
|---|---|
| **正式基準** | `v7_equal`：11 個價量特徵＋截距；Target 到期後，每日全股票等權普通 MSE 一次更新；953 日歷史驗證的官方未年化 Sharpe **+0.00767887**。 |
| **9/24 保存時的主線實驗** | `v30`：營業利益實際年增與 Forecast 年比，七組共同／固定價量模型對照。最新候選 `frozen_r` Sharpe **+0.00435804**，未取代 v7、未使用 test。 |
| **ARIMA 學習支線** | 後續七輪已回測：原 ARIMA(3,1,1)／(5,1,1) Sharpe +0.075442／+0.060424；事後 ACF 與本次共同訓練均未改善整段結果。沿用已反覆研究的 validation，不能當獨立確認；詳見新版本導讀。 |
| **現行選擇原則** | Sharpe 第一、Rank IC 第二；Rank IC 不必為正，MSE 不選模型；不依極端值排除訓練／驗證股票日、不裁切，test 留待最後總體驗證。 |

設定以 [現行預設](research/JPX-experiment-defaults.json)、[正式基準](research/jpx_active_baseline.json) 與 [版本登記](research/jpx_model_versions.json) 為準。上表是既有歷史研究的摘要，本次保存沒有重新訓練或驗證投資績效。v13 會計口徑錯配、EPS 拆季與來源差異等已知問題，均隨結果保留。

## 保存結構

```text
AGENTS.md                         協作及資料保存規則
docs/                             研究地圖、還原指南、續接指令
research/                         保留原專案相對路徑的可讀檔案
  jpx_*/                          計畫、程式、報告、設定及小型結果
  project_archive_20260924/        摘要、原封存清單與還原工具
data/
  index.json                      快照入口
  catalogs/<tag>.json              逐檔位置、大小與 SHA-256
  verification/<tag>.json          上傳／完整性核對紀錄
scripts/                          大型資料下載、驗證及新快照工具
GitHub Releases                   版本化 .tar 分包，不放進 Git 歷史
```

`research/` 的目錄結構對應原專案，但大型檔案須按目錄清單下載後才出現。首次快照標籤為 `snapshot-2026-09-24`；是否已完整存上遠端，以對應的驗證紀錄與 Release 資產核對為準，不能只憑 README 或本機壓縮完成判定。

首次快照已完成全部 **88 個分卷** 的遠端大小與 SHA-256 核對，收錄 **8,288 份大型資料／二進位檔案（約 11 GB）**；另有 **1,918 份原專案可讀檔案** 保存於 Git。下載入口：[完整資料 Release](https://github.com/Kevin-Shih-Kai-Jen/quant-learning-internship-2027/releases/tag/snapshot-2026-09-24)。已實際驗證 [從 GitHub 取回及 gzip 還原](data/verification/download-restore-smoke.json)，並保留 [原研究收錄核對](data/verification/original-research-coverage.json) 與 [Git 檔案逐份核對](data/verification/source-git-coverage.json)。

原本 139 個大型 CSV／PKL 已先做 gzip 無損封存，完整解壓 SHA-256 一致後才移除未壓縮副本；它們的內容仍保留。從 GitHub 取回 gzip 後，舊程式可能還需要第二步還原到原路徑。已有 NPZ、逐筆 trace、排名、修正前／失敗版，以及簡報、PDF、圖表與模板也屬保存範圍；確切收錄項目以快照目錄為準。

## 下載所需資料

本 repo 的檔案與已發布 Release 可免登入讀取；公開入口及 JPX 原始資料位置見 [存取說明](docs/ACCESS.md)。取得專案後，在儲存庫根目錄執行：

```sh
python3 scripts/data_archive.py list
QUANT_TASK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/quant-research.XXXXXX")"
python3 scripts/data_archive.py fetch --experiment jpx_v30_profit_actual_forecast_20260916 --tag snapshot-2026-09-24 --dest "$QUANT_TASK_DIR"
python3 scripts/data_archive.py verify --experiment jpx_v30_profit_actual_forecast_20260916 --tag snapshot-2026-09-24 --dest "$QUANT_TASK_DIR"
```

下載前檢查容量，並記錄本次目錄、tag 與取回清單。單一實驗可能引用較早的共同特徵與封存檔；重跑前依 [資料指南](docs/DATA_GUIDE.md) 一起取回依賴，在本次臨時目錄還原 gzip。歷史程式的本機絕對路徑、套件版本、macOS 動態函式庫與字型仍可能需要適配，這份保存不承諾跨電腦直接執行。

在此儲存庫工作的 Codex 應讀取 `AGENTS.md`。其他獨立 ChatGPT／Codex 聊天室須貼上 [續接指令](docs/FUTURE_CHAT_PROMPT.md)，或將它設為該專案指示；這不會自動改變所有聊天室的設定或存取權。

此儲存庫保存可取得的專案檔案，**不包含未匯出的 ChatGPT／Codex 全部對話、瀏覽器排班資料或帳號狀態**。原封存文件中的「尚未上傳」是本機整理當時的紀錄；目前遠端狀態請看版本化索引與驗證紀錄。
