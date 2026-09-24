# Quant Learning & Internship 2027

量化研究、實驗過程與實習準備的私人保存庫。以 **JPX 股票預測研究**為主線，同時保留產業研究、簡報、文件與網站專案。

這裡把閱讀材料與大型資料分開保存：**Git 管理程式、報告、設定與索引；版本化 GitHub Releases 保存大型資料及二進位產物。** 單獨下載程式碼 ZIP 不等於取得完整研究資料。

## 從這裡開始

| 你想做什麼 | 入口 |
|---|---|
| 用最少內容接續研究 | [最小續接記憶](research/project_archive_20260924/MEMORY_BRIEF.md) |
| 了解每次嘗試、結果與修正 | [完整實驗歷程](research/project_archive_20260924/EXPERIMENT_HISTORY.md) |
| 找到特定研究、成品或資料夾 | [研究地圖](docs/RESEARCH_MAP.md) |
| 查每個實驗的資料量與精確下載前綴 | [實驗與資料下載索引](docs/EXPERIMENT_INDEX.md) |
| 下載、驗證、還原實驗資料 | [資料指南](docs/DATA_GUIDE.md) |
| 讓下一個對話從 GitHub 接手 | [可直接貼上的續接指令](docs/FUTURE_CHAT_PROMPT.md) |
| 查看協作與保存規則 | [AGENTS.md](AGENTS.md) |
| 私人 repo 讀不到或新聊天室無法連線 | [存取說明](docs/ACCESS.md) |
| 確認某次快照包含什麼 | [資料索引](data/index.json) · [完整檔案盤點](data/inventories/snapshot-2026-09-24.json) · [Release 資料目錄](data/catalogs/snapshot-2026-09-24.json) |
| 查上傳與完整性核對證據 | [首次快照驗證紀錄](data/verification/snapshot-2026-09-24.json) |

## 研究目前在哪裡

| 狀態 | 內容 |
|---|---|
| **正式基準** | `v7_equal`：11 個價量特徵＋截距；Target 到期後，每日全股票等權普通 MSE 一次更新；953 日歷史驗證的官方未年化 Sharpe **+0.00767887**。 |
| **最新完成實驗** | `v30`：營業利益實際年增與 Forecast 年比，七組共同／固定價量模型對照。最新候選 `frozen_r` Sharpe **+0.00435804**，未取代 v7、未使用 test。 |
| **未完成支線** | 2026-09-23 ARIMA＋殘差 ACF：已做資料與公式核對，正式回測仍待建模選項確認；合成梯度檢查不是 JPX 績效。 |
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

登入可存取此私人儲存庫的 GitHub 帳號，取得專案後，在儲存庫根目錄執行：

```sh
python3 scripts/data_archive.py list
python3 scripts/data_archive.py fetch --experiment jpx_v30_profit_actual_forecast_20260916
```

單一實驗可能引用較早的共同特徵與封存檔；重跑前依 [資料指南](docs/DATA_GUIDE.md) 一起取回依賴。歷史程式的本機絕對路徑、套件版本、macOS 動態函式庫與字型仍可能需要適配，這份保存不承諾跨電腦直接執行。

此儲存庫保存可取得的專案檔案，**不包含未匯出的 ChatGPT／Codex 全部對話、瀏覽器排班資料或帳號狀態**。原封存文件中的「尚未上傳」是本機整理當時的紀錄；目前遠端狀態請看版本化索引與驗證紀錄。
