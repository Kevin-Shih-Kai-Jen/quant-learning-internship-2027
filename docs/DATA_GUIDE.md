# 資料下載、還原與新快照

此私人儲存庫使用兩層保存。Git 保存可讀內容與索引；大型資料保存在按 tag 版本化的 GitHub Release `.tar` 分包，每包至多 768 MiB。每次使用都以 catalog 的實際檔案清單、大小和 SHA-256 為準。

首次 tag：`snapshot-2026-09-24`。入口是 [data/index.json](../data/index.json)，對應 [catalog](../data/catalogs/snapshot-2026-09-24.json) 和 [遠端驗證紀錄](../data/verification/snapshot-2026-09-24.json)。不要把建立 tag、推送 Git 或看到部分 Release 資產當成全量上傳完成。

## 1. 取得儲存庫與查看快照

需 Python 3，以及能存取本私人儲存庫的 GitHub 登入。下列指令從儲存庫根目錄執行；使用前可用各工具的 `--help` 核對目前介面。

```sh
gh repo clone Kevin-Shih-Kai-Jen/quant-learning-internship-2027
cd quant-learning-internship-2027
python3 scripts/data_archive.py list
```

先看目錄中有哪些實驗、哪些路徑在 Git、哪些屬 Release 資產，以及所需檔案大小。`research/` 對應原專案根目錄；僅 clone 不會取回所有大型檔案。

## 2. 按實驗取回，連同依賴一起處理

```sh
python3 scripts/data_archive.py fetch --experiment jpx_v30_profit_actual_forecast_20260916 --tag snapshot-2026-09-24
python3 scripts/data_archive.py verify --experiment jpx_v30_profit_actual_forecast_20260916
```

不帶 `--tag` 時依工具與 `data/index.json` 選定的快照操作；若要重現特定版本，明確指定並記錄 tag。`--experiment` 的前綴以 `list` 與 catalog 為準。

**下載一個實驗目錄，不等於它的輸入都已到齊。** 例如 v30 會引用 v7 程式、v8 價量輸入、v14／v17 財報特徵與 v27 狀態。先讀原計畫、程式輸入及 [相依查核](../research/project_archive_20260924/PRESERVATION_NOTES.md)，再取回相關兄弟目錄。原本封存的大型檔案還位於另一個前綴：

```sh
python3 scripts/data_archive.py fetch --experiment project_archive_20260924/compressed_data/jpx_v8_soft_rank_20260912 --tag snapshot-2026-09-24
```

原始資料 ZIP 與模板位於 `project_archive_20260924/external_sources/`，同樣按 catalog 取回。工具下載／驗證檔案不等於執行或重新訓練模型。

## 3. 將原 gzip 封存恢復成舊程式路徑

9/24 的本機保存已把 139 個大型 CSV／PKL 壓成 gzip；完整內容經解壓 SHA-256 核對後，才移除原位置副本。因此 GitHub 取回的有些檔案是 `.pkl.gz`／`.csv.gz`，還需要原封存工具還原。

```sh
python3 research/project_archive_20260924/restore_data.py --list
python3 research/project_archive_20260924/restore_data.py --restore --path jpx_v8_soft_rank_20260912/inputs.pkl
```

這會依 `COMPRESSED_DATA_MANIFEST.json` 把檔案恢復到 `research/jpx_v8_soft_rank_20260912/inputs.pkl`，並保留 gzip。也可以對某個已下載齊全的實驗前綴使用 `--path`。工具遇到同路徑不同內容會拒絕覆寫；不要為了繼續而任意刪掉現有檔案。

原工具 `--verify` 會核對整份 gzip 清單，若只按需下載部分封存，缺少其餘檔案不代表已下載的檔案損壞。此時可對所需檔案執行 `--restore --path`，它本身也會核對原始與壓縮內容。全部封存到齊後，再用 `--verify` 作全量核對。詳見 [原還原指南](../research/project_archive_20260924/RESTORE_GUIDE.md)，其中舊電腦絕對路徑需改成目前位置。

## 4. 全量取回與空間

```sh
python3 scripts/data_archive.py fetch --all --tag snapshot-2026-09-24
python3 scripts/data_archive.py verify --all
```

全量取回後如需恢復所有原 CSV／PKL，再執行：

```sh
python3 research/project_archive_20260924/restore_data.py --restore
```

最後一步可能額外需要約 **4.85 GB** 原始資料空間，且不刪 gzip；Release 下載、分包暫存與其他產物也需要空間，精確大小依 catalog 與工具輸出。硬碟有限時，優先取回本次工作所需資料。無損封存釋放的是硬碟空間，不會自動降低模型運行的 RAM 需求。

## 5. 保存新工作

先在新的實驗目錄完成本次研究，保存 plan、來源、設定快照、程式、完整結果、逐步紀錄、audit 與報告。必要時新增簡短記憶與導覽，記錄更正來源；不要重寫舊實驗事實。只有在使用者已授權同步本次產物時進行上傳。

可讀的小檔使用正常 Git 提交／推送。大型資料和可讀檔案快照由獨立工具處理：

```sh
python3 scripts/publish_snapshot.py --workspace /absolute/path/to/project --tag snapshot-YYYY-MM-DD-description
```

`--workspace` 是本次專案根目錄，內容對應 `research/` 下一層；不要傳入混有其他專案的上層目錄，也不要把整個包含 `research/` 的新儲存庫當成原工作區而重複巢狀保存。tag 必須為新的版本名稱，日期與描述對應當次工作。

工具按保存規則複製可讀檔案至 `research/`、將大型檔分包上傳至私人草稿 Release，完成遠端 digest 核對後才更新 catalog。確認其輸出及遠端驗證紀錄後，再核對 Git 中的 `data/index.json`、catalog、verification 是否一致。若任何階段未完成，保留原資料，明確記錄未完成範圍，不覆寫舊快照或宣稱備份完成。

## 保存不等於立即可重跑

- 此次整理沒有重新跑全部回測。檔案 SHA-256 一致是內容保存證據，不是新的模型有效性證明。
- 多版程式包含原電腦絕對路徑；部分使用 macOS `.dylib`、Accelerate、特定套件或字型。另開適配版本處理環境，保留歷史程式。
- 雲端對話、網站服務帳號及未匯出的瀏覽器 `localStorage` 不在檔案快照裡。可重裝套件、排除項與每次實際保存範圍以 catalog／發布紀錄為準。
