# 新對話續接指令

把下面整段貼到新的對話，再接著描述這次要做的工作。該對話需要能存取此私人 GitHub 儲存庫；沒有連線或登入時，應先說明存取障礙；不同聊天室的 GitHub 授權說明見 [ACCESS.md](ACCESS.md)。此指令提供的是檔案入口，不會使未匯出的舊對話自動可見。

---

請接續我的「Quant Learning & Internship 2027」專案。私人儲存庫：

https://github.com/Kevin-Shih-Kai-Jen/quant-learning-internship-2027

請先取得或更新這個儲存庫，讀取：

1. `AGENTS.md`
2. `research/project_archive_20260924/MEMORY_BRIEF.md`
3. `data/index.json` 與本次使用快照的 `data/catalogs/<tag>.json`
4. `docs/RESEARCH_MAP.md`、`docs/DATA_GUIDE.md`，以及與本次任務有關的實驗計畫、報告、設定快照與已知問題。

需要更多背景，再讀 `research/project_archive_20260924/EXPERIMENT_HISTORY.md`。不要一次把所有歷史資料塞進對話，也不要假設能讀到沒有存入 GitHub 或本次未提供的舊對話。

若需要大型資料，使用 `scripts/data_archive.py list` 查位置，按本次實驗前綴取回。先讀 catalog 和程式的輸入相依，將所需兄弟實驗目錄、共用特徵及 `project_archive_20260924/compressed_data/` 下的相關 gzip 一起取回；不要只下載最新版本目錄就假設資料齊全。需要原 `.pkl`／`.csv` 時，使用 `research/project_archive_20260924/restore_data.py` 依清單驗證及還原。請按需下載，核對空間與 SHA-256，保留封存。

正式基準仍是 `v7_equal`；v30 的 `frozen_r` 僅為 validation 候選。現行規則是 Sharpe 第一、Rank IC 第二、不要求 Rank IC 正值；MSE 不選模型，不排除或裁切財報極端值。尚未使用的 test 留待我要求總體驗證。遵循既有資料時序與事件有效性，保留 v13 口徑錯配、EPS 問題和所有修正前／失敗紀錄。新實驗另開版本目錄，不覆蓋歷史結果。

我授權你把**本次任務在此專案產生的新程式、筆記、報告、設定與必要實驗資料**同步到上面這個私人儲存庫：小型可讀檔案用正常 Git 提交／推送，大型資料用 `scripts/publish_snapshot.py` 建立新的快照 Release。先完成工作與適當核對，再同步；不要改公開、覆寫舊快照、刪除原資料、提交密鑰或混入其他專案內容。這不是部署網站或向其他人傳送訊息的授權。

完成後更新必要索引與精簡續接記憶，保留歷史來源可追溯性，核對遠端 commit、Release 資產及驗證紀錄。回報實際同步的位置、核對範圍，以及仍未完成的事項；如果缺權限或上傳不完整，直接說明，不能宣稱已備份。這次的具體工作如下：

---
