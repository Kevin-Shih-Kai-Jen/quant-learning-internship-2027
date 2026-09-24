# 新對話續接指令

把下面整段貼到新的對話，或存入該專案的自訂指示，再描述這次工作。在此儲存庫工作的 Codex 會讀取 `AGENTS.md`；其他獨立聊天室不會因此自動更改設定。該對話仍需能存取私人 GitHub；存取說明見 [ACCESS.md](ACCESS.md)。此指令不會使未匯出的舊對話自動可見。

---

請接續我的「Quant Learning & Internship 2027」專案。私人儲存庫：

https://github.com/Kevin-Shih-Kai-Jen/quant-learning-internship-2027

請先取得或更新這個儲存庫，讀取：

1. `AGENTS.md`
2. `research/project_archive_20260924/MEMORY_BRIEF.md`
3. `data/index.json` 與本次使用快照的 `data/catalogs/<tag>.json`
4. `docs/RESEARCH_MAP.md`、`docs/DATA_GUIDE.md`，以及與本次任務有關的實驗計畫、報告、設定快照與已知問題。

需要更多背景，再讀 `research/project_archive_20260924/EXPERIMENT_HISTORY.md`。不要一次把所有歷史資料塞進對話，也不要假設能讀到沒有存入 GitHub 或本次未提供的舊對話。

私人儲存庫與版本化 Release 是歷史資料的主保存位置，本機只長期保留輕量導讀、工具與索引。需要大型資料時，先用 `scripts/data_archive.py list` 查目錄，為本次任務建立專屬臨時資料目錄，記錄 tag 與取回清單，使用 `fetch --dest` 按需下載並以相同目錄／tag 驗證 SHA-256。下載前檢查容量，包含 TAR 暫存、解壓及新結果所需空間。

讀 catalog 與程式相依，把必要兄弟實驗、共用特徵及 `project_archive_20260924/compressed_data/` 中的 gzip 一起取回。需要原 `.pkl`／`.csv` 時，依 `docs/DATA_GUIDE.md` 將原還原工具與清單放在臨時目錄相同結構下，驗證並還原；不要不小心寫回長期工作區。新實驗另開目錄，保留原研究不變。

正式基準仍是 `v7_equal`；v30 的 `frozen_r` 僅為 validation 候選。現行規則是 Sharpe 第一、Rank IC 第二、不要求 Rank IC 正值；MSE 不選模型，不排除或裁切財報極端值。尚未使用的 test 留待我要求總體驗證。遵循既有資料時序與事件有效性，保留 v13 口徑錯配、EPS 問題和所有修正前／失敗紀錄。新實驗另開版本目錄，不覆蓋歷史結果。

我授權你把**本次任務在此專案產生的新程式、筆記、報告、設定與必要實驗資料**同步到上面這個私人儲存庫：小型可讀檔案用正常 Git 提交／推送，大型資料用 `scripts/publish_snapshot.py` 建立新的快照 Release。先完成適當核對，再同步；不要改公開、覆寫舊快照、刪除雲端研究歷史、提交密鑰或混入其他專案。這不是部署網站或向其他人發訊息的授權。

我也授權你在用完後清理**本次專屬臨時目錄內、可從已驗證私人快照重新取得的下載／還原副本及操作暫存**。本次新產物或變更必須先成功同步，核對遠端 Git 檔案內容、Release／catalog 的 SHA-256 與可取回狀態，再清理；無新成果時也要確認原快照仍可取回且內容一致。不得刪除未同步資料、`sources/` 同步唯讀材料、其他專案、共享且正在使用的檔案或輕量導讀／工具／索引。不確定檔案歸屬或同步未完成時保留檔案，回報原因。

完成後更新必要索引與精簡續接記憶，保留歷史來源可追溯性。回報遠端 commit／Release、核對範圍、已清理的本次副本及保留項；記下下次取回的 tag／前綴。缺權限或上傳不完整就明確說明，不宣稱已備份或已清理。這次的具體工作如下：

---
