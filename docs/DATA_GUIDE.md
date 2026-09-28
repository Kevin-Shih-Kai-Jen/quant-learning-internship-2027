# 資料下載、還原與新快照

此公開儲存庫使用兩層保存（2026-09-28 依使用者指示改為公開）。Git 保存可讀內容與索引；大型資料保存在按 tag 版本化的 GitHub Release `.tar` 分包，每包至多 768 MiB。每次使用都以 catalog 的實際檔案清單、大小和 SHA-256 為準。

**此 GitHub 儲存庫與版本化 Release 是歷史資料的主保存位置。** 依使用者指定，本機長期保留輕量導讀、工具與索引；每次工作建立專屬臨時資料目錄，按需下載，用畢依本指南核對後清理可重取副本。這是資料使用規則，不表示所有獨立聊天室的設定已自動改變；其他聊天室需貼上 [續接指令](FUTURE_CHAT_PROMPT.md) 或設定專案指示。

首次 tag：`snapshot-2026-09-24`。入口是 [data/index.json](../data/index.json)，對應 [catalog](../data/catalogs/snapshot-2026-09-24.json) 和 [遠端驗證紀錄](../data/verification/snapshot-2026-09-24.json)。不要把建立 tag、推送 Git 或看到部分 Release 資產當成全量上傳完成。

## 1. 取得儲存庫與查看快照

讀取與下載需可連線 GitHub 的環境；下載工具需 Python 3，公開檔案與已發布 Release 不需登入。寫入、上傳及發布新 Release 仍需 GitHub 授權。下列指令從儲存庫根目錄執行；使用前可用各工具的 `--help` 核對目前介面。

```sh
git clone https://github.com/Kevin-Shih-Kai-Jen/quant-learning-internship-2027.git
cd quant-learning-internship-2027
python3 scripts/data_archive.py list
```

`list` 列出 Release 資料的頂層分組、檔數與大小；細部路徑查 catalog。`fetch` 會先以匿名 HTTPS 下載公開附件，僅遇 401／403／404 時回退到已登入的 GitHub CLI；正常公開下載不需要 `gh` 登入。Git 與 Release 的完整收錄對照見 [檔案盤點](../data/inventories/snapshot-2026-09-24.json) 或 [實驗下載索引](EXPERIMENT_INDEX.md)。`research/` 對應原專案根目錄；僅 clone 不會取回所有大型檔案。

## 2. 按實驗取回，連同依賴一起處理

先建立只供本次任務使用的臨時目錄，記錄其位置、快照 tag 與取回清單。以下範例沿用同一個終端機中的 `QUANT_TASK_DIR` 變數；不要指向共享資料夾、其他專案或唯讀 `sources/`。下載前確認可用空間能容納所需分包暫存、解壓內容與新結果。

```sh
QUANT_TASK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/quant-research.XXXXXX")"
python3 scripts/data_archive.py fetch --experiment jpx_v30_profit_actual_forecast_20260916 --tag snapshot-2026-09-24 --dest "$QUANT_TASK_DIR"
python3 scripts/data_archive.py verify --experiment jpx_v30_profit_actual_forecast_20260916 --tag snapshot-2026-09-24 --dest "$QUANT_TASK_DIR"
```

檔案會放到本次臨時目錄的 `research/` 下；工具與索引仍留在原儲存庫。`verify` 必須使用與 `fetch` 相同的 `--dest`。需要的程式與輕量設定也應按相同相對結構複製到任務工作目錄，或在新實驗版本中明確設定輸入路徑，勿改寫歷史來源。不要在臨時目錄留下尚未同步的唯一成果。

不帶 `--tag` 時依工具與 `data/index.json` 選定的快照操作；每次工作應明確指定並記錄 tag。`--experiment` 的前綴以 `list` 與 catalog 為準。

**下載一個實驗目錄，不等於它的輸入都已到齊。** 例如 v30 的流程依賴 v7、v8、v14、v27、v28，財報特徵還有較早版本的間接依賴。先讀原計畫、程式輸入及 [相依查核](../research/project_archive_20260924/PRESERVATION_NOTES.md)，再取回相關兄弟目錄。原本封存的大型檔案還位於另一個前綴：

```sh
python3 scripts/data_archive.py fetch --experiment project_archive_20260924/compressed_data/jpx_v8_soft_rank_20260912 --tag snapshot-2026-09-24 --dest "$QUANT_TASK_DIR"
python3 scripts/data_archive.py verify --experiment project_archive_20260924/compressed_data/jpx_v8_soft_rank_20260912 --tag snapshot-2026-09-24 --dest "$QUANT_TASK_DIR"
```

原始資料 ZIP 與模板位於 `project_archive_20260924/external_sources/`，同樣按 catalog 取回；JPX 原始 ZIP 的 tag、分包與免登入入口見 [ACCESS.md](ACCESS.md)。工具下載／驗證檔案不等於執行或重新訓練模型。

## 3. 將原 gzip 封存恢復成舊程式路徑

9/24 的本機保存已把 139 個大型 CSV／PKL 壓成 gzip；完整內容經解壓 SHA-256 核對後，才移除原位置副本。因此 GitHub 取回的有些檔案是 `.pkl.gz`／`.csv.gz`，還需要原封存工具還原。

原工具依它自身位置決定還原根目錄。將工具與清單複製到本次臨時目錄的相同結構後再執行，避免把大型原檔還原回長期儲存庫：

```sh
mkdir -p "$QUANT_TASK_DIR/research/project_archive_20260924"
cp research/project_archive_20260924/restore_data.py research/project_archive_20260924/COMPRESSED_DATA_MANIFEST.json "$QUANT_TASK_DIR/research/project_archive_20260924/"
python3 "$QUANT_TASK_DIR/research/project_archive_20260924/restore_data.py" --list
python3 "$QUANT_TASK_DIR/research/project_archive_20260924/restore_data.py" --restore --path jpx_v8_soft_rank_20260912/inputs.pkl
```

這會依 `COMPRESSED_DATA_MANIFEST.json` 把檔案恢復到 `$QUANT_TASK_DIR/research/jpx_v8_soft_rank_20260912/inputs.pkl`，並保留本次 gzip 副本至工作結束。也可以對某個已下載齊全的實驗前綴使用 `--path`。工具遇到同路徑不同內容會拒絕覆寫；不要為了繼續而任意刪掉現有檔案。

原工具 `--verify` 會核對整份 gzip 清單，若只按需下載部分封存，缺少其餘檔案不代表已下載的檔案損壞。此時可對所需檔案執行 `--restore --path`，它本身也會核對原始與壓縮內容。全部封存到齊後，再用 `--verify` 作全量核對。詳見 [原還原指南](../research/project_archive_20260924/RESTORE_GUIDE.md)，其中舊電腦絕對路徑需改成目前位置。

## 4. 全量取回與空間

預設按需取回；只有本次工作確實需要整套資料且容量足夠時，才在同一個任務專屬目錄使用全量模式：

```sh
python3 scripts/data_archive.py fetch --all --tag snapshot-2026-09-24 --dest "$QUANT_TASK_DIR"
python3 scripts/data_archive.py verify --all --tag snapshot-2026-09-24 --dest "$QUANT_TASK_DIR"
```

全量取回後如需恢復所有原 CSV／PKL，先按上一節放妥還原工具與清單，再執行：

```sh
python3 "$QUANT_TASK_DIR/research/project_archive_20260924/restore_data.py" --restore
```

最後一步可能額外需要約 **4.85 GB** 原始資料空間，且不刪 gzip；Release 下載、分包暫存與其他產物也需要空間，精確大小依 catalog 與工具輸出。硬碟有限時，優先取回本次工作所需資料。無損封存釋放的是硬碟空間，不會自動降低模型運行的 RAM 需求。

## 5. 保存新工作

先在新的實驗目錄完成本次研究，保存 plan、來源、設定快照、程式、完整結果、逐步紀錄、audit 與報告。必要時新增簡短記憶與導覽，記錄更正來源；不要重寫舊實驗事實。只有在使用者已授權同步本次產物時進行上傳。

可讀的小檔使用正常 Git 提交／推送。大型資料和可讀檔案快照由獨立工具處理。下列是上傳入口，**新 tag 必須先完成本節下方的 `--prepare-only`、提交來源及建立同名草稿 Release 流程，再執行此命令**；不要把它當成第一次操作的起點：

```sh
python3 scripts/publish_snapshot.py --workspace /absolute/path/to/project --tag snapshot-YYYY-MM-DD-description
```

`--workspace` 是本次專案根目錄，內容對應 `research/` 下一層；不要傳入混有其他專案的上層目錄，也不要把整個包含 `research/` 的新儲存庫當成原工作區而重複巢狀保存。tag 必須為新的版本名稱，日期與描述對應當次工作。

大型輸入預設放在任務專屬臨時目錄。若新成果也產生在那裡，可用 `--workspace "$QUANT_TASK_DIR/research"` 同步相同結構下的新版本；同步成功前保留所有新產物。若新成果直接寫在 clone 的新 `research/` 實驗目錄，則用 `--workspace ./research`。工具會保留來源，並核對內容；同一路徑不會再次複製到自己。兩種情況都不改寫歷史實驗。

以臨時目錄作為 `--workspace` 時，即使這次沒有還原 gzip，也要先放入固定的歷史壓縮清單，供工具核對既有封存映射。新成果優先保存在常駐 repo 的新實驗目錄，避免臨時目錄成為尚未同步成果的唯一存放處。

```sh
mkdir -p "$QUANT_TASK_DIR/research/project_archive_20260924"
cp research/project_archive_20260924/COMPRESSED_DATA_MANIFEST.json "$QUANT_TASK_DIR/research/project_archive_20260924/"
```

第一次上傳新 tag 時，先準備清單、強制加入列出的可讀檔案（避免原專案 `.gitignore` 漏檔），提交並推送，再建立同名草稿 Release：

```sh
python3 scripts/publish_snapshot.py --workspace /absolute/path/to/project --tag snapshot-YYYY-MM-DD-description --prepare-only
git add -f --pathspec-from-file=data/upload-plans/snapshot-YYYY-MM-DD-description-git-paths.nul --pathspec-file-nul
git add docs scripts archive-config.json data/inventories
git diff --cached --stat
git commit -m "Save research snapshot sources"
git push origin main
gh release create snapshot-YYYY-MM-DD-description --repo Kevin-Shih-Kai-Jen/quant-learning-internship-2027 --draft --target main --title "Research snapshot" --notes-file /path/to/release-notes.md
python3 scripts/publish_snapshot.py --workspace /absolute/path/to/project --tag snapshot-YYYY-MM-DD-description
```

請先撰寫實際 release notes 檔；以上路徑、tag 是需替換的範例。原始快照不會被覆寫，後續 catalog 會引用既有快照中相同資料，只上傳新增檔案；舊大型資料改內容須另建版本路徑。若中斷，使用同一來源與 tag 重跑，工具會核對已上傳資產，不會刪掉遠端資產重傳。

`archive-config.json` 的 `compression_manifest` 指向工作目錄內的壓縮清單；還原出的歷史原檔若與清單一致，且最新已驗證 catalog 中有匹配的 gzip，便沿用該封存而不重複上傳，原路徑內容改變則必須另建實驗路徑。

`compression_manifest_sha256` 鎖定這份清單的完整位元組，初始歷史映射不可改寫；若需建立新映射，必須另外明確設定已驗證的新清單及其雜湊，工具不會自動接受變動。上傳工具也會核對遠端可見性是否符合 `archive-config.json` 的 `public` 設定，不會自行更改 repo 可見性；若不一致，先查明原因，不繞過核對。

工具按保存規則複製可讀檔案至 `research/`、將大型檔分包上傳至尚未發布的草稿 Release，完成遠端 digest 核對後才更新 catalog。它不代做最終 Git commit／push 或 Release 正式發布。先確認輸出與遠端驗證紀錄，核對 `data/index.json`、catalog、verification 一致；再提交／推送本次檔案，將已核對的草稿 Release 發布到這個公開儲存庫，最後重查遠端可取回狀態。公開後的新成果同樣可被外界讀取；沿用本專案同步授權，排除憑證及無關個資，不更改儲存庫可見性。

若任何階段未完成，保留本次資料，明確記錄未完成範圍，不覆寫舊快照或宣稱備份完成。實際包含與排除的內容可查 [保存範圍](UPLOAD_SCOPE.md) 及 `archive-config.json`，不要把同名研究輸出誤當可丟棄快取。

## 6. 用完後清理本次副本

使用者已授權此流程下的本次下載副本清理，符合條件後不必為同一範圍再次要求許可：

1. 對照任務取回清單，辨認下載副本、gzip 還原副本、操作暫存與新建／變更成果。先確認臨時目錄沒有別的工作正在使用，也沒有不明歸屬檔案。
2. 本次若產生新成果或變更，先完成同步、發布與遠端內容核對：Git 不只需本機 commit，還要確認遠端提交及檔案內容；Release 需核對資產大小／SHA-256、逐檔 catalog 與可取回狀態。沒有新成果時，也要確認原快照仍可取回且雜湊一致。
3. 只有已證實可從雲端重取或由已驗證 gzip 重建的本次副本，及無保留價值的操作暫存，才可清理。只在記錄的任務專屬範圍內操作，不沿符號連結擴大刪除，也不以檔名相同當作內容相同。
4. 不刪未同步／校驗失敗的資料、唯讀 `sources/`、其他專案、共享且正在使用的檔案，或常駐的導讀、工具與索引；同步失敗或內容歸屬不明時保留，說明原因。
5. 回報清理的實際範圍，以及下次重取所需的 tag／前綴。雲端原始資料、版本 Release 與歷史過程證據均保持完整。

本節不提供對整個專案的一鍵刪除指令。清理對象由本次清單與核對結果決定；文件中的政策本身也不代表本機檔案已被清理。

## 保存不等於立即可重跑

- 此次整理沒有重新跑全部回測。檔案 SHA-256 一致是內容保存證據，不是新的模型有效性證明。
- 多版程式包含原電腦絕對路徑；部分使用 macOS `.dylib`、Accelerate、特定套件或字型。另開適配版本處理環境，保留歷史程式。
- 雲端對話、網站服務帳號及未匯出的瀏覽器 `localStorage` 不在檔案快照裡。可重裝套件、排除項與每次實際保存範圍以 catalog／發布紀錄為準。

完成全部資料上傳後，提交 `data/index.json`、`data/catalogs/`、`data/verification/` 及本次導讀更動並推送。把本次 catalog 與 SHA256 清單也附加到草稿 Release，再以該提交作為 Release 的 target，解除 draft。發布後核對遠端 tag 對應的提交、所有資產 digest 以及一次按需下載測試，才回報完成。不得以 `--clobber` 覆寫既有資料包。
