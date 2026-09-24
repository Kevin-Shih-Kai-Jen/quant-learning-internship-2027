# 私人 GitHub 保存範圍與上傳前檢查

日期：2026-09-24。此文件描述保存範圍與已完成的唯讀檢查，**不代表遠端上傳已完成**。原始研究檔案沒有因此刪除、重跑或改寫。

原專案：`/Users/coolguy/.codex/.chatgpt-projects/g-p-6a8ad7dc606481919474d5424342a1c3`

Git 保存可閱讀的程式、計畫、報告與小型紀錄；大型歷史實驗資料以同一私人儲存庫的版本化 Release 附件保存。`research/` 保持原專案的兄弟目錄結構。每個排除於 Git 的研究明細，仍須在 Release 封存與檔案清單中找到對應；僅有摘要不算完整保存。

## 需要保存的內容

- 所有 `jpx_*` 實驗目錄：Python/C++ 源碼、原生函式庫、實驗計畫、版本設定、結果、稽核、日誌、特徵 PKL、逐日 predictions/traces NPZ、所有 CSV/CSV.GZ、逐年 ranks、參數／訓練更新、舊版／修正前輸出。`pre_period_fix/`、`superseded_absolute_revision/`、`KNOWN_ISSUES.md` 均屬研究歷史。
- 根目錄 `JPX-current-baseline.md`、`JPX-experiment-defaults.json`、`jpx_active_baseline.json`、`jpx_model_versions.json` 與研究文書、簡報、建置程式。
- `output/` 的報告、PDF、PPTX、交付 ZIP 與資料；`ppt_build/` 的建置程式、模板、素材、驗證紀錄及既有輸出；`qa_*`、`tmp/`、`.chart-data-*` 中的歷史檢查／繪圖資料。這些目錄不能只憑名稱一律當作垃圾排除；體積較大者可保存於 Release。
- `docs/` 與 `scheduling-app/`，以及 `hiwin-application-site/` 的應用程式碼、公開素材、設定、`package.json`、`package-lock.json`。排班資料若存在於瀏覽器，另見下方限制。
- `project_archive_20260924/` 裡的整理文件／清單可以整合到 `docs/`；封存包或外部 ZIP 的重複副本不必重複上傳，但應有明確對應與校驗。
- 外部 `JPX_data.zip`、`JPX-market-regime-experiment.zip` 與原簡報 `reference.pptx`。詳見保存查核文件；未壓縮的外部股票 CSV 已核對與 JPX_data.zip 成員 SHA-256 完全相同，不需再備份第二份。

## 不納入研究上傳的本機操作狀態

以下排除不應誤套到同名研究內容。這是 Git／Release 上傳範圍，不是刪除原檔指令。

| 路徑／類型 | 原因 |
|---|---|
| `hiwin-application-site/node_modules/` | 已安裝套件，可由 lockfile 重裝；保留 lockfile。 |
| `hiwin-application-site/.next/`、`hiwin-application-site/dist/` | 網站建置輸出；原應用程式及 public 素材需保留。 |
| `hiwin-application-site/.wrangler/` | 本機部署設定、SQLite cache 與執行狀態，非研究資料。 |
| `hiwin-application-site/.openai/hosting.json` | 部署專案識別／綁定 metadata，非可攜研究源碼；本次只核對欄位名稱，沒有輸出識別值。 |
| 所有巢狀 `.git/` | 既有 Git 內部資料和設定，不應被另一儲存庫當研究檔案收錄。 |
| `__pycache__/`、`*.pyc`、`.DS_Store` | 執行／作業系統快取。 |
| `github_repo_20260924/` 自身 | 封存原專案時排除 staging，避免遞迴包含自身。 |

若後續新增 `.env`、私鑰、API token、cookies、service-account credential 等檔案，不可因儲存庫是 private 就一起上傳。原始實驗源碼／輸出若真的含憑證，應先隔離並個別處理，不能將掃描後檔案已安全當成永久保證。

## 本次實際完成的敏感資料檢查

- 原專案中 2,007 個不大於 2 MB 的文字檔經規則掃描，未命中 GitHub token、OpenAI key、AWS access key、私鑰標頭，以及常見非空字串 credential assignment 模式。掃描不輸出任何憑證值。
- 掃描跳過 `node_modules/`、`.next/`、`.git/`、新 Git staging 和新整理封存目錄；不解析 PKL，不重跑任何研究。大檔與二進位內容未作全面秘密檢測，因此此結果不是零風險保證。
- 未找到 `.env`、`.npmrc`、`.netrc`、`.git-credentials`、常見 service-account credential、PEM/P12/PFX/KEY 檔案。
- 發現的部署設定為上述 hosting metadata、`.wrangler/deploy/config.json` 和產生的 `dist/server/wrangler.json`。後者的 secret binding 與 vars 皆為空；仍按建置／操作狀態排除。
- 巢狀 `hiwin-application-site/.git/config` 只有 core section；未發現 remote URL 或內嵌遠端認證。未讀取專案外的任何憑證儲存區。
- `JPX_data.zip` 的 16 個 member names 與 `JPX-market-regime-experiment.zip` 的 54 個 member names 經可疑憑證檔名檢查，沒有發現可疑成員；第一包的 `.gitkeep` 只是佔位檔。這是 ZIP 名稱檢查，並非所有成員內容的秘密掃描。

## 私人存取與還原限制

來源 ZIP 沒有看到授權條款檔名；這不代表已取得公開再散布權。因此這次保存採用 **private repository 與其 private Release**。若要改為公開，需另確認原資料、模板與素材的權利及公開範圍；此次未作法律判斷。

最新完成系列 v30 的 `run.py` 引用 v7、v8、v14、v27、v28，這些原目錄在查核時均存在；v27 再引用 v14/v8。目前 active-baseline 設定仍指向 v7，而 2026-09-23 的 ARIMA 目錄是尚待訓練選項確認的數學／資料查核。保存時不能以日期或版本號認定只保留最後一個目錄就足夠。

部分程式包含舊電腦絕對路徑、macOS 動態函式庫和已不存在的舊文書工具版本。保存 bytes 不等於已驗證跨平台重跑。還原需要將 Git 和 Release 明細組回同一結構，再處理環境與路徑；避免重跑覆寫歷史輸出。

排班工具使用瀏覽器 `localStorage` 的 `scheduling-app-mvp-v1` 狀態。若曾實際使用，須從原瀏覽器匯出完整 JSON；本次檔案上傳不包含瀏覽器狀態，也不含 ChatGPT／Codex 完整對話歷史。

實際政策補充：本次保留網站 `dist/` 與 `next-env.d.ts` 作為歷史建置證據；另排除 `.cache/` 與 `.venv/`。以 archive-config.json 與 data/inventories/ 清單為準。可讀檔案會依清單強制加入 Git 並核對內容，避免巢狀 .gitignore 導致遺漏。
