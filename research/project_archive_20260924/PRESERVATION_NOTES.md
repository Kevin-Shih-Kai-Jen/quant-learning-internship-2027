# 保存與重現風險查核

查核日期：2026-09-24。這份文件記錄檔案相依與保存風險，不代表已完成全部資料的備份或重新執行所有實驗。查核僅讀取原檔；未修改 `sources/`、既有實驗或 Git 狀態。

**封存狀態更新：** 查核後已依使用者同意完成大型資料無損封存；下列大型 PKL／CSV 的原位置可能需先依 [還原說明](RESTORE_GUIDE.md) 還原。兩個外部 ZIP 與簡報模板已補存至 `external_sources/`，逐 byte 雜湊核對通過；完整原檔清單見 `PROJECT_MANIFEST_SHA256.json`。本筆記描述的相依仍然有效。

下文 `$PROJECT` 明確指向：

`/Users/coolguy/.codex/.chatgpt-projects/g-p-6a8ad7dc606481919474d5424342a1c3`

## 必須與專案一起保存的外部原始資料

| 現在的位置 | 查核結果 | 保存原因 |
|---|---|---|
| `/Users/coolguy/Desktop/JPX_data/modelling/JPX_data.zip` | 存在；222.93 MB；16 個 ZIP 項目，展開約 1,158.59 MB | 內含股票價格、財報、stock list，以及 options、secondary stock prices、trades 原始資料。多個實驗直接讀取這個路徑。只打包 `$PROJECT` 會漏掉它。 |
| `/Users/coolguy/Desktop/JPX_data/modelling/JPX-market-regime-experiment.zip` | 存在；13.48 MB；54 個 ZIP 項目，展開約 57.60 MB | 包含原 v3 的程式、個股預測／訂單、原始比較基準、`jpx_inputs/nikkei_cme_yahoo_raw.json` 與 Nikkei 每日資料。這不是可用目前報告替代的附件。 |
| `/Users/coolguy/Desktop/JPX_data/raw/train_files/stock_prices.csv` | 存在；217.60 MB；已核對與第一個 ZIP 內 `JPX_data/raw/train_files/stock_prices.csv` 完全相同 | `jpx_neutral_10pct_20260910/run_neutral.py` 直接使用未壓縮路徑；不必再備份第二份相同 CSV，但還原時須重建路徑或明確修正讀取設定。 |

最後一列使用串流 SHA-256，比對原 CSV 及 ZIP 解壓串流；兩者都是：

`bf774a86f834e5338bba74f2356ebb750f93a1cc4e89b064ee07251e8ed851db`

其餘資料沒有全部重新計算雜湊。上表 MB 是十進位檔案長度，不是磁碟實際可回收區塊，也不含 ZIP 解壓後的另存副本。

## 特徵檔是可計算的資料，但不能直接當成可丟棄快取

實驗互相引用兄弟目錄，保留單一「最新版本」無法獨立重跑。代表性的相依如下：

- `$PROJECT/jpx_stock_returns_20260910/features.pkl`、`bars.pkl` 是早期價格／交易流程的共同輸入。原始 CSV、該目錄的 `source/` 與特徵產生程式都必須保留。
- `jpx_return_mean_20260911/price_return_1d.pkl` 與共同價格特徵產生 `ratio_features.pkl`，也被 `jpx_return_t_20260911/feature_builder.py` 使用；之後 return-T／Ridge／交易成本比較再引用這些資料。
- `jpx_v8_soft_rank_20260912/inputs.pkl` 依賴 v5 單日報酬、v7 特徵程式、共同價格特徵和原始價格 ZIP。後續財報系列持續使用這份共同輸入。
- v13/v14/v15 引用 v8、v11 的價格／標準差輸入和較早財報 builder；v17 引用 v14 的財報 records、events 與 signal features。
- v21–v30 多個實驗引用 v7 的運算程式、v8 的輸入、v14／v17 的財報特徵。v27 的 state 特徵引用 v14 builder，v30 再引用 v27。讀取這些程式時不能只搬走最終資料夾。

截至本次清點，排除符號連結重複計數後，專案內有 35 個 `.pkl` 約 4,473.63 MB、7,398 個 `.npz` 約 3,557.01 MB、159 個 `.csv.gz` 約 3,934.04 MB。這些數字是類型合計，不是宣稱可以刪除的容量。

`.pkl` 多數是產生後的特徵／中間資料；在原始資料、程式、精確相依和可用運算環境全部齊全時才有條件重建。`.npz` 的逐日 traces、predictions，以及 `.csv.gz` 的 batch trace、逐年 ranks、selected stocks，則保留「當時究竟如何訓練與得到結果」的證據。摘要、日報酬和最終係數不足以取代這些逐筆紀錄。使用者重視實驗過程，因此宜無損壓縮／搬存，不能直接以報告替代。

特別應保留 `jpx_v12_financial_growth_20260913/pre_period_fix/` 與 `jpx_v17_actual_forecast_revisions_20260915/superseded_absolute_revision/`。它們記錄修正前的假設與輸出；僅因看起來重複而刪除，會破壞修正過程的可追溯性。

## 重現環境限制

- JPX 根目錄沒有找到完整 Python requirements／lockfile。部分實驗有 `runtime_manifest.json`；例如 v13 記錄 Python 3.14.2、NumPy 2.4.6、pandas 3.0.3、macOS arm64 與 clang++。這些歷史記錄應保留；本文件沒有把它們視為所有實驗共用的版本。
- 多版程式透過 `ctypes` 載入同目錄 `.dylib`。有對應 `.cpp` 源碼；v15 `build.py` 明確使用 clang++、C++17、macOS `Accelerate` framework。換到 Linux／Windows 需要修改建置與動態函式庫路徑，不能宣稱下載後跨平台即跑。
- 原程式包含使用者電腦絕對路徑，部分報告也含 `$PROJECT` 的絕對連結。移到 GitHub 或其他電腦時，檔案存在不等於路徑已可用。
- 數值稽核通過不等於資料語意無誤：v13 `KNOWN_ISSUES.md` 記錄個別／合併財報錯配，並指出 v12 同樣規則需追查。應與結果一起保留，避免濃縮後只剩漂亮指標。
- `jpx_arima_acf_learning_20260923/experiment_plan.md` 明確是資料／公式查核與數學核心；正式回測仍待參數共用、殘差截距、訓練排程等選擇。`math_checks.json` 是合成資料核對，不是完成的 JPX 績效。

## 非 JPX 資料也在保存範圍

### 文書、研究簡報與素材

應保留 `$PROJECT/機器人與自動化股票研究企劃書.docx`、`build_robotics_research_proposal.py`、根目錄的大銀研究 Markdown／PPTX、`output/` 的最終研究 PPTX／PDF，以及 `ppt_build/` 的建置程式、素材和版型。舊版文書 helper 已失效，因此最終文件尤其重要。

已查到的外部相依：

| 外部位置 | 目前狀態與含意 |
|---|---|
| `/Users/coolguy/.codex/plugins/cache/openai-curated-remote/openai-templates/0.1.1/skills/artifact-template-team-alignment/assets/reference.pptx` | 存在；兩組研究簡報建置程式直接引用；應另外保存模板。`ppt_build/team_alignment_hiwin/template-starter.pptx` 也需保留，但本次沒有證明它與 reference.pptx 相同。 |
| `/Users/coolguy/.codex/plugins/cache/openai-primary-runtime/documents/26.819.11345/skills/documents` | 已不存在；企劃書 builder 從此處 import `table_geometry`。須在重建時提供相容 helper。 |
| `/Users/coolguy/.codex/plugins/cache/openai-primary-runtime/presentations/26.903.11726/skills/presentations/container_tools/` | 舊路徑下的 `artifact_tool_utils.mjs` 及兩個 inspect helper 已不存在；簡報建置／finalize 引用此版本。 |
| `/Users/coolguy/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3` | 存在；部分簡報 finalize 硬編碼使用。這是環境路徑，不是研究原始資料。 |

簡報還依賴 `@oai/artifact-tool`、JSZip 等套件，這組建置流程未見自己的 package lock。JPX PDF builder 使用 ReportLab 與 macOS 的 `/System/Library/Fonts/STHeiti Light.ttc`、`STHeiti Medium.ttc`；其他電腦可能沒有同字型。最終成品不能都當可立即重建檔案刪掉。

簡報程式保留了公開網站來源 URL，但動態網頁／後續更新網站不等於當時的完整資料快照。本次未下載補齊外部網站，也未重新驗證研究數字。兩張本地 JPG 素材在 `ppt_build/hiwinmikro_robotics_2026/assets/`。

### 網站與排班工具

- `$PROJECT/hiwin-application-site/` 必須保存 `app/`、`public/`、設定、`package.json` 和 `package-lock.json`。`node_modules/` 約 798.91 MB（排除 symlink）是可依 lockfile 重裝的套件；另 `dist/`、`.wrangler/` 是建置／本機執行產物。若要釋放磁碟，這比刪除實驗資料更可控，但重裝仍需要原套件可取得。
- `hiwin-application-site/public/hiwin-application-link.html` 使用特定版本的遠端 JS CDN；完整離線呈現尚未驗證。網站專案備份也不等於服務端部署與帳號的完整備份。
- `$PROJECT/scheduling-app/` 有 4 個應用程式檔案，`docs/ai-scheduling-app-product-plan.md` 是較完整規劃，兩者都應保存。
- 排班工具的使用者輸入儲存在瀏覽器 `localStorage`，鍵名 `scheduling-app-mvp-v1`。這份狀態**不在專案資料夾**；若曾實際填寫員工／排班資料，需由使用該工具的原瀏覽器匯出完整 JSON。程式有匯出 state 功能；本次沒有讀取瀏覽器，也不知道是否存在使用者資料。

## 壓縮與刪除的界線

`.pyc`、`__pycache__`、`.DS_Store`、網站可重裝套件及重建輸出是低價值快取候選。QA 逐頁圖片／preview／montage 可在保留成品和查核紀錄後減量；本次沒有逐張證明它們與所有最終版本相同。

目前 `sources/` 沒有檔案；它仍受 AGENTS.md 唯讀規則保護。此處「以前的資料」清點只涵蓋實際可見的本機檔案與上述外部依賴，不等於 ChatGPT／Codex 每段歷史對話的完整匯出。縮寫摘要有助接續工作，但不會自動釋放 RAM，也不能替代完整實驗紀錄。

任何要移除原始資料或實驗明細的後續步驟，應先確認另一份保存副本已建立、校驗並能還原。這份查核沒有刪除任何原檔。
