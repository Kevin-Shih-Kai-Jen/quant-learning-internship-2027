# JPX 實習作品集發布紀錄 — 2026-10-11

使用者要求將桌面 JPX 專案建立為獨立 GitHub repository，並明確確認公開 `jpx-equity-prediction`，範圍為程式、說明及實驗報告，排除原始大型資料。

- 新作品集：[Kevin-Shih-Kai-Jen/jpx-equity-prediction](https://github.com/Kevin-Shih-Kai-Jen/jpx-equity-prediction)
- 初始版本：[`ee62766597550316b583dd9bad7a716d2184c426`](https://github.com/Kevin-Shih-Kai-Jen/jpx-equity-prediction/commit/ee62766597550316b583dd9bad7a716d2184c426)
- 可讀快照：[repository_snapshot/](repository_snapshot/)，16 個檔案，共 137,330 bytes。
- 來源 7 個 Python、1 份 Markdown 除錯筆記、1 份 PDF 共 9 檔逐位元保留；另新增 README、安裝版本、資料說明、忽略規則及核對紀錄。
- 既有 27 個合成測試全部通過。沒有重跑全量訓練、回測或驗證 PDF 數字。README 已記載同資料訓練與評估、epoch 參數未使用及卷積後只餘一個時間步等限制。
- 從新 repository 的精確 commit 下載 tarball，逐檔核對完整清單、大小及 SHA-256，詳见 [publication_verification.json](publication_verification.json)。
- 原始 CSV、快取、macOS 中繼資料未上傳；桌面來源及資料未更動。未新增大型資料，不需要建立新的 Release；未取回或修改任何歷史資料 Release。

本目錄新增發布紀錄及小型快照，不更換研究基準，不改寫任何舊實驗或失敗紀錄。操作暫存的清理只限本次核對下載目錄，並以本輪最終回報為準。
