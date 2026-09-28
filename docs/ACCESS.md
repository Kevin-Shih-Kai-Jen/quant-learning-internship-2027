# GitHub 儲存庫與資料的存取

儲存庫：[quant-learning-internship-2027](https://github.com/Kevin-Shih-Kai-Jen/quant-learning-internship-2027)

使用者於 **2026-09-28** 明確要求將本 repo 改為公開。公開 Git 檔案與已發布 Release 的附件可免登入讀取及下載；推送、上傳附件與發布新版本仍需具寫入權限的 GitHub 帳號。這次變更僅適用於本 repo，不授權更改其他儲存庫可見性。

## 新聊天室先讀的公開入口

| 內容 | 網頁 | 原始檔案（raw） |
|---|---|---|
| 協作規則 | [AGENTS.md](../AGENTS.md) | [raw AGENTS.md](https://raw.githubusercontent.com/Kevin-Shih-Kai-Jen/quant-learning-internship-2027/main/AGENTS.md) |
| 最小續接記憶 | [MEMORY_BRIEF.md](../research/project_archive_20260924/MEMORY_BRIEF.md) | [raw MEMORY_BRIEF.md](https://raw.githubusercontent.com/Kevin-Shih-Kai-Jen/quant-learning-internship-2027/main/research/project_archive_20260924/MEMORY_BRIEF.md) |
| 版本化資料索引 | [data/index.json](../data/index.json) | [raw data/index.json](https://raw.githubusercontent.com/Kevin-Shih-Kai-Jen/quant-learning-internship-2027/main/data/index.json) |

先讀上述入口，再按本次任務查 [實驗索引](EXPERIMENT_INDEX.md)、[資料指南](DATA_GUIDE.md) 與相關報告。貼上 [FUTURE_CHAT_PROMPT.md](FUTURE_CHAT_PROMPT.md) 可提供續接規則，但不會使未匯出的舊對話自動可見。

聊天室仍須有網路讀取或檔案下載能力；僅能讀網頁的工具未必能下載、解壓大型附件。若某一途徑失敗，嘗試上表 raw 入口或下方 Release 直接下載，並明確回報實際成功的範圍。不要把「能讀 README」當作「已取得全部行情檔」。

## JPX 原始資料的位置

JPX 原始資料不在 Git 程式碼 ZIP 內，也不是名為 `JPX_data.zip` 的獨立 Release 附件；它收錄在以下 TAR 分包裡。

- 版本：[`snapshot-2026-09-24`](https://github.com/Kevin-Shih-Kai-Jen/quant-learning-internship-2027/releases/tag/snapshot-2026-09-24)
- [直接下載 JPX 所在分包](https://github.com/Kevin-Shih-Kai-Jen/quant-learning-internship-2027/releases/download/snapshot-2026-09-24/snapshot-2026-09-24-data-076-project_archive_20260924.tar)
- 分包名稱：`snapshot-2026-09-24-data-076-project_archive_20260924.tar`
- 分包大小：`249374720` bytes；SHA-256：`464662e1d902c994995b878fc1578c7cb5ae5e8193f876d0136d4ccfe338c45a`
- 包內路徑：`research/project_archive_20260924/external_sources/JPX_data.zip`
- ZIP 大小：`222933782` bytes；SHA-256：`aaf930192e68d16924e662ef8190d913b06ac1e9e71c23fb8ce9ee3f418ddb0a`
- 清單依據：[catalog](../data/catalogs/snapshot-2026-09-24.json)／[raw catalog](https://raw.githubusercontent.com/Kevin-Shih-Kai-Jen/quant-learning-internship-2027/main/data/catalogs/snapshot-2026-09-24.json)

在任務專屬臨時目錄下載分包，先核對分包大小與 SHA-256，再取出需要的 ZIP 並核對其大小與 SHA-256。解開 ZIP 時只取本次需要的檔案，保留原有 train／validation／test 使用限制；不要因原始資料齊全就自動使用 test。下載前預留分包、ZIP、解壓內容及新成果所需空間。`scripts/data_archive.py fetch` 會先匿名下載公開附件，僅遇 401／403／404 時才回退到已登入的 GitHub CLI；工具用法及驗證後清理規則見 [DATA_GUIDE.md](DATA_GUIDE.md)。

**2026-09-28 存取檢查範圍：** 上述三份必要文件匿名 GET 均成功；JPX 分包匿名 HEAD 回應 200，前 1,024 bytes 的範圍下載回應 206。另以更新後工具免登入下載一份 10,240 bytes 的小型封存，核對整包及所選檔案 SHA-256 成功，已清除測試副本。詳見 [本次驗證紀錄](../data/verification/public-access-2026-09-28.json)。本次沒有重新下載全量附件、重新計算全部 SHA-256，亦不代表「JPX PCA 規劃校正」聊天室已完成下載。實際使用仍須依 catalog 驗證下載內容；舊快照驗證紀錄保留原日期與原範圍。

## 同一台電腦的 Codex：同步新成果

本機 GitHub CLI 曾確認登入 `Kevin-Shih-Kai-Jen` 並成功推送及上傳 Release。後續寫入可使用同一有效登入；若受限環境的連線檢查失敗，先在正常網路／鑰匙圈權限下唯讀重查，不可直接推論帳號失效。不要要求使用者把密碼或 Token 貼到對話。

本機曾出現 `gh` 已登入、一般 Git 推送卻仍使用舊認證的情況。以下方式曾成功推送；它只讓該次 Git 操作沿用 GitHub CLI 的認證，不修改全機設定：

```sh
git -c credential.helper= -c 'credential.helper=!gh auth git-credential' push origin main
```

公開 repo 的初次取得可直接使用 HTTPS，不需帳號：

```sh
git clone https://github.com/Kevin-Shih-Kai-Jen/quant-learning-internship-2027.git
```

## 先前私人存取問題的紀錄

2026-09-24 曾觀察到 GitHub 插件能讀帳號資料，但此私人 repo 回傳 404，installation 清單為空；本機 CLI 登入與雲端插件是不同存取途徑。後續重新連接後，插件已能讀 repo 的必要文件，但雲端聊天室的大型附件下載仍受登入途徑限制。2026-09-28 依使用者指示改為公開後，應優先使用本頁公開入口。

若未來需要寫入或存取其他私人 repo，仍須在對應環境完成 GitHub 授權；提示詞與本機瀏覽器登入不會自動授權另一個雲端環境。只具唯讀工具的聊天室應明說不能推送，保留新成果，再交給具寫入能力的任務同步。不得假裝已備份或已清理。
