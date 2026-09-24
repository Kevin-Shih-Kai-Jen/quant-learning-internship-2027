# 私人儲存庫的存取

儲存庫：https://github.com/Kevin-Shih-Kai-Jen/quant-learning-internship-2027

## 同一台電腦的 Codex

本次已確認 GitHub CLI 登入 `Kevin-Shih-Kai-Jen`，可建立私人儲存庫、推送內容及上傳 Release。後續 Codex 可以使用同一登入；若受限環境的連線檢查失敗，先在正常網路／鑰匙圈權限下唯讀重查，不可直接推論帳號失效。不要要求使用者把密碼或 Token 貼到對話。

本機曾出現 `gh` 已登入、一般 Git 推送卻仍使用舊認證的情況。本次以下方式已成功推送；它只讓該次 Git 操作沿用 GitHub CLI 的認證，不修改全機設定：

```sh
git -c credential.helper= -c 'credential.helper=!gh auth git-credential' push origin main
```

若第一次取得私人 repo 也遇相同問題，可使用：

```sh
git -c credential.helper= -c 'credential.helper=!gh auth git-credential' clone https://github.com/Kevin-Shih-Kai-Jen/quant-learning-internship-2027.git
```

## 其他 ChatGPT／Codex 聊天室

貼上 [FUTURE_CHAT_PROMPT.md](FUTURE_CHAT_PROMPT.md) 並說明當次任務。新聊天室還需要真正能讀取這個私人 repo 的工具與授權；網址或提示詞本身不會授予存取權，也不會自動帶入舊對話。

2026-09-24 本次實際觀察：GitHub 插件能讀到帳號基本資料，但查此新私人 repo 回傳 404，插件可見 installation 清單為空；這與本機 CLI 已登入是不同的存取途徑。若遇此情況，先在 ChatGPT／Codex 的 Plugins 頁開啟 GitHub，依其連線提示完成 GitHub App 設定，將本 repo 納入允許存取的儲存庫。GitHub App 設定入口可從 https://github.com/settings/installations 查看；若沒有相關安裝，應回插件連線流程完成，而不是擅自改 repo 為公開。

設定介面與連線需求會依版本不同；以實際畫面為準。可參考 [OpenAI 官方插件說明](https://learn.chatgpt.com/docs/plugins)；Codex cloud 的設定亦要求選擇允許存取的 GitHub 儲存庫，見 [官方 cloud 設定](https://learn.chatgpt.com/docs/cloud)。

完成後應以讀取本 repo 的 README 或 MEMORY_BRIEF 成功作為驗證；只看到個人帳號不代表已有 repo 權限。只具備唯讀連接器的聊天室不一定能推送或執行下載工具，須清楚說明能力限制，改交給具 Git／GitHub CLI 存取的 Codex 任務執行。
