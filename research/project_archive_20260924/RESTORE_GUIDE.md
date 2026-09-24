# 實驗資料無損壓縮與還原

大型原始檔以 gzip 無損壓縮保存。`COMPRESSED_DATA_MANIFEST.json` 記錄原路徑、原始大小、SHA-256、修改時間與權限，以及壓縮檔的大小和 SHA-256。每個列入清單的壓縮檔都已驗證能還原成與原檔完全相同的位元組。本輪 139 個大型檔案已完成封存；可依下列方式驗證或還原。

壓縮的目的為釋放硬碟空間。原實驗程式仍然使用原檔路徑，重新執行前，必須還原該實驗依賴的資料，包括共用資料集。還原工具只讀取檔案及計算雜湊，不會執行或反序列化任何 pickle／模型內容。

## 檢視與驗證

以下指令可從任意工作目錄執行，只需 Python 3，無需額外套件。

查看清單與未壓縮／已壓縮總大小：

```sh
python3 /Users/coolguy/.codex/.chatgpt-projects/g-p-6a8ad7dc606481919474d5424342a1c3/project_archive_20260924/restore_data.py --list
```

驗證所有 gzip 檔的大小與 SHA-256，再串流解壓驗證原始內容的大小與 SHA-256；不產生還原檔，也不需要另一份完整備份的空間：

```sh
python3 /Users/coolguy/.codex/.chatgpt-projects/g-p-6a8ad7dc606481919474d5424342a1c3/project_archive_20260924/restore_data.py --verify
```

`VERIFIED` 表示該檔完整通過。請確認最後顯示 `0 error(s)`；若有 `ERROR`，請保留壓縮檔與清單並停止依賴該資料的實驗。

## 只還原需要的實驗

`--path` 接受清單中的「專案相對路徑」，可以指定單檔或資料夾。資料夾會比對所有子層；請先用 `--list` 找到正確路徑。範例：

```sh
python3 /Users/coolguy/.codex/.chatgpt-projects/g-p-6a8ad7dc606481919474d5424342a1c3/project_archive_20260924/restore_data.py --restore --path jpx_v12_financial_growth_20260913/six_financial/training_updates.csv
```

還原某個實驗資料夾：

```sh
python3 /Users/coolguy/.codex/.chatgpt-projects/g-p-6a8ad7dc606481919474d5424342a1c3/project_archive_20260924/restore_data.py --restore --path jpx_v12_financial_growth_20260913
```

資料會回到 `project_archive_20260924` 上一層的原始相對位置。請保留這個目錄結構；搬移專案時，將封存目錄連同程式、筆記及其他資料一起搬移，並修改指令中的絕對路徑。

## 全量還原與空間需求

```sh
python3 /Users/coolguy/.codex/.chatgpt-projects/g-p-6a8ad7dc606481919474d5424342a1c3/project_archive_20260924/restore_data.py --restore
```

完整還原可能需要額外約 4.85 GB，實際值以 `--list` 的 original 總大小與已存在的原檔為準。工具會保留 gzip 副本，因此還原會再次佔用原始資料的空間。每一檔開始前都會確認可用空間至少為該原檔大小加 256 MiB 保留空間；不足時停止，可以清出空間後再次執行。

工具先寫入同一資料夾內的暫存檔，完成大小與 SHA-256 驗證並同步到磁碟後，才以不覆寫方式建立原檔；同時還原檔案權限與修改時間。若同路徑已存在內容完全相同的原檔，會顯示 `SKIP`。若已有不同內容的檔案、符號連結或其他非一般檔案，會拒絕覆寫並停止。清單中的絕對路徑、`..` 路徑與經過符號連結的路徑也會被拒絕。

一般中斷會清除該次暫存檔；若電腦斷電或程序被強制終止，可能留下 `.restore-*.tmp`。確定沒有還原作業正在執行後，可刪除這些暫存檔並重新執行；gzip 副本不受影響。請不要刪除 `compressed_data` 或修改清單。封存仍在同一顆硬碟上，尚不等於異地備份；是否已上傳 GitHub，以實際備份紀錄為準。
