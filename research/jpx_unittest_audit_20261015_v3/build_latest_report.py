from pathlib import Path
import json, shutil
root=Path(__file__).resolve().parent
latest=root/'latest_revision_1124'
r=json.loads((latest/'results/test_results.json').read_text())
issues=[x for x in json.loads((root/'issue_catalog.json').read_text()) if x['id']!='B16']
issues.extend([
 dict(id='B17',category='新增分支／P2',title='傳入非零 std 的分支沒有回傳結果',source_file='features.py',line=44,evidence='以 std=2.0、mean=10.0 呼叫，函式回傳 None；此分支只有複製 DataFrame，尚未做轉換或 return。這是使用可選 std 分支時觸發，目前 common_process 仍使用預設分支。',tests=['test_normalizer_reuse.NormalizerReuseTests.test_nonzero_scalar_std_returns_the_transformed_frame']),
 dict(id='B18',category='參數重用／P2',title='if not std 不能判斷六個特徵的標準差向量',source_file='features.py',line=56,evidence='把 normalize_data 自己回傳的六組 mean/std 用於重播，if not std 引發 NumPy truth value is ambiguous。測試已先把常數欄位尺度替換為 1，單獨確認向量處理問題。若這個新分支要重用訓練的逐特徵參數，就需要支援向量；若只支援純量，則仍缺少這條必要的逐特徵重播流程。',tests=['test_normalizer_reuse.NormalizerReuseTests.test_fitted_featurewise_metadata_can_be_reused'])
])
source=(latest/'source_snapshot/features.py').read_text().splitlines()
for item in issues:
 if item['id']=='B08':item['line']=next(i for i,s in enumerate(source,1) if 'std = scaler.var_' in s)
 if item['id']=='B11':item['line']=next(i for i,s in enumerate(source,1) if 'def create_time_sequence(' in s)
 if item['id']=='B18':item['line']=next(i for i,s in enumerate(source,1) if 'if not std:' in s)
failed=set(r['failure_tests']+r['error_tests'])
assert {t for x in issues for t in x['tests']}==failed
(latest/'issue_catalog.json').write_text(json.dumps(issues,ensure_ascii=False,indent=2)+'\n')
lines=['# JPX unittest v3｜最後一次稽核：2026-10-15 11:24（台北）','',
'**104 項：81 通過、19 項 assertion failure、4 項 error、0 跳過。23 個未通過的方法對應 17 組問題：7 組主流程問題、10 組條件性／邊界問題。** 一項方法包含兩個失敗子案例，因此 unittest 原始輸出顯示 failures=20, errors=4。','',
'你在測試期間持續修改 normalize_data。本報告固定對應 11:24 保存的來源快照；之後的編輯請重跑測試，不要把舊結果套用到新的程式。最新封存來源位於 latest_revision_1124/source_snapshot，雜湊位於同目錄的 source_manifest.json。完整結果位於 results_1124/test_results.json。','',
'我們只新增測試與報告，沒有修改 codes 的模型或資料處理邏輯。使用者在本輪自行修改的 features.py 已分版保存。','',
'## 已解除的問題','',
'先前 std: None 是必填參數，現已改成 std: float | None = None，所以 B16「呼叫漏傳 std」在最後一次測試已通過，不再列為目前問題。feature_columns 也已正確傳入存檔函式，模型歸位至 models.py 後的 forward 測試通過。','',
'## 主流程優先順序','',
'| 編號 | 問題 | 影響 |','|---|---|---|']
for x in issues[:7]:lines.append(f"| {x['id']} | {x['title']} | {x['category']} |")
lines+=['','## 全部可重現問題','']
for x in issues:
 lines += [f"### {x['id']}｜{x['title']}（{x['category']}）",'',f"來源：`{x['source_file']}:{x['line']}`。{x['evidence']}",'','對應測試：','']
 lines += ['- `'+t+'`' for t in x['tests']]
 lines.append('')
lines += ['## 通過的測試及觀察','',
'62 項既有測試的副本已更新模型位置、分數 dict 與建構測試的資料替身，全部通過。新增 42 項中 19 通過、23 未通過。原本 v1/v2 沒有被修改。測試涵蓋張量轉換、資料列篩選、股票與日期排序、建窗和 Target 對齊、訓練／評估 batch、no-grad、預測串接、排名配權、Sharpe、Rank IC，以及真正 PyTorch 序列化的模型／metadata 往返。','',
'- 42 日卷積搭配 kernel=42，LSTM 收到 1 個時間步：記錄為架構選擇。','- expanding folds 重用同一模型及 optimizer：記錄為暖啟動選擇。','- 驗證集分開建 42 日窗會失去每檔前 41 列：是否應帶入訓練尾段歷史需先定義評分範圍。','- main 目前選擇訓練入口；early stopping 尚未實作。未把這些選擇或未新增功能另算 bugs。','',
'## 如何解讀與重跑','',
'```bash','python3.12 -B /Users/coolguy/Desktop/JPX/debug/JPX_debug_v3/run_tests.py','```','',
'上述指令測目前的 codes。要精確重播這份報告，從存檔目錄執行 `python3.12 -B JPX_debug_v3/run_tests.py --codes latest_revision_1124/source_snapshot`。可以在最後加上表列的完整測試名稱，單獨重現。','',
'環境為 Python 3.12.2、PyTorch 2.11.0、NumPy 2.0.0、pandas 2.2.2、SciPy 1.17.1、scikit-learn 1.8.0。只使用 CPU 小型合成資料，未讀市場 CSV、未跑完整訓練，序列化只在記憶體進行。後段存檔／報表測試替換訓練，避免 epochs 介面錯誤遮蔽它們；真實訓練呼叫另有 smoke tests。','',
'標準化下游測試保留對先前必填 std 的相容替身；最後快照已使用可選 std，因此最後一次執行並不需要或啟動該替身。新參數重用測試直接呼叫真正 normalize_data。','',
'本次未量測行覆蓋率，也不能保證找出所有 bugs。未驗證真實拆股／除息與 ExpectedDividend 口徑、Target 可知時間、完整市場池、同分排名政策、MPS/CUDA、長時間資源用量或投資績效。合成無效輸入的失敗不表示真實 CSV 一定含有那些值。','',
'## 版本紀錄','',
'首輪 101 項、必填 std 版本的 102 項，以及這次 104 項結果均保留。102 項版本已獨立保存於 Git commit dee9551fb66ccef385bfba03a169e5b78d3fc946；最新修訂是新增提交，沒有刪除或覆蓋舊 Git 歷史。','']
report='\n'.join(lines)
(latest/'REPORT.md').write_text(report)
(root/'JPX_debug_v3/REPORT.md').write_text(report)
shutil.copytree(latest/'results',root/'JPX_debug_v3/results_1124',dirs_exist_ok=True)
(root/'JPX_debug_v3/README.md').write_text('''# JPX unittest v3

最後測試版本：2026-10-15 11:24（台北）。**104 項：81 通過、19 項 assertion failure、4 項 error。** 23 個未通過方法對應 17 組問題：7 組主流程、10 組條件性／邊界問題。完整清單與證據見 [REPORT.md](REPORT.md)，原始輸出見 [results_1124/test_results.txt](results_1124/test_results.txt)。

執行目前的 codes：

```bash
python3.12 -B /Users/coolguy/Desktop/JPX/debug/JPX_debug_v3/run_tests.py
```

只跑某一組：

```bash
python3.12 -B /Users/coolguy/Desktop/JPX/debug/JPX_debug_v3/run_tests.py test_audit.TrainingContractTests
python3.12 -B /Users/coolguy/Desktop/JPX/debug/JPX_debug_v3/run_tests.py test_normalizer_reuse.NormalizerReuseTests
```

使用 `--report-dir /新結果目錄` 可保存紀錄；用 `--codes /指定程式目錄` 可重播封存快照。執行器會印出實際來源、記錄依賴版本與 SHA-256，且拒絕把零個測試算成成功。退出碼 1 表示測試未全數通過。

原本 v1/v2 保留不動。v3 是 62 項更新介面的回歸測試加上 42 項新增測試；沒有修改 production code，沒有使用 expectedFailure 隱藏問題。使用 CPU 合成資料與記憶體序列化，不讀 CSV、不覆蓋 checkpoint、不啟動完整歷史訓練。

你在執行期間修改了 normalize_data；目前 std 預設 None 已解除原本缺參數問題，但新分支的回傳與逐特徵參數重用仍有失敗。報告只代表上述固定快照。舊 102 項結果仍在 results/；最後結果在 results_1124/。

測試不能保證涵蓋所有 bugs。真實資料語意、Target 到期、GPU 和長時間訓練等未驗證範圍已在完整報告列出。
''')
readme=root/'README.md'
readme.write_text('''# 最後修訂：2026-10-15 11:24

最新結果以 [latest_revision_1124/REPORT.md](latest_revision_1124/REPORT.md) 為準：104 項，81 通過、19 項 assertion failure、4 項 error，17 組問題。來源與雜湊同在 latest_revision_1124；原 102 項來源與結果保留於以下原紀錄，且已保存於 commit dee9551fb66ccef385bfba03a169e5b78d3fc946。重跑目前程式請看 [測試入口說明](JPX_debug_v3/README.md)。

---

'''+readme.read_text())
print('Updated latest report:',r['total'],'tests and',len(issues),'issue groups')
