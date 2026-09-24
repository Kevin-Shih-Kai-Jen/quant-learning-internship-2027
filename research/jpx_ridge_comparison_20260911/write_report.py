from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent
r=json.loads((ROOT/'results.json').read_text());a=json.loads((ROOT/'independent_audit.json').read_text());c=json.loads((ROOT/'return_t_calendar_sensitivity.json').read_text())
assert r['complete'] and a['all_checks_passed']
variants=r['variants'];improved=[v for v in variants if v['drawdown_lower']]
lines=['# 10 個既有版本加入 Ridge：成對比較','',f'10 個版本中，{len(improved)} 個最大回撤降低，{10-len(improved)} 個增加。這是股票預測模型加入 Ridge 的效果；市場狀態模型未加入 Ridge。','',
'同一段 953 個交易區間（2018 年至 2021-12-03，最後一年未滿一年），不含交易成本。原版 OLS 與 Ridge 分別依自己的預測重新排名、選股與設定目標出場價。每邊前三名、該邊資金按 50%／30%／20% 分配；保留錯誤方向訊號的資金轉移規則。','',
'## 原目標價出場規則','',
'| 版本 | 原報酬 | Ridge 報酬 | 原最大回撤 | Ridge 最大回撤 | 回撤改善（百分點） | 原 Sharpe | Ridge Sharpe |',
'|---|---:|---:|---:|---:|---:|---:|---:|']
def row(v):
    o=v['ols'];n=v['ridge']
    return f"| {v['name']} | {o['cumulative_return']:+.2%} | {n['cumulative_return']:+.2%} | {abs(o['max_drawdown']):.2%} | {abs(n['max_drawdown']):.2%} | {v['drawdown_reduction_percentage_points']:+.2f} | {o['sharpe_252_rf0']:.3f} | {n['sharpe_252_rf0']:.3f} |"
lines += [row(v) for v in variants if not v['close_only']]
lines += ['', '## 收盤出場規則','', '收盤出場四組用原執行引擎重新逐筆模擬，包含停牌延後退出與被占用資金。', '', '| 版本 | 原報酬 | Ridge 報酬 | 原最大回撤 | Ridge 最大回撤 | 回撤改善（百分點） | 原 Sharpe | Ridge Sharpe |','|---|---:|---:|---:|---:|---:|---:|---:|']
lines += [row(v) for v in variants if v['close_only']]
lines += ['', '日經基準累積報酬 '+f"{r['benchmark']['cumulative_return']:+.2%}"+'，最大回撤 '+f"{abs(r['benchmark']['max_drawdown']):.2%}"+'。回撤改善為「原回撤幅度減 Ridge 回撤幅度」；正數代表改善。','', '## 損失函數與 λ 的選擇','',
'`Loss = mean((y − alpha − X beta)^2) + lambda * sum(beta_j^2)`。截距 alpha 不受懲罰。保留原特徵單位及價量交乘項，沒有另加標準化、裁切或其他特徵。','',
'每年股票模型仍只用前一年資料。λ 候選固定為 0、0.001、0.01、0.1、1、10、100。每個外層訓練年內做兩個依日期切分的驗證：前半段訓練→下一季驗證，前三季訓練→最後一季驗證；訓練標籤退出日必須已知。按兩折合併均方誤差挑選，不用外層回撤或報酬挑 λ。MSE 的 λ 定義與使用總平方誤差的程式庫 alpha 不可直接混用。','',
'| 股票特徵 | 2018 λ | 2019 λ | 2020 λ | 2021 λ |','|---|---:|---:|---:|---:|']
for family,label in [('level_t','價量水準 T'),('return_t','Return T＋1 日價格 T')]:
    fits=json.loads((ROOT/family/'model_fits.json').read_text())
    lines.append('| '+label+' | '+' | '.join(str(f['lambda']) for f in fits)+' |')
lines += ['', '多數年份選到搜尋上限 100，代表目前候選中以該值的訓練期內驗證 MSE 最低；不代表全域最佳 λ，也不代表它最能降低投資組合回撤。相同股票特徵與訓練資料的配置版本共用同一組係數，避免因配置不同而任意重調 λ。','',
'## 實際多空曝險','', 'Ridge 懲罰係數，並未直接限制投資組合曝險。截距不受懲罰，加上原本的錯誤方向資金轉移，仍可能把資金集中到同一方向。下表為 953 個區間平均持有曝險；單邊天數指多或空一側曝險為零。','',
'| 版本 | 平均多頭 | 平均空頭 | 原單邊天數 | Ridge 單邊天數 |','|---|---:|---:|---:|---:|']
for v in variants:lines.append(f"| {v['name']} | {v['mean_long_exposure']:.2%} | {v['mean_short_exposure']:.2%} | {v['old_one_sided_days']} | {v['one_sided_days']} |")
lines += ['', '## Return T 舊版休市日日期問題與敏感度檢查','',
'重現時發現，Return T 舊版訂單沿用含 2020-10-01 全市場休市日的日期表，而組合執行區間已跳過該日。2020-09-30 訊號所產生的六筆訂單因此沒有進入原執行區間；其前一日訊號的預定退出日也不同。上面的主表保留這個原設定，確保與剛才列出的 OLS 報酬完全對應，且 Ridge 使用相同設定。','',
'額外把 OLS 和 Ridge 的訂單進出場日都映射到實際市場交易區間，係數與訓練維持相同，結果如下。此修正沒有覆寫先前實驗或目前啟用設定。','',
'| Return T 日期修正版 | 累積報酬 | 最大回撤 | Sharpe |','|---|---:|---:|---:|']
for key,label in [('ols','OLS'),('ridge','Ridge')]:
    v=c[key];lines.append(f"| {label} | {v['cumulative_return']:+.2%} | {abs(v['max_drawdown']):.2%} | {v['sharpe_252_rf0']:.3f} |")
lines += ['', '## 各年報酬與回撤','', '| 版本 | 年度 | 原報酬 | Ridge 報酬 | 原回撤 | Ridge 回撤 |','|---|---:|---:|---:|---:|---:|']
for v in variants:
    for annual in v['annual']:
        o=annual['ols'];n=annual['ridge'];lines.append(f"| {v['name']} | {annual['year']} | {o['cumulative_return']:+.2%} | {n['cumulative_return']:+.2%} | {abs(o['max_drawdown']):.2%} | {abs(n['max_drawdown']):.2%} |")
lines += ['', '## 驗證與限制','',
'- 10 組 OLS 基準逐日完整重現，最大每日報酬差為 '+f"{max(v['old_daily_return_reproduction_max_error'] for v in variants):.3g}"+'。','- λ=0 係數與原 OLS 對照；Ridge 與增廣最小平方法交叉檢查。獨立 SVD 重算全部內層候選 λ 的驗證 MSE；檢查日期切分、標籤可用時間與選中股票預測。','- 逐筆成交價、分割調整、損益、日報酬複利及最大回撤重算通過；期末無未結清持倉。','- 本次只測已列出的版本，未新增參數搜尋或修改目前啟用模型。','- 已反覆查看的 2018–2021 年資料屬研究驗證，沒有新增未看過的測試期間；未扣交易、滑價或借券成本。','',
'完整數據見 `results.json`、各版本的 `daily_returns.csv`、`selected_trades.csv` 與 `position_events.csv`；λ 選擇及係數見兩個特徵資料夾。']
(ROOT/'JPX-Ridge-comparison-report.md').write_text('\n'.join(lines)+'\n')
print('Report written; improved',len(improved),'of',len(variants))
