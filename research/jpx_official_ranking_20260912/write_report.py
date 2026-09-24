from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent
r=json.loads((ROOT/'results.json').read_text());a=json.loads((ROOT/'independent_audit.json').read_text())
assert a['all_checks_passed']
t=r['total'];lines=['# 價量水準 T 模型：依 JPX 官方公式進行排名評分','',
f"953 個歷史驗證區間，官方尺度的未年化 Sharpe = **{t['official_style_unannualized_sharpe']:.8f}**。乘 √252 的年化參考值為 **{t['annualized_equivalent_sqrt252']:.6f}**；後者不是競賽排行榜分數。",'',
'## 使用哪一組模型','',
'沿用最初價格與成交量水準的 T 值模型：T5、T22、T60、V5、V22、V60，加上 T5×V5、T22×V22、T60×V60，共九個特徵與截距。價格及成交量做原有股票分割調整。這不是後來的 return T 模型，也没有加入 1 日價格特徵。','',
'每個驗證年度只用前一年資料訓練，使用原來未加 Ridge 的線性 MSE 模型，重新核對訓練樣本數與係數。已知標籤退出日不能超過該年度首次預測日期。沒有 regime、預測正負號資金轉移、目標價出場、交易成本或 softmax。','',
'## 官方權重與公式','',
'官方明確指定多頭取排名最前的 200 檔、空頭取最末的 200 檔，兩邊都從各自最極端的一檔開始給 2，線性下降至第 200 檔的 1。空頭最不看好的股票拿到最大權重。[JPX 官方評分程式](https://www.kaggle.com/code/smeitoma/jpx-competition-metric-definition)','',
'第 i 個位置的原始權重為 `w_i = 2 − (i−1)/199`，i=1…200。每側權重總和 300，平均 1.5。若僅換成該側內部的比例，第一檔是 2/300 = 0.6667%，最後一檔是 1/300 = 0.3333%。這些是側內比例，並非官方要求提交的資金佔比；正式提交仍然只有 Rank。','',
'官方程式的每日分數為：','',
'`S_long = sum(w_i × Target_long_i) / 1.5`','',
'`S_short = sum(w_i × Target_short_i) / 1.5`','',
'`DailySpread = S_long − S_short`','',
'`Score = mean(DailySpread) / std(DailySpread, ddof=1)`','',
'注意官方除的是平均權重 1.5，不是權重總和 300。因此這個 DailySpread 不能直接當作帳戶報酬去複利。若作為說明，轉成多空各占本金 50%、總名目曝險 100% 的理論報酬，則是 DailySpread/400；乘除同一正常數不會改變 mean/std，Sharpe 完全相同。本次直接執行官方原始函式核對分數。','',
'即使某一天所有 g 都是正數，最後 200 名仍視為空頭。g 在這裡只決定排名；將全部 g 同乘一個正常數，排名與官方分數不變。','',
'## 驗證結果','', '| 驗證年度 | 評分區間數 | 官方尺度 Sharpe | 年化參考 Sharpe |','|---|---:|---:|---:|']
for y in r['annual']:lines.append(f"| {y['validation_year']} | {y['scored_days']} | {y['official_style_unannualized_sharpe']:+.6f} | {y['annualized_equivalent_sqrt252']:+.6f} |")
lines += [f"| 全期間 | {t['scored_days']} | {t['official_style_unannualized_sharpe']:+.6f} | {t['annualized_equivalent_sqrt252']:+.6f} |",'',
'全期間是將每日 spread 串接後重新計算，不是四個年度 Sharpe 的平均。年度沿用之前的訓練／驗證切分，首個訊號可能落在上一年末；最後一年資料截至 2021-12-01 訊號，對應 2021-12-03 結果。2018 與 2021 分數為負，顯示這個排序訊號在不同年份並不穩定。','',
'## Target 在 train、validation、test 的角色','', '| 階段 | 模型預測時是否能讀 Target？ | Target 的用途 |','|---|---|---|',
'| 訓練 | 可以作為 y，但不放進輸入特徵 X | 計算 loss，學習係數 |',
'| 本次歷史驗證 | 不可以 | 先生成與保存 g、Rank，再交給評分函式計算結果 |',
'| 正式 Kaggle test | 參賽者看不到當天的真實 Target | 主辦方使用隱藏 Target 計算分數 |','',
'因此「test 不用 Target」應理解為：你的預測函式不使用它；評分仍然需要未來實現的真實報酬。把本次 validation 的 Target 當輸入來計算 g 或 Rank，就會造成資料洩漏。','',
'本次預測函式明確拒絕包含 Target 的輸入表，產出的 ranks 檔案只有 Date、SecuritiesCode、g、Rank、Fallback、ValidationYear；寫檔後才與保留的 Target 合併評分。[官方資料說明](https://www.kaggle.com/competitions/jpx-tokyo-stock-exchange-prediction/data)','',
'## 歷史股票池與後備值','',
f"提供的核心股票檔在這段期間每天有 {t['stocks_per_day_min']}–{t['stocks_per_day_max']} 檔股票。本次主結果使用當日可見核心股票，排除當日 SupervisionFlag=True 的股票，共排除 354 筆股票日觀測；這是歷史可投資集合的近似。正式競賽一定要以 API 的 sample_prediction 指定清單為準，保留原列順序並為全部指定股票提供唯一 Rank。",'',
'原競賽股票池由主辦方按 2021 年末條件建立；本次使用它提供的歷史資料，沒有自行重建 2018–2021 每一天的全市場股票池。提供的 ZIP 沒有各歷史日期的正式 API sample_prediction，也沒有新的正式 test。不要把本次結果當成官方 private leaderboard 成績。[官方評分說明與投資資格](https://www.kaggle.com/code/smeitoma/jpx-competition-metric-definition)','',
f"先前交易回測會略過 T 值算不出的股票，但競賽需要完整排名。本次推論階段將缺少的基礎 T 值填為 0，再重算交乘項；訓練規則沒有改。共有 {t['fallback_rows']:,} 筆排名使用後備值，前後 200 名合計 {t['selected_fallback_rows']:,} 筆，占全部選中部位 {t['selected_fallback_rows']/(953*400):.2%}。後備值是明確的實作選擇，不是官方規定。",'',
f"有 {t['unavailable_target_rows']} 筆歷史 Target 缺失，它們仍參與預測排名，沒有按未來標籤是否存在篩選股票。這 {t['unavailable_target_rows']} 筆都沒有進入前後 200 名，所以全部 {t['scored_days']} 個區間都能完整評分，沒有把未知持倉報酬硬填為 0。",'',
f"若不排除 SupervisionFlag，而對提供的全部核心股票評分，分數為 {r['full_core_universe_sensitivity']['total']['official_style_unannualized_sharpe']:.8f}，年化參考 {r['full_core_universe_sensitivity']['total']['annualized_equivalent_sqrt252']:.6f}。這份敏感度結果用於區分股票池規則差異，沒有依分數挑選較有利的股票池。",'',
'## 驗證與輸出','',
'- 訓練係數與原版模型核對；各年訓練標籤可用時間核對。','- 全部每日排名唯一且連續為 0…N−1；同分以股票代碼排序，沒有按 Target 打破平手。','- 每天確實選多頭 200、空頭 200，兩側不重疊，權重方向及端點核對。','- 獨立重算所有選中股票 g、全部日 spread 和全期間分數；與官方函式一致。','- 測試所有 g 為正時仍保留空頭，以及 g 正比例放大不改變排名。','- 已反覆研究的歷史資料不能視為新的未見測試集；這次也沒有重新調參挑分數。','',
'完整係數見 model_fits.json；每日排名見 ranks_2018.csv.gz 至 ranks_2021.csv.gz；多空部位見 selected_200_each_side.csv.gz；每日分數見 daily_spread_returns.csv。historical_submission_example.csv 只是最後一個歷史訊號日的格式示例，不能直接當作正式 test 的提交檔。']
(ROOT/'JPX-official-ranking-report.md').write_text('\n'.join(lines).replace('没有','沒有')+'\n')
print('Report written')
