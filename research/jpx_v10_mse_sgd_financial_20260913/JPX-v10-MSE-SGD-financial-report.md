# JPX v10：普通 MSE＋逐檔 SGD＋財報衰減

完成日期：2026-09-13。依要求改用 v7 的普通報酬平方誤差，搭配 v9 的逐檔 SGD 與財報衰減。兩組均從零重新訓練；本輪訓練完全不使用 soft rank。

## 結果

同一組 **953 個驗證日**，加入財報特徵後，JPX 未年化 Sharpe 由 **-0.04311345** 變成 **-0.03949089**，差值 **+0.00362255**。財報版稍有改善，但仍為負值，且低於既有 v7、v9 歷史結果。財報版的平均每日報酬預測 MSE 反而較高；不能把本次結果解讀成整體預測能力改善。

| 模型 | JPX 未年化 Sharpe | 平均每日報酬預測 MSE |
| --- | --- | --- |
| v7 普通 MSE／每日整批 | 0.00767887 | 0.000665433 |
| v9 soft rank／SGD＋財報（歷史參照） | -0.01953833 | —（loss 對象不同，不比較） |
| v10 普通 MSE／SGD，無財報 | -0.04311345 | 0.004559378 |
| v10 普通 MSE／SGD＋財報 | -0.03949089 | 0.004982070 |

MSE 先於每個預測日對已知 Target 股票平均，再對 953 日平均；只用於事後評估，不用未來 Target 更新預測。v7 參照使用每日整批更新與特徵值學習率，所以 v7 對 v10 並非只改 batch size 的單因素比較；v10 財報與無財報組才是本輪直接對照。

| 驗證折年 | 日數 | v7 | v9 | v10 無財報 | v10 財報 |
| --- | --- | --- | --- | --- | --- |
| 2018 | 245 | -0.01151377 | -0.03318395 | 0.07963128 | 0.07982330 |
| 2019 | 241 | 0.05006718 | -0.04644647 | -0.08020681 | -0.07510743 |
| 2020 | 242 | 0.05085735 | -0.02795168 | -0.08973267 | -0.08146912 |
| 2021 | 225 | -0.08760481 | 0.03153183 | -0.06341080 | -0.06627525 |

2018、2019、2020 折的財報版 Sharpe 比無財報版略好，2021 折較差。驗證日期實際為 2017-12-29 至 2021-12-01；沿用先前分折標籤，2018 折包含 2017-12-29。沒有依結果挑選年份或更換參數。

## 模型與 loss

`g_i = α + Σⱼ θⱼ xᵢⱼ + β z_i`

原有 11 個價量輸入：T5、T22、T60、V5、V22、V60、P5xV5、P22xV22、P60xV60、PR1、VR1。財報版共 13 個係數，包括 α 和 β；無財報組固定 z=0，β 初始為零且全程為零。

`L_day = (1/N) Σᵢ (g_i − Target_i)²`

每次 SGD batch 只有一檔股票，因此 `L_i=(g_i−Target_i)²`，`∇θ L_i=2(g_i−Target_i)x_i`。沒有開根號，也不把一筆 batch 的 loss 再除以整日股票數。所有已實現 Target 的股票各處理一次，沒有官方排名權重加在訓練 loss 上。

`∂L_i/∂β = 2(g_i − Target_i) z_i`

所以 **z_i=0 時，該筆 β 梯度必為 0，β 保持原值**；並不是把 β 設成零。該筆仍可更新截距與其他係數。財報版共檢查 **2,043,929** 個 z=0 的 batch，全部符合。α 也按 `2(g_i−Target_i)` 更新；其值不會直接改變同日排名，但會影響後續報酬誤差與係數更新。

## 財報特徵保持 v9 定義

`z_i = r_i × [本期 OperatingProfit / NetSales − 去年同期 OperatingProfit / NetSales]`

這是營業利益率的同比「差值」，沒有除以去年利益率，也沒有再乘上營收成長率。營收須為正，營業利益可為負。資料無法完成同比匹配時 z=0。

第一個可用交易日 r=1，後續逐交易日 0.9、0.8、…、0.1，第 11 個交易日起歸零。沿用 v9 嚴格公告時間規則（歷史 15:00 收盤門檻）、公司／季度／會計口徑／期間匹配與訂正處理。當天收盤時還未知的公告不提前使用；Target 成熟後更新時，使用原 SignalDate 已凍結的 z。

本輪直接讀取 [v9 已稽核財報特徵](/Users/coolguy/.codex/.chatgpt-projects/g-p-6a8ad7dc606481919474d5424342a1c3/jpx_v9_financial_decay_sgd_20260912/financial_signal_features.pkl)，沒有重新挑選財報欄位、標準化或截尾。953 日共有 1,864,363 個股票預測，其中 281,866 個 z 非零（15.12%）。原始極端特徵仍保留，最小值 −676.505882；詳見 [v9 財報資料規則與極端值說明](/Users/coolguy/.codex/.chatgpt-projects/g-p-6a8ad7dc606481919474d5424342a1c3/jpx_v9_financial_decay_sgd_20260912/JPX-v9-financial-decay-SGD-report.md)。本輪沒有單獨測試截尾，不能將 MSE 較高完全歸因於極端值。

## 時序與更新方式

2017-01-04 全部係數初始化一次，跨日、跨年接續。每個新成熟日，對該 SignalDate 股票隨機排列（沿用種子 `20260912+YYYYMMDD`）逐檔各更新一次，不重訪更早日期。只在 ExitDate 已到時讀取 Target。保留 2020-10-01 休市規則、SupervisionFlag 排除、缺值回補與相同股票池。

學習率沿用 v9 SGD 試探：先 0.1；成功則接續 0.2…1.0，直到首個未改善候選；0.1 失敗則依次試 0.09…0.01、0.009…0.001，一直到 0.000009…0.000001，採第一個合格者。每個候選都從同一更新前參數與梯度計算。需降低該筆實際候選參數的 MSE，且滿足 Armijo c=1e−4。梯度無窮範數≤1e−10 或全部 46 個候選失敗，記錄後跳過。本輪未採用 v7 每日整批特徵值學習率，也沒有另加學習率衰減。

| 檢查項目 | 無財報 SGD | 財報 SGD |
| --- | --- | --- |
| 訓練日期 | 1199 | 1199 |
| 逐檔 batch | 2325806 | 2325806 |
| 成功更新 | 2325798 | 2325798 |
| 梯度過小跳過 | 5 | 5 |
| 沒有合格學習率而跳過 | 3 | 3 |
| 整日訓練 MSE 在逐檔更新後上升的日期 | 546 | 550 |
| 自身 z=0 卻出現非零 β 梯度 | 0 | 0 |
| 自身 z=0 卻改變 β | 0 | 0 |

每筆成功 SGD 步驟降低的是「該筆」loss，並不保證整日 MSE 也降低。這解釋了為什麼財報版仍有 550 個日期在完成逐檔更新後整日訓練 MSE 上升。預測 MSE 是先前預測留下的值，與成熟日更新前／後 MSE 分開保存。

## 查核與輸出

兩組均完成 2,325,806 個 batch 的時序、學習率條件、β 更新鏈、SignalDate 特徵與 MSE 梯度範數恆等式檢查；所有 1,199 個訓練日的整日 MSE 已重算。額外獨立重算無財報 3,597、財報 5,495 個保存的 batch 梯度與完整學習率試探，每組另有 5 個真實 batch 的有限差分檢查。完整逐步梯度獨立重算的範圍是保存的樣本，並非每個 batch 的完整重演。

每組全部 1,199 個預測日、2,326,022 個預測列，皆以保存的當日係數重新計算分數與唯一整數排名。953 日官方風格多空各 200 檔、2→1 權重的 spread 獨立重算通過。財報組分數最大誤差 3.55e-15，官方 spread 最大誤差 1.07e-14。所有驗證日皆可評分，沒有選到缺 Target 的股票。兩組在第一筆財報特徵可訓練前的 214 個日期，係數完全一致。

最後預測日 2021-12-01；最後參數包含成熟到 2021-12-03 的標籤，不能用最後參數倒推更早日期。財報版最後 β=0.016313010240，α=0.013522329770。

- [最終係數](/Users/coolguy/.codex/.chatgpt-projects/g-p-6a8ad7dc606481919474d5424342a1c3/jpx_v10_mse_sgd_financial_20260913/final_coefficients.csv)
- [完整比較數值](/Users/coolguy/.codex/.chatgpt-projects/g-p-6a8ad7dc606481919474d5424342a1c3/jpx_v10_mse_sgd_financial_20260913/results.json)、[逐日比較](/Users/coolguy/.codex/.chatgpt-projects/g-p-6a8ad7dc606481919474d5424342a1c3/jpx_v10_mse_sgd_financial_20260913/daily_comparison.csv)
- [財報版獨立查核](/Users/coolguy/.codex/.chatgpt-projects/g-p-6a8ad7dc606481919474d5424342a1c3/jpx_v10_mse_sgd_financial_20260913/mse_sgd_financial/audit.json)、[無財報版獨立查核](/Users/coolguy/.codex/.chatgpt-projects/g-p-6a8ad7dc606481919474d5424342a1c3/jpx_v10_mse_sgd_financial_20260913/mse_sgd_control/audit.json)
- [財報版每日參數](/Users/coolguy/.codex/.chatgpt-projects/g-p-6a8ad7dc606481919474d5424342a1c3/jpx_v10_mse_sgd_financial_20260913/mse_sgd_financial/parameter_history.csv)、[每日更新](/Users/coolguy/.codex/.chatgpt-projects/g-p-6a8ad7dc606481919474d5424342a1c3/jpx_v10_mse_sgd_financial_20260913/mse_sgd_financial/training_updates.csv)、[逐筆紀錄](/Users/coolguy/.codex/.chatgpt-projects/g-p-6a8ad7dc606481919474d5424342a1c3/jpx_v10_mse_sgd_financial_20260913/mse_sgd_financial/batch_trace.csv.gz)
- [擬合前計畫](/Users/coolguy/.codex/.chatgpt-projects/g-p-6a8ad7dc606481919474d5424342a1c3/jpx_v10_mse_sgd_financial_20260913/experiment_plan.md)、[來源雜湊](/Users/coolguy/.codex/.chatgpt-projects/g-p-6a8ad7dc606481919474d5424342a1c3/jpx_v10_mse_sgd_financial_20260913/source_hashes.json)

這是未扣交易成本的歷史逐日驗證。相同驗證期已比較多個版本，並非新的獨立 test，也不是排行榜成績。本次依指定方法完整保存，不依結果另調參數。
