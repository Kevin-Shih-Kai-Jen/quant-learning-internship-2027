# JPX：ARIMA 與殘差自相關學習實驗（待確認訓練設定）

**狀態：只完成資料與公式查核，正式回測等待建模選項確認。**

2026-09-23 使用者明確指定：X 是收盤價；ACF 用來預測殘差；差分階數 d=1；未知選擇須先詢問。

目前確定的形式：

```
z[t] = X[t] - X[t-1]
z[t+1] = a + phi*z[t] + theta*e[t] + e[t+1]
e[t+1] = c + b*e[t] + u[t+1]
ehat1 = c + b*e[t]
zhat1 = a + phi*z[t] + theta*e[t] + ehat1
ehat2 = c + b*ehat1
zhat2 = a + phi*zhat1 + theta*ehat1 + ehat2
Xhat1 = X[t] + zhat1
Xhat2 = Xhat1 + zhat2
predicted_target[t] = zhat2 / Xhat1
```

其中 u 是無法提前知道的新衝擊，預測時條件均值設零。b=0 且 c=0 時退化為 ARIMA(1,1,1)。帶殘差 AR(1) 的形式需另行檢驗。

**等待使用者回答的必要選擇：**逐檔參數或全股票共用；殘差截距 c=0／獨立 c／與 a 共用；每日一次到期 Target GD 或 252 日窗重訓。答案到達前，只做資料查核、公式推導及程式基礎核對。不自行決定重估頻率、殘差截距或參數共用方式。


## 已完成的獨立工作

- 原始訓練檔：2017-01-04 至 2021-12-03，2,332,531 筆、2,000 檔。test 未讀取。
- 依既有專案口徑，以當日前的累積 AdjustmentFactor 建立調整價格；全部因子與既有表一致。正式模型仍須明確區分原始 Close 與拆併股調整後 Close。
- 收盤價有 7,608 筆缺值，不能暗自決定填補方式。
- 以因子調整及前值填補重建的報酬，與官方 Target 並非完全相同：2,328,293 筆可比對，其中 57,161 筆差距 >1e-8，最大絕對差 0.0019778113。此處不推斷差異原因。原始 Target 保留。
- model_math.py 完成一階差分、兩步價格還原、Target 預測與穿過殘差遞迴的解析梯度。初始狀態必須由呼叫端指定，核心拒絕缺失價格。
- check_math.py：解析梯度對中央差分最大差距 3.25e-15；未来價格改動不影響過去預測；未到期標籤改動不影響 loss；b=c=0 退化為 ARIMA(1,1,1)。這些是合成數據程式核對，不是 JPX 績效。

## 需在教學中分清楚

1. 差分是 z[t]=X[t]-X[t-1]，不是 X[t]/X[t-1]-1。
2. ACF 是不同時間的相關性，不直接產生新殘差的已知值。用 AR(1) 把它寫成可估計的預測關係，未來 u 的條件均值設為零。
3. 新衝擊 u 應不可預測；若 e 有自相關，e 就不是標準模型最後的白噪音。
4. 樣本 ACF 與帶截距／不帶截距的 lag-1 OLS 斜率不必完全相同；估計方式也需寫清楚。
5. 價格先預測再取比值，是 plug-in 預測，不保證等於報酬的條件期望；對 Target MSE 直接訓練，是另外明確的建模目標。
6. 若新誤差本身設成 AR(1)，整個固定參數線性系統可改寫成具限制的 ARIMA(2,1,1)：(1-bL)(1-phi L)ΔX = a(1-b)+c(1+theta)+(1+theta L)u。新增自相關項也改變了整體模型階數。

## 來源

- JPX 官方競賽資料定義：https://www.kaggle.com/competitions/jpx-tokyo-stock-exchange-prediction/data
- ARIMA 定義：https://otexts.com/fpp3/non-seasonal-arima.html
- 殘差與新衝擊的區別：https://otexts.com/fpp3/dynamic.html
