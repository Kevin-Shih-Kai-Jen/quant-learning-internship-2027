# v12：價量模型加入六個財報特徵（擬合前固定）

2026-09-13。使用者確認：實際值與 Forecast 都加入；實際成長用去年同期；EPS 原值；Forecast 成長的基期是去年對應完整財年的實際值；加入原本價量模型。

18 個共同學習參數：截距 + 11 個既有價量特徵 + 六個財報特徵。PR1、VR1 沿用 v11 各自除以包含 SignalDate 的 22 交易日樣本標準差，不減均值。原營業利益率差值財報項被這六項取代，沒有額外保留。六項持續至下一個相關公告更新，不乘衰減係數。

六項依序：ActualNetSalesGrowth、ActualOperatingProfitGrowth、ActualEPS、ForecastNetSalesGrowth、ForecastOperatingProfitGrowth、ForecastEPS。

成長率統一為 (current−base)/abs(base)，保留負值及負轉正訊號；−100→50=1.5、−100→−60=0.4、−100→−150=−0.5。成長率存小數而非百分數。基期為 0、缺值、無符合比較期或口徑不符，該特徵不可用，零回補並保留旗標。EPS 直接使用來源原值，負值有效，不縮放、不截尾、不做價格比率，也不做未經要求的拆股調整。

實際成長：同公司、會計口徑、季度及財年起日／期間迄日／財年迄日，日期各減一年的同期間數字。累計 2Q 對去年累計 2Q，不改成單季差額。實際 EPS 是最新已知實際報告期間值。

Forecast 成長：可辨識的全年 Forecast 對其目標財年前一完整 FY 實際數字。常規 1Q/2Q/3Q 報表 Forecast 視為 CurrentFiscalYearEndDate 的全年預測；常規 FY 報表 Forecast 視為下一財年預測（目標起日=本 FY 迄日+1天，迄日=本 FY 迄日+1年）。分母必須匹配目標起迄日各減一年的實際 FY，不能用去年季報或去年 Forecast。

ForecastRevision 僅納入 TypeOfCurrentPeriod=FY 的修正，目標為該列 CurrentFiscalYearStartDate/EndDate；其他期間修正可能混有半年預測或半年實際差異，保守排除其對全年特徵的更新。修正缺乏會計口徑時，只在當時已知相同財年口徑唯一時套用，不用未來報表猜口徑。NumericalCorrection 只匹配唯一已知原期間；按實際有提供的欄位修正，空白欄位不當作刪除，明確非數字標記視為不可用。常規完整報表則把該期缺欄視為不可用。

所有原始資料只取使用者 ZIP train_files/financials.csv；實際/Forecast 訂正均依公告時序處理，絕不回填歷史。沿用 max(Date,DisclosedDate) 與 15:00 收盤門檻，當天 15:00 及之後公告下一交易日可用。SignalDate 凍結六項與來源 EventId，Target 成熟日使用原值。非相關舊報表更新不覆蓋最新實際/Forecast 期間，但基期訂正會於訂正日影響後續特徵。

主模型 `six_financial`，對照 `price_only` 將六項設為 0；兩組其他設定完全一致。均用 v10/v11 普通報酬 MSE、逐檔單次 SGD、一次零初始化、連續參數、相同每日亂序種子 20260912+YYYYMMDD。完全沿用原學習率候選與 Armijo c=1e−4，最小 1e−6；若原值 EPS 使所有候選不合格，就記錄跳過，不擅自擴大搜尋範圍。每個新版本只跑一次，不依結果调參數。

比較相同 953 個驗證日官方風格多空各200檔、2→1權重、未年化 Sharpe 與報酬 MSE；另引用 v11 decay_std22。保存全量參數/排名/更新/梯度摘要與獨立稽核。歷史資料已多輪比較，未扣成本，沒有新独立 test。

實作修正紀錄：首次運算後追加期別稽核發現 74 個狀態事件在公司縮短財年時仍保留原先推定較晚 FY 末日。已修正為最新常規報告的 Forecast 目標期覆蓋舊推定期間；年度 ForecastRevision 按財年起日推進，同起日的新修正可更改迄日；舊期間 NumericalCorrection 只修補對應狀態，不把舊目標重新升為最新。原始初步運算存 pre_period_fix，重新從零運算是資料期別程式修正，沒有改特徵公式、參數或依績效調參。
