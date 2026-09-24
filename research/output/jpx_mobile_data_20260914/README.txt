JPX 模型診斷資料包｜2026-09-14

先讀 PDF：10 頁手機直式版。四組均保留 exp(-a/9) 財報事件衰減。
M1 = v14 報酬 MSE、單獨 SGD
M2 = v14 報酬 MSE、SGD + 每日 floor(sqrt(N)) 次整體更新
S1 = v15 sigmoid soft rank、單獨 SGD
S2 = v15 sigmoid soft rank、SGD + 每日 floor(sqrt(N)) 次整體更新
尚未完成不衰減對照，本包沒有不衰減實驗結果。

期間與範圍
- 驗證：2017-12-29 至 2021-12-01，953 日；Rank IC 有定義的 952 日。
- 訓練：1,199 個標籤到期日（含暖身），最後更新 2021-12-03。
- 2020-09-29 的 Target 全為 0，Rank IC 留空，不以 0 替代。
- 每天預測 1,896 至 2,000 檔；前 200 買入、後 200 賣出；邊內權重 2 至 1。
- 所有收益未扣成本。Sharpe 為未年化，std 使用 ddof=1。
- 不含股票全量排名與大型訓練逐步 trace，保留手機可攜的每日數據與案例。

入口檔案
model_summary.csv：4 列模型總表；MeanDailyReturnBP 單位 bp (0.01%)。
comparison.csv / comparison_by_year.csv：原始已稽核比較；ValidationYear 是既有驗證組別標記，不一定和曆年邊界完全相同。
all_models_daily.csv：4 x 953 = 3812 列；不同 loss 不適用的欄位留空。
all_models_training_summary.csv：4 x 1199 = 4796 列；完整逐日更新前後參數在各組 training_updates.csv。
all_models_final_parameters.csv：27 個參數 x 4 組，日期均為 2021-12-03；不得用最後參數回頭當成歷史預測權重。
financial_feature_summary.json：15 個原始財報成分範圍、事件數與門檻計數；其範圍涵蓋原始訊號資料，不能當成已過濾訓練分布。
M1/M2/largest_prediction_errors.csv：刻意選取最大平方誤差的案例，不具代表性；g 是預測報酬，1.0 = 100%。
M1/M2/diagnostic_daily.csv：MSE 分解與零預測基準。
S1/S2/daily_soft_rank_loss.csv：完整有限標籤驗證股票池上的 soft-rank loss；tau = 1。
training_flat_score_baseline.csv/json：已經過訓練門檻篩選的訓練股票池，同分基準與模型 loss。
各組 parameter_history.csv：每日新標籤更新前後參數；Before_ 與 After_ 為更新時點，不是未來參數。
各組 results.json / audit.json：完整結果與稽核摘要。
原始 v14/v15 Markdown 報告：定義、公式與驗證說明。
manifest.json：來源檔案相對工作區路徑與原檔 SHA-256。

逐日欄位字典
Date：訊號日（daily），更新日（training）；訓練原訊號日看 SignalDate，到期日看 ExitDate。
Model：M1 / M2 / S1 / S2。
StocksRanked：完整預測股票數。
FallbackStocks：輸入不足而使用既有缺值處理的股票數，不代表整檔分數固定為 0。
SelectedFallbackStocks：選入多空各 200 檔的 Fallback 股票數。
SelectedMissingTargets / AllMissingTargets：選股／全股票池缺失真實報酬數。
LongWeightedScore / ShortWeightedScore：各邊真實 Target 按 2 至 1 加權並除以權重平均的和。
OfficialDailySpread：LongWeightedScore - ShortWeightedScore。
IllustrativeGrossOneReturn：OfficialDailySpread / 400；僅示意總曝險 1、多空各 50% 的每日報酬。
RankIC：每天 g 與實際 Target 的 Spearman 相關，取每日等權平均。
NormalizedHardRankMSE：預測 g 與 Target 的降序平均同分名次差，除以 N-1 後平方平均。
MaxAbsoluteScore / MaxAbsolutePrediction：當天 g 的最大絕對值；S1/S2 的分數不是報酬率。
ScoredStocks：排名診斷使用的有限標籤股票數。
AllStockForecastMSE：當天全有限標籤股票 mean((g-Target)^2)，僅 MSE 組適用。
ZeroPredictionMSE：當天以 0 預測所有報酬的 mean(Target^2)。
EligibleMSEContribution / ExcludedMSEContribution：可訓練／排除股票的平方誤差和，均除以全當天有限標籤股票數；兩者相加等於全股票 MSE，不是各子群自己的平均。
ExcludedRows：完整驗證股票池中超過訓練門檻的數量。
SoftRankLoss：mean(((soft_rank - true_rank)/(N-1))^2)；用 tau=1。
FlatScoreSoftRankLoss：同一天所有股票 g 相同時的 soft loss，沒有交易訊號。
ScoreRange：當天最大 g 減最小 g。

逐日訓練欄位
KnownLabelStocks：當日到期的有限標籤股票數。
TrainingStocks：套用財報成分 abs > 100 排除門檻後股票數 N。
ThresholdSkippedStocks：本日超門檻未訓練股票數，仍保留預測；不是永久刪除股票。
MissingTargets：缺失到期 Target 的股票數。
LossBeforeSGD / LossAfterSGD / LossAfterFull：當天整個訓練股票池的平均 loss，分別於更新前、逐檔後、整體後計算；MSE 與 Soft loss 不能跨類直接比较。
SGDAttempts / FullAttempts：逐檔／整體更新嘗試數，每步含線性與折價兩個子步。
SGDAccepted / FullAccepted：至少一個子步接受更新的步數。
SGDNoEta / FullNoEta：無可接受學習率的子步數，計數單位可能與 Accepted 不同，不能直接相加當 Attempts。
LinearAccepted / DiscountAccepted：線性／折價子步接受數。
CandidateEvaluations：學習率候選評估數。

參數與財報定義
alpha 與 11 價量係數；每個財報指標 NetSales / OperatingProfit / EPS 另有 BetaActual / BetaQoQ / BetaYoY / BetaRevision / Discount。
PR1、VR1 各除以近 22 日標準差 (ddof=1)，包含訊號日，不扣均值。
財報 G(x,b)=(x-b)/abs(b)。季 Forecast=(全年 Forecast-已知財年累計實績)/(4-已知季度數)。
每個指標 5 個事件成分 u,v,Q,Y,U；財報貢獻=BetaActual*(u-d*v)+d*(BetaQoQ*Q+BetaYoY*Y+BetaRevision*U)。
目前 EPS 也以成長率建模；不是較早版本的 EPS 原值。
Forecast 修正使用相同已知累計實績與季度數、相同去年同期單季實績基期，再取新舊成長率差。
沒有新 Forecast 不新增該次事件；舊事件持續按 exp(-交易日齡/9) 衰減。非零事件累加。
會計基礎不明的 ForecastRevision / NumericalCorrection 沒有重新加入；此資料與 v14/v15 一致。
所有標籤須到期才可训练，使用原訊號日凍結特徵、門檻及種子。沒有年度參數重設。

本包僅整理已完成本地實驗；沒有新的持出測試集或不衰減訓練結果。
