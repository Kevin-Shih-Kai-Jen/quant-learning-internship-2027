import json
import pandas as pd
from financial_features import ROOT

def main():
    r=json.loads((ROOT/'results.json').read_text());m=r['model'];a=json.loads((ROOT/'audit.json').read_text());f=json.loads((ROOT/'feature_summary.json').read_text())
    assert a['passed']
    labels={'v9_sgd':'v9 SGD＋財報衰減特徵','v8_sgd':'原 v8 SGD','v7_equal':'v7 等權（背景參照）'}
    new=r['comparison']['v9_sgd']['official_unannualized_sharpe'];old=r['comparison']['v8_sgd']['official_unannualized_sharpe']
    u=pd.read_csv(ROOT/'sgd/training_updates.csv');hist=pd.read_csv(ROOT/'sgd/parameter_history.csv')
    lines=['# v9：SGD 加入財報衰減特徵','',
        '2026-09-12。新增一個「公告後逐日減弱的營業利益率同比變化」，與原十一個特徵一起訓練。保留 v8 SGD 的名次平方 loss、學習率規則、股票池與資料時間順序。','',
        f'相同 **{r["common_scorable_days"]} 個可評分驗證日**，結果如下。JPX Sharpe 未年化，越高越好。','',
        '| 模型 | JPX Sharpe |','|---|---:|']
    for name,value in r['comparison'].items():lines.append(f'| {labels[name]} | {value["official_unannualized_sharpe"]:.8f} |')
    lines+=['',f'加入財報特徵後，相對原 SGD 的 Sharpe 變化為 **{new-old:+.8f}**。'+('本輪歷史驗證有改善。' if new>old else '本輪歷史驗證未改善。'),'',
        '| 驗證年度 | 共同日數 | 新 SGD | 原 SGD |','|---|---:|---:|---:|']
    for year in r['annual']:lines.append(f'| {year["validation_year"]} | {year["days"]} | {year["v9_sgd"]:.6f} | {year["v8_sgd"]:.6f} |')
    lines+=['','## 特徵與模型','',
        '```text\nm_current = 本期 OperatingProfit / 本期 NetSales\nm_prior = 去年同期 OperatingProfit / 去年同期 NetSales\nΔm = m_current − m_prior\nz = r × Δm\ng = 原十一個特徵的線性分數 + β × z\n```','',
        '這裡採用利益率差值：10% → 12% 時 Δm=0.02，即提高 2 個百分點。β 是由模型學習的新增係數；沒有另乘營收年增率。β 與原係數共同重新學習，原係數不凍結。','',
        '| 財報首次可用後的交易日 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 起 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
        '| r | 1 | 0.9 | 0.8 | 0.7 | 0.6 | 0.5 | 0.4 | 0.3 | 0.2 | 0.1 | 0 |','',
        '10 天是執行前固定的待驗證假設。休市日不推進衰減；新財報取代前一個事件，重新計時。沒有足夠同期資料時，此新增模型輸入為 0，另保留「比較資料不足」狀態。','',
        '## 特徵大多為零，SGD 會怎樣？','',
        '一般逐檔 MSE 中，若 g_i=其他項+βz_i，則 ∂L_i/∂β=2(g_i−y_i)z_i。因此 z_i=0 時，該筆對 β 的梯度為 0；β 保留原值，不會被設成 0。','',
        '本次使用可微分排名的平方誤差，股票 i 的名次參考其他股票，公式為：','',
        '```text\np_ij = sigmoid((g_j − g_i) / τ)，τ=1\nr_soft_i = 1 + Σ(j≠i) p_ij\nL_i = ((r_soft_i − r_true_i)/(N−1))²\n∂L_i/∂β = 2(r_soft_i−r_true_i)/(N−1)²\n             × Σ(j≠i) p_ij(1−p_ij)(z_j−z_i)/τ\n```','',
        '**所以，只有這一檔 z_i=0，仍可能更新 β；同日全部股票 z 都是 0 時，β 的梯度必定為 0。** 對當下預測而言，z_i=0 仍表示該股票自身的新增分數 βz_i=0。','',
        f'實際共檢查 {m["batches"]:,} 個 SGD 批次，其中 **{m["own_zero_feature_nonzero_beta_gradient_batches"]:,} 次**出現自身輸入為 0、β 梯度絕對值仍大於 1e−12；自身輸入為 0 而 β 實際改變的批次共 **{m["own_zero_feature_actual_beta_changes"]:,} 次**。兩項數量採不同判定：前者是梯度門檻，後者包含接受步長後的浮點參數變化。','',
        f'全部輸入為 0 的日期已驗證 β 保持原值。新增特徵關閉時，新運算核心能逐步重現原 v8 的更新；實際資料在第一次財報特徵進入訓練前，也重現了 {a["initial_baseline_state_reproduced_dates"]} 個日期的原 SGD 參數。','',
        '## 資料與時間處理','',
        '使用原始 financials.csv，先依公告時間處理。比對同公司、同會計口徑、同季度類型，以及相差一年的會計年度開始日、結束日與報表期間結束日。使用累計財報對去年同期累計財報；要求兩期營收為正、營業利益有限，虧損保留。','',
        '本輪採歷史收盤訊號的保守時間界線：日本時間 15:00 前公布者可從當天使用，15:00 及之後公布者從下一交易日使用；同時不得早於資料列的 Date。真正公告日期晚於 Date 時，以較晚日期為準。已核對原始 Unix 時戳與日本時間欄位一致。','',
        '純 ForecastRevision 不重新啟動實際營業利益率特徵。NumericalCorrection 只有能從此前正常財報唯一辨認會計口徑，且包含兩個有效數值欄位時才套用；若改變當前利益率或其去年同期比較值，就從更正可用日重算並重新計時。舊期、不影響當前比較的更正不重啟事件，更正前預測保持原值。','',
        f'在原股票池中，正常財報 {f["regular_statement_rows"]:,} 筆、數字更正 {f["correction_rows"]:,} 筆。無法唯一配對口徑的更正 {f["correction_unmatched_or_ambiguous_basis"]} 筆，缺少兩個數值欄位的更正 {f["correction_without_both_numeric_components"]} 筆；這些更正未套用於本次特徵。','',
        f'準備資料共 {f["signal_rows_in_cache"]:,} 個股票日期，非零新特徵 {f["nonzero_feature_rows"]:,} 筆，占 {100*f["nonzero_feature_fraction"]:.2f}%；驗證期 {f["validation_rows"]:,} 個股票日期中，非零 {f["validation_nonzero_feature_rows"]:,} 筆，占 {100*f["validation_nonzero_feature_rows"]/f["validation_rows"]:.2f}%。準備資料包含最末兩個僅供資料對齊的日期，實際預測列數為 {a["forecast_rows_reproduced"]:,}。','',
        '每個 SignalDate 的 z 與 r 都先固定保存。等該日 Target 實現後，SGD 使用當時的 z，不改用訓練當日已繼續衰減的值。參數在 2017 年只初始化一次，之後逐日接續；不重播舊資料、不跨年度重設。','',
        '## 數值尺度與學習紀錄','',
        f'原始特徵最小 {f["feature_min"]:.6f}、最大 {f["feature_max"]:.6f}，皆為比例值。沒有截尾或標準化。例如股票 4883 的一筆營收為 1,000,000、營業利益為 −677,000,000，利益率因此為 −677；這是小分母造成的極端比例。同期差值也會很大，不能當成一般幾個百分點的變化。','',
        f'新增 β 的梯度平方占整體梯度平方和至少 90% 的批次，共 {m["financial_gradient_dominates_batches"]:,} 次。這是尺度診斷，不能單憑它判定績效差異的原因。','',
        '| 項目 | 數量 |','|---|---:|',
        f'| 標籤日期 | {m["label_days"]:,} |',f'| SGD 批次 | {m["batches"]:,} |',f'| 接受更新 | {m["accepted_updates"]:,} |',
        f'| 小梯度跳過 | {m["small_gradient_skips"]:,} |',f'| 找不到合格步長而跳過 | {m["no_acceptable_eta_skips"]:,} |',
        f'| 選到步長上限 1.0 | {m["hit_upper_eta_batches"]:,} |',f'| 當天整體 soft loss 上升的日期 | {m["full_day_loss_increased_days"]:,} |','',
        f'最後 β={m["final_coefficients"]["FinancialMarginYoYDecay"]:.10f}，對應 2021-12-03 已處理完到期標籤後的狀態；最後預測日是 2021-12-01。這個 β 的單位是模型排名分數，不能直接解讀為預測報酬。','',
        '每批依原 v8 規則只算一次梯度，再從相同舊參數試步長：0.1、0.2……1.0；若 0.1 不合格，依序向下試到 0.000001。必須嚴格降低當批 loss 且符合 Armijo 下降條件；最後只提交一個候選。這不保證整天 loss 或未來預測改善。','',
        '## 核對與限制','',
        f'- 獨立重建全部 {a["feature_audit"]["all_signal_rows_independently_reproduced"]:,} 個財報特徵，核對公告時間、來源兩期資料及第十／十一日邊界。\n- 檢查全部 SGD 批次的接受條件、β 接續及使用的特徵確實來自 SignalDate。\n- 抽取真實資料獨立重算 {a["independent_real_batch_replays"]} 個批次的解析梯度與學習率試探，另做 {a["finite_difference_real_batches"]} 個真實批次的有限差分檢查；包含極端財報值和自身特徵為零的情況。\n- 重建全部 {a["forecast_dates_reproduced"]} 日、{a["forecast_rows_reproduced"]:,} 筆預測，整數名次一致；官方日 spread 最大重算差異 {a["max_official_spread_error"]:.3g}。','',
        '沿用相同歷史資料與多空各 200 檔、2→1 排名權重，未扣交易成本。這是已多次使用的歷史驗證，沒有新的保留測試集；單次結果不證明十天衰減成立。稀疏值不妨礙梯度計算，但訊號強弱、有效公告數與極端尺度仍影響學習。','',
        '各筆預測、每日係數、逐批梯度與步長、財報來源事件、更正處理及極端值均保存在本資料夾。`results.json` 是總結，`daily_comparison.csv` 是逐日比較，`audit.json` 是核對結果，`financial_events.csv` 可追溯財報來源。']
    (ROOT/'JPX-v9-financial-decay-SGD-report.md').write_text('\n'.join(lines)+'\n')
    pd.Series(m['final_coefficients'],name='Coefficient').rename_axis('Parameter').to_csv(ROOT/'final_coefficients.csv')
    print('REPORT_WRITTEN',flush=True)

if __name__=='__main__':main()
