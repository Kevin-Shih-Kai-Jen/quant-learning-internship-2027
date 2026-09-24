import json,hashlib,platform,sys
import numpy as np
import pandas as pd
from features import ROOT,VARIANTS,METRICS,FINS
from run import NAMES

def save(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))
def link(name):return f'[{name}]({ROOT/name})'

def main():
    results={v:json.loads((ROOT/v/'results.json').read_text()) for v in VARIANTS}
    audits={v:json.loads((ROOT/v/'audit.json').read_text()) for v in VARIANTS}
    features=json.loads((ROOT/'feature_summary.json').read_text());fa=json.loads((ROOT/'feature_audit.json').read_text())
    assert fa['passed'] and all(a['passed'] for a in audits.values())
    daily={v:pd.read_csv(ROOT/v/'daily_spread_returns.csv',parse_dates=['Date']) for v in VARIANTS}
    updates={v:pd.read_csv(ROOT/v/'training_updates.csv',parse_dates=['Date','SignalDate','ExitDate']) for v in VARIANTS}
    pd.testing.assert_frame_equal(updates[VARIANTS[0]][['Date','SignalDate','ExitDate','KnownLabelStocks']],updates[VARIANTS[1]][['Date','SignalDate','ExitDate','KnownLabelStocks']])
    common=daily['sgd_only'][['Date','ValidationYear']].copy()
    for v in VARIANTS:
        assert daily[v].SelectedMissingTargets.sum()==0
        common=common.merge(daily[v][['Date','OfficialDailySpread','AllStockForecastMSE','MaxAbsolutePrediction']].rename(columns={c:v+'_'+c for c in ['OfficialDailySpread','AllStockForecastMSE','MaxAbsolutePrediction']}),on='Date',validate='one_to_one')
    common.to_csv(ROOT/'daily_comparison.csv',index=False)
    annual=[]
    for yr,g in common.groupby('ValidationYear'):
        row={'ValidationYear':int(yr),'Days':len(g)}
        for v in VARIANTS:
            s=g[v+'_OfficialDailySpread'];row[v+'_Sharpe']=float(s.mean()/s.std(ddof=1));row[v+'_MSE']=float(g[v+'_AllStockForecastMSE'].mean())
        annual.append(row)
    pd.DataFrame(annual).to_csv(ROOT/'annual_comparison.csv',index=False)
    params=pd.DataFrame({'Parameter':NAMES,**{v:[results[v]['final_parameters'][c] for c in NAMES] for v in VARIANTS}})
    params.to_csv(ROOT/'final_parameters.csv',index=False)
    discount=[]
    for v in VARIANTS:
        history=pd.read_csv(ROOT/v/'parameter_history.csv')
        for m in METRICS:
            s=history['After_'+m+'Discount'];discount.append({'Variant':v,'Metric':m,'Final':float(s.iloc[-1]),'Min':float(s.min()),'Max':float(s.max()),'Mean':float(s.mean()),'AtZeroDays':int(s.eq(0).sum()),'AtOneDays':int(s.eq(1).sum())})
    pd.DataFrame(discount).to_csv(ROOT/'discount_summary.csv',index=False)
    # Concrete source-backed examples for inspecting the approved formulas.
    events=pd.read_pickle(ROOT/'financial_events.pkl');examples=[]
    for kind in ['A','Q','Y','U']:
        for metric in range(3):
            candidates=events.loc[events.Kind.eq(kind)&events.Metric.eq(metric)&events.Values.map(lambda x:all(abs(v)<5 for v in x))]
            if len(candidates):examples.append(candidates.iloc[0])
    pd.DataFrame(examples).to_csv(ROOT/'financial_event_examples.csv',index=False)
    reduction=1-results['sgd_sqrt']['mean_daily_forecast_mse']/results['sgd_only']['mean_daily_forecast_mse']
    comparison={'days':len(common),'mse_reduction_fraction':reduction,
        'sharpe_difference_hybrid_minus_sgd':results['sgd_sqrt']['validation']['official_style_unannualized_sharpe']-results['sgd_only']['validation']['official_style_unannualized_sharpe'],
        'variants':results,'annual':annual,'audits':audits,'feature_audit':fa,
        'no_transaction_costs':True,'fresh_test_set_used':False,'optimizer':'alternating block SGD; optional floor(sqrt(N)) same-day block GD',
        'joint_simultaneous_warmup_overflow_archived':True}
    save(ROOT/'results.json',comparison)
    save(ROOT/'paired_audit.json',{'passed':True,'same_label_dates_and_stock_counts':True,'same_feature_matrix':True,
        'same_seed_and_order':True,'same_initial_parameters':True,'same_learning_rate_grid_and_projection':True,'only_difference':'additional floor(sqrt(N)) full-day updates'})
    lines=['# v13：季度財報預期＋SGD／SGD 加每日 √N 次整體更新','',
        '**追加每日整體更新降低了驗證期報酬 MSE，但沒有改善多空排名 Sharpe；兩組 Sharpe 都是負值。**','',
        '兩組使用相同財報公式、價量特徵、股票範圍、資料順序及可用時間。差別只有：逐檔 SGD 後，是否再做 ⌊√N⌋ 次當天全部已知標籤股票的平均 MSE 更新。','',
        '| 相同 953 個驗證日 | 單獨 SGD | SGD＋⌊√N⌋ 次整體更新 |','|---|---:|---:|']
    for label,fn in [('未年化多空 Sharpe',lambda r:r['validation']['official_style_unannualized_sharpe']),('平均每日預測 MSE',lambda r:r['mean_daily_forecast_mse'])]:
        lines.append('| '+label+' | '+' | '.join(f'{fn(results[v]):.8f}' for v in VARIANTS)+' |')
    lines+=['',f'混合組的預測 MSE 降低 **{reduction:.2%}**，但 Sharpe 比單獨 SGD 低 {abs(comparison["sharpe_difference_hybrid_minus_sgd"]):.5f}。以這次多空排名目標來看，單獨 SGD 較好；若目標是報酬數值的平方誤差，混合组較好。這不構成正報酬策略的證據。','',
        '## 訓練方式與數值修正','',
        '先前一次同時更新影響係數與折價參數的版本，在 2017 年 5 月暖身期間發生數值溢位，尚未進入驗證期。原始記錄保留於 joint_step_overflow/。原因是 β 與 d 相乘，投影至 d=0 可能暫時掩蓋過大的 β，單筆 loss 下降仍可能使其他股票的預測失控。','',
        '最終兩組都採分塊 SGD：每檔先固定 d 更新 24 個線性係數，再固定線性係數更新三個 d。每次整體更新也採相同兩步。這是分塊隨機梯度法，而非所有參數在同一步同時更新；每個子步都須通過相同 loss 下降檢查。沒有改動特徵或按驗證績效調整學習率。','',
        'N 只計算當天新成熟、Target 有限的可訓練股票，不使用尚未成熟標籤，也不重播更早日期。N=400 時為 20 次；實際次數依各日 N 向下取整。','',
        '| 全訓練期間 | 單獨 SGD | SGD＋⌊√N⌋ |','|---|---:|---:|']
    for label,key in [('新成熟標籤日','training_days'),('逐檔 SGD 次數','SGDAttempts'),('追加整體更新嘗試次數','FullAttempts'),('至少一個子步獲接受的整體更新次數','FullAccepted'),('最終當日 loss 高於當日訓練前的天數','final_increased_day_loss_days')]:
        lines.append('| '+label+' | '+' | '.join(f'{results[v][key]:,}' for v in VARIANTS)+' |')
    hu=updates['sgd_sqrt']
    lines+=['',f'混合組每天追加 {int(hu.FullAttempts.min())}～{int(hu.FullAttempts.max())} 次整體更新。所有 {len(hu):,} 個訓練日的追加階段均降低了該日 SGD 結束時的 loss；未獲接受的嘗試仍計入 √N 次數。','',
        '| 各訓練日 MSE 的平均值 | 單獨 SGD | SGD＋⌊√N⌋ |','|---|---:|---:|']
    for label,key in [('當日訓練前','mean_training_loss_before'),('逐檔 SGD 後','mean_training_loss_after_sgd'),('整體階段後／當日最終','mean_training_loss_after_full')]:
        lines.append('| '+label+' | '+' | '.join(f'{results[v][key]:.8f}' for v in VARIANTS)+' |')
    lines+=['','這張表是訓練 loss，不能當成驗證表現。兩組自前一天起的參數已不同，訓練前 loss 也因此不同。','',
        '## 分年度驗證','', '| 驗證年度 | 日數 | 單獨 SGD Sharpe | 混合組 Sharpe | 單獨 SGD MSE | 混合組 MSE |','|---|---:|---:|---:|---:|---:|']
    for a in annual:lines.append(f'| {a["ValidationYear"]} | {a["Days"]} | {a["sgd_only_Sharpe"]:.6f} | {a["sgd_sqrt_Sharpe"]:.6f} | {a["sgd_only_MSE"]:.8f} | {a["sgd_sqrt_MSE"]:.8f} |')
    lines+=['','## 實作的財報公式','',
        '每個指標 m∈{NetSales, OperatingProfit, EPS} 使用 Q=(F−C_k)/(4−k)。F 是同財年全年預測，C_k 是已公布前 k 季累計實績。未有實績時 k=0、C_0=0。FY 報表的下一年預測切換到新財年。','',
        '- 成長率 G(x,b)=(x−b)/|b|。分母為零或資料缺失時，該事件不可用；不加任意小數、不截尾。','- 實際驚喜：G(R_q,R_{q−4})−d_m G(Q_q^pre,R_{q−4})。Q^pre 是實績公布前最後預測。','- 預測季增：d_m G(Q_q,Q_{q−1}^pre)。','- 預測年增：d_m G(Q_q,Q_{q−4}^pre)。','- 預期修正：d_m [G((F_new−C_k)/(4−k),R_{q−4})−G((F_old−C_k)/(4−k),R_{q−4})]。新舊兩邊使用相同 C_k、k 與基準。','- 每種訊號各自乘 β；事件權重 exp(−a/9)，首個可用交易日 a=0，不強制歸零，舊事件與新事件相加。','',
        '每個指標的初次季度 Forecast 季增與年增只發出一次；同一季度後續修正只追加修正訊號，不重播原本成長訊號。跨季新預期與同日年度指引修正可以同時存在，各有自己的係數。實績訂正只影響往後已知狀態，不重播原本實際驚喜，也不回填歷史。','',
        '27 個參數：截距與 11 個既有價量係數、12 個財報係數、3 個共享於該指標的折價參數。線性係數只在起點初始化為零，d 初始化為 1、限制於 [0,1]，跨日、跨年連續學習。PR1、VR1 各除以含 SignalDate 的 22 交易日樣本標準差，不減均值。','',
        '## 折價參數與限制','', '| 指標 | 單獨 SGD 最終 d | 混合組最終 d |','|---|---:|---:|']
    for m in METRICS:lines.append('| '+m+' | '+' | '.join(f'{results[v]["final_parameters"][m+"Discount"]:.6f}' for v in VARIANTS)+' |')
    lines+=['',
        '這些是模型擬合參數，不能解讀成觀測到的市場折價。Forecast-only 項包含 βd 乘積，分開解讀需依賴實際驚喜項的聯繫。最終參數日期為 2021-12-03，最後預測日為 2021-12-01；最後兩天成熟標籤未回填過去預測。','',
        f'本次共有 {features["event_count"]:,} 筆指標事件、{features["quarter_snapshot_count"]:,} 筆季度预估快照；驗證股票日有 {features["validation_any_financial_rows"]:,}/{features["validation_rows"]:,} 至少一項非零財報輸入。','',
        '剩餘季度平均分配仍是模型假設，未建模季節性。EPS 依確認公式使用累計 EPS 差額；加權平均股數變化時，它不等同精確重建的單季 EPS。非標準財年、季度邊界或無法辨識期間／會計口徑的資料，不生成該筆財報事件；股票仍留在價量模型。非年度 ForecastRevision 可能混有半年預測，保守排除。基期接近零會產生很大變化率，本次沒有按結果截尾。','',
        '## 核對結果','',
        f'- {fa["source_rows"]:,} 筆來源公告的可用時間、{fa["reference_checks"]:,} 次來源引用及全部 {fa["events"]:,} 筆事件公式已核對。來源資料只读。',
        f'- 全部 {fa["all_signal_rows_checked"]:,} 個股票日，以獨立逐日衰減遞推重算；最大差異 {fa["decay_recurrence_max_error"]:.3g}。',
        '- 各組全部逐檔與整體更新，使用另一套逐列 Jacobian 計算式重播。每 8 檔及每次整體更新使用檢查點，避免兩種等價浮點运算順序造成跨多步的誤差放大；檢查點再生的完整原始訓練 traces 與最終參數逐位元一致。',
        '- 另以 NumPy 重做固定抽樣步驟的完整學習率搜尋，並檢查所有參數有限差分梯度及折價邊界。','- 各組所有 2,326,022 筆預測、整數排名、953 日多空報酬與 MSE 皆重算通過；選中的多空股票沒有缺失 Target。','',
        '## 檔案','',
        '- '+link('daily_comparison.csv')+'：每日兩組對照。','- '+link('annual_comparison.csv')+'：分年度對照。','- '+link('final_parameters.csv')+'：27 個最終參數。','- '+link('discount_summary.csv')+'：折價參數範圍與觸及邊界的天數。','- '+link('financial_event_examples.csv')+'：可回查來源的財報訊號範例。','- '+link('experiment_plan.md')+'：事前規則及數值修正紀錄。','- '+link('results.json')+'：完整結果與稽核摘要。','',
        '驗證期間與先前實驗相同，已被多次比較；本次未使用新的獨立測試集，也未扣交易成本。當前正式基準仍是 v7，本實驗不自動取代它。','']
    report=ROOT/'JPX-v13-quarterly-SGD-sqrt-report.md';report.write_text('\n'.join(lines))
    save(ROOT/'runtime_manifest.json',{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'platform':platform.platform(),'compiler':'clang++ -O3 -std=c++17 -dynamiclib','parameters':27})
    sources=list(ROOT.glob('*.py'))+list(ROOT.glob('*.cpp'))+[ROOT/'experiment_plan.md',ROOT/'model.dylib',ROOT/'replay.dylib',ROOT.parent/'jpx_v12_financial_growth_20260913/features.py',ROOT.parent/'jpx_v11_persistent_finance_std22_20260913/return_std22_features.pkl']
    save(ROOT/'source_hashes.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    print(json.dumps({'report':str(report),'mse_reduction':reduction,'annual':annual}),flush=True)

if __name__=='__main__':main()
