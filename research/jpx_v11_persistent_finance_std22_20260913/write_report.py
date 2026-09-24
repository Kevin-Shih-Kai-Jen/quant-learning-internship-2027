import json,hashlib,sys,platform,os
import numpy as np
import pandas as pd
from features import ROOT,V8,V9,V10,VARIANTS
from run_v11 import NAMES,save

def table(head,rows):
    return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])

def main():
    results={v:json.loads((ROOT/v/'results.json').read_text()) for v in VARIANTS}
    audits={v:json.loads((ROOT/v/'audit.json').read_text()) for v in VARIANTS}
    assert all(a['passed'] for a in audits.values())
    fa=json.loads((ROOT/'feature_audit.json').read_text());fs=json.loads((ROOT/'feature_summary.json').read_text());assert fa['passed']
    dirs={'v10_baseline':V10/'mse_sgd_financial',**{v:ROOT/v for v in VARIANTS}}
    dailies={v:pd.read_csv(p/'daily_spread_returns.csv',parse_dates=['Date'],float_precision='round_trip') for v,p in dirs.items()}
    calendar=dailies['v10_baseline'][['Date','ValidationYear']];common=pd.Index(calendar.Date)
    for d in dailies.values():common=common.intersection(d.loc[d.SelectedMissingTargets.eq(0),'Date'])
    comparison={};annual=[]
    for v,d in dailies.items():
        d=d.loc[d.Date.isin(common)];s=d.OfficialDailySpread
        comparison[v]={'common_days':len(s),'official_unannualized_sharpe':float(s.mean()/s.std(ddof=1)),
            'mean_daily_forecast_mse':float(d.AllStockForecastMSE.mean()),'mean_daily_spread':float(s.mean()),
            'sample_std_daily_spread':float(s.std(ddof=1)),'median_daily_forecast_mse':float(d.AllStockForecastMSE.median())}
    for year,g in calendar.groupby('ValidationYear'):
        dates=common.intersection(g.Date);record={'validation_year':int(year),'days':len(dates)}
        for v,d in dailies.items():
            d=d.loc[d.Date.isin(dates)];s=d.OfficialDailySpread
            record[v]=float(s.mean()/s.std(ddof=1));record[v+'_forecast_mse']=float(d.AllStockForecastMSE.mean())
        annual.append(record)
    combined=calendar.copy()
    for v,d in dailies.items():
        cols=['OfficialDailySpread','AllStockForecastMSE','SelectedMissingTargets']
        combined=combined.merge(d[['Date']+cols].rename(columns={c:v+'_'+c for c in cols}),on='Date',validate='one_to_one')
    combined['CommonScorableDate']=combined.Date.isin(common);combined.to_csv(ROOT/'daily_comparison.csv',index=False)
    contrasts={}
    for key,after,before in [('remove_decay_raw','persistent_raw','v10_baseline'),('std22_with_decay','decay_std22','v10_baseline'),
        ('both_changes','persistent_std22','v10_baseline'),('remove_decay_after_std22','persistent_std22','decay_std22'),
        ('std22_after_remove_decay','persistent_std22','persistent_raw')]:
        contrasts[key]={m:comparison[after][m]-comparison[before][m] for m in ['official_unannualized_sharpe','mean_daily_forecast_mse']}
    # Pair controls coincide before financial features first enter training.
    hist={v:pd.read_csv(p/'parameter_history.csv',parse_dates=['Date'],float_precision='round_trip').set_index('Date') for v,p in dirs.items()}
    h=hist['v10_baseline'].rename(columns={'After_FinancialMarginYoYDecay':'After_FinancialMarginYoYSignal'})
    early=h.index[h.index<pd.Timestamp('2017-11-14')];cols=['After_'+n for n in NAMES]
    np.testing.assert_array_equal(h.loc[early,cols],hist['persistent_raw'].loc[early,cols])
    np.testing.assert_array_equal(hist['decay_std22'].loc[early,cols],hist['persistent_std22'].loc[early,cols])
    pair={'passed':True,'matching_initial_dates':len(early),'raw_pair':'v10_baseline vs persistent_raw','scaled_pair':'decay_std22 vs persistent_std22'}
    save(ROOT/'paired_audit.json',pair)
    summary={'version':'v11','main_variant':'persistent_std22','comparison':comparison,'annual':annual,'contrasts':contrasts,
        'models':results,'validation_days':len(calendar),'common_scorable_days':len(common),
        'features':fs,'loss':'ordinary return MSE','batch_size':1,'same_v10_optimizer':True,'soft_rank_used':False,
        'return_standardization':'PR1 and VR1 each divided by own trailing 22-session sample std including signal day; no demeaning',
        'costs_included':False,'formal_test_used':False,'audit_passed':True}
    save(ROOT/'results.json',summary)
    pd.DataFrame({'Coefficient':NAMES,**{v:[results[v]['final_coefficients'][n] for n in NAMES] for v in VARIANTS}}).to_csv(ROOT/'final_coefficients.csv',index=False)
    save(ROOT/'runtime_manifest.json',{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'platform':platform.platform(),
        'kernel':'byte-identical copy of audited v10 mse.dylib','thread_environment':{k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']}})
    paths=list(ROOT.glob('*.py'))+[ROOT/'mse.cpp',ROOT/'mse.dylib',ROOT/'experiment_plan.md',ROOT/'preflight.json',ROOT/'feature_audit.json',
        ROOT/'financial_persistent_features.pkl',ROOT/'return_std22_features.pkl',V8/'inputs.pkl',V9/'financial_signal_features.pkl',V9/'financial_events.pkl',
        V10/'mse_sgd_financial/daily_spread_returns.csv',ROOT.parent/'jpx_v5_daily_returns_20260912/one_day_returns.pkl']
    save(ROOT/'source_hashes.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    names={'v10_baseline':'原 v10：衰減財報＋原始 return','persistent_raw':'只取消財報衰減',
        'decay_std22':'只標準化 PR1、VR1','persistent_std22':'兩者都改'}
    order=list(dirs)
    rows=[[names[v],f"{comparison[v]['official_unannualized_sharpe']:.8f}",f"{comparison[v]['mean_daily_forecast_mse']:.9f}"] for v in order]
    arows=[[r['validation_year'],r['days']]+[f'{r[v]:.8f}' for v in order] for r in annual]
    mserows=[[r['validation_year']]+[f"{r[v+'_forecast_mse']:.9f}" for v in order] for r in annual]
    crows=[]
    clabels={'remove_decay_raw':'取消衰減：原始 return 下','std22_with_decay':'標準化：有衰減財報下','both_changes':'兩者都改，相較 v10',
        'remove_decay_after_std22':'取消衰減：標準化 return 下','std22_after_remove_decay':'標準化：不衰減財報下'}
    for k,c in contrasts.items():crows.append([clabels[k],f"{c['official_unannualized_sharpe']:+.8f}",f"{c['mean_daily_forecast_mse']:+.9f}"])
    counts=[]
    for key,label in [('batches','逐檔 batch'),('accepted_updates','成功更新'),('small_gradient_skips','梯度過小跳過'),
        ('no_acceptable_eta_skips','沒有合格學習率而跳過'),('full_day_loss_increased_days','更新後整日訓練 MSE 上升的日期'),
        ('own_zero_feature_nonzero_beta_gradient_batches','自身財報 z=0 卻有非零 β 梯度')]:
        counts.append([label]+[f'{results[v][key]:,}' for v in VARIANTS])
    auditrows=[[names[v],f"{audits[v]['audited_batches']:,}",f"{audits[v]['independent_snapshot_batch_replays']:,}",f"{audits[v]['forecast_rows_reproduced']:,}",f"{audits[v]['max_official_spread_error']:.3g}"] for v in VARIANTS]
    best=max(comparison,key=lambda v:comparison[v]['official_unannualized_sharpe'])
    worst_mse={v:d.nlargest(3,'AllStockForecastMSE')[['Date','AllStockForecastMSE']].assign(Date=lambda x:x.Date.dt.strftime('%Y-%m-%d')).to_dict('records') for v,d in dailies.items()}
    save(ROOT/'largest_daily_forecast_mse.json',worst_mse)
    attribution=json.loads((ROOT/'mse_attribution.json').read_text())
    text=f'''# JPX v11：不衰減財報與 PR1／VR1 的 22 日波動縮放

完成日期：2026-09-13。依使用者要求，PR1 和 VR1 都各自除以近 22 個交易日標準差。將「財報取消衰減」和「return 標準化」拆成 2×2 比較。原 v10 作為保存的對照，三個新版本各自從零完整訓練一次。

## 同期結果

{table(['模型','JPX 未年化 Sharpe','平均每日報酬預測 MSE'],rows)}

四組使用相同 {len(common)} 個可評分日（完整驗證期 {len(calendar)} 日）；Sharpe 越高越好，報酬 MSE 越低越好。本轮 Sharpe 最高者是「{names[best]}」。結果僅描述這段歷史驗證，不代表新資料一定相同。

以下直接差值可分辨兩個變動：Sharpe 差值為正表示改善，MSE 差值為負表示改善。

{table(['比較','Sharpe 差值','平均每日 MSE 差值'],crows)}

在本次結果下，較值得保留的是 PR1、VR1 的 22 日縮放，財報先維持原衰減。四組 Sharpe 仍然都是負值；只標準化版在 2019、2021 折也比原 v10 差，改善並非每年一致。PR1、VR1 是同時改動，本輪不能分辨兩者各自貢獻。

兩個改动的效果不必相加，因為所有係數共同訓練且每筆學習率由當筆 MSE 決定。財報持續時間改變會影響 β 及其他係數；return 尺度改變也會改變整個 SGD 軌跡。

## 年度折比較

{table(['驗證折年','日數','原 v10','只取消衰減','只標準化','兩者都改'],arows)}

對應的平均每日報酬 MSE：

{table(['驗證折年','原 v10','只取消衰減','只標準化','兩者都改'],mserows)}

沿用既有驗證折，實際 SignalDate 為 {calendar.Date.min().date()} 至 {calendar.Date.max().date()}；2018 折含 2017-12-29。沒有因結果刪除年份、改變股票池或重跑不同參數。

## 1 日 return 如何標準化

`PR1_scaled(t) = PR1(t) / sd(PR1(t−21), …, PR1(t))`

`VR1_scaled(t) = VR1(t) / sd(VR1(t−21), …, VR1(t))`

PR1 是既有分割調整價格的日報酬；VR1 是既有分割調整成交量的日變動率。兩者各自使用同檔股票、截至當日收盤的 22 個市場交易日，標準差採 ddof=1。不使用未來日，不減平均值、不年化，也沒有截尾。這是波動尺度縮放，不是完整的 `(return−mean)/sd` z-score。

22 日全部須有有限變動率。上市初期、不足 22 筆、視窗有缺值或標準差為 0，就將該輸入視為不可用，沿用零回補與 ReturnFallback 旗標。2020-10-01 休市日不占視窗；股票被 SupervisionFlag 排除預測時，已公開歷史行情仍可計入後續視窗。以完整市場日曆補齊缺日，避免把更久以前的交易日冒充最近 22 天。

樣本中第一個完整縮放日為 2017-02-06。验证期 PR1 縮放不可用 {fs['return_standardization']['PR1']['validation_scaled_missing_rows']:,} 列，VR1 不可用 {fs['return_standardization']['VR1']['validation_scaled_missing_rows']:,} 列；其中 PR1 有 4 列視窗標準差為 0。兩個數量可能重疊，不能相加當作股票總数。

此定義將當日 return 也放進分母；大的當日變動會同時提高分母，因此會調整極端值的尺度。沒有另跑「分母只用前 22 日」版本。全部 PR1 縮放範圍 {fs['return_standardization']['PR1']['scaled_min']:.6f} 至 {fs['return_standardization']['PR1']['scaled_max']:.6f}；VR1 為 {fs['return_standardization']['VR1']['scaled_min']:.6f} 至 {fs['return_standardization']['VR1']['scaled_max']:.6f}。這些是樣本結果，不是人為上下限。

## 財報不衰減如何定義

`z(t) = 最新已知的 [本期營業利益／營收 − 去年同期營業利益／營收]`

從同一個 v9 已稽核的 AvailableDate 開始，權重固定 1，保留到下一個適用財報／訂正事件。沒有 10 日到期，也沒有自行更改資料清理。若新事件已知但无法計算同期差值，z=0；不忽略該新事件而把舊有效財報無限往後填。

保留 v9 公司、季度、期間與會計口徑匹配、負營業利益、營收須為正、公告時間、訂正處理及原始極端值。相比 10 日衰減版，验证期非零財報特徵從 {fs['validation_nonzero_finance_decaying']:,}/{fs['validation_rows']:,}（{fs['validation_nonzero_finance_decaying']/fs['validation_rows']:.2%}）增至 {fs['validation_nonzero_finance_persistent']:,}/{fs['validation_rows']:,}（{fs['validation_nonzero_finance_persistent']/fs['validation_rows']:.2%}）。它现在更像持续的公司状态，而不是公告後短期訊號。

## 財報極端值與 MSE 的具體診斷

在兩種 return 都標準化的條件下，取消財報衰減使平均每日 MSE 增加 {attribution['total_mean_daily_mse_difference']:.9f}。依每個股票的平方誤差差額加總，股票 4883 貢獻這個淨增量的 {attribution['top_stock_share_of_net_difference']:.2%}，前五檔合計 {attribution['top_5_share_of_net_difference']:.2%}。這是不同模型間的事後誤差分解；各係數均已共同重訓，不能把它當成 β 單獨的因果效果。

例如 2021-11-29，4883 的持續財報特徵為 −676.505882，不衰減版的預測報酬 g=17.324479，實際 Target=−0.013661；財報項本身貢獻 +17.370523。保留衰減版的 g=−0.043428。這說明「長期保留極端財報值」在這個模型裡確實伴隨很大的報酬預測誤差；本輪未做截尾或穩健轉換的實驗。

[逐股誤差差額]({ROOT/'mse_difference_by_stock.csv'})與[最大誤差差額觀察值]({ROOT/'largest_mse_difference_observations.csv'})已保存。

## 保持不變的訓練規則

`g_i = α + Σ θ_j x_ij + β z_i`；每筆 loss 為 `(g_i−Target_i)²`，batch=1，梯度 `2(g_i−Target_i)x_i`。所有 13 個係數共同從 2017-01-04 的零初值出發，Target 成熟後逐檔各更新一次，參數跨日跨年延續，沒有歷史重訪。財報和縮放 return 在 SignalDate 凍結，成熟日更新用原來的值。

SGD 程式與 v10 二進位完全相同，沿用亂序種子 `20260912+YYYYMMDD`。學習率先試 0.1，成功則往 0.2…1.0 試至首個未改善；0.1 失敗則往 0.09…0.01、0.009…0.001，直到 0.000009…0.000001 找首個合格者。候選須嚴格降低該筆實際 MSE 且滿足 Armijo c=1e−4。梯度無窮範數≤1e−10 或全部候選失敗則跳過。沒有新增參數正則化、梯度裁切或學習率調整。

{table(['項目','只取消衰減','只標準化','兩者都改'],counts)}

單筆 SGD 降低當筆 loss，並不保證更新完後整日 MSE 降低。MSE 評分先在每個預測日對已知 Target 股票平均，再跨日期平均；它與成熟日更新前／後 MSE 分開保存。官方評分仍採唯一整數名次、多空各 200 檔、2→1 權重。

## 查核與輸出

財報全部 {fa['audited_financial_rows']:,} 列以獨立 as-of 合併重建；兩個 return 共 {fa['audited_return_feature_cells']:,} 個輸入格核對。原始日變動率由調整後價格與成交量重算吻合；22 日標準差用不同數值計算方式逐窗檢查，最大差 {fa['max_std22_error']:.3g}，另有 {fa['scalar_past_window_checks']:,} 個單一歷史窗核對。

{table(['模型','全量 batch 紀錄查核','獨立梯度及學習率完整重算樣本','重建預測列','官方 spread 最大差'],auditrows)}

每個版本全部 1,199 個預測日的分數、股票順序、唯一整數名次與財報輸入均重建；953 日官方風格 spread 與預測 MSE 重算。所有 batch 的 β 更新鏈、MSE 梯度範數恆等式、使用當時凍結輸入與 Target 到期時序均檢查。完整梯度／學習率獨立重演是表列保存樣本，並非全量 batch 的逐步獨立重演。每版亦完成 5 個真實 batch 有限差分檢查。

同樣 return 定義的兩種財報模型，在第一筆財報可訓練前的 {len(early)} 個日期係數完全相同。兩個改動沒有影響其他九個價量特徵、Target 或 ExitDate。最後預測日 2021-12-01，最後參數更新至 2021-12-03；使用逐日參數產生當日預測，不以最後參數回填過去。

- [完整比較數值]({ROOT/'results.json'})、[逐日比較]({ROOT/'daily_comparison.csv'})、[最終係數]({ROOT/'final_coefficients.csv'})
- [財報與標準化特徵查核]({ROOT/'feature_audit.json'})、[三組訓練前計畫]({ROOT/'experiment_plan.md'})
- [兩者都改的查核]({ROOT/'persistent_std22/audit.json'})、[每日更新]({ROOT/'persistent_std22/training_updates.csv'})、[逐筆紀錄]({ROOT/'persistent_std22/batch_trace.csv.gz'})
- [最大預測 MSE 日期]({ROOT/'largest_daily_forecast_mse.json'})、[來源雜湊]({ROOT/'source_hashes.json'})

本輪是相同歷史期上的特徵比較，未扣交易成本、沒有新獨立 test。結果不能證明某個特徵普遍有效；没有依這次結果更改視窗、財報定義或學習率。
'''
    for a,b in [('本轮','本輪'),('改动','改動'),('验证','驗證'),('總数','總數'),('无法','無法'),('现在','現在'),('持续','持續'),('状态','狀態'),('没有','沒有')]:text=text.replace(a,b)
    report=ROOT/'JPX-v11-persistent-finance-std22-report.md';report.write_text(text)
    print(json.dumps({'comparison':comparison,'contrasts':contrasts,'annual':annual,'report':str(report)},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
