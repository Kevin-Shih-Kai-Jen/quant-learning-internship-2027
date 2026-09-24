import json,hashlib,sys,platform
import numpy as np
import pandas as pd
from features import ROOT,V8,V11,FINS,VARIANTS
from run_v12 import NAMES,save

def table(head,rows):return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])

def main():
    models={v:json.loads((ROOT/v/'results.json').read_text()) for v in VARIANTS}
    audits={v:json.loads((ROOT/v/'audit.json').read_text()) for v in VARIANTS}
    assert all(a['passed'] for a in audits.values())
    periodfix=json.loads((ROOT/'period_fix_audit.json').read_text());assert periodfix['passed']
    feature=json.loads((ROOT/'feature_summary.json').read_text());fa=json.loads((ROOT/'feature_audit.json').read_text());diagnostic=json.loads((ROOT/'diagnostics.json').read_text())
    dirs={'price_only':ROOT/'price_only','six_financial':ROOT/'six_financial','v11_decay_std22':V11/'decay_std22','v7_equal':ROOT.parent/'jpx_v7_daily_mse_20260912/v7_equal'}
    dailies={v:pd.read_csv(p/'daily_spread_returns.csv',parse_dates=['Date'],float_precision='round_trip') for v,p in dirs.items()}
    calendar=dailies['price_only'][['Date','ValidationYear']];common=pd.Index(calendar.Date)
    for d in dailies.values():common=common.intersection(d.loc[d.SelectedMissingTargets.eq(0),'Date'])
    comparison={};annual=[]
    for v,d in dailies.items():
        d=d.loc[d.Date.isin(common)];s=d.OfficialDailySpread
        comparison[v]={'common_days':len(s),'official_unannualized_sharpe':float(s.mean()/s.std(ddof=1)),
            'mean_daily_forecast_mse':float(d.AllStockForecastMSE.mean()),'mean_daily_spread':float(s.mean()),'sample_std_daily_spread':float(s.std(ddof=1))}
    for year,g in calendar.groupby('ValidationYear'):
        days=common.intersection(g.Date);rec={'validation_year':int(year),'days':len(days)}
        for v,d in dailies.items():
            z=d.loc[d.Date.isin(days)];s=z.OfficialDailySpread;rec[v]=float(s.mean()/s.std(ddof=1));rec[v+'_forecast_mse']=float(z.AllStockForecastMSE.mean())
        annual.append(rec)
    daily=calendar.copy()
    for v,d in dailies.items():
        cols=['OfficialDailySpread','AllStockForecastMSE','SelectedMissingTargets']
        daily=daily.merge(d[['Date']+cols].rename(columns={c:v+'_'+c for c in cols}),on='Date',validate='one_to_one')
    daily['CommonScorableDate']=daily.Date.isin(common);daily.to_csv(ROOT/'daily_comparison.csv',index=False)
    # Adding five unused dimensions beyond the old D13 kernel must not
    # change price-only states before old v11 finance first becomes available.
    p=pd.read_csv(ROOT/'price_only/parameter_history.csv',parse_dates=['Date'],float_precision='round_trip').set_index('Date')
    old=pd.read_csv(V11/'decay_std22/parameter_history.csv',parse_dates=['Date'],float_precision='round_trip').set_index('Date')
    early=p.index[p.index<pd.Timestamp('2017-11-14')];cols=['After_'+n for n in NAMES[:12]]
    np.testing.assert_array_equal(p.loc[early,cols],old.loc[early,cols])
    save(ROOT/'paired_audit.json',{'passed':True,'price_only_matches_legacy_initial_dates':len(early)})
    summary={'version':'v12','main_variant':'six_financial','control_variant':'price_only','features':NAMES[1:],'coefficient_count':18,
        'comparison':comparison,'annual':annual,'models':models,'common_scorable_days':len(common),'validation_days':len(calendar),
        'actual_growth_base':'previous-year exact same reporting period and accounting basis','forecast_growth_base':'previous corresponding full FY actual',
        'growth_formula':'(current-base)/abs(base); zero or missing base -> missing indicator and zero input','eps_raw':True,'financial_decay':False,
        'period_fix':periodfix,'price_returns_std22':True,'loss':'ordinary return MSE','batch_size':1,'soft_rank_used':False,'audit_passed':True,
        'costs_included':False,'formal_test_used':False,'diagnostics':diagnostic}
    save(ROOT/'results.json',summary)
    coefficients=pd.DataFrame({'Coefficient':NAMES,**{v:[models[v]['final_coefficients'][c] for c in NAMES] for v in VARIANTS}})
    coefficients.to_csv(ROOT/'final_coefficients.csv',index=False)
    save(ROOT/'runtime_manifest.json',{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'platform':platform.platform(),'compiler':'clang++ -O3 -std=c++17 -dynamiclib','kernel_dimensions':18})
    paths=list(ROOT.glob('*.py'))+[ROOT/'mse.cpp',ROOT/'mse.dylib',ROOT/'experiment_plan.md',ROOT/'preflight.json',ROOT/'feature_audit.json',
        ROOT/'period_fix_audit.json',ROOT/'financial_records.pkl',ROOT/'financial_events.pkl',ROOT/'financial_signal_features.pkl',V8/'inputs.pkl',V11/'return_std22_features.pkl',
        ROOT.parent/'jpx_v7_daily_mse_20260912/run_v7.py',ROOT.parent/'jpx_v5_daily_returns_20260912/run_v5.py',
        ROOT.parent/'jpx_official_ranking_20260912/official_metric.py']
    save(ROOT/'source_hashes.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    names={'price_only':'v12 純價量對照（六項財報設為 0）','six_financial':'v12 價量＋六項財報','v11_decay_std22':'v11 價量＋原營益率衰減項','v7_equal':'v7 每日整批 MSE（歷史參照）'}
    rows=[[names[v],f"{d['official_unannualized_sharpe']:.8f}",f"{d['mean_daily_forecast_mse']:.9f}"] for v,d in comparison.items()]
    arows=[[r['validation_year'],r['days'],f"{r['price_only']:.8f}",f"{r['six_financial']:.8f}",f"{r['six_financial_forecast_mse']:.9f}"] for r in annual]
    labels=['實際營收成長','實際營業利益成長','實際 EPS 原值','Forecast 營收成長','Forecast 營業利益成長','Forecast EPS 原值']
    frows=[[labels[k],f"{feature['features'][c]['validation_valid_rows']:,}",f"{feature['features'][c]['validation_valid_rows']/feature['validation_rows']:.2%}",f"{feature['features'][c]['min']:.6g}",f"{feature['features'][c]['max']:.6g}"] for k,c in enumerate(FINS)]
    crows=[[f'β{k+1}',labels[k],f"{models['six_financial']['final_coefficients'][c]:.12f}"] for k,c in enumerate(FINS)]
    counts=[]
    for key,label in [('batches','逐檔 batch'),('accepted_updates','成功更新'),('small_gradient_skips','梯度過小跳過'),('no_acceptable_eta_skips','沒有合格學習率而跳過'),('candidate_evaluations','學習率候選評估'),('full_day_loss_increased_days','整日更新後 MSE 上升日期')]:
        counts.append([label,f"{models['price_only'][key]:,}",f"{models['six_financial'][key]:,}"])
    main=models['six_financial'];a=audits['six_financial'];diag=diagnostic;example=diag['worst_example']
    ratio=comparison['six_financial']['mean_daily_forecast_mse']/comparison['price_only']['mean_daily_forecast_mse']
    delta=comparison['six_financial']['official_unannualized_sharpe']-comparison['price_only']['official_unannualized_sharpe']
    text=f'''# JPX v12：價量模型加入六項實際／Forecast 財報特徵

完成日期：2026-09-13。依使用者確認，實際營收與營業利益成長、實際 EPS 原值、Forecast 營收與營業利益成長、Forecast EPS 原值，六項全部加入原本價量模型，各有可學習係數。

## 完整結果

同一組 **{len(common)} 個驗證日**，新模型 JPX 未年化 Sharpe 為 **{comparison['six_financial']['official_unannualized_sharpe']:.8f}**，純價量對照為 **{comparison['price_only']['official_unannualized_sharpe']:.8f}**，差值 **{delta:+.8f}**。但平均每日報酬預測 MSE 從 **{comparison['price_only']['mean_daily_forecast_mse']:.9f}** 升至 **{comparison['six_financial']['mean_daily_forecast_mse']:.9f}**（約 {ratio:.1f} 倍）。

因此本輪是「排序績效改善，報酬數值誤差明顯變差」，不能只看 Sharpe 就認定整個模型都改善。新模型 g 的大小不宜直接解讀為準確的預期報酬幅度。本文數字已包含財年配對修正，取代先前初步運算。

{table(['模型','JPX 未年化 Sharpe','平均每日報酬預測 MSE'],rows)}

最直接的比較是 v12 的兩組：輸入尺度、SGD、日期、股票池、缺值與排名規則完全相同，只差六個財報輸入是否為 0。v11 與 v7 是已保存的歷史參照，不是重新擬合；v7 也使用不同的每日整批更新及學習率。

{table(['驗證折年','日數','純價量 Sharpe','六財報 Sharpe','六財報預測 MSE'],arows)}

年度折的差異如上，不能只由合併 Sharpe 推論每年都有改善。沿用原折年標籤，實際 SignalDate 為 {calendar.Date.min().date()} 至 {calendar.Date.max().date()}，2018 折包含 2017-12-29。

## 模型寫法與參數

`g = α + Σⱼ θⱼ × 價量特徵ⱼ + β₁ × 實際營收成長 + β₂ × 實際營業利益成長 + β₃ × 實際EPS + β₄ × Forecast營收成長 + β₅ × Forecast營業利益成長 + β₆ × ForecastEPS`

價量項仍為 T5、T22、T60、V5、V22、V60、P5xV5、P22xV22、P60xV60、PR1、VR1。PR1、VR1 沿用 v11，各自除以包含當日的 22 交易日樣本標準差，不減平均值。舊的營業利益率同比差值項被六個新財報項取代。

總共 **18 個參數**：1 個截距、11 個價量係數、6 個財報係數。所有係數共同從零訓練，沒有固定某個財報權重，也不是先固定價量係數再單獨估財報係數。六項財報持續至下一個相關公告更新，不另乘衰減係數。

`實際營收成長 = (本期實際營收 − 去年同期實際營收) / |去年同期實際營收|`

`實際營業利益成長 = (本期實際營業利益 − 去年同期實際營業利益) / |去年同期實際營業利益|`

`Forecast營收成長 = (全年預測營收 − 去年對應完整FY實際營收) / |去年對應完整FY實際營收|`

`Forecast營業利益成長 = (全年預測營業利益 − 去年對應完整FY實際營業利益) / |去年對應完整FY實際營業利益|`

成長率存小數，1.5 代表 +150%。−100→50 得 +150%，−100→−60 得 +40%，−100→−150 得 −50%。EPS 保留來源原值、包括負值，不計算成長率，不除以股價、不標準化、不截尾，也不另做拆股調整。基期是 0 或缺值時，該成長率不可用，零回補並保存有效性旗標；缺一項不會使其他五項一起歸零。

最後六個係數如下，日期為 2021-12-03（包含最後預測的 Target 成熟更新）：

{table(['參數','對應特徵','最後係數'],crows)}

最後截距 α={main['final_coefficients']['alpha']:.12f}。最後參數不是拿來重新生成以前預測；每個 SignalDate 都用當時參數。係數大小有單位差異，不能直接用大小排名特徵重要性。

## 財報期別、Forecast 與公告時序

實際同比匹配同公司、會計口徑、季度與財年起日／期間迄日／財年迄日，各減一年。因此累計 2Q 對去年累計 2Q，不擅自轉為單季差額。

Forecast 的分母使用完整 FY 實際值，不是去年 Forecast，也不是去年同季累計數字。常規季報中的全年 Forecast 配目前財年；常規 FY 決算公告中的 Forecast 配下一財年。因原資料沒有獨立的 next-fiscal-year 起迄欄位，下一財年的起日採本 FY 末日+1 天、末日採本 FY 末日+1 年；要求基期 FY 起迄日完全吻合，無法配對就保留缺值。

期間修正：完整檢查發現初次運算有 74 個狀態事件，在公司縮短財年後仍保留了先前推定較晚的年度截止日。已改成新常規報告的目標財年覆蓋舊推定值；同財年起日的新全年修正也可改變迄日，舊期間訂正不把舊 Forecast 重新升為最新。共影響 {periodfix['feature_rows_changed']:,} 個 SignalDate 股票列，剩餘此類錯配為 0。本文數值全部來自修正後重新運算；舊初步結果只保存於 pre_period_fix，沒有依結果挑選版本。

實例：2753 在 2017-04-03 公告下一財年（2018-03-31 結束）的預測營收 32,000,000,000，分母配同次已知的 FY2017 實際營收 30,564,000,000，成長為 **4.698338%**。這個數字已由原始來源列獨立核對。

一般全年與下一財年 Forecast 的概念可參考 [JPX 官方財報欄位說明](https://jpx.gitbook.io/j-quants-pro/api-reference/statements)。該文件是目前較完整的 API 結構；本研究依使用者提供的舊版 CSV 實際欄位與公告序列實作，沒有把新 API 欄位當作本地資料中存在的欄位，也没有用網路資料補財報數值。

`ForecastRevision` 的期間有可能只是半年預測或半年差異。本輪只採用 TypeOfCurrentPeriod=FY 的全年修正，其他 {feature['dispositions']['nonannual_forecast_revision_excluded']:,} 筆保守排除，避免拿半年值與完整 FY 比較。即使其中某些列可能包含全年數字，也不依其後結果猜用途。

全年 Forecast 修正需在當時找到唯一已知會計口徑；無法判別的 {feature['dispositions']['revision_missing_or_ambiguous_basis']:,} 筆跳過。有提供指定 Forecast 欄位的全年修正共套用 {feature['dispositions']['annual_forecast_revision_applied']:,} 筆。只改股利或没有提供這三個 Forecast 欄位的修正不改模型輸入。

NumericalCorrection 只匹配唯一已知的原期間，按有提供的欄位修正；空白保留原欄，明確非數字標記令該欄不可用。最新實際值、實際同比基期、全年 Forecast 或 FY 實際基期被修正時，只影響訂正公布之後的特徵，絕不回填之前日期。常規完整新報表則將缺欄視為該期不可用。

使用 max(Date, DisclosedDate) 與 DisclosedTime，歷史 15:00 及之後公告，下一交易日才可用；2020-10-01 休市排除。財報數值僅使用提供 ZIP 的 train_files/financials.csv。每個 SignalDate 保存六項數值、有效性與 EventId；Target 成熟後仍用當時凍結的數值更新。

## 六項特徵覆蓋與範圍

{table(['特徵','驗證期可用列','可用比例','全期最小值','全期最大值'],frows)}

共 {feature['validation_rows']:,} 個驗證預測列。來源表選取 {feature['selected_source_rows']:,} 筆相關公告，形成 {feature['event_count']:,} 個六項狀態事件。負基期轉正及虧損縮小均保留。尤其 EPS 原值與成長率的數值尺度差距很大，這次沒有為了結果好看而額外縮放。

## 訓練與 EPS 尺度診斷

每筆 loss=`(g_i−Target_i)²`，梯度=`2(g_i−Target_i) x_i`。沿用單檔 SGD，一個新成熟日期的每檔股票各更新一次，不重訪更早日期，不跨年重設，也沒有 soft rank。隨機排列種子仍為 `20260912+YYYYMMDD`。

學習率沿用 v10/v11：先 0.1，成功則依次試 0.2…1.0 直到首個不改善；0.1 失敗則試 0.09…0.01、0.009…0.001，直到 0.000009…0.000001。每個候選从同一更新前參數與梯度出發，需嚴格降低該筆 loss 且符合 Armijo c=1e−4。梯度無窮範數≤1e−10 或全部候選失敗就跳過，沒有擅自把最小學習率往下延伸。

{table(['項目','純價量對照','六項財報模型'],counts)}

約 **{diag['eps_gradient_dominance_fraction']:.2%}** 的 batch 中，兩個 EPS 梯度合計平方占全部梯度平方範數至少 90%。這是輸入尺度診斷，**不是 EPS 預測重要性的證明**。

新模型有 {diag['no_acceptable_eta_batches']:,} 筆所有學習率候選均不合格，其中 {diag['no_eta_with_norm_squared_above_armijo_limit']:,} 筆的輸入平方範數超過現有最小學習率可滿足 Armijo 的上限，{diag['no_eta_where_eps_inputs_alone_exceed_armijo_norm_limit']:,} 筆光是兩個 EPS 的平方和就足以超過這個上限。對單筆線性平方誤差，條件為 `η ||x||² ≤ 1−c`；本次 η 最小 1e−6、c=1e−4，上限為 999,900。

最大的幾個預測誤差也已拆出財報和 EPS 貢獻。例如 {example['Date']}，股票 {example['SecuritiesCode']} 的 g={example['g']:.6f}，Target={example['Target']:.6f}，財報項合計 {example['FinancialContribution']:.6f}，其中 EPS 項合計 {example['EPSContribution']:.6f}。這些數字是報酬小數；顯示此版的預測幅度有嚴重誤差。此為既有模型的事後分解，並非另做 EPS 單獨消融的因果判定。

本輪忠實保留 EPS 原值與既有 SGD 規則，沒有在看到 MSE 後重新訓練修正參數。每筆成功更新降低當筆 loss，不保證整日 MSE 降低；表中的整日上升日期就是由保存參數重算的結果。

## 獨立查核與檔案

48,142 筆來源公告的時間門檻、45,283 個狀態事件、全部 2,332,001 列特徵已查核；來源引用和分母期間、負值成長與原值 EPS 計算吻合。另以獨立 as-of 合併重建每個 SignalDate 的特徵，沒有用未來公告。

兩組所有 **2,325,806 個逐檔 SGD 步驟（含接受與跳過）**都已用獨立 NumPy 逐步重演，重算誤差、六項梯度與套用保存 η 後的參數。新模型最大最終日參數差 {a['max_sequential_final_state_error']:.3g}、最大逐步 loss 差 {a['max_sequential_loss_error']:.3g}。完整候選學習率搜尋另對保存的 {a['snapshot_full_candidate_searches_replayed']:,} 個 batch 逐一重演，每組另做 5 個真實 batch 有限差分核對。每項財報輸入為 0 時，對應係數在該筆均保持不變。

全部 1,199 個預測日、2,326,022 個預測列的分數及唯一整數排名重建；953 個驗證日的官方風格多空各 200 檔、2→1 權重 spread 和報酬 MSE 獨立重算通過。新模型官方 spread 最大差 {a['max_official_spread_error']:.3g}。驗證期沒有選到缺 Target 的股票。最后預測日為 2021-12-01，最後參數更新至 2021-12-03。

- [完整比較數值]({ROOT/'results.json'})、[逐日比較]({ROOT/'daily_comparison.csv'})、[全部 18 個最終係數]({ROOT/'final_coefficients.csv'})
- [六項財報特徵查核]({ROOT/'feature_audit.json'})、[完整更新查核]({ROOT/'six_financial/audit.json'})
- [每日參數]({ROOT/'six_financial/parameter_history.csv'})、[每日更新]({ROOT/'six_financial/training_updates.csv'})、[逐筆梯度摘要]({ROOT/'six_financial/batch_trace.csv.gz'})
- [EPS 與學習率診斷]({ROOT/'diagnostics.json'})、[最大預測誤差及貢獻]({ROOT/'largest_prediction_errors.csv'})
- [擬合前計畫]({ROOT/'experiment_plan.md'})、[來源雜湊]({ROOT/'source_hashes.json'})

以上是已被多個版本反覆比較的歷史驗證，未扣交易成本，沒有新的獨立 test，也不是排行榜成績。Sharpe 的改善不能當作已驗證的未來收益保證。
'''
    for x,y in [('没有','沒有'),('从同','從同'),('最后','最後')]:text=text.replace(x,y)
    report=ROOT/'JPX-v12-six-financial-features-report.md';report.write_text(text)
    print(json.dumps({'comparison':comparison,'annual':annual,'report':str(report)},ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
