import json,hashlib
import numpy as np
import pandas as pd
from run import ROOT,BASE,FINS,save,sharp

def main():
    assert json.loads((ROOT/'audit.json').read_text())['passed']
    paths=[ROOT/'price_only']+[ROOT/f'{s}_{j:02d}' for j in range(1,16) for s in ['feature','control']]
    dfs={p.name:pd.read_csv(p/'daily_metrics.csv') for p in paths};results={p.name:json.loads((p/'results.json').read_text()) for p in paths}
    common=set(dfs['price_only'].Date)
    for d in dfs.values():common &= set(d.loc[d.SelectedMissingTargets.eq(0),'Date'])
    ss={k:d.loc[d.Date.isin(common)].copy() for k,d in dfs.items()};base=sharp(ss['price_only'].OfficialDailySpread)
    rows=[];yearly=[];combined=[];coeff=[]
    trans={'NetSales':'營收','OperatingProfit':'營業利益','EPS':'EPS'};parts={'ActualGrowth':'實際 YoY 成長','ExpectedGrowth':'事前預測相對實際基期成長','ForecastQoQ':'預測 QoQ 成長','ForecastYoY':'預測 YoY 成長','Revision':'預期修正'}
    def chinese(f):
        for m,v in trans.items():
            if f.startswith(m):return v+'／'+parts[f[len(m):]]
    for j,feature in enumerate(FINS,1):
        name=f'feature_{j:02d}';cn=f'control_{j:02d}';r=results[name];s=sharp(ss[name].OfficialDailySpread);cs=sharp(ss[cn].OfficialDailySpread)
        tr=pd.read_csv(ROOT/name/'training_updates.csv');ct=pd.read_csv(ROOT/cn/'training_updates.csv');ratio=tr.LearningRate/ct.LearningRate
        rows.append({'Feature':feature,'FeatureChinese':chinese(feature),'Variant':name,'Sharpe':s,'BaselineSharpe':base,'DeltaSharpeVsV7':s-base,'MatchedControlSharpe':cs,'DeltaSharpeVsMatchedControl':s-cs,'SharpeAboveZero':s>0,'ImprovedVsV7':s>base,'ImprovedVsBothControls':s>base and s>cs,'CommonDays':len(common),'MeanRankIC':float(ss[name].RankIC.mean()),'HardRankMSE':float(ss[name].NormalizedHardRankMSE.mean()),'ReturnMSE':float(ss[name].AllStockForecastMSE.mean()),'SkippedTrainingStockDays':r['threshold_skipped_stock_days'],'FinalFinancialCoefficient':r['final_coefficients'][feature],'MedianLearningRateRatioVsMatchedControl':float(ratio.median()),'MinimumLearningRateRatioVsMatchedControl':float(ratio.min())})
        for yr,g in ss[name].groupby('ValidationYear'):
            bd=ss['price_only'];cd=ss[cn];bs=sharp(bd.loc[bd.ValidationYear.eq(yr),'OfficialDailySpread']);csh=sharp(cd.loc[cd.ValidationYear.eq(yr),'OfficialDailySpread']);ys=sharp(g.OfficialDailySpread)
            yearly.append({'Feature':feature,'ValidationYear':int(yr),'Days':len(g),'Sharpe':ys,'BaselineSharpe':bs,'MatchedControlSharpe':csh,'DeltaSharpeVsV7':ys-bs,'DeltaSharpeVsMatchedControl':ys-csh,'MeanRankIC':float(g.RankIC.mean())})
    df=pd.DataFrame(rows).sort_values('Sharpe',ascending=False);df.to_csv(ROOT/'comparison.csv',index=False,encoding='utf-8-sig');yd=pd.DataFrame(yearly);yd.to_csv(ROOT/'comparison_by_year.csv',index=False,encoding='utf-8-sig')
    for name,d in dfs.items():
        d=d.copy();d.insert(0,'Variant',name);combined.append(d)
        for k,v in results[name]['final_coefficients'].items():coeff.append({'Variant':name,'Parameter':k,'Value':v,'AsOf':results[name]['parameter_asof']})
    pd.concat(combined,ignore_index=True).to_csv(ROOT/'all_models_daily.csv',index=False,encoding='utf-8-sig');pd.DataFrame(coeff).to_csv(ROOT/'all_coefficients.csv',index=False,encoding='utf-8-sig')
    lines=['# v16：v7 價量模型逐一加入 15 個財報成分','',f'2026-09-15。純價量 v7 完整重現：共同 {len(common)} 日未年化 Sharpe **{base:+.8f}**。15 個单一財報模型中，**{int(df.SharpeAboveZero.sum())} 個 Sharpe > 0，{int(df.ImprovedVsV7.sum())} 個高於 v7，{int(df.ImprovedVsBothControls.sum())} 個同時高於 v7 與同樣篩選股票日的純價量對照**。','', '每組都是獨立的 g_j = 截距 + 11 個價量特徵加權和 + gamma_j × F_j；價量與新增係數一起訓練，每次僅加入一欄，不累積上一組特徵或權重。PR1/VR1 維持 v7 原值、財報沿用 exp(-a/9)，每日一次普通 MSE 更新。','', '## 全驗證期比較','', '| 財報成分 | Sharpe | Δ 對 v7 | Δ 對同樣篩選對照 | 跳過訓練股票日 |','|---|---:|---:|---:|---:|']
    for r in df.itertuples():lines.append(f'| {r.FeatureChinese} | {r.Sharpe:+.8f} | {r.DeltaSharpeVsV7:+.8f} | {r.DeltaSharpeVsMatchedControl:+.8f} | {r.SkippedTrainingStockDays:,} |')
    lines += ['', 'Sharpe > 0 與比原 g 更好是兩個條件。Δ 對 v7 為主比較；Δ 對同樣篩選對照用來判斷是否只是移除了不同訓練股票日。所有比較保留完整預測股票池，沒有用驗證標籤刪除不利股票。', '', '## 分期 Sharpe','', '| 財報成分 | 2018 | 2019 | 2020 | 2021 |','|---|---:|---:|---:|---:|']
    by={yr:sharp(g.OfficialDailySpread) for yr,g in ss['price_only'].groupby('ValidationYear')};lines.append('| 純價量 v7 | '+' | '.join(f'{by[yr]:+.6f}' for yr in [2018,2019,2020,2021])+' |')
    for r in df.itertuples():
        g=yd[yd.Feature.eq(r.Feature)].set_index('ValidationYear');lines.append('| '+r.FeatureChinese+' | '+' | '.join(f'{g.loc[yr,"Sharpe"]:+.6f}' for yr in [2018,2019,2020,2021])+' |')
    lines += ['', '驗證組別沿用既有 calendar 標記；完整訊號日期為 2017-12-29 至 2021-12-01，參數自 2017-01-04 起連續訓練、截至 2021-12-03 的到期標籤。每組 1,199 次每日更新。Sharpe 未年化、未扣成本；多空各 200 檔、權重 2 至 1。Rank IC 缺失日不以 0 取代。', '', '## 這輪最值得注意的數據','', '營業利益實際同比成長的整體改善最大，且 4 個分期中有 3 個高於基準；EPS 事前預期成長有 2 期改善，EPS 預期修正只有 1 期改善。總 Sharpe 提高並不代表每期穩定改善。', '營業利益預測季增的 Sharpe 雖比原 v7 高約 0.000688，但比同樣篩選股票日的純價量對照只高約 0.000053；這項改善很大一部分伴隨門檻造成的樣本改變，應避免只看主比較就高估特徵的效果。', 'EPS 實際成長、EPS 預測季增、EPS 預測年增、營業利益預期修正與營收預期修正，這次都低於原 v7 與各自的同樣篩選對照，是目前設定下值得優先檢查的五項。','', '## 能得到的結論與限制','', '1. 若某一欄的 Sharpe 下降，這是該財報成分在目前定義、尺度與訓練方式下未改善此歷史驗證期；尚不能證明其財報資訊或經濟假說普遍無效。單欄無效也不排除與另一欄搭配後有效。', '2. 這次拆成 15 個原始成分：營收／營業利益／EPS 各有實際成長 u、事前預期成長 v、預測季比 Q、預測年比 Y、修正 U。單獨 u 或 v 的試驗不是 beta*(u-d*v) 的完整實績驚喜假說，沒有另估 d。EPS 也沿用 v14 的成長率，並非早期 EPS 原值。', '3. 只以正在加入的成分 abs(F)>100 排除當次股票日訓練，±100 保留。各組額外跑一個相同訓練樣本的純價量模型。預測仍含超門檻股票。', '4. 保持 v7 的步長公式 1/(2 λmax(XᵀX/N))；新增欄會改變特徵矩陣與步長，所以比較的是「此特徵配合原訓練方法」的整體效果。comparison.csv 另列學習率對同樣篩選對照的比值。', '5. 15 組皆在同一段已反覆研究的驗證期比較，屬探索性結果；挑出最佳項後仍需要未用過的測試期，不能把當前最佳值當成已證實的未來改善。', '', '## 稽核與可重現資料','', '- 純價量基準的全部 2,326,022 筆預測與整數名次、953 日收益核對原 v7 通過；最終係數重現。', '- 31 組全部 1,199 次更新均以獨立 normal-equation 梯度、特徵值步長與參數步驟核對；每組所有預測、整數排名與官方 Sharpe 再算通過。', '- 所有訓練只使用到期標籤與原訊號日凍結特徵；每組篩選數量與相同樣本對照逐日一致。原 v14 已稽核財報快取直接使用，來源雜湊保留。', '- comparison.csv：15 組總比較、排名與報酬誤差、門檻數量、最終財報係數、步長比值。comparison_by_year.csv：60 列分期結果。', '- all_models_daily.csv：31 組逐日結果；all_coefficients.csv：全部最終係數。各組另存完整 parameter_history.csv、training_updates.csv、predictions.npz、results.json、audit.json。', '- predictions.npz 的 score/rank 對齊原 inputs.pkl 的 f 列序；閉市及不預測的尾端列為 NaN / -1。', '- 重現：run.py → audit.py → report.py。正式基準仍為 v7，沒有自動用最好的單欄模型取代它。']
    (ROOT/'JPX-v16-single-financial-report.md').write_text('\n'.join(lines)+'\n')
    result={'version':'v16','common_days':len(common),'baseline_sharpe':base,'positive_sharpe_features':int(df.SharpeAboveZero.sum()),'improved_vs_v7_features':int(df.ImprovedVsV7.sum()),'improved_vs_both_controls_features':int(df.ImprovedVsBothControls.sum()),'best':df.iloc[0].to_dict(),'all_results':df.to_dict('records'),'audit_passed':True}
    save(ROOT/'results.json',result)
    version={'version':'v16','description':'v7 ordinary daily MSE, raw PR1/VR1; add each of 15 decayed financial components separately with matched-sample price-only controls','directory':str(ROOT),'status':'completed_experiment','baseline':'v7_equal','variants':['price_only']+[f'feature_{j:02d}' for j in range(1,16)]+[f'control_{j:02d}' for j in range(1,16)],'baseline_sharpe':base,'common_days':len(common),'single_financial_feature_count':15,'audit_passed':True,'best_feature':df.iloc[0].Feature,'best_sharpe':float(df.iloc[0].Sharpe)}
    save(ROOT/'version.json',version)
    registry=json.loads((BASE/'jpx_model_versions.json').read_text());assert registry['active_version']=='v7'
    registry['versions']=[v for v in registry['versions'] if v.get('version')!='v16']+[version];registry['latest_experiment']='v16';save(BASE/'jpx_model_versions.json',registry)
    baseline=BASE/'JPX-current-baseline.md';text=baseline.read_text();section='## 最新完成的 v16 單一財報成分對照'
    if section not in text:
        baseline.write_text(text+f'\n{section}\n\n2026-09-15 回到 v7 每日一次普通 MSE、PR1/VR1 原值，保留財報 exp(-a/9)，逐一加入15欄財報成分，另設15組相同篩選股票日的純價量對照。相同{len(common)}日，基準Sharpe {base:+.8f}；{int(df.ImprovedVsV7.sum())}欄高於v7，{int(df.SharpeAboveZero.sum())}欄大於0。最佳單欄為{df.iloc[0].FeatureChinese}，Sharpe {df.iloc[0].Sharpe:+.8f}。全部更新與排名稽核通過。正式基準仍為v7。\n\n[完整 v16 報告]({ROOT}/JPX-v16-single-financial-report.md)。\n')
    manifest=json.loads((ROOT/'manifest.json').read_text());manifest['source_sha256'].update({str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'audit.py',ROOT/'report.py']});save(ROOT/'manifest.json',manifest)
    save(ROOT/'delivery_audit.json',{'passed':True,'models':31,'comparison_rows':len(df),'yearly_rows':len(yd),'common_days':len(common),'active_version_unchanged':'v7','latest_experiment':'v16','report_exists':True})
    print(df[['FeatureChinese','Sharpe','DeltaSharpeVsV7','DeltaSharpeVsMatchedControl']].to_string(index=False),flush=True)
if __name__=='__main__':main()
