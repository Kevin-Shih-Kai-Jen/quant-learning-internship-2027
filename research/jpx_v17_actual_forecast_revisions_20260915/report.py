import json,hashlib
import numpy as np
import pandas as pd
from run import ROOT,BASE,JOBS,specs,save,sharp
V16=BASE/'jpx_v16_single_financial_v7_20260915'
OLD={'eps_forecast_actual_qoq':'feature_13','eps_forecast_actual_yoy':'feature_14','sales_revision_relative':'feature_05','profit_revision_relative':'feature_10','eps_revision_relative':'feature_15','eps_actual_joint':'feature_11'}
LABEL={'eps_forecast_actual_qoq':'EPS預測／上一季實際','eps_forecast_actual_yoy':'EPS預測／去年同季實際','sales_revision_relative':'營收修正／舊單季預測','profit_revision_relative':'營業利益修正／舊單季預測','eps_revision_relative':'EPS修正／舊單季預測','eps_actual_qoq':'EPS實際季增單獨','eps_actual_joint':'EPS實際季增＋年增'}
FORMULAS={'eps_forecast_actual_qoq':'(本季預測EPS－上一季實際EPS) / |上一季實際EPS|','eps_forecast_actual_yoy':'(本季預測EPS－去年同季實際EPS) / |去年同季實際EPS|','sales_revision_relative':'(新單季營收預測－舊單季營收預測) / 舊單季營收預測','profit_revision_relative':'(新單季營業利益預測－舊單季營業利益預測) / 舊單季營業利益預測','eps_revision_relative':'(新單季EPS預測－舊單季EPS預測) / 舊單季EPS預測','eps_actual_qoq':'(本季實際EPS－上一季實際EPS) / |上一季實際EPS|','eps_actual_joint':'g + gammaQ × EPSActualQoQ + gammaY × EPSActualGrowth；兩個係數各自估計'}
def fmt(x):return '—' if x is None or pd.isna(x) else f'{x:+.8f}'
def main():
    assert json.loads((ROOT/'audit.json').read_text())['passed']
    frames={n:pd.read_csv(ROOT/n/'daily_metrics.csv') for n,_,_ in specs()};oldnames=set(OLD.values())|{'price_only','feature_12'}
    previous={n:pd.read_csv(V16/n/'daily_metrics.csv') for n in oldnames};common=set(previous['price_only'].Date)
    for d in list(frames.values())+list(previous.values()):common&=set(d.loc[d.SelectedMissingTargets.eq(0),'Date'])
    d={n:v[v.Date.isin(common)].copy() for n,v in frames.items()};p={n:v[v.Date.isin(common)].copy() for n,v in previous.items()};bs=sharp(p['price_only'].OfficialDailySpread)
    rows=[];annual=[]
    for name,features in JOBS:
        result=json.loads((ROOT/name/'results.json').read_text());new=sharp(d[name].OfficialDailySpread);ctrl=sharp(d[name+'_control'].OfficialDailySpread);old=sharp(p[OLD[name]].OfficialDailySpread) if name in OLD else None
        rows.append({'Variant':name,'Label':LABEL[name],'Formula':FORMULAS[name],'Sharpe':new,'BaselineSharpe':bs,'DeltaVsV7':new-bs,'MatchedPriceControlSharpe':ctrl,'DeltaVsMatchedPriceControl':new-ctrl,'OldVariant':OLD.get(name),'OldDefinitionSharpe':old,'DeltaVsOldDefinition':new-old if old is not None else None,'CommonDays':len(common),'TrainingSkippedStocks':result['threshold_skipped_stock_days'],'RankIC':float(d[name].RankIC.mean()),'ReturnMSE':float(d[name].AllStockForecastMSE.mean()),'HardRankMSE':float(d[name].NormalizedHardRankMSE.mean())})
        for year,g in d[name].groupby('ValidationYear'):
            bv=p['price_only'];annual.append({'Variant':name,'Label':LABEL[name],'ValidationYear':int(year),'Days':len(g),'Sharpe':sharp(g.OfficialDailySpread),'BaselineSharpe':sharp(bv.loc[bv.ValidationYear.eq(year),'OfficialDailySpread'])})
    df=pd.DataFrame(rows);df.to_csv(ROOT/'comparison.csv',index=False,encoding='utf-8-sig');ad=pd.DataFrame(annual);ad.to_csv(ROOT/'comparison_by_year.csv',index=False,encoding='utf-8-sig')
    joint=[]
    for name in ['eps_actual_joint','eps_qoq_joint_mask','eps_yoy_joint_mask','eps_actual_joint_control']:
        rr=json.loads((ROOT/name/'results.json').read_text());joint.append({'Variant':name,'Sharpe':sharp(d[name].OfficialDailySpread),'TrainingSkippedStocks':rr['threshold_skipped_stock_days'],'Features':','.join(rr['features'])})
    assert len(set(r['TrainingSkippedStocks'] for r in joint))==1
    jd=pd.DataFrame(joint);jd.to_csv(ROOT/'eps_joint_comparison.csv',index=False,encoding='utf-8-sig')
    # Same-label-day training sample comparisons, independent of ranking order.
    reference=pd.read_csv(ROOT/'eps_actual_joint/training_updates.csv')
    for name in ['eps_qoq_joint_mask','eps_yoy_joint_mask','eps_actual_joint_control']:
        cmp=pd.read_csv(ROOT/name/'training_updates.csv')
        for col in ['Date','SignalDate','ExitDate','KnownLabelStocks','TrainingStocks','ThresholdSkippedStocks']:assert reference[col].equals(cmp[col])
    combined=[];coefs=[]
    for name,_,_ in specs():
        dd=frames[name].copy();dd.insert(0,'Variant',name);combined.append(dd)
        rr=json.loads((ROOT/name/'results.json').read_text())
        for k,v in rr['final_coefficients'].items():coefs.append({'Variant':name,'Parameter':k,'Value':v,'AsOf':rr['parameter_asof']})
    pd.concat(combined,ignore_index=True).to_csv(ROOT/'all_models_daily.csv',index=False,encoding='utf-8-sig');pd.DataFrame(coefs).to_csv(ROOT/'all_coefficients.csv',index=False,encoding='utf-8-sig')
    events=pd.read_pickle(ROOT/'new_financial_events.pkl');rev=events[events.Kind.eq('U')];revision_signs=[]
    for name,g in rev.groupby('Feature'):revision_signs.append({'Feature':name,'Events':len(g),'NegativeOldForecastEvents':int(g.BaseValue.lt(0).sum()),'Formula':'(new-old)/old, signed denominator'})
    pd.DataFrame(revision_signs).to_csv(ROOT/'revision_denominator_diagnostics.csv',index=False,encoding='utf-8-sig')
    audit=json.loads((ROOT/'feature_audit.json').read_text())
    lines=['# v17：Forecast對實績、相對舊預測修正、EPS季增與年增','',f'2026-09-15。依使用者最新公式，三項預期修正改成帶正負號的 `(新單季預測－舊單季預測)/舊單季預測`；EPS預測改對照已知實際值；EPS實際季增與年增各有一個係數共同訓練。共 {len(specs())} 個新模型／對照均已完成並通過稽核，原 v7 與 v16 的相同結果直接引用。','',f'共同 {len(common)} 日（2017-12-29 至 2021-12-01）v7 Sharpe **{bs:+.8f}**。每日一次普通MSE、原值PR1/VR1、exp(-a/9)財報衰減、只在到期標籤上更新，都維持原規則。Sharpe未年化、未扣成本。','', '## 完整期間結果','', '| 新定義／模型 | Sharpe | Δ對v7 | Δ對同樣篩選純價量 | 舊定義或單獨年增 Sharpe |','|---|---:|---:|---:|---:|']
    for r in rows:lines.append(f'| {r["Label"]} | {fmt(r["Sharpe"])} | {fmt(r["DeltaVsV7"])} | {fmt(r["DeltaVsMatchedPriceControl"])} | {fmt(r["OldDefinitionSharpe"])} |')
    lines+=['','舊值欄：EPS預測兩組對照v16的預測對預測；修正三組對照v16以去年同季實績當分母的版本；EPS聯合模型對照v16單獨EPS實際年增。EPS季增單獨沒有舊版本。','', '## EPS 季增與年增的聯合檢驗','', '| 使用聯合模型同一批訓練樣本 | Sharpe | 跳過股票日 |','|---|---:|---:|']
    jl={'eps_actual_joint':'季增＋年增','eps_qoq_joint_mask':'只有季增','eps_yoy_joint_mask':'只有年增','eps_actual_joint_control':'只有價量'}
    for r in joint:lines.append(f'| {jl[r["Variant"]]} | {fmt(r["Sharpe"])} | {r["TrainingSkippedStocks"]:,} |')
    jmap={r['Variant']:r['Sharpe'] for r in joint};jq=jmap['eps_actual_joint'];jbest=max(jmap['eps_qoq_joint_mask'],jmap['eps_yoy_joint_mask'])
    lines+=['',f'聯合模型相對同樣樣本的季增單獨：{jq-jmap["eps_qoq_joint_mask"]:+.8f}；相對年增單獨：{jq-jmap["eps_yoy_joint_mask"]:+.8f}。'+('這輪聯合模型高於兩個單獨版本。' if jq>jbest else '這輪聯合模型沒有高於兩個單獨版本中的較佳者。'),'這只檢驗兩者共同使用的歷史效果，不等於完成季節調整、年度調整或證明彼此獨立。','', '## 公式及避免重複驗證的核對','']
    for n,_ in JOBS:lines.append(f'- {LABEL[n]}：{FORMULAS[n]}。')
    lines+=['','1. EPS預測對實際的兩項特徵，在新／改變全年Forecast首次建立該目標季的預測事件時加入。比較的上一季及去年同季實際值，都須在事件當時已知。舊v16 EPSExpectedGrowth雖然也是預期對去年同期實際成長，卻在當期實績公布時才觸發，時點不同，不能直接視為已有相同試驗。舊結果保持引用，不重新訓練。', '2. 三項修正的新舊單季預測，均先用相同的已知財年累計實績、相同剩餘季度數計算；全年Forecast值未改變時不新增修正事件。分母依本輪最後指定直接用舊單季預測，不取絕對值。', '3. 舊預測為負時，符號可能與「改善」直覺相反：-2改成-1為-50%，-1改成-2為+100%。這是指定公式的性質；原本實際成長／預測對實績的成長率仍用絕對值基期。舊值為0或必要資料缺失時，不建立該成分事件。', '4. EPS實際季增在正常首次單季實績公布且前季實績可用時加入；原EPS實際年增直接沿用v14/v16快取，沒有重定義或重訓舊單獨年增。聯合模型14個參數（價量12個＋季增、年增各1個），全部共同學習，沒有先把兩個比率相加共用係數。', '5. 所有事件依exp(-交易日齡/9)累加；沒有新事件時舊影響仍衰減。abs>100只跳過訓練股票日，聯合模型對兩欄任一超門檻即跳過。預測和多空選股仍保留完整股票池。','', '## 分期 Sharpe','', '| 模型 | 2018 | 2019 | 2020 | 2021 |','|---|---:|---:|---:|---:|']
    for name,_ in JOBS:
        g=ad[ad.Variant.eq(name)].set_index('ValidationYear');lines.append('| '+LABEL[name]+' | '+' | '.join(fmt(g.loc[y,'Sharpe']) for y in [2018,2019,2020,2021])+' |')
    lines+=['','驗證組別沿用既有calendar，不是嚴格曆年分界。這是同一段已反覆研究的歷史驗證，差異未經新保留資料驗證；本輪沒有自動挑選或合併最佳財報項。財報特徵也會影響v7依特徵矩陣決定的步長，结果属于特徵搭配這套優化器的效果。','', '## 資料與稽核','',f'- {audit["new_events"]:,} 筆新事件逐筆核對公式、已知發佈時點及原始累計實績差額；原v14的167,891事件精確重現。',f'- {audit["direct_decayed_sum_checks"]:,} 個訊號截面的衰減結果獨立直接求和核對。舊EPS年增快取逐值沿用。',f'- {len(specs())} 組全部1,199次日更新以獨立梯度、特徵值步長、更新前後loss核對；每組全部2,326,022筆預測、排名與官方Sharpe再計算通過。','- comparison.csv、comparison_by_year.csv：總表與分期。eps_joint_comparison.csv：聯合模型與相同樣本單獨模型。revision_denominator_diagnostics.csv：三項修正的負基期事件數。','- 各組predictions.npz對齊原inputs.pkl的f列序；另存逐日參數、訓練紀錄、結果及稽核。完整事件與特徵快取保存在本資料夾。','- 先前尚未訓練的abs分母中間版本在superseded_absolute_revision，未進入本轮模型或結果。','- 重現流程：features.py → run.py → audit.py → report.py。正式基準仍是v7。']
    (ROOT/'JPX-v17-revised-financial-report.md').write_text('\n'.join(lines)+'\n')
    clean=df.astype(object).where(pd.notna(df),None).to_dict('records')
    save(ROOT/'results.json',{'version':'v17','common_days':len(common),'baseline_sharpe':bs,'comparison':clean,'eps_joint_comparison':joint,'all_revisions_signed_old_forecast_denominator':True,'audit_passed':True})
    src=[ROOT/'features.py',ROOT/'run.py',ROOT/'audit.py',ROOT/'report.py',ROOT/'experiment_plan.md',ROOT/'financial_signal_features.pkl',ROOT/'feature_audit.json',BASE/'jpx_v8_soft_rank_20260912/inputs.pkl']+[V16/n/'results.json' for n in oldnames]
    save(ROOT/'manifest.json',{'source_sha256':{str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in src},'reused_old_models':sorted(oldnames),'revisions_denominator':'signed old quarterly forecast','zero_denominator':'no event','validation_common_days':len(common)})
    ver={'version':'v17','description':'EPS forecast versus known actuals; all three forecast revisions relative to signed old forecast; joint EPS actual QoQ and YoY on v7 daily MSE','directory':str(ROOT),'status':'completed_experiment','variants':[n for n,_,_ in specs()],'baseline':'v7_equal','common_days':len(common),'audit_passed':True,'joint_eps_sharpe':jq,'revision_definition':'(new-old)/old, signed'}
    save(ROOT/'version.json',ver);reg=json.loads((BASE/'jpx_model_versions.json').read_text());assert reg['active_version']=='v7';reg['versions']=[v for v in reg['versions'] if v.get('version')!='v17']+[ver];reg['latest_experiment']='v17';save(BASE/'jpx_model_versions.json',reg)
    b=BASE/'JPX-current-baseline.md';s=b.read_text();heading='## 最新完成的 v17 修訂財報比較'
    if heading not in s:b.write_text(s+f'\n{heading}\n\n2026-09-15 將EPS預測改與已知實際值比較，營收／營業利益／EPS三項修正改成(新單季預測-舊單季預測)/舊單季預測（帶正負號），並驗證EPS實際季增＋年增。共{len(specs())}組模型與對照通過稽核；相同{len(common)}日，聯合EPS模型Sharpe {jq:+.8f}，v7為{bs:+.8f}。正式基準仍為v7。\n\n[完整 v17 報告]({ROOT}/JPX-v17-revised-financial-report.md)。\n')
    save(ROOT/'delivery_audit.json',{'passed':True,'new_models':len(specs()),'primary_models':len(JOBS),'comparison_rows':len(df),'annual_rows':len(ad),'common_days':len(common),'active_baseline':'v7','all_revisions_updated':True})
    print(df[['Label','Sharpe','DeltaVsV7','OldDefinitionSharpe','DeltaVsMatchedPriceControl']].to_string(index=False),flush=True);print(jd.to_string(index=False),flush=True)
if __name__=='__main__':main()
