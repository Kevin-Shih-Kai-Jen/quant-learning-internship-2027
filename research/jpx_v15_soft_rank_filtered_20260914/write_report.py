import json,hashlib,platform,sys
from pathlib import Path
import numpy as np
import pandas as pd
from features import ROOT,V14,V8,V11,VARIANTS

def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    new={v:json.loads((ROOT/v/'results.json').read_text()) for v in VARIANTS}
    score_diagnostics={v:json.loads((ROOT/v/'score_diagnostics.json').read_text()) for v in VARIANTS}
    training_baseline=json.loads((ROOT/'training_flat_score_baseline.json').read_text())
    audits={v:json.loads((ROOT/v/'audit.json').read_text()) for v in VARIANTS}
    assert all(a['passed'] for a in audits.values())
    base=json.loads((ROOT/'v14_comparison_metrics.json').read_text());rows=[];years=[];daily_all={}
    for version in ['v14','v15']:
        for v in VARIANTS:
            src=V14 if version=='v14' else ROOT
            d=pd.read_csv(src/v/'daily_spread_returns.csv',parse_dates=['Date'])
            if version=='v14':
                rank=pd.read_csv(ROOT/f'v14_{v}_ranking_metrics.csv',parse_dates=['Date'])
                d=d.merge(rank[['Date','RankIC','NormalizedHardRankMSE']],on='Date',validate='one_to_one')
            daily_all[version+'_'+v]=d
            rows.append({'Version':version,'Variant':v,'Loss':'Return MSE' if version=='v14' else 'Soft rank',
                'Sharpe':float(d.OfficialDailySpread.mean()/d.OfficialDailySpread.std(ddof=1)),
                'MeanRankIC':float(d.RankIC.mean()),'NormalizedHardRankMSE':float(d.NormalizedHardRankMSE.mean()),'Days':len(d)})
            for yr,g in d.groupby('ValidationYear'):
                years.append({'Version':version,'Variant':v,'ValidationYear':int(yr),'Days':len(g),
                    'Sharpe':float(g.OfficialDailySpread.mean()/g.OfficialDailySpread.std(ddof=1)),'MeanRankIC':float(g.RankIC.mean())})
    comparison=pd.DataFrame(rows);comparison.to_csv(ROOT/'comparison.csv',index=False);pd.DataFrame(years).to_csv(ROOT/'comparison_by_year.csv',index=False)
    for d in daily_all.values():pd.testing.assert_series_equal(d.Date.reset_index(drop=True),next(iter(daily_all.values())).Date.reset_index(drop=True))
    inputs=[V8/'inputs.pkl',V11/'return_std22_features.pkl',V14/'financial_signal_features.pkl',V14/'feature_audit.json']
    source=[*ROOT.glob('*.py'),*ROOT.glob('*.cpp'),*ROOT.glob('*.dylib'),ROOT/'experiment_plan.md']
    manifest={'inputs':{str(p):sha(p) for p in inputs},'source':{p.name:sha(p) for p in source},
        'runtime':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'platform':platform.platform()},
        'compiler':'clang++ -O3 -std=c++17 -dynamiclib -framework Accelerate','cpu_workers':8}
    input_audit=json.loads((ROOT/'input_audit.json').read_text())
    assert input_audit['passed'] and input_audit['source_inputs']==manifest['inputs']
    assert json.loads((ROOT/'preflight.json').read_text())['passed']
    save(ROOT/'manifest.json',manifest)
    result={'version':'v15','variants':new,'audits':audits,'comparison':rows,'score_diagnostics':score_diagnostics,'training_flat_score_baseline':training_baseline,
        'feature_source_version':'v14','same_features_and_training_pools':True,'loss':'sigmoid soft rank normalized squared error',
        'temperature':1.,'financial_training_abs_limit':100.,'excluded_training_rows_per_arm':2294,
        'evaluation_pool_filtered':False,'costs_included':False,'fresh_test_set_used':False,'active_baseline_changed':False}
    save(ROOT/'results.json',result)
    names={'sgd_only':'單獨 SGD','sgd_sqrt':'SGD＋每日 √N 次整體更新'}
    table='| Loss | 訓練方式 | 未年化 Sharpe ↑ | 平均 Rank IC ↑ | 正規化排名 MSE ↓ |\n|---|---|---:|---:|---:|\n'
    for r in rows:table+=f"| {r['Version']} {r['Loss']} | {names[r['Variant']]} | {r['Sharpe']:+.8f} | {r['MeanRankIC']:+.8f} | {r['NormalizedHardRankMSE']:.8f} |\n"
    yearly='| 驗證組別 | v14 單獨 SGD | v15 單獨 SGD | v14 SGD＋√N | v15 SGD＋√N |\n|---|---:|---:|---:|---:|\n'
    for yr in sorted({y['ValidationYear'] for y in years}):
        vals=[next(y['Sharpe'] for y in years if y['ValidationYear']==yr and y['Version']==ver and y['Variant']==v) for ver,v in [('v14','sgd_only'),('v15','sgd_only'),('v14','sgd_sqrt'),('v15','sgd_sqrt')]]
        yearly+=f"| {yr} | "+' | '.join(f'{a:+.8f}' for a in vals)+' |\n'
    training='| Soft-rank 訓練階段 | 單獨 SGD | SGD＋√N |\n|---|---:|---:|\n'
    for label,key in [('更新前整日平均 loss','mean_training_loss_before'),('逐檔 SGD 後','mean_training_loss_after_sgd'),('整體更新後','mean_training_loss_after_full')]:
        training+=f'| {label} | '+ ' | '.join(f'{new[v][key]:.8f}' for v in VARIANTS)+' |\n'
    heldout='| 驗證期 soft-rank loss | 平均 loss ↓ | 優於同分基準的日數 |\n|---|---:|---:|\n'
    for v in VARIANTS:
        s=score_diagnostics[v]
        heldout+=f"| {names[v]} | {s['mean_daily_soft_rank_loss']:.8f} | {s['beat_flat_score_loss_days']} / {s['days']} |\n"
    heldout+=f"| 所有股票同分基準 | {score_diagnostics['sgd_only']['mean_daily_flat_score_soft_rank_loss']:.8f} | — |\n"
    params=pd.DataFrame({names[v]:new[v]['final_parameters'] for v in VARIANTS});params.to_csv(ROOT/'final_parameters.csv',index_label='Parameter')
    direction=[]
    for v in VARIANTS:
        dr=new[v]['validation']['official_style_unannualized_sharpe']-base[v]['sharpe'];di=new[v]['mean_daily_rank_ic']-base[v]['rank_ic']
        direction.append(f"{names[v]}相對同訓練方式的 v14，Sharpe 差 {dr:+.8f}，Rank IC 差 {di:+.8f}。")
    refinements=[(v,date,r) for v,a in audits.items() for date,r in a.get('refined_days',{}).items()]
    refinement_note=''
    if refinements:
        worst=max(max(r['max_relative_errors_loss_before_after_gradinf_norm_descent']) for _,_,r in refinements)
        refinement_note='\n較長區段的獨立浮點重播在 '+ '、'.join(names[v]+' '+date for v,date,_ in refinements)+f' 出現累積差異。這些日期另以每 8 檔股票保存檢查點，全部原訓練參數、trace、snapshots 與 loss 逐位元重現；更密集的獨立重播最大相對誤差為 {worst:.3g}。失敗的初次稽核紀錄與修正後紀錄均保留。\n'
    report='''# v15：Soft rank 與原報酬 MSE 對照

2026-09-14。依要求將 v14 的 loss 換為 sigmoid soft rank，其他資料、特徵、排除條件及訓練時序一致。

## 結果

'''+table+'\n'+' '.join(direction)+'''

這是同一段歷史資料的配對實驗，尚未使用新的保留測試集，也未扣交易成本；不能把這次差異直接視為穩定的未來改善。現有正式基準仍為 v7。

模型輸出 g 現在是排名分數，不再解讀為預測報酬率，因此不拿 (g − Target)² 和 v14 的報酬 MSE 做精準度比較。Rank IC 是每天預測分數與實際報酬的 Spearman 相關；排名 MSE 使用兩者的降序平均同分名次，除以 N−1 後平方。2020-09-29 的實際 Target 全為 0，該日 Rank IC 無定義；其餘 952 日取平均。953 日均保留於 Sharpe 與排名 MSE。

## Loss 與訓練公式

對每個到期標籤日，先排除缺失 Target，以及任一財報成分絕對值大於 100 的股票日，剩餘股票集合為 U，N=|U|。真實名次 ρᵢ 由 U 內 Target 降序排列，同分取平均名次。

Rᵢ = 1 + Σⱼ∈U,j≠i sigmoid((gⱼ − gᵢ)/τ)，τ = 1。

單檔 loss：Lᵢ = ((Rᵢ − ρᵢ)/(N − 1))²。

整體 loss：L = (1/N) Σᵢ∈U Lᵢ。

單檔 SGD 每次使用一檔的 Lᵢ，但 Rᵢ 和梯度都包含 U 中所有其他股票；沒有切斷其他股票的梯度，也沒有抽樣比較對象。被 100 倍門檻排除的股票不進入任何人的 soft-rank 比較。完整股票池仍用於預測與驗證。

每天先做 N 次單檔更新，第二組再做 ⌊√N⌋ 次整體更新。每次先更新 24 個線性係數、再更新 3 個折價參數，折價限制為 [0,1]；兩段各自重新計算梯度。沿用 v14 的投影 Armijo 學習率格點。共同截距無法改變排名，因此其梯度為 0，維持初始值 0。

財報特徵直接讀取已稽核的 v14 快取，保留 exp(−a/9) 衰減；沒有新 Forecast 時不新增該次 Forecast 事件，既有事件繼續衰減。價量 PR1／VR1 仍各自除以近 22 日標準差。

## 各期 Sharpe

'''+yearly+'''
驗證組別沿用既有 calendar 的標記，實際驗證日期為 2017-12-29 至 2021-12-01，共 953 日；暖身期另存。參數沿時間連續更新，沒有年度重設。只在 Target 已到期後使用當初訊號日凍結的特徵與篩選結果。

## 驗證期的 loss 診斷

'''+heldout+'''
同分基準只是診斷，不提供股票間的排序訊號。Soft-rank loss 同時受到名次與分數尺度影響；把所有分數乘以同一個正數，硬排名完全不變，但 soft rank 和 loss 可能改變。當這個倍數趨近 0，所有 soft rank 都會趨近共同的中間名次。因此這個 loss 仍需要搭配 Rank IC 和 Sharpe 判讀，不能只看 loss 下降就認定選股變準。溫度 τ 這次固定為 1，沒有依驗證期結果調整。

## 訓練 loss

'''+training+f'''
每組 1,199 個到期標籤日，2,323,512 次單檔更新，跳過 2,294 個超門檻訓練股票日。混合組另嘗試 52,189 次整體更新。單檔 loss 下降不保證同一天所有股票的平均 loss 下降；完整 SGD 後整日 loss 上升的天數：單獨 SGD {new['sgd_only']['sgd_increased_day_loss_days']}，混合組 {new['sgd_sqrt']['sgd_increased_day_loss_days']}。這些是已看到標籤後的擬合指標，不能替代後續預測績效。

同一批訓練資料上，所有股票同分的平均 loss 為 {training_baseline['sgd_only']['mean_flat_training_loss']:.8f}；單獨 SGD 完成更新後只有 {training_baseline['sgd_only']['after_full_beats_flat_days']} / 1,199 天低於此基準，混合組則為 {training_baseline['sgd_sqrt']['after_full_beats_flat_days']} / 1,199 天。把所有線性係數設為 0 就能達到同分基準，因此這是現有模型可達成的參考值。兩組平均訓練 loss 都高於它，顯示目前的更新方式與分數尺度尚需檢查；本次結果是 soft rank 在既有 SGD 規則下的表現。同分基準只作診斷，沒有選股訊號。

## 驗證與重現

- 合成資料的 81 個有限差分梯度、46 個完整學習率格點步驟通過；檢查同分名次、共同平移、極大分數、零財報梯度，以及被排除股票不影響名次。
- 兩組每一個已記錄的學習率更新都用獨立的分數／Jacobian 程式重播；於第一／中間／最後 SGD 股票及各整體更新保存的狀態校準，並量測校準前誤差。不是宣稱另一個浮點實作從頭到尾逐位元完全相同。
- 單獨 SGD 的 {audits['sgd_only']['independent_numpy_full_grid_substeps']} 個、混合組的 {audits['sgd_sqrt']['independent_numpy_full_grid_substeps']} 個真實資料子步驟，以 NumPy 獨立重跑完整候選格點；其餘逐步重播已選學習率及檢查接受條件。
- 每組全部 2,326,022 筆預測及整數名次重建一致，953 日多空各 200 檔收益和 Sharpe 重算通過。兩組訓練日期、篩選股票、亂數排序種子及更新數與 v14 核對一致。
- 可重現入口：build.py → audit_inputs.py → preflight.py → run.py → audit_results.py → compare_baseline.py → score_diagnostics.py → training_baseline.py → write_report.py。build.py 編譯四個計算／稽核函式庫；輸入與程式 SHA-256 保存在 manifest.json。

逐日排名、訓練紀錄、參數軌跡與稽核分別存於 sgd_only／sgd_sqrt；四組整體比較為 comparison.csv，分期比較為 comparison_by_year.csv。
'''+refinement_note
    report_path=ROOT/'JPX-v15-soft-rank-report.md';report_path.write_text(report)
    version={'version':'v15','description':'v14 frozen financial features and filtered training pool; sigmoid soft-rank loss with full reference-stock differentiation; per-stock SGD versus SGD plus floor(sqrt(N)) daily updates',
        'directory':str(ROOT),'status':'completed_experiment','variants':VARIANTS,'coefficient_count':27,'tau':1.,
        'training_financial_abs_limit':100,'evaluation_stock_pool_filtered':False,
        'variant_sharpes':{v:new[v]['validation']['official_style_unannualized_sharpe'] for v in VARIANTS},
        'variant_rank_ic':{v:new[v]['mean_daily_rank_ic'] for v in VARIANTS},'audit_passed':True}
    save(ROOT/'version.json',version)
    registry_path=ROOT.parent/'jpx_model_versions.json';registry=json.loads(registry_path.read_text())
    registry['versions']=[v for v in registry['versions'] if v['version']!='v15']+[version];registry['latest_experiment']='v15';save(registry_path,registry)
    baseline=ROOT.parent/'JPX-current-baseline.md';text=baseline.read_text();marker='\n## 最新完成的 v15 Soft rank 對照\n'
    text=text.split(marker)[0]+marker+'\n2026-09-14 沿用 v14 財報特徵與 100 倍訓練篩選，只改為 sigmoid soft rank loss。相同 953 日的未年化 Sharpe：'+ '；'.join(f"{names[v]} {new[v]['validation']['official_style_unannualized_sharpe']:+.8f}" for v in VARIANTS)+'。兩組完整更新紀錄與排名核對通過。這是既有歷史期間的實驗，未扣成本，也尚未使用新保留測試集。正式基準仍為 v7。\n\n'+f'[完整 v15 報告]({report_path})。\n';baseline.write_text(text)
    print(comparison.to_string(index=False));print(report_path)
if __name__=='__main__':main()
