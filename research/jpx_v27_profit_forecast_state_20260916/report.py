from pathlib import Path
import json
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent;B=R.parent
PATHS={'g':B/'jpx_v23_g_thresholds_20260916/no_skip','old_forecast':B/'jpx_v25_single_features_unfiltered_20260916/v16_feature_09','old_joint':B/'jpx_v26_profit_forecast_revision_20260916/joint','overwrite':R/'overwrite','change_old':R/'change_old'}
LABELS={'g':'純g','old_forecast':'舊版：g＋累加預測年比','old_joint':'舊版：g＋累加預測年比＋相對修正','overwrite':'A：g＋最新年比（覆蓋）','change_old':'B：g＋r1×(new−old)＋r2×old'}
def sharp(a):return float(np.mean(a)/np.std(a,ddof=1))
def save(n,v):(R/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))
def csv(n,d):d.to_csv(R/n,index=False,encoding='utf-8-sig')
def main():
    assert json.loads((R/'audit.json').read_text())['passed'];fa=json.loads((R/'feature_audit.json').read_text());frames={};updates={};rows=[];annual=[];coefs=[]
    for n,p in PATHS.items():
        d=pd.read_csv(p/'daily_metrics.csv');frames[n]=d;u=pd.read_csv(p/'training_updates.csv');updates[n]=u;rr=json.loads((p/'results.json').read_text())
        pd.testing.assert_series_equal(d.Date,frames['g'].Date)
        for c in ['Date','SignalDate','ExitDate','TrainingStocks','ThresholdSkippedStocks']:pd.testing.assert_series_equal(u[c],updates['g'][c])
        assert len(d)==953 and d.RankIC.notna().sum()==952 and d.SelectedMissingTargets.sum()==0 and u.ThresholdSkippedStocks.sum()==0
        rows.append(dict(Variant=n,Label=LABELS[n],Sharpe=sharp(d.OfficialDailySpread),RankIC=float(d.RankIC.mean()),MSE=float(d.AllStockForecastMSE.mean()),DeltaSharpeVsG=sharp(d.OfficialDailySpread)-sharp(frames['g'].OfficialDailySpread),DeltaRankICVsG=float(d.RankIC.mean()-frames['g'].RankIC.mean()),MedianLearningRate=float(u.LearningRate.median())))
        for y,g in d.groupby('ValidationYear'):annual.append(dict(Variant=n,ValidationYear=int(y),Days=len(g),Sharpe=sharp(g.OfficialDailySpread),RankIC=float(g.RankIC.mean()),MSE=float(g.AllStockForecastMSE.mean())))
        for name,v in rr['final_coefficients'].items():coefs.append(dict(Variant=n,Parameter=name,Value=v,AsOf=rr['parameter_asof']))
    df=pd.DataFrame(rows);ad=pd.DataFrame(annual);by=df.set_index('Variant');rng=np.random.default_rng(20260916);N=953;block=20;reps=4000
    ix=(rng.integers(0,N-block+1,size=(reps,int(np.ceil(N/block))))[:,:,None]+np.arange(block)).reshape(reps,-1)[:,:N];boots={}
    for n,d in frames.items():
        a=d.OfficialDailySpread.to_numpy()[ix];boots[n]={'Sharpe':a.mean(axis=1)/a.std(axis=1,ddof=1),'RankIC':np.nanmean(d.RankIC.to_numpy()[ix],axis=1)}
    ci=[]
    for a,b in [('overwrite','g'),('change_old','g'),('change_old','overwrite'),('overwrite','old_forecast'),('change_old','old_forecast'),('overwrite','old_joint'),('change_old','old_joint')]:
        for m in ['Sharpe','RankIC']:
            lo,hi=np.quantile(boots[a][m]-boots[b][m],[.025,.975]);ci.append(dict(Comparison=a+' - '+b,Metric=m,Difference=float(by.loc[a,m]-by.loc[b,m]),CI95Low=float(lo),CI95High=float(hi)))
    levels=[]
    for n in ['overwrite','change_old']:
        lo,hi=np.quantile(boots[n]['RankIC'],[.025,.975]);levels.append(dict(Variant=n,RankIC=float(by.loc[n,'RankIC']),CI95Low=float(lo),CI95High=float(hi)))
    ev=pd.read_pickle(R/'state_events.pkl');examples=ev[ev.Valid&ev.IsRevision].copy();examples['AbsoluteChange']=examples.DeltaYoY.abs();examples=examples.sort_values('AbsoluteChange',ascending=False).head(30).drop(columns=['Key','BaseKey','References']);csv('largest_state_revisions.csv',examples)
    for name,d in [('comparison.csv',df),('comparison_by_year.csv',ad),('paired_block_bootstrap.csv',pd.DataFrame(ci)),('rank_ic_level_intervals.csv',pd.DataFrame(levels)),('all_coefficients.csv',pd.DataFrame(coefs)),('all_models_daily.csv',pd.concat([d.assign(Variant=n) for n,d in frames.items()]))]:csv(name,d)
    lr_ratio=updates['change_old'].LearningRate/updates['overwrite'].LearningRate
    signals=pd.read_pickle(R/'financial_signal_features.pkl')
    optimization=dict(median_step_ratio_B_to_A=float(lr_ratio.median()),days_step_ratio_below_point_one=int((lr_ratio<.1).sum()),total_update_days=len(lr_ratio),minimum_step_ratio=float(lr_ratio.min()),maximum_absolute_signals={c:float(signals[c].abs().max()) for c in ['ProfitLatestYoY','ProfitYoYChange','ProfitPreviousYoY']})
    save('optimization_diagnostics.json',optimization)
    csv('learning_rate_comparison.csv',pd.DataFrame(dict(Date=updates['overwrite'].Date,LearningRateA=updates['overwrite'].LearningRate,LearningRateB=updates['change_old'].LearningRate,RatioBToA=lr_ratio)))
    result=dict(version='v27',audit_passed=True,comparison=rows,paired_bootstrap=ci,rank_ic_intervals=levels,feature_audit=fa,optimization_diagnostics=optimization,training_financial_abs_limit=None,validation_financial_abs_limit=None,active_baseline='v7')
    save('results.json',result)
    lines=['# v27：最新營業利益預測年比覆蓋與變化＋舊值','','按使用者確認的兩式比較：A=g+beta×new；B=g+r1×(new-old)+r2×old。new與old都是相同去年同季Forecast基期下的營業利益預測年比，不再把差額除以old。g與財報係數共同訓練，不排除或裁切極端值。','','## 全期結果','','953個驗證日，2017-12-29至2021-12-01。Rank IC有效952日，真實Target全部相同日未定義而不補0。Sharpe未年化、未扣成本；MSE僅作診斷。','','| 模型 | Sharpe | 平均Rank IC | MSE |','|---|---:|---:|---:|']
    for q in rows:lines.append(f'| {q["Label"]} | {q["Sharpe"]:+.8f} | {q["RankIC"]:+.8f} | {q["MSE"]:.9f} |')
    lines+=['','## new和old如何更新','','1. 原始單季Forecast仍由(全年Forecast-當時已知累計實績)/(4-已知季度數)取得。基期b是當時已凍結的去年同季單季預測，new=(Fnew-b)/abs(b)。','2. 同一目標季再次出現新值時，old使用上次保存的Fnew，在相同b下重算。不同季度的年比不直接相減。','3. 季度切換時，若有舊全年Forecast，先用同一份最新累計實績與剩餘季度數重建Fold，再計算old=(Fold-b)/abs(b)。因此財報公布同時修正全年Forecast的影響不會被錯當成0。','4. 完全没有可用舊Forecast時，以old=new、差額0初始化。最新季度或去年基期無法定義年比時，訊號清為0但保留該股票，避免把錯季度舊值繼續使用。','5. 新狀態覆蓋舊狀態，不累加歷史事件。new、new-old、old三欄共同乘exp(-距最新狀態更新交易日數/9)；新狀態重置交易日齡，重複相同值不重置。逐列驗證new=(new-old)+old。','6. A共13參數，B共14參數；r1、r2自由估計。當r1=r2時，B的財報部分在代數上等於A，但自由參數個數與最佳化矩陣不同，訓練路徑不保證相同。','', '## 資料時序與訊號數量','',f'狀態更新共{fa["state_events"]:,}筆，其中有效年比狀態{fa["valid_events"]:,}筆、非零預測變化{fa["nonzero_revision_events"]:,}筆（同一季度連續修正{fa["same_quarter_revisions"]:,}筆，季度切換以舊全年Forecast重建的修正{fa["quarter_rollover_rebased_revisions"]:,}筆）。無法定義年比的清空狀態{fa["invalid_state_events"]:,}筆。','', '沿用原財報有效性規則：會計基礎不明的通用修正/更正文件仍未採用。這些計數是本資料與規則能建立的事件，不代表市場全部修正。原v14事件和所有單季Forecast快照已精確重現；此次改變的是訊號表示與更新方式。','', '## 分期Sharpe','','| 模型 | 2018 | 2019 | 2020 | 2021 |','|---|---:|---:|---:|---:|']
    for n in PATHS:
        a=ad[ad.Variant.eq(n)].set_index('ValidationYear');lines.append('| '+LABELS[n]+' | '+' | '.join(f'{a.loc[y,"Sharpe"]:+.6f}' for y in [2018,2019,2020,2021])+' |')
    lines+=['','## 不確定性','','20日連續區塊、4000次同日期配對重抽樣，種子20260916。探索性95%百分位區間，未校正多重比較與過往模型選擇，不是新保留測試集。','','| 比較 | 指標 | 差值 | 95%區間 |','|---|---|---:|---:|']
    for q in ci:lines.append(f'| {q["Comparison"]} | {q["Metric"]} | {q["Difference"]:+.8f} | [{q["CI95Low"]:+.8f}, {q["CI95High"]:+.8f}] |')
    lines+=['','## Rank IC相對0','','| 模型 | Rank IC | 95%區間 |','|---|---:|---:|']
    for q in levels:lines.append(f'| {LABELS[q["Variant"]]} | {q["RankIC"]:+.8f} | [{q["CI95Low"]:+.8f}, {q["CI95High"]:+.8f}] |')
    lines+=['','## 解讀與核對','','- A/B使用完全相同的新狀態資料與當日股票池，只有財報項的參數化不同；步長按自己的Gram矩陣計算，因此比較仍包含最佳化差異。','- 相對舊版還改變了跨事件累加、季度首次訊號、修正分母及衰減重置方式，不能把差值全歸因於單一机制。','- 純g、舊單項及v26组合皆引用已核對的不排除版本；本輪每個新模型1199次更新、2,325,806個訓練股票日，極端值排除0筆。','- 新狀態公式、重建舊Forecast、來源已知時點、每筆訊號只使用最近狀態、new=delta+old已核對；所有梯度、步長、loss、參數更新、2,326,022筆預測、排名、Rank IC、MSE與官方Sharpe核對通過。','- 原始JPX快取與歷史結果未改動，正式基準保持v7。','', '## 最後財報係數','','以下截至2021-12-03最後到期標籤更新後，不是整段使用同一組期末參數。']
    for q in coefs:
        if q['Variant'] in ['overwrite','change_old'] and q['Parameter'].startswith('Profit'):lines.append(f'- {q["Variant"]}／{q["Parameter"]}：{q["Value"]:+.10f}')
    lines+=['','## 本輪結論','','- A的Sharpe高於純g、B低於純g，但A和B的平均Rank IC皆負，沒有通過「Sharpe高於g且Rank IC>0」的雙指標門檻。所有已列配對差值95%區間均跨0，目前只能視為探索性的點估計差異。','- A在2019、2020兩期Sharpe優於g，但2018、2021兩期更差；沒有逐期穩定改善。舊版預測年比單項仍是本表最高Sharpe。','- B涵蓋A的函數形式：r1=r2=beta時即等於A。然而每日一步MSE更新不保證找出更佳的Sharpe解，也不保證更彈性的模型在樣本外優於A。不能據此判定市場不相信修正，或修正值一定錯誤。',f'- B/A步長比中位數{optimization["median_step_ratio_B_to_A"]:.6f}，{optimization["total_update_days"]}個更新日中有{optimization["days_step_ratio_below_point_one"]}日低於0.1，最低{optimization["minimum_step_ratio"]:.6f}。分開表示後的較大特徵會在部分日期影響步長，但尚未以共同步長對照隔離此因素。','- 下一個可定位原因的對照：保持相同狀態與股票池，先約束r1=r2確認重現A，再比較共同步長的自由係數B；這些追加實驗本輪尚未執行。']
    (R/'JPX-v27-forecast-state-report.md').write_text(('\n'.join(lines)+'\n').replace('没有','沒有').replace('机制','機制').replace('组合','組合'))
    version=dict(version='v27',description='Latest operating-profit forecast YoY state: overwrite versus weighted change and old level, common-basis quarter rollovers',directory=str(R),status='completed_experiment',variants=['overwrite','change_old'],audit_passed=True,training_financial_abs_limit=None,validation_financial_abs_limit=None)
    save('version.json',version);reg=json.loads((B/'jpx_model_versions.json').read_text());assert reg['active_version']=='v7';reg['versions']=[v for v in reg['versions'] if v.get('version')!='v27']+[version];reg['latest_experiment']='v27';(B/'jpx_model_versions.json').write_text(json.dumps(reg,ensure_ascii=False,indent=2))
    print(df.to_string(index=False));print(ad.to_string(index=False));print(pd.DataFrame(ci).to_string(index=False));print(pd.DataFrame(levels).to_string(index=False));print(pd.DataFrame(coefs).query("Variant in ['overwrite','change_old']").tail(4).to_string(index=False))
if __name__=='__main__':main()
