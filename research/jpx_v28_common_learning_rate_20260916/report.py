from pathlib import Path
import json
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent;B=R.parent
PATHS={'g_original':B/'jpx_v23_g_thresholds_20260916/no_skip','a_own':B/'jpx_v27_profit_forecast_state_20260916/overwrite','b_own':B/'jpx_v27_profit_forecast_state_20260916/change_old','g_common':R/'g_common','overwrite':R/'overwrite','change_old':R/'change_old'}
LABELS={'g_original':'原純g：原步長','a_own':'A：原各自步長','b_own':'B：原各自步長','g_common':'純g：共同步長','overwrite':'A覆蓋：共同步長','change_old':'B差額＋舊值：共同步長'}
def save(n,d):(R/n).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False))
def csv(n,d):d.to_csv(R/n,index=False,encoding='utf-8-sig')
def sharp(x):return float(np.mean(x)/np.std(x,ddof=1))
def main():
    assert json.loads((R/'audit.json').read_text())['passed']
    frames={};updates={};rows=[];annual=[];coefs=[]
    for n,p in PATHS.items():
        d=pd.read_csv(p/'daily_metrics.csv');frames[n]=d;u=pd.read_csv(p/'training_updates.csv');updates[n]=u;rr=json.loads((p/'results.json').read_text())
        pd.testing.assert_series_equal(d.Date,frames['g_original'].Date,check_exact=True)
        for c in ['Date','SignalDate','ExitDate','TrainingStocks','ThresholdSkippedStocks']:pd.testing.assert_series_equal(u[c],updates['g_original'][c],check_exact=True)
        assert len(d)==953 and d.RankIC.notna().sum()==952 and u.ThresholdSkippedStocks.sum()==0 and d.SelectedMissingTargets.sum()==0
        rows.append(dict(Variant=n,Label=LABELS[n],Sharpe=sharp(d.OfficialDailySpread),RankIC=float(d.RankIC.mean()),MSE=float(d.AllStockForecastMSE.mean())))
        for year,q in d.groupby('ValidationYear'):annual.append(dict(Variant=n,ValidationYear=int(year),Days=len(q),Sharpe=sharp(q.OfficialDailySpread),RankIC=float(q.RankIC.mean()),MSE=float(q.AllStockForecastMSE.mean())))
        for k,v in rr['final_coefficients'].items():coefs.append(dict(Variant=n,Parameter=k,Value=v,AsOf=rr['parameter_asof']))
    df=pd.DataFrame(rows);ad=pd.DataFrame(annual);by=df.set_index('Variant')
    schedule=pd.read_csv(R/'common_learning_rates.csv')
    for n in ['g_common','overwrite','change_old']:pd.testing.assert_series_equal(updates[n].LearningRate,schedule.LearningRate,check_exact=True)
    rate_stats={}
    for n,old in [('g_common','g_original'),('overwrite','a_own'),('change_old','b_own')]:
        ratio=updates[n].LearningRate/updates[old].LearningRate
        rate_stats[n]=dict(changed_days=int((~np.isclose(ratio,1,rtol=1e-10,atol=1e-12)).sum()),days_below_one_tenth=int((ratio<.1).sum()),minimum_ratio=float(ratio.min()),median_ratio=float(ratio.median()))
    rng=np.random.default_rng(20260916);N=953;block=20;reps=4000
    ix=(rng.integers(0,N-block+1,size=(reps,int(np.ceil(N/block))))[:,:,None]+np.arange(block)).reshape(reps,-1)[:,:N];boots={}
    for n,d in frames.items():
        a=d.OfficialDailySpread.to_numpy()[ix];boots[n]={'Sharpe':a.mean(axis=1)/a.std(axis=1,ddof=1),'RankIC':np.nanmean(d.RankIC.to_numpy()[ix],axis=1)}
    ci=[]
    pairs=[('change_old','overwrite'),('overwrite','g_common'),('change_old','g_common'),('overwrite','g_original'),('change_old','g_original'),('g_common','g_original'),('overwrite','a_own'),('change_old','b_own')]
    for a,b in pairs:
        for m in ['Sharpe','RankIC']:
            lo,hi=np.quantile(boots[a][m]-boots[b][m],[.025,.975]);ci.append(dict(Comparison=a+' - '+b,Metric=m,Difference=float(by.loc[a,m]-by.loc[b,m]),CI95Low=float(lo),CI95High=float(hi)))
    levels=[]
    for n in ['g_common','overwrite','change_old']:
        lo,hi=np.quantile(boots[n]['RankIC'],[.025,.975]);levels.append(dict(Variant=n,RankIC=float(by.loc[n,'RankIC']),CI95Low=float(lo),CI95High=float(hi)))
    for name,data in [('comparison.csv',df),('comparison_by_year.csv',ad),('paired_block_bootstrap.csv',pd.DataFrame(ci)),('rank_ic_level_intervals.csv',pd.DataFrame(levels)),('all_coefficients.csv',pd.DataFrame(coefs)),('all_models_daily.csv',pd.concat([d.assign(Variant=n) for n,d in frames.items()]))]:csv(name,data)
    results=dict(version='v28',audit_passed=True,comparison=rows,paired_bootstrap=ci,rank_ic_intervals=levels,learning_rate_stats=rate_stats,common_rate_rule='min(A own spectral rate, B own spectral rate), recomputed per known training batch; all three models use exactly the same scalar each day',training_financial_abs_limit=None,validation_financial_abs_limit=None,active_baseline='v7')
    save('results.json',results)
    lines=['# v28：A、B及純g使用相同每日learning rate','','## 結論','','共同步長後，B的Sharpe點估計高於A；上輪A較好的排序反轉。A、B都略高於共同步長的純g，卻低於原v7純g，平均Rank IC皆負。所有列示差值的95%探索區間均跨0，仍未通過使用者的雙指標門檻。','','這說明原結論對learning rate排程敏感，不能把上輪差距全歸因於預測修正公式；也不能據這輪點估計就認定B穩定勝出。','','## 模型與控制','','A = g + beta × new；B = g + r1 × (new-old) + r2 × old。new與old完全沿用v27同基期營業利益預測年比、季度切換重建、只保留最新狀態及exp(-a/9)。價量及財報係數共同訓練，從零初始化，每日一次普通MSE，PR1/VR1原值，不排除或裁切極端值。','','共同eta_t = min(eta_A,t, eta_B,t)，其中eta_j,t = 1 / (2 lambda_max(X_j,t^T X_j,t / N_t))。每次只由當時已到期訓練批次決定，不按績效挑值。此處固定指同日三組數值相同，可隨日期改變；不是整段固定單一常數。純g共同步長組用來隔離learning rate變化。','','## 全期比較','','953驗證日，2017-12-29至2021-12-01；Rank IC有效952日，真實Target全相同日不補0。Sharpe未年化、未扣成本；MSE是每日橫斷面預測MSE的平均，僅診斷。','','| 模型 | Sharpe | 平均Rank IC | MSE |','|---|---:|---:|---:|']
    for q in rows:lines.append(f'| {q["Label"]} | {q["Sharpe"]:+.8f} | {q["RankIC"]:+.8f} | {q["MSE"]:.9f} |')
    lines+=['','## 分期Sharpe','','分期採用既有ValidationYear標籤，並非重新切割自然年度。','','| 模型 | 2018 | 2019 | 2020 | 2021 |','|---|---:|---:|---:|---:|']
    for n in PATHS:
        a=ad[ad.Variant.eq(n)].set_index('ValidationYear');lines.append('| '+LABELS[n]+' | '+' | '.join(f'{a.loc[y,"Sharpe"]:+.6f}' for y in [2018,2019,2020,2021])+' |')
    lines+=['','## 共同步長核對','','所有1199個更新日，三組LearningRate逐值完全相同，股票日與到期時序也完全一致。以下比較新舊排程；變更日數排除浮點捨入差異。','','| 新模型 | 步長變更日 | 低於原十分之一日數 | 新／原中位數 | 新／原最低值 |','|---|---:|---:|---:|---:|']
    for n,q in rate_stats.items():lines.append(f'| {LABELS[n]} | {q["changed_days"]} | {q["days_below_one_tenth"]} | {q["median_ratio"]:.6f} | {q["minimum_ratio"]:.6f} |')
    lines+=['','## 配對差值與不確定性','','20交易日連續區塊、4000次同日期配對重抽樣，種子20260916，95%百分位探索區間。未校正多重比較及歷史模型選擇，並非新保留測試。','','| 比較 | 指標 | 差值 | 95%區間 |','|---|---|---:|---:|']
    for q in ci:lines.append(f'| {q["Comparison"]} | {q["Metric"]} | {q["Difference"]:+.8f} | [{q["CI95Low"]:+.8f}, {q["CI95High"]:+.8f}] |')
    lines+=['','## Rank IC相對0','','| 模型 | 平均Rank IC | 95%區間 |','|---|---:|---:|']
    for q in levels:lines.append(f'| {LABELS[q["Variant"]]} | {q["RankIC"]:+.8f} | [{q["CI95Low"]:+.8f}, {q["CI95High"]:+.8f}] |')
    lines+=['','## 能與不能推論的部分','','- 這輪控制了相同日期的標量learning rate，沒有控制不同特徵表示下的梯度方向或有效函數更新幅度。即使相同learning rate，A與B也不會自動走同一條訓練路徑。','- B內含A的函數形式：r1=r2=beta時等於A；但每日一步MSE更新、估計誤差與樣本外績效，均不保證較多參數更好。','- 共同排程為兩模型較小步長，並未搜索最佳learning rate。共同步長纯g弱於原g，說明這個排程本身會改變價量學習結果；不能把差額解讀成市場是否信任修正。','- 延用v27財報有效文件規則；會計口徑不明的通用修正文件仍未採用，並非市場全部修正都可納入。來源、事件、衰減及極端值處理本輪未改。','','## 執行與核對','','三個新模型各1199次更新、2,325,806訓練股票日、2,326,022預測。排除0筆極端值；驗證股票池完整。每次共同步長由已知特徵重新計算，梯度、loss、參數歷程、預測、排序、Rank IC、MSE及官方Sharpe獨立重算通過，歷史來源雜湊未改。正式基準維持v7。','','## 最後財報係數','','截至2021-12-03最後到期更新後，僅供追蹤；回測使用各當日參數。']
    for q in coefs:
        if q['Variant'] in ['overwrite','change_old'] and q['Parameter'].startswith('Profit'):lines.append(f'- {q["Variant"]}／{q["Parameter"]}：{q["Value"]:+.10f}')
    (R/'JPX-v28-common-learning-rate-report.md').write_text(('\n'.join(lines)+'\n').replace('纯','純'))
    version=dict(version='v28',description='Common per-day learning rate for operating-profit forecast overwrite, delta+old, and pure price-volume control',directory=str(R),status='completed_experiment',variants=['g_common','overwrite','change_old'],audit_passed=True,training_financial_abs_limit=None,validation_financial_abs_limit=None)
    save('version.json',version);reg=json.loads((B/'jpx_model_versions.json').read_text());assert reg['active_version']=='v7';reg['versions']=[v for v in reg['versions'] if v.get('version')!='v28']+[version];reg['latest_experiment']='v28';(B/'jpx_model_versions.json').write_text(json.dumps(reg,ensure_ascii=False,indent=2))
    print(df.to_string(index=False));print(ad.to_string(index=False));print(pd.DataFrame(ci).to_string(index=False));print(pd.DataFrame(levels).to_string(index=False));print(json.dumps(rate_stats,indent=2))
if __name__=='__main__':main()
