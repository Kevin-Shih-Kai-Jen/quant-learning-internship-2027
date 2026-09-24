from pathlib import Path
import json
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent;B=R.parent
LABELS={'g':'純g（共同步長）','joint_r':'共同訓練：g＋實際R','joint_f':'共同訓練：g＋Forecast F','joint_rf':'共同訓練：g＋R＋F','frozen_r':'固定g：＋實際R','frozen_f':'固定g：＋Forecast F','frozen_rf':'固定g：＋R＋F'}
def save(n,d):(R/n).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False))
def csv(n,d):d.to_csv(R/n,index=False,encoding='utf-8-sig')
def sharp(x):return float(np.mean(x)/np.std(x,ddof=1))
def main():
    assert json.loads((R/'audit.json').read_text())['passed'];diag=json.loads((R/'diagnostics.json').read_text());assert diag['passed'];lr=json.loads((R/'learning_rate_decision.json').read_text())
    frames={};rows=[];annual=[];coefs=[]
    for n,label in LABELS.items():
        d=pd.read_csv(R/n/'daily_metrics.csv');frames[n]=d;rr=json.loads((R/n/'results.json').read_text());u=pd.read_csv(R/n/'training_updates.csv')
        pd.testing.assert_series_equal(d.Date,frames['g'].Date);assert len(d)==953 and d.RankIC.notna().sum()==952 and u.ThresholdSkippedStocks.sum()==0 and d.SelectedMissingTargets.sum()==0
        rows.append(dict(Variant=n,Label=label,Sharpe=sharp(d.OfficialDailySpread),RankIC=float(d.RankIC.mean())))
        for year,q in d.groupby('ValidationYear'):annual.append(dict(Variant=n,ValidationYear=int(year),Days=len(q),Sharpe=sharp(q.OfficialDailySpread),RankIC=float(q.RankIC.mean())))
        for k,v in rr['final_coefficients'].items():coefs.append(dict(Variant=n,Parameter=k,Value=v,AsOf=rr['parameter_asof']))
    df=pd.DataFrame(rows);ad=pd.DataFrame(annual);by=df.set_index('Variant');ranking=df.sort_values(['Sharpe','RankIC'],ascending=False,kind='stable');winner=ranking.iloc[0]
    original=pd.read_csv(B/'jpx_v23_g_thresholds_20260916/no_skip/daily_metrics.csv');frames['original_v7']=original;original_result=dict(Sharpe=sharp(original.OfficialDailySpread),RankIC=float(original.RankIC.mean()))
    rng=np.random.default_rng(20260916);N=953;block=20;reps=4000
    ix=(rng.integers(0,N-block+1,size=(reps,int(np.ceil(N/block))))[:,:,None]+np.arange(block)).reshape(reps,-1)[:,:N];boots={};ci=[]
    for n,d in frames.items():
        a=d.OfficialDailySpread.to_numpy()[ix];boots[n]={'Sharpe':a.mean(axis=1)/a.std(axis=1,ddof=1),'RankIC':np.nanmean(d.RankIC.to_numpy()[ix],axis=1)}
    pairs=[('joint_rf','joint_r'),('joint_rf','joint_f'),('frozen_rf','frozen_r'),('frozen_rf','frozen_f'),('joint_rf','g'),('frozen_rf','g'),('frozen_r','joint_r'),('frozen_f','joint_f'),('frozen_rf','joint_rf'),(winner.Variant,'original_v7')]
    for a,b in pairs:
        for m,col in [('Sharpe','OfficialDailySpread'),('RankIC','RankIC')]:
            lo,hi=np.quantile(boots[a][m]-boots[b][m],[.025,.975]);v=sharp(frames[a][col])-sharp(frames[b][col]) if m=='Sharpe' else float(frames[a][col].mean()-frames[b][col].mean());ci.append(dict(Comparison=a+' - '+b,Metric=m,Difference=v,CI95Low=float(lo),CI95High=float(hi)))
    for n,d in [('comparison.csv',df),('validation_ranking.csv',ranking),('comparison_by_year.csv',ad),('paired_block_bootstrap.csv',pd.DataFrame(ci)),('all_coefficients.csv',pd.DataFrame(coefs)),('all_models_daily.csv',pd.concat([q.assign(Variant=n) for n,q in frames.items()]))]:csv(n,d)
    results=dict(version='v30',audit_passed=True,comparison=rows,validation_winner=winner.Variant,original_v7=original_result,paired_bootstrap=ci,learning_rate_decision=lr,diagnostics=diag,test_evaluated=False,selection_priority=['Sharpe','RankIC'],mse_used_for_selection=False,formal_baseline='v7')
    save('results.json',results)
    lines=['# v30：營業利益實際年增與Forecast年比七組對照','','## 結論','','相同資料、股票日與共同步長下，共同訓練的g+R+F高於兩個單項，這輪沒有重現「實際值和Forecast一起必然更差」。固定g後，R+F略低於R單項，但高於F單項。這些是本輪點估計，區間見下表，不能直接推論資訊互相干擾或無用。', '',f'本輪最高Sharpe為{winner.Label}：{winner.Sharpe:+.8f}，平均Rank IC {winner.RankIC:+.8f}。原v7 Sharpe {original_result["Sharpe"]:+.8f}仍較高。按使用者規則以Sharpe第一、Rank IC第二，MSE不作選擇；test未讀取或評估，正式基準維持v7。','','## 全期結果','','953驗證日（2017-12-29至2021-12-01），Rank IC有效952日，真實Target全同日未定義而不補0。Sharpe未年化、未扣成本。','','| 模型 | Sharpe | 平均Rank IC |','|---|---:|---:|']
    for q in rows:lines.append(f'| {q["Label"]} | {q["Sharpe"]:+.8f} | {q["RankIC"]:+.8f} |')
    lines+=['','## 兩個訊號與固定g的定義','','R=OperatingProfitActualGrowth，單季實際營業利益對去年同季實績的成長率=(本季-去年同季)/abs(去年同季)，沿用v14歷史事件累加exp(-a/9)。F=ProfitLatestYoY，最新單季Forecast對去年同季Forecast的年比，沿用v27最新狀態覆蓋及exp(-a/9)。本輪沒有改成surprise、標準化、裁切或排除極端值。兩欄原本的狀態表示不同，這輪保留以便對照現有模型。','','R沿用既有有效事件規則：實際年增與事前預期成長都能計算才產生A事件，因此此欄不是所有已公布實績的完整涵蓋。通用修正文件的既有會計口徑有效性規則亦未改。','','固定g不是期末係數回填：先跑本輪同共同步長的純g，保存各訊號日當時預測；三個固定g模型訓練與預測皆用該股票原訊號日的g，加上1或2個財報係數。到期後才更新，沒有新增截距，不會改變g。共同訓練組則更新所有g與財報係數。','','## 步長控制', '',f'原v29的1倍排程在新增R後，有{lr["unstable_days"]}日超過完整[g,R,F]矩陣的穩定上限。因此依預先規則，七組一起改為min(原步長,完整矩陣的1/(2 lambda_max))。排除捨入誤差後{lr["common_rate_changed_days"]}日步長改变，最低為原值的{lr["minimum_ratio"]:.6f}，比值中位數{lr["median_ratio"]:.6f}。此決策只看已到期批次特徵，在績效計算前完成。','', '七組同日learning rate完全相同；與v25或v29的數值不可直接當作只改財報項，因為共同排程也不同。本輪主對照使用同排程的g。','','## 診斷發現','']
    corr=pd.read_csv(R/'feature_correlations.csv');contr=pd.read_csv(R/'prediction_contributions.csv');scale=pd.read_csv(R/'feature_scales.csv');cs=pd.read_csv(R/'coefficient_stability.csv');gd=pd.read_csv(R/'g_prediction_drift.csv')
    lines+=['### R與F的關聯','','事件附近指距最近有效R事件或F狀態更新0至2交易日；不改變任何模型的訓練股票池。OppositeSignFraction的分母為該樣本全部列，含0列時需留意。','','| 樣本 | 股票日 | Pearson | Spearman |','|---|---:|---:|---:|']
    for q in corr.itertuples():lines.append(f'| {q.Sample} | {q.StockDays:,} | {q.Pearson:+.6f} | {q.Spearman:+.6f} |')
    lines+=['','全validation的Pearson約0.0052、Spearman約0.0995，未見高度整體相關；事件附近與非零樣本也相近。這不證明條件獨立，但不支持「兩欄幾乎是同一份資訊」作為已確認解釋。']
    lines+=['','### 尺度與預測貢獻','','| 訊號 | 最大絕對值 | 非零絕對值中位數 | 非零絕對值99分位 |','|---|---:|---:|---:|']
    for q in scale.itertuples():lines.append(f'| {q.Feature} | {q.MaxAbsolute:.6f} | {q.NonzeroMedianAbsolute:.6g} | {q.NonzeroP99Absolute:.6f} |')
    lines+=['','抵銷率=sum(abs(R貢獻)+abs(F貢獻)-abs(兩者和))/sum(abs(R貢獻)+abs(F貢獻))；它衡量預測貢獻抵銷，並非抵銷就代表壞事。','','| 模型 | 平均絕對R貢獻 | 平均絕對F貢獻 | 合計抵銷率 |','|---|---:|---:|---:|']
    for q in contr.itertuples():lines.append(f'| {LABELS[q.Variant]} | {q.MeanAbsR:.8f} | {q.MeanAbsF:.8f} | {q.AggregateCancellationFraction:.2%} |')
    lines+=['','共同訓練的R係數正負切換為單項148次、聯合146次，F為單項145次、聯合145次，沒有看到合併後明顯增加的切換。聯合貢獻約7.46%抵銷；有抵銷不等於損失資訊或造成虧損，尤其當它修正另一項的誤差時。']
    lines+=['','係數正負切換、每次梯度、參數更新尺度、g預測漂移另附coefficient_stability.csv、financial_gradient_scales.csv及g_prediction_drift.csv。未經標準化的原始係數大小不能直接當成特徵重要性。','','### 公布季度對齊', '',f'來源事件全期間（非僅validation）的R事件{diag["R_events"]:,}筆；同公告來源列可配到有效F狀態的有{diag["valid_F_updates_matching_R_events"]:,}筆，其目標季度分布為{diag["valid_F_event_relations"]}。逐筆匹配保存在announcement_quarter_alignment.csv。','', 'F的剩餘單季估計=(全年Forecast-已知累計實績)/(4-已知季數)。固定全年Forecast時，累計實績提高會機械性降低剩餘單季估計；因此實際R與未來季度F的反向變動不必然代表市場看法衝突。季度不同，不能把兩欄直接相減叫作當季surprise。此處只辨認資料公式可能產生的關係，沒有量化它對Sharpe的因果效果。','','## 分期Sharpe','','沿用既有ValidationYear標籤。','','| 模型 | 2018 | 2019 | 2020 | 2021 |','|---|---:|---:|---:|---:|']
    for n in LABELS:
        a=ad[ad.Variant.eq(n)].set_index('ValidationYear');lines.append('| '+LABELS[n]+' | '+' | '.join(f'{a.loc[y,"Sharpe"]:+.6f}' for y in [2018,2019,2020,2021])+' |')
    lines+=['','## 配對區間','','20交易日區塊、4000次同日期配對重抽樣，种子20260916；探索性95%區間，未校正多次選擇，不作test結果或本輪選擇門檻。','','| 比較 | 指標 | 差值 | 95%區間 |','|---|---|---:|---:|']
    for q in ci:lines.append(f'| {q["Comparison"]} | {q["Metric"]} | {q["Difference"]:+.8f} | [{q["CI95Low"]:+.8f}, {q["CI95High"]:+.8f}] |')
    lines+=['','## 核對與限制','','七組各1199次更新、2,325,806訓練股票日、2,326,022筆預測，極端值排除0筆。全部更新、梯度、loss、預測、排名、Rank IC與官方Sharpe核對通過；R全部列由事件獨立重建、F全部列由最近狀態重建、固定g與貢獻分解核對通過，歷史來源雜湊未改。','','固定g與共同訓練的差異包含殘差目標、參數適應方式與g更新方式；不能只憑點估計就斷言g被破壞。相關係數、係數正負切換與貢獻抵銷都是診斷線索，並非金融因果證據。這輪未執行標準化、正交化或surprise替代實驗。']
    (R/'JPX-v30-profit-actual-forecast-report.md').write_text(('\n'.join(lines)+'\n').replace('改变','改變').replace('种子','種子'))
    version=dict(version='v30',description='Seven matched-rate joint/frozen-g models of operating-profit actual YoY and latest forecast YoY, with timing/scale/contribution diagnostics',directory=str(R),status='completed_experiment',variants=list(LABELS),audit_passed=True,test_evaluated=False,validation_winner=winner.Variant)
    save('version.json',version);reg=json.loads((B/'jpx_model_versions.json').read_text());assert reg['active_version']=='v7';reg['versions']=[v for v in reg['versions'] if v.get('version')!='v30']+[version];reg['latest_experiment']='v30';reg['latest_validation_candidate']=dict(version='v30',variant=winner.Variant,Sharpe=float(winner.Sharpe),RankIC=float(winner.RankIC),test_evaluated=False,status='validation_candidate_only');(B/'jpx_model_versions.json').write_text(json.dumps(reg,ensure_ascii=False,indent=2))
    print(df.to_string(index=False));print(pd.DataFrame(ci).to_string(index=False));print(json.dumps(diag,ensure_ascii=False));print(ad.to_string(index=False))
if __name__=='__main__':main()
