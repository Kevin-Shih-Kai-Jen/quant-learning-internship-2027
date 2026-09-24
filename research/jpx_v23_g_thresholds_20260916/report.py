from pathlib import Path
import json
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent;B=R.parent
ORDER=['no_skip','skip100','skip10'];LABELS={'no_skip':'不跳過','skip100':'超過100倍跳過','skip10':'超過10倍跳過'}
def sharp(a):return float(np.mean(a)/np.std(a,ddof=1))
def save(name,v):(R/name).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))
def csv(name,df):df.to_csv(R/name,index=False,encoding='utf-8-sig')
def main():
    assert json.loads((R/'audit.json').read_text())['passed']
    frames={};rows=[];annual=[];rates=[]
    for family,path in [('g',R),('eps',B/'jpx_v22_eps_thresholds_20260916')]:
        for n in ORDER:
            d=pd.read_csv(path/n/'daily_metrics.csv');u=pd.read_csv(path/n/'training_updates.csv');key=family+'_'+n;frames[key]=d
            assert len(d)==953 and d.SelectedMissingTargets.sum()==0 and d.RankIC.notna().sum()==952
            pd.testing.assert_series_equal(d.Date,frames['g_no_skip'].Date)
            assert d.loc[d.RankIC.isna(),'Date'].tolist()==['2020-09-29']
            rows.append(dict(Model=key,Family=family,Threshold=n,Label='不跳過（原v7）' if family=='g' and n=='no_skip' else LABELS[n],Sharpe=sharp(d.OfficialDailySpread),RankIC=float(d.RankIC.mean()),MSE=float(d.AllStockForecastMSE.mean()),SkippedTrainingStockDays=int(u.ThresholdSkippedStocks.sum()),TrainingStockDays=int(u.TrainingStocks.sum()),MedianLearningRate=float(u.LearningRate.median())))
            for year,g in d.groupby('ValidationYear'):
                annual.append(dict(Model=key,Family=family,Threshold=n,ValidationYear=int(year),Days=len(g),Sharpe=sharp(g.OfficialDailySpread),RankIC=float(g.RankIC.mean()),MSE=float(g.AllStockForecastMSE.mean())))
    df=pd.DataFrame(rows);ad=pd.DataFrame(annual);by=df.set_index('Model')
    for n in ORDER:
        gu=pd.read_csv(R/n/'training_updates.csv');eu=pd.read_csv(B/'jpx_v22_eps_thresholds_20260916'/n/'training_updates.csv');ratio=eu.LearningRate/gu.LearningRate
        rates.append(dict(Threshold=n,MedianEPSvsGStepRatio=float(ratio.median()),FractionEPSRateBelowHalf=float((ratio<.5).mean())))
    rng=np.random.default_rng(20260916);N=953;block=20;reps=4000
    ix=(rng.integers(0,N-block+1,size=(reps,int(np.ceil(N/block))))[:,:,None]+np.arange(block)).reshape(reps,-1)[:,:N]
    boot={}
    for n,d in frames.items():
        spread=d.OfficialDailySpread.to_numpy()[ix];ic=d.RankIC.to_numpy()[ix]
        boot[n]={'Sharpe':spread.mean(axis=1)/spread.std(axis=1,ddof=1),'RankIC':np.nanmean(ic,axis=1)}
    pairs=[('g_skip100','g_no_skip'),('g_skip10','g_no_skip'),('g_skip10','g_skip100')]+[('eps_'+n,'g_'+n) for n in ORDER]
    ci=[]
    for a,b in pairs:
        for metric in ['Sharpe','RankIC']:
            lo,hi=np.quantile(boot[a][metric]-boot[b][metric],[.025,.975]);ci.append(dict(Comparison=a+' - '+b,Metric=metric,Difference=float(by.loc[a,metric]-by.loc[b,metric]),CI95Low=float(lo),CI95High=float(hi)))
    levels=[]
    for n in frames:
        lo,hi=np.quantile(boot[n]['RankIC'],[.025,.975]);levels.append(dict(Model=n,RankIC=float(by.loc[n,'RankIC']),CI95Low=float(lo),CI95High=float(hi)))
    increments=[]
    for n in ORDER:
        increments.append(dict(Threshold=n,Label=LABELS[n],GSharpe=float(by.loc['g_'+n,'Sharpe']),EPSSharpe=float(by.loc['eps_'+n,'Sharpe']),DeltaSharpe=float(by.loc['eps_'+n,'Sharpe']-by.loc['g_'+n,'Sharpe']),GRankIC=float(by.loc['g_'+n,'RankIC']),EPSRankIC=float(by.loc['eps_'+n,'RankIC']),DeltaRankIC=float(by.loc['eps_'+n,'RankIC']-by.loc['g_'+n,'RankIC'])))
    csv('comparison.csv',df);csv('comparison_by_year.csv',ad);csv('paired_block_bootstrap.csv',pd.DataFrame(ci));csv('rank_ic_level_intervals.csv',pd.DataFrame(levels));csv('eps_vs_g_same_mask.csv',pd.DataFrame(increments));csv('eps_vs_g_learning_rates.csv',pd.DataFrame(rates));csv('all_models_daily.csv',pd.concat([d.assign(Model=n) for n,d in frames.items()]))
    result=dict(version='v23',audit_passed=True,comparison=rows,paired_bootstrap=ci,rank_ic_intervals=levels,eps_vs_g= increments,learning_rates=rates,rank_ic_days=952,sharpe_days=953,excluded_rank_ic_date='2020-09-29',active_baseline='v7')
    save('results.json',result)
    lines=['# v23：純價量g的EPS訓練樣本篩選實驗','','本輪只訓練正式v7的12個價量參數，EPS只用來決定哪些股票日跳過訓練，不進入預測、梯度或學習率矩陣。門檻沿用v22六項衰減累加後EPS訊號的最大絕對值，不是價量特徵或Target的門檻。','','## 主要結果','','同一完整股票池、953個驗證日（2017-12-29至2021-12-01），Sharpe未年化未扣成本。Rank IC是每日股票橫斷面Spearman相關的平均，共952日；2020-09-29全部Target相同，Rank IC未定義，三組皆排除該日而不補0。','','| 純g訓練規則 | Sharpe | 平均Rank IC | MSE（診斷） | 跳過訓練股票日 |','|---|---:|---:|---:|---:|']
    for q in rows[:3]:lines.append(f'| {q["Label"]} | {q["Sharpe"]:+.8f} | {q["RankIC"]:+.8f} | {q["MSE"]:.9f} | {q["SkippedTrainingStockDays"]:,} |')
    lines+=['','純g在不跳過時Sharpe點估計最高；100倍與10倍篩選都降低這次Sharpe。平均Rank IC僅極小幅改善、仍為負，且差值與各模型Rank IC對0的區間均跨0，因此尚未看到篩選穩定改善兩個主要指標的證據。MSE僅有很小變化。']
    lines+=['','## 與加入EPS的模型配對','','下表引用v22，兩邊的訓練股票日逐日完全相同；各自按模型設計矩陣算步長，所以實際步長不完全相同。差值是整套模型結果，不能視為排除最佳化差異後的純EPS資訊效應。','','| 門檻 | 純g Sharpe | g＋EPS Sharpe | ΔSharpe | 純g Rank IC | g＋EPS Rank IC | ΔRank IC |','|---|---:|---:|---:|---:|---:|---:|']
    for q in increments:lines.append('| '+q['Label']+' | '+' | '.join(f'{q[c]:+.8f}' for c in ['GSharpe','EPSSharpe','DeltaSharpe','GRankIC','EPSRankIC','DeltaRankIC'])+' |')
    lines+=['','## 純g分期結果','','分期沿用既有validation calendar，非嚴格曆年起訖。','','| 分期 | 不跳過Sharpe | 100倍Sharpe | 10倍Sharpe | 不跳過Rank IC | 100倍Rank IC | 10倍Rank IC |','|---|---:|---:|---:|---:|---:|---:|']
    for year in [2018,2019,2020,2021]:
        a=ad[(ad.Family=='g')&(ad.ValidationYear==year)].set_index('Threshold');lines.append('| '+str(year)+' | '+' | '.join(f'{a.loc[n,m]:+.6f}' for m in ['Sharpe','RankIC'] for n in ORDER)+' |')
    lines+=['','## 差值不確定性','','20日連續區塊、4000次配對重抽樣、種子20260916；同一日期區塊抽取所有模型。Rank IC未定義值不補0，區塊樣本以其有限值計算平均。探索性95%百分位區間，未校正多重比較與過去模型選擇，不是新保留測試集。','','| 比較（前者減後者） | 指標 | 差值 | 95%區間 |','|---|---|---:|---:|']
    for q in ci:lines.append(f'| {q["Comparison"]} | {q["Metric"]} | {q["Difference"]:+.8f} | [{q["CI95Low"]:+.8f}, {q["CI95High"]:+.8f}] |')
    lines+=['','## 平均Rank IC相對0','','全0預測的Rank IC未定義；這裡測的是各模型平均Rank IC相對無排序相關的0。','','| 模型 | 平均Rank IC | 95%區間 |','|---|---:|---:|']
    for q in levels:lines.append(f'| {q["Model"]} | {q["RankIC"]:+.8f} | [{q["CI95Low"]:+.8f}, {q["CI95High"]:+.8f}] |')
    lines+=['','## 訓練與核對','','- PR1/VR1原值，普通MSE每日一次更新，只用到期Target；從零初始化，跨年連續，共1199次更新。','- 六項EPS篩選訊號及exp(-交易日齡/9)沿用既有快取；10倍即1000%，100倍即10000%，嚴格大於才排除。','- 每日步長1/(2×12維訓練Gram最大特徵值)。篩選可同時改變樣本組成與實際步長。','- 不跳過組的預測在浮點容差內重現原v7且排名完全一致；100倍組精確重現v21 price_mask。','- 三組所有預測、排名、更新、Rank IC、MSE及官方Sharpe獨立核對通過；與v22同門檻逐日訓練股票數相同，來源快取雜湊未變。','- 正式基準保持v7；這次不自動切換門檻，不據單次正負差值宣稱EPS永久有用或無用。']
    (R/'JPX-v23-g-threshold-report.md').write_text('\n'.join(lines)+'\n')
    version=dict(version='v23',description='Pure v7 g with EPS-based training masks: 10, 100, none; Sharpe and Rank IC evaluation',directory=str(R),status='completed_experiment',variants=ORDER,baseline='v7',parameter_count=12,common_days=953,audit_passed=True)
    save('version.json',version);reg=json.loads((B/'jpx_model_versions.json').read_text());assert reg['active_version']=='v7';reg['versions']=[v for v in reg['versions'] if v.get('version')!='v23']+[version];reg['latest_experiment']='v23';(B/'jpx_model_versions.json').write_text(json.dumps(reg,ensure_ascii=False,indent=2))
    print(df.to_string(index=False));print(pd.DataFrame(ci).to_string(index=False));print(pd.DataFrame(levels).to_string(index=False));print(pd.DataFrame(increments).to_string(index=False))
if __name__=='__main__':main()
