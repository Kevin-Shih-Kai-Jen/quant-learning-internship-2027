from pathlib import Path
import json
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent;B=R.parent;V23=B/'jpx_v23_g_thresholds_20260916'
ORDER=['none_none','100_100','10_10','none_100','none_10']
LABELS={'none_none':('不跳過','不跳過'),'100_100':('100倍','100倍'),'10_10':('10倍','10倍'),'none_100':('不跳過','100倍'),'none_10':('不跳過','10倍')}
def sharp(a):return float(np.mean(a)/np.std(a,ddof=1))
def save(name,v):(R/name).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))
def csv(name,d):d.to_csv(R/name,index=False,encoding='utf-8-sig')
def main():
    assert json.loads((R/'audit.json').read_text())['passed']
    rr=json.loads((R/'run_results.json').read_text());frames={};rows=[];annual=[]
    for n in ORDER:
        d=pd.read_csv(R/n/'daily_metrics.csv');frames[n]=d
        pd.testing.assert_series_equal(d.Date,frames['none_none'].Date)
        q=rr[n];rows.append(dict(Variant=n,TrainingFilter=LABELS[n][0],ValidationFilter=LABELS[n][1],Sharpe=q['sharpe'],RankIC=q['rank_ic'],MSE=q['mean_mse'],ValidationExcludedStockDays=q['excluded_stock_days'],ValidationExcludedPct=q['excluded_pct'],ValidationRetainedStockDays=q['retained_stock_days'],MinDailyStocks=q['min_daily_stocks'],TrainingSkippedStockDays=q['training_skipped_stock_days'],ReplacedLongStockDays=q['old_long_excluded_total'],ReplacedShortStockDays=q['old_short_excluded_total']))
        for year,g in d.groupby('ValidationYear'):annual.append(dict(Variant=n,ValidationYear=int(year),Days=len(g),Sharpe=sharp(g.OfficialDailySpread),RankIC=float(g.RankIC.mean()),MSE=float(g.AllStockForecastMSE.mean()),ExcludedStocks=int(g.ValidationExcludedStocks.sum())))
    for n,source in [('100_none','skip100'),('10_none','skip10')]:frames[n]=pd.read_csv(V23/source/'daily_metrics.csv');pd.testing.assert_series_equal(frames[n].Date,frames['none_none'].Date)
    df=pd.DataFrame(rows);ad=pd.DataFrame(annual);csv('comparison.csv',df);csv('comparison_by_year.csv',ad);csv('all_models_daily.csv',pd.concat([d.assign(Variant=n) for n,d in frames.items()]))
    rng=np.random.default_rng(20260916);N=953;block=20;reps=4000
    ix=(rng.integers(0,N-block+1,size=(reps,int(np.ceil(N/block))))[:,:,None]+np.arange(block)).reshape(reps,-1)[:,:N]
    boot={};point={}
    for n,d in frames.items():
        a=d.OfficialDailySpread.to_numpy()[ix];boot[n]={'Sharpe':a.mean(axis=1)/a.std(axis=1,ddof=1),'RankIC':np.nanmean(d.RankIC.to_numpy()[ix],axis=1)};point[n]={'Sharpe':sharp(d.OfficialDailySpread),'RankIC':float(d.RankIC.mean())}
    pairs=[('100_100','none_none'),('10_10','none_none'),('10_10','100_100'),('100_100','100_none'),('10_10','10_none'),('100_100','none_100'),('10_10','none_10'),('none_100','none_none'),('none_10','none_none')]
    ci=[]
    for a,b in pairs:
        for m in ['Sharpe','RankIC']:
            lo,hi=np.quantile(boot[a][m]-boot[b][m],[.025,.975]);ci.append(dict(Comparison=a+' - '+b,Metric=m,Difference=point[a][m]-point[b][m],CI95Low=float(lo),CI95High=float(hi)))
    levels=[]
    for n in ORDER:
        lo,hi=np.quantile(boot[n]['RankIC'],[.025,.975]);levels.append(dict(Variant=n,RankIC=point[n]['RankIC'],CI95Low=float(lo),CI95High=float(hi)))
    stages=[]
    for n,previous in [('100_100','100_none'),('10_10','10_none')]:
        stages.append(dict(Threshold=LABELS[n][0],PreviousSharpe=point[previous]['Sharpe'],FilteredSharpe=point[n]['Sharpe'],DeltaSharpe=point[n]['Sharpe']-point[previous]['Sharpe'],PreviousRankIC=point[previous]['RankIC'],FilteredRankIC=point[n]['RankIC'],DeltaRankIC=point[n]['RankIC']-point[previous]['RankIC']))
    csv('paired_block_bootstrap.csv',pd.DataFrame(ci));csv('rank_ic_level_intervals.csv',pd.DataFrame(levels));csv('validation_filter_increment.csv',pd.DataFrame(stages))
    result=dict(version='v24',audit_passed=True,comparison=rows,paired_bootstrap=ci,rank_ic_intervals=levels,validation_filter_increment=stages,training_reused_unchanged=True,active_baseline='v7')
    save('results.json',result)
    lines=['# v24：純g的訓練與驗證同步篩選','','純價量v7，12個價量參數。沿用v23各門檻已訓練的每日預測；六項EPS只作股票日篩選，未放入模型。本輪變動僅是驗證股票池與排除後重新排名、選股。','','## 主實驗','','相同953個驗證日，2017-12-29至2021-12-01，Sharpe未年化、未扣成本。各組驗證股票池不同。','','| 訓練跳過 | 驗證跳過 | Sharpe | 平均Rank IC | 驗證排除股票日 | 排除占比 |','|---|---|---:|---:|---:|---:|']
    for q in rows[:3]:lines.append(f'| {q["TrainingFilter"]} | {q["ValidationFilter"]} | {q["Sharpe"]:+.8f} | {q["RankIC"]:+.8f} | {q["ValidationExcludedStockDays"]:,} | {q["ValidationExcludedPct"]:.3f}% |')
    lines+=['','Rank IC是各日剩餘股票的橫斷面Spearman相關再取平均，有效952日。2020-09-29真實Target全部相同，Rank IC未定義，不補0。MSE留在CSV作診斷；不同股票池下不可把MSE或Rank IC變化全部解讀為模型能力改變。','','## 增加驗證篩選的效果','','固定各門檻原本的訓練結果，只比較v23完整股票池驗證與本輪篩選後驗證。','','| 訓練門檻 | 前輪Sharpe | 本輪Sharpe | 差值 | 前輪Rank IC | 本輪Rank IC |','|---|---:|---:|---:|---:|---:|']
    for q in stages:lines.append('| '+q['Threshold']+' | '+' | '.join(f'{q[c]:+.8f}' for c in ['PreviousSharpe','FilteredSharpe','DeltaSharpe','PreviousRankIC','FilteredRankIC'])+' |')
    lines+=['','## 相同驗證股票池的訓練對照','','原v7訓練不跳過，只在驗證選股前套用相同門檻，與主實驗配對。這有助分辨改變驗證股票池與訓練篩選的效果；訓練篩選仍可能改變實際學習率。','','| 訓練跳過 | 驗證跳過 | Sharpe | 平均Rank IC |','|---|---|---:|---:|']
    for n in ['none_100','100_100','none_10','10_10']:
        q=df.set_index('Variant').loc[n];lines.append(f'| {q.TrainingFilter} | {q.ValidationFilter} | {q.Sharpe:+.8f} | {q.RankIC:+.8f} |')
    lines+=['','## 選股規則與篩選時點','','- 門檻沿用六項當日已知、按exp(-交易日齡/9)衰減累加後EPS訊號；任一絕對值嚴格>100或>10即排除。10倍是1000%，100倍是10000%。','- 決定是否排除不使用未來收益或未來財報；這是事前可執行的股票池篩選。','- 先排除，再依保存的當日g預測重新排名。剩餘股票最高200檔做多、最低200檔做空，權重與原v7一致。不是建立原持股後刪除已知虧損樣本。','- 三組每日最少股票數分別1896、1884、1763，均足夠維持多空各200檔。','- 排除數是股票日，不是獨立股票數；驗證與訓練期間不同，因此排除筆數不必相同。','- 此為篩選股票池後的策略回測，雖沿用官方收益公式，但不等同在JPX完整指定股票池上提交全體排名的結果。','','## 分期主實驗','','| 分期 | 不跳過Sharpe | 100倍Sharpe | 10倍Sharpe | 不跳過Rank IC | 100倍Rank IC | 10倍Rank IC |','|---|---:|---:|---:|---:|---:|---:|']
    for year in [2018,2019,2020,2021]:
        a=ad[ad.ValidationYear.eq(year)].set_index('Variant');lines.append('| '+str(year)+' | '+' | '.join(f'{a.loc[n,m]:+.6f}' for m in ['Sharpe','RankIC'] for n in ORDER[:3])+' |')
    lines+=['','## 不確定性','','20日區塊、4000次同日配對重抽樣，種子20260916。探索性95%百分位區間，未校正多重比較與過去模型選擇，不是新保留測試集。命名為「訓練門檻_驗證門檻」，none代表不跳過。','','| 比較（前者減後者） | 指標 | 差值 | 95%區間 |','|---|---|---:|---:|']
    for q in ci:lines.append(f'| {q["Comparison"]} | {q["Metric"]} | {q["Difference"]:+.8f} | [{q["CI95Low"]:+.8f}, {q["CI95High"]:+.8f}] |')
    lines+=['','## Rank IC相對0','','| 組別 | 平均Rank IC | 95%區間 |','|---|---:|---:|']
    for q in levels:lines.append(f'| {q["Variant"]} | {q["RankIC"]:+.8f} | [{q["CI95Low"]:+.8f}, {q["CI95High"]:+.8f}] |')
    lines+=['','## 本輪結論','','依目前點估計，訓練與驗證都使用100倍或10倍門檻，Sharpe均低於原v7；Rank IC雖略往0改善，仍為負。100倍組加入驗證篩選後相對前輪略升，10倍組反而下降。所有已列Sharpe／Rank IC差值探索區間均跨0，各組Rank IC相對0區間也跨0，尚未看到穩定正向排序能力或可靠整體改善。','', '原v7只在驗證排除100倍時Sharpe為+0.00800069，比原始+0.00767887高約0.000322，但此差值區間同樣跨0，不據此選為新基準。']
    lines+=['','## 核對與重現','','未重新訓練。三組v23來源的預測、參數歷程、訓練更新與快取雜湊皆未變；每筆保留預測與當日參數重算一致。五個驗證情境逐日獨立核對篩選集合、排名、收益、Rank IC、MSE及官方公式Sharpe。原v7不跳過組完整重現。正式基準維持v7。','','comparison.csv主表含5情境；validation_filter_increment.csv為前後輪差異；all_models_daily.csv含完整逐日收益與排除、替補數；paired_block_bootstrap.csv與rank_ic_level_intervals.csv為區間；各組validation_selection.npz保留驗證股票集合與重排名（-1表示未評分），原始scores仍在v23保存。']
    (R/'JPX-v24-train-validation-filter-report.md').write_text('\n'.join(lines)+'\n')
    version=dict(version='v24',description='Pure g with matched training and validation EPS thresholds, plus same-pool unfiltered-training controls',directory=str(R),status='completed_experiment',variants=ORDER,baseline='v7',common_days=953,audit_passed=True,training_reused=True)
    save('version.json',version);reg=json.loads((B/'jpx_model_versions.json').read_text());assert reg['active_version']=='v7';reg['versions']=[v for v in reg['versions'] if v.get('version')!='v24']+[version];reg['latest_experiment']='v24';(B/'jpx_model_versions.json').write_text(json.dumps(reg,ensure_ascii=False,indent=2))
    print(df.to_string(index=False));print(pd.DataFrame(ci).to_string(index=False));print(pd.DataFrame(levels).to_string(index=False));print(pd.DataFrame(stages).to_string(index=False))
if __name__=='__main__':main()
