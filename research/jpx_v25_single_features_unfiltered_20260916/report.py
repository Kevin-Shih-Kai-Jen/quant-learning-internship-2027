from pathlib import Path
import json
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent;B=R.parent
METRICS={'NetSales':'營收','OperatingProfit':'營業利益','EPS':'EPS'}
PARTS={'ActualGrowth':'實際年增','ExpectedGrowth':'實績公布時事前預期成長','ForecastQoQ':'預測對前季預測','ForecastYoY':'預測對去年同季預測','Revision':'舊版修正（去年實績基期）','ForecastActualQoQ':'預測對前季實績','ForecastActualYoY':'預測對去年同季實績','RevisionRelative':'修正相對舊單季預測','ActualQoQ':'實際季增'}
def label(feature):
    for m,v in METRICS.items():
        if feature.startswith(m):return v+'／'+PARTS[feature[len(m):]]
def save(name,v):(R/name).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))
def csv(name,d):d.to_csv(R/name,index=False,encoding='utf-8-sig')
def sharp(a):return float(np.mean(a)/np.std(a,ddof=1))
def main():
    assert json.loads((R/'audit.json').read_text())['passed']
    catalog=json.loads((R/'catalog.json').read_text());base=pd.read_csv(B/'jpx_v16_single_financial_v7_20260915/price_only/daily_metrics.csv');bs=sharp(base.OfficialDailySpread);bi=float(base.RankIC.mean());bu=pd.read_csv(B/'jpx_v23_g_thresholds_20260916/no_skip/training_updates.csv')
    rng=np.random.default_rng(20260916);N=953;block=20;reps=4000;ix=(rng.integers(0,N-block+1,size=(reps,int(np.ceil(N/block))))[:,:,None]+np.arange(block)).reshape(reps,-1)[:,:N]
    def bootstrap(d):
        a=d.OfficialDailySpread.to_numpy()[ix];return {'Sharpe':a.mean(axis=1)/a.std(axis=1,ddof=1),'RankIC':np.nanmean(d.RankIC.to_numpy()[ix],axis=1)}
    bb=bootstrap(base);rows=[];annual=[];intervals=[];coef=[];daily=[base.assign(Variant='v7')];defs=[]
    for item in catalog:
        name=item['name'];feature=item['feature'];d=pd.read_csv(R/name/'daily_metrics.csv');old=pd.read_csv(Path(item['source'])/'daily_metrics.csv');rr=json.loads((R/name/'results.json').read_text());u=pd.read_csv(R/name/'training_updates.csv');pd.testing.assert_series_equal(d.Date,base.Date);pd.testing.assert_series_equal(old.Date,base.Date)
        for c in ['Date','SignalDate','ExitDate','KnownLabelStocks','TrainingStocks']:pd.testing.assert_series_equal(u[c],bu[c])
        assert len(d)==953 and d.RankIC.notna().sum()==952 and u.ThresholdSkippedStocks.sum()==0 and d.SelectedMissingTargets.sum()==0
        s=sharp(d.OfficialDailySpread);ic=float(d.RankIC.mean());os=sharp(old.OfficialDailySpread);oi=float(old.RankIC.mean());oldstep=pd.read_csv(Path(item['source'])/'training_updates.csv');ratio=u.LearningRate/oldstep.LearningRate
        rows.append(dict(Variant=name,Feature=feature,Label=label(feature),Status=item['status'],Reused=item['reuse'],OldSkippedStockDays=item['old_skipped'],NewSkippedStockDays=0,OldSharpe=os,Sharpe=s,DeltaVsOld=s-os,DeltaVsG=s-bs,OldRankIC=oi,RankIC=ic,DeltaRankICVsG=ic-bi,MSE=float(d.AllStockForecastMSE.mean()),OldMSE=float(old.AllStockForecastMSE.mean()),SharpeAboveG=s>bs,RankICAboveZero=ic>0,MeetsPointEstimateCriteria=s>bs and ic>0,MedianRateRatioVsOld=float(ratio.median()),FractionRateBelowTenthOfOld=float((ratio<.1).mean()),MinRateRatioVsOld=float(ratio.min())))
        for year,g in d.groupby('ValidationYear'):
            bg=base[base.ValidationYear.eq(year)];og=old[old.ValidationYear.eq(year)];annual.append(dict(Variant=name,Feature=feature,Label=label(feature),Status=item['status'],ValidationYear=int(year),Sharpe=sharp(g.OfficialDailySpread),OldSharpe=sharp(og.OfficialDailySpread),GSharpe=sharp(bg.OfficialDailySpread),RankIC=float(g.RankIC.mean()),GRankIC=float(bg.RankIC.mean()),MSE=float(g.AllStockForecastMSE.mean())))
        nb=bootstrap(d);ob=bootstrap(old)
        for m,newpoint,oldpoint,basepoint in [('Sharpe',s,os,bs),('RankIC',ic,oi,bi)]:
            for ref,refboot,pointref in [('g',bb[m],basepoint),('old100',ob[m],oldpoint)]:
                lo,hi=np.quantile(nb[m]-refboot,[.025,.975]);intervals.append(dict(Variant=name,Feature=feature,Metric=m,Reference=ref,Difference=newpoint-pointref,CI95Low=float(lo),CI95High=float(hi)))
        lo,hi=np.quantile(nb['RankIC'],[.025,.975]);intervals.append(dict(Variant=name,Feature=feature,Metric='RankIC',Reference='zero',Difference=ic,CI95Low=float(lo),CI95High=float(hi)))
        for p,value in rr['final_coefficients'].items():coef.append(dict(Variant=name,Parameter=p,Value=value,AsOf=rr['parameter_asof']))
        daily.append(d.assign(Variant=name));defs.append(dict(Variant=name,Feature=feature,Label=label(feature),Status=item['status'],SourceCache=item['cache'],PreviousModel=item['source'],Definition='Formula and event timing unchanged from previous model'))
    df=pd.DataFrame(rows);ad=pd.DataFrame(annual);ci=pd.DataFrame(intervals);current=df[df.Status.eq('current')].sort_values('Sharpe',ascending=False);legacy=df[df.Status.eq('legacy_replaced')];eps=current[current.Feature.str.startswith('EPS')]
    for name,d in [('comparison_all.csv',df),('comparison_current.csv',current),('comparison_legacy.csv',legacy),('comparison_eps.csv',eps),('comparison_by_year.csv',ad),('paired_block_bootstrap.csv',ci),('all_coefficients.csv',pd.DataFrame(coef)),('all_models_daily.csv',pd.concat(daily)),('feature_catalog.csv',pd.DataFrame(defs))]:csv(name,d)
    summary=dict(version='v25',audit_passed=True,retrained=18,reused=3,current_features=len(current),legacy_features=len(legacy),current_sharpe_above_g=int(current.SharpeAboveG.sum()),current_rank_ic_above_zero=int(current.RankICAboveZero.sum()),current_meets_point_criteria=int(current.MeetsPointEstimateCriteria.sum()),g_sharpe=bs,g_rank_ic=bi,g_mse=float(base.AllStockForecastMSE.mean()),training_threshold=None,validation_threshold=None,comparison=rows)
    save('results.json',summary)
    lines=['# v25：單指標＋g取消極端值排除','','使用者指定後續訓練不跳過極端值；已記錄至JPX-experiment-defaults.json與專案基準文件。v16原15項、v17修正版6項共21個單指標版本；18項曾實際排除訓練樣本，已重訓；3項原本未排除任何樣本，已核對並沿用。','','目前定義16項，另外5項為已被修正版取代的歷史對照，分開呈現，不重新納入現行模型。純g、g＋單一財報都使用完整訓練與驗證股票池，不裁切大值。EPS算法、事件有效性與缺值規則原樣保留。','','## 主要結果','',f'純g基準Sharpe {bs:+.8f}，平均Rank IC {bi:+.8f}。953個驗證日，2017-12-29至2021-12-01；Rank IC有效952日；Sharpe未年化未扣成本。現行16項中，{int(current.SharpeAboveG.sum())}項Sharpe高於g，{int(current.RankICAboveZero.sum())}項平均Rank IC大於0，{int(current.MeetsPointEstimateCriteria.sum())}項同時滿足兩個點估計條件。','', '| 現行單項特徵＋g | 舊100倍Sharpe | 不排除Sharpe | Δ對g | 不排除Rank IC | 舊跳過股票日 |','|---|---:|---:|---:|---:|---:|']
    for q in current.itertuples():lines.append(f'| {q.Label} | {q.OldSharpe:+.8f} | {q.Sharpe:+.8f} | {q.DeltaVsG:+.8f} | {q.RankIC:+.8f} | {q.OldSkippedStockDays:,} |')
    lines+=['','## 六項現行EPS','','| EPS特徵 | 舊100倍Sharpe | 不排除Sharpe | 不排除Rank IC |','|---|---:|---:|---:|']
    for q in eps.itertuples():lines.append(f'| {q.Label} | {q.OldSharpe:+.8f} | {q.Sharpe:+.8f} | {q.RankIC:+.8f} |')
    lines+=['','六項EPS在取消門檻後，Sharpe點估計均變成負值。這表示原先單項EPS結果對極端值訓練規則敏感；不能直接推論EPS資訊普遍無用。門檻同時影響樣本與Gram矩陣決定的每日步長，並未隔離兩者作用。','', '## 不確定性','','所有區間由相同日期配對、20日連續區塊、4000次重抽樣（種子20260916）取得。為探索性95%百分位區間，未校正多重比較與過去模型選擇；不是新保留測試集。當前最高Sharpe不是已驗證的勝者。','','| 現行特徵 | ΔSharpe對g的95%區間 | 平均Rank IC的95%區間 |','|---|---|---|']
    for q in current.itertuples():
        a=ci[(ci.Variant==q.Variant)&(ci.Metric=='Sharpe')&(ci.Reference=='g')].iloc[0];b=ci[(ci.Variant==q.Variant)&(ci.Metric=='RankIC')&(ci.Reference=='zero')].iloc[0];lines.append(f'| {q.Label} | [{a.CI95Low:+.6f}, {a.CI95High:+.6f}] | [{b.CI95Low:+.6f}, {b.CI95High:+.6f}] |')
    lines+=['','## 已被替代的歷史公式','','下列只為完整重現曾做過的單項實驗，不能與修正版混為同一特徵，亦不作現行候選。旧修正使用去年同季實績作基期，現行修正使用舊單季預測；舊EPS預測對預測已由預測對實績取代。','','| 舊公式 | 舊100倍Sharpe | 不排除Sharpe | 不排除Rank IC |','|---|---:|---:|---:|']
    for q in legacy.itertuples():lines.append(f'| {q.Label} | {q.OldSharpe:+.8f} | {q.Sharpe:+.8f} | {q.RankIC:+.8f} |')
    lines+=['','## 設定與核對','','- 每組12個價量參數＋1個財報係數共同更新；從零初始化、跨年連續、只用已到期Target、每日一次普通MSE；PR1/VR1原值，財報exp(-交易日齡/9)。','- 每組1199次更新、2,325,806個有Target的訓練股票日；因極端值跳過0筆。驗證完整股票池，多空各200檔，952有效Rank IC日，不補全體Target相同日的IC為0。','- 沿用3項：營收預測對前季預測、營收預測對去年同季預測、舊版營收修正。它們原本就沒有樣本觸發100倍門檻，因此無需重算；輸出逐位元核對相同。','- 已重訓18項的全部梯度、日步長與參數更新在訓練時獨立核對；21項所有預測、排名、更新時序、Rank IC、MSE與官方Sharpe再核對通過。','- 原始快取與历史結果雜湊未變。源事件缺失／零分母／會計基礎有效性規則未改，只取消依數值大小排除，不擅自填補無法定義的成長率。','- 正式基準維持v7，後續預設不排除極端值。比較的是單項特徵搭配原學習率規則的完整模型，不是固定g的額外係數實驗。','', '## 檔案','','comparison_current.csv現行16項；comparison_eps.csv六項EPS；comparison_legacy.csv已替代5項；comparison_all.csv完整21項。comparison_by_year.csv分期、paired_block_bootstrap.csv相對g／舊版及Rank IC對0區間、all_models_daily.csv逐日結果、all_coefficients.csv期末係數。各模型另存完整預測、參數與更新。']
    (R/'JPX-v25-unfiltered-single-features-report.md').write_text('\n'.join(lines)+'\n')
    version=dict(version='v25',description='Rerun prior single financial feature plus v7 g without extreme-value exclusion; 18 retrained, 3 unchanged reused',directory=str(R),status='completed_experiment',retrained_models=18,reused_models=3,current_features=16,legacy_features=5,training_financial_abs_limit=None,evaluation_stock_pool_filtered=False,audit_passed=True)
    save('version.json',version);reg=json.loads((B/'jpx_model_versions.json').read_text());assert reg['active_version']=='v7';reg['versions']=[v for v in reg['versions'] if v.get('version')!='v25']+[version];reg['latest_experiment']='v25';(B/'jpx_model_versions.json').write_text(json.dumps(reg,ensure_ascii=False,indent=2))
    print(current[['Label','OldSharpe','Sharpe','RankIC','DeltaVsG']].to_string(index=False));print('EPS');print(eps[['Label','OldSharpe','Sharpe','RankIC']].to_string(index=False));print('Counts',summary['current_sharpe_above_g'],summary['current_rank_ic_above_zero']);print(ci[(ci.Reference=='g')&(ci.Metric=='Sharpe')].to_string(index=False))
if __name__=='__main__':main()
