from pathlib import Path
import importlib.util,json,zipfile
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent;B=R.parent
spec=importlib.util.spec_from_file_location('v22',R/'run.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
LABELS={'skip100':'超過100倍跳過','skip10':'超過10倍跳過','no_skip':'完全不跳過'}
def sharp(a):return float(np.mean(a)/np.std(a,ddof=1))
def csv(df,name):df.to_csv(R/name,index=False,encoding='utf-8-sig')
def main():
    assert json.loads((R/'audit.json').read_text())['passed']
    f,x,z,groups,cal=r.load();target=f.Target.to_numpy();extreme=np.abs(z).max(axis=1)
    val=f.SignalDate.isin(cal.Date).to_numpy();known=val&np.isfinite(target)
    dailies={n:pd.read_csv(R/n/'daily_metrics.csv') for n in r.JOBS}
    updates={n:pd.read_csv(R/n/'training_updates.csv') for n in r.JOBS}
    zero=np.array([np.mean(target[groups[d]][np.isfinite(target[groups[d]])]**2) for d in cal.Date])
    rows=[];annual=[];strata=[];tails=[];rates=[]
    for n,d in dailies.items():
        assert len(d)==953 and d.SelectedMissingTargets.sum()==0
        pd.testing.assert_series_equal(d.Date,dailies['skip100'].Date)
        u=updates[n];result=json.loads((R/n/'results.json').read_text())
        with np.load(R/n/'predictions.npz') as p:score=p['score']
        error=(score[known]-target[known])**2;absolute=np.abs(score[val]);total=float(error.sum())
        row=dict(Variant=n,Label=LABELS[n],Sharpe=sharp(d.OfficialDailySpread),DailyMeanMSE=float(d.AllStockForecastMSE.mean()),PooledStockDayMSE=float(error.mean()),RootMeanDailyMSEPct=float(np.sqrt(d.AllStockForecastMSE.mean())*100),DeltaSharpeVs100=sharp(d.OfficialDailySpread)-sharp(dailies['skip100'].OfficialDailySpread),MSEChangeVs100Pct=float((d.AllStockForecastMSE.mean()/dailies['skip100'].AllStockForecastMSE.mean()-1)*100),DailyMSEP95=float(d.AllStockForecastMSE.quantile(.95)),DailyMSEMax=float(d.AllStockForecastMSE.max()),AbsPredictionP99=float(np.quantile(absolute,.99)),AbsPredictionP999=float(np.quantile(absolute,.999)),AbsPredictionMax=float(absolute.max()),PredictionsAbove100Pct=int((absolute>1).sum()),RankIC=float(d.RankIC.mean()),TrainingStockDays=int(u.TrainingStocks.sum()),SkippedTrainingStockDays=int(u.ThresholdSkippedStocks.sum()),SkipPct=float(u.ThresholdSkippedStocks.sum()/u.KnownLabelStocks.sum()*100),MedianLearningRate=float(u.LearningRate.median()),MSEImprovementVsZeroPct=float((1-d.AllStockForecastMSE.mean()/zero.mean())*100))
        rows.append(row)
        for year,g in d.groupby('ValidationYear'):
            annual.append(dict(Variant=n,ValidationYear=int(year),Days=len(g),Sharpe=sharp(g.OfficialDailySpread),DailyMeanMSE=float(g.AllStockForecastMSE.mean()),DailyMSEMax=float(g.AllStockForecastMSE.max())))
        for label,mask in [('abs<=10',extreme<=10),('10<abs<=100',(extreme>10)&(extreme<=100)),('abs>100',extreme>100)]:
            mask=mask&known;err=(score[mask]-target[mask])**2
            strata.append(dict(Variant=n,Stratum=label,StockDays=int(mask.sum()),SamplePct=float(mask.sum()/known.sum()*100),MSE=float(err.mean()),SquaredErrorSharePct=float(err.sum()/total*100),ZeroForecastMSE=float(np.mean(target[mask]**2))))
        ids=np.flatnonzero(known);ids=ids[np.argsort(-np.abs(score[ids]))[:10]]
        for i in ids:
            j=int(np.argmax(np.abs(z[i])))
            tails.append(dict(Variant=n,Date=str(f.SignalDate.iloc[i].date()),SecuritiesCode=int(f.SecuritiesCode.iloc[i]),Prediction=float(score[i]),Target=float(target[i]),MaxFeature=r.FINS[j],MaxFeatureValue=float(z[i,j]),SquaredError=float((score[i]-target[i])**2)))
        rate_ratio=u.LearningRate/updates['skip100'].LearningRate
        rates.append(dict(Variant=n,MedianRate=float(u.LearningRate.median()),MinRate=float(u.LearningRate.min()),MedianRateRatioVs100=float(rate_ratio.median()),RateRatioP01Vs100=float(rate_ratio.quantile(.01)),FractionRateBelowTenthOf100=float((rate_ratio<.1).mean())))
    comparison=pd.DataFrame(rows);ad=pd.DataFrame(annual);sd=pd.DataFrame(strata)
    csv(comparison,'comparison.csv');csv(ad,'comparison_by_year.csv');csv(sd,'mse_by_feature_magnitude.csv');csv(pd.DataFrame(tails),'largest_predictions.csv');csv(pd.DataFrame(rates),'learning_rates.csv')
    csv(pd.concat([d.assign(Variant=n,ZeroForecastMSE=zero) for n,d in dailies.items()]),'all_models_daily.csv')
    rng=np.random.default_rng(20260916);N=953;block=20;reps=4000
    indices=(rng.integers(0,N-block+1,size=(reps,int(np.ceil(N/block))))[:,:,None]+np.arange(block)).reshape(reps,-1)[:,:N]
    boot={}
    for n,d in dailies.items():
        a=d.OfficialDailySpread.to_numpy()[indices]
        boot[n]={'Sharpe':a.mean(axis=1)/a.std(axis=1,ddof=1),'MSE':d.AllStockForecastMSE.to_numpy()[indices].mean(axis=1)}
    ci=[]
    for a,b in [('skip10','skip100'),('no_skip','skip100'),('skip10','no_skip')]:
        for metric,column in [('Sharpe','Sharpe'),('MSE','DailyMeanMSE')]:
            low,high=np.quantile(boot[a][metric]-boot[b][metric],[.025,.975])
            ci.append(dict(Comparison=a+' - '+b,Metric=metric,Difference=float(comparison.set_index('Variant').loc[a,column]-comparison.set_index('Variant').loc[b,column]),CI95Low=float(low),CI95High=float(high)))
    csv(pd.DataFrame(ci),'paired_block_bootstrap.csv')
    baseline=pd.read_csv(B/'jpx_v16_single_financial_v7_20260915/price_only/daily_metrics.csv')
    pd.testing.assert_series_equal(baseline.Date,dailies['skip100'].Date)
    result=dict(version='v22',audit_passed=True,comparison=rows,paired_bootstrap=ci,learning_rates=rates,zero_forecast_daily_mean_mse=float(zero.mean()),v7_reference=dict(sharpe=sharp(baseline.OfficialDailySpread),mse=float(baseline.AllStockForecastMSE.mean())),validation_days=N,validation_stock_days=int(val.sum()),validation_known_target_stock_days=int(known.sum()))
    r.save(R/'results.json',result)
    lines=['# v22：EPS極端值訓練門檻比較','','固定g＋六項EPS、18個參數共同訓練，只更改跳過訓練的門檻。三組都用相同完整股票池評估，100倍組精確重現v21。','','## 整體結果','','953個驗證日，2017-12-29至2021-12-01；官方Sharpe未年化、未扣成本。MSE為每天全部已知Target股票的MSE再等權平均，預測與Target均採小數報酬。','', '| 訓練規則 | Sharpe | 預測MSE | 相對100倍MSE變化 | 跳過股票日 |','|---|---:|---:|---:|---:|']
    for q in rows:lines.append(f'| {q["Label"]} | {q["Sharpe"]:+.8f} | {q["DailyMeanMSE"]:.9f} | {q["MSEChangeVs100Pct"]:+.2f}% | {q["SkippedTrainingStockDays"]:,} ({q["SkipPct"]:.3f}%) |')
    lines+=['',f'原v7純價量參考：Sharpe {result["v7_reference"]["sharpe"]:+.8f}、MSE {result["v7_reference"]["mse"]:.9f}。它未加入EPS且訓練篩選不同，僅作參考。','', '## 預測尾端與穩定性','','MSE衡量預測誤差，不等於排名品質或預測穩定度。以下另列所有驗證股票日的預測尾端；100%代表預測報酬的小數值1。','', '| 規則 | 絕對預測P99 | P99.9 | 最大 | 超過100%筆數 | 每日MSE最大 |','|---|---:|---:|---:|---:|---:|']
    for q in rows:lines.append(f'| {q["Label"]} | {q["AbsPredictionP99"]:.4%} | {q["AbsPredictionP999"]:.4%} | {q["AbsPredictionMax"]:.2%} | {q["PredictionsAbove100Pct"]:,} | {q["DailyMSEMax"]:.6f} |')
    lines+=['','## 相同特徵幅度分組的MSE','','分組取六項衰減後EPS特徵的最大絕對值。每個分組在三模型中的股票日完全相同；分組只供分析，不改變評分股票池。','','| 規則 | 分組 | 股票日 | MSE | 占全部平方誤差 |','|---|---|---:|---:|---:|']
    for q in strata:lines.append(f'| {LABELS[q["Variant"]]} | {q["Stratum"]} | {q["StockDays"]:,} | {q["MSE"]:.8f} | {q["SquaredErrorSharePct"]:.2f}% |')
    lines+=['','## 分期結果','','分期使用既有validation calendar，並非嚴格曆年起訖。','','| 分期 | 100倍Sharpe | 10倍Sharpe | 不跳過Sharpe | 100倍MSE | 10倍MSE | 不跳過MSE |','|---|---:|---:|---:|---:|---:|---:|']
    for year in [2018,2019,2020,2021]:
        t=ad[ad.ValidationYear.eq(year)].set_index('Variant');lines.append('| '+str(year)+' | '+' | '.join([f'{t.loc[n,"Sharpe"]:+.6f}' for n in r.JOBS]+[f'{t.loc[n,"DailyMeanMSE"]:.7f}' for n in r.JOBS])+' |')
    lines+=['','## 不確定性','','三組逐日配對，20日連續區塊、4000次重抽樣、種子20260916。95%百分位區間只用於探索，未校正多重比較與過往反覆嘗試，不是新保留集；MSE差值越低越好。','','| 比較 | 指標 | 差值 | 95%區間 |','|---|---|---:|---:|']
    for q in ci:lines.append(f'| {q["Comparison"]} | {q["Metric"]} | {q["Difference"]:+.8f} | [{q["CI95Low"]:+.8f}, {q["CI95High"]:+.8f}] |')
    lines+=['','## 這輪的主要發現','','1. 10倍組Sharpe點估計略高於100倍組，但其差值95%區間跨0，不能認定10倍門檻可靠地更好；三組兩兩Sharpe差值區間都跨0。','2. 10倍組日均MSE比100倍組增加29.22%，不跳過組則減少16.77%。三項MSE配對差值的探索性區間均不跨0，但尚未校正反覆試驗。','3. 訓練排除更嚴格，不保證完整股票池的預測較穩。10倍組最大絕對預測1089.15%，100倍組473.24%，不跳過組204.46%；>100倍訊號的驗證樣本僅0.148%，卻占10倍組總平方誤差32.87%。在訓練沒看過的大值區域做線性外推，是與這個現象相符的機制，但此實驗未單獨證明原因。','4. 不跳過組步長相對100倍組的每日比值中位數為0.272，32.86%的更新日不到對照十分之一。它同時改變了g與EPS的學習速度，較低MSE與較差排名可能和此有關，不能全部歸因於極端值。','5. 三組MSE都高於零預測參考，顯示低MSE本身不足以證明有可用的選股訊息。不跳過組RankIC也較差，與Sharpe下降的方向一致。','6. 正式基準不更換；100倍保留作後續EPS研究的既有對照。10倍與不跳過的比較留作研究結果，尚無可靠Sharpe勝出者。']
    lines+=['','## 設定與解讀限制','','- 六個EPS特徵、累計差分算法、全年預測剩餘季度推算、exp(-交易日齡/9)事件累加全部沿用v21；PR1/VR1原值。','- 門檻套在當日衰減累加後特徵；任一絕對值嚴格大於10或100即跳過股票日訓練，不裁切數值、不刪除驗證股票。10倍是1000%，不是10%。','- 三組均每日一次MSE更新，從零初始化、跨年連續、只使用已到期Target，共1199次更新；每天步長公式相同。','- 步長為1/(2×Gram最大特徵值)。門檻改變會使實際步長也改變，因此此實驗比較的是整套訓練規則，尚未分離資料效應與步長效應。','- 更低MSE不保證更高Sharpe；要同時看RankIC、預測幅度、分層誤差與投組損益。','- 這次未單獨測定六項EPS是否有用，不將門檻組間差異解讀為EPS本身的因果貢獻。','',f'零預測（所有股票預測報酬為0）在相同驗證樣本的日均MSE為 {zero.mean():.9f}，僅作預測誤差的簡單參考，不具有可辨識的選股排序。','', '## 核對與檔案','','完整預測、排名、梯度、更新、MSE與官方Sharpe已獨立核對；100倍組與v21預測、排名、參數歷程及更新完全一致，來源快取雜湊未變。正式基準保持v7。','','comparison.csv、comparison_by_year.csv、mse_by_feature_magnitude.csv、learning_rates.csv、largest_predictions.csv、all_models_daily.csv、paired_block_bootstrap.csv提供詳細數據。各模型目錄包含完整更新及係數歷程。']
    (R/'JPX-v22-threshold-report.md').write_text('\n'.join(lines)+'\n')
    version=dict(version='v22',description='Joint g plus six EPS: training thresholds 100, 10, none; same full evaluation universe',directory=str(R),status='completed_experiment',variants=r.JOBS,baseline='v21_all_eps',common_days=N,audit_passed=True)
    r.save(R/'version.json',version);reg=json.loads((B/'jpx_model_versions.json').read_text());assert reg['active_version']=='v7';reg['versions']=[v for v in reg['versions'] if v.get('version')!='v22']+[version];reg['latest_experiment']='v22';r.save(B/'jpx_model_versions.json',reg)
    print(comparison.to_string(index=False));print(ad.to_string(index=False));print(sd.to_string(index=False));print(pd.DataFrame(rates).to_string(index=False));print(pd.DataFrame(ci).to_string(index=False));print('Zero forecast MSE:',zero.mean())
if __name__=='__main__':main()
