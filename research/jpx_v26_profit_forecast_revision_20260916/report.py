from pathlib import Path
import json
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent;B=R.parent
PATHS={'g':B/'jpx_v23_g_thresholds_20260916/no_skip','forecast':B/'jpx_v25_single_features_unfiltered_20260916/v16_feature_09','revision':B/'jpx_v25_single_features_unfiltered_20260916/v17_profit_revision_relative','joint':R/'joint'}
LABELS={'g':'純價量g','forecast':'g＋營業利益預測對去年預測','revision':'g＋營業利益預期修正','joint':'g＋兩項營業利益特徵'}
def sharp(a):return float(np.mean(a)/np.std(a,ddof=1))
def save(name,v):(R/name).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))
def csv(name,d):d.to_csv(R/name,index=False,encoding='utf-8-sig')
def main():
    assert json.loads((R/'audit.json').read_text())['passed']
    frames={};rows=[];annual=[];training={};coeffs=[]
    for name,path in PATHS.items():
        d=pd.read_csv(path/'daily_metrics.csv');frames[name]=d;u=pd.read_csv(path/'training_updates.csv');training[name]=u
        pd.testing.assert_series_equal(d.Date,frames['g'].Date)
        for c in ['Date','SignalDate','ExitDate','TrainingStocks','ThresholdSkippedStocks']:pd.testing.assert_series_equal(u[c],training['g'][c])
        assert len(d)==953 and d.RankIC.notna().sum()==952 and d.SelectedMissingTargets.sum()==0 and u.ThresholdSkippedStocks.sum()==0
        rows.append(dict(Variant=name,Label=LABELS[name],Sharpe=sharp(d.OfficialDailySpread),RankIC=float(d.RankIC.mean()),MSE=float(d.AllStockForecastMSE.mean()),DeltaSharpeVsG=sharp(d.OfficialDailySpread)-sharp(frames['g'].OfficialDailySpread),DeltaRankICVsG=float(d.RankIC.mean()-frames['g'].RankIC.mean()),MedianLearningRate=float(u.LearningRate.median())))
        for year,g in d.groupby('ValidationYear'):annual.append(dict(Variant=name,ValidationYear=int(year),Days=len(g),Sharpe=sharp(g.OfficialDailySpread),RankIC=float(g.RankIC.mean()),MSE=float(g.AllStockForecastMSE.mean())))
        q=json.loads((path/'results.json').read_text())
        for p,v in q['final_coefficients'].items():coeffs.append(dict(Variant=name,Parameter=p,Value=v,AsOf=q['parameter_asof']))
    df=pd.DataFrame(rows);ad=pd.DataFrame(annual);by=df.set_index('Variant');rng=np.random.default_rng(20260916);N=953;block=20;reps=4000
    ix=(rng.integers(0,N-block+1,size=(reps,int(np.ceil(N/block))))[:,:,None]+np.arange(block)).reshape(reps,-1)[:,:N]
    boots={}
    for n,d in frames.items():
        a=d.OfficialDailySpread.to_numpy()[ix];boots[n]={'Sharpe':a.mean(axis=1)/a.std(axis=1,ddof=1),'RankIC':np.nanmean(d.RankIC.to_numpy()[ix],axis=1)}
    ci=[]
    for n in ['g','forecast','revision']:
        for m in ['Sharpe','RankIC']:
            lo,hi=np.quantile(boots['joint'][m]-boots[n][m],[.025,.975]);ci.append(dict(Comparison='joint - '+n,Metric=m,Difference=float(by.loc['joint',m]-by.loc[n,m]),CI95Low=float(lo),CI95High=float(hi)))
    lo,hi=np.quantile(boots['joint']['RankIC'],[.025,.975]);level=dict(RankIC=float(by.loc['joint','RankIC']),CI95Low=float(lo),CI95High=float(hi))
    rates=[]
    for n in ['g','forecast','revision']:
        ratio=training['joint'].LearningRate/training[n].LearningRate;rates.append(dict(Reference=n,MedianJointRateRatio=float(ratio.median()),FractionBelowHalf=float((ratio<.5).mean())))
    for name,d in [('comparison.csv',df),('comparison_by_year.csv',ad),('paired_block_bootstrap.csv',pd.DataFrame(ci)),('all_coefficients.csv',pd.DataFrame(coeffs)),('all_models_daily.csv',pd.concat([d.assign(Variant=n) for n,d in frames.items()])),('learning_rate_comparison.csv',pd.DataFrame(rates))]:csv(name,d)
    result=dict(version='v26',audit_passed=True,features=['OperatingProfitForecastYoY','OperatingProfitRevisionRelative'],comparison=rows,paired_bootstrap=ci,joint_rank_ic_interval=level,joint_highest_sharpe=bool(by.loc['joint','Sharpe']==by.Sharpe.max()),joint_highest_rank_ic=bool(by.loc['joint','RankIC']==by.RankIC.max()),training_financial_abs_limit=None,validation_financial_abs_limit=None,active_baseline='v7')
    save('results.json',result)
    lines=['# v26：營業利益預測年比＋預期修正＋g','','依使用者指定，將營業利益／預測對去年預測與營業利益／預期修正共同加入g。14個參數（價量12個、財報2個）共同訓練；不排除或裁切極端值，完整股票池驗證。','','## 四組結果','','953個驗證日，2017-12-29至2021-12-01；平均Rank IC有效952日，排除全體Target相同的2020-09-29而不補0。Sharpe未年化、未扣成本。','','| 模型 | Sharpe | 平均Rank IC | MSE（診斷） |','|---|---:|---:|---:|']
    for q in rows:lines.append(f'| {q["Label"]} | {q["Sharpe"]:+.8f} | {q["RankIC"]:+.8f} | {q["MSE"]:.9f} |')
    lines+=['',f'組合Sharpe高於純g，但比單獨預測年比低 {by.loc["forecast","Sharpe"]-by.loc["joint","Sharpe"]:.8f}；這次沒有出現組合Sharpe最高。組合平均Rank IC是四組最高，但仍為負，因此未同時達到Sharpe改善與Rank IC>0的條件。','', '## 分期結果','','分期沿用既有validation calendar，非嚴格曆年起訖。','','| 模型 | 2018 Sharpe | 2019 Sharpe | 2020 Sharpe | 2021 Sharpe |','|---|---:|---:|---:|---:|']
    for n in PATHS:
        a=ad[ad.Variant.eq(n)].set_index('ValidationYear');lines.append('| '+LABELS[n]+' | '+' | '.join(f'{a.loc[y,"Sharpe"]:+.6f}' for y in [2018,2019,2020,2021])+' |')
    lines+=['','## 差值不確定性','','20日連續區塊、4000次同日期配對重抽樣，種子20260916。探索性95%百分位區間，未校正多重比較與過去模型選擇，不是新的保留測試集。','','| 比較（組合減對照） | 指標 | 差值 | 95%區間 |','|---|---|---:|---:|']
    for q in ci:lines.append(f'| {q["Comparison"]} | {q["Metric"]} | {q["Difference"]:+.8f} | [{q["CI95Low"]:+.8f}, {q["CI95High"]:+.8f}] |')
    lines+=['',f'組合平均Rank IC的95%區間：[{level["CI95Low"]:+.8f}, {level["CI95High"]:+.8f}]。','', '## 這次如何加入修正','','模型：prediction(i,t) = g(i,t;theta) + bY*Y(i,t) + bR*R(i,t)。Y與R均沿用exp(-交易日齡/9)事件累加訊號。','', '- Y：在目標季首次符合條件的Forecast事件，(本季單季預測-去年同季凍結預測)/abs(去年同季凍結預測)。','- R：在全年Forecast改變時，使用同一已知累計實績與剩餘季度數，先求新舊單季預測，再算(新單季預測-舊單季預測)/舊單季預測。分母帶正負號，沿用使用者先前公式。','- 資料內部的最新預測會更新；已產生的Y事件則繼續衰減，R另外加入。兩個係數各自估計，不強制把舊Y值覆蓋成新的Y。','', '所以這輪測的是「初始相對去年預測資訊＋後續修正訊息能否互補」，不等同於測試「每次修正後重算最新預測年增，並取代舊訊號」。','', '用單一事件、不考慮衰減的代數例子說明：設去年預測為b、修正前預測f0、修正後f1。Y0=(f0-b)/abs(b)，R=(f1-f0)/f0，則Y1=Y0+R*f0/abs(b)。兩個不同分母使得固定係數的Y與R相加不必等於更新後的Y1；本輪亦沒有加入這個動態比例交叉項。此說明不改動本輪特徵。','', '## 設定與核對','','- 普通MSE每日一次更新，PR1/VR1原值、從零初始化、跨年連續，只使用已到期Target。每組1199次更新、2,325,806個訓練股票日、極端值排除0筆。','- 四組逐日訓練股票數、驗證股票池完全一致。價量與財報係數共同學習；不同矩陣維度會改變實際步長，因此仍是整套模型比較。','- 組合全部梯度、步長、loss與參數更新在訓練時獨立核對；所有2,326,022筆預測、排名、更新時序、Rank IC、MSE與官方公式Sharpe另核對通過。','- 三個對照引用已核對的不排除版本，不需重訓；原始快取及對照檔案雜湊未變。','- 正式基準維持v7。本次點估計沒有支持「組合Sharpe最高」，不據此認定修正值沒有資訊；尚未測定精確最新Forecast覆蓋的另一種表示方式。','', '## 最後兩項財報係數','','以下是2021-12-03最後到期標籤更新後的係數，回測每日使用當時係數，未拿期末參數回填。係數大小不能直接当特徵重要性。']
    joint=json.loads((R/'joint/results.json').read_text())
    for f in result['features']:lines.append(f'- {f}：{joint["final_coefficients"][f]:+.10f}')
    (R/'JPX-v26-profit-forecast-revision-report.md').write_text('\n'.join(lines)+'\n')
    version=dict(version='v26',description='Joint operating-profit forecast YoY and relative forecast revision plus v7 g, unfiltered',directory=str(R),status='completed_experiment',parameter_count=14,features=result['features'],main_sharpe=float(by.loc['joint','Sharpe']),main_rank_ic=float(by.loc['joint','RankIC']),audit_passed=True)
    save('version.json',version);reg=json.loads((B/'jpx_model_versions.json').read_text());assert reg['active_version']=='v7';reg['versions']=[v for v in reg['versions'] if v.get('version')!='v26']+[version];reg['latest_experiment']='v26';(B/'jpx_model_versions.json').write_text(json.dumps(reg,ensure_ascii=False,indent=2))
    print(df.to_string(index=False));print(ad.to_string(index=False));print(pd.DataFrame(ci).to_string(index=False));print(level)
if __name__=='__main__':main()
