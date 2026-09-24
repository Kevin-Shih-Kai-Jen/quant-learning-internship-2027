from pathlib import Path
import json
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent;B=R.parent
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))
def sharp(s):return float(np.mean(s)/np.std(s,ddof=1))
def main():
    assert json.loads((R/'audit.json').read_text())['passed']
    labels={'v7':'原v7純價量','all_eps':'g＋全部六項EPS','price_mask':'純價量：同樣本','price_mask_step':'純價量：同樣本同主模型步長'}
    frames={'v7':pd.read_csv(B/'jpx_v16_single_financial_v7_20260915/price_only/daily_metrics.csv')}
    for n in ['all_eps','price_mask','price_mask_step']:frames[n]=pd.read_csv(R/n/'daily_metrics.csv')
    dates=frames['v7'].Date
    for n,d in frames.items():
        pd.testing.assert_series_equal(dates,d.Date);assert d.SelectedMissingTargets.sum()==0 and len(d)==953
    base=sharp(frames['v7'].OfficialDailySpread);rows=[];annual=[]
    for name,d in frames.items():
        rows.append(dict(Variant=name,Label=labels[name],Sharpe=sharp(d.OfficialDailySpread),DeltaVsV7=sharp(d.OfficialDailySpread)-base,MeanDailyGrossOneReturnBP=float(d.IllustrativeGrossOneReturn.mean()*10000),DailyGrossOneReturnStdBP=float(d.IllustrativeGrossOneReturn.std()*10000),RankIC=float(d.RankIC.mean()),ReturnMSE=float(d.AllStockForecastMSE.mean()),NormalizedHardRankMSE=float(d.NormalizedHardRankMSE.mean())))
        for year,g in d.groupby('ValidationYear'):annual.append(dict(Variant=name,ValidationYear=int(year),Days=len(g),Sharpe=sharp(g.OfficialDailySpread),MeanDailyGrossOneReturnBP=float(g.IllustrativeGrossOneReturn.mean()*10000)))
    pd.DataFrame(rows).to_csv(R/'comparison.csv',index=False,encoding='utf-8-sig');ad=pd.DataFrame(annual);ad.to_csv(R/'comparison_by_year.csv',index=False,encoding='utf-8-sig')
    pd.concat([d.assign(Variant=n) for n,d in frames.items()]).to_csv(R/'all_models_daily.csv',index=False,encoding='utf-8-sig')
    coefs=[]
    for n in ['all_eps','price_mask','price_mask_step']:
        rr=json.loads((R/n/'results.json').read_text())
        for p,v in rr['final_coefficients'].items():coefs.append(dict(Variant=n,Parameter=p,Value=v,AsOf=rr['parameter_asof']))
    pd.DataFrame(coefs).to_csv(R/'all_coefficients.csv',index=False,encoding='utf-8-sig')
    features=[
      ('EPSActualQoQ','EPS實際季增','實績公布','(本季實績-前季實績)/abs(前季實績)'),
      ('EPSActualGrowth','EPS實際年增','實績公布；沿用舊事件有效性條件','(本季實績-去年同季實績)/abs(去年同季實績)'),
      ('EPSExpectedGrowth','實績公布時的事前預期成長','實績公布；預測凍結於公布前','(事前本季預測-去年同季實績)/abs(去年同季實績)'),
      ('EPSForecastActualQoQ','EPS預測對前季實績','Forecast事件','(本季預測-已知前季實績)/abs(已知前季實績)'),
      ('EPSForecastActualYoY','EPS預測對去年同季實績','Forecast事件','(本季預測-已知去年同季實績)/abs(已知去年同季實績)'),
      ('EPSRevisionRelative','EPS預期修正','同基準單季Forecast改變','(新單季預測-舊單季預測)/舊單季預測')]
    pd.DataFrame(features,columns=['Feature','Label','Trigger','Formula']).to_csv(R/'feature_definitions.csv',index=False,encoding='utf-8-sig')
    rng=np.random.default_rng(20260916);N=953;block=20;reps=4000
    indices=(rng.integers(0,N-block+1,size=(reps,int(np.ceil(N/block))))[:,:,None]+np.arange(block)).reshape(reps,-1)[:,:N]
    estimates={}
    for n,d in frames.items():
        sample=d.OfficialDailySpread.to_numpy()[indices];estimates[n]=sample.mean(axis=1)/sample.std(axis=1,ddof=1)
    ci=[]
    for n in ['v7','price_mask','price_mask_step']:
        low,high=np.quantile(estimates['all_eps']-estimates[n],[.025,.975]);ci.append(dict(Comparison='all_eps - '+n,DeltaSharpe=sharp(frames['all_eps'].OfficialDailySpread)-sharp(frames[n].OfficialDailySpread),CI95Low=float(low),CI95High=float(high)))
    pd.DataFrame(ci).to_csv(R/'paired_block_bootstrap.csv',index=False,encoding='utf-8-sig')
    tr=pd.read_csv(R/'all_eps/training_updates.csv');pr=pd.read_csv(R/'price_mask/training_updates.csv');ratio=tr.LearningRate/pr.LearningRate
    lr=dict(median_main_vs_price_step=float(ratio.median()),min_main_vs_price_step=float(ratio.min()),fraction_main_step_below_half=float((ratio<.5).mean()))
    result=json.loads((R/'all_eps/results.json').read_text());main=result['sharpe']
    lines=['# v21：g＋全部EPS特徵共同訓練','', '依使用者要求，保留目前EPS計算方法與JPX提供的數值，將價量g與六個EPS係數共同訓練。主模型Sharpe為負且接近零，低於原v7及兩個相同樣本純價量對照。','', '## 整段回測','', '共同953個驗證日，2017-12-29至2021-12-01。官方Sharpe未年化、未扣成本；總Sharpe由整段每日損益直接計算，不平均分期Sharpe。','', '| 模型 | Sharpe | Δ對原v7 | 預測MSE |','|---|---:|---:|---:|']
    for row in rows:lines.append(f'| {row["Label"]} | {row["Sharpe"]:+.8f} | {row["DeltaVsV7"]:+.8f} | {row["ReturnMSE"]:.9f} |')
    lines+=['', '相同樣本對照同樣排除3,025個到期訓練股票日；另加的同一步長對照，也使用主模型的每日學習率。因此這輪合併版低於純價量，不能只歸因於訓練股票變少或步長不同。這仍是指定模型／訓練設定下的整體效果，不是每個EPS特徵的獨立因果貢獻。','', '## 模型內容','', 'prediction(i,t) = g(i,t; theta) + b1×EPSActualQoQ + b2×EPSActualGrowth + b3×EPSExpectedGrowth + b4×EPSForecastActualQoQ + b5×EPSForecastActualYoY + b6×EPSRevisionRelative。','', 'g的12個參數（截距及11個價量特徵）與6個EPS係數全部更新，合計18個。沒有固定v7歷史g，沒有加入營收、營業利益、交叉項或額外EPS原值水準。','', '| EPS特徵 | 觸發時點 | 公式 |','|---|---|---|']
    for col,label,trigger,formula in features:lines.append(f'| {label} | {trigger} | {formula} |')
    lines+=['', '「實績公布時的事前預期成長」與「Forecast事件時對去年同季實績」觸發時間不同，所以各自保留係數。未將已取消的預測對預測季增／年增加回。實際年增與事前預期成長分別估計係數，沒有額外固定成一個r-df差值或新增折價約束。','', '## 訓練設定','', '- EPS算法原樣保留：累計差分、全年Forecast剩餘季度推算、基期與事件有效性條件均使用既有快取。未採用外部文件修正或v20股數調整估算。','- 全部財報事件按exp(-交易日齡/9)累加，沒有新事件時既有影響繼續衰減。','- PR1／VR1使用正式v7原值。普通MSE，每天對已到期的Target更新一次；從零開始、跨年連續學習。沒有改成soft rank、每股SGD或sqrt(N)多次更新。','- 每日步長為1/(2×訓練設計矩陣Gram最大特徵值)。主模型與同一步長對照共用完整18維矩陣計算的步長；一般純價量對照使用自己的12維矩陣。','- 六項衰減後EPS訊號任一絕對值>100即跳過該股票日訓練。100是比率100倍，非100%；此為沿用的訓練規則，不把大值判定為資料錯誤。','- 每組1,199次更新，使用2,322,781個到期訓練股票日、跳過3,025個；預測及選股仍保留完整股票池，共2,326,022筆歷史預測。','', '## 分期Sharpe','', '| 模型 | 2018 | 2019 | 2020 | 2021 |','|---|---:|---:|---:|---:|']
    for n in frames:
        a=ad[ad.Variant.eq(n)].set_index('ValidationYear');lines.append('| '+labels[n]+' | '+' | '.join(f'{a.loc[y,"Sharpe"]:+.6f}' for y in [2018,2019,2020,2021])+' |')
    lines+=['', '分期沿用既有validation calendar，非嚴格曆年起訖。','', '## 差值不確定性','', '共同日期配對，20個連續交易日為區塊，重抽樣4,000次，種子20260916。以下為探索性95%百分位區間，未調整反覆嘗試模型，也不是新的保留測試集。','', '| 比較 | ΔSharpe | 95%區間 |','|---|---:|---:|']
    for row in ci:lines.append(f'| {row["Comparison"]} | {row["DeltaSharpe"]:+.8f} | [{row["CI95Low"]:+.8f}, {row["CI95High"]:+.8f}] |')
    lines+=['', '三個差值區間均包含0。因此這次點估計沒有改善，但還不能認定加入EPS必然造成負貢獻。', '', '## 最後的EPS係數','', '以下截至2021-12-03最後到期標籤更新之後，不是整段回測固定使用的係數。係數大小受特徵尺度、相關性影響，不能直接當作重要性排序。','', '| 特徵 | 係數 |','|---|---:|']
    for feature,label,_,_ in features:lines.append(f'| {label} | {result["final_coefficients"][feature]:+.10f} |')
    lines+=['', '## 核對與交付','', '三個模型各自全部2,326,022筆預測與排名、1,199次更新、訓練時間與門檻、953日收益及官方Sharpe均獨立重算通過。原資料快取雜湊未變，主模型與同步長對照的步長逐日相同。','', '這輪只回答「目前六項EPS合併、與g共同學習是否改善」：結果沒有改善。不能據此推論每個EPS項目都無效，也尚未測試逐項移除的效果。正式基準仍為v7，本次不自動切換模型。','', '- comparison.csv／comparison_by_year.csv：總表與分期。','- feature_definitions.csv：六項完整定義。','- all_models_daily.csv／all_coefficients.csv：逐日結果與期末係數。','- 各組 parameter_history.csv／training_updates.csv：完整係數歷程與更新。','- paired_block_bootstrap.csv：差值區間。','- 重現：run.py → audit.py → report.py。']
    (R/'JPX-v21-all-eps-joint-report.md').write_text('\n'.join(lines)+'\n')
    save(R/'results.json',dict(version='v21',comparison=rows,bootstrap=ci,learning_rate_diagnostics=lr,features=[r[0] for r in features],eps_algorithm_unchanged=True,jointly_train_price_and_eps=True,audit_passed=True))
    version=dict(version='v21',description='Jointly train v7 price-volume and six unchanged EPS features with matched-sample and matched-step controls',directory=str(R),status='completed_experiment',variants=['all_eps','price_mask','price_mask_step'],parameter_count=18,main_sharpe=main,baseline='v7_equal',common_days=953,eps_algorithm_unchanged=True,audit_passed=True)
    save(R/'version.json',version);reg=json.loads((B/'jpx_model_versions.json').read_text());assert reg['active_version']=='v7';reg['versions']=[v for v in reg['versions'] if v.get('version')!='v21']+[version];reg['latest_experiment']='v21';save(B/'jpx_model_versions.json',reg)
    p=B/'JPX-current-baseline.md';s=p.read_text();title='## 最新完成的 v21 全EPS共同訓練'
    if title not in s:p.write_text(s+f'\n{title}\n\n2026-09-16依使用者要求，維持目前EPS算法，把價量g與六項EPS係數一起訓練。953日Sharpe {main:+.8f}，原v7 {base:+.8f}；同樣本價量對照 {rows[2]["Sharpe"]:+.8f}，同樣本同主模型步長對照 {rows[3]["Sharpe"]:+.8f}。獨立核對通過，正式基準維持v7。\n\n[完整 v21 報告]({R}/JPX-v21-all-eps-joint-report.md)。\n')
    print(pd.DataFrame(rows).to_string(index=False));print(ad.to_string(index=False));print(pd.DataFrame(ci).to_string(index=False));print(json.dumps(lr))
if __name__=='__main__':main()
