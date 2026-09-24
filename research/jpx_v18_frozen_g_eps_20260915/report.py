from pathlib import Path
import json
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent;B=R.parent
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))
def sharp(s):return float(np.mean(s)/np.std(s,ddof=1))
def main():
    assert json.loads((R/'audit.json').read_text())['passed']
    names={'g':'原 v7 價量 g','qoq':'固定 g + EPS 實際季增','yoy':'固定 g + EPS 實際年增','joint':'固定 g + EPS 季增 + 年增'}
    d={'g':pd.read_csv(B/'jpx_v16_single_financial_v7_20260915/price_only/daily_metrics.csv')}
    for n in ['qoq','yoy','joint']:d[n]=pd.read_csv(R/n/'daily_metrics.csv')
    for v in d.values():
        assert len(v)==953 and v.SelectedMissingTargets.sum()==0
        pd.testing.assert_series_equal(d['g'].Date,v.Date)
    oldnames={'qoq':'eps_qoq_joint_mask','yoy':'eps_yoy_joint_mask','joint':'eps_actual_joint'}
    rows=[];annual=[];coefs=[]
    bs=sharp(d['g'].OfficialDailySpread)
    for n,v in d.items():
        sh=sharp(v.OfficialDailySpread)
        old=sharp(pd.read_csv(B/'jpx_v17_actual_forecast_revisions_20260915'/oldnames[n]/'daily_metrics.csv').OfficialDailySpread) if n in oldnames else bs
        rows.append(dict(Variant=n,Label=names[n],Sharpe=sh,DeltaVsV7=sh-bs,PreviousJointlyTrainedSharpe=old,MeanDailyReturnBP=float(v.IllustrativeGrossOneReturn.mean()*10000),DailyReturnStdBP=float(v.IllustrativeGrossOneReturn.std()*10000),MeanRankIC=float(v.RankIC.mean()),ReturnMSE=float(v.AllStockForecastMSE.mean())))
        for year,a in v.groupby('ValidationYear'):annual.append(dict(Variant=n,ValidationYear=int(year),Days=len(a),Sharpe=sharp(a.OfficialDailySpread)))
        if n!='g':
            result=json.loads((R/n/'results.json').read_text())
            for k,val in result['final_coefficients'].items():coefs.append(dict(Variant=n,Parameter=k,Value=val,AsOf='2021-12-03'))
    comparison=pd.DataFrame(rows);comparison.to_csv(R/'comparison.csv',index=False,encoding='utf-8-sig')
    ann=pd.DataFrame(annual);ann.to_csv(R/'comparison_by_year.csv',index=False,encoding='utf-8-sig')
    pd.concat([v.assign(Variant=n) for n,v in d.items()]).to_csv(R/'all_models_daily.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(coefs).to_csv(R/'all_coefficients.csv',index=False,encoding='utf-8-sig')
    rng=np.random.default_rng(20260915);length=953;block=20;reps=4000
    idx=((rng.integers(0,length-block+1,size=(reps,int(np.ceil(length/block)))))[:,:,None]+np.arange(block)).reshape(reps,-1)[:,:length]
    boot={n:(v.OfficialDailySpread.to_numpy()[idx]) for n,v in d.items()}
    estimates={n:a.mean(axis=1)/a.std(axis=1,ddof=1) for n,a in boot.items()}
    ci=[]
    for a,b in [('qoq','g'),('yoy','g'),('joint','g'),('joint','qoq'),('joint','yoy')]:
        low,high=np.quantile(estimates[a]-estimates[b],[.025,.975]);ci.append(dict(Comparison=a+' - '+b,DeltaSharpe=sharp(d[a].OfficialDailySpread)-sharp(d[b].OfficialDailySpread),CI95Low=float(low),CI95High=float(high)))
    pd.DataFrame(ci).to_csv(R/'paired_block_bootstrap.csv',index=False,encoding='utf-8-sig')
    lines=['# v18：固定 g，只訓練 EPS 季增／年增','', '已完成三組財報係數訓練及獨立核對。固定的是 v7 在各歷史訊號日產生的預測路徑；財報 loss 不再改動價量參數。這一輪三組 Sharpe 都為正，但均低於原 v7；季增＋年增也沒有超過任何單獨一組。','', '## 共同期間結果','', '953 個驗證交易日，2017-12-29 至 2021-12-01；官方 Sharpe 未年化、未扣交易成本。','', '| 模型 | Sharpe | 相對 v7 | 舊 v17 價量共同訓練 |','|---|---:|---:|---:|']
    for a in rows:lines.append(f'| {a["Label"]} | {a["Sharpe"]:+.8f} | {a["DeltaVsV7"]:+.8f} | {a["PreviousJointlyTrainedSharpe"]:+.8f} |')
    lines+=['','## 這次究竟固定什麼','', '令 g(i,t) 為原 v7 在當時產生並保存的預測，Q(i,t)、Y(i,t) 為既有 EPS 實際季增與年增的衰減特徵。','', '- 季增：prediction = g + bQ × Q。','- 年增：prediction = g + bY × Y。','- 聯合：prediction = g + bQ × Q + bY × Y。','', '只有 bQ、bY 學習，沒有新增截距；g 不依財報訓練重新計算，也不是將期末價量參數套回過去。三組使用完全相同的 g，原 v7 自身的历史價量學習路徑保持原樣。','', 'Target 到期之後，每日對當時合格股票做一次普通 MSE 更新：L = mean[(g + F b - Target)^2]；gradient = 2 Fᵀ(g + F b - Target)/N；b_next = b - eta × gradient。財報係數從 0 起始、跨年連續更新。','', '三組使用同一每日 eta：沿用 v17 聯合模型的 1/(2 × 最大特徵值)，設計矩陣為 [原價量, Q, Y]，在同一合格樣本計算。此矩陣只用來定步長；不更新價量。這避免季增、年增、聯合三者連步長都不同。並未改為財報專屬的較大步長或搜尋最佳學習率。','', 'Q 或 Y 任一絕對值 > 100 均跳過該股票日訓練，三組共同使用 2,323,958 個訓練股票日、排除 1,848 個，1,199 次到期更新。預測均保留全股票池，共 2,326,022 筆。原 v7 g 來源沒有套財報排除門檻；它是固定來源，不是重新以共同門檻訓練的價量對照。','', '## 分期 Sharpe','', '| 模型 | 2018 | 2019 | 2020 | 2021 |','|---|---:|---:|---:|---:|']
    for n in d:
        a=ann[ann.Variant.eq(n)].set_index('ValidationYear');lines.append('| '+names[n]+' | '+' | '.join(f'{a.loc[y,"Sharpe"]:+.6f}' for y in [2018,2019,2020,2021])+' |')
    lines+=['','分期標籤沿用原 validation calendar，不是嚴格曆年邊界；總 Sharpe 直接串接全部每日損益計算，不平均各年 Sharpe。','', '## 差距的不確定性','', '把各模型同一天的損益一起抽取，使用 20 個連續交易日區塊、4,000 次重抽樣，固定種子 20260915。以下為探索性的差值區間，未校正反覆選模型，也不是新保留測試集。','', '| 差值 | 實際 ΔSharpe | 95% 區間 |','|---|---:|---:|']
    for a in ci:lines.append(f'| {a["Comparison"]} | {a["DeltaSharpe"]:+.8f} | [{a["CI95Low"]:+.8f}, {a["CI95High"]:+.8f}] |')
    lines+=['','## 可以如何解讀','', '1. 以這個固定 g、MSE、共同步長及既有財報定義，沒有看到新增 EPS 季增或年增提高整段 Sharpe。高於零不等於有額外貢獻，需與相同的 g 比較。','2. 聯合組低於單獨兩組，說明這個訓練設定下合用未帶來改善；不能單憑此結果判定資訊重疊、因果干擾或年增永久無用。差值區間與分期結果應一起看。','3. 舊 v17 單獨季增較好的表現，在固定原 v7 預測後沒有保留。因此先前優勢不能直接歸因於一個獨立的季增加分項。舊模型也會重估價量權重，單一特徵的步長略不同，且用更新時的價量參數重算訓練預測；新舊差異不是純粹的因果分解。','4. MSE 最小化並不直接最佳化頭尾 200 檔的排序損益；CSV 同時保留 MSE、Rank IC、日損益，可檢查改善預測誤差是否真的改善選股。','', '## 保留的資料限制與稽核','', '- 財報仍沿用 exp(-交易日齡/9) 事件累加；沒有新事件不等於舊影響立即歸零。','- EPS 拆季保留原程式的累計 EPS 差分近似；股份數變化時需另行檢查。EPS 年增的既有事件觸發條件也原樣保留，這次沒有同時修改資料假設。','- 三組全部預測、排名、1,199 次到期更新、共同訓練樣本、共同步長以及官方 Sharpe 都獨立重算通過。保存的 g 與來源逐值相同。','- 這是已反覆研究過的同一歷史驗證期，尚未取得新的樣本外證據。正式基準仍為 v7。','', '## 檔案','', '- comparison.csv：完整比較、日報酬均值／標準差、Rank IC、MSE。','- comparison_by_year.csv、all_models_daily.csv：分期與逐日結果。','- paired_block_bootstrap.csv：差值區間。','- 各模型資料夾：完整預測、逐日係數與每次梯度更新。','- run.py → audit.py → report.py：重現流程；manifest.json 記錄輸入雜湊。']
    (R/'JPX-v18-frozen-g-report.md').write_text('\n'.join(lines)+'\n')
    save(R/'results.json',dict(comparison=rows,bootstrap=ci,bootstrap_block=block,bootstrap_replicates=reps,audit_passed=True))
    ver=dict(version='v18',description='Freeze point-in-time v7 g; learn EPS QoQ, YoY, and joint coefficients with shared training mask and step schedule',directory=str(R),status='completed_experiment',variants=['qoq','yoy','joint'],baseline='v7_equal',common_days=953,audit_passed=True,variant_sharpes={a['Variant']:a['Sharpe'] for a in rows})
    save(R/'version.json',ver)
    reg=json.loads((B/'jpx_model_versions.json').read_text());assert reg['active_version']=='v7'
    reg['versions']=[v for v in reg['versions'] if v.get('version')!='v18']+[ver];reg['latest_experiment']='v18';save(B/'jpx_model_versions.json',reg)
    p=B/'JPX-current-baseline.md';s=p.read_text();title='## 最新完成的 v18 固定 g 比較'
    if title not in s:p.write_text(s+f'\n{title}\n\n固定原 v7 各訊號日預測，只訓練 EPS 季增／年增係數。三組共同樣本及步長，953 日 Sharpe：季增 {rows[1]["Sharpe"]:+.8f}、年增 {rows[2]["Sharpe"]:+.8f}、聯合 {rows[3]["Sharpe"]:+.8f}，皆低於 v7 {bs:+.8f}。正式基準仍為 v7。\n\n[完整 v18 報告]({R}/JPX-v18-frozen-g-report.md)。\n')
    print(comparison.to_string(index=False));print(ann.to_string(index=False));print(pd.DataFrame(ci).to_string(index=False))
if __name__=='__main__':main()
