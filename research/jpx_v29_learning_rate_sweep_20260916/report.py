from pathlib import Path
import json
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent;B=R.parent
MODELS={'g_common':'純g','overwrite':'A：新值覆蓋','change_old':'B：差額＋舊值'}
MULT={'c025':.25,'c05':.5,'c1':1.,'c15':1.5,'c19':1.9}
def save(n,d):(R/n).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False))
def csv(n,d):d.to_csv(R/n,index=False,encoding='utf-8-sig')
def sharp(x):return float(np.mean(x)/np.std(x,ddof=1))
def main():
    assert json.loads((R/'audit.json').read_text())['passed']
    frames={};rows=[];annual=[];coefs=[]
    for prefix,c in MULT.items():
        for model in MODELS:
            n=prefix+'_'+model;p=R/n;d=pd.read_csv(p/'daily_metrics.csv');frames[n]=d;rr=json.loads((p/'results.json').read_text());u=pd.read_csv(p/'training_updates.csv')
            pd.testing.assert_series_equal(d.Date,frames['c025_g_common'].Date)
            assert len(d)==953 and d.RankIC.notna().sum()==952 and u.ThresholdSkippedStocks.sum()==0 and d.SelectedMissingTargets.sum()==0
            rows.append(dict(Variant=n,Model=model,Label=MODELS[model],Multiplier=c,Sharpe=sharp(d.OfficialDailySpread),RankIC=float(d.RankIC.mean()),ValidationDays=len(d)))
            for year,q in d.groupby('ValidationYear'):annual.append(dict(Variant=n,Model=model,Multiplier=c,ValidationYear=int(year),Days=len(q),Sharpe=sharp(q.OfficialDailySpread),RankIC=float(q.RankIC.mean())))
            for k,v in rr['final_coefficients'].items():coefs.append(dict(Variant=n,Parameter=k,Value=v,AsOf=rr['parameter_asof']))
    df=pd.DataFrame(rows);ad=pd.DataFrame(annual);ranking=df.sort_values(['Sharpe','RankIC'],ascending=False,kind='stable').reset_index(drop=True);ranking.insert(0,'ValidationRank',np.arange(1,len(ranking)+1));best=ranking.groupby('Model',sort=False).head(1);winner=ranking.iloc[0];bestg=best[best.Model.eq('g_common')].iloc[0]
    original=pd.read_csv(B/'jpx_v23_g_thresholds_20260916/no_skip/daily_metrics.csv');pd.testing.assert_series_equal(original.Date,frames[winner.Variant].Date);frames['original_v7']=original
    original_result=dict(Sharpe=sharp(original.OfficialDailySpread),RankIC=float(original.RankIC.mean()))
    comparisons=[]
    for q in best.itertuples():
        for ref in ['c1_'+q.Model,'original_v7']:
            if q.Variant!=ref and (q.Variant,ref) not in comparisons:comparisons.append((q.Variant,ref))
    prefix=next(p for p,c in MULT.items() if c==winner.Multiplier)
    for ref in [prefix+'_g_common',bestg.Variant]:
        if winner.Variant!=ref and (winner.Variant,ref) not in comparisons:comparisons.append((winner.Variant,ref))
    rng=np.random.default_rng(20260916);N=953;block=20;reps=4000
    ix=(rng.integers(0,N-block+1,size=(reps,int(np.ceil(N/block))))[:,:,None]+np.arange(block)).reshape(reps,-1)[:,:N];boots={};ci=[]
    for n in set(a for pair in comparisons for a in pair):
        d=frames[n];a=d.OfficialDailySpread.to_numpy()[ix];boots[n]={'Sharpe':a.mean(axis=1)/a.std(axis=1,ddof=1),'RankIC':np.nanmean(d.RankIC.to_numpy()[ix],axis=1)}
    for a,b in comparisons:
        for m,col in [('Sharpe','OfficialDailySpread'),('RankIC','RankIC')]:
            lo,hi=np.quantile(boots[a][m]-boots[b][m],[.025,.975]);v=(sharp(frames[a][col])-sharp(frames[b][col])) if m=='Sharpe' else float(frames[a][col].mean()-frames[b][col].mean())
            ci.append(dict(Comparison=a+' - '+b,Metric=m,Difference=v,CI95Low=float(lo),CI95High=float(hi)))
    for name,data in [('comparison.csv',df),('validation_ranking.csv',ranking),('best_per_model.csv',best),('comparison_by_year.csv',ad),('paired_block_bootstrap.csv',pd.DataFrame(ci)),('all_coefficients.csv',pd.DataFrame(coefs)),('all_models_daily.csv',pd.concat([d.assign(Variant=n) for n,d in frames.items()]))]:csv(name,data)
    def native(row):return json.loads(row.to_json(force_ascii=False))
    candidate=dict(version='v29',variant=winner.Variant,model=winner.Model,multiplier=float(winner.Multiplier),selection_scope='all existing validation, 953 days',selection_order=['Sharpe descending','RankIC descending as tie-breaker'],rank_ic_positive_required=False,mse_used_for_selection=False,Sharpe=float(winner.Sharpe),RankIC=float(winner.RankIC),test_evaluated=False,status='validation_selected_candidate_only',training_loss='ordinary MSE',learning_rate='multiplier * min(A spectral rate, B spectral rate) each known training day',source_manifest=str(R/'manifest.json'),formal_baseline='v7')
    save('selected_candidate.json',candidate)
    result=dict(version='v29',audit_passed=True,ranking=json.loads(ranking.to_json(orient='records',force_ascii=False)),best_per_model=json.loads(best.to_json(orient='records',force_ascii=False)),selected_candidate=candidate,original_v7=original_result,paired_bootstrap=ci,test_evaluated=False,active_baseline='v7',training_financial_abs_limit=None,validation_financial_abs_limit=None)
    save('results.json',result)
    lines=['# v29：五種learning rate倍率的validation比較','','## 使用者指定的選擇規則','','Sharpe第一、Rank IC第二；Rank IC負值不直接淘汰，MSE不參與選擇。全期Sharpe相同才以Rank IC破同分。沿用普通MSE訓練loss，沒有改為直接最佳化Sharpe。這輪使用全部既有953日validation選候選，test未讀取或評估，正式v7不自動更換。','','## 結果', '',f'本輪15組中最高Sharpe為{MODELS[winner.Model]}、{winner.Multiplier:g}倍：Sharpe {winner.Sharpe:+.8f}，平均Rank IC {winner.RankIC:+.8f}。已記為待test驗證候選。原v7 Sharpe {original_result["Sharpe"]:+.8f}、Rank IC {original_result["RankIC"]:+.8f}。','','Sharpe未年化、未扣成本；驗證953日（2017-12-29至2021-12-01），Rank IC有效952日，Target全相同日未定義而不補0。所有倍率使用相同完整股票池。','','| 倍率 | 純g Sharpe | A覆蓋 Sharpe | B差額＋舊值 Sharpe |','|---|---:|---:|---:|']
    for c in MULT.values():lines.append(f'| {c:g} | '+' | '.join(f'{df[(df.Model==m)&(df.Multiplier==c)].Sharpe.iloc[0]:+.8f}' for m in MODELS)+' |')
    lines+=['','| 倍率 | 純g Rank IC | A覆蓋 Rank IC | B差額＋舊值 Rank IC |','|---|---:|---:|---:|']
    for c in MULT.values():lines.append(f'| {c:g} | '+' | '.join(f'{df[(df.Model==m)&(df.Multiplier==c)].RankIC.iloc[0]:+.8f}' for m in MODELS)+' |')
    lines+=['','## 各模型最佳倍率','','| 模型 | 倍率 | Sharpe | Rank IC |','|---|---:|---:|---:|']
    for q in best.itertuples():lines.append(f'| {q.Label} | {q.Multiplier:g} | {q.Sharpe:+.8f} | {q.RankIC:+.8f} |')
    lines+=['','## 每種倍率控制什麼','','A=g+beta×new；B=g+r1×(new-old)+r2×old。new、old均為v27同基期下的營業利益預測年比，使用最新狀態覆蓋、季度切換重建舊全年預測及exp(-a/9)衰減。','','eta_t(c)=c×min(eta_A,t,eta_B,t)，c固定為0.25、0.5、1、1.5或1.9。倍率整段固定、基準步長逐日變動。同一倍率下三模型的每日learning rate逐值相同。價量與財報共同從零訓練，每日一次更新，PR1/VR1原值，不排除極端值或裁切數值。財報來源、有效文件規則及EPS算法未改。','','## 各模型最佳設定的分期結果','','分期沿用既有ValidationYear標籤，不重新選每年的倍率。','','| 模型／倍率 | 2018 Sharpe | 2019 Sharpe | 2020 Sharpe | 2021 Sharpe |','|---|---:|---:|---:|---:|']
    for q in best.itertuples():
        a=ad[ad.Variant.eq(q.Variant)].set_index('ValidationYear');lines.append(f'| {q.Label}／{q.Multiplier:g} | '+' | '.join(f'{a.loc[y,"Sharpe"]:+.6f}' for y in [2018,2019,2020,2021])+' |')
    lines+=['','## 選擇後的探索性配對區間','','20日連續區塊、4000次同日期配對重抽樣，種子20260916。這些區間沒有校正15組挑最佳與過去模型選擇，不是test結果，也不作本輪排名門檻。','','| 比較 | 指標 | 差值 | 95%區間 |','|---|---|---:|---:|']
    for q in ci:lines.append(f'| {q["Comparison"]} | {q["Metric"]} | {q["Difference"]:+.8f} | [{q["CI95Low"]:+.8f}, {q["CI95High"]:+.8f}] |')
    lines+=['','## 判讀方式','','這輪測的是五個預先指定倍率的validation表現；最高者只是這個網格與資料期間的候選，不代表learning rate越大越好，亦不代表test已改善。保留完整五點曲線，避免只呈現最佳結果。正式基準維持v7，test留待之後總體驗證。','','## 核對','','15組各1199次更新、2,325,806訓練股票日、2,326,022預測，極端值排除0筆。每組已獨立核對所有更新梯度、已到期標籤時序、learning rate、參數歷程、預測、排名、Rank IC及官方Sharpe。c=1的三組與v28逐筆完全重現。同倍率三模型步長與股票日完全相同；歷史資料、特徵與結果未改。MSE僅留在低層數值核對記錄，不參與validation候選排名。']
    (R/'JPX-v29-learning-rate-sweep-report.md').write_text('\n'.join(lines)+'\n')
    version=dict(version='v29',description='Validation sweep of 0.25/0.5/1/1.5/1.9 times common learning rate; Sharpe primary, Rank IC secondary, no MSE selection',directory=str(R),status='completed_experiment',variants=df.Variant.tolist(),audit_passed=True,test_evaluated=False,selected_candidate=candidate,training_financial_abs_limit=None,validation_financial_abs_limit=None)
    save('version.json',version);reg=json.loads((B/'jpx_model_versions.json').read_text());assert reg['active_version']=='v7';reg['versions']=[v for v in reg['versions'] if v.get('version')!='v29']+[version];reg['latest_experiment']='v29';reg['latest_validation_candidate']=candidate;(B/'jpx_model_versions.json').write_text(json.dumps(reg,ensure_ascii=False,indent=2))
    print(ranking.to_string(index=False));print(best.to_string(index=False));print(ad[ad.Variant.isin(best.Variant)].to_string(index=False));print(pd.DataFrame(ci).to_string(index=False));print(json.dumps(candidate,ensure_ascii=False))
if __name__=='__main__':main()
