from pathlib import Path
import argparse,importlib.util,json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
R=Path(__file__).resolve().parent;spec=importlib.util.spec_from_file_location('v31',R/'run.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
LABELS={'shared_v7':'原v7：全部股票共用12參數','stock_specific':'逐檔v7：每股獨立12參數'}
def csv(n,d):d.to_csv(R/n,index=False,encoding='utf-8-sig')
def sharp(x):return float(np.mean(x)/np.std(x,ddof=1))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--task-dir',type=Path,required=True);args=ap.parse_args();audit=json.loads((R/'audit.json').read_text());assert audit['passed'];f,x,groups,cal=r.load(args.task_dir);frames={n:pd.read_csv(R/n/'daily_metrics.csv') for n in LABELS};pd.testing.assert_series_equal(frames['shared_v7'].Date,frames['stock_specific'].Date)
 common=np.logical_and.reduce([d.OfficialDailySpread.notna().to_numpy() for d in frames.values()]);rows=[];annual=[];results={n:json.loads((R/n/'results.json').read_text()) for n in LABELS}
 for n,d in frames.items():
  q=d[common];rows.append(dict(Variant=n,Label=LABELS[n],Sharpe=sharp(q.OfficialDailySpread),RankIC=float(q.RankIC.mean()),MatchedScoredDays=len(q),ScoredDaysAll=int(d.OfficialDailySpread.notna().sum()),RankICDays=int(q.RankIC.notna().sum()),MSE=float(q.AllStockForecastMSE.mean()),MeanAbsoluteScore=float(q.MeanAbsoluteScore.mean()),ParameterVectors=results[n]['parameter_vectors'],TotalParameters=results[n]['total_parameters']))
  for year,g in q.groupby('ValidationYear'):annual.append(dict(Variant=n,ValidationYear=int(year),Days=len(g),Sharpe=sharp(g.OfficialDailySpread),RankIC=float(g.RankIC.mean())))
 df=pd.DataFrame(rows);ad=pd.DataFrame(annual);by=df.set_index('Variant');rng=np.random.default_rng(20260925);N=int(common.sum());block=20;reps=4000;ix=(rng.integers(0,N-block+1,size=(reps,int(np.ceil(N/block))))[:,:,None]+np.arange(block)).reshape(reps,-1)[:,:N];boot={}
 for n,d in frames.items():
  q=d[common];a=q.OfficialDailySpread.to_numpy()[ix];boot[n]={'Sharpe':a.mean(axis=1)/a.std(axis=1,ddof=1),'RankIC':np.nanmean(q.RankIC.to_numpy()[ix],axis=1)}
 intervals=[]
 for metric in ['Sharpe','RankIC']:
  lo,hi=np.quantile(boot['stock_specific'][metric]-boot['shared_v7'][metric],[.025,.975]);intervals.append(dict(Metric=metric,Difference=float(by.loc['stock_specific',metric]-by.loc['shared_v7',metric]),CI95Low=float(lo),CI95High=float(hi)))
 predictions={};history={}
 for n in LABELS:
  with np.load(R/n/'predictions.npz') as p:predictions[n]=p['score'];history[n]=p['own_known_updates']
 valid=f.SignalDate.isin(cal.Date).to_numpy();y=f.Target.to_numpy();stock=[]
 for code,ids in f.groupby('SecuritiesCode').indices.items():
  ids=ids[valid[ids]&np.isfinite(y[ids])];row=dict(SecuritiesCode=int(code),ValidationObservations=len(ids))
  for n in LABELS:
   pred=predictions[n][ids];row[n+'_TimeSeriesRankIC']=float(spearmanr(pred,y[ids]).statistic) if len(ids)>=20 and np.std(pred)>0 and np.std(y[ids])>0 else None
   row[n+'_MeanAbsoluteScore']=float(np.mean(np.abs(pred))) if len(ids) else None
  if len(ids):row.update(FirstValidationKnownUpdates=int(history['stock_specific'][ids[0]]),LastValidationKnownUpdates=int(history['stock_specific'][ids[-1]]))
  stock.append(row)
 stock=pd.DataFrame(stock);comparable=stock[['shared_v7_TimeSeriesRankIC','stock_specific_TimeSeriesRankIC']].dropna();stock_diag=dict(stock_time_series_comparable=len(comparable),fraction_stock_time_series_ic_improved=float((comparable.stock_specific_TimeSeriesRankIC>comparable.shared_v7_TimeSeriesRankIC).mean()),median_shared_stock_time_series_ic=float(comparable.shared_v7_TimeSeriesRankIC.median()),median_individual_stock_time_series_ic=float(comparable.stock_specific_TimeSeriesRankIC.median()),validation_zero_own_history_stock_days=int((history['stock_specific'][valid]==0).sum()),validation_under12_own_updates_stock_days=int((history['stock_specific'][valid]<12).sum()),validation_stock_days=int(valid.sum()))
 np.testing.assert_array_equal(history['shared_v7'],history['stock_specific'])
 for n,d in [('comparison.csv',df),('comparison_by_year.csv',ad),('paired_block_bootstrap.csv',pd.DataFrame(intervals)),('per_stock_diagnostics.csv',stock),('all_models_daily.csv',pd.concat([d.assign(Variant=n) for n,d in frames.items()]))]:csv(n,d)
 r.save(R/'diagnostics.json',stock_diag)
 res=dict(version='v31',comparison=rows,paired_block_bootstrap=intervals,diagnostics=stock_diag,audit_passed=True,MSE_only=True,MLE_cancelled_by_user=True,test_evaluated=False,formal_baseline='v7',validation_period=['2017-12-29','2021-12-01'],source_snapshot='snapshot-2026-09-24');r.save(R/'results.json',res)
 diff=by.loc['stock_specific','Sharpe']-by.loc['shared_v7','Sharpe'];direction='高於' if diff>0 else '低於';nstock=results['stock_specific']['parameter_vectors']
 lines=['# v31：每股獨立v7參數的MSE回測報告','','2026-09-25。依使用者最後確認，本輪只用MSE，不跑MLE。參考私人儲存庫AGENTS.md、MEMORY_BRIEF、data/index及正式v7報告；原始輸入由已核對快照按需取回。','','## 結果', '',f'逐檔模型在相同{N}個可評分validation日期，Sharpe {direction}原v7，差值{diff:+.8f}。逐檔獨立參數共{nstock:,}套、每套12個，合計{nstock*12:,}個；原v7只有全市場共用12個。這是目前特徵與每日一步訓練方式的實驗，不能推廣成所有逐檔模型都好或不好。','','| 模型 | Sharpe | 平均橫斷面Rank IC | 日均預測MSE（診斷） |','|---|---:|---:|---:|']
 for q in rows:lines.append(f'| {q["Label"]} | {q["Sharpe"]:+.8f} | {q["RankIC"]:+.8f} | {q["MSE"]:.9f} |')
 lines+=['','Sharpe未年化、未扣交易成本；仍為2018–2021既有validation，並非新test。每天按全股票預測排序，多空各200檔、側內2→1權重。Sharpe第一、Rank IC第二，MSE不參與選擇。', '',f'預測股票池共953個validation日。共同可評分日{N}個；無法評分的入選缺失Target日完整保留，不補0、不改排名。Rank IC在真實Target全相同日未定義，不補0。', '', '## 分期結果','','| 分期 | 日期數 | 共用v7 Sharpe | 逐檔 Sharpe | 共用 Rank IC | 逐檔 Rank IC |','|---|---:|---:|---:|---:|---:|']
 for year in sorted(ad.ValidationYear.unique()):
  a=ad[(ad.Variant=='shared_v7')&(ad.ValidationYear==year)].iloc[0];b=ad[(ad.Variant=='stock_specific')&(ad.ValidationYear==year)].iloc[0];lines.append(f'| {year} | {a.Days} | {a.Sharpe:+.6f} | {b.Sharpe:+.6f} | {a.RankIC:+.6f} | {b.RankIC:+.6f} |')
 lines+=['','## 逐檔模型如何訓練','','保留v7的T5/22/60、V5/22/60、三個同期間乘積、原值PR1/VR1，加截距共12項。theta_i由SecuritiesCode索引；每檔只吃自己的已到期標籤、特徵與歷史參數。2017零初始化與暖身、跨年接續；新股票從零開始，不使用其他股票的係數。缺Target跳過該筆訓練，預測股票池不因標籤缺失改變。沒有財報、標準化、裁切、極端值篩選、Ridge或歷史回放。','','L_i=(x_i^T theta_i-y_i)^2；grad_i=2 x_i (x_i^T theta_i-y_i)。沿用v7的曲率步長規則，但每檔每日只有1筆，因此eta_i=1/(2||x_i||²)，theta_i_new=theta_i-x_i(x_i^T theta_i-y_i)/||x_i||²。','','每次只更新一次，當批訓練誤差因此可接近0；不代表下一期預測精確。參數保留過去更新的影響，但本次loss只含新到期的該筆資料，沒有每次把歷史全部重新擬合。線性平方誤差梯度可參考[Stanford CS229 LMS說明](https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-supervised-learning/)，上述逐檔步長與投影性質由本輪公式直接推得並核對。','','參數與步長都只依自己股票決定，避免跨股參數影響。相較共用v7，同時改變了樣本共享與曲率步長的計算層級，不能宣稱純粹只改係數數量、步長數值完全不變。','','## 個股股性診斷', '',f'有足夠validation資料可比較的股票{stock_diag["stock_time_series_comparable"]:,}檔，其中逐檔「跨時間Rank IC」優於共用模型的比例{stock_diag["fraction_stock_time_series_ic_improved"]:.2%}。各股時間序列Rank IC中位數：共用{stock_diag["median_shared_stock_time_series_ic"]:+.6f}、逐檔{stock_diag["median_individual_stock_time_series_ic"]:+.6f}。此指標是在每檔股票自己的不同日期比較，不能替代每天全市場橫斷面排名與Sharpe。', '',f'validation股票日共{stock_diag["validation_stock_days"]:,}，尚無自身已知標籤的股票日{stock_diag["validation_zero_own_history_stock_days"]:,}，少於12次自身更新的股票日{stock_diag["validation_under12_own_updates_stock_days"]:,}。所有股票仍納入預測，不因資料短或極端值而篩掉。逐股明細見per_stock_diagnostics.csv，完整係數見stock_specific/final_stock_parameters.csv。','','## 配對不確定性','','20個連續交易日區塊、4000次共同日期重抽樣，種子20260925。探索性95%區間，未校正過往反覆模型選擇。','','| 逐檔減共用 | 差值 | 95%區間 |','|---|---:|---:|']
 for q in intervals:lines.append(f'| {q["Metric"]} | {q["Difference"]:+.8f} | [{q["CI95Low"]:+.8f}, {q["CI95High"]:+.8f}] |')
 lines+=['','## 核對、來源及可重取性','','- 兩模型各1199個到期批次，2,325,806筆有效訓練股票日與2,326,022筆預測；逐檔模型每筆只更新對應股票的12個參數。','- 原共用v7排名逐筆重現來源快照；逐檔模型按每檔自己的到期順序獨立重播全部參數與預測，沒有使用其他股票標籤。','- 每日排序、Rank IC、選股Target完整性、官方Sharpe及冷啟動更新數均獨立核對；保存逐筆參數更新軌跡，非僅期末係數。','- 原v7保持正式基準；未讀取或使用保留test。未跑MLE、未跑每檔歷史最小平方法或其他調參。','- 來源Release：snapshot-2026-09-24；輸入與對照的SHA-256見manifest.json，取回核對見retrieval_verification.json。新程式、結果、PDF與必要逐筆證據依私人repo保存流程同步，同步／清理狀態另列交付紀錄。']
 (R/'JPX-v31-stock-specific-mse-report.md').write_text('\n'.join(lines)+'\n');r.save(R/'version.json',dict(version='v31',status='completed_experiment',date='2026-09-25',directory=R.name,variants=list(LABELS),formal_baseline='v7',test_evaluated=False,audit_passed=True))
 print(df.to_string(index=False));print(ad.to_string(index=False));print(pd.DataFrame(intervals).to_string(index=False));print(json.dumps(stock_diag))
if __name__=='__main__':main()
