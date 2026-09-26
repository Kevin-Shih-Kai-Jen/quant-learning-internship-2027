from pathlib import Path
import json, argparse, importlib.util
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
R=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('v32',R/'run.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
LABELS={'shared_online':'原v7：共用參數、每日更新','stock_online':'原v31：逐檔參數、每日更新','shared_frozen':'共用參數、年度固定','stock_single_frozen':'逐檔只逐筆訓練、年度固定','stock_chunk_frozen':'逐檔分段複習、年度固定'}
def csv(n,v):v.to_csv(R/n,index=False,encoding='utf-8-sig')
def sharp(v):return float(np.mean(v)/np.std(v,ddof=1))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--task-dir',type=Path,required=True);a=ap.parse_args();assert json.loads((R/'audit.json').read_text())['passed'];f,x,groups,cal=r.load(a.task_dir);frames={n:pd.read_csv(R/n/'daily_metrics.csv') for n in LABELS};common=np.logical_and.reduce([d.OfficialDailySpread.notna().to_numpy() for d in frames.values()]);rows=[];annual=[];raw=json.loads((R/'raw_results.json').read_text())
 for n,d in frames.items():
  pd.testing.assert_series_equal(d.Date,frames['shared_online'].Date);g=d[common];rows.append(dict(Variant=n,Label=LABELS[n],Sharpe=sharp(g.OfficialDailySpread),RankIC=float(g.RankIC.mean()),MSE=float(g.MSE.mean()),CommonScoredDays=len(g),AllScoredDays=raw[n]['scored_days'],RankICDays=int(g.RankIC.notna().sum()),ZeroHistoryStockDays=int(d.ZeroTrainingHistoryStocks.sum())))
  for year,q in g.groupby('ValidationYear'):annual.append(dict(Variant=n,ValidationYear=int(year),Days=len(q),Sharpe=sharp(q.OfficialDailySpread),RankIC=float(q.RankIC.mean()),MSE=float(q.MSE.mean())))
 comparison=pd.DataFrame(rows);ad=pd.DataFrame(annual);lookup=comparison.set_index('Variant');N=int(common.sum());rng=np.random.default_rng(20260926);ix=(rng.integers(0,N-20+1,size=(4000,int(np.ceil(N/20))))[:,:,None]+np.arange(20)).reshape(4000,-1)[:,:N];boots={}
 for n,d in frames.items():
  d=d[common];sp=d.OfficialDailySpread.to_numpy()[ix];boots[n]=dict(Sharpe=sp.mean(axis=1)/sp.std(axis=1,ddof=1),RankIC=np.nanmean(d.RankIC.to_numpy()[ix],axis=1))
 pairs=[('stock_chunk_frozen',n) for n in ['stock_single_frozen','shared_frozen','shared_online','stock_online']]+[('stock_single_frozen','stock_online'),('shared_frozen','shared_online')];cis=[]
 for aa,bb in pairs:
  for m in ['Sharpe','RankIC']:
   lo,hi=np.quantile(boots[aa][m]-boots[bb][m],[.025,.975]);cis.append(dict(ModelA=aa,ModelB=bb,Metric=m,Difference=float(lookup.loc[aa,m]-lookup.loc[bb,m]),CI95Low=float(lo),CI95High=float(hi)))
 stage=[];foldrows=[];coef=[];manifest=json.loads((R/'manifest.json').read_text())
 for fold in manifest['folds']:
  year=fold['validation_year'];p=np.load(R/f'fold_{year}'/'stage_parameters.npz');counts=p['train_counts'];pars=p['coefficients'];s=pd.read_csv(R/f'fold_{year}'/'training_stage_diagnostics.csv');stage.append(s);foldrows.append(dict(**fold,median_stock_train_rows=float(np.median(counts)),stock_train_rows_lt12=int((counts<12).sum()),stock_train_rows_min=int(counts.min()),stock_train_rows_max=int(counts.max())))
  for st,theta in enumerate(pars):
   z=pd.DataFrame(theta,columns=r.NAMES);z.insert(0,'Stage',r.STAGES[st]);z.insert(0,'SecuritiesCode',p['codes']);z.insert(0,'ValidationYear',year);coef.append(z)
  p.close()
 st=pd.concat(stage,ignore_index=True);summary=[]
 for (year,stage),d in st.groupby(['ValidationYear','Stage'],sort=False):
  summary.append(dict(ValidationYear=int(year),Stage=stage,TrainingUpdates=int(d.Updates.sum()),EqualStockMeanHistoryMSE=float(d.HistoryMSE.mean()),ObservationWeightedHistoryMSE=float(np.nansum(d.HistoryMSE*d.TrainRows)/d.TrainRows.sum()),MedianCoefficientNorm=float(d.CoefficientNorm.median())))
 valid=f.SignalDate.isin(cal.Date).to_numpy();scores={n:np.load(R/n/'predictions.npz')['score'] for n in ['stock_single_frozen','stock_chunk_frozen']};stock=[];y=f.Target.to_numpy()
 for code,idx in f.groupby('SecuritiesCode').indices.items():
  ids=idx[valid[idx]&np.isfinite(y[idx])];q=dict(SecuritiesCode=int(code),ValidationObservations=len(ids))
  for n in scores:
   pred=scores[n][ids];q[n+'_TimeSeriesIC']=float(spearmanr(pred,y[ids]).statistic) if len(ids)>=20 and np.std(pred)>0 and np.std(y[ids])>0 else None
  stock.append(q)
 stock=pd.DataFrame(stock);comparable=stock.dropna();fraction=float((comparable.stock_chunk_frozen_TimeSeriesIC>comparable.stock_single_frozen_TimeSeriesIC).mean());ss=pd.DataFrame(summary);fold_table=pd.DataFrame(foldrows)
 for name,frame in [('comparison.csv',comparison),('comparison_by_year.csv',ad),('paired_block_bootstrap.csv',pd.DataFrame(cis)),('all_models_daily.csv',pd.concat([d.assign(Variant=n) for n,d in frames.items()])),('training_stages_summary.csv',ss),('fold_summary.csv',fold_table),('all_stage_coefficients.csv',pd.concat(coef)),('per_stock_diagnostics.csv',stock)]:csv(name,frame)
 mainci=next(v for v in cis if v['ModelB']=='stock_single_frozen' and v['Metric']=='Sharpe');diag=dict(comparable_stocks=len(comparable),fraction_stock_time_ic_improved=fraction,history_mse_is_diagnostic_only=True)
 res=dict(version='v32',date='2026-09-26',comparison=rows,folds=foldrows,paired_intervals=cis,diagnostics=diag,audit_passed=True,formal_baseline='v7',test_evaluated=False);r.save(R/'results.json',res)
 main=lookup.loc['stock_chunk_frozen'];control=lookup.loc['stock_single_frozen'];base=lookup.loc['shared_online'];lines=['# v32：逐檔歷史分塊複習，年度固定 validation','','2026-09-26。使用者指定整段歷史切塊，僅在每次 validation 前訓練一次；validation 整年度固定參數。','','## 本輪結果','',f'主實驗 Sharpe {main.Sharpe:+.8f}、Rank IC {main.RankIC:+.8f}；直接對照（同樣逐檔與年度固定，只做逐筆階段）Sharpe {control.Sharpe:+.8f}、Rank IC {control.RankIC:+.8f}。差值 ΔSharpe {main.Sharpe-control.Sharpe:+.8f}、ΔRank IC {main.RankIC-control.RankIC:+.8f}。原 v7 每日更新 Sharpe {base.Sharpe:+.8f}。','','| 模型 | Sharpe | Rank IC | 日均預測MSE：診斷 |','|---|---:|---:|---:|']
 for q in rows:lines.append(f'| {q["Label"]} | {q["Sharpe"]:+.8f} | {q["RankIC"]:+.8f} | {q["MSE"]:.9f} |')
 lines+=['',f'預測股票池保留全部953個原validation日期，共同可評分日{N}。不因極端值或缺Target刪預測股票；缺Target不訓練，入選缺Target的日期不補0，詳細日誌保留。Rank IC的真實Target全相同日為未定義。Sharpe未年化、未扣成本；Sharpe第一、Rank IC第二，MSE不選模型。這是已反覆開發的歷史validation，不是新test。','','## 各年度結果','','| 年度 | 日期數 | 分段複習 | 逐筆固定對照 | 原v7每日更新 |','|---|---:|---:|---:|---:|']
 for year in sorted(ad.ValidationYear.unique()):
  z=ad[ad.ValidationYear==year].set_index('Variant');lines.append(f'| {year} | {int(z.iloc[0].Days)} | {z.loc["stock_chunk_frozen","Sharpe"]:+.6f} | {z.loc["stock_single_frozen","Sharpe"]:+.6f} | {z.loc["shared_online","Sharpe"]:+.6f} |')
 lines+=['','## 實際訓練方式','','每個年度fold，每檔自己的12參數從零開始；同一fold各阶段接續，不按chunk歸零。原v7的11價量特徵與截距不變，無財報、標準化、正則或極端值篩選。只使用該fold首個訊號日收盤已到期的Target，且SignalDate早於該日。全年不再更新；下一fold把已到期歷史納入expanding training，再重新訓練。','','1. 全歷史逐筆各更新一次。','2. 從歷史起點切不重疊5筆區塊，每完整塊更新2次。','3. 重新把全歷史切22筆區塊，每完整塊更新4次。','4. 重新把全歷史切60筆區塊，每完整塊更新7次。','5. 每個歷史日曆年度各成一塊，按舊到新更新floor(sqrt(n))次。','', '各層重用相同歷史，不增加獨立樣本數。n是該檔實際有效觀測筆數，尾塊保留並以實際n計次，年度不使用365。每次用当前參數重算區塊平均loss。新股票若訓練時完全無自身歷史，全年保持零係數並納入排名，沒有借用共用參數。','','| validation | 擬合時點：收盤 | 歷史訓練列 | 有歷史股票 | 每股筆數中位數 | 逐筆＋複習總更新 |','|---|---|---:|---:|---:|---:|']
 for q in foldrows:lines.append(f'| {q["validation_year"]} | {q["fit_at_signal_close"]} | {q["train_rows"]:,} | {q["unique_training_stocks"]:,} | {q["median_stock_train_rows"]:.0f} | {q["training_updates"]:,} |')
 lines+=['','## 常態MLE與MSE的關係','','採最小化平均負log likelihood，固定sigma=1作數值尺度：NLL=0.5 log(2π)+0.5 mean((Xθ-y)²)。此值不表示真實誤差波動為100%，沒有估計或校準報酬分布。每區塊η_NLL=1/λmax(XᵀX/n)，等價MSE步長為其一半，因此係數更新與MSE相同，已逐步核對。沒有把log likelihood最小化，也未新增σ參數。此種常態等變異數MLE的平均係數等價最小平方，見[CMU講義](https://www.stat.cmu.edu/~cshalizi/mreg/15/lectures/06/lecture-06.pdf)。','','## 如何解讀差異','',f'主實驗減直接對照的ΔSharpe={mainci["Difference"]:+.8f}；20日區塊、4000次配對重抽樣，探索性95%區間[{mainci["CI95Low"]:+.8f}, {mainci["CI95High"]:+.8f}]。未校正多次validation選模；區間若跨0，不確認穩定優勢或劣勢。', '', '主實驗與stock_single_frozen只差後續歷史複習，可以評估整套複習的影響；但複習增加計算量，也改變更新順序，本輪未做等計算量對照，不能歸因於5/22/60本身比任意複習更好。與原v7/v31每日更新相比，另有validation凍結與每fold重建流程的差異，不能混成單一原因。', '',f'能比較個股跨時間Rank IC的股票{len(comparable):,}檔，分段複習高於逐筆固定對照的比例{fraction:.2%}。這與每日橫斷面Rank IC不同，明細見per_stock_diagnostics.csv。', '',f'逐檔年度固定兩組各有無訓練歷史股票日{int(main.ZeroHistoryStockDays):,}。逐檔模型的冷啟動股票保留零預測；沒有因樣本不足排除股票。訓練每階段的全歷史MSE只用於檢查複習，不作選模。','','不能用本次結果直接證明「參數太多」或「資料不足」：沒有改每股12參數，也沒有在同一未來期間控制性比較不同歷史長度。不同fold年代的市場狀況不同，不能把逐年差異直接當學習曲線。正式基準仍v7，test未使用。','','## 核對與保存','','全部fold逐股逐塊、每次參數更新以獨立殘差梯度重播，SVD核對學習步長；時間界線、NLL/MSE等價、年度凍結預測、排序、Rank IC與官方Sharpe均通過。保存每步12維係數，不只保留年度期末結果。原v7/v31參照預測從已核對私人快照取回，原內容不變，同日指標重新計算。','','來源快照snapshot-2026-09-25-v31-stock-mse，原始輸入解壓與SHA256已核對。新結果使用獨立v32版本；同步與本次暫存清理狀態見data/verification下的交付紀錄。']
 (R/'JPX-v32-chunk-replay-report.md').write_text('\n'.join(lines)+'\n');r.save(R/'version.json',dict(version='v32',status='completed_experiment',date='2026-09-26',formal_baseline='v7',audit_passed=True,test_evaluated=False));print(comparison.to_string(index=False));print(ad.to_string(index=False));print(json.dumps(mainci))
if __name__=='__main__':main()
