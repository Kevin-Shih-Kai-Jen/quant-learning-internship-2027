from pathlib import Path
import json,zipfile
R=Path(__file__).resolve().parent
source=(R.parent/'jpx_v30_profit_actual_forecast_20260916/mobile_pdf.py').read_text().split("page(1,")[0]
source=source.replace("out=R.parent/'output/pdf/JPX-v30-profit-actual-forecast-mobile-20260916.pdf'","out=R/'output/pdf/JPX-v32-chunk-replay-mobile-20260926.pdf';out.parent.mkdir(parents=True,exist_ok=True)")
source=source.replace('v30｜實際值與Forecast七組對照','v32｜分段複習與年度固定').replace('v30  /  2026-09-16','v32  /  2026-09-26');exec(source)
res=json.loads((R/'results.json').read_text());lookup=df.set_index('Variant');labels={'shared_online':'原 v7：每日更新','stock_online':'原 v31：逐檔每日更新','shared_frozen':'共用參數：年度固定','stock_single_frozen':'逐檔逐筆：年度固定','stock_chunk_frozen':'逐檔複習：年度固定'}
page(1,'比原 v7 高，仍未超越直接對照')
p('每次 validation 前完整訓練一次；validation 整年度固定參數。每檔自己的 12 個係數，原價量特徵不變。',11.5)
table([['模型','Sharpe','Rank IC']]+[[labels[q.Variant],f'{q.Sharpe:+.6f}',f'{q.RankIC:+.6f}'] for q in df.itertuples()],[154,94,94],9.8)
p('直接比較：分段複習減逐筆固定',14,teal)
p('ΔSharpe -0.007119；ΔRank IC -0.000243。<br/>兩組都逐檔、同資料、年度固定；因此這個比較才能看整套額外複習的效果。',10.5)
p('各年度 Sharpe',14,teal)
rows=[]
for year in sorted(yr.ValidationYear.unique()):
 a=yr[yr.ValidationYear==year].set_index('Variant');rows.append([str(year),f'{a.loc["stock_chunk_frozen","Sharpe"]:+.6f}',f'{a.loc["stock_single_frozen","Sharpe"]:+.6f}'])
table([['年份','分段複習','逐筆固定對照']]+rows,[74,134,134],11)
p('2020 年複習較好；另外三年較差。<br/>目前沒有看到跨年度一致的複習優勢。',11,teal)
p('相同 953 個 validation 日期，2018–2021 分期。Sharpe 未年化、未扣成本；Sharpe 優先、Rank IC 次之。MSE 只診斷，test 未使用。',10,gray)
p('原 v7 仍是正式基準。全年固定與每日更新是不同設定，不能把兩者的差距全部算成複習的貢獻。',10,gray)
page(2,'完整流程與這次能回答的問題')
p('按你確認的方式，整段歷史逐塊複習',14,teal)
p('每個年度 fold 重新從零開始，各階段接續：<br/>逐筆 1 次 → 5 筆塊 2 次 → 22 筆塊 4 次 → 60 筆塊 7 次 → 各歷史年度 √n 取整數次。',10.5)
p('只用該次預測前已到期的報酬。尾塊保留；n 用實際有效筆數，年度不直接用 365。未來年度不參與當次訓練。',10.5)
p('常態 MLE 這次等價於 MSE',14,teal)
p('最小化平均負 log likelihood，固定誤差尺度：<br/>NLL = 常數 + 0.5 × MSE。<br/>配合相應步長，兩者係數更新完全相同；沒有新增波動參數，也未校準報酬分布。',10.5)
p('差距的不確定性仍大',14,teal)
p('分段複習減逐筆固定，20 日區塊配對重抽樣 4,000 次：<br/>ΔSharpe 95% 區間 [-0.079007, +0.072478]<br/>ΔRank IC 95% 區間 [-0.003278, +0.003074]',10.5)
p('區間都包含 0；既有 validation 也已反覆用於開發，不能確認長期穩定優勢或劣勢。',10.5)
p('這次尚未回答「資料是否不足」',14,teal)
p('重複複習没有增加獨立樣本，也沒有減少每股 12 個參數。未做同一未來期間的資料長度比較，或相同計算量的其他更新方式對照。',10.5)
p('共 8,215,561 次逐檔更新已以獨立梯度重播；全部五組預測、排名與官方指標核對通過。無極端值篩選，完整軌跡保存到私人版本化快照。',9.5,gray)
c.save()
archive=R/'output/JPX-v32-chunk-replay-data-20260926.zip'
files=['comparison.csv','comparison_by_year.csv','paired_block_bootstrap.csv','all_models_daily.csv','training_stages_summary.csv','fold_summary.csv','per_stock_diagnostics.csv','all_stage_coefficients.csv','JPX-v32-chunk-replay-report.md','experiment_plan.md','config.json','experiment_defaults_snapshot.json','manifest.json','audit.json','results.json','version.json']
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
 z.write(out,out.name)
 for n in files:z.write(R/n,n)
 for variant in labels:
  for n in ['daily_metrics.csv','results.json','audit.json']:z.write(R/variant/n,variant+'/'+n)
print(json.dumps({'pdf':str(out),'zip':str(archive),'zip_bytes':archive.stat().st_size}))
