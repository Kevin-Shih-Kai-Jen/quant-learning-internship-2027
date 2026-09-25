from pathlib import Path
import json, zipfile
R=Path(__file__).resolve().parent
source=(R.parent/'jpx_v30_profit_actual_forecast_20260916/mobile_pdf.py').read_text().split("page(1,")[0]
source=source.replace("out=R.parent/'output/pdf/JPX-v30-profit-actual-forecast-mobile-20260916.pdf'", "out=R/'output/pdf/JPX-v31-stock-specific-mse-mobile-20260925.pdf';out.parent.mkdir(parents=True,exist_ok=True)")
source=source.replace('v30｜實際值與Forecast七組對照','v31｜逐檔獨立參數MSE回測').replace('v30  /  2026-09-16','v31  /  2026-09-25')
exec(source)
page(1,'每檔獨立參數，這次未改善')
p('只用 MSE。保留原 v7 價量特徵，每檔股票各有一套係數，只用自己的已到期報酬更新。沒有加入財報。',11.5)
p('2,000 套 × 12 個係數 = 24,000 個參數。<br/>對照組：原 v7 全市場共用 12 個參數。',11.5,teal)
labels={'shared_v7':'原 v7 共用','stock_specific':'逐檔獨立'}
table([['模型','Sharpe','Rank IC']]+[[labels[q.Variant],f'{q.Sharpe:+.6f}',f'{q.RankIC:+.6f}'] for q in df.itertuples()],[116,113,113],11)
p('逐檔減共用：Sharpe -0.026876；Rank IC -0.002032。兩個主要指標都下降，這輪不支持取代原 v7。',11.5,teal)
p('每年表現是否一致？',14,teal)
years=sorted(yr.ValidationYear.unique())
rows=[]
for yy in years:
    a=yr[(yr.ValidationYear==yy)&(yr.Variant=='shared_v7')].iloc[0];b=yr[(yr.ValidationYear==yy)&(yr.Variant=='stock_specific')].iloc[0]
    rows.append([str(yy),f'{a.Sharpe:+.6f}',f'{b.Sharpe:+.6f}'])
table([['年份','共用 Sharpe','逐檔 Sharpe']]+rows,[74,134,134],11)
p('2018、2019、2020 逐檔較差；2021 逐檔較好，但兩組仍為負。因此不能說每個時期都變差。',10.5)
p('953 個相同 validation 日期，2018–2021 分期。Sharpe 未年化、未扣交易成本。Sharpe 優先、Rank IC 次之；MSE 僅供診斷。',10,gray)
p('原 v7 預測與排名已重現；逐檔版每次更新及預測已獨立重播核對。未排除極端值、未用 test，正式基準維持 v7。',10,gray)
page(2,'為什麼還不能否定「股性」？')
p('1｜這個更新方式相當積極',14,teal)
p('逐檔每日只有一筆新到期資料。沿用 v7 的步長規則，會得到：',10.5)
p('η = 1 / (2 ||x||²)<br/>θ新 = θ舊 − x(xᵀθ舊 − y) / ||x||²',12)
p('因此每次更新幾乎完全貼合剛到期的那一筆。當筆訓練誤差接近 0，不代表未來更準；下一期預測仍可能大幅變動。',10.5)
p('本次改了參數共享方式，也改了步長計算層級。結果不能完全歸因於「各股獨立」本身。',10.5,teal)
p('2｜個股內部的排序也未普遍改善',14,teal)
p('2,000 檔中，45.70% 的個股跨時間 Rank IC 改善。中位數從 +0.001468 變成 -0.002517。這與每天全市場排序的 Rank IC 是不同診斷。',10.5)
p('3｜差距仍有不確定性',14,teal)
p('20 日區塊配對重抽樣，4,000 次：<br/>ΔSharpe 95% 區間 [-0.108148, +0.055851]<br/>ΔRank IC 95% 區間 [-0.008392, +0.003593]',10.5)
p('區間包含 0。觀察結果較差，但不能宣稱已證明長期一定較差；既有 validation 也已反覆用於開發。',10.5)
p('判讀與下一個可分開驗證的問題',13,teal)
p('目前不換模型。若繼續，先固定這次逐檔特徵與資料，只降低更新幅度，確認是否過度追逐單筆報酬；這個後續實驗尚未執行。',10.5)
p('完整 CSV、逐股診斷與設定見隨附資料包；完整預測與逐筆係數軌跡另存私人版本化快照。來源、公式與核對細節見完整版報告。',9.5,gray)
c.save()
archive=R/'output/JPX-v31-stock-specific-mse-data-20260925.zip'
names=['comparison.csv','comparison_by_year.csv','paired_block_bootstrap.csv','per_stock_diagnostics.csv','all_models_daily.csv','diagnostics.json','JPX-v31-stock-specific-mse-report.md','experiment_plan.md','experiment_defaults_snapshot.json','manifest.json','audit.json','results.json','version.json']
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    z.write(out,out.name)
    for name in names:z.write(R/name,name)
    for variant in ['shared_v7','stock_specific']:
        for name in ['daily_metrics.csv','training_updates.csv','results.json','audit.json']:z.write(R/variant/name,variant+'/'+name)
    z.write(R/'stock_specific/final_stock_parameters.csv','stock_specific/final_stock_parameters.csv')
print(json.dumps({'pdf':str(out),'zip':str(archive),'zip_bytes':archive.stat().st_size}))
