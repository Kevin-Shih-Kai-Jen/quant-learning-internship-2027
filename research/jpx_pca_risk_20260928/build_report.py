"""Build a self-contained Chinese HTML report and exact research figures."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'final_results'
R=json.loads((OUT/'results.json').read_text())
A=json.loads((OUT/'data_audit.json').read_text())
market,paired=R['cases']; v7=paired['portfolios']['v7']; ew=paired['portfolios']['equal_long']
pct=lambda x:f'{100*x:.2f}%'
pc1=paired['components'][0]

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
    'axes.spines.right':False,'svg.fonttype':'none','savefig.facecolor':'white'})
colors=['#246a8d','#d56b3a','#32947d']
components=pd.DataFrame(paired['components'])
fig,ax=plt.subplots(figsize=(10,4.8))
x=np.arange(11);width=.25
for offset,(col,label,color) in enumerate([
    ('market_variance_share','Universe: sum of stock variances',colors[0]),
    ('v7_portfolio_variance_share','v7: portfolio variance',colors[1]),
    ('equal_long_portfolio_variance_share','Equal-long: portfolio variance',colors[2])]):
    vals=np.r_[components[col].iloc[:10],components[col].iloc[10:].sum()]*100
    ax.bar(x+(offset-1)*width,vals,width,label=label,color=color)
ax.set_xticks(x,[f'PC{i}' for i in range(1,11)]+['PC11+'])
ax.set_ylabel('Share of the stated variance (%)');ax.set_ylim(0,105)
ax.set_title(f'Same PCA basis: {paired["stocks"]:,} stocks, {paired["dates"]} jointly observed daily returns',loc='left',weight='bold')
ax.legend(frameon=False,fontsize=8.8,loc='upper right');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.text(.09,.015,'Fixed holdings dated 2021-12-01; sample risk only. Joint-observation dates end 2021-11-29.',fontsize=9,color='#555')
fig.tight_layout(rect=(0,.04,1,1));fig.savefig(OUT/'risk_comparison.svg');fig.savefig('/workspace/scratch/c2c7725b1e28/jpx_pca_risk_chart.png',dpi=150);plt.close(fig)

fig,(ax,bx)=plt.subplots(1,2,figsize=(10,4))
for c,color in [(market,colors[0]),(paired,colors[1])]:
    d=pd.DataFrame(c['components']);label=f'{c["dates"]} dates / {c["stocks"]:,} stocks'
    ax.plot(d.PC,d.market_variance_share*100,label=label,color=color)
    bx.plot(d.PC,d.cumulative_market_share*100,label=label,color=color)
ax.set_yscale('log');ax.set_ylabel('Explained variance (%) — log scale');ax.set_xlabel('PC number')
bx.set_ylabel('Cumulative explained variance (%)');bx.set_xlabel('PC number');bx.set_ylim(0,101)
for a in [ax,bx]:a.grid(alpha=.15);a.legend(frameon=False,fontsize=8)
fig.suptitle('Two explicitly different estimation samples',x=.08,ha='left',weight='bold')
fig.tight_layout();fig.savefig(OUT/'eigenspectrum.svg');plt.close(fig)

top=components.nlargest(5,'v7_portfolio_variance_share')
top_rows='\n'.join(f'| PC{int(z.PC)} | {pct(z.market_variance_share)} | {pct(z.v7_portfolio_variance_share)} |' for z in top.itertuples())
pc6_pos=[x for x in paired['top_loadings'] if x['PC']==6 and x['side']=='positive'][:5]
pc6_neg=[x for x in paired['top_loadings'] if x['PC']==6 and x['side']=='negative'][:5]
stock_lines=lambda rows:'、'.join(f'{x["SecuritiesCode"]} {x["Name"]}（{x["loading"]:+.4f}）' for x in rows)
note=f'''# JPX PCA：市場與實際持倉風險

分析日：2026-09-28。歷史訊號切點：2021-12-01 收盤後。正式基準：v7_equal。

最值得你注意的結果是：在同一組 PCA 基底下，PC1 解釋股票池總變異數的 **{pct(paired['pc1_share'])}**，卻只占 v7 多空持倉變異數的 **{pct(v7['first_pc_share'])}**；若改為同一股池的等權多頭，PC1 會占其風險 **{pct(ew['first_pc_share'])}**。你的「先找共同波動，再看持倉是否集中在那些方向」這個用途成立，而且這個例子直接展示了兩種占比為何不能互換。

這是固定 2021-12-01 權重的歷史共變異數診斷。不能把它解讀成 v7 每日調倉的實現波動、未來風險保證或新的樣本外績效。

## 先確認資料與估計範圍

你的 raw.zip 內 stock_prices.csv 完整，共 **2,332,531 筆**、2017-01-04～2021-12-03，SHA-256 與歷史紀錄一致。這次另由 Release 還原 bars 快取，全部 2,332,531 筆 OHLCV 及累積調整因子都與 ZIP 原行情吻合。先前提到的 2021-04-08 截斷發生於助理的解壓暫存副本，不是使用者上傳的原始檔；正式運算拒絕該副本、直接讀 ZIP。v7 持倉與排名封存也已核對 SHA-256。

| 版本 | 日期數 | 股票數 | 用途與限制 |
|---|---:|---:|---|
| 252 日市場版 | {market['dates']} | {market['stocks']:,} | {market['start']}～{market['end']}，連續市場交易日，只保留整段報酬完整股票。 |
| 全部持倉比較版 | {paired['dates']} | {paired['stocks']:,} | 同一 252 日窗口中的共同可觀測日期，實際最末日為 {paired['end']}，涵蓋全部 400 持倉。 |

原當日排名股池有 {R['base_universe']:,} 檔，與原行情排除當日監理旗標後一致。stock_list 的較晚公司／產業資訊只用來標示結果，沒有拿 Universe0 或期末市值倒選歷史股票。

252 日版本排除 {R['base_universe']-market['stocks']} 檔，其中涉及 8 檔持倉、總絕對權重 {pct(1-R['primary_coverage_gross'])}。因此另外將這 8 檔加入完整股池，再取全部股票都有有效日報酬的 160 日。共排除 {R['joint_omitted_dates']} 日（{pct(R['joint_omitted_dates']/252)}），這會產生「只看共同可觀測日期」的選擇偏差，尤其可能漏掉流動性差或異常日期，不能視為完整 252 日風險估計。先計算相鄰市場交易日報酬再選日期，沒有把跨缺口的多日漲跌當作單日報酬。

## 無指定持倉：市場在怎麼一起變動？

252 日版本 PC1 占 **{pct(market['pc1_share'])}**，前 5 個合計 **{pct(market['first5_share'])}**，前 10 個 **{pct(market['first10_share'])}**。解釋 80% 總變異數需要 {market['pcs_for_80pct']} 個方向。它不是所有股票都被一個因素完全支配。

完整持倉比較版中，PC1 有 {pct(paired['pc1_positive_loading_fraction'])} 的股票 loading 為正，分數與同股池等權報酬相關係數為 {paired['pc1_score_ew_correlation']:.4f}。依這兩項觀察，可以把它暫時解讀為「股票大致一起漲跌」的方向。這是從 loading 與報酬關係作的詮釋，不代表已找到造成漲跌的經濟原因。

無指定持倉的市場占比，分母是各股票變異數加總；等權投資組合是另一個明確指定權重的持倉。因此市場 PC1 占 19.33%，同時等權多頭 PC1 風險占 99.75%，沒有矛盾。

## 有持倉：同一個方向對 v7 有多重要？

以下全部使用共同 160 日、{paired['stocks']:,} 檔股票的同一套 PCA。v7 多空各 200 檔，兩側原始排名權重 2→1，轉成多頭 +50%、空頭 −50% 的資本權重，gross=1、net=0。

| 投資組合／部位 | PC1 占自身變異數 | 估計單日標準差 | 部位尺度 |
|---|---:|---:|---|
| 等權多頭 | {pct(ew['first_pc_share'])} | {pct(ew['training_covariance_daily_volatility'])} | 多頭 100% |
| v7 多頭側 | {pct(paired['portfolios']['long_sleeve']['first_pc_share'])} | {pct(paired['portfolios']['long_sleeve']['training_covariance_daily_volatility'])} | 多頭 50% |
| v7 空頭側 | {pct(paired['portfolios']['short_sleeve']['first_pc_share'])} | {pct(paired['portfolios']['short_sleeve']['training_covariance_daily_volatility'])} | 空頭 50% |
| v7 多空合計 | {pct(v7['first_pc_share'])} | {pct(v7['training_covariance_daily_volatility'])} | 多頭 50%＋空頭 50% |

多頭側的 PC1 曝險為 **{pc1['long_sleeve_exposure']:+.8f}**，空頭側為 **{pc1['short_sleeve_exposure']:+.8f}**，加起來只剩 **{pc1['v7_exposure']:+.8f}**。兩側各自很受共同漲跌影響，但放在一起大致抵銷。不能把「多頭側風險占比＋空頭側風險占比」直接相加，必須先加有正負號的曝險，再平方算風險。

v7 最大的單一風險方向是 **PC{v7['largest_risk_pc']}（{pct(v7['largest_risk_pc_share'])}）**，前 10 個方向合計 **{pct(v7['first10_share'])}**。剩餘方向合計 {pct(1-v7['first10_share'])}，不應只盯 PC1 或前兩個方向。就這個樣本而言，v7 沒有把大部分風險壓在單一 PC；這仍不代表尾部風險、流動性風險或未估到的風險很低。

| v7 風險最高方向 | 占市場總變異數 | 占 v7 變異數 |
|---|---:|---:|
{top_rows}

PC6 主要正 loading：{stock_lines(pc6_pos)}。

PC6 主要負 loading：{stock_lines(pc6_neg)}。

這兩組表示相反的共變動方向。查看互動報告可切換前 10 個 PC、比較兩側股票與產業。產業表的「平方 loading 占比」只是該方向的組成，不是產業對整個投資組合的風險貢獻。PC 的正負號可以整體翻轉，不影響風險結論。

## 對照你的手寫規劃

| 你的步驟 | 保留的部分 | 要修正或補上的部分 |
|---|---|---|
| 找最大波動方向 | 正確，PCA 找樣本變異數最大的單位方向。 | 先用每檔股票每日報酬率，並各自減平均；不直接對價格或原始 CSV 的 12 欄做此分析。 |
| 刪掉已知方向的投影 | 正確，是正交化。 | 已知方向須為單位向量，對所有已知方向扣投影；矩陣乘法後也要重新正交化與正規化。 |
| 不斷乘 covariance | 正確，是 power iteration。 | 正規化分母是向量長度 √(Σvᵢ²)，不是元素加總；用收斂判準，不能固定 20 次就認定成功。 |
| 得到 PC1 | 投影確實能得到 PC1 分數。 | v₁ 是方向，fₜ₁ 是當天有正負號的座標，λ₁ 才是整段樣本的變異數。三者不能都叫「最大波動值」。 |
| 整段 PC 得出的占比 | 你澄清的「整段統計」方向正確。 | 必須用 λₖ/Σλ，等價于中心化分數的平方加總占比；不是將有正負號的 PC 分數直接加總再相除。 |
| 加入資產權重 w | 正確，這正是持倉版。 | 先算 βₖ=wᵀvₖ，再算 λₖβₖ²；wᵀr 本身是組合報酬，不能再隨意乘 v。 |

v 的長度為 1，不代表它的權重加總或總絕對權重為 1。若把 v 當成實際可交易組合，需要另外說明資本尺度。因此 λ 不能直接當成一元資本的投資組合風險。

## 你可以手算對照的核心公式

設 R 是 T 個日期 × N 檔股票的報酬矩陣。X 的第 i 欄是 R 的第 i 欄減去其樣本平均。

```text
Σ = XᵀX / (T − 1)
Σvₖ = λₖvₖ，且 vₖᵀvₖ = 1
fₜₖ = (rₜ − μ)ᵀvₖ
市場解釋比例 = λₖ / Σⱼλⱼ
持倉曝險 βₖ = wᵀvₖ
PC k 的持倉變異數 = λₖβₖ²
持倉風險占比 = λₖβₖ² / (wᵀΣw)
```

以完整持倉比較版的 PC1 為例，λ₁={pc1['eigenvalue']:.8f}，β₁={pc1['v7_exposure']:.10f}，總持倉變異數={v7['training_covariance_variance']:.12g}。代入 λ₁β₁² / 總變異數，就得到 {pct(v7['first_pc_share'])}。

你的手寫 2×2 矩陣 [[2,1],[1,3]] 可得到 λ₁=3.618034、λ₂=1.381966，PC1 方向約 (0.525731,0.850651)，解釋比例 72.36%。原筆記若將左上角寫作 Var(A)=1，應改成 2。

## 維度、門檻與下一步

252×1,925 的中心化矩陣，最多支持 min(251,1925)=251 個非零方向，實際就是 251；160×1,933 版本最多 159 個，實際也是 159。這裡的欄數是股票數，不是原始行情 CSV 欄數，也不是 alpha 模型參數數。

100%/N×2 不是統計顯著性門檻。尤其 N 大於樣本日期數時，很多方向在樣本中無法估計，不能拿 1/N 當平均可識別風險。可以先看 scree plot、累積解釋比例與集中程度；若要設定正式異常門檻，需要再訂虛無模型、重抽樣方式與樣本外穩定性檢查，不能事後看結果挑門檻。

本次「多空抵銷 PC1」值得追蹤，但正式評估應使用每個歷史時點之前的資料重估 PCA，再套用當時保存的權重，並在之後日期檢查風險。兩個不同窗口的 PC 序號不保證是同一個方向，近似重根也可能旋轉；比較時要看方向對齊或子空間。本次尚未做這項滾動樣本外風險驗證，未使用保留 test，也沒有證明 PCA 會提高 Sharpe。

## 核對與來源

- 原始 ZIP 的 CSV：{R['source_hashes']['raw']}。
- bars 封存還原：{R['source_hashes']['bars']}。直接 ZIP 原行情全量對照通過。
- v7 持倉：{R['source_hashes']['holdings']}。
- 來源程式／設定 commit：{R['source_commit']}；資料 Release：snapshot-2026-09-24，查詢 catalog：snapshot-2026-09-26-v32-chunk-replay。
- 8 項單元測試通過；真實資料特徵方程、正交性、重建、power iteration、曝險與變異數分解通過。最大完整持倉變異數分解絕對誤差 {v7['reconciliation_absolute_error']:.3g}。
- 價格報酬不含股息，沒有估交易成本、借券成本或尾部損失。本次未修改既有 v7 模型。
- 方法定義參考：[scikit-learn PCA 官方文件](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)。實作以 NumPy SVD 計算，並以獨立 power iteration 核對 PC1。

互動入口：[JPX_PCA_report.html](JPX_PCA_report.html)。可下載後直接用瀏覽器開啟，無需網路。完整數值與日期／股票清單在 final_results；重跑方式見 README。
'''
(ROOT/'JPX_PCA_report.md').write_text(note)

template=r'''<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>JPX PCA｜市場與持倉風險</title><style>
:root{--ink:#183440;--muted:#5e737b;--paper:#f5f7f5;--blue:#246a8d;--orange:#d56b3a;--green:#32947d;--line:#dbe5e5}*{box-sizing:border-box}body{margin:0;background:var(--paper);font:16px/1.7 system-ui,-apple-system,"Noto Sans TC",sans-serif;color:var(--ink)}main{max-width:1150px;margin:auto;padding:38px 24px 60px}header{border-top:5px solid var(--blue);padding-top:24px;margin-bottom:26px}.eyebrow{font-size:12px;letter-spacing:.12em;color:var(--muted)}h1{font-size:clamp(26px,4vw,42px);line-height:1.3;margin:12px 0}h2{font-size:21px;margin:0 0 16px}h3{font-size:17px;margin:16px 0 7px}p{margin:8px 0 16px}.lead{max-width:850px;font-size:18px}.muted,small{color:var(--muted)}section{background:#fff;border:1px solid var(--line);border-radius:15px;padding:24px;margin:20px 0}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.card{padding:20px;background:white;border:1px solid var(--line);border-radius:12px}.value{font-size:30px;font-weight:750;letter-spacing:-.04em;line-height:1.4}.card small{display:block;font-size:12px}.blue{color:var(--blue)}.orange{color:var(--orange)}.green{color:var(--green)}.notice{border-left:4px solid #c59b50;background:#fbf8ef;padding:12px 16px;font-size:14px}.controls{display:flex;flex-wrap:wrap;gap:20px;align-items:center;margin-bottom:15px}select{font:inherit;color:var(--ink);background:#fff;padding:8px 12px;border:1px solid #8a9ba1;border-radius:7px;max-width:100%}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border-bottom:1px solid var(--line);text-align:left;padding:11px 12px;vertical-align:top}th{background:#f2f6f6;white-space:nowrap}td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}.two{display:grid;grid-template-columns:1fr 1fr;gap:24px}.formula{font:16px/1.9 ui-monospace,monospace;background:#f2f6f6;padding:16px;border-radius:8px;overflow:auto;white-space:pre-wrap}.barrow{display:grid;grid-template-columns:100px 1fr 65px;gap:10px;align-items:center;font-size:13px;margin:9px 0}.track{height:13px;border-radius:4px;background:#e9eeee}.bar{height:100%;border-radius:4px;background:var(--orange)}.legend{font-size:13px;margin:12px 0}.dot{display:inline-block;width:10px;height:10px;margin:0 7px 0 15px}.chart svg{width:100%;height:auto;min-width:560px}summary{cursor:pointer;font-weight:650;padding:6px 0}details{margin:15px 0}a{color:var(--blue)}footer{font-size:13px;color:var(--muted)}button{background:var(--blue);color:#fff;border:0;padding:9px 16px;border-radius:7px;cursor:pointer;font:inherit}.exposure{display:flex;flex-wrap:wrap;gap:18px;justify-content:space-around;background:#f2f6f6;padding:18px;border-radius:10px}.exposure strong{display:block;font-size:22px}.pcnote{font-size:14px;color:var(--muted)}@media(max-width:720px){main{padding:20px 14px}.cards{grid-template-columns:1fr 1fr}.card{padding:15px}.value{font-size:27px}section{padding:18px}.two{grid-template-columns:1fr}th,td{padding:9px 8px}.lead{font-size:16px}.barrow{grid-template-columns:80px 1fr 55px}}@media print{body{background:#fff}section,.card{break-inside:avoid}select,button{display:none}}
</style></head><body><main>
<header><div class="eyebrow">QUANT LEARNING · JPX · 2026-09-28</div><h1>市場與持倉，風險集中在哪裡？</h1>
<p class="lead">PCA 找出股票一起變動的方向。加入持倉權重後，才能知道這些方向對你的組合有多重要。</p>
<p class="muted">歷史持倉切點 2021-12-01 · v7_equal · 多 200／空 200 · 歷史風險診斷</p></header>
<div class="cards" id="cards"></div>
<section><h2>先看同一組方向下的差別</h2><p>下列主要比較都使用同一組 PCA：<strong id="paired-size"></strong>。v7 的多頭、空頭各自暴露在市場共同方向，合併後大致抵銷。</p>
<div class="exposure" id="exposure"></div><p class="pcnote">曝險有正負號，先相加再平方。兩側風險占比不能直接相加。</p>
<div class="notice" id="sample-note"></div></section>
<section><div class="controls"><label>分析樣本 <select id="sample"><option value="1">全部 400 持倉 · 共同 160 日</option><option value="0">市場參考 · 完整 252 日</option></select></label></div>
<h2 id="chart-title"></h2><p class="muted" id="chart-note"></p><div class="scroll chart" id="comparison"></div><div id="legend" class="legend"></div>
<p class="pcnote">市場占比的分母是各股票變異數加總；持倉占比的分母是該組合變異數。PC11+ 是其餘所有方向合計。</p>
<details><summary>查看全部主成分數值</summary><button id="download">下載目前比較 CSV</button><div class="scroll" id="all-components"></div></details></section>
<section><h2>哪些方向占 v7 最多風險？</h2><p>市場排序第一的方向，未必是持倉最重要的方向。這裡按 v7 的風險占比重新排序。</p><div id="risk-ranking"></div>
<p class="pcnote" id="rest-risk"></p></section>
<section><div class="controls"><h2 style="margin:0">看懂一個方向的組成</h2><label>主成分 <select id="pc"></select></label></div>
<p id="pc-context"></p><div class="two"><div><h3>正 loading 較大的股票</h3><div class="scroll" id="positive"></div></div><div><h3>負 loading 較大的股票</h3><div class="scroll" id="negative"></div></div></div>
<p class="pcnote">表格分別列出最大、最小的 loading；若某側沒有足夠對應正／負值，只列實際符合符號的股票。正負號可整體翻轉，風險不變。產業名稱取自較晚的股票名單，僅作說明。</p>
<details><summary>產業組成與解讀限制</summary><div id="sectors"></div><p class="pcnote">平方 loading 加總描述方向組成，並非產業對持倉的風險貢獻。PCA 提供共變動描述，沒有辨識經濟因果。</p></details></section>
<section><h2>你的手寫規劃，怎麼接到這些數字？</h2><div class="two"><div><h3>不指定持倉</h3><div class="formula">X = 每檔報酬 − 該檔樣本平均
Σ = XᵀX / (T − 1)
Σvₖ = λₖvₖ
解釋比例 = λₖ / Σⱼλⱼ</div><p>你說「整段 PC 得出的比例」方向正確，但要用整段<strong>變異數</strong>，不能直接加總正負分數。</p></div><div><h3>加入持倉 w</h3><div class="formula">βₖ = wᵀvₖ
此方向風險 = λₖβₖ²
持倉風險占比
= λₖβₖ² / (wᵀΣw)</div><p>v 是方向、fₜₖ 是當日投影分數、λₖ 是分數的樣本變異數。β 才回答你的持倉壓了多少在這個方向。</p></div></div>
<details><summary>你的規劃需要補上的四件事</summary><ol><li>用報酬率並先中心化；本次沒有除以每檔標準差，因為研究的是共變異數。</li><li>每次乘矩陣後扣除已知方向投影，再用向量長度 √(Σvᵢ²) 正規化；檢查收斂，不固定迭代次數。</li><li>v 長度為 1，不等於總資本權重為 1。λ 本身不是一元資本組合的風險。</li><li>「100% ÷ 股票數 × 2」不是顯著性門檻；樣本能估計的方向最多是 min(T−1,N)。</li></ol></details>
<p id="rank-note" class="notice"></p></section>
<section><h2>資料、核對與範圍</h2><p>原始 ZIP 共 2,332,531 筆，2017-01-04～2021-12-03；與研究快取的全部 OHLCV 和累積調整因子逐筆一致。v7 持倉、排名及行情封存雜湊均已驗證。</p>
<p>原始 ZIP 完整。先前提到的截斷發生於助理的解壓暫存副本，正式計算已改為直接讀 ZIP。</p>
<p>8 項單元測試及真實資料的正交性、重建、特徵方程、power iteration 與組合變異數分解核對通過。</p>
<div class="notice">這是固定一日權重套用歷史共變異數的結果。未含股息、成本或尾部風險，沒有模擬每日調倉，尚未做滾動樣本外風險驗證，未使用保留 test。樣本估不到的方向不代表未來風險為零。</div>
<details><summary>252 日版本未覆蓋的 8 檔持倉</summary><div class="scroll" id="missing"></div><p class="pcnote">共同 160 日版本已包含這 8 檔，全部 400 持倉均有納入。</p></details>
<p><a href="https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html" target="_blank" rel="noopener">PCA 方法參考：scikit-learn 官方文件</a></p></section>
<footer>報告可離線開啟。HTML 內含本次結果與互動程式。完整數值、排除日期與重跑方式另附於研究目錄。報告日期：2026-09-28。</footer>
<noscript>請開啟 JavaScript 查看互動結果，或閱讀隨附的 JPX_PCA_report.md。</noscript>
</main><script>
const DATA=__DATA__;
const $=id=>document.getElementById(id),e=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct=(x,n=2)=>(x*100).toFixed(n)+'%',num=x=>Number(x).toLocaleString('en-US');
const [M,P]=DATA.cases,V=P.portfolios.v7,E=P.portfolios.equal_long,ONE=P.components[0];
const card=(label,value,note,color)=>`<div class="card"><small>${label}</small><div class="value ${color}">${value}</div><small>${note}</small></div>`;
$('cards').innerHTML=card('PC1 · 市場總變異數',pct(P.pc1_share),'共同 160 日／同一基底','blue')+card('PC1 · v7 多空風險',pct(V.first_pc_share),'多頭 50%＋空頭 50%','orange')+card('PC1 · 等權多頭風險',pct(E.first_pc_share),'同股池／多頭 100%','green')+card('v7 最大單一風險方向','PC'+V.largest_risk_pc,'占組合變異數 '+pct(V.largest_risk_pc_share),'orange');
$('paired-size').textContent=`${num(P.stocks)} 檔股票、${P.dates} 個共同有效日報酬`;
$('exposure').innerHTML=[['多頭 PC1 曝險',ONE.long_sleeve_exposure],['空頭 PC1 曝險',ONE.short_sleeve_exposure],['合計 PC1 曝險',ONE.v7_exposure]].map(([label,x])=>`<div><small>${label}</small><strong>${x>=0?'+':''}${x.toFixed(8)}</strong></div>`).join('');
$('sample-note').textContent=`持倉日為 ${DATA.asof_after_close}。估計取自原 252 日窗口，共同有效樣本為 ${P.start}～${P.end}，保留 ${P.dates} 日、排除 ${DATA.joint_omitted_dates} 日。結果只代表共同可觀測日期；缺值相關的風險可能未被充分反映。`;
const riskRows=[...P.components].sort((a,b)=>b.v7_portfolio_variance_share-a.v7_portfolio_variance_share).slice(0,8);
$('risk-ranking').innerHTML=riskRows.map(r=>`<div class="barrow"><span>PC${r.PC}</span><div class="track"><div class="bar" style="width:${r.v7_portfolio_variance_share/riskRows[0].v7_portfolio_variance_share*100}%"></div></div><strong>${pct(r.v7_portfolio_variance_share)}</strong></div>`).join('');
$('rest-risk').textContent=`按市場變異數排序的前 10 個 PC，合計占 v7 風險 ${pct(V.first10_share)}；其餘方向仍占 ${pct(1-V.first10_share)}。橫條按最大一項縮放，右側為實際占比。`;
$('rank-note').textContent=`252 日 × ${num(M.stocks)} 檔股票 → 最多 251 個非零方向，實際 251 個。共同 160 日 × ${num(P.stocks)} 檔 → 最多 159 個，實際 159 個。這裡每一欄是一檔股票。`;
function table(headers,rows){return '<table><thead><tr>'+headers.map(x=>'<th>'+e(x)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(row=>'<tr>'+row.map(x=>'<td>'+e(x)+'</td>').join('')+'</tr>').join('')+'</tbody></table>'}
$('missing').innerHTML=table(['代碼','公司','缺報酬日數','資本權重'],DATA.missing_252_holdings.map(r=>[r.SecuritiesCode,r.Name,r.missing_days,pct(r.Weight,4)]));
for(let i=1;i<=10;i++)$('pc').add(new Option('PC'+i,i));$('pc').value='6';
function current(){return DATA.cases[Number($('sample').value)]}
function series(){return Number($('sample').value)===1?[
 ['market_variance_share','市場總變異數','#246a8d'],['v7_portfolio_variance_share','v7 多空風險','#d56b3a'],['equal_long_portfolio_variance_share','等權多頭風險','#32947d']]:[
 ['market_variance_share','市場總變異數','#246a8d'],['covered_v7_portfolio_variance_share','v7 已覆蓋部位風險','#d56b3a']]}
function renderComparison(){const c=current(),s=series(),width=960,height=320,left=52,bottom=266,plotH=224,step=79;
 $('chart-title').textContent=`${c.dates} 日樣本：市場與持倉的變異數占比`;
 $('chart-note').textContent=Number($('sample').value)===1?`${num(c.stocks)} 檔股票 · 全部 400 持倉 · 固定權重估計單日標準差 ${pct(c.portfolios.v7.training_covariance_daily_volatility)}`:`${num(c.stocks)} 檔股票 · ${c.start}～${c.end} · 僅覆蓋原持倉總絕對權重 ${pct(DATA.primary_coverage_gross)}，未覆蓋部位沒有補零或重新歸一化；橘色不是完整組合風險。`;
 let svg=`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="主成分風險占比比較"><rect width="100%" height="100%" fill="white"/>`;
 for(let v=0;v<=100;v+=25){let y=bottom-plotH*v/100;svg+=`<line x1="${left}" y1="${y}" x2="935" y2="${y}" stroke="#e3e9e9"/><text x="42" y="${y+4}" text-anchor="end" fill="#5e737b" font-size="12">${v}%</text>`}
 for(let k=0;k<11;k++){const xx=left+18+k*step;for(let j=0;j<s.length;j++){const [key,label,color]=s[j];const val=k<10?c.components[k][key]:c.components.slice(10).reduce((a,r)=>a+r[key],0);const hh=val*plotH;svg+=`<rect x="${xx+j*18}" y="${bottom-hh}" width="16" height="${hh}" fill="${color}"><title>${e(label)} ${k<10?'PC'+(k+1):'PC11+'}：${pct(val,4)}</title></rect>`}svg+=`<text x="${xx+21}" y="287" text-anchor="middle" fill="#183440" font-size="12">${k<10?'PC'+(k+1):'PC11+'}</text>`}
 $('comparison').innerHTML=svg+'</svg>';$('legend').innerHTML=s.map(([key,label,color])=>`<span><i class="dot" style="background:${color}"></i>${e(label)}</span>`).join('');
 $('all-components').innerHTML=table(['主成分',...s.map(x=>x[1]),'累積市場解釋比例'],c.components.map(r=>['PC'+r.PC,...s.map(x=>pct(r[x[0]],4)),pct(r.cumulative_market_share,4)]));renderPC();}
function renderPC(){const c=current(),pc=Number($('pc').value),r=c.components[pc-1],s=Number($('sample').value)===1?'v7':'covered_v7';
 $('pc-context').textContent=`PC${pc} 解釋市場變異數 ${pct(r.market_variance_share)}，占${s==='v7'?'完整 v7':'已覆蓋 v7 部位'}風險 ${pct(r[s+'_portfolio_variance_share'])}。`;
 for(const side of ['positive','negative']){let rows=c.top_loadings.filter(r=>r.PC===pc&&r.side===side&&(side==='positive'?r.loading>0:r.loading<0)).slice(0,8);$(side).innerHTML=rows.length?table(['代碼／公司','產業','loading'],rows.map(r=>[r.SecuritiesCode+' '+r.Name,r.sector,(r.loading>0?'+':'')+r.loading.toFixed(4)])):'<p class="muted">這個方向沒有對應符號的主要股票。</p>';}
 const sectors=c.sector_profiles.filter(x=>x.PC===pc).slice(0,8);$('sectors').innerHTML=table(['產業','平方 loading 占比','股票數'],sectors.map(x=>[x['17SectorName'],pct(x.loading_energy),x.stock_count]));}
$('sample').addEventListener('change',renderComparison);$('pc').addEventListener('change',renderPC);
$('download').addEventListener('click',()=>{const c=current(),rows=c.components,keys=Object.keys(rows[0]);const csv=keys.join(',')+'\n'+rows.map(r=>keys.map(k=>r[k]).join(',')).join('\n');const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download='JPX_PCA_'+c.name+'.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),2000)});
renderComparison();
</script></body></html>'''
payload=json.dumps(R,ensure_ascii=False,separators=(',',':')).replace('</',r'<\/')
(ROOT/'JPX_PCA_report.html').write_text(template.replace('__DATA__',payload))
print('Built reports and figures')
