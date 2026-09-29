"""Describe April 2018 daily risk concentration from immutable, verified A/PCA states."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'jpx_pca_random_2018_20260929/results'
audit=json.loads((SOURCE/'generation_audit.json').read_text())
rows=[];sources=[];W=np.linspace(2.,1.,200)
for entry in audit['states']:
    if '_2018-04-' not in entry['path']:
        continue
    path=SOURCE/entry['path']
    with path.open('rb') as f:
        assert hashlib.file_digest(f,'sha256').hexdigest()==entry['sha256']
    with np.load(path,allow_pickle=False) as s:
        order=np.argsort(s['baseline_rank']);w=np.zeros(len(order))
        w[order[:200]]=.5*W/W.sum()
        w[order[-200:][::-1]]=-.5*W/W.sum()
        np.testing.assert_allclose(np.sum(np.abs(w)),1.,atol=1e-14)
        beta=w[s['keep']]@s['v']
        variance=s['eigenvalues']*beta**2
        k=int(np.argmax(variance))
        assert k==int(s['kmax']) and variance.sum()>0
        concentration=float(variance[k]/variance.sum())
        assert 0<=concentration<=1
        rows.append({'Date':str(s['date']),'max_PC':k+1,'max_risk_share':concentration,
                     'above_30pct':bool(concentration>.3),
                     'covered_total_variance':float(variance.sum()),
                     'max_component_variance':float(variance[k]),
                     'covered_gross':float(np.abs(w[s['keep']]).sum())})
    sources.append(entry)
d=pd.DataFrame(rows).sort_values('Date');assert len(d)==20
d.to_csv(ROOT/'daily_risk_share.csv',index=False)
bins=np.linspace(0,1,11);counts,_=np.histogram(d.max_risk_share,bins=bins)
hist=[{'lower_pct':int(round(bins[i]*100)),'upper_pct':int(round(bins[i+1]*100)),
       'days':int(counts[i])} for i in range(10)]
stats={'days':20,'mean':float(d.max_risk_share.mean()),
       'quantiles':{str(q):float(d.max_risk_share.quantile(q)) for q in [0,.25,.5,.75,1]},
       'above_30pct_days':int(d.above_30pct.sum()),'equal_30pct_days':int(d.max_risk_share.eq(.3).sum()),
       'minimum_date':str(d.loc[d.max_risk_share.idxmin(),'Date']),
       'maximum_date':str(d.loc[d.max_risk_share.idxmax(),'Date']),
       'max_PC_counts':{str(k):int(v) for k,v in d.max_PC.value_counts().sort_index().items()},
       'mean_covered_gross':float(d.covered_gross.mean()),'minimum_covered_gross':float(d.covered_gross.min()),
       'histogram_10pct_bins':hist,'baseline':'A / v7_equal',
       'denominator':'Estimated variance of the PCA-covered part of A, sum_k lambda_k*(w_A_covered^T v_k)^2',
       'source_state_hashes_verified':sources,'Target_used':False,'test_used':False,
       'scope':'Daily as-of estimates, April 2018 only; no new backtest, no regime gate applied'}
(ROOT/'summary.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n')
fig,axs=plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
axs[0].hist(d.max_risk_share*100,bins=bins*100,color='#4477aa',edgecolor='white')
axs[0].axvline(30,color='#bb4422',linestyle='--',label='Proposed threshold: 30%')
axs[0].set(xlabel='Largest PC contribution / covered portfolio variance (%)',ylabel='Trading days',
           title='Distribution across 20 signal days',xlim=(0,100),yticks=range(0,9,2))
axs[0].legend(fontsize=8)
dates=pd.to_datetime(d.Date)
axs[1].plot(dates,d.max_risk_share*100,'o-',color='#4477aa',markersize=4)
axs[1].axhline(30,color='#bb4422',linestyle='--')
axs[1].set(title='Daily risk concentration',ylabel='Share (%)',ylim=(0,100))
fig.autofmt_xdate(rotation=30)
fig.suptitle('April 2018 | Original model A | PCA-covered estimated variance')
fig.savefig(ROOT/'april_risk_distribution.png',dpi=150);plt.close(fig)
lines=['# 2018 年 4 月：原模型最大 PC 風險貢獻占比分布','',
       '本次只描述已保存的每日事前風險估計，沒有重新訓練、產生新策略、讀取 Target 或使用保留 test。',
       'C_t = max_k[lambda_k (w_A^T v_k)^2] / sum_k[lambda_k (w_A^T v_k)^2]。分子是原持倉的最大方向變異貢獻；分母限於 PCA 能涵蓋的原持倉部分，不是含缺資料股票在內的全資產風險，也不是波動率相加。',
       '每日使用當時最多 252 個市場交易日估計，沒有先用完整 4 月估一套方向回套。月統計是上述每日比例的等權描述，並非先加總變異再相除。','',
       '## 結果','',
       f'- 20 個訊號日；嚴格超過 30%：{stats["above_30pct_days"]} 天（50%），等於 30%：0 天。',
       f'- 平均 {stats["mean"]:.2%}；中位數 {stats["quantiles"]["0.5"]:.2%}。',
       f'- 第 25% / 75% 分位數：{stats["quantiles"]["0.25"]:.2%} / {stats["quantiles"]["0.75"]:.2%}（線性插值）。',
       f'- 最低 {stats["quantiles"]["0"]:.2%}（{stats["minimum_date"]}）；最高 {stats["quantiles"]["1"]:.2%}（{stats["maximum_date"]}）。',
       '- 最大持倉風險貢獻方向：PC1 2 天、PC2 6 天、PC3 12 天。每天 PC 編號可能改變，並不代表跨日期固定的同一經濟因子。',
       f'- PCA 涵蓋的 gross 持倉比平均 {stats["mean_covered_gross"]:.2%}，最低 {stats["minimum_covered_gross"]:.2%}。這是部位涵蓋率，不是風險涵蓋率；未涵蓋股票風險未知。','',
       '![分布與每日占比](april_risk_distribution.png)','',
       '依使用者提出的 30% 門檻，當月有一半日期超標。這只確認該特徵在 4 月的分布；尚未比較其他月份，也未證明 30% 能解釋 B_MAX 的改善。不根據本次分布另改門檻。','',
       '## 每日資料','',
       '| 訊號日期 | 最大方向 | 風險占比 | 超過 30% |',
       '|---|---:|---:|---|']
for r in d.itertuples():
    lines.append(f'| {r.Date} | PC{r.max_PC} | {r.max_risk_share:.2%} | {"是" if r.above_30pct else "否"} |')
lines.extend(['','## 來源與保存','',
              '來源為 ../jpx_pca_random_2018_20260929/results/states 的 20 份每日 NPZ；逐檔 SHA-256 已對照原 generation_audit.json，並核對最大方向與原 B_MAX 選擇一致。原實驗程式及小型結果保存於 Git 提交 f14b6a1701b9e0a6092aa59c67a10a806bc4900b。',
              '本次衍生程式、完整每日 CSV、摘要和圖表均屬小型檔。原大型 PCA 狀態仍沿原 sync_status.json 列為待發布 Release，不能因本次小型衍生結果已保存就視為原大型資料已封存完成。沒有刪除來源或輸入。'])
(ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:v for k,v in stats.items() if k!='source_state_hashes_verified'},ensure_ascii=False))
