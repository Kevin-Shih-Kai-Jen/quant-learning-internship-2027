"""Create a complete, selection-aware report for the fixed 25-model JPX grid."""
from pathlib import Path
import os,json,hashlib,zipfile,shutil,sqlite3
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
RUN=ROOT/'arima_grid';OUT=ROOT.parent/'outputs'
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'mplcache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

def fmt(x):return f'{x:+.6f}' if pd.notna(x) else '未定義'

def plot_matrix(ax,values,title,digits=4):
    v=np.asarray(values,dtype=float);bound=max(float(np.nanmax(np.abs(v))) if np.isfinite(v).any() else 0.,1e-6)
    im=ax.imshow(np.ma.masked_invalid(v),cmap='RdBu',norm=TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound))
    ax.set_xticks(range(5),range(1,6));ax.set_yticks(range(5),range(1,6))
    ax.set_xlabel('MA order s');ax.set_ylabel('AR order r');ax.set_title(title,loc='left',fontweight='bold')
    for i in range(5):
        for j in range(5):ax.text(j,i,f'{v[i,j]:+.{digits}f}' if np.isfinite(v[i,j]) else 'N/A',ha='center',va='center',color='white' if abs(v[i,j])>bound*.6 else '#182532',fontsize=9)
    return im

def main():
    s=json.loads((RUN/'results.json').read_text());assert s['complete'] and len(s['models'])==25 and s['total_fits']==200000
    table=pd.read_csv(RUN/'leaderboard.csv');annual=pd.read_csv(RUN/'annual.csv');base=s['baseline_common'];top=table.iloc[0]
    con=sqlite3.connect(f'file:{RUN}/fits.sqlite?mode=ro',uri=True)
    assert con.execute('select count(*) from fits').fetchone()[0]==200000
    assert con.execute('select count(*) from (select oi,count(*) n from fits group by oi having n=8000)').fetchone()[0]==25
    con.close()
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'figure.facecolor':'#f8fafc'})
    fig,axes=plt.subplots(1,2,figsize=(12,5.8))
    for ax,col,title in zip(axes,['sharpe','mean_rank_ic'],['Daily spread Sharpe','Mean daily Rank IC']):
        v=table.pivot(index='p',columns='q',values=col).sort_index().sort_index(axis=1)
        im=plot_matrix(ax,v,title);fig.colorbar(im,ax=ax,fraction=.045,pad=.04)
    fig.suptitle('JPX | ARIMA(r,1,s): all 25 combinations',x=.05,ha='left',fontweight='bold',fontsize=16)
    fig.text(.05,.025,f'Expanding annual fits | {s["common_days"]} common validation days | Before costs, unannualized | Test untouched',fontsize=9)
    fig.tight_layout(rect=[.02,.06,.99,.93]);fig.savefig(OUT/'JPX-ARIMA-grid-heatmaps.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,10))
    for ax,year in zip(axes.flat,[2018,2019,2020,2021]):
        v=annual.loc[annual.Year.eq(year)].pivot(index='p',columns='q',values='sharpe').sort_index().sort_index(axis=1)
        plot_matrix(ax,v,f'{year} validation'+(' (partial year)' if year==2021 else ''))
    fig.suptitle('JPX | Annual Sharpe by AR and MA order',x=.06,ha='left',fontsize=16,fontweight='bold')
    fig.text(.06,.015,'Each panel uses its own symmetric color scale; compare the printed values.',fontsize=9)
    fig.tight_layout(rect=[.02,.04,.99,.94]);fig.savefig(OUT/'JPX-ARIMA-grid-annual-heatmaps.png',dpi=150);plt.close(fig)
    resultrows='\n'.join(f'| {int(r.GridRank)} | ({int(r.p)},1,{int(r.q)}) | {fmt(r.sharpe)} | {fmt(r.mean_rank_ic)} | {int(r.PositiveYears)}/4 | {int(r.YearsAboveV7)}/4 | {r.FallbackFraction:.2%} | {int(r.OtherFitFailures)} |' for _,r in table.iterrows())
    annualwide=annual.pivot(index=['p','q'],columns='Year',values='sharpe')
    annualrows='\n'.join(f'| ({int(p)},1,{int(q)}) | '+' | '.join(fmt(v) for v in row)+' |' for (p,q),row in annualwide.iterrows())
    cirows='\n'.join(f'| ({int(r.p)},1,{int(r.q)}) | {fmt(r.SharpeMinusV7)} | [{fmt(r.Marginal95Low)}, {fmt(r.Marginal95High)}] | [{fmt(r.GridSimultaneous95Low)}, {fmt(r.GridSimultaneous95High)}] |' for _,r in table.iterrows())
    first=s['models'][0]['annual'];baseannual='\n'.join(f"| {a['year']} | {fmt(a['v7']['sharpe'])} | {fmt(a['v7']['mean_rank_ic'])} | {a['v7']['days']} |" for a in first)
    count_better=int(table.sharpe.gt(base['sharpe']).sum());count_sig=int(table.GridSimultaneous95Low.gt(0).sum())
    failures=sum(sum(v for k,v in m['fit_statuses'].items() if k not in ['ok','insufficient_history']) for m in s['models'])
    prefixes=sum(m['prefix_checks'] for m in s['models']);forward=sum(m['invalid_forward_observations'] for m in s['models'])
    excluded='無' if not s['excluded_dates'] else '、'.join(s['excluded_dates'])
    score_compare='高於' if top.sharpe>base['sharpe'] else '低於'
    report=f'''# JPX ARIMA(r,1,s) 全部 25 組回測報告

使用者指定 1≤r≤5、1≤s≤5；本報告完整列出 25 組，沒有只挑選成功或高分組合。沿用無漂移、expanding window、年度估計與驗證、JPX 明天→後天目標；共完成 200,000 次股票年度估計。使用者提出研究方向，助理實作與核對。

## 主要結果

在全部模型共同的 {s['common_days']} 個可評分驗證日，驗證集最高 Sharpe 為 **ARIMA({int(top.p)},1,{int(top.q)})：{top.sharpe:+.6f}**，平均 Rank IC {top.mean_rank_ic:+.6f}。既有 v7 的 Sharpe 為 {base['sharpe']:+.6f}，平均 Rank IC {base['mean_rank_ic']:+.6f}；網格第一名的樣本 Sharpe {score_compare} v7。

25 組中，{count_better} 組的樣本 Sharpe 高於 v7；{count_sig} 組在本次固定 25 組網格的近似同時信賴區間下，其 Sharpe 差異下界高於零。這仍不是全新 test 證據，不能自動推廣成未來會獲利的結論。正式 test 未讀取、未評分；v7 未被自動替換。

所有 Sharpe 均為官方每日多空報酬差的平均／樣本標準差，未年化、未扣成本。Rank IC 是每日原始預測分數與 Target 的 Spearman 相關係數，再取平均。單日標籤或分數全同時 IC 未定義，不以零替代。

![全部組合的 Sharpe 與 Rank IC](JPX-ARIMA-grid-heatmaps.png)

## 完整 25 組排行榜

按共同日期 Sharpe 降冪排列；同分時依平均 Rank IC，再依較小 r、s 排序。這只是驗證集內的排序規則。

| 名次 | ARIMA(r,1,s) | Sharpe | 平均 Rank IC | 正 Sharpe 年數 | 勝 v7 年數 | 後備股票日比例 | 其他估計失敗數 |
|---:|---|---:|---:|---:|---:|---:|---:|
{resultrows}

「其他估計失敗」不含最低歷史量不足；全網格其他估計失敗合計 {failures:,} 次。每組失敗原因、重試數與被選入前後 200 檔的後備次數見資料包與完整 CSV。所有失敗使用事前固定零分後備，股票不被刪除。

## 每組的分年 Sharpe

| ARIMA(r,1,s) | 2018 | 2019 | 2020 | 2021（部分年度） |
|---|---:|---:|---:|---:|
{annualrows}

既有 v7 在相同共同日期的分年參照：

| 驗證年標籤 | Sharpe | Rank IC | 共同日數 |
|---|---:|---:|---:|
{baseannual}

![全部組合的分年 Sharpe](JPX-ARIMA-grid-annual-heatmaps.png)

年度不同代表不同歷史區間，並不是已經用客觀規則辨認出的市場 regime。本輪沒有再挑特定年份或反轉訊號來提高結果。

## 25 組選模帶來的不確定性

全期第一名相對 v7 的 Sharpe 差為 {top.SharpeMinusV7:+.6f}；單組邊際 95% 區間 [{top.Marginal95Low:+.6f}, {top.Marginal95High:+.6f}]；固定網格近似同時 95% 區間 [{top.GridSimultaneous95Low:+.6f}, {top.GridSimultaneous95High:+.6f}]。

方法：共同日期上做 20 日循環區塊配對 bootstrap，2,000 次，seed=20260923；所有模型與基準每次共用同一批抽樣日期。邊際區間取各模型差值分布的 2.5%／97.5% 分位。網格同時區間使用各次抽樣相對觀測差值的最大絕對偏差，其 95% 分位作為共同半徑 {s['bootstrap']['simultaneous_centered_max_deviation_radius']:.6f}。

這是固定 25 候選的近似同時區間，依賴區塊長度與時間相依假設，未校正先前反覆使用驗證資料的研究決策。不能把它稱為已消除過擬合、已證明泛化，或將最高 Sharpe 視為未來預期績效。也没有重新對驗證集勝者做一輪事後參數搜尋。

| ARIMA(r,1,s) | Sharpe 減 v7 | 單組邊際 95% 區間 | 固定網格同時 95% 區間 |
|---|---:|---|---|
{cirows}

## 模型與排名公式

ΔP[t] = φ₁ΔP[t−1] + … + φᵣΔP[t−r] + ε[t] + θ₁ε[t−1] + … + θₛε[t−s]

P 為因果調整收盤價；d=1 表示價格差分，並非百分比報酬。站在 t 收盤後，用截至 t 的價格與估計狀態預測 P[t+1|t]、P[t+2|t]，並以：

score[t] = P[t+2|t] / P[t+1|t] − 1

作為排名分數。兩個點預測的比值是報酬代理，不保證等於隨機價格比值的精確條件期望。未來創新誤差用條件期望零，不能把預測的價格變動當作新 error，也不能使用隔天實際價格。

每日所有合格股票完整唯一排名，分數由高到低，同分按股票代碼升冪。依 JPX 官方規則選前後各 200 檔，名次權重由 2 線性降到 1；不能另行調整投入金額或挑某天不交易。保持有限極端值，不做績效導向裁切。

## Expanding window 與共同評估日期

| 驗證年標籤 | 年度參數估計區間 | 日數 | 驗證訊號區間 |
|---|---|---:|---|
| 2018 | 2017-01-04～2017-12-28 | 246 | 2017-12-29～2018-12-27 |
| 2019 | 2017-01-04～2018-12-27 | 491 | 2018-12-28～2019-12-27 |
| 2020 | 2017-01-04～2019-12-27 | 732 | 2019-12-30～2020-12-29 |
| 2021 | 2017-01-04～2020-12-29 | 975 | 2020-12-30～2021-12-01 |

每檔股票獨立估計，至少 126 個有效訓練價格。每年重新估計係數；年內固定係數，每日讀入新價格更新 forward filter。Expanding 代表下一年度保留所有累積歷史，不等於每天重估係數。缺失日期保留，不壓縮時間；因果價格調整與前輪一致。

原驗證池 953 日、1,864,363 個股票日，各模型的原始可評分日數也保存在 CSV。為公平比較，本報告排行只用全部 25 組與 v7 皆能評分的 {s['common_days']} 日；排除日期：{excluded}。沿用原始 Target，不以模型自行重建標籤覆寫。2021 非完整年度；年度邊界含前一年末訊號是沿用原始報酬日期對齊。

估計均採相同 statsmodels 0.14.6 Gaussian 狀態空間最大概似、AR 平穩與 MA 可逆約束、無漂移。L-BFGS 最多 500 步；未收斂時從當前參數再跑最多 1,000 步。尺度變換只用訓練資料。前輪 ARIMA(5,1,5) 用 252 日窗口，本次已重新估計為 expanding，不能直接沿用其舊績效。

## 核對與限制

- 200,000 次股票年度紀錄完整；所有年度訓練結束均早於第一個預測日，expanding 起點一致；工作函式只接收價格，沒有 Target。
- 共 {prefixes:,} 次內建截斷預測核對、{s['independent_causal_checks']:,} 次獨立截斷／未來價格擾動核對通過；未使用 smoothed states。
- 逐組檢查模型收斂、參數、根與訓練創新變異數；完整 forward filter 觀测到的數值失效合計 {forward:,} 次。若年內發生數值失效，只從當時起使用後備，不回頭改寫之前排名。
- 25 組完整排名與官方 Sharpe 獨立重算通過，原 v7 每日報酬也重現。股票池與日期鍵完全一致。
- 原始 Target 與本地因果價格比值有少量微小差異，前輪已記錄，成因未確定；所有比較仍使用相同原始 Target。
- 本次是反覆使用歷史驗證資料的探索。未納入成本與成交限制，也未宣稱完成所有模型殘差、參數漂移及 regime 診斷。

## 如何用這份報告寫企劃

先把研究假設與模型參數分開：r 表示價格差分的落後期數，s 表示估計創新誤差的落後期數。網格回測回答這 25 組在既定資料與流程下如何表現，不能僅憑最高分推論所有 shock 有用或某個階數具有普遍金融意義。

檢查全期績效之外，也看分年是否一致、附近階數是否有相近表現、失敗與後備比例是否過高。這些是評估結果是否值得後續研究的線索；不要把挑出的「看起來穩定」再當作未參與選模的證據。正式測試集繼續保留，基準不自動替換。

## 檔案與重跑

- 完整排行榜、分年表、每組每日績效、預測分數／名次、參數、估計紀錄与檢查皆保存在結果包。
- `fits.sqlite` 是每次估計的可續跑檢查點，含兩步價格預測、分數、後備標記、參數及日誌；不只保存勝者。
- 圖片與 CSV 可單獨開啟；程式和事前設定在資料包中。原市場資料與原參考專案未重新分發，重跑需使用本機既有資料。
- [JPX 官方評分](https://www.kaggle.com/code/smeitoma/jpx-competition-metric-definition)、[ARIMA 模型](https://otexts.com/fpp3/arima.html)、[ARIMA 多步預測](https://otexts.com/fpp3/arima-forecasting.html)。
'''
    report=report.replace('没有','沒有').replace('观测','觀測').replace('觀测','觀測').replace('紀錄与','紀錄與')
    local_report=report
    for plot_name in ['JPX-ARIMA-grid-heatmaps.png','JPX-ARIMA-grid-annual-heatmaps.png']:
        local_report=local_report.replace(f']({plot_name})',f']({OUT/plot_name})')
    (OUT/'JPX-ARIMA-grid-report.md').write_text(local_report)
    for src,dst in [('leaderboard.csv','JPX-ARIMA-grid-leaderboard.csv'),('annual.csv','JPX-ARIMA-grid-annual.csv'),('results.json','JPX-ARIMA-grid-results.json')]:shutil.copy2(RUN/src,OUT/dst)
    instructions='''# JPX fixed 25-model ARIMA grid
Python 3.14.2, statsmodels 0.14.6. Set OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
Use the original local reference project and prepared inputs from arima012_expanding (prices.npz, labels.pkl, baseline.pkl, calendar.csv, preparation.json, baseline_daily_metrics.csv); original market inputs are not redistributed. Adjust BASE and SOURCE paths for another computer. Hashes are recorded in preparation.json.
Scripts reside together. Run:
python run_arima_grid.py prepare
python run_arima_grid.py pilot --workers 6
python run_arima_grid.py run --workers 6
python evaluate_arima_grid.py
python report_arima_grid.py
run resumes all completed (order,year,stock) estimates from arima_grid/fits.sqlite. Never change fitting code or plan and reuse its database; hash guards prevent mixing specifications. Evaluation can be resumed at complete model boundaries. Every model is included before the final report is written.
Predictions under results/models/p*_q*/predictions.npz share keys in results/prediction_keys.npz. The database contains full two-step forecast prices and detailed logs; use export_arima_grid.py to make a chosen model's row-level CSV.
The original official metric implementation from the local reference project is required for independent evaluation. Formal test is untouched. All 25 models were selected within reused validation; reported top scores are exploratory.
'''
    package=OUT/'JPX-ARIMA-grid-code-and-results.zip'
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for name in ['run_arima_grid.py','evaluate_arima_grid.py','report_arima_grid.py','export_arima_grid.py','arima_grid_plan.md']:z.write(ROOT/name,name)
        for name in ['results.json','preparation.json','leaderboard.csv','annual.csv','baseline_daily.csv','prediction_keys.npz','pilot.log','run.log','fits.sqlite']:z.write(RUN/name,'results/'+name)
        for f in (RUN/'models').rglob('*'):
            if f.is_file():z.write(f,'results/'+str(f.relative_to(RUN)))
        z.writestr('JPX-ARIMA-grid-report.md',report)
        for name in ['JPX-ARIMA-grid-heatmaps.png','JPX-ARIMA-grid-annual-heatmaps.png']:z.write(OUT/name,name)
        z.writestr('README.md',instructions)
        z.writestr('requirements.txt','numpy==2.4.6\npandas==3.0.3\nscipy==1.17.1\nstatsmodels==0.14.6\nmatplotlib==3.10.9\n')
    with zipfile.ZipFile(package) as z:
        assert z.testzip() is None
        assert hashlib.sha256(z.read('run_arima_grid.py')).hexdigest()==s['runner_sha256']
        assert hashlib.sha256(z.read('evaluate_arima_grid.py')).hexdigest()==s['evaluator_sha256']
    print('DELIVERED',str(package),package.stat().st_size,flush=True)

if __name__=='__main__':main()
