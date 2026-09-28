"""Package the verified JPX MA(2) experiment and concise learning notes."""
from pathlib import Path
import json,hashlib,shutil,zipfile
import pandas as pd
ROOT=Path(__file__).resolve().parent
RUN=ROOT/'arima012_expanding'
OUT=ROOT.parent/'outputs'

def main():
    s=json.loads((RUN/'results.json').read_text())
    audit=json.loads((RUN/'independent_audit.json').read_text())
    numeric=json.loads((RUN/'numerical_audit.json').read_text())
    full=json.loads((RUN/'full_filter_numerical_audit.json').read_text())
    assert audit['passed'] and full['passed'] and not numeric['invalid_numerical_fits']
    assert hashlib.sha256((ROOT/'run_arima012_expanding.py').read_bytes()).hexdigest()==s['run_code_sha256']
    old=json.loads((ROOT/'arima110_expanding/results.json').read_text())
    a,b=s['common_arima'],s['common_v7'];prev=old['common_arima']
    delta=s['paired_sharpe_difference'];lo,hi=delta['exploratory_95pct_interval']
    verdict='高於' if a['sharpe']>b['sharpe'] else '低於'
    uncertainty='區間跨零，不能斷言新模型穩定較好或較差。' if lo<=0<=hi else '區間未跨零，但未校正反覆探索驗證集的選擇偏誤，不能當成正式 test 證據。'
    prior={r['year']:r['arima'] for r in old['annual']}
    annual=pd.DataFrame([{'ValidationYear':r['year'],'ARIMA012_Sharpe':r['arima']['sharpe'],
        'ARIMA110_Sharpe':prior[r['year']]['sharpe'],'v7_Sharpe':r['v7']['sharpe'],
        'ARIMA012_RankIC':r['arima']['mean_rank_ic'],'v7_RankIC':r['v7']['mean_rank_ic']} for r in s['annual']])
    annualrows='\n'.join(f"| {r['year']} | {r['arima']['sharpe']:+.6f} | {prior[r['year']]['sharpe']:+.6f} | {r['v7']['sharpe']:+.6f} | {r['arima']['mean_rank_ic']:+.6f} |" for r in s['annual'])
    fits=pd.DataFrame(json.loads((RUN/'fit_audits.json').read_text()))
    bounds=fits.groupby('year').agg(start=('train_start','first'),end=('train_last','first'),slots=('training_slots','first'),first=('first_signal','first'),last=('last_signal','first'))
    boundaryrows='\n'.join(f'| {y} | {r.start}～{r.end} | {r.slots} | {r.first}～{r.last} |' for y,r in bounds.iterrows())
    names={'ok':'成功估計','insufficient_history':'歷史不足','fit_not_converged':'未收斂','fit_or_filter_exception':'數值求解失敗','constant_or_invalid_scale':'價格尺度無效'}
    statuses='\n'.join(f'| {names.get(k,k)} | {v:,} |' for k,v in s['fit_statuses'].items())
    report=f'''# JPX：ARIMA(0,1,2) expanding-window 回測

使用者提出以 MA 誤差資訊預測價格變動；助理實作與核對。沿用無漂移、expanding window、年度驗證與明天→後天的 JPX 排名區間。沒有另行搜尋階數、加入漂移或依結果翻轉訊號。

## 結果

同 {s['common_days']} 個驗證日、{s['forecast_rows']:,} 個股票日，ARIMA(0,1,2) 的 Sharpe {verdict}既有 v7。正式 test 未讀取、未評分，基準沒有自動更換。

| 模型 | Sharpe（未年化、未扣成本） | 平均每日 Rank IC |
|---|---:|---:|
| 本輪 ARIMA(0,1,2)，expanding | {a['sharpe']:+.6f} | {a['mean_rank_ic']:+.6f} |
| 前輪 ARIMA(1,1,0)，expanding | {prev['sharpe']:+.6f} | {prev['mean_rank_ic']:+.6f} |
| v7_equal，既有歷史參照 | {b['sharpe']:+.6f} | {b['mean_rank_ic']:+.6f} |

本輪減 v7 的 Sharpe 差 {delta['arima_minus_v7']:+.6f}；20 日區塊配對 bootstrap 2,000 次、固定 seed 20260923，探索性 95% 區間 [{lo:+.6f}, {hi:+.6f}]。{uncertainty}

前輪 ARIMA(0,1,1) 全部分數為零，其 Sharpe 僅來自股票代碼同分排序，不把它當成有預測訊號的模型來宣稱增益。

| ValidationYear | 本輪 MA(2) Sharpe | 前輪 AR(1) Sharpe | v7 Sharpe | 本輪 Rank IC |
|---|---:|---:|---:|---:|
{annualrows}

## MA(2) 這次有產生訊號嗎？

有 {s['second_step_nonzero_stock_days']:,} 個股票日的分數非零；全部同分的日期為 {s['constant_score_days']} 日，每日最少有 {s['minimum_daily_distinct_scores']:,} 個不同分數。這回答了「能否產生不同排序分數」，但是否有用，仍要看上面的驗證 Sharpe，不能從分數非零直接推論有預測力。

模型為：

ΔP[t] = ε[t] + θ₁·ε[t−1] + θ₂·ε[t−2]

站在今天，令 ê 為根據目前資訊估計的創新誤差：

- 預測明天價格變動 = θ₁·ê[t] + θ₂·ê[t−1]。
- 預測後天價格變動 = θ₂·ê[t]。
- 預測第三天價格變動 = 0。

因此明天預測價格 = 今天價格 + 第一步變動；後天預測價格 = 明天預測價格 + 第二步變動。昨天的誤差透過第一步進入明天與後天的價格水準，並未被漏掉。未觀測的未來 error 使用條件期望零，不能把預測的價格變動當作新的 error。

score[t] = 預測 P[t+2|t] / 預測 P[t+1|t] − 1

此分數是價格點預測的比值代理，不保證等於隨機報酬比值的精確條件期望。缺值時使用 forward state 中的估計，沒有偷看未來平滑狀態。MA 使用過去估計的創新誤差；它沒有辨認具體新聞事件，也不等於可以預測尚未出現的 surprise 本身。

## 固定設定與時間切分

- 每檔股票獨立估計 ARIMA(0,1,2)，無漂移、MA 可逆，至少 126 個有效訓練價格。Gaussian 最大概似、L-BFGS 500 次，未收斂續跑最多 1,000 次；沒有用驗證 Sharpe 挑選起點或優化器。
- 每年使用從最早可用日期開始的所有已知歷史重新估計；年內固定係數，每天只更新已觀測價格的模型狀態。Expanding window 不代表每天重估參數。
- 與之前共用相同因果調整價格、股票池與原始 Target。缺失日期保留，尺度只由訓練資料決定，不裁切有限極端值。
- 依分數由高到低完整唯一排名，同分按股票代碼升冪。官方前後各 200 檔、名次權重 2→1，不另行決定投入金額。
- 資料不足、未收斂、數值無效或非正預測價格時使用固定零分後備，仍保留股票於當日排名，報告後備選中數量。
- Sharpe 優先、Rank IC 次之；MSE 僅診斷。沒有因績效較差而翻轉分數或更換回測區間。

| 驗證年度標籤 | 參數估計資料 | 訓練日数 | 驗證訊號日 |
|---|---|---:|---|
{boundaryrows}

ValidationYear 與既有模型一致；訊號和報酬實現日期有位移，年度邊界因此含前一年末訊號。2021 只到現有驗證資料終點，並非完整年度。

## 估計狀態與核對

共 {s['stock_year_fits']:,} 個股票年度模型，{s['retried_fits']} 個需要第二次優化嘗試。

| 狀態 | 數量 |
|---|---:|
{statuses}

後備 {s['fallback_stock_days']:,} 個股票日（{s['fallback_fraction']:.2%}），其中 {s['selected_fallback_stock_days']:,} 個進入前後 200 檔。這份績效包含事前固定後備規則，不能只報成功模型。

- 全部年度訓練結束早於第一個預測日；expanding 起點一致；預測函式不接收 Target。
- {s['third_step_flat_checks']:,} 次 MA(2) 第三步截斷核對通過，第三步與第二步價格的最大差 {s['third_step_max_price_gap']:.3g}。
- {s['prefix_checks']} 次內建截斷核對，加上 {audit['independent_prefix_and_future_perturbation_checks']} 次獨立截斷／未來資料擾動檢查通過；修改未來資料不會改變早期預測，最大價格預測差 {audit['max_price_forecast_error']:.3g}。
- {full['accepted_fits_checked']:,} 個接受模型的完整 forward filter 創新變異數有效；沒有依驗證績效剔除模型。
- 官方 Sharpe 獨立重算差 {s['official_formula_max_error']:.3g}；原 v7 每日報酬重現，日期與股票鍵完全對齊。
- 原始 Target 未改寫。部分標籤與本地因果價格比值有微小差異，原因未確定，所有比較仍共用原始標籤。

## 寫企劃時可下的結論

「MA(2) 的第二個誤差項能讓目前的 JPX 預測區間產生不同股票分數」是結構與程式核對支持的結論；「這些誤差能穩定改善排名」則需要驗證結果支持，不能混為一談。

與 ARIMA(1,1,0) 的比較同時更換 AR 與 MA 結構，不是單獨增加一個 MA 項的實驗。v7 輸入、模型與更新方式也不同，只是既有績效參照。反覆使用相同驗證資料會產生選擇偏誤，正 Sharpe、較佳單一年份或較佳平均 Rank IC 均不能直接當成泛化證據。

本輪未納入成本與成交限制，也未宣稱已完成全部残差／模型適配診斷。沒有自動換掉 v7，也沒有繼續試其他階數。

## 來源與重跑

- [MA 模型定義](https://otexts.com/fpp3/MA.html)
- [ARIMA 多步預測](https://otexts.com/fpp3/arima-forecasting.html)
- [JPX 官方資料](https://www.kaggle.com/competitions/jpx-tokyo-stock-exchange-prediction/data)
- [官方評分程式](https://www.kaggle.com/code/smeitoma/jpx-competition-metric-definition)

實際 statsmodels 0.14.6。資料包包含實作、事前設定、參數與失敗紀錄、逐筆預測排名、每日與分年結果、獨立核對及環境版本。原始市場資料與來源專案未重新分發，重跑需調整本機來源路徑。
'''
    report=report.replace('日数','日數').replace('残差','殘差')
    OUT.mkdir(exist_ok=True)
    (OUT/'JPX-ARIMA012-expanding-report.md').write_text(report)
    annual.to_csv(OUT/'ARIMA012-expanding-annual-comparison.csv',index=False)
    shutil.copy2(RUN/'results.json',OUT/'ARIMA012-expanding-results.json')
    scripts=['run_arima012_expanding.py','audit_arima012_expanding.py','validate_arima012_expanding_numerics.py','report_arima012_expanding.py','arima012_expanding_plan.md']
    records=['results.json','preparation.json','calendar.csv','fit_audits.json','daily_metrics.csv','baseline_daily_metrics.csv','predictions.csv.gz','independent_audit.json','numerical_audit.json','full_filter_numerical_audit.json','run.log','pilot.log']
    instructions='''# Reproduce JPX ARIMA(0,1,2)
Python 3.14.2. Install requirements.txt. Set OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
Use a fresh directory; never reuse another experiment's checkpoints. Change BASE and RAW paths in scripts to the original local JPX project and ZIP. Source files are not redistributed. Raw ZIP hash is in preparation.json.
Place scripts and plan in one directory; run:
python run_arima012_expanding.py prepare
python run_arima012_expanding.py pilot --workers 6
python run_arima012_expanding.py run --workers 6
python validate_arima012_expanding_numerics.py
python validate_arima012_expanding_numerics.py --future
python run_arima012_expanding.py evaluate
python audit_arima012_expanding.py
python report_arima012_expanding.py
For the report, place prior-arima110-results.json at arima110_expanding/results.json next to the scripts. It is the saved previous experiment, not a new refit.
The original run reused identical prepared inputs from the previous experiment; prepare rebuilds them from original train_files and the reference project. Formal test is never loaded.
'''
    package=OUT/'JPX-ARIMA012-expanding-code-and-results.zip'
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as z:
        for n in scripts:z.write(ROOT/n,n)
        for n in records:z.write(RUN/n,'results/'+n)
        z.write(OUT/'JPX-ARIMA012-expanding-report.md','JPX-ARIMA012-expanding-report.md')
        z.write(OUT/'ARIMA012-expanding-annual-comparison.csv','ARIMA012-expanding-annual-comparison.csv')
        z.write(ROOT.parent/'JPX-experiment-preferences.json','JPX-experiment-preferences.json')
        z.write(ROOT/'arima110_expanding/results.json','prior-arima110-results.json')
        z.writestr('README.md',instructions)
        z.writestr('requirements.txt','numpy==2.4.6\npandas==3.0.3\nscipy==1.17.1\nstatsmodels==0.14.6\n')
    with zipfile.ZipFile(package) as z:
        assert z.testzip() is None
        assert hashlib.sha256(z.read('run_arima012_expanding.py')).hexdigest()==s['run_code_sha256']
    print('DELIVERED',str(package),package.stat().st_size,flush=True)

if __name__=='__main__':main()
