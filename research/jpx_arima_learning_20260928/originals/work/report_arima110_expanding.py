"""Package verified second exploratory JPX experiment and learning notes."""
from pathlib import Path
import json, shutil, zipfile, hashlib
import pandas as pd
ROOT=Path(__file__).resolve().parent
RUN=ROOT/'arima110_expanding'
OUT=ROOT.parent/'outputs'

def main():
    s=json.loads((RUN/'results.json').read_text())
    audit=json.loads((RUN/'independent_audit.json').read_text())
    numerical=json.loads((RUN/'numerical_audit.json').read_text())
    full=json.loads((RUN/'full_filter_numerical_audit.json').read_text())
    assert audit['passed'] and full['passed']
    assert not numerical['invalid_numerical_fits']
    assert hashlib.sha256((ROOT/'run_arima110_expanding.py').read_bytes()).hexdigest()==s['run_code_sha256']
    a,b=s['common_arima'],s['common_v7']
    delta=s['paired_sharpe_difference'];lo,hi=delta['exploratory_95pct_interval']
    annual=pd.DataFrame([{'ValidationYear':r['year'],'ARIMA110_Sharpe':r['arima']['sharpe'],
        'v7_Sharpe':r['v7']['sharpe'],'ARIMA110_RankIC':r['arima']['mean_rank_ic'],
        'v7_RankIC':r['v7']['mean_rank_ic']} for r in s['annual']])
    rows='\n'.join(f"| {r['year']} | {r['arima']['sharpe']:+.6f} | {r['v7']['sharpe']:+.6f} | {r['arima']['mean_rank_ic']:+.6f} | {r['v7']['mean_rank_ic']:+.6f} |" for r in s['annual'])
    fits=json.loads((RUN/'fit_audits.json').read_text())
    boundaries=pd.DataFrame(fits).groupby('year').agg(train_start=('train_start','first'),train_end=('train_last','first'),first_signal=('first_signal','first'),last_signal=('last_signal','first'),training_dates=('training_slots','first'))
    boundaryrows='\n'.join(f'| {y} | {r.train_start}～{r.train_end} | {r.training_dates} | {r.first_signal}～{r.last_signal} |' for y,r in boundaries.iterrows())
    statuses='\n'.join(f'| {k} | {v:,} |' for k,v in s['fit_statuses'].items())
    verdict='高於' if a['sharpe']>b['sharpe'] else '低於'
    uncertainty='區間跨過零，不能斷言新模型穩定較優或較差。' if lo<=0<=hi else '此區間未跨零，但未校正反覆探索驗證集的選擇偏誤，不能當成正式 test 證據。'
    report=f'''# JPX：ARIMA(1,1,0) 與 expanding window

本輪使用者提出模型、expanding window 與年度驗證，並確認對齊明天→後天的 JPX 目標；助理實作與核對。這份是實驗紀錄與學習說明，不取代使用者自行撰寫的企劃書。

## 結果

同 {s['common_days']} 個驗證日、{s['forecast_rows']:,} 個股票日，新設定 Sharpe {verdict}既有 v7。未自動更換基準。

| 模型 | Sharpe（未年化、未扣成本） | 平均每日 Rank IC |
|---|---:|---:|
| ARIMA(1,1,0)，expanding | {a['sharpe']:+.6f} | {a['mean_rank_ic']:+.6f} |
| v7_equal，既有歷史結果 | {b['sharpe']:+.6f} | {b['mean_rank_ic']:+.6f} |

新模型減 v7 的 Sharpe 差為 {delta['arima_minus_v7']:+.6f}；20 日區塊配對 bootstrap 2,000 次的探索性 95% 區間為 [{lo:+.6f}, {hi:+.6f}]。{uncertainty}

| ValidationYear | ARIMA Sharpe | v7 Sharpe | ARIMA Rank IC | v7 Rank IC |
|---|---:|---:|---:|---:|
{rows}

## 後天可以使用預測的明天嗎？

可以。這叫遞迴預測。今天 t 產生的兩步預測，都只能使用截至 t 的資訊。先用今日已知價格變動預測明日變動，再將預測的變動代入同一模型，推到後天；不需要訓練另一個模型，也不需要偷看明天實際價格。

本輪採無漂移 ARIMA(1,1,0)，P 為因果調整收盤價：

ΔP[t] = φ·ΔP[t−1] + ε[t]

在今天及昨天價格都有觀測值時：

1. ΔP[t] = P[t] − P[t−1]
2. 預測 ΔP[t+1|t] = φ·ΔP[t]
3. 預測 P[t+1|t] = P[t] + φ·ΔP[t]
4. 預測 ΔP[t+2|t] = φ·預測 ΔP[t+1|t] = φ²·ΔP[t]
5. 預測 P[t+2|t] = 預測 P[t+1|t] + φ²·ΔP[t]
6. 排名分數 = 預測 P[t+2|t] / 預測 P[t+1|t] − 1

例：昨天 98、今天 100、φ=0.5，預測明天 101、後天 101.5，排名分數約 0.495%。係數是假設值，非所有股票共用。

預測期間尚未觀測的 ε 使用條件期望零；真實未來仍會有誤差。q=0 表示不加入落後創新誤差的 MA 項，並不代表沒有 shock 或 shock 不會經由 AR 傳遞。明天公布真實价格後，可以更新明天的新排名，不能回頭修正今天的歷史排名。

兩個點預測的比值只是報酬代理，不保證等於報酬比值的精確條件期望。遇到缺失價格，使用 forward filter 中的估計狀態，不能套用需要兩個已知相鄰價格的簡式。

## 本轮固定設定

- 每檔股票獨立 ARIMA(1,1,0)、無漂移；將「AR(1,1)」明確解讀為 AR 一階、差分一次、MA 零階。
- 訓練從最早可用歷史持續擴大，每年重估一次參數；驗證年內固定參數，每日使用已公布價格更新狀態。至少 126 個有效訓練價格。
- 這是每年擴大訓練的 walk-forward，不是每天重估係數。前一驗證年的資料，只有到了下一年度才加入參數估計。
- 排名由高至低，每日完整唯一名次，同分依股票代碼；前 200 做多、後 200 做空，官方名次權重 2→1。不可另行自選倉位或不交易。
- 採用已核對的因果價格調整及原始 Target；不裁切有限極端值。最大概似估計、AR 平穩約束、L-BFGS 最多 500 次，未收斂則同設定續跑最多 1,000 次。
- 尺度轉換只用年度訓練資料；缺值保留於日曆。資料不足、估計失敗或預測價格非正等無效情況，固定以零報酬分數後備，股票仍參加排名。
- Sharpe 優先、Rank IC 次之；MSE 僅診斷。formal test 未讀取、未評分。

| 驗證年度標籤 | 參數估計資料日期 | 訓練日數 | 驗證訊號日期 |
|---|---|---:|---|
{boundaryrows}

延用既有 ValidationYear 分組以保持可比；訊號日與報酬實現日有位移，因此邊界含前一年末交易日。2021 只到現有驗證資料終點，非完整年度。

## 核對與失敗處理

共 {s['stock_year_fits']:,} 個股票年度模型：

| 狀態 | 數量 |
|---|---:|
{statuses}

後備 {s['fallback_stock_days']:,} 個股票日（{s['fallback_fraction']:.2%}），其中 {s['selected_fallback_stock_days']:,} 個進入前後各 200 檔。估計或數值失敗的原因逐筆留存，不能刪掉後只報成功模型的績效。

- 全部年度估計資料早於第一個預測日，且 expanding 起點不變；預測函式不接收 Target。
- {s['recursive_checks']:,} 次觀測相鄰價格的遞迴公式核對通過，最大價格差 {s['recursive_max_price_error']:.3g}。
- {s['prefix_checks']} 次內建截斷檢查，加上 {audit['independent_prefix_and_future_perturbation_checks']} 次獨立截斷／未來價格擾動檢查通過。修改未來價格不改變當下預測。
- 所有 {full['accepted_fits_checked']:,} 個接受模型的完整 forward filter 創新變異數有效；沒有依驗證績效修正或剔除模型。
- 官方 Sharpe 獨立重算差異 {s['official_formula_max_error']:.3g}；原 v7 每日報酬亦重現。
- 原始 Target 未改寫。前輪核對發現少量 Target 與本地調整價格比值有微小差異，原因未確定；兩個模型仍共用原始標籤。

## 寫企劃時還要考慮什麼

1. 上一輪 ARIMA(5,1,5) 已轉成明天→後天的變化率。其結果不能歸因於忘記轉換；輸入是價格差分，不是百分比報酬。
2. 本輪同時更改階數與歷史窗口，因此與上一輪的差異無法單獨歸因於移除 MA。v7 也是既有設定的參照，並未重跑成相同估計方式。
3. Expanding 保留更多歷史，可能穩定估計，也可能讓舊 regime 主導係數；不能保證更能適應市場改變。
4. 無漂移的 ARIMA(1,1,0) 有特殊結構：相鄰觀測完整時，第二日的預測價格變動為 φ²·ΔP[t]。即使 φ 為負，兩步後方向也會轉回與今天變動同號；對齊預測區間會改變策略含義。
5. 簡化模型仍需要檢查殘差、自相關與分年穩定性。本輪未聲稱已通過全部統計診斷；只完成這個固定設定的回測與數值／時間因果核對。
6. 驗證資料已反覆用於模型探索，不能視為全新泛化證據；成本、成交限制亦未納入官方分數。模型應先依 Sharpe 判斷，不能只看 Rank IC 或某一年。

## 來源與重跑

- [JPX 官方資料說明](https://www.kaggle.com/competitions/jpx-tokyo-stock-exchange-prediction/data)
- [JPX 官方評分程式](https://www.kaggle.com/code/smeitoma/jpx-competition-metric-definition)
- [statsmodels ARIMA](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html)，本次版本 0.14.6。
- 資料包包含程式、事前設定、參數紀錄、每日排名及報酬、核對結果；原始市場資料與原參考專案不重新分發。程式中的本機路徑需依電腦調整。
'''
    report=report.replace('真实','真實').replace('价格','價格').replace('本轮','本輪')
    OUT.mkdir(exist_ok=True)
    (OUT/'JPX-ARIMA110-expanding-report.md').write_text(report)
    annual.to_csv(OUT/'ARIMA110-expanding-annual-comparison.csv',index=False)
    shutil.copy2(RUN/'results.json',OUT/'ARIMA110-expanding-results.json')
    scripts=['run_arima110_expanding.py','audit_arima110_expanding.py','validate_arima110_expanding_numerics.py','report_arima110_expanding.py','arima110_expanding_plan.md']
    records=['results.json','preparation.json','calendar.csv','fit_audits.json','daily_metrics.csv','baseline_daily_metrics.csv','predictions.csv.gz','independent_audit.json','numerical_audit.json','full_filter_numerical_audit.json','run.log','pilot_aligned.log']
    instructions='''# Reproduce the experiment
Python 3.14.2. Set OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
Use a fresh directory for this specification; never reuse a prior experiment's jobs.
Change BASE and RAW absolute paths in the scripts to your local reference project and JPX ZIP. Data hash is in preparation.json. This archive omits raw data and reference project files.
Place scripts and arima110_expanding_plan.md together, then run:
python run_arima110_expanding.py prepare
python run_arima110_expanding.py pilot --workers 6
python run_arima110_expanding.py run --workers 6
python validate_arima110_expanding_numerics.py
python validate_arima110_expanding_numerics.py --future
python run_arima110_expanding.py evaluate
python audit_arima110_expanding.py
python report_arima110_expanding.py
The original run reused the identical prepared inputs from the previous experiment; prepare rebuilds them from original train_files and the existing baseline. Formal test is not used.
Checkpoints are written in arima110_expanding/jobs. The old unaligned pilot was excluded from this final run and archive; all delivered scores use t+1 to t+2.
'''
    package=OUT/'JPX-ARIMA110-expanding-code-and-results.zip'
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as z:
        for name in scripts:z.write(ROOT/name,name)
        for name in records:z.write(RUN/name,'results/'+name)
        z.write(OUT/'JPX-ARIMA110-expanding-report.md','JPX-ARIMA110-expanding-report.md')
        z.write(ROOT.parent/'JPX-experiment-preferences.json','JPX-experiment-preferences.json')
        z.writestr('README.md',instructions)
        z.writestr('requirements.txt','numpy==2.4.6\npandas==3.0.3\nscipy==1.17.1\nstatsmodels==0.14.6\n')
    with zipfile.ZipFile(package) as z:
        assert z.testzip() is None
        assert hashlib.sha256(z.read('run_arima110_expanding.py')).hexdigest()==s['run_code_sha256']
    print('DELIVERED',package,package.stat().st_size,flush=True)

if __name__=='__main__':main()
