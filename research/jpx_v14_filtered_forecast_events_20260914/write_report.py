from pathlib import Path
import json,pickle,hashlib,platform,sys
import numpy as np
import pandas as pd
from features import ROOT,V8,VARIANTS,FINS
from run import NAMES

def save(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))
def main():
    with (V8/'inputs.pkl').open('rb') as h:data=pickle.load(h)
    labels=data[0][['SignalDate','SecuritiesCode','Target']].rename(columns={'SignalDate':'Date'}).copy();del data
    stats=json.loads((ROOT/'feature_summary.json').read_text());fa=json.loads((ROOT/'feature_audit.json').read_text())
    results={v:json.loads((ROOT/v/'results.json').read_text()) for v in VARIANTS}
    audits={v:json.loads((ROOT/v/'audit.json').read_text()) for v in VARIANTS}
    assert fa['passed'] and all(a['passed'] for a in audits.values())
    summaries={};daily={};annual=[]
    for v in VARIANTS:
        ranks=pd.concat([pd.read_csv(ROOT/v/f'ranks_{year}.csv.gz',parse_dates=['Date']) for year in [2018,2019,2020,2021]],ignore_index=True)
        r=ranks.merge(labels,on=['Date','SecuritiesCode'],validate='one_to_one');r=r.loc[r.Target.notna()].copy()
        r['SE']=(r.g-r.Target)**2
        rows=[]
        for date,g in r.groupby('Date'):
            good=g.TrainingEligible
            ic=float(g.g.rank().corr(g.Target.rank())) if g.Target.nunique()>1 and g.g.nunique()>1 else np.nan
            rows.append({'Date':date,'RankIC':ic,'ZeroPredictionMSE':float((g.Target**2).mean()),
                'EligibleMSEContribution':float(g.loc[good,'SE'].sum()/len(g)),
                'ExcludedMSEContribution':float(g.loc[~good,'SE'].sum()/len(g)),
                'ExcludedRows':int((~good).sum()),'ExcludedAbsPredictionGT1':int(g.loc[~good,'g'].abs().gt(1).sum())})
        d=pd.read_csv(ROOT/v/'daily_spread_returns.csv',parse_dates=['Date']).merge(pd.DataFrame(rows),on='Date',validate='one_to_one')
        np.testing.assert_allclose(d.AllStockForecastMSE,d.EligibleMSEContribution+d.ExcludedMSEContribution,rtol=1e-11,atol=1e-12)
        d.to_csv(ROOT/v/'diagnostic_daily.csv',index=False);daily[v]=d
        summaries[v]={'mean_rank_ic':float(d.RankIC.mean()),'rank_ic_defined_days':int(d.RankIC.notna().sum()),
            'zero_prediction_mse':float(d.ZeroPredictionMSE.mean()),'mean_forecast_mse':float(d.AllStockForecastMSE.mean()),
            'eligible_mse_contribution':float(d.EligibleMSEContribution.mean()),'excluded_mse_contribution':float(d.ExcludedMSEContribution.mean()),
            'excluded_share_of_full_pool_mse':float(d.ExcludedMSEContribution.sum()/d.AllStockForecastMSE.sum()),
            'excluded_validation_rows_with_known_target':int(d.ExcludedRows.sum()),'max_absolute_prediction':float(r.g.abs().max()),
            'abs_prediction_gt_1_rows':int(r.g.abs().gt(1).sum())}
        for yr,g in d.groupby('ValidationYear'):
            s=g.OfficialDailySpread;annual.append({'Variant':v,'ValidationYear':int(yr),'Days':len(g),
                'Sharpe':float(s.mean()/s.std(ddof=1)),'MSE':float(g.AllStockForecastMSE.mean()),'RankIC':float(g.RankIC.mean())})
    paired=daily['sgd_only'][['Date','ValidationYear']].copy()
    for v,d in daily.items():
        cols=['OfficialDailySpread','AllStockForecastMSE','RankIC','ExcludedMSEContribution']
        paired=paired.merge(d[['Date']+cols].rename(columns={c:v+'_'+c for c in cols}),on='Date',validate='one_to_one')
    paired.to_csv(ROOT/'daily_comparison.csv',index=False);pd.DataFrame(annual).to_csv(ROOT/'annual_comparison.csv',index=False)
    u={v:pd.read_csv(ROOT/v/'training_updates.csv') for v in VARIANTS}
    cols=['Date','SignalDate','KnownLabelStocks','TrainingStocks','ThresholdSkippedStocks']
    pd.testing.assert_frame_equal(u['sgd_only'][cols],u['sgd_sqrt'][cols])
    pd.DataFrame({'Parameter':NAMES,**{v:[results[v]['final_parameters'][c] for c in NAMES] for v in VARIANTS}}).to_csv(ROOT/'final_parameters.csv',index=False)
    for v in VARIANTS:
        fin=pd.read_pickle(ROOT/'financial_signal_features.pkl')
        forbidden=np.flatnonzero(~fin.TrainingEligible.to_numpy())
        # Saved skip inventory is prediction-day information, without labels.
        with (V8/'inputs.pkl').open('rb') as h:base=pickle.load(h)[0]
        skipped=base.loc[forbidden,['SignalDate','SecuritiesCode']].copy();skipped['MaxAbsFinancialInput']=fin.iloc[forbidden].TrainingFeatureMaxAbs.to_numpy()
        skipped.to_csv(ROOT/'training_filter_inventory.csv',index=False)
        break
    result={'version':'v14','variants':results,'diagnostics':summaries,'annual':annual,'audits':audits,
        'feature_audit':fa,'same_filtered_training_pool':True,'training_limit_absolute':100.,'evaluation_pool_filtered':False,
        'no_new_forecast_no_new_event':True,'old_events_continue_decay':True,
        'unverified_revision_and_correction_rows_excluded':True,'costs_included':False,'fresh_test_set_used':False}
    save(ROOT/'results.json',result)
    save(ROOT/'paired_audit.json',{'passed':True,'same_filtered_stock_days':True,'same_features':True,'same_seed_initial_state_and_optimizer':True,
        'only_within_v14_difference':'additional floor(sqrt(filtered finite-label N)) daily full updates','complete_evaluation_stock_pool_preserved':True})
    lines=['# v14：100 倍訓練門檻與 Forecast 新消息規則','',
        '本輪已加入絕對值大於 100 的財報特徵訓練篩選，並把「沒有新的全年 Forecast」的當次新增 Forecast 訊號設為零。舊事件仍按照 exp(−a/9) 衰減。兩組沿用相同分塊 SGD，混合組另加每日 ⌊√N⌋ 次整體更新。','',
        '## 本次規則','',
        '- 篩選值為當日 15 個財報成長成分，已衰減並加總、尚未乘 d 或 β。任一 |值|>100，該股票日不參與 SGD 或整體 loss；±100 保留。不是對原始營收金額、Target 或股票代碼做篩選。','- 標記在 SignalDate 凍結，標籤成熟時沿用。N 是通過篩選且 Target 已知的股票數。','- 無新 Forecast、缺少該欄或同財年該指標 Forecast 未變，當次不新增該指標的 Forecast 季增、年增或修正訊號。各指標分開判斷。','- 只因累計實績更新而改變剩餘季度預估時，更新內部預期狀態，供後續實際驚喜比較，但不新增 Forecast 事件。舊 Forecast 事件沒有被清除。','- 實際驚喜仍獨立計算；新財年第一次明確公布 Forecast 視為新值。','- 為排除 v13 已確認的合併／個別錯配，本輪不套用缺乏明確會計口徑的 ForecastRevision 和 NumericalCorrection。保留常規報表內明確口徑的 Forecast 更新；未猜測修正歸屬。','',
        f'來源中保守排除 {stats["counts"].get("unverified_accounting_basis_ForecastRevision",0):,} 筆 ForecastRevision 與 {stats["counts"].get("unverified_accounting_basis_NumericalCorrection",0):,} 筆 NumericalCorrection；仍產生 {stats["counts"]["events_U"]:,} 筆指標層的预期修正事件。此做法降低來源覆蓋，並不宣稱已全面核實所有其他財報資料語意。','',
        '## 完整股票池的驗證結果','',
        '| 相同 953 日 | 單獨 SGD | SGD＋⌊√N⌋ |','|---|---:|---:|']
    for label,values in [('未年化多空 Sharpe',[results[v]['validation']['official_style_unannualized_sharpe'] for v in VARIANTS]),
        ('平均每日預測 MSE',[results[v]['mean_daily_forecast_mse'] for v in VARIANTS]),
        ('平均每日 Rank IC',[summaries[v]['mean_rank_ic'] for v in VARIANTS]),
        ('全部預測為零的 MSE',[summaries[v]['zero_prediction_mse'] for v in VARIANTS])]:
        lines.append('| '+label+' | '+' | '.join(f'{x:.8f}' for x in values)+' |')
    lines+=['','Rank IC 是預測分數與實際報酬的排名相關；952 日可定義，另一日全部 Target 為零。MSE 是按日等權的平均股票平方誤差。零預測只用於誤差基準，沒有選股排序能力。','',
        '| 訓練／評估篩選統計 | 單獨 SGD | SGD＋⌊√N⌋ |','|---|---:|---:|']
    for label,key in [('原本有已知 Target 的訓練股票日','KnownLabelStocks'),('因 100 倍門檻跳過的訓練股票日','ThresholdSkippedStocks'),('實際逐檔訓練次數','TrainingStocks'),('額外整體更新嘗試次數','FullAttempts')]:
        lines.append('| '+label+' | '+' | '.join(f'{results[v][key]:,}' for v in VARIANTS)+' |')
    lines+=['',
        '只跳過訓練，不會自動把該檔的預測值截斷或從選股池移除。本輪完整保留它們的預測及驗證，避免只在容易的子樣本上評估。','',
        '| 完整驗證 MSE 的來源 | 單獨 SGD | SGD＋⌊√N⌋ |','|---|---:|---:|']
    for label,key in [('可訓練股票日的 MSE 貢獻','eligible_mse_contribution'),('超門檻股票日的 MSE 貢獻','excluded_mse_contribution')]:
        lines.append('| '+label+' | '+' | '.join(f'{summaries[v][key]:.8f}' for v in VARIANTS)+' |')
    lines+=['','兩項貢獻相加等於完整股票池 MSE，分母使用各日全部有限 Target 股票數，沒有更換評估分母。','',
        '## 預測誤差小，是否代表更準？','',
        '**是。相同資料、相同 MSE 定義下，較小的驗證 MSE 就代表報酬數值預測更準。**它不保證股票之間的排序更準，因為排名只看誰高誰低。','',
        '| 示意 | 股票 A | 股票 B |','|---|---:|---:|','| 實際報酬 | 10% | 5% |','| 舊預測 | 30% | 20% |','| 新預測 | 7% | 8% |','',
        '舊預測雖排序正確，MSE=(0.20²+0.15²)/2=0.03125；新預測排序相反，但 MSE=(−0.03²+0.03²)/2=0.0009，數值誤差更小。應依目標同時查看 MSE 與排名指標。','',
        '## 分年度對照','', '| 年度 | 日數 | 單獨 SGD Sharpe | 混合 Sharpe | 單獨 SGD MSE | 混合 MSE |','|---|---:|---:|---:|---:|---:|']
    for year in [2018,2019,2020,2021]:
        a=next(x for x in annual if x['Variant']=='sgd_only' and x['ValidationYear']==year);b=next(x for x in annual if x['Variant']=='sgd_sqrt' and x['ValidationYear']==year)
        lines.append(f'| {year} | {a["Days"]} | {a["Sharpe"]:.6f} | {b["Sharpe"]:.6f} | {a["MSE"]:.8f} | {b["MSE"]:.8f} |')
    lines+=['','## 核對與比較限制','',
        f'- {fa["events"]:,} 筆事件及 {fa["all_signal_rows_checked"]:,} 個股票日的公式、公告時間與衰減重算通過。','- 已測试 ±100 邊界、負值、無新 Forecast、缺欄不產生新事件，以及無法辨識口徑的修正列不更新狀態。','- 原 1808 SourceRow 22861 未出現在新事件或來源引用中。','- 兩組訓練資料、篩選標記、初始參數、每日順序和學習率規則完全相同。','- 每筆實際使用的訓練股票都通過 SignalDate 的 100 倍檢查；全部逐檔／整體更新和全部排名、MSE、多空報酬皆重算核對。','',
        '與 v13 比較時，本次同時更動了口徑處理、新事件條件及訓練篩選，不能把差異歸因某一項。v13 本身有已知資料問題，只可作歷史診斷參考。本輪没有新的獨立測試期間，也未扣交易成本。正式基準仍為 v7。','',
        '## 檔案','']
    for name,label in [('daily_comparison.csv','每日對照'),('annual_comparison.csv','分年度對照'),('training_filter_inventory.csv','訓練門檻標記清單'),('final_parameters.csv','27 個最終參數'),('experiment_plan.md','實作規則'),('results.json','完整結果與稽核')]:
        lines.append(f'- [{label}]({ROOT/name})')
    (ROOT/'JPX-v14-filtered-forecast-report.md').write_text('\n'.join(lines)+'\n')
    save(ROOT/'runtime_manifest.json',{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'platform':platform.platform(),'compiler':'clang++ -O3 -std=c++17 -dynamiclib'})
    sources=list(ROOT.glob('*.py'))+list(ROOT.glob('*.cpp'))+[ROOT/'experiment_plan.md',ROOT/'model.dylib',ROOT/'replay.dylib',ROOT/'financial_events.pkl',ROOT/'financial_signal_features.pkl']
    save(ROOT/'source_hashes.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    print(json.dumps({'variants':{v:{'Sharpe':results[v]['validation']['official_style_unannualized_sharpe'],'MSE':results[v]['mean_daily_forecast_mse'],'skipped_training':results[v]['ThresholdSkippedStocks']} for v in VARIANTS},'diagnostics':summaries}),flush=True)
if __name__=='__main__':main()
