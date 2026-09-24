from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;BASE=R.parent
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))
def main():
    assert json.loads((R/'audit.json').read_text())['passed']
    rr=json.loads((R/'results.json').read_text());pred=pd.read_pickle(R/'predictions.pkl');v=pred[pred.IsValidation];bound=v[v.TrainingEligible]
    yearly=pd.read_csv(R/'comparison_by_year.csv');days=[]
    fig,axes=plt.subplots(2,1,figsize=(9,7),constrained_layout=True)
    for ax,(label,df) in zip(axes,[('Including extreme events',v),('Both growth rates within 100x',bound)]):
        d=df.groupby('AvailableDate').agg(N=('SquaredError','size'),SSE=('SquaredError','sum'),BaselineSSE=('BaselineSquaredError','sum')).reset_index()
        d['CumulativeMSE']=d.SSE.cumsum()/d.N.cumsum();d['CumulativeBaselineMSE']=d.BaselineSSE.cumsum()/d.N.cumsum();d['CumulativeMSEImprovement']=1-d.CumulativeMSE/d.CumulativeBaselineMSE;d['Scope']=label;days.append(d)
        ax.plot(d.AvailableDate,d.CumulativeMSE,label='a + B * QoQ',color='#007c82');ax.plot(d.AvailableDate,d.CumulativeBaselineMSE,label='Historical mean YoY',color='#bd7841',linestyle='--');ax.set_title(label);ax.set_ylabel('Cumulative MSE (ratio squared)');ax.grid(alpha=.2);ax.legend()
    fig.suptitle('EPS YoY prediction | expanding OLS | 2018 warmup, 2019-2021 evaluation',fontsize=12)
    fig.savefig(R/'cumulative_mse.png',dpi=180);plt.close(fig)
    pd.concat(days,ignore_index=True).to_csv(R/'cumulative_error_both_scopes.csv',index=False)
    worst=v.sort_values('SquaredError',ascending=False).head(30)
    worst[['Code','AvailableDate','SourceRow','CurrentEPS','PreviousQuarterEPS','PreviousYearEPS','QoQ','YoY','PredictedYoY','HistoricalMeanPrediction','SquaredError','BaselineSquaredError','TrainingEligible']].to_csv(R/'largest_errors.csv',index=False)
    # Algebraic identity, not a fitted predictive benchmark: both rates share current EPS.
    row_intercept=(v.PreviousQuarterEPS-v.PreviousYearEPS)/v.PreviousYearEPS.abs()
    row_slope=v.PreviousQuarterEPS.abs()/v.PreviousYearEPS.abs()
    reconstructed=row_intercept+row_slope*v.QoQ
    np.testing.assert_allclose(reconstructed,v.YoY,atol=2e-11,rtol=2e-12)
    tail=dict(events=int((~v.TrainingEligible).sum()),fraction=float((~v.TrainingEligible).mean()),share_of_model_squared_error=float(v.loc[~v.TrainingEligible,'SquaredError'].sum()/v.SquaredError.sum()),algebra_identity_max_abs_error=float(np.max(np.abs(reconstructed-v.YoY))))
    save(R/'error_diagnostics.json',tail)
    rows=[]
    for name,k in [('全部可用事件（含極端值）','including_extremes'),('季增、年增均不超過100倍','primary')]:rows.append(dict(Scope=name,**rr[k]))
    pd.DataFrame(rows).to_csv(R/'comparison.csv',index=False,encoding='utf-8-sig')
    a=rr['last_coefficients']['a'];b=rr['last_coefficients']['B']
    lines=['# v19：用 EPS 季增預測年增','', '這個設計適合衡量「只使用季增的共同線性模型，能比歷史平均更準確地預測年增多少」。它不能直接衡量能取代多少投資資訊，也不是預先知道未公布財報。兩個比率都使用同一次公布的本季 EPS。','', '## 模型與時間順序','', 'Q = (本季 EPS - 上季 EPS) / |上季 EPS|','', 'Y = (本季 EPS - 去年同季 EPS) / |去年同季 EPS|','', '預測 Y = a + B Q；係數透過最小化過去資料的 mean[(Y - a - B Q)^2] 估計。使用全部合格歷史資料的 OLS 精確解，沒有 SGD 步長誤差、沒有股票價量 g 或報酬 Target。','', '每個財報可用日：先用更早日期的配對估計 a、B，對當天全體事件預測並記錄誤差，再將當天合格事件加入後續訓練。同一天其他公司年增也不會被偷用來預測當天。係數在當天預測後才有機會更新。','', '首個資料完整年2018作初始訓練，2019-2021分期作歷史未來批次評估。沿用原 validation calendar，起訖為2018-12-28至2021-12-01，實際有配對事件的日數為600；每筆財報事件等權，沒有每日複製財報，也没有指數衰減。第一次驗證預測已有5,432個合格歷史配對。','', '原始資料開始時缺乏去年同期基準，最初2018分期僅有6筆過去配對可供估計；因此沒有把該冷啟動期間當主要評估。所有較早的逐筆估計仍保留於 event_predictions.csv，IsValidation 欄標示正式評估範圍。','', '## 用什麼定義「季增增加多少解釋力」','', '對照模型只預測過去同一批合格樣本的平均年增，不使用季增。兩者以相同新事件比較：','', 'MSE改善率 = 1 - MSE(a + B × 季增) / MSE(歷史平均年增)。','', '正值代表降低未來批次誤差；0代表與對照相當；負值代表更差。這是相對歷史平均的樣本外 MSE 改善率，不應寫成「年增有多少百分比資料被取代」，也不能用訓練期 R² 代替。','', '## 整段結果','', '| 評估範圍 | 事件數 | 模型 MSE | 平均值對照 MSE | MSE改善率 |','|---|---:|---:|---:|---:|']
    for row in rows:lines.append(f'| {row["Scope"]} | {row["N"]:,} | {row["MSE"]:.6f} | {row["BaselineMSE"]:.6f} | {100*row["MSEImprovement"]:+.3f}% |')
    lines+=['', '全部事件的 MSE 比平均值對照高1.135%；兩個比率都不超過100倍時，MSE比對照低1.978%。因此這個共同線性模型在一般範圍只提供小幅改善，對極端情況的線性外推則較差。','', '門檻是原始比率100，即10,000%，不是100%。超門檻只跳過訓練；全部可用事件的主表仍保留它們。第二列是額外的條件式診斷，不能拿它代表完整樣本。也不會將值裁切成100。','', f'極端事件180筆，占全部評估的 {tail["fraction"]*100:.3f}%，卻占模型總平方誤差的 {tail["share_of_model_squared_error"]*100:.3f}%。因此完整 MSE 特別受這些案例支配。最大30筆誤差及原始EPS分母已輸出供檢查。','', '比率以1=100%輸入，MSE單位是比率平方。一般範圍 RMSE=5.800127，相當於580.013個百分點；絕對誤差中位數=0.643773，相當於64.377個百分點。相對改善約2%不代表絕對預測誤差已經很小。','', '## 到後期有沒有改善','', '| 分期 | 全部事件 MSE改善 | 100倍內 MSE改善 | 100倍內 RMSE |','|---|---:|---:|---:|']
    for year in [2019,2020,2021]:
        aa=yearly[(yearly.Scope=='including_extremes')&(yearly.ValidationYear==year)].iloc[0];bb=yearly[(yearly.Scope=='within_100')&(yearly.ValidationYear==year)].iloc[0]
        lines.append(f'| {year} | {aa.MSEImprovement*100:+.3f}% | {bb.MSEImprovement*100:+.3f}% | {bb.RMSE:.6f} |')
    lines+=['', '一般範圍各分期皆有小幅相對改善，但改善幅度從2.851%降到1.457%；2021的絕對MSE也較高。不能說資料增加後誤差就持續下降，因為新年度的目標分布和難度也可能改變。','', f'最後一個驗證事件使用的係數為 a={a:.8f}、B={b:.8f}，當時累積27,243筆訓練事件。這是期末估計，不是整段回測都使用這兩個數字。','', '## 一個會影響「重疊」解讀的數學關係','', '設 p=上季EPS、z=去年同季EPS。因為本季EPS = p + |p|Q，所以：','', 'Y = (p-z)/|z| + (|p|/|z|) Q。','', '只要兩個舊基準都不為零且基準一致，這是恆等式。我們逐筆核對通過。也就是說，若除了季增還保留上季與去年同季的EPS基準，年增可以精確還原，不需要學習。','', '本次 a、B 是所有公司共用、隨時間擴充估计的兩個係數，沒有提供每家公司的基準比值。因此低 MSE 改善不等於季增與年增彼此沒有重疊；它只表示「單靠季增＋共同線性係數」難以取代個別公司的年增差異。季節、產業、公司差異也未被本模型分開測試。','', '## 資料與限制','', '- 只使用正常財年、正常季度、會計基礎可辨認的首次實績事件；同公司／會計基礎／季度不重複計數。缺少本季或兩個基期、基期為零均不建立配對。','- 本次配對不要求公司曾發布Forecast，避免用預測有無篩選這個實際成長率問題。與v17共有的季增、年增數值逐值相同；因此样本數多於舊年增事件特徵的交集。','- 保留既有EPS累計值差分拆季近似；股數變動時不能視為嚴格的單季EPS，需另行修正驗證。本次結果只適用於現有資料定義。','- expanding window 避免模型使用未來批次，但2019-2021仍是以前反覆研究的歷史期間，不是新的保留測試集。','- 未訓練股價模型、未比較Sharpe、未進行季節調整，也未將残差加入g。正式價量基準維持v7。','', '## 交付資料','', '- comparison.csv、comparison_by_year.csv：整段與分期比較。','- event_predictions.csv：每筆QoQ、真實YoY、預測、係數、訓練數量、MSE。','- parameter_history.csv：逐批expanding係數與樣本數。','- cumulative_error_both_scopes.csv、cumulative_mse.png：到各時點的累積MSE與對照。','- largest_errors.csv：最大誤差案例與EPS原值。','- run.py → audit.py → report.py：重現流程。']
    (R/'JPX-v19-qoq-predict-yoy-report.md').write_text('\n'.join(lines)+'\n')
    version=dict(version='v19',description='Expanding OLS predicts raw EPS YoY from QoQ; intercept-only historical-mean comparison; 2018 warmup and 2019-2021 evaluation',directory=str(R),status='completed_experiment',audit_passed=True,financial_prediction_only=True,mse_improvement_all=rr['including_extremes']['MSEImprovement'],mse_improvement_within_100=rr['primary']['MSEImprovement'])
    save(R/'version.json',version);reg=json.loads((BASE/'jpx_model_versions.json').read_text());assert reg['active_version']=='v7';reg['versions']=[v for v in reg['versions'] if v.get('version')!='v19']+[version];reg['latest_experiment']='v19';save(BASE/'jpx_model_versions.json',reg)
    p=BASE/'JPX-current-baseline.md';text=p.read_text();title='## 最新完成的 v19 季增預測年增'
    if title not in text:p.write_text(text+f'\n{title}\n\n2018暖身後，2019-2021 expanding OLS：EPS年增 = a + B × 季增。完整21,993個事件 MSE相對歷史平均高1.135%；兩比率均不超過100倍的21,813事件 MSE低1.978%。此為財報間的預測實驗，未改股價模型，正式基準仍為v7。\n\n[完整 v19 報告]({R}/JPX-v19-qoq-predict-yoy-report.md)。\n')
    print(json.dumps(tail));print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
