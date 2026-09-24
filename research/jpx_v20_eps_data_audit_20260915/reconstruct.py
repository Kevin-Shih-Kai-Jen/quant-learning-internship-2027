from pathlib import Path
import json
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent;B=R.parent
def num(x):return pd.to_numeric(x,errors='coerce')
def growth(a,b):return (a-b)/abs(b) if np.isfinite(a) and np.isfinite(b) and b!=0 else np.nan
def main():
    raw=pd.read_pickle(B/'jpx_v14_filtered_forecast_events_20260914/financial_records.pkl').set_index('SourceRow').to_dict('index')
    selected=pd.read_pickle(R/'selected_cases.pkl');quarters=[];cases=[]
    for e in selected.itertuples():
        vals={}
        for label,refs,value in [('Current',e.CurrentRefs,e.CurrentEPS),('Previous',e.PreviousRefs,e.PreviousQuarterEPS),('Year',e.YearRefs,e.PreviousYearEPS)]:
            records=sorted([dict(raw[k],SourceRow=k) for k in refs],key=lambda r:r['CurrentPeriodEndDate']);cur=records[-1];pre=records[-2] if len(records)>1 else None
            assert len(records) in [1,2]
            eps=float(cur['EarningsPerShare']);shares=float(num(cur['AverageNumberOfShares']));profit=float(num(cur['Profit']))
            dc=(cur['CurrentPeriodEndDate']-cur['CurrentFiscalYearStartDate']).days+1
            if pre is None:
                prev_eps=prev_s=prev_p=dp=0.;sq=shares;naive=eps;reweighted=eps;rounderr=.005
            else:
                prev_eps=float(pre['EarningsPerShare']);prev_s=float(num(pre['AverageNumberOfShares']));prev_p=float(num(pre['Profit']))
                dp=(pre['CurrentPeriodEndDate']-pre['CurrentFiscalYearStartDate']).days+1
                sq=(dc*shares-dp*prev_s)/(dc-dp)
                assert sq>0 and dc>dp
                naive=eps-prev_eps
                reweighted=(eps*shares-prev_eps*prev_s)/sq
                rounderr=.005*(shares+prev_s)/sq
            np.testing.assert_allclose(naive,value,atol=0,rtol=0)
            direct=(profit-prev_p)/sq
            item=dict(CaseID=e.CaseID,SampleType=e.SampleType,Role=label,Code=e.Code,Quarter=cur['TypeOfCurrentPeriod'],SourceRows=','.join(map(str,refs)),CumulativeEPS=eps,PriorCumulativeEPS=prev_eps,CumulativeAverageShares=shares,PriorCumulativeAverageShares=prev_s,CumulativeDays=dc,PriorCumulativeDays=dp,ImpliedQuarterAverageShares=sq,CumulativeProfit=profit,PriorCumulativeProfit=prev_p,NaiveQuarterEPS=naive,ShareAdjustedEPSSensitivity=reweighted,ShareAdjustedEPSMinusNaive=reweighted-naive,ShareAdjustedEPSLower=reweighted-rounderr,ShareAdjustedEPSUpper=reweighted+rounderr,ProfitBasedQuarterEPSSensitivity=direct,ReportedProfitZeroButEPSNonzero=bool(profit==0 and eps!=0),YtdShareChangeRatio=(shares-prev_s)/prev_s if prev_s else 0.,NaiveRoundingHalfWidth=.01 if pre is not None else .005)
            quarters.append(item);vals[label]=item
        q=growth(vals['Current']['ShareAdjustedEPSSensitivity'],vals['Previous']['ShareAdjustedEPSSensitivity']);y=growth(vals['Current']['ShareAdjustedEPSSensitivity'],vals['Year']['ShareAdjustedEPSSensitivity'])
        qc=growth(vals['Current']['ProfitBasedQuarterEPSSensitivity'],vals['Previous']['ProfitBasedQuarterEPSSensitivity']);yc=growth(vals['Current']['ProfitBasedQuarterEPSSensitivity'],vals['Year']['ProfitBasedQuarterEPSSensitivity'])
        cases.append(dict(CaseID=e.CaseID,SampleType=e.SampleType,Code=e.Code,AvailableDate=str(e.AvailableDate.date()),SourceRow=e.SourceRow,OriginalQoQ=e.QoQ,ShareAdjustedQoQSensitivity=q,QoQChangePercentagePoints=(q-e.QoQ)*100,OriginalYoY=e.YoY,ShareAdjustedYoYSensitivity=y,YoYChangePercentagePoints=(y-e.YoY)*100,OriginalYearBaseEPS=e.PreviousYearEPS,ShareAdjustedYearBaseEPS=vals['Year']['ShareAdjustedEPSSensitivity'],YearBaseSensitivityLower=vals['Year']['ShareAdjustedEPSLower'],YearBaseSensitivityUpper=vals['Year']['ShareAdjustedEPSUpper'],ProfitBasedQoQSensitivity=qc,ProfitBasedYoYSensitivity=yc,OriginalOver100=bool(max(abs(e.QoQ),abs(e.YoY))>100),ShareAdjustedOver100=bool(max(abs(q),abs(y))>100),YearBaseFromCumulativeSubtraction=bool(len(e.YearRefs)>1),AnyYtdAverageSharesChangeAbove1Pct=any(abs(v['YtdShareChangeRatio'])>.01 for v in vals.values()),MaximumQuarterEPSChange=max(abs(v['ShareAdjustedEPSMinusNaive']) for v in vals.values())))
    d=pd.DataFrame(cases);qd=pd.DataFrame(quarters);d.to_csv(R/'eps_sensitivity_by_case.csv',index=False);qd.to_csv(R/'quarter_reconstruction_details.csv',index=False)
    summary=dict(selected_cases=len(d),ordinary_cases=int(d.SampleType.eq('ordinary').sum()),extreme_cases=int(d.SampleType.eq('extreme').sum()),quarter_checks=len(qd),naive_values_reproduced_exactly=True,all_extreme_year_base_abs_le_009=bool(d[d.SampleType.eq('extreme')].OriginalYearBaseEPS.abs().le(.090000001).all()),extreme_year_bases_direct_1q=int((d.SampleType.eq('extreme')&~d.YearBaseFromCumulativeSubtraction).sum()),extreme_year_bases_differenced=int((d.SampleType.eq('extreme')&d.YearBaseFromCumulativeSubtraction).sum()),over100_classification_sensitivity_changes=int(d.OriginalOver100.ne(d.ShareAdjustedOver100).sum()),ordinary_growth_sign_changes=int(((np.sign(d.OriginalQoQ)!=np.sign(d.ShareAdjustedQoQSensitivity))|(np.sign(d.OriginalYoY)!=np.sign(d.ShareAdjustedYoYSensitivity)))[d.SampleType.eq('ordinary')].sum()),zero_reported_profit_nonzero_eps_quarters=int(qd.ReportedProfitZeroButEPSNonzero.sum()),status='Sensitivity calculation only, not an official corrected EPS feed',method='Day-weighted reconstruction of quarterly average shares, EPS-implied numerator and separately rounded Profit; no retraining or cache mutation')
    (R/'reconstruction_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(d[['CaseID','Code','OriginalYearBaseEPS','ShareAdjustedYearBaseEPS','OriginalYoY','ShareAdjustedYoYSensitivity','OriginalQoQ','ShareAdjustedQoQSensitivity','MaximumQuarterEPSChange']].to_string(index=False));print(json.dumps(summary))
if __name__=='__main__':main()
