from pathlib import Path
import json
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent;B=R.parent
def main():
    raw=pd.read_pickle(B/'jpx_v14_filtered_forecast_events_20260914/financial_records.pkl').set_index('SourceRow').to_dict('index')
    pairs=pd.read_pickle(R/'all_actual_events.pkl');pred=pd.read_pickle(R/'predictions.pkl');checks=0
    assert not pairs.duplicated(['Code','Basis','FiscalStart','FiscalEnd','FiscalQuarter']).any()
    for e in pairs[pairs.ValidPair].itertuples():
        for field,refs,q,year in [('CurrentEPS',e.CurrentRefs,e.FiscalQuarter,0),('PreviousQuarterEPS',e.PreviousRefs,e.FiscalQuarter-1 if e.FiscalQuarter>1 else 4,0 if e.FiscalQuarter>1 else -1),('PreviousYearEPS',e.YearRefs,e.FiscalQuarter,-1)]:
            source=[raw[k] for k in refs];assert len(source)>0
            assert all(r['Code']==e.Code and r['Basis']==e.Basis for r in source)
            assert all(r['EffectiveTimestamp']<=e.EffectiveTimestamp for r in source)
            fs=e.FiscalStart+pd.DateOffset(years=year)
            assert all(r['CurrentFiscalYearStartDate']==fs for r in source)
            vals={dict(zip(['1Q','2Q','3Q','FY'],[1,2,3,4]))[r['TypeOfCurrentPeriod']]:r['EarningsPerShare'] for r in source}
            value=vals[q]-(vals[q-1] if q>1 else 0)
            np.testing.assert_allclose(value,getattr(e,field),atol=0,rtol=0);checks+=1
        np.testing.assert_allclose([e.QoQ,e.YoY],[(e.CurrentEPS-e.PreviousQuarterEPS)/abs(e.PreviousQuarterEPS),(e.CurrentEPS-e.PreviousYearEPS)/abs(e.PreviousYearEPS)],rtol=0,atol=0)
    valid=pairs[pairs.ValidPair]
    oldq=pd.read_pickle(B/'jpx_v17_actual_forecast_revisions_20260915/new_financial_events.pkl');oldq=oldq[oldq.Feature.eq('EPSActualQoQ')]
    merged=valid.merge(oldq[['SourceRow','Value']],on='SourceRow',validate='one_to_one');np.testing.assert_array_equal(merged.QoQ,merged.Value)
    oldy=pd.read_pickle(B/'jpx_v14_filtered_forecast_events_20260914/financial_events.pkl');oldy=oldy[oldy.Kind.eq('A')&oldy.Metric.eq(2)]
    merged_y=valid.merge(oldy[['TriggerSourceRow','Values']],left_on='SourceRow',right_on='TriggerSourceRow',validate='one_to_one');np.testing.assert_array_equal(merged_y.YoY,[v[0] for v in merged_y.Values])
    # Independent replay of all forecast batches against strictly earlier events.
    for day,g in pred.groupby('AvailableDate'):
        train=valid[valid.TrainingEligible & valid.AvailableDate.lt(day)]
        X=np.column_stack([np.ones(len(train)),train.QoQ]);coef=np.linalg.lstsq(X,train.YoY,rcond=None)[0]
        assert g.TrainingN.eq(len(train)).all() and g.LastTrainingDate.eq(train.AvailableDate.max()).all()
        np.testing.assert_allclose(g.PredictedYoY,coef[0]+coef[1]*g.QoQ,atol=2e-10,rtol=1e-9)
        np.testing.assert_allclose(g.HistoricalMeanPrediction,train.YoY.mean(),atol=1e-13)
        np.testing.assert_allclose(g.SquaredError,(g.PredictedYoY-g.YoY)**2,atol=0,rtol=0)
    result=json.loads((R/'results.json').read_text());v=pred[pred.IsValidation]
    for label,df in [('primary',v[v.TrainingEligible]),('including_extremes',v)]:
        assert len(df)==result[label]['N'];np.testing.assert_allclose(result[label]['MSE'],np.mean((df.PredictedYoY-df.YoY)**2),rtol=1e-14)
    calendar=pd.read_csv(B/'jpx_v16_single_financial_v7_20260915/price_only/daily_metrics.csv')
    dates=pd.to_datetime(calendar.loc[calendar.ValidationYear.ge(2019),'Date'])
    assert len(v)==len(valid[valid.AvailableDate.isin(dates)])
    audit=dict(passed=True,source_quarter_value_checks=checks,growth_pairs_checked=len(valid),previous_qoq_values_reproduced=len(merged),previous_yoy_values_reproduced=len(merged_y),predictions_checked=len(pred),expanding_batches_checked=int(pred.AvailableDate.nunique()),all_training_dates_strictly_before_prediction=True)
    (R/'audit.json').write_text(json.dumps(audit,indent=2));print(audit)
if __name__=='__main__':main()
