from pathlib import Path
import importlib.util,json,pickle,hashlib
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent;BASE=R.parent;V14=BASE/'jpx_v14_filtered_forecast_events_20260914'
s=importlib.util.spec_from_file_location('financial_builder',V14/'features.py');old=importlib.util.module_from_spec(s);s.loader.exec_module(old)
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))
class Builder(old.Builder):
    def __init__(self):super().__init__();self.pairs=[]
    def emit(self,row,key,m,kind,values,refs,detail):
        if m!=2 or kind!='A':return
        current,cr=self.quarter(key);previous,pr=self.quarter(old.previous_quarter(key));year,yr=self.quarter(old.shift_year(key))
        self.pairs.append(dict(Code=key[0],Basis=key[1],FiscalStart=key[2],FiscalEnd=key[3],FiscalQuarter=key[4],SourceRow=row.SourceRow,EffectiveTimestamp=row.EffectiveTimestamp,AvailableDate=row.AvailableDate,CurrentEPS=float(current[2]),PreviousQuarterEPS=float(previous[2]),PreviousYearEPS=float(year[2]),QoQ=old.grow(current[2],previous[2]),YoY=old.grow(current[2],year[2]),CurrentRefs=cr[2],PreviousRefs=pr[2],YearRefs=yr[2]))
def main():
    records=pd.read_pickle(V14/'financial_records.pkl');builder=Builder();builder.run(records)
    pairs=pd.DataFrame(builder.pairs).sort_values(['AvailableDate','EffectiveTimestamp','Code','SourceRow']).reset_index(drop=True)
    pairs['ValidPair']=np.isfinite(pairs[['QoQ','YoY']]).all(axis=1)
    pairs['TrainingEligible']=pairs.ValidPair & pairs.QoQ.abs().le(100)&pairs.YoY.abs().le(100)
    pairs.to_pickle(R/'all_actual_events.pkl');pairs.to_csv(R/'all_actual_events.csv',index=False)
    with (BASE/'jpx_v8_soft_rank_20260912/inputs.pkl').open('rb') as h:_,_,_,calendar=pickle.load(h)
    # The first year with paired annual observations is the warmup period.
    calendar=calendar[calendar.ValidationYear.ge(2019)].copy()
    start=calendar.Date.min();end=calendar.Date.max();yearmap=calendar.set_index('Date').ValidationYear.to_dict()
    events=pairs[pairs.ValidPair & pairs.AvailableDate.le(end)].copy()
    # Complete past-only OLS, fitted before each availability-day batch.
    prior=[];outputs=[];history=[];checks=0
    for day,batch in events.groupby('AvailableDate',sort=True):
        train=pd.concat(prior,ignore_index=True) if prior else events.iloc[:0]
        n=len(train);estimable=n>=2 and train.QoQ.var(ddof=0)>0
        a=b=mean=None
        if estimable:
            q=train.QoQ.to_numpy();y=train.YoY.to_numpy();mean=float(y.mean())
            b=float((q-q.mean())@(y-y.mean())/((q-q.mean())@(q-q.mean())));a=float(mean-b*q.mean())
            assert train.AvailableDate.max()<day
            independent=np.linalg.lstsq(np.column_stack([np.ones(n),q]),y,rcond=None)[0]
            np.testing.assert_allclose([a,b],independent,atol=2e-12,rtol=2e-10);checks+=1
            out=batch.copy();out['TrainingN']=n;out['LastTrainingDate']=train.AvailableDate.max();out['Intercept']=a;out['Slope']=b
            out['PredictedYoY']=a+b*out.QoQ;out['HistoricalMeanPrediction']=mean
            out['Error']=out.PredictedYoY-out.YoY;out['SquaredError']=out.Error**2
            out['BaselineSquaredError']=(mean-out.YoY)**2
            out['ValidationYear']=out.AvailableDate.map(yearmap);out['IsValidation']=out.AvailableDate.isin(calendar.Date)
            outputs.append(out)
        history.append(dict(Date=str(day.date()),TrainingN=n,LastTrainingDate=str(train.AvailableDate.max().date()) if n else None,Intercept=a,Slope=b,HistoricalMeanYoY=mean,NewValidPairs=len(batch),NewEligiblePairs=int(batch.TrainingEligible.sum()),Estimable=bool(estimable)))
        prior.append(batch[batch.TrainingEligible])
    pred=pd.concat(outputs,ignore_index=True);pred.to_pickle(R/'predictions.pkl');pred.to_csv(R/'event_predictions.csv',index=False)
    pd.DataFrame(history).to_csv(R/'parameter_history.csv',index=False)
    valid=pred[pred.IsValidation];assert valid.AvailableDate.min()>=start
    primary=valid[valid.TrainingEligible]
    def stats(v):
        mse=float(v.SquaredError.mean());base=float(v.BaselineSquaredError.mean())
        return dict(N=len(v),Dates=int(v.AvailableDate.nunique()),MSE=mse,RMSE=float(np.sqrt(mse)),BaselineMSE=base,BaselineRMSE=float(np.sqrt(base)),MSEImprovement=1-mse/base,MAE=float(v.Error.abs().mean()),MedianAbsoluteError=float(v.Error.abs().median()))
    result=dict(version='v19',target='EPS actual YoY',predictor='EPS actual QoQ',formula='(current-old)/abs(old)',raw_events_no_decay=True,training='past availability days expanding OLS; joint abs<=100 mask; no forecast requirement',validation_start=str(start.date()),validation_end=str(end.date()),all_actual_events=len(pairs),valid_pairs=int(pairs.ValidPair.sum()),invalid_pairs=int((~pairs.ValidPair).sum()),excluded_from_training=int((pairs.ValidPair&~pairs.TrainingEligible).sum()),primary=stats(primary),including_extremes=stats(valid),validation_extreme_events=int((~valid.TrainingEligible).sum()),ols_independent_fit_checks=checks)
    yearly=[]
    for scope,v in [('within_100',primary),('including_extremes',valid)]:
        for year,g in v.groupby('ValidationYear'):yearly.append(dict(Scope=scope,ValidationYear=int(year),**stats(g)))
    pd.DataFrame(yearly).to_csv(R/'comparison_by_year.csv',index=False)
    daily=primary.groupby('AvailableDate').agg(N=('SquaredError','size'),SSE=('SquaredError','sum'),BaselineSSE=('BaselineSquaredError','sum')).reset_index()
    daily['MSE']=daily.SSE/daily.N;daily['BaselineMSE']=daily.BaselineSSE/daily.N
    daily['CumulativeN']=daily.N.cumsum();daily['CumulativeMSE']=daily.SSE.cumsum()/daily.CumulativeN
    daily['CumulativeBaselineMSE']=daily.BaselineSSE.cumsum()/daily.CumulativeN
    daily['CumulativeMSEImprovement']=1-daily.CumulativeMSE/daily.CumulativeBaselineMSE
    daily.to_csv(R/'daily_and_cumulative_error.csv',index=False)
    result['first_validation_training_n']=int(primary.TrainingN.iloc[0]);result['last_validation_training_n']=int(primary.TrainingN.iloc[-1]);result['last_coefficients']={k:float(primary.iloc[-1][v]) for k,v in [('a','Intercept'),('B','Slope')]}
    save(R/'results.json',result)
    save(R/'manifest.json',dict(source_sha256={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [V14/'financial_records.pkl',V14/'features.py',R/'run.py',BASE/'jpx_v8_soft_rank_20260912/inputs.pkl']}))
    print(json.dumps(result,ensure_ascii=False,indent=2));print(pd.DataFrame(yearly).to_string(index=False))
if __name__=='__main__':main()
