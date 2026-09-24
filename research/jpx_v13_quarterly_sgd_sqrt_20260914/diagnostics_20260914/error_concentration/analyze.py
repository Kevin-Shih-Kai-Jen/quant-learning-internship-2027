from pathlib import Path
import json,pickle,gc
import numpy as np
import pandas as pd
OUT=Path(__file__).resolve().parent
ROOT=OUT.parent.parent
with (ROOT.parent/'jpx_v8_soft_rank_20260912/inputs.pkl').open('rb') as h:data=pickle.load(h)
labels=data[0][['SignalDate','SecuritiesCode','Target']].rename(columns={'SignalDate':'Date'}).copy();del data;gc.collect()
results={};parts={}
for variant in ['sgd_only','sgd_sqrt']:
    values=[]
    for year in [2018,2019,2020,2021]:
        r=pd.read_csv(ROOT/variant/f'ranks_{year}.csv.gz',parse_dates=['Date'],usecols=['Date','SecuritiesCode','g','FinancialContribution','Rank'])
        r=r.merge(labels,on=['Date','SecuritiesCode'],validate='one_to_one');r=r.loc[r.Target.notna()].copy()
        r['Error']=(r.g-r.Target)**2;r['AbsError']=abs(r.g-r.Target)
        r['ZeroError']=r.Target**2;r['PriceComponentError']=(r.g-r.FinancialContribution-r.Target)**2
        values.append(r)
    r=pd.concat(values,ignore_index=True);del values;gc.collect()
    n=r.groupby('Date').Target.transform('count');days=r.Date.nunique()
    r['Contribution']=r.Error/n/days
    daily=r.groupby('Date').agg(MSE=('Error','mean'),MAE=('AbsError','mean'),ZeroMSE=('ZeroError','mean'),PriceComponentMSE=('PriceComponentError','mean'),ForecastStd=('g','std'),TargetStd=('Target','std'))
    mse=float(daily.MSE.mean());ranked=r.sort_values('Contribution',ascending=False)
    bystock=r.groupby('SecuritiesCode').Contribution.sum().sort_values(ascending=False)
    bydate=r.groupby('Date').Contribution.sum().sort_values(ascending=False)
    report={'days':int(days),'finite_target_rows':len(r),'daily_mean_MSE':mse,'zero_prediction_daily_mean_MSE':float(daily.ZeroMSE.mean()),
        'MSE_divided_by_zero_baseline':float(mse/daily.ZeroMSE.mean()),'mean_daily_MAE':float(daily.MAE.mean()),
        'fixed_fitted_price_component_MSE_not_refit':float(daily.PriceComponentMSE.mean()),
        'median_abs_error':float(r.AbsError.median()),'prediction_abs_quantiles':{str(q):float(r.g.abs().quantile(q)) for q in [.5,.9,.99,.999,1]},
        'target_abs_quantiles':{str(q):float(r.Target.abs().quantile(q)) for q in [.5,.9,.99,.999,1]},
        'abs_prediction_gt_100pct_rows':int(r.g.abs().gt(1).sum()),'abs_prediction_gt_20pct_rows':int(r.g.abs().gt(.2).sum()),
        'top_stock_error_shares':{str(k):float(v/mse) for k,v in bystock.head(10).items()},
        'top_5_stocks_error_share':float(bystock.head(5).sum()/mse),'top_10_dates_error_share':float(bydate.head(10).sum()/mse),
        'top_100_stock_days_error_share':float(ranked.head(100).Contribution.sum()/mse),
        'top_01pct_stock_days_error_share':float(ranked.head(int(np.ceil(len(r)*.001))).Contribution.sum()/mse),
        'median_daily_prediction_std':float(daily.ForecastStd.median()),'median_daily_target_std':float(daily.TargetStd.median())}
    bystock.rename('DailyMeanMSEContribution').to_csv(OUT/f'{variant}_error_by_stock.csv')
    bydate.rename('DailyMeanMSEContribution').to_csv(OUT/f'{variant}_error_by_date.csv')
    ranked.head(100).to_csv(OUT/f'{variant}_largest_errors.csv',index=False)
    daily.to_csv(OUT/f'{variant}_daily_error.csv');results[variant]=report
    parts[variant]=r[['Date','SecuritiesCode','Contribution']].rename(columns={'Contribution':variant}).copy()
    del r,ranked;gc.collect()
diff=parts['sgd_only'].merge(parts['sgd_sqrt'],on=['Date','SecuritiesCode'],validate='one_to_one')
diff['MSEImprovement']=diff.sgd_only-diff.sgd_sqrt
stock=diff.groupby('SecuritiesCode').MSEImprovement.sum().sort_values(ascending=False)
gain=float(stock.sum());results['difference']={'total_mse_improvement':gain,'top_5_stocks_share_of_net_improvement':float(stock.head(5).sum()/gain),'top_10_stocks':{str(k):float(v/gain) for k,v in stock.head(10).items()}}
stock.to_csv(OUT/'mse_improvement_by_stock.csv')
(OUT/'summary.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
print(json.dumps(results,ensure_ascii=False),flush=True)
