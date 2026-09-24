"""Build a sample-universe cap-weighted price index from published share counts.
No 2021 stock-list capitalization is used. This is not an official market index.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from jpx_common import price_t_features, digest, json_write

SHARES='NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock'


def build_proxy(raw, financials, out, coverage_floor=.90):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    traded=(raw.Close.notna() & (raw.Volume>0)).groupby(raw.Date).any()
    cal=pd.DatetimeIndex(traded.index[traded]).sort_values()
    raw=raw.loc[raw.Date.isin(cal)].sort_values(['SecuritiesCode','Date']).copy()
    raw['PriorFactor']=raw.groupby('SecuritiesCode').AdjustmentFactor.transform(lambda a:a.cumprod().shift(1,fill_value=1.))
    raw['EconomicPrice']=raw.Close/raw.PriorFactor
    codes=sorted(raw.SecuritiesCode.unique())
    prices=raw.pivot(index='Date',columns='SecuritiesCode',values='EconomicPrice').reindex(cal)
    observed=prices.notna()
    # Stale marks value existing exposure only; they are never fabricated executions.
    prices=prices.ffill()
    factors=raw.pivot(index='Date',columns='SecuritiesCode',values='PriorFactor').reindex(cal).ffill().fillna(1.)
    f=financials.copy();f['Shares']=pd.to_numeric(f[SHARES],errors='coerce')
    for c in ['DisclosedDate','CurrentPeriodEndDate']:f[c]=pd.to_datetime(f[c],errors='coerce')
    f=f.loc[f.SecuritiesCode.isin(codes) & f.Shares.gt(0) & f.CurrentPeriodEndDate.notna() & f.DisclosedDate.notna() & (f.CurrentPeriodEndDate<=f.DisclosedDate)].copy()
    f['SecuritiesCode']=f.SecuritiesCode.astype(int)
    # Conservatively use every filing on the next actual stock session, even morning filings.
    idx=cal.searchsorted(f.DisclosedDate,side='right');f=f.loc[idx<len(cal)].copy();f['AvailableDate']=cal[idx[idx<len(cal)]]
    f=f.sort_values(['SecuritiesCode','DisclosedDate','DisclosedTime','DisclosureNumber'])
    events=[]
    for code,group in f.groupby('SecuritiesCode',sort=True):
        latest_period=pd.Timestamp.min
        for row in group.itertuples():
            if row.CurrentPeriodEndDate<latest_period:continue
            latest_period=row.CurrentPeriodEndDate
            k=cal.searchsorted(row.CurrentPeriodEndDate,side='right')-1
            factor=float(factors.at[cal[k],code]) if k>=0 else 1.
            # Count refers to period-end share units. Future split changes cancel
            # between economic share units and the causally adjusted price.
            events.append({'SecuritiesCode':code,'AvailableDate':row.AvailableDate,
                'DisclosedDate':row.DisclosedDate,'DisclosedTime':row.DisclosedTime,
                'PeriodEnd':row.CurrentPeriodEndDate,'DisclosureNumber':row.DisclosureNumber,
                'ReportedShares':row.Shares,'PeriodEndFactor':factor,'EconomicShares':row.Shares*factor})
    e=pd.DataFrame(events).drop_duplicates(['SecuritiesCode','AvailableDate'],keep='last')
    assert (e.AvailableDate>e.DisclosedDate).all()
    shares=e.pivot(index='AvailableDate',columns='SecuritiesCode',values='EconomicShares').reindex(index=cal,columns=codes).ffill()
    caps=prices*shares
    eligible=caps.notna() & caps.gt(0)
    coverage=eligible.sum(axis=1)/prices.notna().sum(axis=1)
    ready=coverage.ge(coverage_floor)
    if not ready.any():raise ValueError('No date meets the prespecified share-count coverage floor')
    start=ready.index[ready][0]
    weights=caps.div(caps.sum(axis=1),axis=0).fillna(0.)
    returns=prices.div(prices.shift(1)).sub(1)
    # New stocks only enter next day's weights; corporate share updates do not
    # themselves generate index return or a capital-injection jump.
    prev_w=weights.shift(1).fillna(0.)
    invalid=(prev_w.gt(0)&~np.isfinite(returns)).any(axis=1)
    if invalid.loc[start:].any():raise ValueError('A held index weight lacks a valid marked return')
    market_return=(prev_w*returns.fillna(0.)).sum(axis=1)
    index=(1+market_return.loc[start:].iloc[1:]).cumprod()*100
    index=pd.concat([pd.Series([100.],index=[start]),index])
    result=pd.DataFrame({'Close':index,'MarketReturn':market_return.reindex(index.index),
        'CoveredStocks':eligible.sum(axis=1).reindex(index.index),
        'ObservedUniverseStocks':prices.notna().sum(axis=1).reindex(index.index),
        'ShareCoverage':coverage.reindex(index.index),
        'StaleMarkedWeight':weights.where(~observed,0.).sum(axis=1).reindex(index.index),
        'LargestStockWeight':weights.max(axis=1).reindex(index.index)})
    result.loc[start,'MarketReturn']=np.nan
    t,_=price_t_features(result.Close);result=result.join(t);result.index.name='Date'
    result.to_csv(out/'market_proxy_daily.csv');e.to_csv(out/'share_disclosure_events.csv',index=False)
    pd.DataFrame({'Date':cal,'Coverage':coverage,'CoveredStocks':eligible.sum(axis=1),'PricedStocks':prices.notna().sum(axis=1)}).to_csv(out/'coverage_all_dates.csv',index=False)
    # Compact daily audit sample; full weights reconstruct from raw files + this module.
    top=[]
    for date in result.index:
        for code,w in weights.loc[date].nlargest(10).items():top.append({'Date':date,'SecuritiesCode':code,'Weight':w,'EstimatedMarketCap':caps.at[date,code]})
    pd.DataFrame(top).to_csv(out/'top10_daily_weights.csv',index=False)
    audit={'source':'user stock_prices and financials; stock_list snapshot NOT used',
        'universe':'fixed provided price-file universe; historical listing survivorship/selection bias remains',
        'weight':'previous session economic price * latest published economic issued shares, normalized each day',
        'shares':'issued incl treasury, not free float; last published, not actual daily share register',
        'filing_timing':'usable next actual stock session; weights lag one additional session for return',
        'share_units':'period-end count times cumulative adjustment factor strictly before period-end',
        'coverage_floor':coverage_floor,'coverage_floor_unit':'stock count, not total-market capitalization',
        'start':str(start.date()),'end':str(result.index[-1].date()),'rows':len(result),
        'valid_T_rows':int(result[['T5','T22','T60']].notna().all(axis=1).sum()),
        'share_disclosure_events':len(e),'represented_codes':int(e.SecuritiesCode.nunique()),
        'start_coverage':float(result.ShareCoverage.iloc[0]),'min_coverage_after_start':float(result.ShareCoverage.min()),
        'last_coverage':float(result.ShareCoverage.iloc[-1]),'max_stale_marked_weight':float(result.StaleMarkedWeight.max()),
        'max_single_stock_weight':float(result.LargestStockWeight.max()),
        'weight_sum_max_error':float((weights.loc[start:].sum(axis=1)-1).abs().max()),
        'market_return_min':float(result.MarketReturn.min()),'market_return_max':float(result.MarketReturn.max()),
        'no_backfilled_2021_market_caps':True,'not_official_index':True,'share_restating_caveat':'financial statement share counts may require issuer-level validation when restated around corporate actions',
        'market_price_return_excludes_dividends':True}
    json_write(out/'proxy_audit.json',audit)
    print(json.dumps(audit),flush=True)
    return result


if __name__=='__main__':
    raw=pd.read_csv('jpx_inputs/stock_prices.csv',parse_dates=['Date'])
    f=pd.read_csv('jpx_inputs/financials.csv',low_memory=False)
    build_proxy(raw,f,'jpx_market_proxy_results')
