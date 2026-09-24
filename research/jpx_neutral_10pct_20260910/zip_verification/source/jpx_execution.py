"""Stateful execution: zero-volume exits carry; occupied capital is not reused."""
import numpy as np
import pandas as pd

ORDER_COLUMNS=['SignalDate','PreviousSignalDate','EntryDate','ExitDate','ValidationYear',
               'SecuritiesCode','RawPrediction','PreviousRawPrediction','LagFallback','U',
               'Prediction','Target','Weight','SourceSide','SourceRank','Transferred']


def execute_hold(orders, bars, intervals, benchmark, close_only=False):
    """Orders are already ranked using prior information; no future reranking.

    Reserve gross marked notional of carried positions; divide remaining equity
    among today's original slots in their original proportions. Missing entries
    or already-held symbols leave their slot cash. A zero-volume day carries the
    last observed economic mark, solely for valuation, never as a fake fill.
    """
    orders=orders[ORDER_COLUMNS].copy()
    for c in ('SignalDate','PreviousSignalDate','EntryDate','ExitDate'):
        orders[c]=pd.to_datetime(orders[c])
    orders=orders.rename(columns={'ExitDate':'PlannedExitDate','Weight':'RequestedWeight'})
    bars=bars.copy(); bars['Date']=pd.to_datetime(bars.Date)
    lookup=bars.set_index(['Date','SecuritiesCode']).to_dict('index')
    by_entry={date:frame for date,frame in orders.groupby('EntryDate',sort=True)}
    timeline=intervals.copy().sort_values('EntryDate')
    for c in ('SignalDate','EntryDate','ExitDate'): timeline[c]=pd.to_datetime(timeline[c])
    benchmark=benchmark.copy(); benchmark.index=pd.to_datetime(benchmark.index)
    equity=100.; opened=[]; records=[]; daily=[]; events=[]
    next_id=0
    barfields=['Open','High','Low','Close','Volume','CumulativeFactor']
    def bar_at(date,code):
        bar=lookup.get((date,code))
        if bar is None:
            raise ValueError(f'Missing source row for held security {code} on {date}; absence is not proven zero volume')
        return bar
    for interval in timeline.to_dict('records'):
        entrydate,exitdate=interval['EntryDate'],interval['ExitDate']
        start_equity=equity
        reserved=sum(p['EntryNotional']*p['_mark'] for p in opened)
        available=max(0.,equity-reserved)
        held_codes={p['SecuritiesCode'] for p in opened}
        selected=by_entry.get(entrydate,orders.iloc[:0])
        new_executed=0
        for row in selected.to_dict('records'):
            next_id+=1
            code=row['SecuritiesCode']; bar=lookup.get((entrydate,code),{})
            executable=(np.isfinite(bar.get('Close',np.nan)) and bar.get('Close',0)>0 and bar.get('Volume',0)>0)
            overlap=code in held_codes
            notional=available*abs(row['RequestedWeight']) if executable and not overlap else 0.
            p={**row,'TradeId':next_id,'ExitDate':row['PlannedExitDate'],
               'EntryEquity':equity,'AvailableCapitalAtEntry':available,'ReservedCapitalAtEntry':reserved,
               'EntryNotional':notional,'Weight':np.sign(row['RequestedWeight'])*notional/equity,
               'ExecutedWeight':np.sign(row['RequestedWeight'])*notional/equity,
               'Executed':notional>0,'Touched':False,'DelaySessions':0,
               'UnderlyingReturn':0.,'CloseUnderlyingReturn':0.,'PositionReturn':0.,
               'PnlContribution':0.,'CloseOnlyContribution':0.,'RealizedPnl':0.,
               'ActionFactor':np.nan,'EntryPriceOnExitScale':np.nan,'TargetPriceOnExitScale':np.nan,
               'ExitPrice':np.nan,'EntrySinglePrice':bool(executable and bar.get('High')==bar.get('Low')),
               'ExitSinglePrice':False,'_mark':1.}
            p.update({'Entry'+c:bar.get(c,np.nan) for c in barfields})
            p.update({'Exit'+c:np.nan for c in barfields})
            p['LongExposure']=max(p['Weight'],0.);p['ShortExposure']=max(-p['Weight'],0.)
            if notional<=0:
                p['ExitReason']='held_symbol_cash' if overlap else ('entry_unavailable_cash' if not executable else 'no_available_capital_cash')
                records.append(p)
            else:
                new_executed+=1; held_codes.add(code); opened.append(p)
        long=sum(p['EntryNotional']*p['_mark'] for p in opened if p['Weight']>0)/equity
        short=sum(p['EntryNotional']*p['_mark'] for p in opened if p['Weight']<0)/equity
        assert long+short<=1+1e-12, 'Carried exposure exceeds equity; explicit margin policy required'
        pnl_today=0.; pending=[]; hit_count=0; carry_count=0; exit_count=0
        for p in opened:
            assert p['PlannedExitDate']<=exitdate
            bar=bar_at(exitdate,p['SecuritiesCode'])
            if bar['Volume']==0:
                p['DelaySessions']+=1;carry_count+=1;pending.append(p)
                events.append({'TradeId':p['TradeId'],'SecuritiesCode':p['SecuritiesCode'],
                    'IntervalEntryDate':entrydate,'ValuationDate':exitdate,'Event':'no_trade_carry',
                    'Pnl':0.,'Contribution':0.,'MarkRatio':p['_mark'],
                    'ReservedNotional':p['EntryNotional']*p['_mark']})
                continue
            if not (bar['Volume']>0 and all(np.isfinite(bar[c]) for c in barfields)):
                raise ValueError(f'Positive/unknown volume with incomplete held exit OHLC: {p["SecuritiesCode"]}, {exitdate}')
            factor=bar['CumulativeFactor']/p['EntryCumulativeFactor']
            entry_scale=p['EntryClose']*factor
            target=entry_scale*(1+p['Prediction'])
            touched=(not close_only) and bar['Low']<=target<=bar['High']
            fill=target if touched else bar['Close']
            mark=fill/entry_scale
            change=np.sign(p['Weight'])*p['EntryNotional']*(mark-p['_mark'])
            pnl_today+=change;hit_count+=int(touched);exit_count+=1
            p.update({'Exit'+c:bar[c] for c in barfields})
            p.update({'ExitDate':exitdate,'Touched':bool(touched),'ActionFactor':factor,
                      'EntryPriceOnExitScale':entry_scale,'TargetPriceOnExitScale':target,'ExitPrice':fill,
                      'UnderlyingReturn':mark-1,'CloseUnderlyingReturn':bar['Close']/entry_scale-1,
                      'PositionReturn':np.sign(p['Weight'])*(mark-1),
                      'PnlContribution':p['Weight']*(mark-1),
                      'CloseOnlyContribution':p['Weight']*(bar['Close']/entry_scale-1),
                      'RealizedPnl':np.sign(p['Weight'])*p['EntryNotional']*(mark-1),
                      'ExitSinglePrice':bar['High']==bar['Low'],
                      'ExitReason':'range_touch' if touched else 'close_no_touch'})
            events.append({'TradeId':p['TradeId'],'SecuritiesCode':p['SecuritiesCode'],
                'IntervalEntryDate':entrydate,'ValuationDate':exitdate,'Event':p['ExitReason'],
                'Pnl':change,'Contribution':change/start_equity,'MarkRatio':mark,'ReservedNotional':0.})
            records.append(p)
        opened=pending;equity+=pnl_today
        assert equity>0
        daily.append({**interval,'StrategyReturn':pnl_today/start_equity,
            'StrategyEquity':equity,'LongExposure':long,'ShortExposure':short,'CashWeight':1-long-short,
            'NetExposure':long-short,'SelectedPositions':len(selected),'ExecutedPositions':new_executed,
            'TouchedPositions':hit_count,'ExitedPositions':exit_count,
            'TransferredSlots':int(selected.Transferred.sum()),'CarriedPositionsAtExit':carry_count,
            'ReservedCapitalAtEntry':reserved,'AvailableCapitalAtEntry':available,
            'StartEquity':start_equity,'Pnl':pnl_today,
            'IndexReturn':benchmark.loc[exitdate]/benchmark.loc[entrydate]-1})
    if opened:
        raise ValueError('Positions remain unliquidated at dataset end; extend data or explicitly report open holdings')
    t=pd.DataFrame(records).drop(columns='_mark').sort_values('TradeId').reset_index(drop=True)
    d=pd.DataFrame(daily);d['IndexEquity']=100*(1+d.IndexReturn).cumprod()
    e=pd.DataFrame(events)
    np.testing.assert_allclose(d.StrategyEquity,100*(1+d.StrategyReturn).cumprod(),atol=1e-10)
    np.testing.assert_allclose(d.Pnl.sum(),t.RealizedPnl.sum(),atol=1e-10)
    return t,d,e


def execute_with_close_diagnostic(orders,bars,intervals,benchmark):
    t,d,e=execute_hold(orders,bars,intervals,benchmark)
    _,c,_=execute_hold(orders,bars,intervals,benchmark,close_only=True)
    d['CloseOnlySamePositionsReturn']=c.StrategyReturn.to_numpy()
    d['CloseOnlySamePositionsEquity']=c.StrategyEquity.to_numpy()
    return t,d,e
