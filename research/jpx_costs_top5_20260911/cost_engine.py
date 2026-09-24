import numpy as np
import pandas as pd

def replay_costs(templates,timeline,rates,save_ledger=True):
    """Replay frozen gross executions with path-dependent net equity and costs.

    This is a cost sensitivity overlay, not a quote/order-book fill simulator.
    Unchanged gross executions are inputs; no future outcomes drive selections.
    """
    cr,sr,lr=[rates[k]/10000 for k in ('commission_bp','spread_bp','slippage_bp')]
    br=rates['borrow_annual'];c=cr+sr+lr
    by_entry={date:g.to_dict('records') for date,g in templates.groupby('EntryDate',sort=True)}
    equity=100.;opened=[];ledger=[];daily=[]
    for interval in timeline.to_dict('records'):
        entry,exit=interval['EntryDate'],interval['ExitDate'];start=equity
        reserved=sum(p['NetEntryNotional'] for p in opened)
        available=equity-reserved
        assert available>=-1e-10
        rows=by_entry.get(entry,[])
        wsum=sum(abs(row['RequestedWeight']) for row in rows if row['Executed'])
        base=max(0.,available)/(1+c*wsum)
        entry_value=base*wsum;entry_cost=entry_value*c
        components={'Commission':entry_value*cr,'Spread':entry_value*sr,'Slippage':entry_value*lr,'Borrow':0.}
        newcount=0
        for row in rows:
            n=base*abs(row['RequestedWeight']) if row['Executed'] else 0.
            if n==0:continue
            p={**row,'NetEntryNotional':n,'NetEntryEquity':start,'NetEntryFee':n*c,
               'NetCommission':n*cr,'NetSpread':n*sr,'NetSlippage':n*lr,'NetBorrow':0.}
            opened.append(p);newcount+=1
        total=sum(p['NetEntryNotional'] for p in opened)
        assert total<=equity-entry_cost+1e-10
        long=sum(p['NetEntryNotional'] for p in opened if p['RequestedWeight']>0)/start
        short=sum(p['NetEntryNotional'] for p in opened if p['RequestedWeight']<0)/start
        pnl=0.;exit_value=0.;remaining=[]
        for p in opened:
            end=min(exit,p['ExitDate'])
            days=(end-max(entry,p['EntryDate'])).days
            assert days>=0
            borrowing=p['NetEntryNotional']*br*days/365 if p['RequestedWeight']<0 else 0.
            p['NetBorrow']+=borrowing;components['Borrow']+=borrowing
            if p['ExitDate']>exit:
                remaining.append(p);continue
            n=p['NetEntryNotional'];mark=1+p['UnderlyingReturn'];v=n*mark
            gross=n*np.sign(p['RequestedWeight'])*(mark-1)
            pnl+=gross;exit_value+=v
            for key,rate in [('Commission',cr),('Spread',sr),('Slippage',lr)]:
                components[key]+=v*rate;p['Net'+key]+=v*rate
            p.update(NetExitNotional=v,NetGrossPnl=gross,NetExitFee=v*c,
                NetTotalCost=sum(p['Net'+k] for k in components),NetHoldingCalendarDays=(p['ExitDate']-p['EntryDate']).days)
            p['NetPnlAfterCosts']=gross-p['NetTotalCost']
            if save_ledger:ledger.append(p)
        opened=remaining
        costs=sum(components.values());equity+=pnl-costs
        assert equity>0,'Account depleted in this scenario'
        daily.append({**interval,'StartNetEquity':start,'NetEquity':equity,'NetReturn':(pnl-costs)/start,
            'GrossPnlOnNetCapital':pnl,'TotalCost':costs,**components,
            'EntryTurnover':entry_value/start,'ExitTurnover':exit_value/start,
            'EntryValue':entry_value,'ExitValue':exit_value,'LongExposure':long,'ShortExposure':short,
            'ReservedCapital':reserved,'NewPositions':newcount,'CarriedPositionsAtExit':len(opened)})
    assert not opened
    d=pd.DataFrame(daily);t=pd.DataFrame(ledger)
    np.testing.assert_allclose(d.NetEquity,100*(1+d.NetReturn).cumprod(),atol=1e-10,rtol=1e-12)
    if save_ledger:
        np.testing.assert_allclose(t.NetPnlAfterCosts.sum(),equity-100,atol=1e-10,rtol=0)
        for k in components:np.testing.assert_allclose(t['Net'+k].sum(),d[k].sum(),atol=1e-10,rtol=0)
    return d,t
