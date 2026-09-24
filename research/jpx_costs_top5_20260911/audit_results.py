from pathlib import Path
import json
import numpy as np
import pandas as pd
from cost_engine import replay_costs
ROOT=Path(__file__).resolve().parent

def main():
    # Analytic roundtrip: unchanged prices, half long/half short, Friday to Monday.
    time=pd.DataFrame([{'SignalDate':pd.Timestamp('2024-01-04'),'EntryDate':pd.Timestamp('2024-01-05'),'ExitDate':pd.Timestamp('2024-01-08'),'ValidationYear':2024}])
    t=pd.DataFrame([{'EntryDate':time.EntryDate.iloc[0],'ExitDate':time.ExitDate.iloc[0],
        'Executed':True,'RequestedWeight':w,'UnderlyingReturn':0.} for w in [.5,-.5]])
    rates={'commission_bp':5,'spread_bp':2.5,'slippage_bp':2.5,'borrow_annual':.03}
    d,l=replay_costs(t,time,rates)
    c=.001;n=100/(1+c)
    expected=n*(1-c)-n*.5*.03*3/365
    np.testing.assert_allclose(d.NetEquity.iloc[0],expected,atol=1e-12,rtol=0)
    np.testing.assert_allclose(l.NetBorrow.sum(),n*.5*.03*3/365,atol=1e-12)
    # Exit cost is charged on exit value: long +20%, short -10% price return.
    t['UnderlyingReturn']=[.2,-.1]
    d,l=replay_costs(t,time,rates)
    assert np.isclose(l.iloc[0].NetExitNotional,l.iloc[0].NetEntryNotional*1.2)
    assert np.isclose(l.iloc[1].NetExitNotional,l.iloc[1].NetEntryNotional*.9)
    # A cash/unavailable slot incurs no charge and is not reallocated.
    t.loc[1,'Executed']=False
    d,l=replay_costs(t,time,rates)
    assert len(l)==1 and l.NetBorrow.sum()==0
    assert np.isclose(l.NetEntryNotional.iloc[0],100*.5/(1+c*.5))
    # Delayed short over two intervals: Friday to Tuesday = 4 calendar days.
    timeline=pd.concat([time,pd.DataFrame([{'SignalDate':pd.Timestamp('2024-01-05'),'EntryDate':pd.Timestamp('2024-01-08'),'ExitDate':pd.Timestamp('2024-01-09'),'ValidationYear':2024}])],ignore_index=True)
    t=pd.DataFrame([{'EntryDate':pd.Timestamp('2024-01-05'),'ExitDate':pd.Timestamp('2024-01-09'),'Executed':True,'RequestedWeight':-.5,'UnderlyingReturn':.1},
        {'EntryDate':pd.Timestamp('2024-01-05'),'ExitDate':pd.Timestamp('2024-01-08'),'Executed':True,'RequestedWeight':.5,'UnderlyingReturn':.02},
        {'EntryDate':pd.Timestamp('2024-01-08'),'ExitDate':pd.Timestamp('2024-01-09'),'Executed':True,'RequestedWeight':.5,'UnderlyingReturn':.03}])
    d,l=replay_costs(t,timeline,rates)
    sh=l.loc[l.RequestedWeight.lt(0)].iloc[0]
    assert np.isclose(sh.NetBorrow,sh.NetEntryNotional*.03*4/365)
    assert d.CarriedPositionsAtExit.iloc[0]==1 and d.CarriedPositionsAtExit.iloc[1]==0
    assert np.isclose(d.ReservedCapital.iloc[1],sh.NetEntryNotional)

    results=json.loads((ROOT/'results.json').read_text());assert results['complete'] and len(results['models'])==5
    checks=[]
    for model in results['models']:
        old=pd.read_csv(ROOT/model['key']/'gross_daily_returns.csv')
        for scenario,spec in model['scenarios'].items():
            folder=ROOT/model['key']/scenario
            d=pd.read_csv(folder/'daily_returns.csv');t=pd.read_csv(folder/'trades.csv',parse_dates=['EntryDate','ExitDate'])
            r=spec['rates'];commission=r['commission_bp']/10000;spread=r['spread_bp']/10000;slippage=r['slippage_bp']/10000
            np.testing.assert_allclose(t.NetExitNotional,t.NetEntryNotional*(1+t.UnderlyingReturn),atol=1e-10,rtol=1e-12)
            values=t.NetEntryNotional+t.NetExitNotional
            for key,rate in [('Commission',commission),('Spread',spread),('Slippage',slippage)]:
                np.testing.assert_allclose(t['Net'+key],values*rate,atol=1e-10,rtol=1e-12)
            days=(t.ExitDate-t.EntryDate).dt.days
            borrow=np.where(t.RequestedWeight.lt(0),t.NetEntryNotional*r['borrow_annual']*days/365,0.)
            np.testing.assert_allclose(t.NetBorrow,borrow,atol=1e-10,rtol=1e-12)
            np.testing.assert_allclose(t.NetGrossPnl,t.NetEntryNotional*np.sign(t.RequestedWeight)*t.UnderlyingReturn,atol=1e-10,rtol=1e-12)
            np.testing.assert_allclose(t.NetPnlAfterCosts.sum(),d.NetEquity.iloc[-1]-100,atol=1e-10,rtol=0)
            wealth=(1+d.NetReturn).cumprod();peak=np.maximum.accumulate(np.r_[1.,wealth])[1:]
            assert abs(float(wealth.iloc[-1]-1)-spec['performance']['cumulative_return'])<1e-12
            assert abs(float((wealth/peak-1).min())-spec['performance']['max_drawdown'])<1e-12
            if scenario=='zero':np.testing.assert_allclose(d.NetReturn,old.StrategyReturn,atol=1e-12,rtol=0)
            for key in ['Commission','Spread','Slippage','Borrow']:
                assert abs(d[key].sum()-spec['costs_in_starting_capital_units'][key])<1e-9
            checks.append({'model':model['key'],'scenario':scenario,'trades':len(t),'days':len(d),'verified':True})
        for a,b in [('zero','low'),('low','middle'),('middle','high')]:
            assert model['scenarios'][a]['performance']['cumulative_return']>=model['scenarios'][b]['performance']['cumulative_return']
        t=pd.read_csv(ROOT/model['key']/'gross_trade_templates.csv',parse_dates=['EntryDate','ExitDate'])
        time=pd.read_csv(ROOT/model['key']/'gross_daily_returns.csv',parse_dates=['SignalDate','EntryDate','ExitDate'])[['SignalDate','EntryDate','ExitDate','ValidationYear']]
        for rate,be in model['break_even_all_in_per_leg_bp_by_borrow_rate'].items():
            r={'commission_bp':be,'spread_bp':0.,'slippage_bp':0.,'borrow_annual':float(rate)}
            d,_=replay_costs(t,time,r,False)
            assert abs(d.NetEquity.iloc[-1]-100)<.0001
    report={'all_checks_passed':True,'analytic_checks':['roundtrip entry reserve','entry and exit value costs','3-day weekend borrow','unavailable cash slot','4-day carry and reserved capital'],
        'scenario_checks':checks,'break_even_roots_checked':10,'unchanged_gross_dates_and_returns':True}
    (ROOT/'independent_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':main()
