from pathlib import Path
import sys,json,zipfile,hashlib,gc
import numpy as np
import pandas as pd
from cost_engine import replay_costs
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'jpx_stock_returns_20260910'
sys.path.insert(0,str(BASE/'source'))
from jpx_execution import execute_hold,ORDER_COLUMNS
from jpx_common import performance
EXP=Path('/Users/coolguy/Desktop/JPX_data/modelling/JPX-market-regime-experiment.zip')
def rates(c,s,l,b):return dict(commission_bp=c,spread_bp=s,slippage_bp=l,borrow_annual=b)
SCENARIOS={'zero':rates(0,0,0,0),'low':rates(1,1,1,.01),'middle':rates(5,2.5,2.5,.03),'high':rates(10,5,5,.10),
    'commission_only':rates(5,0,0,0),'commission_spread':rates(5,2.5,0,0),'commission_spread_slippage':rates(5,2.5,2.5,0)}
SPECS=[('dynamic_close','動態配置／收盤出場','regime',True),('fixed30_close','固定多30%空70%／收盤出場','fixed_30_70',True),
    ('fixed50_close','固定多50%空50%／收盤出場','baseline_replay',True),('dynamic_target','原始動態配置／目標價出場','regime',False),
    ('fixed30_target','固定多30%空70%／目標價出場','fixed_30_70',False)]
def save(path,x):path.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))

def breakeven(t,time,borrow):
    lo,hi=0.,100.
    for _ in range(32):
        mid=(lo+hi)/2;d,_=replay_costs(t,time,rates(mid,0,0,borrow),False)
        if d.NetEquity.iloc[-1]>=100:lo=mid
        else:hi=mid
    return (lo+hi)/2

def main():
    allresults=[];z=zipfile.ZipFile(EXP)
    benchmark=pd.read_csv(z.open('jpx_inputs/nikkei_benchmark.csv'),parse_dates=['Date']).set_index('Date').Close
    bars=pd.read_pickle(BASE/'bars.pkl')
    for key,name,zkey,close in SPECS:
        out=ROOT/key;out.mkdir(exist_ok=True)
        reference=pd.read_csv(z.open(f'jpx_market_regime_results/{zkey}/daily_returns.csv'),parse_dates=['SignalDate','EntryDate','ExitDate'])
        saved=pd.read_csv(z.open(f'jpx_market_regime_results/{zkey}/selected_trades.csv'),parse_dates=['SignalDate','PreviousSignalDate','EntryDate','ExitDate','PlannedExitDate'])
        saved['Weight']=saved.RequestedWeight;saved['ExitDate']=saved.PlannedExitDate
        orders=saved[ORDER_COLUMNS]
        time=reference[['SignalDate','EntryDate','ExitDate','ValidationYear']]
        subset=bars.loc[bars.SecuritiesCode.isin(orders.SecuritiesCode.unique())&bars.Date.between(time.EntryDate.min(),time.ExitDate.max())]
        t,d,e=execute_hold(orders,subset,time,benchmark,close_only=close)
        target=reference.CloseOnlySamePositionsReturn if close else reference.StrategyReturn
        np.testing.assert_allclose(d.StrategyReturn,target,atol=1e-12,rtol=0)
        t.to_csv(out/'gross_trade_templates.csv',index=False);d.to_csv(out/'gross_daily_returns.csv',index=False)
        variants={}
        for scenario,rate in SCENARIOS.items():
            daily,ledger=replay_costs(t,time,rate)
            if scenario=='zero':np.testing.assert_allclose(daily.NetReturn,target,atol=1e-12,rtol=0)
            folder=out/scenario;folder.mkdir(exist_ok=True)
            daily.to_csv(folder/'daily_returns.csv',index=False);ledger.to_csv(folder/'trades.csv',index=False)
            variants[scenario]={'performance':performance(daily.NetReturn),'rates':rate,
                'costs_in_starting_capital_units':{k:float(daily[k].sum()) for k in ['Commission','Spread','Slippage','Borrow','TotalCost']},
                'average_roundtrip_turnover':float((daily.EntryTurnover+daily.ExitTurnover).mean()),
                'annual':[{'year':int(y),'performance':performance(g.NetReturn)} for y,g in daily.groupby('ValidationYear')]}
            print('RESULT',key,scenario,variants[scenario]['performance']['cumulative_return'],abs(variants[scenario]['performance']['max_drawdown']),flush=True)
        be={str(b):breakeven(t,time,b) for b in [0.,.03]}
        gross=variants['zero'];item={'key':key,'name':name,'close_only':close,'scenarios':variants,'break_even_all_in_per_leg_bp_by_borrow_rate':be,
            'gross_executed_trades':int(t.Executed.sum()),'gross_short_trades':int((t.Executed&t.RequestedWeight.lt(0)).sum()),
            'gross_mean_short_exposure':float(d.ShortExposure.mean()),'gross_mean_daily_return_bp':gross['performance']['daily_mean']*10000,
            'zero_cost_baseline_max_daily_error':float(abs(target-pd.read_csv(out/'zero/daily_returns.csv').NetReturn).max())}
        allresults.append(item);save(out/'results.json',item);save(ROOT/'results.json',{'models':allresults,'complete':len(allresults)==5})
        print('BREAK_EVEN',key,be,flush=True);gc.collect()
    save(ROOT/'results.json',{'models':allresults,'complete':True,'scenarios':SCENARIOS,'benchmark':performance(reference.IndexReturn),
        'cost_rates_are_hypothetical':True,'costs_are_execution_debits_not_new_fill_prices':True,'stock_models_retrained':False,'test_used':False,'capital_scale':100})
    save(ROOT/'source_hashes.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'experiment_plan.md',ROOT/'cost_engine.py',ROOT/'run_experiment.py',BASE/'source/jpx_execution.py']})
    print('COMPLETE',flush=True)
if __name__=='__main__':main()
