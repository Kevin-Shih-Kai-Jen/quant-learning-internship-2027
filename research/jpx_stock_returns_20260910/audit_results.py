from pathlib import Path
import pandas as pd
import numpy as np
import zipfile,json,hashlib
R=Path(__file__).resolve().parent
f=pd.read_pickle(R/'features.pkl').set_index(['SecuritiesCode','SignalDate'])
with zipfile.ZipFile('/Users/coolguy/Desktop/JPX_data/modelling/JPX_data.zip') as z:
    raw=pd.read_csv(z.open('JPX_data/raw/train_files/stock_prices.csv'),usecols=['SecuritiesCode','Date','Close','Volume','AdjustmentFactor'],parse_dates=['Date'])
market_open=(raw.Close.notna()&raw.Volume.gt(0)).groupby(raw.Date).any()
calendar=market_open.index[market_open]
checks=0;split_checks=0;zero_checks=0
for code,stock in raw.groupby('SecuritiesCode'):
    if code%19!=0 and not stock.AdjustmentFactor.ne(1).any():continue
    stock=stock.sort_values('Date').set_index('Date')
    stock=stock.reindex(calendar[(calendar>=stock.index.min())&(calendar<=stock.index.max())])
    if len(stock)<=60:continue
    splitpositions=np.flatnonzero(stock.AdjustmentFactor.fillna(1).ne(1))+1
    positions=set(np.linspace(60,len(stock)-1,5,dtype=int))|set(splitpositions)
    for t in sorted(positions):
        if not 60<=t<len(stock):continue
        now=stock.iloc[t];computed=f.loc[(code,stock.index[t])]
        for k in (5,22,60):
            before=stock.iloc[t-k]
            factor=stock.AdjustmentFactor.iloc[t-k:t].fillna(1).prod()
            pr=now.Close/(before.Close*factor)-1 if before.Close>0 else np.nan
            vr=now.Volume*factor/before.Volume-1 if before.Volume>0 else np.nan
            np.testing.assert_allclose(computed[f'PR{k}'],pr,rtol=1e-11,atol=1e-12,equal_nan=True)
            np.testing.assert_allclose(computed[f'VR{k}'],vr,rtol=1e-11,atol=1e-12,equal_nan=True)
            checks+=2;split_checks+=int(factor!=1);zero_checks+=int(before.Volume==0)
newcols=['PR5','PR22','PR60','VR5','VR22','VR60','PR5xVR5','PR22xVR22','PR60xVR60']
for code,stock in f.reset_index().groupby('SecuritiesCode'):
    stock=stock.sort_values('SignalDate').set_index('SignalDate')
    date=pd.Timestamp('2020-10-01')
    if date in stock.index:
        pos=stock.index.get_loc(date)
        if pos==0:
            assert stock.loc[date,newcols].isna().all()
            assert not stock.loc[date,'ReturnEligible']
            continue
        np.testing.assert_allclose(stock.loc[date,newcols].astype(float),stock.iloc[pos-1][newcols].astype(float),equal_nan=True)
orders=pd.read_csv(R/'requested_orders.csv',parse_dates=['SignalDate','EntryDate','ExitDate'])
d=pd.read_csv(R/'daily_returns.csv',parse_dates=['SignalDate','EntryDate','ExitDate'])
t=pd.read_csv(R/'selected_trades.csv')
sig=pd.read_csv(R.parent/'jpx_signed_band_20260910/market_signals.csv',parse_dates=['SignalDate','EntryDate','ExitDate'])
pd.testing.assert_frame_equal(d[['SignalDate','EntryDate','ExitDate','ValidationYear']],sig[['SignalDate','EntryDate','ExitDate','ValidationYear']],check_dtype=False)
o=orders.merge(sig[['SignalDate','Regime']],on='SignalDate',validate='many_to_one')
long=np.where(o.Regime.eq(1),.7,np.where(o.Regime.eq(-1),.3,.5))
budget=np.where(o.SourceSide.eq('long'),long,1-long)
np.testing.assert_allclose(o.Weight.abs(),budget*o.SourceRank.map({1:.5,2:.3,3:.2}),atol=1e-14)
assert (orders.SignalDate<orders.EntryDate).all() and (orders.EntryDate<orders.ExitDate).all()
assert (orders.Weight*orders.Prediction>0).all()
assert len(d)==953 and not orders.duplicated(['SignalDate','SecuritiesCode']).any()
for fit in json.loads((R/'model_fits.json').read_text()):
    assert fit['last_training_label_exit']<=fit['fit_asof']
result={'independent_raw_endpoint_checks':checks,'split_window_checks':split_checks,'zero_volume_denominator_checks':zero_checks,
    'market_closure_carry_verified':True,'same_market_dates_and_regime_budgets':True,'validation_intervals':len(d),
    'requested_positions':len(orders),'executed_positions':int(t.Executed.sum()),'delayed_positions':int(t.DelaySessions.gt(0).sum()),
    'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (R/'source').glob('*.py')},'all_passed':True}
(R/'independent_audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
print(json.dumps(result,ensure_ascii=False))
