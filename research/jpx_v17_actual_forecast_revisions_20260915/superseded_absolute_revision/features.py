from pathlib import Path
import sys,pickle,json,importlib.util,hashlib
from collections import Counter
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent;V14=BASE/'jpx_v14_filtered_forecast_events_20260914'
spec=importlib.util.spec_from_file_location('v14_events',V14/'features.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
NEW=['EPSForecastActualQoQ','EPSForecastActualYoY','NetSalesRevisionRelative','OperatingProfitRevisionRelative','EPSActualQoQ']
FINS=NEW+['EPSActualGrowth']
def save(path,x):path.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
class Builder(old.Builder):
    def __init__(self):super().__init__();self.new=[];self.skipped=Counter()
    def emit(self,row,key,m,kind,values,refs,detail):
        # Preserve/reproduce all old events; new features use their own availability tests.
        super().emit(row,key,m,kind,values,refs,detail)
        name=None;basekey=None;baserefs=[]
        if m==2 and kind in ['Q','Y']:
            name='EPSForecastActual'+('QoQ' if kind=='Q' else 'YoY')
            basekey=old.previous_quarter(key) if kind=='Q' else old.shift_year(key)
            bv,br=self.quarter(basekey);a=detail['Expected'];b=bv[m];baserefs=br[m]
        elif m in [0,1] and kind=='U':
            name=old.METRICS[m]+'RevisionRelative';a=detail['NewExpected'];b=detail['OldExpected']
        elif m==2 and kind=='A':
            name='EPSActualQoQ';basekey=old.previous_quarter(key);bv,br=self.quarter(basekey)
            a=detail['Actual'];b=bv[m];baserefs=br[m]
        if name is None:return
        value=old.grow(a,b)
        if not np.isfinite(value):self.skipped[name]+=1;return
        self.new.append({'NewEventId':len(self.new),'Feature':name,'Code':key[0],'Key':key,'Metric':m,'Kind':kind,'EffectiveTimestamp':row.EffectiveTimestamp,'AvailablePosition':row.AvailablePosition,'AvailableDate':row.AvailableDate,'SourceRow':row.SourceRow,'Value':float(value),'ComparedValue':float(a),'BaseValue':float(b),'BaseKey':basekey,'BaseActualReferences':baserefs,'References':sorted(set(refs+baserefs)),**{k:v for k,v in detail.items() if k not in ['Base']}})

def build():
    records=pd.read_pickle(V14/'financial_records.pkl');b=Builder();original,_,_=b.run(records)
    stored=pd.read_pickle(V14/'financial_events.pkl');pd.testing.assert_frame_equal(original,stored,check_exact=True)
    events=pd.DataFrame(b.new);events.to_pickle(ROOT/'new_financial_events.pkl')
    np.testing.assert_allclose(events.Value,(events.ComparedValue-events.BaseValue)/events.BaseValue.abs(),rtol=0,atol=0)
    lookup=records.set_index('SourceRow');checks=0
    # Independently derive each actual comparison baseline from raw cumulative rows.
    for e in events.itertuples():
        assert all(lookup.loc[k,'EffectiveTimestamp']<=e.EffectiveTimestamp for k in e.References)
        assert lookup.loc[e.SourceRow,'AvailablePosition']==e.AvailablePosition
        if e.BaseKey is not None:
            key=e.BaseKey;refs=e.BaseActualReferences;assert refs
            rows=lookup.loc[refs];assert rows.Code.eq(key[0]).all() and rows.Basis.eq(key[1]).all()
            assert rows.CurrentFiscalYearStartDate.eq(key[2]).all() and rows.CurrentFiscalYearEndDate.eq(key[3]).all()
            mapping={'1Q':1,'2Q':2,'3Q':3,'FY':4};vals={mapping[r.TypeOfCurrentPeriod]:getattr(r,old.ACTUAL[e.Metric]) for r in rows.itertuples()}
            expected=vals[key[4]]-(vals[key[4]-1] if key[4]>1 else 0.)
            np.testing.assert_allclose(expected,e.BaseValue,atol=0,rtol=0)
        elif e.Kind=='U':
            np.testing.assert_allclose(e.ComparedValue,(e.NewAnnual-e.Cumulative)/(4-e.KnownQuarters),atol=0,rtol=0)
            np.testing.assert_allclose(e.BaseValue,(e.OldAnnual-e.Cumulative)/(4-e.KnownQuarters),atol=0,rtol=0)
        checks+=1
    print('EVENTS',len(events),'independent source checks',checks,flush=True)
    with (BASE/'jpx_v8_soft_rank_20260912/inputs.pkl').open('rb') as h:f,x,groups,calendar=pickle.load(h)
    market=pd.DatetimeIndex(groups).difference(pd.DatetimeIndex(['2020-10-01'])).sort_values();positions=market.searchsorted(f.SignalDate)
    z=np.zeros((len(f),len(NEW)));field={c:j for j,c in enumerate(NEW)}
    eg={code:g.sort_values(['AvailablePosition','EffectiveTimestamp','NewEventId']) for code,g in events.groupby('Code')}
    samplechecks=0;maxerr=0.
    for code,ix in f.groupby('SecuritiesCode').indices.items():
        if code not in eg:continue
        rows=list(eg[code].itertuples(index=False));j=0;state=np.zeros(len(NEW));last=0
        for n,i in enumerate(ix):
            pos=positions[i]
            while j<len(rows) and rows[j].AvailablePosition<=pos:
                e=rows[j];state*=np.exp(-(e.AvailablePosition-last)/9);state[field[e.Feature]]+=e.Value;last=e.AvailablePosition;j+=1
            state*=np.exp(-(pos-last)/9);last=pos;z[i]=state
            if n%101==0 or n==len(ix)-1:
                independent=np.zeros(len(NEW))
                for e in rows[:j]:independent[field[e.Feature]]+=e.Value*np.exp(-(pos-e.AvailablePosition)/9)
                np.testing.assert_allclose(independent,state,atol=2e-10,rtol=2e-12)
                maxerr=max(maxerr,float(np.abs(independent-state).max()));samplechecks+=1
    out=pd.DataFrame(z,columns=NEW);out['EPSActualGrowth']=pd.read_pickle(V14/'financial_signal_features.pkl').EPSActualGrowth.to_numpy()
    out.to_pickle(ROOT/'financial_signal_features.pkl')
    save(ROOT/'feature_audit.json',{'passed':True,'old_167891_events_reproduced_exactly':True,'new_events':len(events),'event_counts':events.Feature.value_counts().to_dict(),'invalid_or_zero_baseline_events_skipped':dict(b.skipped),'all_event_formulas_checked':checks,'all_event_references_known_at_event_time':True,'actual_bases_independently_rebuilt_from_raw_ytd':True,'signal_rows':len(out),'direct_decayed_sum_checks':samplechecks,'direct_decayed_sum_max_abs_error':maxerr,'old_EPSActualGrowth_reused_exactly':True})
    save(ROOT/'feature_summary.json',{'ranges':{col:{'min':float(out[col].min()),'max':float(out[col].max()),'over100_signal_rows':int(out[col].abs().gt(100).sum())} for col in FINS},'source_sha256':{str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [V14/'financial_records.pkl',V14/'financial_events.pkl',V14/'financial_signal_features.pkl',ROOT/'features.py']}})
    print('FEATURES COMPLETE',len(out),samplechecks,flush=True)
if __name__=='__main__':build()
