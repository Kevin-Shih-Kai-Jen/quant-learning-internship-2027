from pathlib import Path
import importlib.util,pickle,json,hashlib
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent;B=R.parent;V14=B/'jpx_v14_filtered_forecast_events_20260914'
spec=importlib.util.spec_from_file_location('old_builder',V14/'features.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
COLS=['ProfitLatestYoY','ProfitYoYChange','ProfitPreviousYoY']
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def transition(previous,key,value,base,rebased_prior=np.nan):
    valid=bool(np.isfinite(value) and np.isfinite(base) and base!=0)
    if previous is not None and previous['key']==key and previous['valid']==valid:
        if not valid or (previous['value']==value and previous['base']==base):return None
    same=bool(valid and previous is not None and previous['valid'] and previous['key']==key)
    prior=previous['value'] if same else rebased_prior if valid and np.isfinite(rebased_prior) else value
    basis='same_quarter_previous' if same else 'prior_annual_rebased' if valid and np.isfinite(rebased_prior) else 'initial_no_prior'
    revision=bool(valid and prior!=value)
    new=(value-base)/abs(base) if valid else 0.;before=(prior-base)/abs(base) if valid else 0.
    return dict(key=key,value=value,base=base,valid=valid,revision=revision,prior=prior,new=new,old=before,delta=new-before,basis=basis)
class Builder(old.Builder):
    def __init__(self):super().__init__();self.states={};self.state_events=[]
    def process(self,row):
        start=len(self.snapshots);before_annual=self.annual.copy();result=super().process(row)
        for snap in self.snapshots[start:]:
            key=snap['Key'];code=key[0];basekey=old.shift_year(key);prior_year=self.frozen.get(basekey,{'v':np.full(3,np.nan),'refs':[[],[],[]],'time':pd.NaT})
            value=snap['Values'][1];base=prior_year['v'][1];previous=self.states.get(code)
            oldannual,oldrefs=before_annual.get(key[:4],(np.full(3,np.nan),[[],[],[]]))
            rebased=(oldannual[1]-snap['Cumulative'][1])/(4-snap['KnownQuarters'])
            state=transition(previous,key,value,base,rebased)
            if state is None:continue
            extra=previous['refs'] if state['basis']=='same_quarter_previous' else oldrefs[1] if state['basis']=='prior_annual_rebased' else []
            state['refs']=list(snap['References'][1])
            self.states[code]=state
            self.state_events.append(dict(StateEventId=len(self.state_events),Code=code,Key=key,BaseKey=basekey,EffectiveTimestamp=row.EffectiveTimestamp,AvailablePosition=row.AvailablePosition,AvailableDate=row.AvailableDate,SourceRow=row.SourceRow,Valid=state['valid'],IsRevision=state['revision'],OldBasisType=state['basis'],OldAnnual=float(oldannual[1]),NewForecast=float(value),OldForecast=float(state['prior']),BaseForecast=float(base),NewYoY=state['new'],OldYoY=state['old'],DeltaYoY=state['delta'],Annual=float(snap['Annual'][1]),Cumulative=float(snap['Cumulative'][1]),KnownQuarters=snap['KnownQuarters'],References=sorted(set(snap['References'][1]+prior_year['refs'][1]+extra)),BaseForecastTime=prior_year['time']))
        return result
def main():
    # Distinguish same-quarter revisions, repeated disclosures and new-quarter initialization.
    p=transition(None,'q',120.,100.);assert p['new']==p['old'] and p['delta']==0
    q=transition(p,'q',130.,100.);np.testing.assert_allclose([q['new'],q['old'],q['delta']],[.3,.2,.1])
    s=transition(q,'q',135.,100.);np.testing.assert_allclose([s['new'],s['old'],s['delta']],[.35,.3,.05])
    assert transition(s,'q',135.,100.) is None
    nextq=transition(s,'q2',160.,80.);assert nextq['new']==nextq['old']==1 and nextq['delta']==0
    assert transition(s,'q2',10.,0.)['new']==0
    rollover=transition(s,'q2',70.,50.,rebased_prior=60.);np.testing.assert_allclose([rollover['new'],rollover['old'],rollover['delta']],[.4,.2,.2]);assert rollover['revision']
    records=pd.read_pickle(V14/'financial_records.pkl');builder=Builder();events,snapshots,_=builder.run(records)
    pd.testing.assert_frame_equal(events,pd.read_pickle(V14/'financial_events.pkl'),check_exact=True)
    pd.testing.assert_frame_equal(snapshots,pd.read_pickle(V14/'quarter_forecasts.pkl'),check_exact=True)
    ev=pd.DataFrame(builder.state_events);ev.to_pickle(R/'state_events.pkl')
    lookup=records.set_index('SourceRow');snapmap={(s.SourceRow,s.Key):s for s in snapshots.itertuples()}
    for e in ev.itertuples():
        assert all(lookup.loc[k,'EffectiveTimestamp']<=e.EffectiveTimestamp for k in e.References)
        assert lookup.loc[e.SourceRow,'AvailablePosition']==e.AvailablePosition
        ss=snapmap[e.SourceRow,e.Key]
        np.testing.assert_allclose((e.Annual-e.Cumulative)/(4-e.KnownQuarters),e.NewForecast,equal_nan=True)
        np.testing.assert_allclose(ss.Values[1],e.NewForecast,equal_nan=True)
        if e.OldBasisType=='prior_annual_rebased':np.testing.assert_allclose(e.OldForecast,(e.OldAnnual-e.Cumulative)/(4-e.KnownQuarters),atol=0,rtol=0)
        if e.Valid:
            assert pd.notna(e.BaseForecastTime) and e.BaseForecastTime<e.EffectiveTimestamp
            np.testing.assert_allclose(e.NewYoY,(e.NewForecast-e.BaseForecast)/abs(e.BaseForecast),rtol=0,atol=0)
            np.testing.assert_allclose(e.OldYoY,(e.OldForecast-e.BaseForecast)/abs(e.BaseForecast),rtol=0,atol=0)
            np.testing.assert_allclose(e.DeltaYoY,e.NewYoY-e.OldYoY,rtol=0,atol=0)
    with (B/'jpx_v8_soft_rank_20260912/inputs.pkl').open('rb') as h:f,x,groups,cal=pickle.load(h)
    market=pd.DatetimeIndex(groups).difference(pd.DatetimeIndex(['2020-10-01'])).sort_values();positions=market.searchsorted(f.SignalDate)
    data=np.zeros((len(f),3));event_ids=np.full(len(f),-1,np.int32);event_pos=np.full(len(f),-1,np.int32)
    eventgroups={c:g.sort_values(['AvailablePosition','EffectiveTimestamp','StateEventId']) for c,g in ev.groupby('Code')}
    for code,ids in f.groupby('SecuritiesCode').indices.items():
        if code not in eventgroups:continue
        g=eventgroups[code];p=g.AvailablePosition.to_numpy();j=np.searchsorted(p,positions[ids],side='right')-1;valid=j>=0;dest=ids[valid];selected=g.iloc[j[valid]]
        values=selected[['NewYoY','DeltaYoY','OldYoY']].to_numpy();age=positions[dest]-selected.AvailablePosition.to_numpy();assert (age>=0).all()
        data[dest]=values*np.exp(-age[:,None]/9);event_ids[dest]=selected.StateEventId;event_pos[dest]=selected.AvailablePosition
        # Independent sequential last-event selection, not an accumulated event sum.
        rows=list(g.itertuples());cursor=0;last=None;sequential=np.zeros((len(ids),3))
        for n,i in enumerate(ids):
            while cursor<len(rows) and rows[cursor].AvailablePosition<=positions[i]:last=rows[cursor];cursor+=1
            if last is not None:
                expected=np.array([last.NewYoY,last.DeltaYoY,last.OldYoY])*np.exp(-(positions[i]-last.AvailablePosition)/9)
                sequential[n]=expected
        np.testing.assert_allclose(data[ids],sequential,atol=0,rtol=0)
    assert np.isfinite(data).all();np.testing.assert_allclose(data[:,0],data[:,1]+data[:,2],atol=2e-12,rtol=2e-12)
    pd.DataFrame(data,columns=COLS).to_pickle(R/'financial_signal_features.pkl');np.savez_compressed(R/'signal_event_mapping.npz',event_id=event_ids,event_position=event_pos)
    details=ev.drop(columns=['References','Key','BaseKey']).copy();details.to_csv(R/'state_events.csv',index=False,encoding='utf-8-sig')
    audit=dict(passed=True,old_events_and_forecast_snapshots_exactly_reproduced=True,state_events=len(ev),valid_events=int(ev.Valid.sum()),nonzero_revision_events=int(ev.IsRevision.sum()),same_quarter_revisions=int((ev.IsRevision&ev.OldBasisType.eq('same_quarter_previous')).sum()),quarter_rollover_rebased_revisions=int((ev.IsRevision&ev.OldBasisType.eq('prior_annual_rebased')).sum()),zero_delta_valid_states=int((ev.Valid&~ev.IsRevision).sum()),invalid_state_events=int((~ev.Valid).sum()),mapped_stock_days=len(f),forecast_formulas_checked=True,references_known_at_event_time=True,last_event_only_mapping_checked_all_rows=True,new_equals_delta_plus_old=True,transition_tests_passed=True)
    save(R/'feature_audit.json',audit);print(json.dumps(audit),flush=True)
    save(R/'feature_manifest.json',dict(source_sha256={str(p.relative_to(B)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [V14/'features.py',V14/'financial_records.pkl',V14/'financial_events.pkl',V14/'quarter_forecasts.pkl',B/'jpx_v8_soft_rank_20260912/inputs.pkl',R/'features.py']},decay='exp(-age_since_latest_state_update/9)',new_quarter='rebase prior annual forecast with same current YTD and remaining quarters; if unavailable old=new',invalid_baseline='clear signal to zero, retain stock'))
if __name__=='__main__':main()
