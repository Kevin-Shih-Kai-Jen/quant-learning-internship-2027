import numpy as np
import pandas as pd

def allocate_slots(day,weights):
    """Same sign-transfer contract as the old allocator, generalized to N slots."""
    weights=np.asarray(weights,dtype=float)
    if weights.ndim!=2 or weights.shape[0]!=2 or weights.shape[1]<1 or not np.isfinite(weights).all() or (weights<0).any() or weights.sum()>1+1e-12:
        raise ValueError('Expected two equally sized rows of nonnegative slot budgets')
    n=weights.shape[1]
    up=day.sort_values(['Prediction','SecuritiesCode'],ascending=[False,True])
    down=day.sort_values(['Prediction','SecuritiesCode'],ascending=[True,True])
    slots=[]
    for side,ordered,amounts in [(1,up,weights[0]),(-1,down,weights[1])]:
        for rank,(idx,row) in enumerate(ordered.head(n).iterrows(),1):
            slots.append((idx,side,rank,amounts[rank-1],float(row.Prediction)))
    chosen=[];used=set();transfers=[]
    for idx,side,rank,amount,g in slots:
        if side*g>0:
            used.add(idx);chosen.append((idx,side*amount,side,rank,False))
        elif side*g<0:transfers.append((side,rank,amount))
    for source_side,rank,amount in transfers:
        new_side=-source_side
        ranked=down if new_side<0 else up
        candidates=ranked.loc[(ranked.Prediction*new_side>0)&~ranked.index.isin(used)]
        if len(candidates):
            idx=candidates.index[0];used.add(idx)
            chosen.append((idx,new_side*amount,source_side,rank,True))
    if not chosen:return pd.DataFrame(columns=list(day.columns)+['Weight','SourceSide','SourceRank','Transferred'])
    result=day.loc[[r[0] for r in chosen]].copy()
    result['Weight']=[r[1] for r in chosen]
    result['SourceSide']=['long' if r[2]>0 else 'short' for r in chosen]
    result['SourceRank']=[r[3] for r in chosen]
    result['Transferred']=[r[4] for r in chosen]
    assert result.SecuritiesCode.is_unique
    assert (result.Weight*result.Prediction>0).all()
    assert result.Weight.abs().sum()<=1+1e-12
    return result
