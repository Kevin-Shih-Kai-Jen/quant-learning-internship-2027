from pathlib import Path
import json
import numpy as np
import pandas as pd
from allocation import allocate_slots

weights=np.array([np.full(10,.07),np.full(10,.03)])
def run(values):
    frame=pd.DataFrame({'SecuritiesCode':np.arange(len(values))+1000,'Prediction':values})
    result=allocate_slots(frame,weights)
    assert not result.SecuritiesCode.duplicated().any()
    if len(result):
        assert (result.Weight*result.Prediction>0).all()
        assert result.Weight.abs().sum()<=1+1e-12
    return result

balanced=run(np.r_[np.arange(20,0,-1)/100,np.arange(-1,-21,-1)/100])
assert len(balanced)==20 and not balanced.Transferred.any()
assert set(balanced.loc[balanced.Weight>0,'SecuritiesCode'])==set(range(1000,1010))
assert set(balanced.loc[balanced.Weight<0,'SecuritiesCode'])==set(range(1030,1040))
np.testing.assert_allclose(balanced.loc[balanced.Weight>0,'Weight'],.07)
np.testing.assert_allclose(balanced.loc[balanced.Weight<0,'Weight'],-.03)
positive=run(np.arange(30,0,-1)/100)
assert len(positive)==20 and (positive.Weight>0).all() and positive.Transferred.sum()==10
assert set(positive.SecuritiesCode)==set(range(1000,1020))
np.testing.assert_allclose(positive.Weight.sum(),1.)
negative=run(-np.arange(1,31)/100)
assert len(negative)==20 and (negative.Weight<0).all() and negative.Transferred.sum()==10
assert set(negative.SecuritiesCode)==set(range(1010,1030))
np.testing.assert_allclose(negative.Weight.sum(),-1.)
onepositive=run(np.r_[.01,-np.arange(1,40)/100])
assert len(onepositive)==20 and onepositive.Transferred.sum()==9
np.testing.assert_allclose(onepositive.loc[onepositive.Weight>0,'Weight'].sum(),.07)
np.testing.assert_allclose(-onepositive.loc[onepositive.Weight<0,'Weight'].sum(),.93)
assert run(np.zeros(25)).empty
ties=run(np.r_[np.full(15,.01),np.full(15,-.01)])
assert set(ties.loc[ties.Weight>0,'SecuritiesCode'])==set(range(1000,1010))
assert set(ties.loc[ties.Weight<0,'SecuritiesCode'])==set(range(1015,1025))
small=run([.01,.02,-.01,-.02])
assert len(small)==4 and small.Weight.abs().sum()<1
result={'balanced_signs':True,'all_positive_transfer':True,'all_negative_transfer':True,
    'scarce_positive_transfer':True,'zero_predictions_cash':True,'deterministic_tie_break':True,
    'insufficient_candidates_no_duplicates':True,'all_passed':True}
(Path(__file__).resolve().parent/'allocation_contract_checks.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
