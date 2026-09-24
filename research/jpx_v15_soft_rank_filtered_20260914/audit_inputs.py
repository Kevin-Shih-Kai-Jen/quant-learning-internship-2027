import json,hashlib
import numpy as np
from features import ROOT,V14,V8,V11,load

def main():
    f,x,groups,calendar,fin=load()
    peak=np.abs(x[:,12:]).max(axis=1);expected=np.isfinite(x[:,12:]).all(axis=1)&(peak<=100)
    np.testing.assert_array_equal(fin.TrainingEligible.to_numpy(),expected)
    np.testing.assert_allclose(fin.TrainingFeatureMaxAbs.to_numpy(),peak,rtol=0,atol=0)
    assert json.loads((V14/'feature_audit.json').read_text())['passed']
    inputs={}
    for p in [V8/'inputs.pkl',V11/'return_std22_features.pkl',V14/'financial_signal_features.pkl',V14/'feature_audit.json']:
        h=hashlib.sha256()
        with p.open('rb') as handle:
            for b in iter(lambda:handle.read(8*1024*1024),b''):h.update(b)
        inputs[str(p)]=h.hexdigest()
    result={'passed':True,'financial_source':'v14 frozen audited cache','all_signal_rows':len(f),
        'eligible_rows':int(expected.sum()),'ineligible_rows':int((~expected).sum()),'all_feature_thresholds_recomputed':True,'source_inputs':inputs}
    (ROOT/'input_audit.json').write_text(json.dumps(result,indent=2));print(result)
if __name__=='__main__':main()
