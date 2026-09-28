"""Audit every accepted fit using saved parameters, never refit or select by score."""
from pathlib import Path
import json, warnings, shutil, sys
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from run_arima515 import RUN,tasks


def check(task, future=False):
    code,year,start,train_end,positions,y,dates=task
    stem=RUN/'jobs'/f'{year}_{code}'
    r=json.loads(stem.with_suffix('.json').read_text())
    if r['status']!='ok':return {'code':code,'year':year,'checked':False}
    z=(y-r['anchor'])/r['scale']
    params=np.load(stem.with_suffix('.npz'))['params']
    reason=None;minimum=None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            f=ARIMA(z if future else z[:train_end],order=(5,1,5),trend='n').filter(params,cov_type='none')
        mask=np.isfinite(z if future else z[:train_end]);mask[:int(f.loglikelihood_burn)]=False
        v=f.filter_results.forecasts_error_cov[0,0,mask]
        minimum=float(np.min(v)) if np.isfinite(v).all() else None
        if not np.isfinite(v).all() or not (v>0).all() or not np.isfinite(f.llf) or f.llf==0:
            reason='Non-positive/non-finite innovation variance or degenerate log likelihood'
        elif not future and not np.isclose(f.llf,r['attempts'][-1]['llf'],rtol=1e-9,atol=1e-7):
            raise AssertionError('Saved likelihood not reproducible from saved parameters')
    except AssertionError:raise
    except Exception as exc:reason=f'{type(exc).__name__}: {exc}'
    return {'code':code,'year':year,'checked':True,'invalid':reason is not None,
            'reason':reason,'min_innovation_variance':minimum}


if __name__=='__main__':
    future='--future' in sys.argv
    with ProcessPoolExecutor(max_workers=6) as pool:
        results=list(pool.map(partial(check,future=future),list(tasks()),chunksize=10))
    bad=[r for r in results if r.get('invalid')]
    if future:
        report={'accepted_fits_checked':sum(r['checked'] for r in results),
                'invalid_full_forward_filters':bad,'passed':not bad,'predictions_modified':False}
        (RUN/'full_filter_numerical_audit.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2),flush=True)
        sys.exit(0 if not bad else 1)
    backup=RUN/'invalid_numerical_backups';backup.mkdir(exist_ok=True)
    for r in bad:
        stem=RUN/'jobs'/f"{r['year']}_{r['code']}"
        for ext in ['.json','.npz']:shutil.copy2(stem.with_suffix(ext),backup/stem.with_suffix(ext).name)
        a=json.loads(stem.with_suffix('.json').read_text())
        p=np.load(stem.with_suffix('.npz'));n=len(p['score'])
        a['original_status']=a['status'];a['status']='invalid_numerical_fit'
        a['numerical_audit_reason']=r['reason'];a['fallback_forecasts']=n
        a['min_innovation_variance']=r['min_innovation_variance']
        a['prefix_checks']=[]
        np.savez_compressed(stem.with_suffix('.npz'),score=np.zeros(n),fallback=np.ones(n,bool),
                            forecast_prices=np.full((n,2),np.nan),params=np.empty(0))
        stem.with_suffix('.json').write_text(json.dumps(a,ensure_ascii=False,indent=2,allow_nan=False))
    summary={'accepted_fits_checked':sum(r['checked'] for r in results),
             'invalid_numerical_fits':bad,'changed_to_prespecified_zero_score_fallback':len(bad),
             'selection_uses_validation_performance':False}
    (RUN/'numerical_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
