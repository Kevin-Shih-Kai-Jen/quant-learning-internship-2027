"""Training-only ADF decisions and ACF/PACF diagnostics. Never loads Target."""
import argparse
import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, acf, pacf
from common import CONFIG, EXP, INPUT, OUT, YEARS, save, sha, hashes


def longest_segment(x):
    valid = np.isfinite(x)
    edges = np.diff(np.r_[False, valid, False].astype(np.int8))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    if not len(starts): return 0, 0
    idx = max(range(len(starts)), key=lambda i: (ends[i] - starts[i], ends[i]))
    return int(starts[idx]), int(ends[idx])


def decide_d(train, dates, code, year):
    """The public interface accepts the training prefix only, not future prices."""
    start, stop = longest_segment(train)
    record = {'code': int(code), 'year': int(year), 'train_end_exclusive': len(train),
              'train_last': str(dates[-1]), 'finite_training_prices': int(np.isfinite(train).sum()),
              'segment_start_index': start, 'segment_end_exclusive': stop,
              'segment_first': str(dates[start]) if stop > start else None,
              'segment_last': str(dates[stop - 1]) if stop > start else None,
              'segment_prices': stop-start, 'chosen_d': None, 'status': 'unresolved',
              'tests': [], 'acf': None, 'pacf': None}
    x = np.asarray(train[start:stop], dtype=float)
    if np.isfinite(train).sum() < CONFIG['arima_min_observations']:
        record['status'] = 'insufficient_fit_history'
        return record
    for d in CONFIG['d_candidates']:
        series = np.diff(x, n=d) if d else x.copy()
        test = {'d': d, 'sample_size_before_lags': len(series)}
        if len(series) < CONFIG['adf_min_observations']:
            test['status'] = 'insufficient_contiguous_history'
            record['tests'].append(test)
            break
        scale = np.std(series)
        if not np.isfinite(scale) or scale <= 1e-14 * max(1., abs(np.mean(series))):
            test['status'] = 'constant_or_invalid_series'
            record['tests'].append(test)
            break
        # Stable affine normalization leaves the ADF with a constant invariant.
        series = (series - np.mean(series)) / scale
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                stat, pval, lag, nobs, crit, icbest = adfuller(
                    series, maxlag=CONFIG['adf_maxlag'], regression=CONFIG['adf_regression'],
                    autolag=CONFIG['adf_autolag'])
            rejected = bool(stat < crit['5%'])
            test.update(status='ok', statistic=float(stat), pvalue=float(pval), lag=int(lag),
                        effective_nobs=int(nobs), critical_values={k: float(v) for k,v in crit.items()},
                        aic=float(icbest), reject_unit_root_5pct=rejected,
                        warnings=sorted(set(str(w.message) for w in caught)))
            record['tests'].append(test)
            if rejected:
                record.update(chosen_d=d, status='selected')
                record['acf'] = acf(series, nlags=CONFIG['acf_pacf_maxlag'], fft=True).tolist()
                record['pacf'] = pacf(series, nlags=CONFIG['acf_pacf_maxlag'], method='ywm').tolist()
                break
        except Exception as exc:
            test.update(status='exception', exception=f'{type(exc).__name__}: {exc}')
            record['tests'].append(test)
            break
    return record


def run(workers):
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(INPUT / 'prices.npz') as z:
        prices, dates, codes, vp, years = (z[k] for k in ['prices','dates','codes','valid_positions','years'])
    started = time.monotonic()
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {}
        for year in YEARS:
            end = int(vp[years == year][0])
            for ci, code in enumerate(codes):
                pending[pool.submit(decide_d, prices[:end,ci], dates[:end], code, year)] = (year,ci)
        for done, fut in enumerate(as_completed(pending),1):
            year,ci = pending[fut]
            row = fut.result(); row['ci'] = ci
            results.append(row)
            if done % 250 == 0 or done == len(pending):
                counts = pd.Series([r['chosen_d'] if r['chosen_d'] is not None else -1 for r in results]).value_counts().to_dict()
                progress = {'completed': done, 'total': len(pending), 'seconds': round(time.monotonic()-started,1),
                            'd_counts': {str(k): int(v) for k,v in counts.items()}}
                save(OUT / 'adf_progress.json', progress)
                print(json.dumps(progress), flush=True)
    results.sort(key=lambda r: (r['year'],r['ci']))
    save(OUT / 'adf_decisions.json', {'specification_hashes': hashes(), 'price_sha256': sha(INPUT/'prices.npz'),
                                   'records': results})
    flat = []
    for r in results:
        row = {k:v for k,v in r.items() if k not in ['tests','acf','pacf']}
        for t in r['tests']:
            for k in ['statistic','pvalue','lag','effective_nobs','reject_unit_root_5pct']:
                row[f'd{t["d"]}_{k}'] = t.get(k)
            row[f'd{t["d"]}_critical5'] = t.get('critical_values',{}).get('5%')
        flat.append(row)
    pd.DataFrame(flat).to_csv(OUT/'adf_decisions.csv',index=False)
    counts = []
    for year in YEARS:
        rows = [r for r in results if r['year']==year]
        counts.append({'year':year, 'total':len(rows), **{f'd{d}':sum(r['chosen_d']==d for r in rows) for d in [0,1,2]},
                       'unresolved':sum(r['chosen_d'] is None for r in rows)})
    save(EXP/'adf_summary.json', {'counts':counts, 'new_fits':25*sum(r['chosen_d'] in [0,2] for r in results),
                               'reused_fits':25*sum(r['chosen_d']==1 for r in results),
                               'zero_fallback_fits':25*sum(r['chosen_d'] is None for r in results),
                               'elapsed_seconds':time.monotonic()-started,'formal_test_read':False})
    print(json.dumps(json.loads((EXP/'adf_summary.json').read_text())),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=6)
    run(parser.parse_args().workers)
