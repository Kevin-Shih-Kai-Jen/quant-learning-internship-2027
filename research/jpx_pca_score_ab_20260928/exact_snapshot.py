"""A one-signal-date check of the exact PC6 discussed in the original report."""
from pathlib import Path
import argparse
import hashlib
import zipfile
import numpy as np
import pandas as pd
from run_ab import (RAW_SHA, RANK_SHA, raw_member, sha, save, records, build_returns,
                    rank_weights, ablate, spread_from_ranks, spearman)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--zip', type=Path, required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError('New output directory required')
    a.out.mkdir()
    date = pd.Timestamp('2021-12-01')
    rankfile = a.input / 'research/jpx_v7_daily_mse_20260912/v7_equal/ranks_2021.csv.gz'
    assert sha(rankfile) == RANK_SHA[2021]
    frame = pd.read_csv(rankfile, usecols=['Date', 'SecuritiesCode', 'g', 'Rank'], parse_dates=['Date'],
                        float_precision='round_trip')
    frame = frame.loc[frame.Date.eq(date)].sort_values('SecuritiesCode').reset_index(drop=True)
    codes, g = frame.SecuritiesCode.to_numpy(), frame.g.to_numpy()
    arank, aw = rank_weights(g, codes)
    np.testing.assert_array_equal(arank, frame.Rank)
    with zipfile.ZipFile(a.zip) as z:
        with z.open(raw_member(z)) as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == RAW_SHA
        with z.open(raw_member(z)) as stream:
            prices = pd.read_csv(stream, usecols=['Date', 'SecuritiesCode', 'Close', 'Volume', 'AdjustmentFactor'],
                                 parse_dates=['Date'])
    panel, _ = build_returns(prices)
    window = panel.loc[:date, codes].tail(252)
    keep = window.notna().all() & window.std(ddof=1).gt(0)
    selected_codes = sorted(set(window.columns[keep]) | set(codes[aw != 0]))
    joint = window[selected_codes].dropna(axis=0)
    assert joint.shape == (160, 1933)
    x = joint.to_numpy() - joint.mean().to_numpy()
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    v = vt[:159].T.copy()
    for k in range(v.shape[1]):
        if v[np.argmax(np.abs(v[:, k])), k] < 0:
            v[:, k] *= -1
    l = s[:159] ** 2 / 159
    fitpos = np.searchsorted(codes, selected_codes)
    beta = aw[fitpos] @ v
    risk = l * beta**2
    np.testing.assert_allclose(l[5], .013623253945793, rtol=1e-10)
    np.testing.assert_allclose(beta[5], .002847851170449, rtol=1e-10)
    np.testing.assert_allclose(risk[5] / risk.sum(), .09764861379843556, rtol=1e-10)
    b = g.copy()
    b[fitpos], slope, fraction = ablate(g[fitpos], v[:, 5])
    brank, bw = rank_weights(b, codes)
    fullv = np.full(len(codes), np.nan)
    fullv[fitpos] = v[:, 5]
    prediction = pd.DataFrame({'SecuritiesCode': codes, 'A_score': g, 'B_score': b,
        'A_rank': arank, 'B_rank': brank, 'PC6_loading': fullv})
    prediction.to_csv(a.out / 'predictions_before_labels.csv', index=False)
    # The original forward Target is only read after the prediction file exists.
    with zipfile.ZipFile(a.zip) as z:
        with z.open(raw_member(z)) as stream:
            labels = pd.read_csv(stream, usecols=['Date', 'SecuritiesCode', 'Target'], parse_dates=['Date'])
    target = labels.loc[labels.Date.eq(date)].set_index('SecuritiesCode').Target.reindex(codes).to_numpy()
    rows = []
    for label, score, rank, w in [('A', g, arank, aw), ('B_exact_PC6', b, brank, bw)]:
        spread, missing = spread_from_ranks(rank, target)
        assert missing == 0
        e = w[fitpos] @ v
        r = l * e**2
        rows.append({'variant': label, 'official_daily_spread': spread, 'RankIC': spearman(score, target),
            'gross_one_scaled_spread': spread / 400, 'PC6_exposure_covered': float(e[5]),
            'PC6_variance_covered': float(r[5]), 'covered_total_variance': float(r.sum()),
            'PC6_variance_share_covered': float(r[5] / r.sum()),
            'covered_gross': float(np.sum(np.abs(w[fitpos]))),
            'replaced_long_vs_A': int(np.sum((rank < 200) & (arank >= 200))),
            'replaced_short_vs_A': int(np.sum((rank >= len(rank)-200) & (arank < len(rank)-200)))})
    result = {'signal_date': '2021-12-01', 'target_return_dates': ['2021-12-02', '2021-12-03'],
        'pca_days': 160, 'pca_stocks': 1933, 'ranked_stocks': 1996, 'exact_original_PC6_reproduced': True,
        'score_slope': slope, 'removed_score_variance_share': fraction, 'results': rows,
        'single_date_spread_difference_B_minus_A': rows[1]['official_daily_spread']-rows[0]['official_daily_spread'],
        'Sharpe_computable': False, 'formal_test_used': False,
        'interpretation': 'Single date only. Does not prove long-run usefulness; no retrospective use of this vector on earlier dates.'}
    save(a.out / 'result.json', result)
    print(result)


if __name__ == '__main__':
    main()
