"""Reproducible, descriptive JPX covariance PCA. No alpha-model retraining.

Inputs are hash-verified original bars, stored v7 holdings/ranks, stock metadata,
and the complete raw CSV read DIRECTLY from the uploaded ZIP. Outputs are NEW.
"""
from pathlib import Path
import argparse, csv, hashlib, json, platform, zipfile, io
import numpy as np
import pandas as pd
from jpx_pca import fit_pca, risk, sha256, returns_from_raw

ASOF = pd.Timestamp('2021-12-01')
LOOKBACK = 252
EXPECTED = {
    'bars': '25e2bb691ed156b083727649a9f74fdf2bc70a8d958050f48e68aecd3c9d8162',
    'holdings': '7563b3f6ade1940d2e9b77c31eb67303c752a4c0b275c639ecf120304c507dfc',
    'ranks': 'd0029f7dc9c89362183250321f6583a56df820b337a7233bdfba2ea65adfe42a',
    'raw': 'bf774a86f834e5338bba74f2356ebb750f93a1cc4e89b064ee07251e8ed851db',
    'metadata': '88662c9cba6c0029cb1850032423c509d94536a177a79515b4069102c6674f3a',
}


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def table_records(frame):
    return json.loads(frame.to_json(orient='records',date_format='iso',double_precision=15))


def audit_raw(raw, bars, archive_path):
    # Re-read the ZIP member directly; an earlier extracted working copy was incomplete.
    with zipfile.ZipFile(archive_path) as z, z.open('raw/train_files/stock_prices.csv') as stream:
        reader=csv.reader(io.TextIOWrapper(stream)); header=next(reader); bad=[]; count=0
        for line,row in enumerate(reader,2):
            count+=1
            if len(row)!=len(header):bad.append({'line':line,'field_count':len(row),'row_id':row[0]})
    assert not bad and count==len(raw)==len(bars)==2332531
    complete=raw.copy()
    overlap=complete.merge(bars,on=['Date','SecuritiesCode'],suffixes=('_raw','_cache'),validate='one_to_one')
    assert len(overlap)==len(complete)
    match={}
    for col in ['Open','High','Low','Close','Volume']:
        a,b=overlap[col+'_raw'],overlap[col+'_cache']
        same=a.eq(b)|(a.isna()&b.isna())
        assert same.all(),col
        match[col]={'matched_rows':int(same.sum()),'max_abs_difference':float((a-b).abs().max())}
    # Validate cached cumulative adjustment, not merely unadjusted closes.
    complete=complete.sort_values(['SecuritiesCode','Date'])
    complete['RebuiltFactor']=complete.groupby('SecuritiesCode').AdjustmentFactor.transform(
        lambda x:x.fillna(1).cumprod().shift(1,fill_value=1))
    factor=complete[['Date','SecuritiesCode','RebuiltFactor']].merge(
        bars[['Date','SecuritiesCode','CumulativeFactor']],on=['Date','SecuritiesCode'],validate='one_to_one')
    np.testing.assert_allclose(factor.RebuiltFactor,factor.CumulativeFactor,rtol=1e-13,atol=1e-13)
    return {'raw_rows':count,'raw_last_date':str(raw.Date.max().date()),'malformed_rows':bad,
        'last_complete_raw_session':str(raw.Date.max().date()),'overlap_rows':len(overlap),'overlap_checks':match,
        'cached_factor_max_difference':float((factor.RebuiltFactor-factor.CumulativeFactor).abs().max()),
        'issue':'Source ZIP is COMPLETE. An earlier 187695104-byte extracted temporary copy was incomplete; it is rejected and never used for final PCA.',
        'cache_status':'Archive bytes verified. All 2332531 original raw rows match cached OHLCV and cumulative factors.'}


def returns_panel(bars):
    # No Target is read and no post-asof price enters estimation.
    b=bars.loc[bars.Date.le(ASOF)].copy()
    assert not b.duplicated(['Date','SecuritiesCode']).any()
    assert np.isfinite(b.CumulativeFactor).all() and b.CumulativeFactor.gt(0).all()
    traded=b.Close.gt(0)&b.Volume.gt(0)
    market=traded.groupby(b.Date).any().sort_index()
    cal=pd.DatetimeIndex(market.index[market])
    b['AdjustedClose']=b.Close/b.CumulativeFactor
    close=b.pivot(index='Date',columns='SecuritiesCode',values='AdjustedClose').reindex(cal)
    vol=b.pivot(index='Date',columns='SecuritiesCode',values='Volume').reindex(cal)
    good=close.gt(0)&np.isfinite(close)&vol.gt(0)&np.isfinite(vol)
    # Daily returns are formed BEFORE selecting common dates: no gap bridging.
    panel=(close/close.shift(1)-1).where(good&good.shift(1,fill_value=False))
    return panel, [str(d.date()) for d in market.index[~market]]


def fit_case(panel, name, weights, meta, out, scope):
    fit=fit_pca(panel,panel.index.max(),lookback=len(panel),min_observations=60)
    assert len(fit.columns)==len(panel.columns) and not fit.excluded
    x=fit.training-fit.mean; v=fit.directions; l=fit.eigenvalues
    scores=x@v
    orth=float(np.max(np.abs(v.T@v-np.eye(len(l)))))
    eigerr=float(np.linalg.norm(x.T@scores/(len(x)-1)-v*l)/np.linalg.norm(v*l))
    reconstruction=float(np.linalg.norm(x-scores@v.T)/np.linalg.norm(x))
    np.testing.assert_allclose(np.var(scores,axis=0,ddof=1),l,rtol=1e-10,atol=1e-14)
    assert orth<1e-10 and eigerr<1e-10 and reconstruction<1e-10
    # Independent power iteration through the covariance operator; no NxN allocation.
    p=np.random.default_rng(27).normal(size=len(fit.columns));p/=np.linalg.norm(p)
    for iteration in range(2000):
        q=x.T@(x@p);q/=np.linalg.norm(q)
        if min(np.linalg.norm(q-p),np.linalg.norm(q+p))<1e-11:break
        p=q
    agreement=float(abs(q@v[:,0]));assert agreement>1-1e-9
    ratios=fit.explained_variance_ratio
    components=pd.DataFrame({'PC':np.arange(1,len(l)+1),'eigenvalue':l,
        'market_variance_share':ratios,'cumulative_market_share':np.cumsum(ratios)})
    portfolios={}
    for key,w in weights.items():
        tab,summary=risk(fit,w)
        summary.update(first_pc_share=float(tab.portfolio_variance_share.iloc[0]),
            first5_share=float(tab.portfolio_variance_share.iloc[:5].sum()),
            first10_share=float(tab.portfolio_variance_share.iloc[:10].sum()),
            largest_risk_pc=int(tab.loc[tab.portfolio_variance_share.idxmax(),'PC']),
            largest_risk_pc_share=float(tab.portfolio_variance_share.max()),
            effective_risk_components=float(1/(tab.portfolio_variance_share**2).sum()))
        portfolios[key]=summary
        for c in ['exposure','variance_contribution','portfolio_variance_share']:
            components[key+'_'+c]=tab[c].to_numpy()
    loadings=pd.DataFrame(v[:,:10],index=fit.columns,columns=[f'PC{k}' for k in range(1,11)]).rename_axis('SecuritiesCode')
    labels=meta.set_index('SecuritiesCode')[['Name','17SectorName','33SectorName']]
    loadings=loadings.join(labels)
    selected=[];sector_profiles=[]; correlations=[]
    for k in range(10):
        col=f'PC{k+1}'
        for side,frame in [('positive',loadings.nlargest(12,col)),('negative',loadings.nsmallest(12,col))]:
            for code,row in frame.iterrows():
                selected.append({'PC':k+1,'side':side,'SecuritiesCode':code,'Name':row.Name,
                    'sector':row['17SectorName'],'loading':float(row[col]),'loading_square':float(row[col]**2)})
        sectors=loadings.assign(squared=loadings[col]**2).groupby('17SectorName').agg(
            loading_energy=('squared','sum'),stock_count=('squared','size')).reset_index()
        sectors['PC']=k+1;sector_profiles.extend(table_records(sectors.sort_values('loading_energy',ascending=False)))
        for sector,codes in loadings.groupby('17SectorName').groups.items():
            ids=[fit.columns.index(c) for c in codes]
            corr=float(np.corrcoef(scores[:,k],fit.training[:,ids].mean(axis=1))[0,1])
            correlations.append({'PC':k+1,'sector':sector,'score_sector_ew_correlation':corr})
    components.to_csv(out/(name+'_components.csv'),index=False)
    pd.DataFrame(selected).to_csv(out/(name+'_top_loadings.csv'),index=False)
    # Actual dated observations and membership make the complete-case choice inspectable.
    pd.DataFrame({'Date':fit.dates.strftime('%Y-%m-%d')}).to_csv(out/(name+'_dates.csv'),index=False)
    pd.DataFrame({'SecuritiesCode':fit.columns}).to_csv(out/(name+'_universe.csv'),index=False)
    info={'name':name,'scope':scope,'start':str(fit.dates.min().date()),'end':str(fit.dates.max().date()),
        'dates':len(fit.dates),'stocks':len(fit.columns),'rank':len(l),'maximum_centered_rank':min(len(x)-1,len(fit.columns)),
        'pc1_share':float(ratios[0]),'pc1_positive_loading_fraction':float((v[:,0]>0).mean()),
        'pc1_score_ew_correlation':float(np.corrcoef(scores[:,0],fit.training.mean(axis=1))[0,1]),
        'first5_share':float(ratios[:5].sum()),'first10_share':float(ratios[:10].sum()),
        'pcs_for_50pct':int(np.searchsorted(np.cumsum(ratios),.5)+1),
        'pcs_for_80pct':int(np.searchsorted(np.cumsum(ratios),.8)+1),
        'pcs_for_90pct':int(np.searchsorted(np.cumsum(ratios),.9)+1),
        'effective_variance_components':float(1/(ratios**2).sum()),'portfolios':portfolios,
        'audit':{'orthogonality_max_abs_error':orth,'relative_eigen_equation_error':eigerr,
            'relative_training_reconstruction_error':reconstruction,'evr_sum':float(ratios.sum()),
            'power_iteration_pc1_abs_cosine':agreement,'power_iterations':iteration+1},
        'sector_profiles':sector_profiles,'sector_correlations':correlations,'top_loadings':selected,
        'components':table_records(components)}
    print(name,json.dumps({k:info[k] for k in ['stocks','dates','rank','pc1_share','first5_share','first10_share','portfolios']},ensure_ascii=False),flush=True)
    return info


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--zip',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();b=a.input
    paths={'bars':b/'archive/research/jpx_stock_returns_20260910/bars.pkl',
        'holdings':b/'archive/research/jpx_v7_daily_mse_20260912/v7_equal/selected_200_each_side.csv.gz',
        'ranks':b/'archive/research/jpx_v7_daily_mse_20260912/v7_equal/ranks_2021.csv.gz',
        'metadata':b/'stock_list.csv'}
    for k,f in paths.items():assert sha256(f)==EXPECTED[k],k
    if a.out.exists():raise FileExistsError('Use a new output directory')
    a.out.mkdir(parents=True)
    bars=pd.read_pickle(paths['bars']);bars.SecuritiesCode=bars.SecuritiesCode.astype(str)
    with zipfile.ZipFile(a.zip) as z:
        with z.open('raw/train_files/stock_prices.csv') as stream:
            assert hashlib.file_digest(stream,'sha256').hexdigest()==EXPECTED['raw']
        with z.open('raw/train_files/stock_prices.csv') as stream:
            raw=pd.read_csv(stream,dtype={'SecuritiesCode':str,'SupervisionFlag':str},parse_dates=['Date'])
    print('Direct ZIP read:',len(raw),raw.Date.min(),raw.Date.max(),flush=True)
    audit=audit_raw(raw,bars,a.zip);save(a.out/'data_audit.json',audit)
    expected_universe=sorted(raw.loc[raw.Date.eq(ASOF)&raw.SupervisionFlag.eq('False'),'SecuritiesCode'])
    del raw
    meta=pd.read_csv(paths['metadata'],dtype={'SecuritiesCode':str})
    holdings=pd.read_csv(paths['holdings'],dtype={'SecuritiesCode':str},parse_dates=['Date'])
    held=holdings.loc[holdings.Date.eq(ASOF)].copy()
    w=held.set_index('SecuritiesCode').IllustrativeSigned50_50Weight
    assert len(w)==400 and w.index.is_unique
    np.testing.assert_allclose([w.abs().sum(),w.sum()],[1,0],atol=1e-12)
    for side,sgn in [('long',1),('short',-1)]:
        h=held.loc[held.Side.eq(side)].sort_values('SideRank')
        assert len(h)==200
        np.testing.assert_allclose(h.IllustrativeSigned50_50Weight,sgn*.5*np.linspace(2,1,200)/300,atol=1e-15)
    ranks=pd.read_csv(paths['ranks'],dtype={'SecuritiesCode':str},parse_dates=['Date'])
    universe=sorted(ranks.loc[ranks.Date.eq(ASOF),'SecuritiesCode'])
    assert universe==expected_universe and len(universe)>=400 and set(w.index).issubset(universe)
    panel,closed=returns_panel(bars)
    train=panel.reindex(columns=universe).loc[:ASOF].tail(LOOKBACK)
    complete=train.notna().all()&train.std(ddof=1).gt(0)
    core=train.loc[:,complete]
    missing=w.loc[~w.index.isin(core.columns)]
    covered=w.loc[w.index.isin(core.columns)]
    exclusion=pd.DataFrame({'SecuritiesCode':train.columns[~complete],
        'missing_days':train.loc[:,~complete].isna().sum().to_numpy()})
    exclusion=exclusion.merge(meta[['SecuritiesCode','Name','17SectorName']],on='SecuritiesCode',validate='one_to_one')
    exclusion['Weight']=exclusion.SecuritiesCode.map(w).fillna(0)
    exclusion.to_csv(a.out/'excluded_252.csv',index=False)
    held[['Date','SecuritiesCode','Side','SideRank','IllustrativeSigned50_50Weight']].to_csv(a.out/'holdings_20211201.csv',index=False)
    cases=[]
    cases.append(fit_case(core,'market_252',{'covered_v7':covered},meta,a.out,
        '252 consecutive market sessions. Only complete stocks; covered_v7 excludes 8 missing-data positions WITHOUT rescaling and is not full-portfolio risk.'))
    # Supplement: same broad universe plus all otherwise excluded actual holdings.
    # Keep only dates with observed daily returns across this common universe.
    joint_codes=sorted(set(core.columns)|set(w.index))
    joint=train[joint_codes].dropna(axis=0)
    assert len(joint)>=60
    equal=pd.Series(1/len(joint_codes),index=joint_codes)
    long=w.loc[w>0];short=w.loc[w<0]
    cases.append(fit_case(joint,'matched_full_portfolio',{'v7':w,'equal_long':equal,
        'long_sleeve':long,'short_sleeve':short},meta,a.out,
        '160 observed daily-return dates from the same 252-session window. All 400 holdings covered; same PCA basis for universe, equal-long and v7. Conditional on joint observability; not a complete 252-day estimate.'))
    omissions=pd.DataFrame({'Date':train.index.difference(joint.index).strftime('%Y-%m-%d')})
    omissions.to_csv(a.out/'omitted_joint_dates.csv',index=False)
    long_ret=joint.reindex(columns=w.index).to_numpy()@w.clip(lower=0).to_numpy()
    short_ret=joint.reindex(columns=w.index).to_numpy()@w.clip(upper=0).to_numpy()
    long_var=float(np.var(long_ret,ddof=1));short_var=float(np.var(short_ret,ddof=1))
    cross=float(2*np.cov(long_ret,short_ret,ddof=1)[0,1])
    total=float(np.var(long_ret+short_ret,ddof=1))
    np.testing.assert_allclose(total,long_var+short_var+cross,rtol=1e-12)
    output={'asof_after_close':str(ASOF.date()),'window_sessions':LOOKBACK,
        'raw_rows':audit['raw_rows'],'cache_rows':len(bars),'cache_start':str(bars.Date.min().date()),'cache_end':str(bars.Date.max().date()),
        'universe_source':'Stored v7 ranks at signal date, excluding then-current supervision; Universe0 snapshot NOT used',
        'base_universe':len(universe),'marketwide_closed_dates':closed,'primary_coverage_gross':float(covered.abs().sum()),
        'missing_252_holdings':table_records(exclusion.loc[exclusion.Weight.ne(0)]),
        'joint_omitted_dates':len(omissions),'joint_kept_date_fraction':len(joint)/LOOKBACK,
        'sleeve_reconciliation':{'long_variance':long_var,'short_variance':short_var,'twice_long_short_covariance':cross,'net_variance':total},
        'cases':cases,'return_type':'split-adjusted close-to-close simple price returns, no dividends',
        'no_clipping':True,'no_return_imputation':True,'no_stock_standardization':True,'formal_test_used':False,
        'scope':'Historical descriptive risk diagnostics with weights fixed at 2021-12-01. Not a dynamic-strategy backtest or evidence of forecast skill.',
        'metadata_caveat':'Company/sector labels are the supplied later snapshot for display only, not historical predictors or selection.',
        'source_hashes':EXPECTED,'source_commit':'5959badfe4703284ba5fbaf18a3ea626581a7d18',
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__},
        'method_reference':'https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html'}
    save(a.out/'results.json',output)
    print('COMPLETE',a.out,flush=True)


if __name__=='__main__':main()
