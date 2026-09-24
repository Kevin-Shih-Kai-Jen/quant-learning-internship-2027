"""Independent raw-ZIP, source-reference, quarter-key and publication-time audit.

Reads existing caches but never invokes the financial feature builder.
Writes only this experiment's source/timing audit artifacts.
"""
from pathlib import Path
import hashlib
import io
import json
import pickle
import re
import zipfile

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent
B = R.parent
ARCHIVE = Path('/Users/coolguy/Desktop/JPX_data/modelling/JPX_data.zip')
MEMBER = 'JPX_data/raw/train_files/financials.csv'
QNUM = {'1Q': 1, '2Q': 2, '3Q': 3, 'FY': 4}
DATE_FIELDS = ['Date', 'DisclosedDate', 'CurrentFiscalYearStartDate',
               'CurrentPeriodEndDate', 'CurrentFiscalYearEndDate']
NUMERIC_FIELDS = ['SecuritiesCode', 'EarningsPerShare', 'Profit',
                  'AverageNumberOfShares', 'DisclosedUnixTime', 'DisclosureNumber']
TEXT_FIELDS = ['TypeOfDocument', 'TypeOfCurrentPeriod', 'DisclosedTime']


def finite_number(v):
    n = pd.to_numeric(v, errors='coerce')
    return float(n) if pd.notna(n) and np.isfinite(n) else np.nan


def same_number(a, b):
    x, y = finite_number(a), finite_number(b)
    return bool((np.isnan(x) and np.isnan(y)) or x == y)


def basis(document):
    return re.sub(r'^(1Q|2Q|3Q|FY|OtherPeriod)FinancialStatements_', '', document)


def effective(record):
    return max(pd.Timestamp(record['Date']), pd.Timestamp(record['DisclosedDate'])) + pd.Timedelta(record['DisclosedTime'])


def main():
    cases = pd.read_pickle(R / 'selected_cases.pkl')
    cache = pd.read_pickle(B / 'jpx_v14_filtered_forecast_events_20260914/financial_records.pkl').set_index('SourceRow')
    exported = pd.read_csv(R / 'selected_source_records.csv', dtype=str).set_index('SourceRow')
    with zipfile.ZipFile(ARCHIVE) as archive:
        payload = archive.read(MEMBER)
    raw = pd.read_csv(io.BytesIO(payload), dtype=str)
    raw.index.name = 'SourceRow'
    ids = sorted({int(v) for c in cases.itertuples() for refs in [c.CurrentRefs, c.PreviousRefs, c.YearRefs] for v in refs})
    assert ids == sorted(exported.index.astype(int).tolist())
    with (B / 'jpx_v8_soft_rank_20260912/inputs.pkl').open('rb') as h:
        _, _, groups, _ = pickle.load(h)
    market = pd.DatetimeIndex(groups).sort_values().difference(pd.DatetimeIndex(['2020-10-01']))
    closes = market + pd.Timedelta(hours=15)

    def availability(timestamp):
        eligible = np.flatnonzero(closes > timestamp)
        return (int(eligible[0]), market[eligible[0]]) if len(eligible) else (len(market), pd.NaT)

    checks, failures = [], []
    for source_id in ids:
        src = raw.loc[source_id]
        cached = cache.loc[source_id]
        exp = exported.loc[str(source_id)]
        for field in NUMERIC_FIELDS + DATE_FIELDS + TEXT_FIELDS:
            if field in NUMERIC_FIELDS:
                cache_ok, export_ok = same_number(src[field], cached[field]), same_number(src[field], exp[field])
            elif field in DATE_FIELDS:
                cache_ok = pd.Timestamp(src[field]) == pd.Timestamp(cached[field])
                export_ok = pd.Timestamp(src[field]) == pd.Timestamp(exp[field])
            else:
                cache_ok, export_ok = str(src[field]) == str(cached[field]), str(src[field]) == str(exp[field])
            row = dict(SourceRow=source_id, Field=field, RawValue=src[field], CacheValue=str(cached[field]),
                       ExportValue=str(exp[field]), CacheMatches=bool(cache_ok), ExportMatches=bool(export_ok))
            checks.append(row)
            if not (cache_ok and export_ok): failures.append(row)
        ts = effective(src)
        pos, day = availability(ts)
        assert cached['Basis'] == basis(src['TypeOfDocument'])
        assert cached['Code'] == int(float(src['SecuritiesCode']))
        assert cached['EffectiveTimestamp'] == ts
        assert cached['AvailablePosition'] == pos and cached['AvailableDate'] == day

    # Prepare all raw regular statements for the chosen companies.  Independently
    # confirm the supplied references are the last known row for their quarter.
    relevant = raw[pd.to_numeric(raw.SecuritiesCode, errors='coerce').isin(cases.Code)].copy()
    relevant = relevant[relevant.TypeOfDocument.str.match(r'^(1Q|2Q|3Q|FY)FinancialStatements_', na=False)].copy()
    relevant['Code'] = pd.to_numeric(relevant.SecuritiesCode).astype(int)
    relevant['Basis'] = relevant.TypeOfDocument.map(basis)
    relevant['EffectiveTimestamp'] = relevant.apply(effective, axis=1)
    relevant['FiscalStart'] = pd.to_datetime(relevant.CurrentFiscalYearStartDate)
    relevant['FiscalEnd'] = pd.to_datetime(relevant.CurrentFiscalYearEndDate)
    relevant['Quarter'] = relevant.TypeOfCurrentPeriod.map(QNUM)
    relevant['PeriodEnd'] = pd.to_datetime(relevant.CurrentPeriodEndDate)
    relevant['SourceRow'] = relevant.index
    relevant.index.name = None
    relevant['DisclosureOrder'] = pd.to_numeric(relevant.DisclosureNumber)
    relevant = relevant.sort_values(['EffectiveTimestamp', 'DisclosureOrder', 'SourceRow'], kind='stable')
    detail = []
    case_rows = []
    for case in cases.itertuples():
        case_raw = raw.loc[case.SourceRow]
        case_ts = effective(case_raw)
        case_pos, case_day = availability(case_ts)
        assert case_ts == case.EffectiveTimestamp and case_day == case.AvailableDate
        assert pd.Timestamp(case_raw.CurrentFiscalYearStartDate) == case.FiscalStart
        assert pd.Timestamp(case_raw.CurrentFiscalYearEndDate) == case.FiscalEnd
        assert QNUM[case_raw.TypeOfCurrentPeriod] == case.FiscalQuarter
        case_ref_ok = True
        for role, refs, q, year_offset in [
            ('current', case.CurrentRefs, case.FiscalQuarter, 0),
            ('previous_quarter', case.PreviousRefs, case.FiscalQuarter - 1 if case.FiscalQuarter > 1 else 4,
             0 if case.FiscalQuarter > 1 else -1),
            ('previous_year', case.YearRefs, case.FiscalQuarter, -1),
        ]:
            expected_start = case.FiscalStart + pd.DateOffset(years=year_offset)
            expected_end = case.FiscalEnd + pd.DateOffset(years=year_offset)
            expected_quarters = {q} if q == 1 else {q - 1, q}
            actual_quarters = set()
            for source_id in refs:
                src = raw.loc[source_id]
                source_q = QNUM[src.TypeOfCurrentPeriod]
                actual_quarters.add(source_q)
                source_ts = effective(src)
                source_pos, source_day = availability(source_ts)
                same_company_basis = int(float(src.SecuritiesCode)) == case.Code and basis(src.TypeOfDocument) == case.Basis
                fiscal_period_correct = (pd.Timestamp(src.CurrentFiscalYearStartDate) == expected_start and
                                         pd.Timestamp(src.CurrentFiscalYearEndDate) == expected_end and
                                         pd.Timestamp(src.CurrentPeriodEndDate) == expected_start + pd.DateOffset(months=3 * source_q) - pd.Timedelta(days=1))
                known = source_ts <= case_ts
                # Prior-period source rows must predate the current release.
                if role != 'current': known = known and source_ts < case_ts
                known_at_signal = source_day <= case_day and source_pos <= case_pos
                candidates = relevant[
                    relevant.Code.eq(case.Code) & relevant.Basis.eq(case.Basis) &
                    relevant.FiscalStart.eq(expected_start) & relevant.FiscalEnd.eq(expected_end) &
                    relevant.Quarter.eq(source_q) & relevant.EffectiveTimestamp.le(case_ts)
                ]
                latest_row = int(candidates.iloc[-1].SourceRow)
                latest_known = latest_row == source_id
                unix_jst = pd.to_datetime(float(src.DisclosedUnixTime), unit='s', utc=True).tz_convert('Asia/Tokyo').tz_localize(None)
                declared_timestamp = pd.Timestamp(src.DisclosedDate) + pd.Timedelta(src.DisclosedTime)
                unix_matches = unix_jst == declared_timestamp
                row_ok = same_company_basis and fiscal_period_correct and known and known_at_signal and latest_known and unix_matches
                case_ref_ok &= row_ok
                detail.append(dict(CaseID=case.CaseID, Role=role, SourceRow=source_id, SourceQuarter=source_q,
                                   ExpectedQuarter=q, SourceEffectiveTimestamp=str(source_ts),
                                   SourceAvailableDate=str(source_day.date()), CaseEffectiveTimestamp=str(case_ts),
                                   SameCompanyAndBasis=bool(same_company_basis), FiscalPeriodCorrect=bool(fiscal_period_correct),
                                   KnownAtRelease=bool(known), KnownAtSignal=bool(known_at_signal),
                                   LatestKnownRow=latest_row, IsLatestKnown=bool(latest_known),
                                   UnixDisclosureTimestampMatches=bool(unix_matches), Passed=bool(row_ok)))
            assert actual_quarters == expected_quarters
        unix_jst = pd.to_datetime(float(case_raw.DisclosedUnixTime), unit='s', utc=True).tz_convert('Asia/Tokyo').tz_localize(None)
        case_rows.append(dict(CaseID=case.CaseID, Code=case.Code, SampleType=case.SampleType,
                              SourceRow=case.SourceRow, FiscalQuarter=case.FiscalQuarter,
                              EffectiveTimestamp=str(case_ts), AvailableDate=str(case_day.date()),
                              DisclosedUnixJST=str(unix_jst), DateDiffersFromDisclosedDate=case_raw.Date != case_raw.DisclosedDate,
                              AtOrAfterHistoricalClose=case_ts.time() >= pd.Timestamp('15:00').time(),
                              RefOccurrences=len(case.CurrentRefs) + len(case.PreviousRefs) + len(case.YearRefs),
                              AllRawFieldsMatch=not any(f['SourceRow'] in case.CurrentRefs + case.PreviousRefs + case.YearRefs for f in failures),
                              AllReferencesAndPeriodsPass=bool(case_ref_ok),
                              PriorTrainingDayStrictlyEarlier=case.LastTrainingDate < case.AvailableDate))
    synthetic = []
    for stamp, expected in [('2020-09-30 14:59:59', '2020-09-30'), ('2020-09-30 15:00:00', '2020-10-02'),
                            ('2020-10-01 12:00:00', '2020-10-02'), ('2020-10-02 14:59:59', '2020-10-02'),
                            ('2020-10-02 15:00:00', '2020-10-05')]:
        _, actual = availability(pd.Timestamp(stamp))
        passed = actual == pd.Timestamp(expected)
        assert passed
        synthetic.append(dict(EffectiveTimestamp=stamp, ExpectedAvailableDate=expected, ActualAvailableDate=str(actual.date()), Passed=passed))
    pd.DataFrame(checks).to_csv(R / 'source_raw_field_checks.csv', index=False)
    pd.DataFrame(detail).to_csv(R / 'source_reference_timing_checks.csv', index=False)
    pd.DataFrame(case_rows).to_csv(R / 'source_timing_cases.csv', index=False)
    result = dict(
        passed=not failures and all(r['Passed'] for r in detail) and all(r['PriorTrainingDayStrictlyEarlier'] for r in case_rows),
        raw_archive=str(ARCHIVE), raw_member=MEMBER, raw_member_sha256=hashlib.sha256(payload).hexdigest(),
        selected_cases=len(cases), unique_source_rows=len(ids), source_field_checks=len(checks),
        fields_checked=NUMERIC_FIELDS + DATE_FIELDS + TEXT_FIELDS,
        raw_cache_and_export_mismatches=failures, reference_occurrences_checked=len(detail),
        all_selected_cached_derived_timestamps_basis_and_market_positions_match=True,
        all_references_are_latest_known_regular_statement_rows=all(r['IsLatestKnown'] for r in detail),
        selected_reference_checks_failed=[r for r in detail if not r['Passed']],
        selected_at_or_after_close_cases=sum(r['AtOrAfterHistoricalClose'] for r in case_rows),
        selected_date_disclosed_date_mismatch_cases=sum(r['DateDiffersFromDisclosedDate'] for r in case_rows),
        market_excludes_2020_10_01=pd.Timestamp('2020-10-01') not in market,
        synthetic_close_and_market_closure_tests=synthetic,
        limitations=[
            'This checks the 20 selected cases and 87 underlying raw dataset rows, not the entire financial dataset.',
            'Issuer-original figures and true dissemination times require independent issuer documents; raw ZIP matching alone does not establish their correctness.',
            'The availability convention uses the first recorded trading close strictly after publication (15:00 JST for this historical sample), conservatively deferring exact-close disclosures.',
            'The trading calendar is independently mapped from existing price-date groups with 2020-10-01 explicitly excluded; this is not a separate official exchange-calendar audit.',
            'Generic NumericalCorrection rows remain excluded by the existing feature design; latest-known checks cover regular statements with an explicit accounting basis.',
            'This subaudit does not endorse subtracting cumulative EPS across changing share denominators or split bases; EPS economic comparability is audited separately.',
        ],
    )
    (R / 'source_timing_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    assert result['passed']


if __name__ == '__main__':
    main()
