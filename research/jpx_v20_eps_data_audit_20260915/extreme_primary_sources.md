# Extreme EPS cases: primary-document review

This is a bounded source check of E01–E10, dated2026-09-15. A confirmed cumulative figure is not automatically a confirmed standalone quarterly EPS. Later reports and corrections are documentary evidence only and must not be backfilled before their publication.

| Case | Company | Event date | Status |
|---|---|---|---|
| E01 | Okamura (7994) | 2021-08-04 | current_and_year_base_directly_confirmed |
| E02 | Hitachi Construction Machinery (6305) | 2021-10-26 | cumulative_confirmed_standalone_mismatch_and_near_zero_base_confirmed |
| E03 | Remixpoint (3825) | 2019-11-14 | all_four_cumulative_inputs_confirmed_standalone_pending |
| E04 | Hokkan Holdings (5902) | 2021-08-06 | current_and_year_base_directly_confirmed_search_index_document_text |
| E05 | Chugoku Marine Paints (4617) | 2020-01-31 | material_standalone_eps_mismatch_confirmed |
| E06 | Yamada Consulting Group (4792) | 2020-08-06 | current_and_year_base_directly_confirmed |
| E07 | Yoshimura Food Holdings (2884) | 2021-07-15 | later_material_restatement_confirmed_original_primary_pdf_unavailable |
| E08 | FreeBit (3843) | 2020-12-11 | all_four_cumulative_inputs_confirmed_standalone_pending |
| E09 | Daishinku (6962) | 2019-02-12 | current_9m_and_both_year_base_cumulative_confirmed_current_h1_pending |
| E10 | Santec (6777) | 2019-05-10 | current_standalone_mismatch_confirmed_prior_cumulative_confirmed |

The clearest data-definition mismatch is E05: directly disclosed prior/current standalone EPS .12 and13.39 differ from cumulative subtraction .01 and13.31. E07 is a separate, later-restatement issue. E01,E04,E06 show that some near-zero denominators are actually disclosed Q1 EPS, not splitting artifacts. E02 also explicitly reports priorQ2EPS .04.

## E01 — Okamura

Near-zero prior EPS0.01 was explicitly disclosed for a first quarter; it is not produced by subtracting cumulative EPS. Weighted share base changed across years. No stock-split cause established.

Original model event date: 2021-08-04.

- [Primary source 1](https://www2.jpx.co.jp/disc/79940/140120210803477498.pdf) — document date: 2021-08-04; available by event date: Yes. PDF page1: current Apr-Jun EPS33.04, prior Apr-Jun EPS0.01; attributable profit3294m vs1m. Page2 weighted-average shares99711729 vs110141071.

Unresolved: Previous sequential quarter EPS61.34 has not been independently reconciled to primary documents in this sub-review.

## E02 — Hitachi Construction Machinery

Prior Q2 .04 is independently disclosed, so near-zero base is real at reported precision. Current cumulative subtraction differs by.01 from separately reported quarterlyEPS; cannot infer its entire extreme YoY is a splitting bug.

Original model event date: 2021-10-26.

- [Primary source 1](https://www.hitachicm.com/content/dam/hitachicm/global/en/ir/library/results/docs/20211026-HCM-Financial-E.pdf) — document date: 2021-10-26; available by event date: Yes. Page1 six-month EPS149.96 vs0.99, attributable profit31889m vs211m.
- [Primary source 2](https://www.hitachicm.com/content/dam/hitachicm/global/en/ir/library/securities-report/docs/Annual-Securities-Report-57th-term.pdf) — document date: 2021 annual securities report; exact issue date not checked; available by event date: Not established. Printed page122: prior year quarterly basic EPS Q1 .95,Q2 .04,Q3 13.71,Q4 33.92; cumulativeQ2 .99.
- [Primary source 3](https://www.hitachicm.com/content/dam/hitachicm/global/en/ir/library/securities-report/docs/Annual-Securities-Report-58th-term.pdf) — document date: 2022 annual securities report; approval2022-06-28; available by event date: No. Printed page129: current year cumulativeQ1 EPS33.51,Q2 149.96, but independently stated standaloneQ2 EPS116.44. Naive difference is116.45.

Unresolved: Exact publication time of standalone Q2 EPS116.44 before annual report not established.

## E03 — Remixpoint

Cumulative input values fully matched to primary releases: current-69.66 minus(-7.61)=-62.05; prior6.87-6.91=-.04. This validates input transcription and arithmetic, but separately reported single-quarterEPS still not checked.

Original model event date: 2019-11-14.

- [Primary source 1](https://www2.jpx.co.jp/disc/38250/140120191114427111.pdf) — document date: 2019-11-14; available by event date: Yes. Page1 six-month current EPS-69.66 and prior-yearEPS6.87. Page2 weighted-average shares57891705 and56977149; share issuance count changed.
- [Primary source 2](https://www2.jpx.co.jp/disc/38250/140120190814488344.pdf) — document date: 2019-08-14; available by event date: Yes. Page1 currentQ1EPS-7.61 and priorQ1EPS6.91; page2 cumulative weighted-average shares57370141 and56961318. Together withNov14H1release allfour subtraction inputs are confirmed.

Unresolved: Standalone current and priorQ2EPS not independently confirmed.

## E04 — Hokkan Holdings

Near-zero priorEPS .09 explicitly disclosed forQ1, not cumulative subtraction. Issuer also notes new revenue-recognition standard applied in current year; impact on EPS not quantified here.

Original model event date: 2021-08-06.

- [Primary source 1](https://www2.jpx.co.jp/disc/59020/140120210806480930.pdf) — document date: 2021-08-06; available by event date: Yes. Primary PDF text returned in web search (direct open failed): page1 current first-quarterEPS128.29 and prior first-quarterEPS.09, attributable profit1563m vs1m.

Unresolved: Sequential priorquarterEPS-126.92 and split/restatement history not fully audited.

## E05 — Chugoku Marine Paints

Confirmed cumulative subtraction artifact: model lastyear quarter .01 vs directly disclosed .12; current13.31 vs13.39. At reported standalone precision annualgrowth=(13.39-.12)/abs(.12)=110.583333, compared with1330 using naive values. Weighted-share changes plus rounding matter; .12 is later documentary audit evidence, not automatically a Jan31 tradable replacement.

Original model event date: 2020-01-31.

- [Primary source 1](https://www2.jpx.co.jp/disc/46170/140120191226441901.pdf) — document date: 2020-01-31; available by event date: Yes. Original earnings release confirms nine-month EPS28.99 and prior-year-7.89. Text search finds no standalone13.39 or.12.
- [Primary source 2](https://www.cmp.co.jp/IR/2020/2020_f-report_3q.pdf) — document date: 2020-02-12; available by event date: No. PDF page4 (printed1): standalone Oct-Dec EPS2018=.12 and2019=13.39, whereas cumulative values are-7.89 and28.99. PDF page16: buybacks4824300 shares in prior9M and1972100 shares in current9M.

Unresolved: Earlier release of standalone values on or beforeJan31 has not been established.

## E06 — Yamada Consulting Group

Prior EPS .01 explicitly disclosed forQ1, not a cumulative difference. Profit0 million does not establish exact zero profit.

Original model event date: 2020-08-06.

- [Primary source 1](https://www2.jpx.co.jp/disc/47920/140120200806476646.pdf) — document date: 2020-08-06; available by event date: Yes. Page1 current Apr-JunEPS-13.20, priorApr-JunEPS.01. Prior profit displayed0 million under less-than-million truncation convention.

Unresolved: Prior sequentialquarter25.69 and corporate-action history not independently verified.

## E07 — Yoshimura Food Holdings

Later restatement removes the near-zero prior base and changes currentEPS. These2022 values were not knowable at2021 event and must not be backfilled into historical forecasts. Original document found in third-party archive but not used as primary confirmation.

Original model event date: 2021-07-15.

- [Primary source 1](https://www2.jpx.co.jp/disc/28840/140120220511539996.pdf) — document date: 2022-05-11; available by event date: No. Page1 explicitly corrects release originally published2021-07-15. Page2 corrected currentQ1 EPS10.69 and priorQ1 EPS1.06, attributable profit254m and23m. Original localEPS9.84 and.01.

Unresolved: Original issuer/JPX release currently not readable; exact restatement reasons not yet reviewed.

## E08 — FreeBit

Cumulative input values matched: current64.27-9.96=54.31; prior-8.20-(-8.14)=-.06. Standalone EPS equivalence remains unverified.

Original model event date: 2020-12-11.

- [Primary source 1](https://www2.jpx.co.jp/disc/38430/140120201210433457.pdf) — document date: 2020-12-11; available by event date: Yes. Page1 six-monthEPS64.27,current, and-8.20,prior year.
- [Primary source 2](https://www2.jpx.co.jp/disc/38430/140120200911491554.pdf) — document date: 2020-09-11; available by event date: Yes. Primary PDF text returned in search (directopenfailed): currentQ1EPS9.96 and priorQ1EPS-8.14. Together withDec11H1release, allfour inputs confirmed.

Unresolved: Independently stated standalone current/priorQ2EPS not confirmed.

## E09 — Daishinku

The52.40-52.33=.07 base is a difference of reported cumulative EPS. Published correction did not change9M EPS52.40. Disclosed split2016-10-01 predates all2017-2018quarters being compared, so split alone does not explain this .07 denominator.

Original model event date: 2019-02-12.

- [Primary source 1](https://www.kds.info/wp-content/uploads/2018/05/2017.12-correction-of-financial-result-jp-180517.pdf) — document date: 2018-05-17; available by event date: Yes. Corrected and original9M summary both show EPS52.40. Notes disclose five-to-one reverse split effective2016-10-01, comparative EPS retrospectively adjusted as if occurred start of prior fiscalyear.
- [Primary source 2](https://www.kds.info/wp-content/uploads/2018/05/2017.9-correction-of-financial-result-jp-180517.pdf) — document date: 2018-05-17; available by event date: Yes. Corrected H1 summary EPS52.33, prioryear basis uses same5:1 split note.
- [Primary source 3](https://www.kds.info/wp-content/uploads/2019/02/2018.12-financial-result-en-190212.pdf) — document date: 2019-02-12; available by event date: Yes. English official release indexed PDF text confirms nine-month currentEPS-49.33 and prior52.40; currentnetprofit-398m vsprior423m.

Unresolved: CurrentH1EPS8.55 and separately reported singlequarterEPS not directly checked.

## E10 — Santec

Current standaloneEPS mismatch. Prior base .01 is difference of rounded47.33 and47.32; no independently stated priorQ4 EPS confirmed yet. Dataset million-yen rounding hides the underlying tiny profit difference.

Original model event date: 2019-05-10.

- [Primary source 1](https://www.santec.com/dcms_media/other/yuuhou20190620.pdf) — document date: 2019-06-20; available by event date: No. PDF page58/printed55 cumulative currentEPS49.85 and57.35, explicit standaloneQ4EPS7.49 rather than naive7.50. PriorFY profit556597 thousand yen andEPS47.33 shown elsewhere in same report.
- [Primary source 2](https://www.santec.com/dcms_media/other/tanshin3q2017.pdf) — document date: 2018-02-05; available by event date: Yes. Page1 prior-year9M EPS47.32.

Unresolved: PrioryearQ4EPS and availability of7.49 byMay10 have not yet been established.

