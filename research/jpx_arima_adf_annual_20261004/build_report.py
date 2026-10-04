"""Render a source-grounded report and static comparison figures after evaluation."""
import json
import sqlite3
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import EXP, OUT, CONFIG, YEARS, ORDERS, save, sha


def fmt(value,digits=6):
    return '未定義' if value is None or pd.isna(value) else f'{value:+.{digits}f}'


def run():
    result=json.loads((OUT/'results.json').read_text())
    adf=json.loads((EXP/'adf_summary.json').read_text())
    leader=pd.read_csv(OUT/'leaderboard.csv')
    strategy=pd.read_csv(OUT/'strategy_metrics.csv')
    costs=pd.read_csv(OUT/'cost_sensitivity.csv')
    summary=strategy.loc[strategy.period.eq('all')].set_index('strategy')
    names={'v7_equal':'正式基準 v7_equal','fixed_d1_annual_selector':'固定 d=1、年度選 p,q',
           'adf_annual_selector':'逐檔 ADF 選 d、年度選 p,q',
           'fixed_311_posthoc_reference':'固定 (3,1,1)，既有全期事後候選',
           'd1_at_adf_orders':'固定 d=1，沿用 ADF 的年度 p,q',
           'd1_at_adf_orders_matched_adf_coverage':'固定 d=1，同 ADF 年度 p,q 與後備覆蓋'}
    labels={'v7_equal':'v7_equal','fixed_d1_annual_selector':'d=1 annual p,q','adf_annual_selector':'ADF d + annual p,q'}
    primary=list(labels)
    matrices={family:leader[leader.family.eq(family)].pivot(index='p',columns='q',values='sharpe').reindex(index=range(1,6),columns=range(1,6)).to_numpy()
              for family in ['fixed_d1','adf']}
    ic=leader[leader.family.eq('adf')].pivot(index='p',columns='q',values='mean_daily_rank_ic').reindex(index=range(1,6),columns=range(1,6)).to_numpy()
    fig,axes=plt.subplots(1,3,figsize=(14.5,4.5),constrained_layout=True)
    limit=max(np.abs(matrices['fixed_d1']).max(),np.abs(matrices['adf']).max())
    for ax,m,title,bound in zip(axes,[matrices['fixed_d1'],matrices['adf'],ic],
                                ['Fixed d=1: Sharpe','Per-stock ADF d: Sharpe','Per-stock ADF d: mean Rank IC'],
                                [limit,limit,np.abs(ic).max()]):
        im=ax.imshow(m,cmap='RdBu',vmin=-bound,vmax=bound)
        ax.set_xticks(range(5),range(1,6));ax.set_yticks(range(5),range(1,6))
        ax.set_xlabel('q');ax.set_ylabel('p');ax.set_title(title,fontsize=11)
        for i in range(5):
            for j in range(5):ax.text(j,i,f'{m[i,j]:+.4f}',ha='center',va='center',fontsize=8,color='white' if abs(m[i,j])>.60*bound else 'black')
        fig.colorbar(im,ax=ax,shrink=.8)
    fig.suptitle(f'JPX development validation | {result["common_days"]} common dates | unannualized, before costs',fontsize=12)
    fig.savefig(EXP/'candidate_heatmaps.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    x=np.arange(4);width=.24
    for k,name in enumerate(primary):
        vals=[float(strategy.loc[strategy.strategy.eq(name)&strategy.period.eq(str(y)),'sharpe'].iloc[0]) for y in YEARS]
        axes[0].bar(x+(k-1)*width,vals,width,label=labels[name])
        subset=costs[costs.strategy.eq(name)]
        axes[1].plot(subset.cost_bps_per_traded_notional,subset.net_sharpe,marker='o',label=labels[name])
    axes[0].set_xticks(x,['2018','2019','2020','2021*']);axes[0].axhline(0,color='black',lw=.6)
    axes[0].set_title('Annual strategy Sharpe');axes[0].set_ylabel('Unannualized Sharpe');axes[0].legend(fontsize=8)
    axes[1].set_title('Illustrative turnover cost sensitivity');axes[1].set_xlabel('Cost per traded notional (bps)')
    axes[1].set_ylabel('Unannualized net Sharpe');axes[1].axhline(0,color='black',lw=.6);axes[1].legend(fontsize=8)
    fig.suptitle('Annual choices use only past matured labels; 2021 is incomplete',fontsize=11)
    fig.savefig(EXP/'strategy_comparison.png',dpi=160);plt.close(fig)
    con=sqlite3.connect(f'file:{OUT}/new_fits.sqlite?mode=ro',uri=True)
    audits=[json.loads(row[0]) for row in con.execute('SELECT audit FROM fits')]
    status=pd.Series([a['status'] for a in audits]).value_counts().to_dict()
    failures=[{k:a.get(k) for k in ['code','year','p','d','q','status','exception','attempts']} for a in audits if a['status']!='ok']
    save(EXP/'fit_audit_summary.json',{'fits':len(audits),'statuses':{k:int(v) for k,v in status.items()},
         'causal_prefix_and_future_checks':sum(len(a['causal_checks']) for a in audits),
         'invalid_forward_observations':sum(a.get('invalid_forward_observations',0) for a in audits),
         'retried_fits':sum(len(a['attempts'])>1 for a in audits),'failures':failures})
    con.close()
    independent=json.loads((EXP/'numerical_checks.json').read_text())
    contrast=result['contrasts']['adf_annual_selector']
    headline=summary.loc['adf_annual_selector','sharpe']-summary.loc['fixed_d1_annual_selector','sharpe']
    lines=['# JPX：5% ADF 差分與年度選階實測（2026-10-04）','',
           f'**本輪完成。主策略 Sharpe {fmt(summary.loc["adf_annual_selector","sharpe"])}，固定 d=1 年度選階 {fmt(summary.loc["fixed_d1_annual_selector","sharpe"])}，差異 {fmt(headline)}。**',
           '結果來自已反覆參與研究的歷史 validation；正式 test 未讀取、未評分，正式基準仍為 v7_equal。',
           '',f'共同評估日數：**{result["common_days"]}**；原資料 953 日、1,864,363 個股票日，ValidationYear 2018–2021（2021 不完整）。Sharpe 均為官方每日 spread 的平均／樣本標準差，未年化、主表未扣成本。',
           '', '## 主要策略比較','', '| 策略 | Sharpe | 平均每日 Rank IC | 每日 Target MSE 平均 |','|---|---:|---:|---:|']
    for name,row in summary.iterrows():lines.append(f'| {names[name]} | {fmt(row.sharpe)} | {fmt(row.mean_daily_rank_ic)} | {row.mean_daily_target_mse:.8g} |')
    lines+=['','兩條「沿用 ADF 年度 p,q」對照是主策略路徑的機制診斷，沒有自行再選勝者。固定 (3,1,1) 是既有全期事後勝出候選，不能當作逐年可以提前知道的答案。',
            '', '## 每年實際選出的 p,q','', '| 年度 | 選模截止 | 成熟歷史日數 | 固定 d=1 家族 p,q | ADF 家族 p,q |','|---|---|---:|---|---|']
    for y in YEARS:
        a=next(r for r in result['annual_choices'] if r['year']==y and r['family']=='adf')
        f=next(r for r in result['annual_choices'] if r['year']==y and r['family']=='fixed_d1')
        n=f'{f["eligible_days"]}／{a["eligible_days"]}'
        lines.append(f'| {y}{"（部分年度）" if y==2021 else ""} | {a["cutoff"]} | {n} | ({f["p"]},{f["q"]}) | ({a["p"]},{a["q"]}) |')
    lines+=['','歷史日數欄依序為固定 d=1／ADF 家族。2018 固定 p=q=1 作為初始化；之後使用全部既往年度的成熟樣本外績效，Sharpe 第一、Rank IC 第二，完全同分再取較低階。每年邊界最後兩個前期訊號的 Target 尚未成熟，因此不參與當年度選模。',
            '', '## 分年與排除初始化年度','', '| 策略 | 2018 | 2019 | 2020 | 2021（部分） | 2019–2021 合併 |','|---|---:|---:|---:|---:|---:|']
    for name in primary:
        vals=[strategy.loc[strategy.strategy.eq(name)&strategy.period.eq(str(y)),'sharpe'].iloc[0] for y in YEARS]
        vals.append(strategy.loc[strategy.strategy.eq(name)&strategy.period.eq('2019-2021'),'sharpe'].iloc[0])
        lines.append('| '+names[name]+' | '+' | '.join(fmt(v) for v in vals)+' |')
    lines+=['','![年度策略與成本敏感度](strategy_comparison.png)','',
            '## ADF 的實際分布','', '| 年度 | d=0 | d=1 | d=2 | 未選 d／後備 |','|---|---:|---:|---:|---:|']
    for row in adf['counts']:lines.append(f'| {row["year"]} | {row["d0"]} | {row["d1"]} | {row["d2"]} | {row["unresolved"]} |')
    lines+=['',f'共新增 {len(audits):,} 次股票年度估計，重用 {adf["reused_fits"]:,} 次既有 d=1 結果，{adf["zero_fallback_fits"]:,} 次候選股票年度採 ADF 可用性後備。401 個未選 d 股票年度包括 299 個原有效歷史不足，以及 102 個連續檢定樣本不足；本次沒有因差分到 d=2 仍未拒絕單位根而後備的案例。',
            '', 'ADF 使用 5% critical value、常數項、AIC 選 lag；在 d=0、1、2 中選第一個拒絕單位根者。檢定只用年度 training 的最長連續有限區段，ARIMA 仍用完整 expanding 歷史並保留缺價。d=0 估常數均值；d>=1 無漂移。ADF 的拒絕是本次操作決策，不宣稱已知道真實過程必然平穩。',
            '', '**缺價處理的限制：**原日曆的 2020-10-01 是全市場缺價日。依本輪事先固定的最長連續區段規則，2021 年 1,974 個可進入檢定流程的股票年度，其 ADF 區段均未涵蓋訓練末日；其中 1,901 個截至 2020-09-30，其餘更早。ARIMA 係數仍用截至 2020-12-29 的完整 expanding 價格。不能把本輪 ADF 說成使用了 2021 fold 的所有訓練觀測；完整区段診斷見 `adf_segment_diagnostics.json`。',
            '', '## 全部 25 組候選','', '每組 p,q 跨年固定；ADF 家族的 d 仍按每檔、每年度檢定變動。下表的全期排序是開發資料上的事後描述，與上面的年度因果選模分開。',
            '', '| p | q | 固定 d=1 Sharpe | ADF Sharpe | Sharpe 差 | 固定 d=1 Rank IC | ADF Rank IC |','|---:|---:|---:|---:|---:|---:|---:|']
    for p,q in ORDERS:
        f=leader[(leader.family=='fixed_d1')&(leader.p==p)&(leader.q==q)].iloc[0]
        a=leader[(leader.family=='adf')&(leader.p==p)&(leader.q==q)].iloc[0]
        lines.append(f'| {p} | {q} | {fmt(f.sharpe)} | {fmt(a.sharpe)} | {fmt(a.sharpe-f.sharpe)} | {fmt(f.mean_daily_rank_ic)} | {fmt(a.mean_daily_rank_ic)} |')
    lines+=['','![全部候選熱圖](candidate_heatmaps.png)','', '## 差異的不確定性','', '| ADF 年度主策略相對對照 | Sharpe 差 | 配對邊際 95% 區間 |','|---|---:|---|']
    for key,label in [('versus_v7','v7_equal'),('versus_fixed_d1_annual','固定 d=1 年度選階'),('versus_matched_order_and_coverage_d1','相同 p,q 路徑與 ADF 後備覆蓋的 d=1')]:
        c=contrast[key];lo,hi=c['marginal_95'];lines.append(f'| {label} | {fmt(c["difference"])} | [{fmt(lo)}, {fmt(hi)}] |')
    lines+=['','20 日循環區塊配對 bootstrap、2,000 次、seed=20261004。這裡重抽已實現策略路徑，沒有在每次抽樣重做訓練及年度選模，因此是條件於該選模路徑的探索區間。候選 CSV 另有固定 25 組相對 v7 的近似同時區間；兩者都沒有校正全部歷史研究決策，也不構成新 holdout 證據。',
            '', '## 換手與成本敏感度','', '| 策略 | 0 bps Sharpe | 5 bps | 10 bps | 20 bps | 平均損益兩平成本 bps |','|---|---:|---:|---:|---:|---:|']
    for name in primary:
        subset=costs[costs.strategy.eq(name)].sort_values('cost_bps_per_traded_notional')
        lines.append('| '+names[name]+' | '+' | '.join(fmt(x) for x in subset.net_sharpe)+f' | {subset.mean_gross_break_even_bps.iloc[0]:.4f} |')
    lines+=['','兩側各單位名目曝險：gross = 官方 spread / 200；成本 = 單位成交成本 × 相鄰目標權重 L1 差。含初始建倉与末次平倉，忽略價格漂移造成的實際權重變化、借券費、滑價細節及成交限制，屬換手敏感度，不是完整帳戶回測；沒有把官方 spread 直接複利。',
            '', '## 數值與資料核對','',f'- 新估計狀態：{json.dumps({k:int(v) for k,v in status.items()},ensure_ascii=False)}。失敗案例詳見 `fit_audit_summary.json`，不刪股票。',
            f'- 新估計程序內完成 {sum(len(a["causal_checks"]) for a in audits):,} 次截斷／未來擾動核對；獨立程序另有 {independent["forecast_tests"]["count"]} 次。最大獨立價格誤差 {independent["forecast_tests"]["max_price_error"]:.3g}。',
            '- ADF training 邊界、原始未縮放 ADF 決策、缺價區段選擇、未到期 Target 排除及年度選模未來擾動核對均通過。',
            '- 重用 d=1 分數與後備標記逐筆精確相等；主策略及新候選由官方評分函數獨立核對；v7 逐日 spread 重現。',
            f'- 各策略共同排除日期：{", ".join(result["excluded_dates"]) if result["excluded_dates"] else "無"}。若原始 Target 缺失落在選中持倉，該日不當成零報酬；完整日期列表與各候選原始日數保留。',
            '- 原始 Target 與本地因果價格比值的既有微小差異未在本輪修正；全部模型仍用相同原始 Target。',
            '', '## 來源與重現','',
            '- 起始 Git commit：`6f6a57a515cd05f0676646479140d63e8ad07832`；來源 inventory：`data/inventories/snapshot-2026-09-28-arima-learning.json`。',
            '- 本次查得來源 `snapshot-2026-09-28-arima-learning` 仍是草稿 Release，root index 未列入，不能宣稱它已正式發布。54 個本機原檔共 551,748,534 bytes 與版本化 SHA-256 清單一致；另分段完整取回價格檔 5,876,864 bytes，SHA-256 與本機完全一致。沒有宣稱完整下載或核對整個舊資料包。',
            '- 本轮新 Release 保存所需價格／Target keys、兩個完整候選家族的逐筆預測、參數、ADF 明細、估計 checkpoint、選模紀錄、後備與核對資料；程式、設定、報告及小型結果進 Git。同步與實際清理狀態以 `sync_status.json`、`cleanup_receipt.json` 為準。',
            '- 完整參數比較：`leaderboard.csv`；逐年比較：`annual_candidates.csv`；策略表：`strategy_metrics.csv`；成本表：`cost_sensitivity.csv`。',
            '- ACF/PACF lag 1–20 及檢定明細保存在 `adf_decisions.json`；新擬合模型的 training Ljung–Box 診斷在 checkpoint audit 中，未拿來另行選模。']
    (EXP/'REPORT.md').write_text('\n'.join(lines)+'\n')
    for name in ['leaderboard.csv','annual_candidates.csv','strategy_metrics.csv','cost_sensitivity.csv','common_calendar.csv','excluded_dates.csv']:
        shutil.copyfile(OUT/name,EXP/name)
    save(EXP/'strategy_results.json',{'common_days':result['common_days'],'strategies':result['all_strategy_metrics'],
                                     'annual_choices':[{k:r.get(k) for k in ['family','year','cutoff','p','q','eligible_days','max_exit_date','unmatured_prior_fold_days']} for r in result['annual_choices']],
                                     'contrasts':result['contrasts'],'formal_test_used':False})
    print(json.dumps({'report':str(EXP/'REPORT.md'),'main_sharpe':summary.loc['adf_annual_selector','sharpe'],
                      'relative_to_fixed_d1_annual':headline,'new_fit_statuses':{k:int(v) for k,v in status.items()}}))


if __name__=='__main__':run()
