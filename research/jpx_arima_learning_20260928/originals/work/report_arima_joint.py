"""Deliver the four-arm, two-order Target-MSE experiment without duplicating inputs."""
import os,json,zipfile,shutil,hashlib
import numpy as np
import pandas as pd
from run_arima_joint import ROOT,RUN,SOURCE,PS,sha
OUT=ROOT.parent/'outputs'
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'mplcache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
NAMES={'A':'原始 ARIMA（MLE）','B':'原始 ARIMA＋事後 ACF','C':'Target MSE 訓練 ARIMA','D':'Target MSE 共同訓練 ARIMA＋ACF'}
def f(v):return f'{v:+.6f}'

def main():
    s=json.loads((RUN/'results.json').read_text());assert s['complete'] and s['optimizer_jobs']==32000
    m=pd.read_csv(RUN/'metrics.csv').set_index(['p','Group']);annual=pd.read_csv(RUN/'annual.csv');contrasts=pd.read_csv(RUN/'contrasts.csv')
    colors=['#2a5270','#7ba8b4','#b67442','#8a5680'];plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'figure.facecolor':'#f8fafc'})
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    for ax,p in zip(axes,PS):
        vals=[m.loc[(p,g),'sharpe'] for g in NAMES];bars=ax.bar(list(NAMES),vals,color=colors);ax.bar_label(bars,fmt='%+.4f',padding=3,fontsize=9);ax.axhline(0,color='#78818c',lw=.8);ax.set_title(f'ARIMA({p},1,1)',loc='left',fontweight='bold');ax.set_ylabel('Daily spread Sharpe');ax.margins(y=.2);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('JPX | Four training designs on the same validation dates',x=.06,ha='left',fontsize=15,fontweight='bold')
    fig.text(.06,.025,'A: original   B: post-fit ACF   C: Target-MSE ARIMA   D: jointly trained ACF\nUnannualized, before costs; formal test untouched.',fontsize=9)
    fig.tight_layout(rect=[.02,.10,.99,.93]);fig.savefig(OUT/'JPX-ARIMA-joint-comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.6))
    for ax,p in zip(axes,PS):
        for g,color in zip(NAMES,colors):
            rows=annual.loc[annual.p.eq(p)&annual.Group.eq(g)].sort_values('Year');ax.plot(rows.Year,rows.sharpe,'o-',color=color,label=g)
        ax.set_xticks([2018,2019,2020,2021],['2018','2019','2020','2021*']);ax.set_title(f'ARIMA({p},1,1)',loc='left',fontweight='bold');ax.axhline(0,color='#78818c',lw=.8);ax.legend(ncol=4,frameon=False);ax.set_ylabel('Daily spread Sharpe')
    fig.suptitle('JPX | Annual validation comparison',x=.06,ha='left',fontsize=15,fontweight='bold');fig.text(.06,.025,'*2021 is partial. A/B use MLE; C/D use the same Target-MSE definition.',fontsize=9);fig.tight_layout(rect=[.02,.07,.99,.91]);fig.savefig(OUT/'JPX-ARIMA-joint-annual.png',dpi=160);plt.close(fig)
    fullrows='\n'.join(f'| {g}：{NAMES[g]} | {f(m.loc[(3,g),"sharpe"])} | {f(m.loc[(5,g),"sharpe"])} |' for g in NAMES)
    metricrows='\n'.join(f'| ({p},1,1) | {g} | {f(m.loc[(p,g),"mean_rank_ic"])} | {m.loc[(p,g),"TargetMSE"]:.9f} |' for p in PS for g in NAMES)
    annualrows=[]
    for p in PS:
        for y in [2018,2019,2020,2021]:
            a=annual.loc[annual.p.eq(p)&annual.Year.eq(y)].set_index('Group');annualrows.append(f'| ({p},1,1) | {y} | '+' | '.join(f(a.loc[g,'sharpe']) for g in NAMES)+' |')
    cirows='\n'.join(f'| ({int(r.p)},1,1) | {r.Contrast} | {f(r.SharpeDifference)} | [{f(r.Marginal95Low)}, {f(r.Marginal95High)}] | [{f(r.Simultaneous95Low)}, {f(r.Simultaneous95High)}] |' for _,r in contrasts.iterrows())
    trs=[];lrs=[];statuses=[]
    for r in s['training_summary']:
        trs.append(f'| ({r["p"]},1,1) | {r["Group"]} | {r["optimized"]:,} | {r["converged"]:,} | {r["training_mse_relative_change"]:+.2%} | {r["median_function_calls"]:.0f} |')
        statuses.append(f'- ({r["p"]},1,1) {r["Group"]}：'+ '、'.join(f'{k}={v}' for k,v in r['statuses'].items()))
        if r['Group']=='D':lrs.append(f'| ({r["p"]},1,1) | {r["median_lambda1"]:.6f} | {r["median_lambda2"]:.6f} | {r["lambda1_at_zero"]}/{r["optimized"]} | {r["lambda2_at_zero"]}/{r["optimized"]} | {r["lambda1_at_one"]}/{r["optimized"]} | {r["lambda2_at_one"]}/{r["optimized"]} |')
    conclusions=[]
    for p in PS:
        delta=float(m.loc[(p,'D'),'sharpe']-m.loc[(p,'C'),'sharpe']);c=contrasts.loc[contrasts.p.eq(p)&contrasts.Contrast.eq('D-C')].iloc[0]
        interval='仍包含零，沒有足夠證據判定優劣' if c.Simultaneous95Low<=0<=c.Simultaneous95High else ('完全高於零，本次固定對比下呈現正差異' if c.Simultaneous95Low>0 else '完全低於零，本次固定對比下呈現負差異')
        conclusions.append(f'- **ARIMA({p},1,1)：D 相對 C 的 Sharpe 差為 {delta:+.6f}**；六項對比的近似同時區間{interval}。')
    auditrows='\n'.join(f'| ({r["p"]},1,1) | {r["Group"]} | {r["FallbackFraction"]:.2%} | {r["SelectedFallbackStocks"]:,} |' for r in s['prediction_summary'])
    report=f'''# JPX：ARIMA 與 ACF 共同訓練四組對照

## 完整結果表

依使用者要求，將有無 ACF、原訓練方式及 Target MSE 訓練放在同一張表。A、B 直接沿用前輪預測，沒有重訓；C、D 是本輪實際訓練與回測的新版本。所有指標在共同可評分日期上重新計算，若有排除日期，A、B 指標也會隨比較日期而改變。兩個 ARIMA 階數分開比較，没有混成一個投資組合。

| 設計 | ARIMA(3,1,1) Sharpe | ARIMA(5,1,1) Sharpe |
|---|---:|---:|
{fullrows}

全部使用相同 **{s['common_days']} 個驗證日**，未年化、未扣交易成本。v7 同期 Sharpe：{f(m.loc[(0,'v7'),'sharpe'])}。正式 test 未使用；本輪沿用已多次研究的驗證資料，所以不是獨立確認模型作用的證據。

![完整四組比較](JPX-ARIMA-joint-comparison.png)

## 最重要的判讀

{chr(10).join(conclusions)}

- **C−A**：更換 ARIMA 訓練目標與最佳化流程的效果。
- **D−C**：同樣 Target MSE 下，加入共同訓練 ACF 修正部分的增量效果，是本輪主要對比。
- **D−B**：整個新流程對比事後 ACF 的變化；訓練目標與修正強度都改了，不能將差異全部歸因於共同訓練。

這些對比衡量同一歷史驗證流程下的預測表現，不是價格受某種 shock 影響的因果證明。模型保留為研究候選，沒有自動替換原版本。

## 各組究竟訓練了什麼

| 組別 | ARIMA 參數 | ACF 與修正強度 | 損失函數 |
|---|---|---|---|
| A | 原最大概似估計 | 無 ACF 修正 | 原價格模型的 Gaussian 最大概似 |
| B | 固定 A 的參數 | 殘差 ACF lag1、2，強度固定1 | 不再訓練 |
| C | 從 A 出發重新最佳化 AR／MA 係數 | 無 ACF 修正 | JPX Target MSE |
| D | 從 A 出發重新最佳化 AR／MA 係數 | 每個候選重新算殘差 ACF，同時學 λ₁、λ₂ | 與 C 相同的 JPX Target MSE |

Target MSE = 平均[(原始 Target[t] − 預測報酬[t])²]，其中預測報酬[t]=P̂[t+2|t]/P̂[t+1|t]−1。不是預測價格的 MSE，也不是要求兩組的 MSE 數字相同。最佳化時乘上 10⁶ 改善數值尺度，不改變最小值的位置。

對 D，e[t] 是目前候選 ARIMA 的原始一步價格殘差（實際−預測），ρₕ 是它的年度訓練 ACF：

u₁ = λ₁ρ₁e[t]，u₂ = λ₂ρ₂e[t]，0≤λ₁,λ₂≤1。

修正轉回價格單位後：P̂*₁=P̂₁+u₁；P̂*₂=P̂₂+(1+φ₁+θ₁)u₁+u₂。ARIMA 係數變動時，訓練殘差、ρ₁、ρ₂ 都會重新計算，並與 λ₁、λ₂ 一同影響最終 Target MSE。ACF 本身是殘差的統計量，不是額外兩個自由迴歸係數。

兩組保留原始訓練尺度、innovation variance sigma² 與無漂移設定，只最佳化預測相關的 AR／MA 係數；sigma² 在報酬 MSE 下不是直接可識別的自由尺度，因此未一同訓練。AR 平穩／MA 可逆轉換沿用 statsmodels。

## 排名與 MSE 分開檢查

| ARIMA | 組別 | 平均 Rank IC | 驗證 Target MSE |
|---|---|---:|---:|
{metricrows}

Target MSE 按所有可觀察股票日平均；Rank IC 按每日相關再平均；Sharpe 則來自前後各200檔的加權多空報酬差，三者不是同一件事。MSE 下降不保證排名改善，訓練 MSE 下降更不等於未見資料的績效改善。

## 分年 Sharpe

| ARIMA | 驗證年 | A 原始 | B 事後 ACF | C MSE | D 共同訓練 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(annualrows)}

![分年比較](JPX-ARIMA-joint-annual.png)

2021 是部分年度，驗證訊號截至2021-12-01。年度切分與前輪相同，包含依報酬日期對齊的前一年末訊號。沒有挑年份、翻轉訊號或將年度辨認結果當成已完成的 regime 模型。

## 訓練是否成功，以及 λ 學到什麼

共32,000件 C/D 股票年度工作，包含不具訓練條件的後備；實際進入最佳化筆數如下。訓練 MSE 變化以各股票年度的訓練樣本數加權，與同批資料的原 MLE 初始化比較，不能與驗證 MSE 混淆。

| ARIMA | 組別 | 進入最佳化 | 回報收斂 | 訓練 MSE 相對變化 | 目標函數呼叫次數中位數 |
|---|---|---:|---:|---:|---:|
{chr(10).join(trs)}

{chr(10).join(statuses)}

未收斂但已有有效候選時，依事前設定保留訓練 MSE 最低的已評估參數，包含原始初始化；不是把未收斂紀錄刪掉。原 MLE 不可用時沿用原零分後備；若原 MLE 可用但成熟 Target 不足或初始訓練預測不合法，C、D 均保留原 A 預測，並記錄未執行最佳化。

| ARIMA | λ₁ 中位數 | λ₂ 中位數 | λ₁≈0 | λ₂≈0 | λ₁≈1 | λ₂≈1 |
|---|---:|---:|---|---|---|---|
{chr(10).join(lrs)}

邊界判斷容差10⁻⁶。λ接近0表示在本次局部訓練結果中少用該修正，並不單獨證明該訊號沒有作用；也要看ρ與殘差大小。λ接近1不代表已證明有用。

C、D皆從原MLE出發，D的λ從0出發；使用相同 L-BFGS-B 設定（最多100步、最多1500次函數評估、ftol=10⁻⁹、gtol=10⁻⁵、差分步長10⁻⁶、maxls=20）。不同參數數量使實際函數評估次數與耗時不同；沒有保證得到全域最小值。最佳化器可能在停止條件檢查前略超過 maxfun，實際次數已保留。

在 {s['paired_optimized_records']:,} 筆 C、D 都有最佳化的配對中，D 的訓練 MSE 仍高於 C 超過10⁻¹²的有 **{s['D_training_loss_greater_than_C_count']:,} 筆**。雖然D在λ=0時可表示C，但不同局部搜尋路徑與固定預算不保證找到較低損失；這是本次試跑的最佳化限制，不能把它誤認為較大模型的理論最小訓練損失一定較高。

## 配對差異的不確定性

| ARIMA | 對比 | Sharpe 差異 | 單項95%區間 | 六項近似同時95%區間 |
|---|---|---:|---|---|
{cirows}

所有對比使用相同日期與相同抽樣：20日循環區塊、2,000次、seed=20260924。單項區間為bootstrap百分位；同時區間用六個對比的最大中心化絕對偏差95%分位作共同半徑。這僅針對本輪固定六項比較，未校正之前25組選模與後續反覆探索，也不是未來年度Sharpe的預測區間。

## 資料界線、後備與核對

- Expanding 起點2017-01-04，年度訓練／驗證沿用原953日切分。每個worker只接收該年度已成熟的Target切片；若訓練價格截止T，最多使用訊號T−2的Target，因為其出場價在T才實現。最後兩個尚未成熟的訊號標籤不參與訓練。
- 跳過殘差初始化前段，至少100個有效成熟Target。C、D使用完全相同的訓練股票日，候選參數不能藉由排除難預測資料來降低MSE。
- 年內ARIMA參數與ACF固定，隨已發生價格更新forward states。ACF在訓練配適時使用整個訓練前綴，因此訓練MSE是配適誤差，不是每個歷史起點當時可取得資訊下的獨立預測誤差。
- 缺失日期保留；ACF只使用中心化後有效配對，至少100個有效殘差、lag1和lag2各80對。不加入殘差截距、反向修正或λ大於1的放大，也沒有額外正則化或驗證集早停。
- 沿用JPX完整唯一排名、同分按股票代碼、前後各200檔、2到1線性權重。非法／非正預測給零分，數值失效只從當時起影響後續預測；股票不被刪掉，不裁切有限極端值。
- 本輪 {s['causal_checks']} 次固定樣本的截斷／未來資料擾動核對通過；原參數初始化與存檔價格預測一致，λ=0時兩組目標一致，訓練最佳損失不高於初始化，平穩／可逆根與λ界線均核對。
- 八個版本的完整排名及官方Sharpe獨立重算通過；A、B與前輪逐日結果重現，v7沿用前輪核對過的同期參照。來源資料雜湊一致，正式test未讀取。所有指標使用相同 {s['common_days']} 日，排除日期：{'無' if not s['excluded_dates'] else '、'.join(s['excluded_dates'])}。
- 原始Target與因果價格比值間的微小差異沿用前輪揭露，使用原始Target，不自行覆寫。結果未納入交易成本、成交限制或新的獨立測試資料。

| ARIMA | 新組別 | 零分後備股票日比例 | 入選前後200的後備次數 |
|---|---|---:|---:|
{auditrows}

## 檔案

資料包包含固定設定、訓練與評估程式、壓縮參數檢查點、訓練稽核、四個新版本的每日分數／名次、每日績效與分年表。為節省磁碟，不重複包含原價格、訓練Target快取或上一輪大型資料包；重跑仍依賴本機原始JPX資料與前輪結果。

此表可用於企劃中的「共同訓練是否提供增量價值」段落。若要主張ARIMA具有可重複的真實預測價值，仍需先固定研究決策，再用未參與選模的資料確認；不能僅根據這張已重複使用驗證資料的表下定論。[時間序列驗證參考](https://otexts.com/fpp3/tscv.html)
'''
    report=report.replace('没有','沒有')
    local=report
    for name in ['JPX-ARIMA-joint-comparison.png','JPX-ARIMA-joint-annual.png']:local=local.replace(f']({name})',f']({OUT/name})')
    (OUT/'JPX-ARIMA-joint-report.md').write_text(local)
    for name in ['metrics.csv','annual.csv','contrasts.csv','results.json']:shutil.copy2(RUN/name,OUT/f'JPX-ARIMA-joint-{name}')
    package=OUT/'JPX-ARIMA-joint-code-and-results.zip'
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for name in ['run_arima_joint.py','evaluate_arima_joint.py','report_arima_joint.py','arima_joint_plan.md','run_arima_grid.py','run_arima_acf.py','evaluate_arima_grid.py','evaluate_arima_acf.py']:z.write(ROOT/name,name)
        for path in RUN.iterdir():
            if path.is_file() and path.name!='training_targets.npz' and not path.name.endswith(('-wal','-shm')):z.write(path,'arima_joint/'+path.name)
        z.write(SOURCE/'prediction_keys.npz','arima_joint/prediction_keys.npz')
        z.writestr('JPX-ARIMA-joint-report.md',report)
        for name in ['JPX-ARIMA-joint-comparison.png','JPX-ARIMA-joint-annual.png']:z.write(OUT/name,name)
        z.writestr('README.md','''# Joint ARIMA / residual ACF Target-MSE trial
Keep scripts in the original work directory alongside prior arima_grid/arima_acf results. Original market ZIP and reference project containing official_metric are dependencies. No raw market or training Target cache is redistributed.
Python 3.14.2, statsmodels 0.14.6, numpy 2.4.6, pandas 3.0.3, scipy 1.17.1, matplotlib 3.10.9.
Set OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1.
python run_arima_joint.py prepare
python run_arima_joint.py pilot --workers 6
python run_arima_joint.py run --workers 6
python evaluate_arima_joint.py
python report_arima_joint.py
fits.sqlite audit blobs are zlib-compressed JSON, keyed by (p, yi, ci); yi=0..3 means 2018..2021, ci indexes original grid security codes. Each record includes both C and D results. Final p*_C/D_predictions.npz arrays align to prediction_keys.npz (date and stock ascending). Price predictions can be reconstructed from parameters and original source prices.
Preserve code/plan hashes when resuming checkpoints. Optimizer convergence is separate from keeping the lowest valid training-objective candidate. Training input contains only labels whose t+2 price was available before validation. Formal test remains unused; this is exploratory reused-validation evidence.
''')
    with zipfile.ZipFile(package) as z:
        assert z.testzip() is None
        for name,key in [('run_arima_joint.py','runner_sha256'),('evaluate_arima_joint.py','evaluator_sha256')]:assert hashlib.sha256(z.read(name)).hexdigest()==s[key]
    print('DELIVERED',str(package),package.stat().st_size,flush=True)
if __name__=='__main__':main()
