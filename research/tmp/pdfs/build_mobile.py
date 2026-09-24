from pathlib import Path
import json, shutil, zipfile, hashlib, math, html
import pandas as pd
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors

ROOT=Path.cwd(); OUT=ROOT/'output'; DATA=OUT/'jpx_mobile_data_20260914'; PDF=OUT/'pdf/JPX-model-diagnostics-mobile-20260914.pdf'
V14=ROOT/'jpx_v14_filtered_forecast_events_20260914'; V15=ROOT/'jpx_v15_soft_rank_filtered_20260914'
pdfmetrics.registerFont(TTFont('CJK','/System/Library/Fonts/STHeiti Light.ttc',subfontIndex=0))
pdfmetrics.registerFont(TTFont('CJKB','/System/Library/Fonts/STHeiti Medium.ttc',subfontIndex=0))
MODELS=[('M1',V14,'sgd_only','MSE / SGD'),('M2',V14,'sgd_sqrt','MSE / SGD + √N'),('S1',V15,'sgd_only','Soft rank / SGD'),('S2',V15,'sgd_sqrt','Soft rank / SGD + √N')]
res={k:json.loads((d/a/'results.json').read_text()) for k,d,a,_ in MODELS}
comp=pd.read_csv(V15/'comparison.csv'); annual=pd.read_csv(V15/'comparison_by_year.csv')
fs=json.loads((V14/'feature_summary.json').read_text()); flat=json.loads((V15/'training_flat_score_baseline.json').read_text())
source=[]
def copy(src, dest):
    dest=DATA/dest;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
    source.append({'source':str(src.relative_to(ROOT)),'file':str(dest.relative_to(DATA)),'sha256':hashlib.sha256(src.read_bytes()).hexdigest()})
for fn in ['comparison.csv','comparison_by_year.csv','training_flat_score_baseline.csv','training_flat_score_baseline.json','JPX-v15-soft-rank-report.md','delivery_audit.json']:
    copy(V15/fn,fn)
copy(V14/'feature_summary.json','financial_feature_summary.json')
copy(V14/'JPX-v14-filtered-forecast-report.md','JPX-v14-filtered-forecast-report.md')
daily=[]; trainings=[]; params=[]; mse_diag={}
for k,d,a,label in MODELS:
    p=d/a
    for fn in ['results.json','daily_spread_returns.csv','training_updates.csv','parameter_history.csv','audit.json']:
        copy(p/fn,Path(k)/fn)
    dd=pd.read_csv(p/'daily_spread_returns.csv')
    if k.startswith('M'):
        diag=pd.read_csv(p/'diagnostic_daily.csv');mse_diag[k]=diag
        copy(p/'diagnostic_daily.csv',Path(k)/'diagnostic_daily.csv')
        copy(p/'largest_prediction_errors.csv',Path(k)/'largest_prediction_errors.csv')
        rr=pd.read_csv(V15/f'v14_{a}_ranking_metrics.csv')
        dd=dd.drop(columns=['RankIC','NormalizedHardRankMSE'],errors='ignore').merge(rr[['Date','RankIC','NormalizedHardRankMSE']],on='Date',validate='one_to_one')
        dd=dd.merge(diag[['Date','AllStockForecastMSE','ZeroPredictionMSE','EligibleMSEContribution','ExcludedMSEContribution','ExcludedRows']],on='Date',validate='one_to_one')
    else:
        for fn in ['daily_soft_rank_loss.csv','score_diagnostics.json']:
            copy(p/fn,Path(k)/fn)
        dd=dd.merge(pd.read_csv(p/'daily_soft_rank_loss.csv')[['Date','SoftRankLoss','FlatScoreSoftRankLoss','ScoreRange']],on='Date',validate='one_to_one')
    dd.insert(0,'Model',k);daily.append(dd)
    tr=pd.read_csv(p/'training_updates.csv'); tr.insert(0,'Model',k)
    trainings.append(tr[[c for c in tr if not c.startswith(('Before_','After_'))]])
    for name,value in res[k]['final_parameters'].items(): params.append({'Model':k,'Parameter':name,'Value':value,'AsOf':res[k]['parameter_asof']})
    assert len(dd)==953
    spread=dd.OfficialDailySpread
    assert abs(spread.mean()/spread.std(ddof=1)-res[k]['validation']['official_style_unannualized_sharpe'])<1e-12
all_daily=pd.concat(daily,ignore_index=True)
all_daily.to_csv(DATA/'all_models_daily.csv',index=False,encoding='utf-8-sig')
pd.concat(trainings).to_csv(DATA/'all_models_training_summary.csv',index=False,encoding='utf-8-sig')
paramdf=pd.DataFrame(params);paramdf.pivot(index='Parameter',columns='Model',values='Value').to_csv(DATA/'all_models_final_parameters.csv',encoding='utf-8-sig')
summary=[]
for i,(k,d,a,label) in enumerate(MODELS):
    r=res[k];row=comp.iloc[i].to_dict();row['Model']=k;row['MeanDailyReturnBP']=r['validation']['mean_illustrative_gross_one_return']*10000;row['ReturnMSE']=r.get('mean_daily_forecast_mse');summary.append(row)
pd.DataFrame(summary).to_csv(DATA/'model_summary.csv',index=False,encoding='utf-8-sig')

W,H=390,680; M=24; CW=W-2*M
NAVY=colors.HexColor('#153347');TEAL=colors.HexColor('#007C82');INK=colors.HexColor('#172D3A');MUTED=colors.HexColor('#536B78');BG=colors.HexColor('#F3F7F9');RED=colors.HexColor('#A64B3C');RULE=colors.HexColor('#D7E2E8')
c=canvas.Canvas(str(PDF),pagesize=(W,H));c.setTitle('JPX 模型診斷數據｜手機閱讀版');c.setAuthor('Quant Learning & Internship 2027')
page=0;y=0
style=ParagraphStyle('body',fontName='CJK',fontSize=12.3,leading=19,textColor=INK,wordWrap='CJK',spaceAfter=0)
small=ParagraphStyle('small',parent=style,fontSize=10,leading=15,textColor=MUTED)
def p(text,size=None,color=None,gap=10):
    global y
    st=ParagraphStyle('p',parent=style,fontSize=size or style.fontSize,leading=(size*1.55 if size else style.leading),textColor=color or INK)
    para=Paragraph(text,st);_,h=para.wrap(CW,600)
    assert y-h>=40,('overflow',page,y,h,text[:60])
    para.drawOn(c,M,y-h);y-=h+gap

def begin(title,subtitle=''):
    global page,y
    if page:c.showPage()
    page+=1;c.setFillColor(NAVY);c.rect(0,H-8,W,8,fill=1,stroke=0)
    c.setFillColor(TEAL);c.setFont('CJKB',10);c.drawString(M,H-32,'JPX  /  MODEL DIAGNOSTICS')
    y=H-53;p(title,22,NAVY,10)
    if subtitle:p(subtitle,10.3,MUTED,15)
    c.setStrokeColor(RULE);c.line(M,31,W-M,31);c.setFont('CJK',9);c.setFillColor(MUTED)
    c.drawString(M,17,'2026-09-14  ·  手機離線閱讀版');c.drawRightString(W-M,17,f'{page:02d} / 10')

def heading(t):p(t,14.5,TEAL,7)
def table(rows,widths=None,font=11.2,rowheight=None):
    global y
    widths=widths or [CW/len(rows[0])]*len(rows[0]);wrap=[]
    for ir,row in enumerate(rows):
        st=ParagraphStyle('cell',parent=style,fontName='CJKB' if ir==0 else 'CJK',fontSize=font,leading=font*1.35,textColor=colors.white if ir==0 else INK)
        wrap.append([Paragraph(html.escape(str(x)),st) for x in row])
    tb=Table(wrap,colWidths=widths,rowHeights=rowheight,hAlign='LEFT')
    tb.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,BG]),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),0.9 if rowheight else 6),('BOTTOMPADDING',(0,0),(-1,-1),0.9 if rowheight else 6),('LINEBELOW',(0,-1),(-1,-1),.5,RULE)]))
    _,h=tb.wrap(CW,H);assert y-h>=40,('table overflow',page,y,h);tb.drawOn(c,M,y-h);y-=h+12

def fmt(x,n=5):return f'{x:+.{n}f}'

begin('你的模型，問題在哪裡？','先看現有實驗的數據，再決定下一輪要改什麼。')
p('四組 Sharpe 都是負值。Soft rank 的單獨 SGD 最接近 0，但目前還沒有證據能把改善歸因於取消衰減。',14)
table([['模型','訓練方式','Sharpe'],['M1｜MSE','單獨 SGD',fmt(comp.iloc[0].Sharpe)],['M2｜MSE','SGD + √N',fmt(comp.iloc[1].Sharpe)],['S1｜Soft','單獨 SGD',fmt(comp.iloc[2].Sharpe)],['S2｜Soft','SGD + √N',fmt(comp.iloc[3].Sharpe)]],[100,136,106])
heading('優先查三件事')
p('1. MSE 降低約 93.7%，Sharpe 卻變差：報酬數值與選股排名可能在改善不同的事。')
p('2. Soft rank 的平均訓練 loss 仍高於同分基準：先查更新方式與分數尺度。')
p('3. 大於 100 倍只跳過訓練，仍然參與預測：極端財報值依然可能支配選股。')
p('本文件四組都保留 exp(-a/9) 衰減。「不衰減 × 兩種 loss」尚未完成，不放入已完成結果。',10.5,MUTED)

begin('四組的完整比較','共同驗證期間：2017-12-29 至 2021-12-01。')
table([['模型','Rank IC ↑','排名 MSE ↓']]+[[k,fmt(comp.iloc[i].MeanRankIC,6),f'{comp.iloc[i].NormalizedHardRankMSE:.6f}'] for i,(k,*_) in enumerate(MODELS)],[64,139,139])
table([['模型','平均每日報酬','報酬 MSE ↓']]+[[k,f'{res[k]["validation"]["mean_illustrative_gross_one_return"]*10000:+.3f} bp',f'{res[k]["mean_daily_forecast_mse"]:.6f}' if k.startswith('M') else '不適用'] for k,*_ in MODELS],[64,139,139])
p('每組 953 個收益日；Rank IC 有定義的為 952 日。每天排名 1,896 至 2,000 檔，買前 200、賣後 200；各邊權重由 2 線性降到 1。',11.5)
p('Sharpe = 每日多空收益平均 ÷ 樣本標準差，未年化、未扣成本。表中每日報酬為官方 spread ÷ 400，相當於多空各 50% 的示意報酬；1 bp = 0.01%。',11.5)
p('Rank IC 是每天分數與 Target 的 Spearman 相關；排名 MSE 是平均同分名次差除以 N-1 後平方。Soft rank 的 g 是排序分數，不能當作報酬率。',11.5)
p('Target 對應訊號日之後第一個交易日收盤至第二個交易日收盤。2020-09-29 的 Target 全為 0，該日 Rank IC 無定義。',10,MUTED)

begin('改善是否跨年度成立？','下表沿用既有驗證組別標記，參數沒有每年重設。')
for ver,title in [('v14','MSE：未年化 Sharpe'),('v15','Soft rank：未年化 Sharpe')]:
    heading(title);ss=annual[annual.Version.eq(ver)]
    table([['組別 / 日數','SGD','SGD + √N']]+[[f'{yr} / {int(ss[ss.ValidationYear.eq(yr)].Days.iloc[0])}',fmt(ss[(ss.ValidationYear.eq(yr))&ss.Variant.eq('sgd_only')].Sharpe.iloc[0]),fmt(ss[(ss.ValidationYear.eq(yr))&ss.Variant.eq('sgd_sqrt')].Sharpe.iloc[0])] for yr in [2018,2019,2020,2021]],[114,114,114])
p('沒有一組四期皆為正。S1 的總 Sharpe 較好，但 2019 組別從 M1 的 +0.03022 變成 -0.06432；改善並非每期一致。')
p('因此下一輪應一起看整體與分期數據，並保留新的測試期間。這些數據本身尚未檢驗差異的統計顯著性。',11,MUTED)

begin('Loss 下降，為何沒選得更好？','以下是看到標籤後，同一批訓練資料上的每日平均 loss。')
for prefix,title in [('M','報酬 MSE'),('S','Soft-rank loss')]:
    heading(title)
    table([['訓練階段',prefix+'1｜SGD',prefix+'2｜SGD+√N']]+[[name,f'{res[prefix+"1"][field]:.6f}',f'{res[prefix+"2"][field]:.6f}'] for name,field in [('更新前','mean_training_loss_before'),('逐檔 SGD 後','mean_training_loss_after_sgd'),('整體更新後','mean_training_loss_after_full')]],[128,107,107])
p('MSE：M2 的驗證報酬 MSE 從 M1 的 0.021107 降到 0.001336，但 Sharpe 從 -0.01760 降至 -0.02987。數值誤差降低，這次沒有帶來更好的多空排名。',11.6)
p('逐檔 SGD 後，整天平均 loss 上升的日數：M1 553、M2 1,162、S1 196、S2 672（每組共 1,199 日）。單檔更新只要求該檔 loss 改善。',11.6)
p('每日 SGD 做 N 次；混合組另做 floor(√N) 次整體更新。每組共 2,323,512 次單檔更新，混合組另嘗試 52,189 次整體更新。',10.5,MUTED)

begin('Soft rank 的同分基準','τ 固定為 1。兩組都沿用既有 SGD 與學習率搜尋規則。')
table([['資料 / 指標','S1','S2'],['訓練最終 loss','0.124854','0.105364'],['訓練同分基準','0.083344','0.083344'],['訓練勝過基準日數','9 / 1,199','45 / 1,199'],['驗證 soft loss','0.143688','0.127718'],['驗證同分基準','0.083327','0.083327'],['驗證勝過基準日數','1 / 953','1 / 953']],[152,95,95])
p('把所有線性係數設成 0，就能達到「所有股票同分」的 loss。因此現有更新流程尚未把訓練目標降到這個簡單的可達成基準。')
p('同分不提供選股訊號。所有 g 同乘正數，硬排名不變，但 soft rank 和 loss 會改變；倍數趨近 0 時，各 soft rank 趨近中間名次。')
heading('現行目標的定義')
p('R(i) = 1 + Σ(j ≠ i) sigmoid((g(j) - g(i)) / τ)<br/>L(i) = ((R(i) - ρ(i)) / (N - 1))²<br/>整體 L = mean(L(i))',11.5)
p('ρ 是依真實 Target 排序的平均同分名次。訓練比較池先套用 100 倍門檻；驗證 soft loss 則使用完整有限標籤股票池，兩期基準因此分開計算。',10,MUTED)

begin('極端值仍會進入預測','M1 最大平方誤差案例，並非一般股票的代表。g = 1 代表 100% 報酬。')
err=pd.read_csv(V14/'sgd_only/largest_prediction_errors.csv').head(2)
for row in err.itertuples():
    heading(f'{row.Date}｜股票 {row.SecuritiesCode}')
    table([['欄位','數值'],['預測報酬 g',f'{row.g:+.6f}'],['實際 Target',f'{row.Target:+.6f} ({row.Target*100:+.3f}%)'],['財報對 g 的貢獻',f'{row.FinancialContribution:+.6f}'],['最大財報成分絕對值',f'{row.TrainingFeatureMaxAbs:,.3f}'],['可進入訓練','否']],[175,167],10.5)
frac=mse_diag['M1'].ExcludedMSEContribution.mean()/mse_diag['M1'].AllStockForecastMSE.mean()
p(f'M1 中，被訓練門檻排除的股票對完整驗證 MSE 的貢獻占 {frac:.1%}（按每日總股票數正規化後再平均）。',11.3)
p('原規則只跳過 abs(財報成分) > 100 的股票日，±100 保留，預測仍含全股票池。共跳過 2,294 個到期訓練股票日；不能把「不訓練」理解成「不選股」。',10.5,MUTED)

begin('保留目前實作的公式','這一頁記錄已跑完的 v14 / v15，方便你對照原本想法。')
heading('財報事件與季預測')
p('成長率 G(x,b) = (x - b) / |b|。基期為 0 或資料不可用時，不建立該成分事件。',11.2)
p('季預測 q = (全年 Forecast - 已知累計實績 C(k)) / (4 - k)，k = 0, 1, 2, 3。修正前後固定同一個 k、C(k) 與比較基期。',11.2)
p('每一指標 m 保留五個成分：u＝實際 YoY 成長；v＝事前季預測相對相同實際基期的成長；Q／Y＝季預測相較上一季／去年同季預測的成長；U＝新舊季預測在相同基期下的成長率差。',11.2)
heading('模型對財報的作用')
p('各指標的財報貢獻 = βA × (u - d × v)<br/>　+ d × (βQ × Q + βY × Y + βU × U)',11.5)
p('g = 截距 + 11 個價量特徵的加權和<br/>　+ NetSales、OperatingProfit、EPS 三者的財報貢獻。',11.2)
p('事件以 exp(-a/9) 累加，a 是交易日齡。沒有新的 Forecast 不新增該次事件，既有事件繼續衰減。PR1、VR1 各除以近 22 日標準差，不扣均值。',11.2)
p('共有 24 個線性參數與 3 個折價 d；d 限制在 [0,1]。每步先更新線性係數，再重算梯度更新 d。只在 Target 到期後，使用訊號日凍結的特徵與門檻結果。',10.5,MUTED)

labels={'alpha':'截距','NetSales':'營收','OperatingProfit':'營業利益','EPS':'EPS'}
names=list(res['M1']['final_parameters'])
def short(n):
    if n=='alpha':return '截距'
    for m in ['NetSales','OperatingProfit','EPS']:
        if n.startswith(m):return labels[m]+' '+{'BetaActual':'A','BetaQoQ':'Q','BetaYoY':'Y','BetaRevision':'U','Discount':'d'}[n[len(m):]]
    return n
for prefix,title in [('M','MSE：全部 27 個參數'),('S','Soft rank：全部 27 個參數')]:
    begin(title,'截至 2021-12-03 最後到期標籤更新後；不是整段期間固定係數。')
    table([['參數',prefix+'1｜SGD',prefix+'2｜SGD+√N']]+[[short(n),f'{res[prefix+"1"]["final_parameters"][n]:+.5f}',f'{res[prefix+"2"]["final_parameters"][n]:+.5f}'] for n in names],[142,100,100],font=10,rowheight=15.3)
    p('A＝實績落差；Q＝季比；Y＝年比；U＝預期修正；d＝折價。不同 loss 的 g 尺度不同，不能只比係數大小判斷重要性。',10,MUTED,5)
    if prefix=='S':p('S2 的 EPS d = 0：此時 EPS 的 Q、Y、U 作用為 0，但 βA × u 仍保留。',10,MUTED,0)

begin('帶走的資料與下一步','這份資料包只整理已完成結果，未啟動新一輪訓練。')
heading('可優先驗證的三個假設')
p('① 更新規則：追蹤逐檔／整體 loss、同分基準和分數分布，確認訓練是否有效降低既定目標。',11.5)
p('② 極端值：拆開可訓練與被排除股票對 MSE、前後 200 檔收益的貢獻，確認問題集中在哪裡。',11.5)
p('③ 不衰減：保持資料、模型、loss 和訓練規則一致，再比較事件永久累加或只留最新值；這兩種定義需要分清楚。',11.5)
heading('資料包裡有什麼')
p('model_summary.csv：四組總表。<br/>comparison_by_year.csv：分期比較。<br/>all_models_daily.csv：3,812 列逐日驗證數據。<br/>all_models_training_summary.csv：4,796 列逐日訓練紀錄。<br/>all_models_final_parameters.csv：四組 27 個參數。',10.7)
p('M1 / M2 / S1 / S2 資料夾另有完整每日參數軌跡、訓練紀錄、結果與稽核；MSE 附極端錯誤案例，Soft rank 附逐日 loss。README 說明每個欄位。',10.7)
p('來源：工作區已稽核的 v14 與 v15 輸出；完整相對路徑與 SHA-256 在 manifest.json。資料包不含數百萬列股票全量排名與大型訓練軌跡。',10,MUTED)
p('所有結果來自同一段已反覆研究的歷史驗證資料；不衰減尚待實驗，正式基準仍為 v7。本文件將觀察與待驗證假設分開呈現。',10,MUTED)
assert page==10
c.save()

README='''JPX 模型診斷資料包｜2026-09-14

先讀 PDF：10 頁手機直式版。四組均保留 exp(-a/9) 財報事件衰減。
M1 = v14 報酬 MSE、單獨 SGD
M2 = v14 報酬 MSE、SGD + 每日 floor(sqrt(N)) 次整體更新
S1 = v15 sigmoid soft rank、單獨 SGD
S2 = v15 sigmoid soft rank、SGD + 每日 floor(sqrt(N)) 次整體更新
尚未完成不衰減對照，本包沒有不衰減實驗結果。

期間與範圍
- 驗證：2017-12-29 至 2021-12-01，953 日；Rank IC 有定義的 952 日。
- 訓練：1,199 個標籤到期日（含暖身），最後更新 2021-12-03。
- 2020-09-29 的 Target 全為 0，Rank IC 留空，不以 0 替代。
- 每天預測 1,896 至 2,000 檔；前 200 買入、後 200 賣出；邊內權重 2 至 1。
- 所有收益未扣成本。Sharpe 為未年化，std 使用 ddof=1。
- 不含股票全量排名與大型訓練逐步 trace，保留手機可攜的每日數據與案例。

入口檔案
model_summary.csv：4 列模型總表；MeanDailyReturnBP 單位 bp (0.01%)。
comparison.csv / comparison_by_year.csv：原始已稽核比較；ValidationYear 是既有驗證組別標記，不一定和曆年邊界完全相同。
all_models_daily.csv：4 x 953 = 3812 列；不同 loss 不適用的欄位留空。
all_models_training_summary.csv：4 x 1199 = 4796 列；完整逐日更新前後參數在各組 training_updates.csv。
all_models_final_parameters.csv：27 個參數 x 4 組，日期均為 2021-12-03；不得用最後參數回頭當成歷史預測權重。
financial_feature_summary.json：15 個原始財報成分範圍、事件數與門檻計數；其範圍涵蓋原始訊號資料，不能當成已過濾訓練分布。
M1/M2/largest_prediction_errors.csv：刻意選取最大平方誤差的案例，不具代表性；g 是預測報酬，1.0 = 100%。
M1/M2/diagnostic_daily.csv：MSE 分解與零預測基準。
S1/S2/daily_soft_rank_loss.csv：完整有限標籤驗證股票池上的 soft-rank loss；tau = 1。
training_flat_score_baseline.csv/json：已經過訓練門檻篩選的訓練股票池，同分基準與模型 loss。
各組 parameter_history.csv：每日新標籤更新前後參數；Before_ 與 After_ 為更新時點，不是未來參數。
各組 results.json / audit.json：完整結果與稽核摘要。
原始 v14/v15 Markdown 報告：定義、公式與驗證說明。
manifest.json：來源檔案相對工作區路徑與原檔 SHA-256。

逐日欄位字典
Date：訊號日（daily），更新日（training）；訓練原訊號日看 SignalDate，到期日看 ExitDate。
Model：M1 / M2 / S1 / S2。
StocksRanked：完整預測股票數。
FallbackStocks：輸入不足而使用既有缺值處理的股票數，不代表整檔分數固定為 0。
SelectedFallbackStocks：選入多空各 200 檔的 Fallback 股票數。
SelectedMissingTargets / AllMissingTargets：選股／全股票池缺失真實報酬數。
LongWeightedScore / ShortWeightedScore：各邊真實 Target 按 2 至 1 加權並除以權重平均的和。
OfficialDailySpread：LongWeightedScore - ShortWeightedScore。
IllustrativeGrossOneReturn：OfficialDailySpread / 400；僅示意總曝險 1、多空各 50% 的每日報酬。
RankIC：每天 g 與實際 Target 的 Spearman 相關，取每日等權平均。
NormalizedHardRankMSE：預測 g 與 Target 的降序平均同分名次差，除以 N-1 後平方平均。
MaxAbsoluteScore / MaxAbsolutePrediction：當天 g 的最大絕對值；S1/S2 的分數不是報酬率。
ScoredStocks：排名診斷使用的有限標籤股票數。
AllStockForecastMSE：當天全有限標籤股票 mean((g-Target)^2)，僅 MSE 組適用。
ZeroPredictionMSE：當天以 0 預測所有報酬的 mean(Target^2)。
EligibleMSEContribution / ExcludedMSEContribution：可訓練／排除股票的平方誤差和，均除以全當天有限標籤股票數；兩者相加等於全股票 MSE，不是各子群自己的平均。
ExcludedRows：完整驗證股票池中超過訓練門檻的數量。
SoftRankLoss：mean(((soft_rank - true_rank)/(N-1))^2)；用 tau=1。
FlatScoreSoftRankLoss：同一天所有股票 g 相同時的 soft loss，沒有交易訊號。
ScoreRange：當天最大 g 減最小 g。

逐日訓練欄位
KnownLabelStocks：當日到期的有限標籤股票數。
TrainingStocks：套用財報成分 abs > 100 排除門檻後股票數 N。
ThresholdSkippedStocks：本日超門檻未訓練股票數，仍保留預測；不是永久刪除股票。
MissingTargets：缺失到期 Target 的股票數。
LossBeforeSGD / LossAfterSGD / LossAfterFull：當天整個訓練股票池的平均 loss，分別於更新前、逐檔後、整體後計算；MSE 與 Soft loss 不能跨類直接比较。
SGDAttempts / FullAttempts：逐檔／整體更新嘗試數，每步含線性與折價兩個子步。
SGDAccepted / FullAccepted：至少一個子步接受更新的步數。
SGDNoEta / FullNoEta：無可接受學習率的子步數，計數單位可能與 Accepted 不同，不能直接相加當 Attempts。
LinearAccepted / DiscountAccepted：線性／折價子步接受數。
CandidateEvaluations：學習率候選評估數。

參數與財報定義
alpha 與 11 價量係數；每個財報指標 NetSales / OperatingProfit / EPS 另有 BetaActual / BetaQoQ / BetaYoY / BetaRevision / Discount。
PR1、VR1 各除以近 22 日標準差 (ddof=1)，包含訊號日，不扣均值。
財報 G(x,b)=(x-b)/abs(b)。季 Forecast=(全年 Forecast-已知財年累計實績)/(4-已知季度數)。
每個指標 5 個事件成分 u,v,Q,Y,U；財報貢獻=BetaActual*(u-d*v)+d*(BetaQoQ*Q+BetaYoY*Y+BetaRevision*U)。
目前 EPS 也以成長率建模；不是較早版本的 EPS 原值。
Forecast 修正使用相同已知累計實績與季度數、相同去年同期單季實績基期，再取新舊成長率差。
沒有新 Forecast 不新增該次事件；舊事件持續按 exp(-交易日齡/9) 衰減。非零事件累加。
會計基礎不明的 ForecastRevision / NumericalCorrection 沒有重新加入；此資料與 v14/v15 一致。
所有標籤須到期才可训练，使用原訊號日凍結特徵、門檻及種子。沒有年度參數重設。

本包僅整理已完成本地實驗；沒有新的持出測試集或不衰減訓練結果。
'''
(DATA/'README.txt').write_text(README,encoding='utf-8')
manifest={'created':'2026-09-14','scope':'v14/v15 completed decayed experiments','no_decay_completed':False,'sources':source,'derived_counts':{'daily_rows':len(all_daily),'training_rows':sum(len(t) for t in trainings),'parameter_values':len(paramdf)},'pdf_pages':10}
(DATA/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'jpx_mobile_artifact_stats.json').write_text(json.dumps({'pdf':str(PDF),'mse_excluded_contribution_share_M1':frac,'mse_reduction':1-res['M2']['mean_daily_forecast_mse']/res['M1']['mean_daily_forecast_mse'],'models':summary},indent=2),encoding='utf-8')
print(json.dumps({'pdf':str(PDF),'pages':page,'data_rows':len(all_daily),'training_rows':sum(len(t) for t in trainings),'M1_excluded_mse_share':frac},ensure_ascii=False))
