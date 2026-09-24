from pathlib import Path
import json,html,zipfile
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph,Table,TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
R=Path(__file__).resolve().parent;out=R.parent/'output/pdf/JPX-v30-profit-actual-forecast-mobile-20260916.pdf'
df=pd.read_csv(R/'comparison.csv');yr=pd.read_csv(R/'comparison_by_year.csv')
pdfmetrics.registerFont(TTFont('C','/System/Library/Fonts/STHeiti Light.ttc',subfontIndex=0));pdfmetrics.registerFont(TTFont('B','/System/Library/Fonts/STHeiti Medium.ttc',subfontIndex=0))
W,H=390,680;M=24;cw=W-2*M;navy=colors.HexColor('#153347');teal=colors.HexColor('#007C82');gray=colors.HexColor('#536B78');light=colors.HexColor('#F2F7F9')
c=canvas.Canvas(str(out),pagesize=(W,H));c.setTitle('v30｜實際值與Forecast七組對照');c.setAuthor('Quant Learning & Internship 2027');y=0
st=ParagraphStyle('s',fontName='C',fontSize=11.5,leading=17,wordWrap='CJK',textColor=navy)
def p(t,size=11.5,color=navy,gap=9):
    global y
    ss=ParagraphStyle('x',parent=st,fontSize=size,leading=size*1.48,textColor=color);a=Paragraph(t,ss);_,h=a.wrap(cw,600);assert y-h>=41,(y,h,t);a.drawOn(c,M,y-h);y-=h+gap

def page(n,title):
    global y
    if n>1:c.showPage()
    c.setFillColor(navy);c.rect(0,H-7,W,7,fill=1,stroke=0);y=H-31;p('JPX  /  v30  /  2026-09-16',10,teal);p(title,21)
    c.setFont('C',9);c.setFillColor(gray);c.drawString(M,18,'不排除極端值  ·  手機版');c.drawRightString(W-M,18,f'{n} / 2')
def table(rows,widths,font=10):
    global y
    rr=[]
    for i,row in enumerate(rows):
        s=ParagraphStyle('t',parent=st,fontName='B' if i==0 else 'C',fontSize=font,leading=font*1.3,textColor=colors.white if i==0 else navy)
        rr.append([Paragraph(html.escape(str(v)),s) for v in row])
    t=Table(rr,colWidths=widths);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),navy),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,light]),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5)]));_,h=t.wrap(cw,600);assert y-h>=41,(y,h);t.drawOn(c,M,y-h);y-=h+10




page(1,'合併兩項，沒有一致變差')
p('R＝營業利益實際年增；F＝最新Forecast年比。四組共同訓練，三組固定相同每日g，只訓練財報係數。',11.5)
p('同日七組使用相同learning rate與股票池。953日validation；Sharpe未年化、未扣成本。Sharpe第一、Rank IC第二，MSE不參與選擇。',10,gray)
labels={'g':'純g（共同步長）','joint_r':'共同：g＋R','joint_f':'共同：g＋F','joint_rf':'共同：g＋R＋F','frozen_r':'固定g：＋R','frozen_f':'固定g：＋F','frozen_rf':'固定g：＋R＋F'}
table([['模型','Sharpe','Rank IC']]+[[labels[q.Variant],f'{q.Sharpe:+.6f}',f'{q.RankIC:+.6f}'] for q in df.itertuples()],[146,98,98],10.5)
p('共同訓練：合併高於兩個單項。<br/>固定g：合併略低於R，但高於F。<br/>所以這輪不支持「放在一起必然更差」。',11.5,teal)
p('固定g的合併減R：ΔSharpe -0.000244；<br/>95%探索區間[-0.003057, +0.002649]。<br/>差距很小，目前不能確認穩定負貢獻。',10.5)
p('新增R後，步長為何調整？',14,teal)
p('原排程在94個更新日超過完整模型穩定上限，因此七組統一使用較保守的共同步長。這輪要對照本表純g，不能直接把差異當作只改財報項。',10.5)
p('最高是固定g＋R，Sharpe +0.004358；原v7仍較高（+0.007679）。正式基準維持v7，test未使用。',10.5)
p('七組各1199次更新與2,326,022筆預測已核對。固定g使用原訊號日保存的預測，沒有期末參數回填；極端值排除0筆。',9.5,gray)
page(2,'目前更值得注意的是季度定義')
p('1｜沒有看到高度整體重疊',14,teal)
p('全validation：Pearson約0.0052，Spearman約0.0995。更新附近與兩欄皆非零時也相近。這不證明獨立，但不支持兩欄幾乎相同的解釋。',10.5)
p('2｜合併沒有增加係數正負切換',14,teal)
table([['係數','單項切換次數','聯合切換次數'],['R','148','146'],['F','145','145']],[94,124,124],10)
p('聯合模型的兩項預測貢獻約7.46%互相抵銷；抵銷本身不代表壞事，也可能是修正另一項誤差。',10.5)
p('3｜實際與Forecast常是不同季',14,teal)
p('同公告可配到有效F的21,135筆來源事件，F全部指向實際值之後的季度。這是來源全期間的事件檢查，不僅限validation。',10.5)
p('剩餘單季Forecast＝<br/>(全年Forecast-已知累計實績) / 剩餘季數。',11)
p('全年Forecast不變時，已知實績越高，剩餘季度的推算Forecast會越低。這是換算公式的關係，不能直接解讀成市場看法互相矛盾。',10.5)
p('要測「是否超出預期」，應另配同一季：當季實際值與公布前對該季的最後Forecast。本輪沒有執行這個替代實驗。',11,teal)
p('資料範圍與限制',13,teal)
p('R沿用舊版事件累加，F沿用最新狀態覆蓋，均exp(-a/9)。R僅涵蓋既有有效事件；未擴充財報、改算法或排除極端值。相關性與抵銷都是線索，尚未證明Sharpe差異的原因。',9.5,gray)
c.save()
archive=R.parent/'output/JPX-v30-profit-actual-forecast-data-20260916.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    z.write(out,out.name)
    for name in ['comparison.csv','validation_ranking.csv','comparison_by_year.csv','paired_block_bootstrap.csv','all_models_daily.csv','all_coefficients.csv','common_learning_rates.csv','learning_rate_decision.json','feature_correlations.csv','feature_scales.csv','announcement_quarter_alignment.csv','prediction_contributions.csv','financial_gradient_scales.csv','coefficient_stability.csv','g_prediction_drift.csv','diagnostics.json','JPX-v30-profit-actual-forecast-report.md','experiment_plan.md','experiment_defaults_snapshot.json','manifest.json','audit.json','results.json','run.py','audit.py','report.py','diagnostics.py']:
        z.write(R/name,name)
    for variant in df.Variant:
        for name in ['daily_metrics.csv','training_updates.csv','parameter_history.csv','results.json','audit.json','update_audit.json']:
            z.write(R/variant/name,variant+'/'+name)
print(json.dumps({'pdf':str(out),'zip':str(archive),'zip_bytes':archive.stat().st_size}))
