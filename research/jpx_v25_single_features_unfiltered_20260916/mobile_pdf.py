from pathlib import Path
import json,html,zipfile
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph,Table,TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
R=Path(__file__).resolve().parent;out=R.parent/'output/pdf/JPX-v25-unfiltered-single-features-mobile-20260916.pdf'
df=pd.read_csv(R/'comparison_current.csv');eps=pd.read_csv(R/'comparison_eps.csv');result=json.loads((R/'results.json').read_text())
pdfmetrics.registerFont(TTFont('C','/System/Library/Fonts/STHeiti Light.ttc',subfontIndex=0));pdfmetrics.registerFont(TTFont('B','/System/Library/Fonts/STHeiti Medium.ttc',subfontIndex=0))
W,H=390,680;M=24;cw=W-2*M;navy=colors.HexColor('#153347');teal=colors.HexColor('#007C82');gray=colors.HexColor('#536B78');light=colors.HexColor('#F2F7F9')
c=canvas.Canvas(str(out),pagesize=(W,H));c.setTitle('v25｜單項財報加g取消極端值排除');c.setAuthor('Quant Learning & Internship 2027');y=0
st=ParagraphStyle('s',fontName='C',fontSize=11.5,leading=17,wordWrap='CJK',textColor=navy)
def p(t,size=11.5,color=navy,gap=9):
    global y
    ss=ParagraphStyle('x',parent=st,fontSize=size,leading=size*1.48,textColor=color);a=Paragraph(t,ss);_,h=a.wrap(cw,600);assert y-h>=41,(y,h,t);a.drawOn(c,M,y-h);y-=h+gap

def page(n,title):
    global y
    if n>1:c.showPage()
    c.setFillColor(navy);c.rect(0,H-7,W,7,fill=1,stroke=0);y=H-31;p('JPX  /  v25  /  2026-09-16',10,teal);p(title,21)
    c.setFont('C',9);c.setFillColor(gray);c.drawString(M,18,'g＋單項財報  ·  不排除極端值');c.drawRightString(W-M,18,f'{n} / 2')
def table(rows,widths,font=10):
    global y
    rr=[]
    for i,row in enumerate(rows):
        s=ParagraphStyle('t',parent=st,fontName='B' if i==0 else 'C',fontSize=font,leading=font*1.3,textColor=colors.white if i==0 else navy)
        rr.append([Paragraph(html.escape(str(v)),s) for v in row])
    t=Table(rr,colWidths=widths);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),navy),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,light]),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5)]));_,h=t.wrap(cw,600);assert y-h>=41,(y,h);t.drawOn(c,M,y-h);y-=h+10

page(1,'取消極端值排除，重做單項實驗')
p('後續訓練預設已改為不排除極端值。這次18個受影響版本重訓，3個從未排除樣本的版本直接沿用並核對。',11.5)
p('現行16項中，7項Sharpe高於純g；但平均Rank IC全部仍為負，沒有一項同時達到兩個點估計條件。',11.5,teal)
p('純g：Sharpe +0.007679，Rank IC −0.000734。953日驗證；Sharpe未年化、未扣成本。',10,gray)
p('六項現行EPS：不排除後都轉負',14,teal)
short={'EPSActualQoQ':'實際季增','EPSActualGrowth':'實際年增','EPSExpectedGrowth':'實績公布時事前預期成長','EPSForecastActualQoQ':'預測對前季實績','EPSForecastActualYoY':'預測對去年同季實績','EPSRevisionRelative':'修正相對舊單季預測'}
eps=eps.set_index('Feature').loc[list(short)].reset_index()
table([['EPS特徵','舊100倍Sharpe','不排除Sharpe']]+[[short[r.Feature],f'{r.OldSharpe:+.6f}',f'{r.Sharpe:+.6f}'] for r in eps.itertuples()],[145,99,98],10)
p('這表示先前EPS結果對訓練門檻敏感。取消排除也會改變實際學習率，因此不能直接推論EPS資訊普遍無用。',11)
p('Sharpe點估計最高的現行項目',14,teal)
p('營業利益實際年增＋g：Sharpe +0.014509，高於純g；Rank IC仍為−0.000162。仍需獨立資料確認，不能直接視為通過。',11)
p('保留原EPS算法、exp(-a/9)、每日一次MSE、原值PR1/VR1。所有模型都在完整股票池訓練與驗證；沒有裁切大值。',10,gray)
page(2,'現行16項：同一完整股票池')
p('每一列都是g加一項特徵，13個係數共同訓練；按本次Sharpe由高到低排列。',10.5)
labels={'實績公布時事前預期成長':'事前預期成長¹','預測對去年同季預測':'預測對去年預測','預測對去年同季實績':'預測對去年實績','修正相對舊單季預測':'預期修正'}
rows=[]
for r in df.itertuples():
    name=r.Label
    for a,b in labels.items():name=name.replace(a,b)
    rows.append([name,f'{r.Sharpe:+.6f}',f'{r.RankIC:+.6f}'])
table([['單項財報＋g','Sharpe','Rank IC']]+rows,[164,89,89],9.5)
p('¹ 實績公布時觸發，使用公布前凍結的預測；與Forecast事件時對實績比較不同。',9,gray,gap=6)
p('另有5項已被取代的舊公式，僅保留在詳細報告作歷史對照，未重新納入現行模型。',9,gray,gap=6)
p('全部逐日預測、排名、更新與指標已核對。95%重抽樣區間屬探索性，未校正多重比較與過去模型挑選；正式基準仍v7。',9.5,teal)
c.save()
archive=R.parent/'output/JPX-v25-unfiltered-single-features-data-20260916.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    z.write(out,out.name)
    for name in ['comparison_all.csv','comparison_current.csv','comparison_legacy.csv','comparison_eps.csv','comparison_by_year.csv','paired_block_bootstrap.csv','all_coefficients.csv','all_models_daily.csv','feature_catalog.csv','JPX-v25-unfiltered-single-features-report.md','experiment_plan.md','manifest.json','audit.json','results.json','run.py','audit.py','report.py','catalog.json']:
        z.write(R/name,name)
    z.write(R.parent/'JPX-experiment-defaults.json','JPX-experiment-defaults.json')
    for q in json.loads((R/'catalog.json').read_text()):
        for name in ['parameter_history.csv','training_updates.csv','results.json','audit.json','update_audit.json']:
            z.write(R/q['name']/name,q['name']+'/'+name)
print(json.dumps({'pdf':str(out),'zip':str(archive),'zip_bytes':archive.stat().st_size}))
