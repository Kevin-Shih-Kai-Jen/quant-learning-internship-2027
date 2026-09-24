from pathlib import Path
import json,html,zipfile
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph,Table,TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
R=Path(__file__).resolve().parent;out=R.parent/'output/pdf/JPX-v17-revised-financial-mobile-20260915.pdf'
df=pd.read_csv(R/'comparison.csv');yr=pd.read_csv(R/'comparison_by_year.csv');base=pd.read_csv(R.parent/'jpx_v16_single_financial_v7_20260915/price_only/daily_metrics.csv');joint=pd.read_csv(R/'eps_joint_comparison.csv')
pdfmetrics.registerFont(TTFont('C','/System/Library/Fonts/STHeiti Light.ttc',subfontIndex=0));pdfmetrics.registerFont(TTFont('B','/System/Library/Fonts/STHeiti Medium.ttc',subfontIndex=0))
W,H=390,680;M=24;cw=W-2*M;navy=colors.HexColor('#153347');teal=colors.HexColor('#007C82');gray=colors.HexColor('#536B78');light=colors.HexColor('#F2F7F9')
c=canvas.Canvas(str(out),pagesize=(W,H));c.setTitle('v17｜修正預期公式與EPS聯合訓練');c.setAuthor('Quant Learning & Internship 2027');y=0
st=ParagraphStyle('s',fontName='C',fontSize=11.5,leading=17,wordWrap='CJK',textColor=navy)
def p(t,size=11.5,color=navy,gap=9):
    global y
    ss=ParagraphStyle('x',parent=st,fontSize=size,leading=size*1.48,textColor=color);a=Paragraph(t,ss);_,h=a.wrap(cw,600);assert y-h>=41,(y,h,t);a.drawOn(c,M,y-h);y-=h+gap

def page(n,title):
    global y
    if n>1:c.showPage()
    c.setFillColor(navy);c.rect(0,H-7,W,7,fill=1,stroke=0);y=H-31;p('JPX  /  v17  /  2026-09-15',10,teal);p(title,21)
    c.setFont('C',9);c.setFillColor(gray);c.drawString(M,18,'保留 exp(-a/9)  ·  手機版');c.drawRightString(W-M,18,f'{n} / 2')
def table(rows,widths,font=10):
    global y
    rr=[]
    for i,row in enumerate(rows):
        s=ParagraphStyle('t',parent=st,fontName='B' if i==0 else 'C',fontSize=font,leading=font*1.3,textColor=colors.white if i==0 else navy)
        rr.append([Paragraph(html.escape(str(v)),s) for v in row])
    t=Table(rr,colWidths=widths);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),navy),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,light]),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5)]));_,h=t.wrap(cw,600);assert y-h>=41,(y,h);t.drawOn(c,M,y-h);y-=h+10

def short(s):return s.replace('／',' ').replace('事前預測相對實際基期成長','事前预期成長').replace('事前预期','事前預期').replace('實際 YoY 成長','實際年增').replace('預測 YoY 成長','預測年增').replace('預測 QoQ 成長','預測季增')
page(1,'三項預期修正，分母已改正')
p('營收、營業利益、EPS 三項都依你最後指定，改為：',12)
p('(新單季預測 - 舊單季預測) / 舊單季預測',12,teal)
p('新舊預測都用相同已公布累計實績與剩餘季度數計算。分母直接保留舊值正負號；-2改成-1為-50%。舊值為0時不建立該事件。',11)
p('完整953日的v7基準Sharpe：+0.00767887。下列皆未年化、未扣成本。',11)
table([['新模型','Sharpe','Δ 對 v7']]+[[r.Label,f'{r.Sharpe:.6f}',f'{r.DeltaVsV7:+.6f}'] for r in df.itertuples()],[170,82,90],10)
p('EPS預測對已知實際：上一季、去年同季兩種比較仍各用實際基期的絕對值當分母。只有「預期修正」依本次指定用帶正負號的舊預測。',10.5,gray)
p('EPS修正提高；營收與營業利益修正仍低於v7。公式符合你的定義，不保證其選股效果一定改善。',11)
page(2,'EPS季增＋年增，效果如何？')
p('聯合模型：g + gammaQ × 實際季增 + gammaY × 實際年增。兩個係數各自學習，價量係數也共同更新。',12)
names={'eps_actual_joint':'季增＋年增','eps_qoq_joint_mask':'只有季增','eps_yoy_joint_mask':'只有年增','eps_actual_joint_control':'只有價量'}
table([['相同訓練樣本','Sharpe']]+[[names[r.Variant],f'{r.Sharpe:+.6f}'] for r in joint.itertuples()],[222,120],12)
p('聯合模型高於單獨年增，但低於單獨季增。四組都套用相同聯合門檻，排除1,848個到期訓練股票日，預測仍保留全股票池。',11.5)
p('條件一致時，季增單獨的表現較好；這一輪尚未看到加入年增能進一步提高Sharpe。聯合訓練本身不等於季節或年度調整。',11.5)
p('沒有重跑完全相同的舊模型。舊EPS「事前預期成長」雖與實績比較，但在實績公布時才觸發；新版本在Forecast事件時與已知實績比較，時點不同，因此需要重跑。',10.5,gray)
p('所有新試驗維持v7每日一次普通MSE、PR1/VR1原值、exp(-a/9)與100倍訓練門檻。共16組新模型及對照、88,251筆新事件，更新、公式、排名與收益核對通過。',10.5,gray)
p('結果來自同一段已反覆研究的歷史驗證，未使用新保留測試期。完整公式、分年度表現、同樣樣本對照和負基期事件數在資料包中；正式基準仍為v7。',10.5,gray)
c.save()
archive=R.parent/'output/JPX-v17-revised-financial-data-20260915.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    z.write(out,out.name)
    for name in ['comparison.csv','comparison_by_year.csv','eps_joint_comparison.csv','revision_denominator_diagnostics.csv','all_models_daily.csv','all_coefficients.csv','JPX-v17-revised-financial-report.md','experiment_plan.md','manifest.json','audit.json','feature_audit.json','results.json','delivery_audit.json']:
        z.write(R/name,name)
print(json.dumps({'pdf':str(out),'pdf_bytes':out.stat().st_size,'zip':str(archive),'zip_bytes':archive.stat().st_size}))
