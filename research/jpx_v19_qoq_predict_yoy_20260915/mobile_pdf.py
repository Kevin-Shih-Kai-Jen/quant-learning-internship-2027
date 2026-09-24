from pathlib import Path
import json,html,zipfile
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph,Table,TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
R=Path(__file__).resolve().parent;out=R.parent/'output/pdf/JPX-v19-qoq-predict-yoy-mobile-20260915.pdf'
df=pd.read_csv(R/'comparison.csv');yr=pd.read_csv(R/'comparison_by_year.csv');base=pd.read_csv(R.parent/'jpx_v16_single_financial_v7_20260915/price_only/daily_metrics.csv')
pdfmetrics.registerFont(TTFont('C','/System/Library/Fonts/STHeiti Light.ttc',subfontIndex=0));pdfmetrics.registerFont(TTFont('B','/System/Library/Fonts/STHeiti Medium.ttc',subfontIndex=0))
W,H=390,680;M=24;cw=W-2*M;navy=colors.HexColor('#153347');teal=colors.HexColor('#007C82');gray=colors.HexColor('#536B78');light=colors.HexColor('#F2F7F9')
c=canvas.Canvas(str(out),pagesize=(W,H));c.setTitle('v19｜季增能預測年增多少');c.setAuthor('Quant Learning & Internship 2027');y=0
st=ParagraphStyle('s',fontName='C',fontSize=11.5,leading=17,wordWrap='CJK',textColor=navy)
def p(t,size=11.5,color=navy,gap=9):
    global y
    ss=ParagraphStyle('x',parent=st,fontSize=size,leading=size*1.48,textColor=color);a=Paragraph(t,ss);_,h=a.wrap(cw,600);assert y-h>=41,(y,h,t);a.drawOn(c,M,y-h);y-=h+gap

def page(n,title):
    global y
    if n>1:c.showPage()
    c.setFillColor(navy);c.rect(0,H-7,W,7,fill=1,stroke=0);y=H-31;p('JPX  /  v19  /  2026-09-15',10,teal);p(title,21)
    c.setFont('C',9);c.setFillColor(gray);c.drawString(M,18,'Expanding OLS  ·  手機版');c.drawRightString(W-M,18,f'{n} / 2')
def table(rows,widths,font=10):
    global y
    rr=[]
    for i,row in enumerate(rows):
        s=ParagraphStyle('t',parent=st,fontName='B' if i==0 else 'C',fontSize=font,leading=font*1.3,textColor=colors.white if i==0 else navy)
        rr.append([Paragraph(html.escape(str(v)),s) for v in row])
    t=Table(rr,colWidths=widths);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),navy),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,light]),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5)]));_,h=t.wrap(cw,600);assert y-h>=41,(y,h);t.drawOn(c,M,y-h);y-=h+10

page(1,'季增能預測多少年增？')
p('模型：預測年增 = a + B × 季增',13,teal)
p('季增 = (本季EPS - 上季EPS) / |上季EPS|<br/>年增 = (本季EPS - 去年同季EPS) / |去年同季EPS|',10.5)
p('這次直接使用每次財報的比率，不衰減、不每天複製。2018作初始訓練；2019-2021每個新公布日，先用更早日期資料估計，再預測當天整批事件。',11)
p('採用過去全部合格樣本的最小平方解，排除SGD步長的影響。第一次驗證已有5,432筆訓練配對。',10.5,gray)
p('判斷依據：MSE相對改善',14,teal)
p('1 - 模型MSE / 歷史平均年增的MSE',12)
table([['評估範圍','模型MSE','對照MSE','改善']]+[[('全部事件' if i==0 else '100倍內'),f'{r.MSE:.3f}',f'{r.BaselineMSE:.3f}',f'{r.MSEImprovement*100:+.2f}%'] for i,r in enumerate(df.itertuples())],[91,85,85,81],10)
p('全部21,993個事件：模型較差1.13%。<br/>兩個比率均在100倍內的21,813個事件：模型改善1.98%。',11)
p('超過100倍仍只跳過訓練，不從完整評估刪掉。180筆極端事件，占約0.82%，對平方誤差的影響很大；第二列僅作條件式診斷。',10.5,gray)
p('比率1代表100%。100倍內的RMSE仍達580.01個百分點，絕對誤差中位數64.38個百分點。相對改善不代表誤差已經很小。',10.5)
page(2,'如何解讀，以及不能下的結論')
p('100倍內的分期MSE改善：2019為2.85%、2020為2.06%、2021為1.46%。資料逐漸增加，改善幅度沒有持續提高。',11)
# The plot uses two separate axes because extreme ratios strongly change the MSE scale.
c.drawImage(str(R/'cumulative_mse.png'),M,y-267,width=cw,height=267,preserveAspectRatio=True,anchor='c');y-=279
p('這是共同線性模型的誤差改善率，不能直接說「取代1.98%的年增資訊」，也不能推論Sharpe改善。',11,teal)
p('兩個比率本來就共用本季EPS。若 p 是上季EPS、z 是去年同季EPS，則年增恆等於：',10.5)
p('(p - z) / |z| + (|p| / |z|) × 季增',11,teal)
p('保留兩個舊基準就能還原年增；本次只給季增、讓所有公司共用a與B，測的是這個簡單近似能做到多少。',10.5)
p('沿用既有累計EPS差分拆季近似，股數變化需另查。未改價量模型，正式基準仍是v7。完整數據、逐筆預測與核對結果附於資料包。',10,gray)
c.save()
archive=R.parent/'output/JPX-v19-qoq-predict-yoy-data-20260915.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    z.write(out,out.name)
    for name in ['comparison.csv','comparison_by_year.csv','event_predictions.csv','parameter_history.csv','cumulative_error_both_scopes.csv','cumulative_mse.png','largest_errors.csv','JPX-v19-qoq-predict-yoy-report.md','manifest.json','audit.json','results.json','error_diagnostics.json','run.py','audit.py','report.py']:
        z.write(R/name,name)
print(json.dumps({'pdf':str(out),'zip':str(archive),'zip_bytes':archive.stat().st_size}))
