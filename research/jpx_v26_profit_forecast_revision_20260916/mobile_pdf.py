from pathlib import Path
import json,html,zipfile
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph,Table,TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
R=Path(__file__).resolve().parent;out=R.parent/'output/pdf/JPX-v26-profit-forecast-revision-mobile-20260916.pdf'
df=pd.read_csv(R/'comparison.csv');yr=pd.read_csv(R/'comparison_by_year.csv');base=pd.read_csv(R.parent/'jpx_v16_single_financial_v7_20260915/price_only/daily_metrics.csv')
pdfmetrics.registerFont(TTFont('C','/System/Library/Fonts/STHeiti Light.ttc',subfontIndex=0));pdfmetrics.registerFont(TTFont('B','/System/Library/Fonts/STHeiti Medium.ttc',subfontIndex=0))
W,H=390,680;M=24;cw=W-2*M;navy=colors.HexColor('#153347');teal=colors.HexColor('#007C82');gray=colors.HexColor('#536B78');light=colors.HexColor('#F2F7F9')
c=canvas.Canvas(str(out),pagesize=(W,H));c.setTitle('v26｜營業利益預測加修正');c.setAuthor('Quant Learning & Internship 2027');y=0
st=ParagraphStyle('s',fontName='C',fontSize=11.5,leading=17,wordWrap='CJK',textColor=navy)
def p(t,size=11.5,color=navy,gap=9):
    global y
    ss=ParagraphStyle('x',parent=st,fontSize=size,leading=size*1.48,textColor=color);a=Paragraph(t,ss);_,h=a.wrap(cw,600);assert y-h>=41,(y,h,t);a.drawOn(c,M,y-h);y-=h+gap

def page(n,title):
    global y
    if n>1:c.showPage()
    c.setFillColor(navy);c.rect(0,H-7,W,7,fill=1,stroke=0);y=H-31;p('JPX  /  v26  /  2026-09-16',10,teal);p(title,21)
    c.setFont('C',9);c.setFillColor(gray);c.drawString(M,18,'不排除極端值  ·  手機版');c.drawRightString(W-M,18,f'{n} / 2')
def table(rows,widths,font=10):
    global y
    rr=[]
    for i,row in enumerate(rows):
        s=ParagraphStyle('t',parent=st,fontName='B' if i==0 else 'C',fontSize=font,leading=font*1.3,textColor=colors.white if i==0 else navy)
        rr.append([Paragraph(html.escape(str(v)),s) for v in row])
    t=Table(rr,colWidths=widths);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),navy),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,light]),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5)]));_,h=t.wrap(cw,600);assert y-h>=41,(y,h);t.drawOn(c,M,y-h);y-=h+10

page(1,'營業利益：預測＋修正＋g')
p('價量g、預測對去年同季預測、相對舊單季預測的修正，各有係數共同學習，共14個參數。',12)
p('訓練與驗證均不排除極端值。953日驗證；Sharpe未年化、未扣成本；Rank IC有效952日。',10.5,gray)
labels={'g':'純價量g','forecast':'g＋預測對去年預測','revision':'g＋預期修正','joint':'g＋預測＋修正'}
table([['模型','Sharpe','Rank IC']]+[[labels[r.Variant],f'{r.Sharpe:+.6f}',f'{r.RankIC:+.6f}'] for r in df.itertuples()],[146,98,98],10.5)
p('組合Sharpe高於純g，但低於單獨加入預測年比，這次沒有成為最高。Rank IC是四組中最高，但仍略為負值。',11)
table([['分期Sharpe','純g','g＋預測','組合']]+[[str(y)]+[f'{yr[(yr.Variant==n)&(yr.ValidationYear==y)].Sharpe.iloc[0]:+.5f}' for n in ['g','forecast','joint']] for y in [2018,2019,2020,2021]],[84,86,86,86],10)
p('相對兩個主要對照',14,teal)
p('對純g：ΔSharpe +0.003837。<br/>對g＋預測年比：ΔSharpe −0.001141。<br/>兩項差值的95%區間都跨0，尚無可靠勝出證據。',11)
p('全部1199次更新與2,326,022筆歷史預測、排名及驗證指標已核對。正式基準維持v7。',10.5,gray)
page(2,'加入修正，不等於覆蓋舊訊號')
p('你提出的理由是：預測變更後，模型應及時使用更新的資訊。這次修正資訊確實有加入，但實作方式需要區分。',11.5)
p('本輪模型',14,teal)
p('預測報酬＝g＋bY×預測年比訊號＋bR×修正訊號。兩個係數獨立估計，兩欄事件各自按exp(-a/9)累加衰減。',11)
p('資料內部的最新Forecast會更新；舊年比事件仍繼續衰減，修正事件另行加入，沒有把舊年比直接換成新的年比。',11)
p('為何兩者不完全相同？',14,teal)
p('預測年比的分母是「去年同季預測絕對值」；修正的分母是「本季舊預測」。不同基期的兩個比率相加，不一定剛好等於更新後的年比。',11)
p('這次測到的是兩項訊息的組合效果；尚未測試「每次修正後重算最新預測年比並覆蓋舊值」的另一種模型。',11,teal)
p('Sharpe差值的不確定性',14,teal)
ci=pd.read_csv(R/'paired_block_bootstrap.csv');ci=ci[ci.Metric=='Sharpe'];ref={'joint - g':'組合減純g','joint - forecast':'組合減g＋預測','joint - revision':'組合減g＋修正'}
table([['比較','95%差值區間']]+[[ref[r.Comparison],f'[{r.CI95Low:+.5f}, {r.CI95High:+.5f}]'] for r in ci.itertuples()],[145,197],10)
p('20日區塊、4,000次同日期配對重抽樣，未校正多重比較與過去模型選擇。未達顯著差異不等於證明修正資訊無效。',9.5,gray)
c.save()
archive=R.parent/'output/JPX-v26-profit-forecast-revision-data-20260916.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    z.write(out,out.name)
    for name in ['comparison.csv','comparison_by_year.csv','paired_block_bootstrap.csv','all_models_daily.csv','all_coefficients.csv','learning_rate_comparison.csv','JPX-v26-profit-forecast-revision-report.md','experiment_plan.md','manifest.json','audit.json','results.json','run.py','audit.py','report.py']:
        z.write(R/name,name)
    for name in ['daily_metrics.csv','training_updates.csv','parameter_history.csv','results.json','audit.json','update_audit.json']:
        z.write(R/'joint'/name,'joint/'+name)
print(json.dumps({'pdf':str(out),'zip':str(archive),'zip_bytes':archive.stat().st_size}))
