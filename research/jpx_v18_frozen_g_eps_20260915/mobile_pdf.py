from pathlib import Path
import json,html,zipfile
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph,Table,TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
R=Path(__file__).resolve().parent;out=R.parent/'output/pdf/JPX-v18-frozen-g-mobile-20260915.pdf'
df=pd.read_csv(R/'comparison.csv');yr=pd.read_csv(R/'comparison_by_year.csv');base=pd.read_csv(R.parent/'jpx_v16_single_financial_v7_20260915/price_only/daily_metrics.csv')
pdfmetrics.registerFont(TTFont('C','/System/Library/Fonts/STHeiti Light.ttc',subfontIndex=0));pdfmetrics.registerFont(TTFont('B','/System/Library/Fonts/STHeiti Medium.ttc',subfontIndex=0))
W,H=390,680;M=24;cw=W-2*M;navy=colors.HexColor('#153347');teal=colors.HexColor('#007C82');gray=colors.HexColor('#536B78');light=colors.HexColor('#F2F7F9')
c=canvas.Canvas(str(out),pagesize=(W,H));c.setTitle('v18｜固定g，EPS季增與年增');c.setAuthor('Quant Learning & Internship 2027');y=0
st=ParagraphStyle('s',fontName='C',fontSize=11.5,leading=17,wordWrap='CJK',textColor=navy)
def p(t,size=11.5,color=navy,gap=9):
    global y
    ss=ParagraphStyle('x',parent=st,fontSize=size,leading=size*1.48,textColor=color);a=Paragraph(t,ss);_,h=a.wrap(cw,600);assert y-h>=41,(y,h,t);a.drawOn(c,M,y-h);y-=h+gap

def page(n,title):
    global y
    if n>1:c.showPage()
    c.setFillColor(navy);c.rect(0,H-7,W,7,fill=1,stroke=0);y=H-31;p('JPX  /  v18  /  2026-09-15',10,teal);p(title,21)
    c.setFont('C',9);c.setFillColor(gray);c.drawString(M,18,'保留 exp(-a/9)  ·  手機版');c.drawRightString(W-M,18,f'{n} / 2')
def table(rows,widths,font=10):
    global y
    rr=[]
    for i,row in enumerate(rows):
        s=ParagraphStyle('t',parent=st,fontName='B' if i==0 else 'C',fontSize=font,leading=font*1.3,textColor=colors.white if i==0 else navy)
        rr.append([Paragraph(html.escape(str(v)),s) for v in row])
    t=Table(rr,colWidths=widths);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),navy),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,light]),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5)]));_,h=t.wrap(cw,600);assert y-h>=41,(y,h);t.drawOn(c,M,y-h);y-=h+10

page(1,'固定 g，財報有額外貢獻嗎？')
p('固定原 v7 在歷史當天的預測，只訓練 EPS 財報係數。三組都低於原 v7；合用也沒有超過任一單獨組。',12)
p('共同953日，2017-12-29 至 2021-12-01。Sharpe未年化、未扣交易成本。',10.5,gray)
table([['模型','Sharpe','相對 v7']]+[[r.Label,f'{r.Sharpe:+.6f}',f'{r.DeltaVsV7:+.6f}'] for r in df.itertuples()],[166,88,88],10.5)
p('三組的預測式',14,teal)
p('季增：g + bQ × 季增<br/>年增：g + bY × 年增<br/>聯合：g + bQ × 季增 + bY × 年增',12)
p('只學 bQ、bY，沒有新增截距。g 保留 v7 原本的歷史預測路徑，財報 loss 不會回頭修改價量參數。',11)
p('共同設定',14,teal)
p('普通 MSE，每日一次更新；只使用已到期報酬。三組共用同一批股票與每日學習步長，保留 exp(-a/9) 衰減。',11)
p('任一財報值絕對值 > 100 就跳過該股票日訓練，共排除1,848筆；合格2,323,958筆。預測保留完整股票池。',10.5,gray)
page(2,'如何解讀這次的差距')
p('聯合相對單獨季增，Sharpe低0.000308；相對單獨年增，低0.000192。差距很小。',12)
p('20日區塊、4,000次配對重抽樣：聯合減季增的95%區間為 [-0.005400, +0.005515]，包含零。其他比較的區間也都包含零。',11)
p('因此現在的結論是「尚未看到改善」，還不能證明年增與季增有穩定的負向干擾。',12,teal)
table([['分期','原 g','季增','年增','合用']]+[[str(y)]+[f'{yr[(yr.Variant==n)&(yr.ValidationYear==y)].Sharpe.iloc[0]:+.4f}' for n in ['g','qoq','yoy','joint']] for y in [2018,2019,2020,2021]],[50,73,73,73,73],10)
p('分期沿用既有驗證標籤，不是嚴格曆年。季增在2020、2021改善，但2018、2019較差，效果沒有跨期一致。',10.5)
p('舊版的季增優勢沒有保留',14,teal)
p('舊 v17 讓價量一起學習，單獨季增 Sharpe 為0.011213；固定 g 後是0.006584。舊優勢不能直接視為獨立的季增加分效果。新舊步長及訓練時價量預測處理也有差異，不能把全部差距歸因於某一原因。',10.5)
p('本輪沿用既有 EPS 累計值差分拆季近似與事件定義，沒有同時修正資料。這仍是反覆研究過的同一歷史期間；正式基準維持 v7。',10.5,gray)
p('逐日預測、排名、每次更新與官方Sharpe均核對通過。資料包含完整比較、逐日損益、分期、係數歷史與差值區間。',10.5,gray)
c.save()
archive=R.parent/'output/JPX-v18-frozen-g-data-20260915.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    z.write(out,out.name)
    for name in ['comparison.csv','comparison_by_year.csv','all_models_daily.csv','all_coefficients.csv','paired_block_bootstrap.csv','JPX-v18-frozen-g-report.md','experiment_plan.md','manifest.json','audit.json','update_audit.json','results.json','run.py','audit.py','report.py']:
        z.write(R/name,name)
    for model in ['qoq','yoy','joint']:
        for name in ['parameter_history.csv','training_updates.csv','results.json']:
            z.write(R/model/name,model+'/'+name)
print(json.dumps({'pdf':str(out),'pdf_bytes':out.stat().st_size,'zip':str(archive),'zip_bytes':archive.stat().st_size}))
