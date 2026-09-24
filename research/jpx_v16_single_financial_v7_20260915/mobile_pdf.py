from pathlib import Path
import json,html,zipfile
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph,Table,TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
R=Path(__file__).resolve().parent;out=R.parent/'output/pdf/JPX-v16-single-financial-mobile-20260915.pdf'
df=pd.read_csv(R/'comparison.csv');yr=pd.read_csv(R/'comparison_by_year.csv');base=pd.read_csv(R/'price_only/daily_metrics.csv')
pdfmetrics.registerFont(TTFont('C','/System/Library/Fonts/STHeiti Light.ttc',subfontIndex=0));pdfmetrics.registerFont(TTFont('B','/System/Library/Fonts/STHeiti Medium.ttc',subfontIndex=0))
W,H=390,680;M=24;cw=W-2*M;navy=colors.HexColor('#153347');teal=colors.HexColor('#007C82');gray=colors.HexColor('#536B78');light=colors.HexColor('#F2F7F9')
c=canvas.Canvas(str(out),pagesize=(W,H));c.setTitle('v16｜逐一加入財報特徵');c.setAuthor('Quant Learning & Internship 2027');y=0
st=ParagraphStyle('s',fontName='C',fontSize=11.5,leading=17,wordWrap='CJK',textColor=navy)
def p(t,size=11.5,color=navy,gap=9):
    global y
    ss=ParagraphStyle('x',parent=st,fontSize=size,leading=size*1.48,textColor=color);a=Paragraph(t,ss);_,h=a.wrap(cw,600);assert y-h>=41,(y,h,t);a.drawOn(c,M,y-h);y-=h+gap

def page(n,title):
    global y
    if n>1:c.showPage()
    c.setFillColor(navy);c.rect(0,H-7,W,7,fill=1,stroke=0);y=H-31;p('JPX  /  v16  /  2026-09-15',10,teal);p(title,21)
    c.setFont('C',9);c.setFillColor(gray);c.drawString(M,18,'保留 exp(-a/9)  ·  手機版');c.drawRightString(W-M,18,f'{n} / 2')
def table(rows,widths,font=10):
    global y
    rr=[]
    for i,row in enumerate(rows):
        s=ParagraphStyle('t',parent=st,fontName='B' if i==0 else 'C',fontSize=font,leading=font*1.3,textColor=colors.white if i==0 else navy)
        rr.append([Paragraph(html.escape(str(v)),s) for v in row])
    t=Table(rr,colWidths=widths);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),navy),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,light]),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5)]));_,h=t.wrap(cw,600);assert y-h>=41,(y,h);t.drawOn(c,M,y-h);y-=h+10

def short(s):return s.replace('／',' ').replace('事前預測相對實際基期成長','事前预期成長').replace('事前预期','事前預期').replace('實際 YoY 成長','實際年增').replace('預測 YoY 成長','預測年增').replace('預測 QoQ 成長','預測季增')
page(1,'15 個特徵，10 個提高 Sharpe')
p('原 v7 Sharpe：<b>+0.00767887</b>。這輪 15 組皆大於 0，其中 10 組高於 v7、5 組較低；判斷有無改善應看相對基準差額。',11.7)
table([['單一財報成分','Sharpe','Δ 對 v7']]+[[short(r.FeatureChinese),f'{r.Sharpe:.6f}',f'{r.DeltaSharpeVsV7:+.6f}'] for r in df.itertuples()],[170,82,90],10)
p('相同 953 日，2017-12-29 至 2021-12-01。Sharpe 未年化、未扣成本。每組只有一個財報成分，價量係數與新增係數一起訓練。',10.5,gray)
p('「事前預期成長」指事前季預測相對去年同季實際值的成長；不是 Forecast 原值。EPS 也使用成長率。',10.5,gray)
page(2,'如何判讀這輪結果')
p('最有改善的是營業利益實際年增：Sharpe +0.011694，比 v7 高 +0.004015；相同訓練樣本下仍提高 +0.004314。四個分期中，有三期高於原 v7。',12)
table([['驗證組別','v7','+ 營業利益實際年增']]+[[int(y),f'{g.OfficialDailySpread.mean()/g.OfficialDailySpread.std(ddof=1):+.5f}',f'{yr[(yr.Feature=="OperatingProfitActualGrowth")&(yr.ValidationYear==y)].Sharpe.iloc[0]:+.5f}'] for y,g in base.groupby('ValidationYear')],[87,89,166],10.7)
p('目前值得優先檢查的五項：EPS 實際年增、EPS 預測季增、EPS 預測年增、營業利益預期修正、營收預期修正。這五項皆低於原 v7 與相同樣本的純價量對照。',11.5)
p('營業利益預測季增雖比 v7 高約 0.000688，但比相同樣本對照僅高約 0.000053；多數差異伴隨訓練樣本篩選，不宜全部算成特徵帶來的改善。',11.5)
p('實驗固定：每日一次普通 MSE 更新、PR1/VR1 原值、財報 exp(-a/9)。每次只依所加入成分的 abs(F)>100 排除訓練；預測仍含全股票池。',10.5,gray)
p('這是單欄原始成分檢查，沒有同時估計 beta × (u - d × v) 的完整驚喜假說。新增欄也會影響 v7 的步長；結果限於此定義、尺度與訓練法。',10.5,gray)
p('31 組（基準 + 15 單欄 + 15 篩選對照）均通過更新、排名及收益核對。這是同一歷史驗證期的探索；尚不能以單次下降否定整個財報假說，也不能把最佳結果視為新測試集表現。',10.5,gray)
c.save()
archive=R.parent/'output/JPX-v16-single-financial-data-20260915.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    z.write(out,out.name)
    for name in ['comparison.csv','comparison_by_year.csv','all_models_daily.csv','all_coefficients.csv','JPX-v16-single-financial-report.md','experiment_plan.md','manifest.json','audit.json','results.json','delivery_audit.json']:
        z.write(R/name,name)
print(json.dumps({'pdf':str(out),'pdf_bytes':out.stat().st_size,'zip':str(archive),'zip_bytes':archive.stat().st_size}))
