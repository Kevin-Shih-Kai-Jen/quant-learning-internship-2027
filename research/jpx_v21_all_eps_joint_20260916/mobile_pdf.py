from pathlib import Path
import json,html,zipfile
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph,Table,TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
R=Path(__file__).resolve().parent;out=R.parent/'output/pdf/JPX-v21-all-eps-joint-mobile-20260916.pdf'
df=pd.read_csv(R/'comparison.csv');yr=pd.read_csv(R/'comparison_by_year.csv');base=pd.read_csv(R.parent/'jpx_v16_single_financial_v7_20260915/price_only/daily_metrics.csv')
pdfmetrics.registerFont(TTFont('C','/System/Library/Fonts/STHeiti Light.ttc',subfontIndex=0));pdfmetrics.registerFont(TTFont('B','/System/Library/Fonts/STHeiti Medium.ttc',subfontIndex=0))
W,H=390,680;M=24;cw=W-2*M;navy=colors.HexColor('#153347');teal=colors.HexColor('#007C82');gray=colors.HexColor('#536B78');light=colors.HexColor('#F2F7F9')
c=canvas.Canvas(str(out),pagesize=(W,H));c.setTitle('v21｜g加上全部EPS共同訓練');c.setAuthor('Quant Learning & Internship 2027');y=0
st=ParagraphStyle('s',fontName='C',fontSize=11.5,leading=17,wordWrap='CJK',textColor=navy)
def p(t,size=11.5,color=navy,gap=9):
    global y
    ss=ParagraphStyle('x',parent=st,fontSize=size,leading=size*1.48,textColor=color);a=Paragraph(t,ss);_,h=a.wrap(cw,600);assert y-h>=41,(y,h,t);a.drawOn(c,M,y-h);y-=h+gap

def page(n,title):
    global y
    if n>1:c.showPage()
    c.setFillColor(navy);c.rect(0,H-7,W,7,fill=1,stroke=0);y=H-31;p('JPX  /  v21  /  2026-09-16',10,teal);p(title,21)
    c.setFont('C',9);c.setFillColor(gray);c.drawString(M,18,'保留 exp(-a/9)  ·  手機版');c.drawRightString(W-M,18,f'{n} / 2')
def table(rows,widths,font=10):
    global y
    rr=[]
    for i,row in enumerate(rows):
        s=ParagraphStyle('t',parent=st,fontName='B' if i==0 else 'C',fontSize=font,leading=font*1.3,textColor=colors.white if i==0 else navy)
        rr.append([Paragraph(html.escape(str(v)),s) for v in row])
    t=Table(rr,colWidths=widths);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),navy),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,light]),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5)]));_,h=t.wrap(cw,600);assert y-h>=41,(y,h);t.drawOn(c,M,y-h);y-=h+10

page(1,'g＋全部EPS，共同訓練結果')
p('依你的要求，保留目前EPS算法，讓價量g與六個EPS係數一起學習，共18個參數。',12)
p('共同953日，2017-12-29至2021-12-01。Sharpe未年化、未扣交易成本。',10.5,gray)
table([['模型','Sharpe','Δ對v7']]+[[r.Label,f'{r.Sharpe:+.6f}',f'{r.DeltaVsV7:+.6f}'] for r in df.itertuples()],[166,88,88],10)
p('合併版Sharpe接近零，低於原v7及兩個純價量對照。同樣本、同步長的對照仍為正，差距不能只歸因於樣本篩選或步長改變。',11)
table([['分期','原v7','g＋六項EPS']]+[[str(y)]+[f'{yr[(yr.Variant==n)&(yr.ValidationYear==y)].Sharpe.iloc[0]:+.5f}' for n in ['v7','all_eps']] for y in [2018,2019,2020,2021]],[62,140,140],11)
p('保留的設定',14,teal)
p('每日一次普通MSE、原值PR1/VR1、exp(-a/9)衰減。從零初始化，跨年連續訓練，只用已到期Target更新。',10.5)
p('任一EPS訊號絕對值超過100倍即跳過該股票日訓練。三組共同使用2,322,781筆、排除3,025筆；預測仍保留完整股票池。',10.5,gray)
page(2,'六個EPS係數，全部各自估計')
fdefs=pd.read_csv(R/'feature_definitions.csv');coef=pd.read_csv(R/'all_coefficients.csv');coef=coef[coef.Variant=='all_eps'].set_index('Parameter')
table([['EPS特徵','最後係數']]+[[r.Label,f'{coef.loc[r.Feature,"Value"]:+.7f}'] for r in fdefs.itertuples()],[239,103],10.5)
p('價量12個參數也共同更新。表中係數是2021-12-03最後更新後的數值；回測各日使用當時係數，沒有用期末參數回填。',10.5)
p('事件時間有差別',14,teal)
p('「事前預期成長」在實績公布時使用公布前已凍結預測；「預測對前季／去年實績」在Forecast事件時觸發。各自保留係數，沒有把已刪除的預測對預測版本加回。',10.5)
p('差距的不確定性',14,teal)
ci=pd.read_csv(R/'paired_block_bootstrap.csv');labels={'all_eps - v7':'對原v7','all_eps - price_mask':'對同樣本g','all_eps - price_mask_step':'對同樣本同步長g'}
table([['合併版減去對照','95%差值區間']]+[[labels[r.Comparison],f'[{r.CI95Low:+.5f}, {r.CI95High:+.5f}]'] for r in ci.itertuples()],[159,183],10)
p('20日區塊、4,000次配對重抽樣，區間都跨過0。屬探索性比較，未校正反覆嘗試模型，尚不能認定EPS必然拖累表現。',10,gray)
p('這輪回答的是「全部合併是否改善」；不能因此認定每個EPS都無效。完整預測、排名、更新和官方指標均已核對，正式基準維持v7。',10.5,teal)
c.save()
archive=R.parent/'output/JPX-v21-all-eps-joint-data-20260916.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    z.write(out,out.name)
    for name in ['comparison.csv','comparison_by_year.csv','feature_definitions.csv','all_models_daily.csv','all_coefficients.csv','paired_block_bootstrap.csv','JPX-v21-all-eps-joint-report.md','experiment_plan.md','manifest.json','audit.json','update_audit.json','results.json','run.py','audit.py','report.py']:
        z.write(R/name,name)
    for model in ['all_eps','price_mask','price_mask_step']:
        for name in ['parameter_history.csv','training_updates.csv','results.json']:
            z.write(R/model/name,model+'/'+name)
print(json.dumps({'pdf':str(out),'zip':str(archive),'zip_bytes':archive.stat().st_size}))
