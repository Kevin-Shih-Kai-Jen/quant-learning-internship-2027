from pathlib import Path
import json,html,zipfile
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph,Table,TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
R=Path(__file__).resolve().parent;out=R.parent/'output/pdf/JPX-v24-train-validation-filters-mobile-20260916.pdf'
df=pd.read_csv(R/'comparison.csv');yr=pd.read_csv(R/'comparison_by_year.csv');base=pd.read_csv(R.parent/'jpx_v16_single_financial_v7_20260915/price_only/daily_metrics.csv')
pdfmetrics.registerFont(TTFont('C','/System/Library/Fonts/STHeiti Light.ttc',subfontIndex=0));pdfmetrics.registerFont(TTFont('B','/System/Library/Fonts/STHeiti Medium.ttc',subfontIndex=0))
W,H=390,680;M=24;cw=W-2*M;navy=colors.HexColor('#153347');teal=colors.HexColor('#007C82');gray=colors.HexColor('#536B78');light=colors.HexColor('#F2F7F9')
c=canvas.Canvas(str(out),pagesize=(W,H));c.setTitle('v24｜純g訓練與驗證同步篩選');c.setAuthor('Quant Learning & Internship 2027');y=0
st=ParagraphStyle('s',fontName='C',fontSize=11.5,leading=17,wordWrap='CJK',textColor=navy)
def p(t,size=11.5,color=navy,gap=9):
    global y
    ss=ParagraphStyle('x',parent=st,fontSize=size,leading=size*1.48,textColor=color);a=Paragraph(t,ss);_,h=a.wrap(cw,600);assert y-h>=41,(y,h,t);a.drawOn(c,M,y-h);y-=h+gap

def page(n,title):
    global y
    if n>1:c.showPage()
    c.setFillColor(navy);c.rect(0,H-7,W,7,fill=1,stroke=0);y=H-31;p('JPX  /  v24  /  2026-09-16',10,teal);p(title,21)
    c.setFont('C',9);c.setFillColor(gray);c.drawString(M,18,'純價量g  ·  EPS僅作篩選');c.drawRightString(W-M,18,f'{n} / 2')
def table(rows,widths,font=10):
    global y
    rr=[]
    for i,row in enumerate(rows):
        s=ParagraphStyle('t',parent=st,fontName='B' if i==0 else 'C',fontSize=font,leading=font*1.3,textColor=colors.white if i==0 else navy)
        rr.append([Paragraph(html.escape(str(v)),s) for v in row])
    t=Table(rr,colWidths=widths);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),navy),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,light]),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5)]));_,h=t.wrap(cw,600);assert y-h>=41,(y,h);t.drawOn(c,M,y-h);y-=h+10

page(1,'訓練、驗證都跳過極端值')
p('純v7價量g，EPS只用來篩選股票。沿用上一輪三組的每日預測，驗證時先排除，再重新排名與選股。',12)
p('953日驗證，Rank IC有效952日；Sharpe未年化、未扣成本。門檻是六項衰減後EPS訊號的絕對值。',10.5,gray)
primary=df.iloc[:3]
table([['訓練與驗證規則','Sharpe','平均Rank IC']]+[[r.TrainingFilter,f'{r.Sharpe:+.6f}',f'{r.RankIC:+.6f}'] for r in primary.itertuples()],[143,99,100],10.5)
p('兩個門檻的Sharpe仍低於完全不跳過的原v7；Rank IC略微靠近0，但仍為負。',11)
table([['門檻','驗證排除股票日','占比']]+[[r.ValidationFilter,f'{r.ValidationExcludedStockDays:,}',f'{r.ValidationExcludedPct:.3f}%'] for r in primary.itertuples()],[99,153,90],10.5)
p('每次排除後，補足多空各200檔',14,teal)
p('每天剩餘股票最少仍有1,763檔。依g的當日預測，重新選最高200檔做多、最低200檔做空，收益權重維持原規則。',10.5)
p('相較上一輪只在訓練跳過',14,teal)
stages=pd.read_csv(R/'validation_filter_increment.csv')
table([['門檻','前輪Sharpe','本輪Sharpe']]+[[r.Threshold,f'{r.PreviousSharpe:+.6f}',f'{r.FilteredSharpe:+.6f}'] for r in stages.itertuples()],[84,129,129],10.5)
p('100倍組增加驗證篩選後略升；10倍組反而下降。跨門檻的驗證股票池不同，不能把變化全歸因於預測能力。',10.5,teal)
page(2,'改善來自訓練，還是股票池？')
p('另讓原v7只在驗證排除，和「訓練也排除」比較；每一對使用相同的驗證股票池。',11)
order=['none_100','100_100','none_10','10_10'];q=df.set_index('Variant')
table([['訓練篩選','驗證篩選','Sharpe']]+[[q.loc[n,'TrainingFilter'],q.loc[n,'ValidationFilter'],f'{q.loc[n,"Sharpe"]:+.6f}'] for n in order],[114,114,114],10.5)
p('在相同驗證池中，訓練也篩選的Sharpe點估計均較低。原v7只在驗證排除100倍時為+0.008001，僅比原本+0.007679高一點。',11)
p('主實驗相對原v7的差值區間',14,teal)
ci=pd.read_csv(R/'paired_block_bootstrap.csv');rows=[]
for n,label in [('100_100 - none_none','100倍皆篩'),('10_10 - none_none','10倍皆篩')]:
    for metric in ['Sharpe','RankIC']:
        a=ci[(ci.Comparison==n)&(ci.Metric==metric)].iloc[0];rows.append([label+' '+metric,f'[{a.CI95Low:+.5f}, {a.CI95High:+.5f}]'])
table([['比較指標','95%差值區間']]+rows,[151,191],10)
p('20日區塊、4,000次配對重抽樣。探索性結果，未校正多重比較與過去模型挑選，不是新保留測試集。',9.5,gray)
p('可執行的策略篩選，但評分範圍不同',14,teal)
p('篩選只用當日已知EPS訊號，不使用未來收益。此結果是在剩餘股票池上的策略回測，和JPX要求全體股票排名的完整股票池評分有別。',10.5)
p('預測、篩選集合、重排名、損益及Rank IC均已獨立核對；原v7重現通過，既有訓練結果未改動，正式基準維持v7。',10.5,teal)
c.save()
archive=R.parent/'output/JPX-v24-train-validation-filters-data-20260916.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    z.write(out,out.name)
    for name in ['comparison.csv','comparison_by_year.csv','paired_block_bootstrap.csv','rank_ic_level_intervals.csv','validation_filter_increment.csv','all_models_daily.csv','JPX-v24-train-validation-filter-report.md','experiment_plan.md','manifest.json','audit.json','results.json','run.py','audit.py','report.py']:
        z.write(R/name,name)
    for model in ['none_none','100_100','10_10','none_100','none_10']:
        for name in ['daily_metrics.csv','results.json']:
            z.write(R/model/name,model+'/'+name)
print(json.dumps({'pdf':str(out),'zip':str(archive),'zip_bytes':archive.stat().st_size}))
