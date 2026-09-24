from pathlib import Path
import json,html,zipfile
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph,Table,TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
R=Path(__file__).resolve().parent;out=R.parent/'output/pdf/JPX-v23-g-thresholds-mobile-20260916.pdf'
df=pd.read_csv(R/'comparison.csv');yr=pd.read_csv(R/'comparison_by_year.csv');base=pd.read_csv(R.parent/'jpx_v16_single_financial_v7_20260915/price_only/daily_metrics.csv')
pdfmetrics.registerFont(TTFont('C','/System/Library/Fonts/STHeiti Light.ttc',subfontIndex=0));pdfmetrics.registerFont(TTFont('B','/System/Library/Fonts/STHeiti Medium.ttc',subfontIndex=0))
W,H=390,680;M=24;cw=W-2*M;navy=colors.HexColor('#153347');teal=colors.HexColor('#007C82');gray=colors.HexColor('#536B78');light=colors.HexColor('#F2F7F9')
c=canvas.Canvas(str(out),pagesize=(W,H));c.setTitle('v23｜純g的EPS篩選對照');c.setAuthor('Quant Learning & Internship 2027');y=0
st=ParagraphStyle('s',fontName='C',fontSize=11.5,leading=17,wordWrap='CJK',textColor=navy)
def p(t,size=11.5,color=navy,gap=9):
    global y
    ss=ParagraphStyle('x',parent=st,fontSize=size,leading=size*1.48,textColor=color);a=Paragraph(t,ss);_,h=a.wrap(cw,600);assert y-h>=41,(y,h,t);a.drawOn(c,M,y-h);y-=h+gap

def page(n,title):
    global y
    if n>1:c.showPage()
    c.setFillColor(navy);c.rect(0,H-7,W,7,fill=1,stroke=0);y=H-31;p('JPX  /  v23  /  2026-09-16',10,teal);p(title,21)
    c.setFont('C',9);c.setFillColor(gray);c.drawString(M,18,'純價量g  ·  EPS僅作篩選');c.drawRightString(W-M,18,f'{n} / 2')
def table(rows,widths,font=10):
    global y
    rr=[]
    for i,row in enumerate(rows):
        s=ParagraphStyle('t',parent=st,fontName='B' if i==0 else 'C',fontSize=font,leading=font*1.3,textColor=colors.white if i==0 else navy)
        rr.append([Paragraph(html.escape(str(v)),s) for v in row])
    t=Table(rr,colWidths=widths);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),navy),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,light]),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5)]));_,h=t.wrap(cw,600);assert y-h>=41,(y,h);t.drawOn(c,M,y-h);y-=h+10

page(1,'純g模型，只改訓練篩選')
p('只訓練v7的12個價量參數。六項EPS訊號只用來判斷哪些股票日跳過訓練，不放進模型預測。',12)
p('953個驗證日；Rank IC有效952日。Sharpe未年化、未扣成本。EPS門檻沿用衰減累加後的絕對值。',10.5,gray)
g=df[df.Family=='g']
table([['純g訓練規則','Sharpe','平均Rank IC']]+[[r.Label,f'{r.Sharpe:+.6f}',f'{r.RankIC:+.6f}'] for r in g.itertuples()],[145,98,99],10.5)
p('不跳過的Sharpe最高。兩種篩選的Rank IC雖稍微較好，但仍為負，尚未同時改善兩項主要指標。',11)
table([['分期Sharpe','不跳過','100倍','10倍']]+[[str(y)]+[f'{yr[(yr.Model==n)&(yr.ValidationYear==y)].Sharpe.iloc[0]:+.5f}' for n in ['g_no_skip','g_skip100','g_skip10']] for y in [2018,2019,2020,2021]],[81,87,87,87],10)
p('篩選沒有改變評分股票池',14,teal)
p('100倍組跳過3,025個訓練股票日；10倍組跳過32,541個。三組仍對相同完整股票池預測、排序與評分。',10.5)
p('100倍組精確重現先前純g對照，不跳過組重現原v7。全部1199次更新與逐日預測、排名、Rank IC、官方Sharpe已核對。',10.5)
p('不能把極小差值當成可靠改善',14,teal)
p('純g三組的Sharpe及Rank IC差值95%區間都跨0。這次沒有支持改用10倍或100倍篩選的證據，正式基準維持原v7。',10.5,teal)
page(2,'相同樣本，再比較加入EPS')
inc=pd.read_csv(R/'eps_vs_g_same_mask.csv')
table([['訓練門檻','純g Sharpe','g＋EPS Sharpe']]+[[r.Label,f'{r.GSharpe:+.6f}',f'{r.EPSSharpe:+.6f}'] for r in inc.itertuples()],[123,109,110],10.5)
table([['訓練門檻','純g Rank IC','g＋EPS Rank IC']]+[[r.Label,f'{r.GRankIC:+.6f}',f'{r.EPSRankIC:+.6f}'] for r in inc.itertuples()],[123,109,110],10.5)
p('三個門檻下，加入全部EPS的Sharpe點估計都比同樣本純g低。10倍組的Rank IC有微小提升，但仍為負，且差值區間包含0。',11)
p('這個對照解決了什麼？',14,teal)
p('兩邊每天使用同一批訓練股票，所以不會把「樣本被排除」與「EPS放進模型」完全混在一起。',11)
p('還有什麼尚未分離？',14,teal)
p('純g用12維矩陣算學習率，加入EPS則用18維。實際步長仍可能不同，所以這是整套模型比較，不能直接當成EPS資訊本身的因果貢獻。',11)
p('Rank IC的判讀',14,teal)
p('六個模型的平均Rank IC都略為負，95%區間也都跨0，目前沒有穩定正向排序能力的證據。全0預測的Rank IC未定義，不是可直接比較的0分模型。',10.5)
p('20日區塊、4,000次配對重抽樣；探索性區間未校正多重比較與過去模型挑選。2020-09-29全體Target相同，Rank IC不補0，排除於平均。',9.5,gray)
c.save()
archive=R.parent/'output/JPX-v23-g-thresholds-data-20260916.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    z.write(out,out.name)
    for name in ['comparison.csv','comparison_by_year.csv','paired_block_bootstrap.csv','rank_ic_level_intervals.csv','eps_vs_g_same_mask.csv','eps_vs_g_learning_rates.csv','all_models_daily.csv','JPX-v23-g-threshold-report.md','experiment_plan.md','manifest.json','audit.json','update_audit.json','results.json','run.py','audit.py','report.py']:
        z.write(R/name,name)
    for model in ['skip100','skip10','no_skip']:
        for name in ['parameter_history.csv','training_updates.csv','results.json']:
            z.write(R/model/name,model+'/'+name)
print(json.dumps({'pdf':str(out),'zip':str(archive),'zip_bytes':archive.stat().st_size}))
