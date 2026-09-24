from pathlib import Path
import json,html,zipfile
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph,Table,TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
R=Path(__file__).resolve().parent;out=R.parent/'output/pdf/JPX-v22-eps-thresholds-mobile-20260916.pdf'
df=pd.read_csv(R/'comparison.csv');yr=pd.read_csv(R/'comparison_by_year.csv');base=pd.read_csv(R.parent/'jpx_v16_single_financial_v7_20260915/price_only/daily_metrics.csv')
pdfmetrics.registerFont(TTFont('C','/System/Library/Fonts/STHeiti Light.ttc',subfontIndex=0));pdfmetrics.registerFont(TTFont('B','/System/Library/Fonts/STHeiti Medium.ttc',subfontIndex=0))
W,H=390,680;M=24;cw=W-2*M;navy=colors.HexColor('#153347');teal=colors.HexColor('#007C82');gray=colors.HexColor('#536B78');light=colors.HexColor('#F2F7F9')
c=canvas.Canvas(str(out),pagesize=(W,H));c.setTitle('v22｜EPS極端值訓練門檻比較');c.setAuthor('Quant Learning & Internship 2027');y=0
st=ParagraphStyle('s',fontName='C',fontSize=11.5,leading=17,wordWrap='CJK',textColor=navy)
def p(t,size=11.5,color=navy,gap=9):
    global y
    ss=ParagraphStyle('x',parent=st,fontSize=size,leading=size*1.48,textColor=color);a=Paragraph(t,ss);_,h=a.wrap(cw,600);assert y-h>=41,(y,h,t);a.drawOn(c,M,y-h);y-=h+gap

def page(n,title):
    global y
    if n>1:c.showPage()
    c.setFillColor(navy);c.rect(0,H-7,W,7,fill=1,stroke=0);y=H-31;p('JPX  /  v22  /  2026-09-16',10,teal);p(title,21)
    c.setFont('C',9);c.setFillColor(gray);c.drawString(M,18,'保留 exp(-a/9)  ·  手機版');c.drawRightString(W-M,18,f'{n} / 2')
def table(rows,widths,font=10):
    global y
    rr=[]
    for i,row in enumerate(rows):
        s=ParagraphStyle('t',parent=st,fontName='B' if i==0 else 'C',fontSize=font,leading=font*1.3,textColor=colors.white if i==0 else navy)
        rr.append([Paragraph(html.escape(str(v)),s) for v in row])
    t=Table(rr,colWidths=widths);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),navy),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,light]),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5)]));_,h=t.wrap(cw,600);assert y-h>=41,(y,h);t.drawOn(c,M,y-h);y-=h+10

page(1,'EPS極端值，要跳過多少？')
p('三組都使用價量g＋全部六項EPS，共18個參數共同訓練；EPS算法保持原樣。',12)
p('953日驗證，2017-12-29至2021-12-01。Sharpe未年化、未扣成本；MSE越低表示預測誤差越小。',10.5,gray)
table([['訓練規則','Sharpe','預測MSE']]+[[r.Label,f'{r.Sharpe:+.6f}',f'{r.DailyMeanMSE:.7f}'] for r in df.itertuples()],[140,98,104],11)
p('10倍組Sharpe略高，但MSE比100倍組差29.22%；不跳過組MSE較低16.77%，Sharpe卻最差。MSE與選股獲利衡量不同事情。',11)
table([['分期','100倍','10倍','不跳過']]+[[str(y)]+[f'{yr[(yr.Variant==n)&(yr.ValidationYear==y)].Sharpe.iloc[0]:+.5f}' for n in ['skip100','skip10','no_skip']] for y in [2018,2019,2020,2021]],[60,94,94,94],10.5)
p('比較方式',14,teal)
p('門檻套在六項衰減累加後EPS訊號：任一絕對值大於門檻就跳過該股票日訓練。10倍＝1000%，100倍＝10000%。',10.5)
p('三組仍對同一完整股票池預測及評分。跳過訓練數：100倍組3,025筆；10倍組32,541筆；不跳過組0筆。',10.5)
p('原v7純價量Sharpe為+0.007679，作歷史參考。此次三組兩兩Sharpe差值區間均跨0，尚無可靠勝出者。',10.5,teal)
page(2,'為什麼篩得更嚴，誤差更大？')
table([['預測尾端','100倍','10倍','不跳過'],['絕對預測P99']+[f'{r.AbsPredictionP99:.2%}' for r in df.itertuples()],['最大絕對預測']+[f'{r.AbsPredictionMax:.2%}' for r in df.itertuples()],['絕對預測>100%']+[str(r.PredictionsAbove100Pct) for r in df.itertuples()]],[99,81,81,81],10)
p('只有訓練跳過，預測仍然保留',14,teal)
p('超過100倍的EPS訊號只占約0.15%驗證股票日，卻占10倍組32.87%的總平方誤差。篩選較嚴仍可能對大值做出過度線性外推；這是合理機制，尚未單獨證明因果。',11)
p('不跳過，也改變了學習速度',14,teal)
p('步長公式相同，但大值會壓小實際步長。不跳過組對100倍組的每日步長比值中位數是0.272；約33%更新日不到對照十分之一，價量g也一起受到影響。',11)
p('改善幅度仍有不確定性',14,teal)
ci=pd.read_csv(R/'paired_block_bootstrap.csv');ci=ci[ci.Metric=='Sharpe'];labels={'skip10 - skip100':'10倍減100倍','no_skip - skip100':'不跳過減100倍','skip10 - no_skip':'10倍減不跳過'}
table([['ΔSharpe比較','95%區間']]+[[labels[r.Comparison],f'[{r.CI95Low:+.4f}, {r.CI95High:+.4f}]'] for r in ci.itertuples()],[155,187],10)
p('20日區塊、4,000次配對重抽樣，未校正多重比較與過往模型嘗試；屬探索性結果。',9.5,gray)
p('三組MSE都高於零預測的0.0005922。不跳過的MSE較低，仍不足以證明有用的選股訊息。正式v7與既有100倍實驗對照均保留。',10.5,teal)
c.save()
archive=R.parent/'output/JPX-v22-eps-thresholds-data-20260916.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    z.write(out,out.name)
    for name in ['comparison.csv','comparison_by_year.csv','mse_by_feature_magnitude.csv','learning_rates.csv','largest_predictions.csv','all_models_daily.csv','paired_block_bootstrap.csv','JPX-v22-threshold-report.md','experiment_plan.md','manifest.json','audit.json','update_audit.json','results.json','run.py','audit.py','report.py']:
        z.write(R/name,name)
    for model in ['skip100','skip10','no_skip']:
        for name in ['parameter_history.csv','training_updates.csv','results.json']:
            z.write(R/model/name,model+'/'+name)
print(json.dumps({'pdf':str(out),'zip':str(archive),'zip_bytes':archive.stat().st_size}))
