from pathlib import Path
import json,html,zipfile
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph,Table,TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
R=Path(__file__).resolve().parent;out=R.parent/'output/pdf/JPX-v29-learning-rate-sweep-mobile-20260916.pdf'
df=pd.read_csv(R/'comparison.csv');yr=pd.read_csv(R/'comparison_by_year.csv')
pdfmetrics.registerFont(TTFont('C','/System/Library/Fonts/STHeiti Light.ttc',subfontIndex=0));pdfmetrics.registerFont(TTFont('B','/System/Library/Fonts/STHeiti Medium.ttc',subfontIndex=0))
W,H=390,680;M=24;cw=W-2*M;navy=colors.HexColor('#153347');teal=colors.HexColor('#007C82');gray=colors.HexColor('#536B78');light=colors.HexColor('#F2F7F9')
c=canvas.Canvas(str(out),pagesize=(W,H));c.setTitle('v29｜learning rate倍率比較');c.setAuthor('Quant Learning & Internship 2027');y=0
st=ParagraphStyle('s',fontName='C',fontSize=11.5,leading=17,wordWrap='CJK',textColor=navy)
def p(t,size=11.5,color=navy,gap=9):
    global y
    ss=ParagraphStyle('x',parent=st,fontSize=size,leading=size*1.48,textColor=color);a=Paragraph(t,ss);_,h=a.wrap(cw,600);assert y-h>=41,(y,h,t);a.drawOn(c,M,y-h);y-=h+gap

def page(n,title):
    global y
    if n>1:c.showPage()
    c.setFillColor(navy);c.rect(0,H-7,W,7,fill=1,stroke=0);y=H-31;p('JPX  /  v29  /  2026-09-16',10,teal);p(title,21)
    c.setFont('C',9);c.setFillColor(gray);c.drawString(M,18,'不排除極端值  ·  手機版');c.drawRightString(W-M,18,f'{n} / 2')
def table(rows,widths,font=10):
    global y
    rr=[]
    for i,row in enumerate(rows):
        s=ParagraphStyle('t',parent=st,fontName='B' if i==0 else 'C',fontSize=font,leading=font*1.3,textColor=colors.white if i==0 else navy)
        rr.append([Paragraph(html.escape(str(v)),s) for v in row])
    t=Table(rr,colWidths=widths);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),navy),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,light]),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5)]));_,h=t.wrap(cw,600);assert y-h>=41,(y,h);t.drawOn(c,M,y-h);y-=h+10




page(1,'提高學習率，這輪沒有更好')
p('依你的規則：Sharpe優先，Rank IC其次。MSE不參與選擇。全部使用既有validation，test留待之後總體驗證。',12,teal)
p('共同learning rate × 固定倍率<br/>倍率：0.25、0.5、1、1.5、1.9；每個倍率都跑純g、A覆蓋、B差額＋舊值，共15組。',11)
p('953日驗證，Sharpe未年化、未扣成本。每日一次更新、財報exp(-a/9)、不排除極端值。',10,gray)
models=['g_common','overwrite','change_old'];multipliers=[.25,.5,1,1.5,1.9]
def val(m,c,col):return df[(df.Model==m)&(df.Multiplier==c)][col].iloc[0]
table([['倍率','純g Sharpe','A Sharpe','B Sharpe']]+[[f'{k:g}']+[f'{val(m,k,"Sharpe"):+.6f}' for m in models] for k in multipliers],[45,99,99,99],10)
p('三個模型都在1倍取得最高Sharpe。縮小至0.25或0.5倍、放大至1.5或1.9倍，Sharpe都低於原本的1倍。',11)
p('這輪15組的候選',14,teal)
p('B：g＋r1×(new-old)＋r2×old<br/>倍率1；Sharpe +0.002224<br/>平均Rank IC -0.001371。',12)
p('原正式v7的Sharpe為+0.007679，仍高於本輪最佳。B是這個15組網格中的候選，並未取代v7，也未通過test驗證。',11)
p('三個1倍模型與上一輪v28的全部預測、排名及參數完全重現。同倍率下三模型的每日learning rate與股票日逐值相同。',10,gray)
p('目前能說的是：在本輪範圍內，單純放大步長沒有改善Sharpe，不能把低Sharpe直接歸因於learning rate太小。',11,teal)
page(2,'Rank IC：第二順位，完整保留')
table([['倍率','純g Rank IC','A Rank IC','B Rank IC']]+[[f'{k:g}']+[f'{val(m,k,"RankIC"):+.6f}' for m in models] for k in multipliers],[45,99,99,99],10)
p('1.5倍的Rank IC較1倍好，但Sharpe轉負。依你指定的排序，選1倍；Rank IC負值不直接淘汰，MSE不列為選擇條件。',11)
p('A＝g＋β×new；B＝g＋r1×(new-old)＋r2×old。new、old皆為同基期營業利益預測年比，最新狀態覆蓋舊狀態。',10.5)
p('倍率整段固定；基準步長逐日取A、B原步長較小值。价量與財報共同訓練，資料與訊號規則沿用上一輪。'.replace('价','價'),10.5)
p('1倍最佳設定的分期Sharpe',14,teal)
table([['分期','純g','A','B']]+[[str(vy)]+[f'{yr[(yr.Model==m)&(yr.Multiplier==1)&(yr.ValidationYear==vy)].Sharpe.iloc[0]:+.5f}' for m in models] for vy in [2018,2019,2020,2021]],[78,88,88,88],9.8)
p('Validation選候選，test再驗證',14,teal)
p('本輪按完整validation Sharpe排序，再以Rank IC破同分。最佳點已記錄，原v7保持正式基準。test本輪未讀取或評估。',11)
p('B減同倍率純g的ΔSharpe＝+0.001412；探索性95%區間[-0.001951, +0.004220]。區間未校正挑選最佳模型，不作本輪選擇門檻。',10,gray)
p('每組1199次更新、2,325,806訓練股票日、2,326,022筆預測已核對，極端值排除0筆。完整數據與共同步長排程均在ZIP。',9.5,gray)
c.save()
archive=R.parent/'output/JPX-v29-learning-rate-sweep-data-20260916.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    z.write(out,out.name)
    for name in ['comparison.csv','validation_ranking.csv','best_per_model.csv','comparison_by_year.csv','paired_block_bootstrap.csv','all_models_daily.csv','all_coefficients.csv','common_learning_rates.csv','JPX-v29-learning-rate-sweep-report.md','experiment_plan.md','experiment_defaults_snapshot.json','selected_candidate.json','manifest.json','audit.json','results.json','run.py','audit.py','report.py']:
        z.write(R/name,name)
    for variant in df.Variant:
        for name in ['daily_metrics.csv','training_updates.csv','parameter_history.csv','results.json','audit.json','update_audit.json']:
            z.write(R/variant/name,variant+'/'+name)
    previous=R.parent/'jpx_v27_profit_forecast_state_20260916'
    for name in ['features.py','feature_manifest.json','feature_audit.json','experiment_plan.md']:
        z.write(previous/name,'v27_feature_definition/'+name)
print(json.dumps({'pdf':str(out),'zip':str(archive),'zip_bytes':archive.stat().st_size}))
