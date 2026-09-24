from pathlib import Path
import json,html,zipfile
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph,Table,TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
R=Path(__file__).resolve().parent;out=R.parent/'output/pdf/JPX-v28-common-learning-rate-mobile-20260916.pdf'
df=pd.read_csv(R/'comparison.csv');yr=pd.read_csv(R/'comparison_by_year.csv')
pdfmetrics.registerFont(TTFont('C','/System/Library/Fonts/STHeiti Light.ttc',subfontIndex=0));pdfmetrics.registerFont(TTFont('B','/System/Library/Fonts/STHeiti Medium.ttc',subfontIndex=0))
W,H=390,680;M=24;cw=W-2*M;navy=colors.HexColor('#153347');teal=colors.HexColor('#007C82');gray=colors.HexColor('#536B78');light=colors.HexColor('#F2F7F9')
c=canvas.Canvas(str(out),pagesize=(W,H));c.setTitle('v28｜相同學習率對照');c.setAuthor('Quant Learning & Internship 2027');y=0
st=ParagraphStyle('s',fontName='C',fontSize=11.5,leading=17,wordWrap='CJK',textColor=navy)
def p(t,size=11.5,color=navy,gap=9):
    global y
    ss=ParagraphStyle('x',parent=st,fontSize=size,leading=size*1.48,textColor=color);a=Paragraph(t,ss);_,h=a.wrap(cw,600);assert y-h>=41,(y,h,t);a.drawOn(c,M,y-h);y-=h+gap

def page(n,title):
    global y
    if n>1:c.showPage()
    c.setFillColor(navy);c.rect(0,H-7,W,7,fill=1,stroke=0);y=H-31;p('JPX  /  v28  /  2026-09-16',10,teal);p(title,21)
    c.setFont('C',9);c.setFillColor(gray);c.drawString(M,18,'不排除極端值  ·  手機版');c.drawRightString(W-M,18,f'{n} / 2')
def table(rows,widths,font=10):
    global y
    rr=[]
    for i,row in enumerate(rows):
        s=ParagraphStyle('t',parent=st,fontName='B' if i==0 else 'C',fontSize=font,leading=font*1.3,textColor=colors.white if i==0 else navy)
        rr.append([Paragraph(html.escape(str(v)),s) for v in row])
    t=Table(rr,colWidths=widths);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),navy),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,light]),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5)]));_,h=t.wrap(cw,600);assert y-h>=41,(y,h);t.drawOn(c,M,y-h);y-=h+10



page(1,'相同學習率後，B略高於A')
p('A＝g＋β×new<br/>B＝g＋r1×(new-old)＋r2×old',13,teal)
p('new、old沿用相同基期的營業利益預測年比。三組每天使用完全相同的learning rate，價量與財報係數共同訓練。',11)
p('953日驗證，Rank IC有效952日；Sharpe未年化、未扣成本。保留exp(-a/9)，不排除極端值。',10,gray)
labels={'g_original':'原純g','a_own':'A覆蓋','b_own':'B差額＋舊值','g_common':'純g','overwrite':'A覆蓋','change_old':'B差額＋舊值'}
common=df[df.Variant.isin(['g_common','overwrite','change_old'])]
table([['共同學習率','Sharpe','Rank IC']]+[[labels[q.Variant],f'{q.Sharpe:+.6f}',f'{q.RankIC:+.6f}'] for q in common.itertuples()],[146,98,98],10.5)
p('B的Sharpe與Rank IC點估計均高於A，但三組Rank IC都負，尚未通過你的雙指標標準。',11)
p('相較上一輪，排序反轉',14,teal)
lookup=df.set_index('Variant')
table([['模型','原各自學習率\nSharpe','共同學習率\nSharpe']]+[[labels[a],f'{lookup.loc[a,"Sharpe"]:+.6f}',f'{lookup.loc[b,"Sharpe"]:+.6f}'] for a,b in [('g_original','g_common'),('a_own','overwrite'),('b_own','change_old')]],[116,113,113],10.3)
p('A從+0.011122降到+0.001063；B變化較小。原先A領先的結果對學習率設定敏感，不能全部歸因於修正公式。',11)
p('原v7純g仍高於這次三組。共同學習率也改變了價量g本身的學習結果，因此要同時看「同學習率純g」及「原v7」。',10.5)
p('每組1199次更新、2,325,806個訓練股票日；每日步長、樣本、全部預測與指標已核對。正式基準維持v7。',9.5,gray)
page(2,'這次固定了什麼？')
p('同一天，三組使用同一個數值',14,teal)
p('先依當天已到期批次，計算A、B各自的原步長，再取較小值給A、B與純g：<br/>共同 η(t)＝min(ηA(t), ηB(t))。',11)
p('步長可以逐日變動；這次固定的是模型之間的步長一致，並非整段歷史使用單一常數。沒有依回測成績挑選步長。',11)
p('B高於A，但差距仍不確定',14,teal)
ci=pd.read_csv(R/'paired_block_bootstrap.csv');refs={'change_old - overwrite':'B減A','overwrite - g_common':'A減共同步長g','change_old - g_common':'B減共同步長g'}
table([['Sharpe差值','差值','95%區間']]+[[refs[q.Comparison],f'{q.Difference:+.6f}',f'[{q.CI95Low:+.5f}, {q.CI95High:+.5f}]'] for q in ci[(ci.Metric=='Sharpe')&ci.Comparison.isin(refs)].itertuples()],[104,78,160],9.5)
p('三項區間都跨0，目前無法認定B穩定勝出。20日區塊、4000次配對重抽樣，未校正反覆選模型。',10,gray)
table([['分期Sharpe','共同g','A','B']]+[[str(vy)]+[f'{yr[(yr.Variant==n)&(yr.ValidationYear==vy)].Sharpe.iloc[0]:+.5f}' for n in ['g_common','overwrite','change_old']] for vy in [2018,2019,2020,2021]],[78,88,88,88],9.8)
p('如何理解這個結果',14,teal)
p('這輪排除了「同日learning rate數值不同」的因素。但A、B的特徵及梯度仍不同，相同步長不代表完全相同的更新路徑。',10.5)
p('可以說：目前沒有穩定勝出的證據。還不能說：修正資訊一定沒用，或市場一定不相信修正。',11,teal)
p('訊號、資料有效性與時間規則沿用v27。完整ZIP包含每日結果、共同步長排程、參數歷程、MSE及核對紀錄。',9.5,gray)
c.save()
archive=R.parent/'output/JPX-v28-common-learning-rate-data-20260916.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    z.write(out,out.name)
    for name in ['comparison.csv','comparison_by_year.csv','paired_block_bootstrap.csv','rank_ic_level_intervals.csv','all_models_daily.csv','all_coefficients.csv','common_learning_rates.csv','JPX-v28-common-learning-rate-report.md','experiment_plan.md','manifest.json','audit.json','results.json','run.py','audit.py','report.py']:
        z.write(R/name,name)
    for variant in ['g_common','overwrite','change_old']:
        for name in ['daily_metrics.csv','training_updates.csv','parameter_history.csv','results.json','audit.json','update_audit.json']:
            z.write(R/variant/name,variant+'/'+name)
    previous=R.parent/'jpx_v27_profit_forecast_state_20260916'
    for name in ['features.py','feature_manifest.json','feature_audit.json','experiment_plan.md']:
        z.write(previous/name,'v27_feature_definition/'+name)
print(json.dumps({'pdf':str(out),'zip':str(archive),'zip_bytes':archive.stat().st_size}))
