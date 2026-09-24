from pathlib import Path
import json,html,zipfile
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph,Table,TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
R=Path(__file__).resolve().parent;out=R.parent/'output/pdf/JPX-v27-profit-forecast-state-mobile-20260916.pdf'
df=pd.read_csv(R/'comparison.csv');yr=pd.read_csv(R/'comparison_by_year.csv')
pdfmetrics.registerFont(TTFont('C','/System/Library/Fonts/STHeiti Light.ttc',subfontIndex=0));pdfmetrics.registerFont(TTFont('B','/System/Library/Fonts/STHeiti Medium.ttc',subfontIndex=0))
W,H=390,680;M=24;cw=W-2*M;navy=colors.HexColor('#153347');teal=colors.HexColor('#007C82');gray=colors.HexColor('#536B78');light=colors.HexColor('#F2F7F9')
c=canvas.Canvas(str(out),pagesize=(W,H));c.setTitle('v27｜預測覆蓋與差額加權');c.setAuthor('Quant Learning & Internship 2027');y=0
st=ParagraphStyle('s',fontName='C',fontSize=11.5,leading=17,wordWrap='CJK',textColor=navy)
def p(t,size=11.5,color=navy,gap=9):
    global y
    ss=ParagraphStyle('x',parent=st,fontSize=size,leading=size*1.48,textColor=color);a=Paragraph(t,ss);_,h=a.wrap(cw,600);assert y-h>=41,(y,h,t);a.drawOn(c,M,y-h);y-=h+gap

def page(n,title):
    global y
    if n>1:c.showPage()
    c.setFillColor(navy);c.rect(0,H-7,W,7,fill=1,stroke=0);y=H-31;p('JPX  /  v27  /  2026-09-16',10,teal);p(title,21)
    c.setFont('C',9);c.setFillColor(gray);c.drawString(M,18,'不排除極端值  ·  手機版');c.drawRightString(W-M,18,f'{n} / 2')
def table(rows,widths,font=10):
    global y
    rr=[]
    for i,row in enumerate(rows):
        s=ParagraphStyle('t',parent=st,fontName='B' if i==0 else 'C',fontSize=font,leading=font*1.3,textColor=colors.white if i==0 else navy)
        rr.append([Paragraph(html.escape(str(v)),s) for v in row])
    t=Table(rr,colWidths=widths);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),navy),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,light]),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5)]));_,h=t.wrap(cw,600);assert y-h>=41,(y,h);t.drawOn(c,M,y-h);y-=h+10


page(1,'新值覆蓋：這次表現較好')
p('A＝g＋β×new<br/>B＝g＋r1×(new-old)＋r2×old',13,teal)
p('new、old 是相同基期的營業利益預測年比。價量g與財報係數共同訓練；新狀態覆蓋舊狀態，保留 exp(-a/9) 衰減。',11)
p('953日驗證；Rank IC有效952日。Sharpe未年化、未扣成本，訓練與驗證均不排除極端值。',10,gray)
labels={'g':'純價量g','old_forecast':'舊版：累加預測年比','old_joint':'舊版：預測＋修正','overwrite':'A：新值覆蓋','change_old':'B：差額＋舊值'}
table([['模型','Sharpe','Rank IC']]+[[labels[r.Variant],f'{r.Sharpe:+.6f}',f'{r.RankIC:+.6f}'] for r in df.itertuples()],[146,98,98],10.5)
p('A高於純g，B低於純g；兩者Rank IC仍負，尚未達到你的雙指標標準。A也沒有超過舊版的預測年比單項。',11)
table([['分期Sharpe','純g','A覆蓋','B差額＋舊值']]+[[str(vy)]+[f'{yr[(yr.Variant==n)&(yr.ValidationYear==vy)].Sharpe.iloc[0]:+.5f}' for n in ['g','overwrite','change_old']] for vy in [2018,2019,2020,2021]],[78,88,88,88],9.8)
p('A僅在2019、2020優於g，2018、2021更差。<br/>A減g的ΔSharpe＝+0.003443；95%區間<br/>[-0.022255, +0.030650]，尚無穩定勝出證據。',10.5)
p('每日一次MSE更新；共1199次更新與2,326,022筆預測核對通過。正式基準維持v7。',9.5,gray)
page(2,'修正值如何更新，結果如何解讀')
p('同基準，才能比較新舊值',14,teal)
p('以b代表去年同季的預測：<br/>new＝(新單季預測-b) / |b|<br/>old＝(舊單季預測-b) / |b|<br/>差額＝new-old，不再除以old。',11)
p('同一季再次修正時，old使用上次的new。換季時，舊全年預測也要扣除最新累計實績、除以剩餘季數，讓新舊值使用相同基準。沒有可用舊預測才設old＝new、差額＝0。',11)
p('新狀態取代舊狀態，三個訊號共同乘 exp(-a/9)。a是距最新狀態更新的交易日數；相同值重複公布不重置。已逐筆核對 new＝差額＋old。',10.5)
p('保留4,608次非零修正，其中4,603次需換季重建舊預測。沿用原有效文件規則，會計口徑不明的通用修正文件仍未採用。',10,gray)
p('為什麼B不一定比較好？',14,teal)
p('當r1＝r2＝β，B在數學上就等於A。但多一個自由係數，加上不同特徵尺度，會改變每日MSE訓練路徑；更彈性不保證Sharpe更高。',11)
p('1199個更新日中，B有25日的步長不到A的十分之一。這是待驗證的原因，不能直接解讀為「市場不相信修正」或「修正資訊無用」。',10.5)
ci=pd.read_csv(R/'paired_block_bootstrap.csv')
refs={'change_old - overwrite':'B減A','change_old - g':'B減g'}
table([['Sharpe差值','95%區間']]+[[refs[r.Comparison],f'[{r.CI95Low:+.5f}, {r.CI95High:+.5f}]'] for r in ci[(ci.Metric=='Sharpe')&ci.Comparison.isin(refs)].itertuples()],[110,232],10)
p('20日区塊、4000次配對重抽樣；全部差值區間跨0。這是同一歷史期間的探索結果，未校正反覆選模型。'.replace('区','區'),9.5,gray)
p('下一步可固定共同步長，比較A與B，隔離訓練方式的影響。本輪尚未執行這項追加實驗。',10,teal)
c.save()
archive=R.parent/'output/JPX-v27-profit-forecast-state-data-20260916.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    z.write(out,out.name)
    for name in ['comparison.csv','comparison_by_year.csv','paired_block_bootstrap.csv','rank_ic_level_intervals.csv','all_models_daily.csv','all_coefficients.csv','learning_rate_comparison.csv','optimization_diagnostics.json','largest_state_revisions.csv','state_events.csv','JPX-v27-forecast-state-report.md','experiment_plan.md','manifest.json','feature_manifest.json','feature_audit.json','audit.json','results.json','features.py','run.py','audit.py','report.py']:
        z.write(R/name,name)
    for variant in ['overwrite','change_old']:
        for name in ['daily_metrics.csv','training_updates.csv','parameter_history.csv','results.json','audit.json','update_audit.json']:
            z.write(R/variant/name,variant+'/'+name)
print(json.dumps({'pdf':str(out),'zip':str(archive),'zip_bytes':archive.stat().st_size}))
