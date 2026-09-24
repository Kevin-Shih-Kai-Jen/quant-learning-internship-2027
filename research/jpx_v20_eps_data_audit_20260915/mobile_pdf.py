from pathlib import Path
import json,html,zipfile
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph,Table,TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
R=Path(__file__).resolve().parent;out=R.parent/'output/pdf/JPX-v20-eps-data-audit-mobile-20260915.pdf'
df=pd.read_csv(R/'eps_sensitivity_by_case.csv')
pdfmetrics.registerFont(TTFont('C','/System/Library/Fonts/STHeiti Light.ttc',subfontIndex=0));pdfmetrics.registerFont(TTFont('B','/System/Library/Fonts/STHeiti Medium.ttc',subfontIndex=0))
W,H=390,680;M=24;cw=W-2*M;navy=colors.HexColor('#153347');teal=colors.HexColor('#007C82');gray=colors.HexColor('#536B78');light=colors.HexColor('#F2F7F9')
c=canvas.Canvas(str(out),pagesize=(W,H));c.setTitle('v20｜20筆EPS資料核對');c.setAuthor('Quant Learning & Internship 2027');y=0
st=ParagraphStyle('s',fontName='C',fontSize=11.5,leading=17,wordWrap='CJK',textColor=navy)
def p(t,size=11.5,color=navy,gap=9):
    global y
    ss=ParagraphStyle('x',parent=st,fontSize=size,leading=size*1.48,textColor=color);a=Paragraph(t,ss);_,h=a.wrap(cw,600);assert y-h>=41,(y,h,t);a.drawOn(c,M,y-h);y-=h+gap

def page(n,title):
    global y
    if n>1:c.showPage()
    c.setFillColor(navy);c.rect(0,H-7,W,7,fill=1,stroke=0);y=H-31;p('JPX  /  v20  /  2026-09-15',10,teal);p(title,21)
    c.setFont('C',9);c.setFillColor(gray);c.drawString(M,18,'資料核對  ·  模型尚未重訓');c.drawRightString(W-M,18,f'{n} / 3')
def table(rows,widths,font=10):
    global y
    rr=[]
    for i,row in enumerate(rows):
        s=ParagraphStyle('t',parent=st,fontName='B' if i==0 else 'C',fontSize=font,leading=font*1.3,textColor=colors.white if i==0 else navy)
        rr.append([Paragraph(html.escape(str(v)),s) for v in row])
    t=Table(rr,colWidths=widths);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),navy),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,light]),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5)]));_,h=t.wrap(cw,600);assert y-h>=41,(y,h);t.drawOn(c,M,y-h);y-=h+10

page(1,'找到一個需要先修的資料問題')
p('20筆抽查已完成。累計EPS直接相減，部分案例會把年增率嚴重放大；這是已有官方文件佐證的資料定義問題。',12)
p('來源與時間核對通過',14,teal)
p('10筆一般＋10筆極端案例，合計87笔來源、1,218項欄位均與原始JPX CSV一致。季度對應和當時可用時間也通過核對。這不等於EPS拆季方法正確。',11)
p('最明確案例：中國塗料 4617',14,teal)
table([['項目','舊程式','官方單季'],['本季EPS','13.31','13.39'],['去年同季EPS','0.01','0.12'],['年增率','133,000%','11,058.33%']],[130,106,106],12)
p('舊方法：累計EPS相減。公司另列的單季EPS顯示，去年基準由0.01變成0.12；算出的年增率相差約12倍。',11)
p('官方單季數字見2020-02-12季度報告，晚於原2020-01-31訊號日。這份文件用來查錯，不能回填成1月31日已知資料。',10.5,gray)
p('還有來源差異與低基期',14,teal)
p('4572的上期累計EPS，原CSV為-62.50，公司公告是-62.56，原因待查。另有直接公告0.01的低基期，以及2884於2022年才發布的更正，須分別處理。',10.5)
p('下一步只先修EPS的單季來源、可比基準與版本時間，再重跑原比較。本輪沒有重訓，也沒有證明修正後Sharpe一定提高。',11,teal)
p('主要證據：公司季報PDF第4頁。完整網址、逐案確認值與查核缺口都附在資料包。',9.5,gray)
page(2,'一般案例：10筆核對')
p('固定種子按年份／季度抽樣，兩個原始比率均在±200%內，每家公司一筆。不是整個資料集的代表性錯誤率估計。',10.5)
p('下表年增以比率表示，1＝100%。',10.5,gray)
table([['案例／代碼','舊年增','股數敏感度年增']]+[[r.CaseID+' / '+str(r.Code),f'{r.OriginalYoY:+.4f}',f'{r.ShareAdjustedYoYSensitivity:+.4f}'] for r in df[df.SampleType=='ordinary'].itertuples()],[112,104,126],11)
p('一般10筆，在這個敏感度計算中，季增與年增方向皆未翻轉；最大年增差約0.33個百分點。',11)
p('敏感度不是官方真值',14,teal)
p('利用累計EPS、平均股數和期間天數，估算股數變動可能造成的差異。仍需同一股數加權與股票分割基礎，且公布EPS本身有捨入誤差。',10.5)
p('N09原CSV與公司公告的上期累計值不一致，因此這個敏感度計算也受原始輸入影響。外部文件為逐案部分核對，未取得全部舊基期原件；來源檔有明確標記。',10.5,gray)
page(3,'極端案例：分母都很小')
p('取v19平方誤差最高的10家公司。下表是原模型去年EPS基準，不是修正值。',10.5)
table([['案例／代碼','去年EPS','基準取得方式']]+[[r.CaseID+' / '+str(r.Code),f'{r.OriginalYearBaseEPS:.2f}',('累計相減' if r.YearBaseFromCumulativeSubtraction else '第一季直接值')] for r in df[df.SampleType=='extreme'].itertuples()],[112,94,136],11)
p('10筆中，4筆基準是第一季直接EPS，6筆由累計相減。基準絕對值都不超過0.09。即使原值讀對，成長率也可能被小分母與兩位小數精度放大。',11)
p('不能直接用粗略淨利重算EPS',14,teal)
p('部分資料的Profit顯示0百萬，EPS卻為0.01。這不代表真正獲利是0；淨利欄的顯示精度可能已經丟失重要數字。',10.5)
p('E05的精度敏感度跨越100倍訓練門檻；E10的去年基準敏感度跨0。這些是條件式敏感度，不能當作真實EPS信賴區間。',10.5)
p('報告結論：先修資料，再判斷特徵。舊EPS實驗保留作紀錄；純價量v7維持原樣。',11,teal)
c.save()
print(str(out))
