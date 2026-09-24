import json
import pandas as pd
from run_v8 import ROOT,VARIANTS,NAMES

def main():
    result=json.loads((ROOT/'results.json').read_text());audit=json.loads((ROOT/'audit.json').read_text())
    names={'daily':'v8 每天整批','sgd':'v8 逐檔更新','mini128':'v8 每 128 檔','v7_equal':'v7 報酬 MSE 等權','v7_jpx':'v7 報酬 MSE 官方權重','v5':'v5 年度最小平方法'}
    lines=['# JPX v8：同一檔股票的名次差平方','',
        '2026-09-12。三組已完成逐日訓練與歷史驗證，使用同一組十一個特徵、股票池、資料揭曉時間與評分方式。每天整批是執行前指定的主要比較組；未依結果挑選主要組。','',
        '## 結果','',f'以下使用六組共同可評分的 **{result["common_scorable_dates"]} 天**。JPX Sharpe 為日 spread 的平均除以樣本標準差，未年化；數值越高越好。','',
        '| 方法 | JPX Sharpe（未年化） |','|---|---:|']
    for v,c in result['comparison'].items():lines.append(f'| {names[v]} | {c["official_unannualized_sharpe"]:.8f} |')
    lines+=['','| 驗證年度 | 共同天數 | 每天整批 | 逐檔 | 每 128 檔 | v7 等權 | v5 |','|---|---:|---:|---:|---:|---:|---:|']
    for a in result['annual']:lines.append(f'| {a["validation_year"]} | {a["days"]} | {a["daily"]:.6f} | {a["sgd"]:.6f} | {a["mini128"]:.6f} | {a["v7_equal"]:.6f} | {a["v5"]:.6f} |')
    main=result['comparison']['daily']['official_unannualized_sharpe'];old=result['comparison']['v7_equal']['official_unannualized_sharpe']
    lines+=['',f'預先指定的 v8 每天整批相對 v7 等權，Sharpe 差為 **{main-old:+.8f}**。這是同一段歷史資料的研究比較，不能當成未來績效保證；驗證資料已多次使用，沒有動用新的保留測試集。','',
        '## 你指定的 loss 如何實作','',
        '每一個誤差都比較**同一檔股票 i** 的真實名次與預測名次，再平方；不開根號。沒有把「股票 A 名次減股票 B 名次」當成訓練樣本。','',
        '```text\n模型分數：g_i = alpha + X_i · beta\n真實名次：r_i = 當天 Target 由高至低排序（相同 Target 取平均名次）\n可微分預測名次：r_soft_i = 1 + Σ(j ≠ i) sigmoid((g_j - g_i) / τ)\nτ = 1，三組皆固定\n批次 loss = mean_i∈B [ ((r_i - r_soft_i) / (N - 1))² ]\n```','',
        'N 是當天全部有已知 Target 的股票數，B 是本次要計算誤差的股票集合。除以 N−1 讓不同日期的名次尺度一致。訓練使用已同意的 soft rank 近似；最後提交評分時仍把分數排成唯一整數名次。**訓練並非對離散整數名次直接求梯度，也不保證每次更新都降低實際整數名次誤差。**','',
        '計算股票 i 排第幾必須參考同日其他股票；即使 B 只有一檔或 128 檔，計算名次時仍參考完整 N 檔，也包含其他股票分數隨參數改變的梯度。alpha 對所有股票增加同樣分數，不改變排名，因此梯度為零並維持初始值 0。','',
        '## 三組更新與時間順序','',
        '- 每天整批：同日所有有標籤股票的平均誤差，只更新一次。\n- 逐檔：每次用一檔的名次誤差更新，當天每檔走過一次。\n- 每 128 檔：每批 128 檔；最後不足 128 檔仍獨立成批，以該批實際檔數平均。','',
        '三組在同一天使用相同固定隨機順序。參數只在 2017-01-04 初始化為零一次，跨日與跨年接續；不重播歷史樣本。Target 於 ExitDate 收盤揭曉後才進入更新，再使用當天可得特徵產生當天預測。2020-10-01 處理已揭曉標籤，但不在全市場休市日產生預測。最後預測日 2021-12-01，最後一批標籤處理至 2021-12-03。','',
        '保留 v5 的 T5、T22、T60、V5、V22、V60、三個同窗口交乘項，以及 PR1、VR1；PR1 與 VR1 不另行標準化或截尾。未定義輸入逐項用 0 代入並保留旗標。使用歷史核心股票池，排除當日 SupervisionFlag；訓練只排除 Target 缺失的股票，預測不依未來 Target 是否缺失來篩選。','',
        '## 學習率的接受與停止條件','',
        '每批先固定舊參數並只算一次梯度。由 0.1 起試：若能降低 loss，依序試 0.2、0.3……到 1.0；第一次不再改善時停止，選已接受者中 loss 最小的候選。若 0.1 不合格，就試 0.09……0.01，再試 0.009……，最低到 0.000001；接受第一個合格者。每次試探皆從同一組舊參數出發，最後只提交一次更新。','',
        '候選 loss 必須有限、嚴格低於原 loss，且符合 L_new ≤ L_old − 0.0001 × η × ‖gradient‖²。梯度最大絕對值 ≤ 1e−10 就跳過；全部學習率不合格也跳過。下降很快本身不是拒絕理由。上限 1.0 是預先固定的試驗邊界，達到上限不表示已找到全域最佳學習率。','',
        '| 方法 | 標籤日數 | 股票誤差項總數 | 批次總數 | 接受更新 | 小梯度跳過 | 無合格步長跳過 | 選到上限 1.0 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for v in VARIANTS:
        r=result['variants'][v];lines.append(f'| {names[v]} | {r["label_days"]:,} | {r["training_stocks"]:,} | {r["batches"]:,} | {r["accepted_updates"]:,} | {r["small_gradient_skips"]:,} | {r["no_acceptable_eta_skips"]:,} | {r["hit_upper_eta_batches"]:,} |')
    lines+=['','| 方法 | 當天整體 soft loss 上升的天數 | 平均 Rank IC | 平均標準化整數名次 MSE |','|---|---:|---:|---:|']
    for v in VARIANTS:
        r=result['variants'][v];lines.append(f'| {names[v]} | {r["full_day_loss_increased_days"]} | {r["mean_rank_ic"]:.8f} | {r["mean_normalized_hard_rank_mse"]:.8f} |')
    lines+=['',
        '上表 Rank IC 是每日 Target 與預測分數的 Spearman 相關，先按各自數值取平均名次；Target 或分數全相同的日期不計相關。整數名次 MSE 只在有標籤的同一批股票中重新編名次，以 N−1 標準化後平方平均；分數同分以證券代碼決定順序。這兩項用全部 953 個驗證日各自可用標籤計算，與前表的共同可評分日篩選不同。','',
        '逐檔／128 檔更新只保證所接受的一次更新降低當批 soft loss，整天的 soft loss 仍可能上升。更低的名次 MSE 也不必然提高只使用兩端各 200 檔、且依報酬幅度計算的 JPX Sharpe。','',
        '## 驗證範圍','',
        '預測維持 JPX 高分多頭 200 檔、低分空頭 200 檔，兩側按極端程度使用 2→1 權重。沒有 regime、Ridge、softmax 或成本；這裡是競賽形式歷史排名評分，不是完整可交易淨值回測。選中的 Target 缺失時，該天不可評分，沒有補 0。','',
        '- 編譯計算與獨立 NumPy 計算的 loss、解析梯度、有限差分梯度一致，涵蓋單檔、整批、剩餘小批與向下調學習率。\n- 三組所有批次均檢查接受條件、批次大小與完整覆蓋，並核對資料揭曉時間及跨年度參數接續。\n- 每組抽取五個實際訓練日期，獨立重算已存的第一／中間／最後批次梯度與完整步長試探；不是逐一獨立重算所有 SGD 梯度。\n- 三組所有暖身與驗證預測均從當時參數重建，核對整數排名；各驗證日官方 spread 另行重算一致。','',
        '## 檔案','',
        '- `results.json`：整體與年度比較。\n- `daily_comparison.csv`：逐日比較與共同可評分日旗標。\n- `audit.json`：獨立核對結果及各學習率出現次數。\n- 各組資料夾：逐日參數、訓練彙總、每批 loss／梯度範數／步長紀錄、梯度抽樣快照、每日排名與選股。\n- `run_v8.py`、`native.py`、`soft_rank.cpp`：可重跑實作；`experiment_plan.md` 記錄執行前設定。','',
        '重新執行順序：準備輸入 → 三組各自完整訓練 → 比較 → 稽核 → 產生報告。輸入沿用本工作區既有特徵與使用者提供的 JPX_data.zip，來源檔未修改。']
    (ROOT/'JPX-v8-same-stock-rank-loss-report.md').write_text('\n'.join(lines)+'\n')
    coefficients=pd.DataFrame({v:result['variants'][v]['final_coefficients'] for v in VARIANTS}).reindex(NAMES)
    coefficients.index.name='Parameter';coefficients.to_csv(ROOT/'final_coefficients.csv')
    print('REPORT_WRITTEN',flush=True)

if __name__=='__main__':main()
