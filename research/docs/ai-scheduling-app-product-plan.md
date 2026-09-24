# AI 協作排班 App：產品、技術與商業規劃 v0.2

更新日期：2026-09-04

## 0. 目前結論

這個產品值得做，但首版必須收斂成：

> 為台灣 3–20 人、單一據點、一般工時制的餐飲／零售小團隊，收集成員可上班時段與自然語言需求，產生 1–3 個有實質差異、可解釋的候選班表，讓管理者審核、修改與發布。

暫定商業模式為 `NT$990／群組／年`，員工免費。這個價格在技術成本上可行；真正的成本風險不是 AI，而是人工客服、複雜產業規則與多據點客製化。

首發順序建議：

1. Web（可安裝，但不承諾完整離線）+ Windows，先用 5–10 個真實團隊驗證。
2. 約 20–30 個穩定付費群組後上 Google Play。
3. 證明留存與續訂後，再支付 Apple Developer 年費並上 iOS。

這份規劃假設第一客群是餐飲／零售。若第一客群是醫療、護理、航空、保全、24 小時照護、大型連鎖或跨門市，必須重做部分規則與成本模型，不能直接沿用此 MVP。

### v0.2 審查後的主要改動

- 產品不再被定義為「AI 日曆」，而是共用同一狀態的「流程控制台＋班表畫布」。
- 加入每一期班表、成員提交、班次安排三層狀態機。
- 桌面班表改為覆蓋／人員雙視圖，搭配待補班次列與右側情境檢查器。
- 手機不縮小桌面表格，成員端改用 agenda 與「常態＋本期例外」。
- 候選方案必須先滿足相同硬限制與最大可達覆蓋，再比較公平、偏好與改動量。
- 候選改為 1–3 個；明確區分最佳、可行、已證明無解、逾時與模型錯誤，不把逾時說成無解。
- 拖放、AI 修改與重新求解都必須先預覽影響、可以復原，且不能直接覆寫目前草稿。
- 發布後補上「通知送達／已查看／已確認／有問題」閉環與逐班版本差異。
- 背景求解、通知與發布加入 transactional outbox、idempotency 與 compare-and-swap，避免漏執行或重複執行。
- 自然語言中含「這週日」等相對日期時，不可只靠文字 hash 快取。
- 第三方 AI 使用前需獨立說明並取得明確許可，同時保留完全不使用 AI 的結構化路徑。
- 重新定義備份：不可把 Supabase Storage 當成資料庫備份；權威 schedule snapshot 留在 Postgres。
- 付款改為 server-side entitlement 與事件對帳；到期仍可讀取和匯出，不綁架既有資料。
- 離線首版只承諾查看已同步 Published 班表與保存成員本機草稿，不承諾管理員離線編班。

## 1. 產品定位

### 一句話

把成員散亂的排班需求，轉成可確認、可追蹤的規則，再產生數個公平且可解釋的班表。

### 不只賣「AI 排班」

台灣已有產品提供免費 AI 配班與 LINE 報班，因此「有 AI」本身不是差異化。第一版應強調：

- 繁體中文自然語言轉規則，而且每條都可確認。
- 不是只給一張黑箱答案，而是在有實質差異時比較 2–3 個不同取向的方案。
- 無解時解釋是哪幾條規則、哪些班次或技能造成衝突。
- 管理者能用對話修改，但系統會顯示實際規則差異。
- 公平分數透明，且不把核准休假或受保護特徵當成負面訊號。
- 所有輸入、求解器版本與原始輸出都能稽核及重播，發布後的變更可追蹤。

### 產品不做的承諾

- 不承諾「找到所有可能班表」。組合數會快速爆炸，只找限時內可用且彼此有實質差異的 1–3 個方案。
- 不宣稱「保證符合勞基法」。產品提供規則檢查、版本與警示，最終制度設定與合法性仍由雇主負責。
- 不把預定班表說成實際出勤紀錄、打卡或薪資依據。
- 不讓 AI 產生任意程式碼並直接執行。

## 2. 使用者與角色

### 主要角色

- Owner：群組、帳務、資料匯出／刪除、權限管理。
- Scheduler：建立週期、班次、人力需求與規則，產生、修改、發布班表。
- Member：提交自己的可用時間與偏好，只查看已發布班表。
- Viewer：只讀已發布班表；首版可延後。

### 權限底線

- Member 看不到其他人的自由文字、請假原因或私人備註。
- Scheduler 只看到排班必要資訊，不要求成員輸入健康、家庭或請假原因。
- 員工不能建立或修改法律基線；AI 也不能改動法律基線。
- 任何人都不能在發布後原地覆寫班表，只能建立新版本。

## 3. 核心使用流程

### A. 管理者建立群組

1. 建立 Group：名稱、時區、排班週起始日、提交截止時間。
2. 選擇產業與工時制度；MVP 只完整支援一般工時制。
3. 建立職位、技能與班別模板，例如早班 09:00–17:00。
4. 設定每個班次的人數與必要技能。
5. 用一次性邀請連結邀請成員；可把連結手動分享到 LINE，但不購買 LINE／SMS 發送服務。
6. 設定公司規則，例如最多連上 5 天、每人每週目標時數。

### B. 成員提交需求

1. 日曆選擇「可以、偏好、不可以」。
2. 可補充文字，例如「週三下課後 18:30 才能到，週五希望晚班」。
3. AI 只把文字轉成白名單規則，顯示逐條預覽與原句來源。
4. 日期、時間、程度或對象不明確時，要求補充。
5. 成員確認後才生效；未確認的 AI 結果不能進求解器。
6. 截止後的變更成為新版本，需管理者接受。

### C. 截止與缺交

- 到期只凍結輸入快照並通知管理者，不在所有群組同一時間自動求解。
- 「沒回覆」預設是未知，不可偷偷當成全天可上班。
- 管理者可選擇：沿用週期模板、暫時排除、或明確設為可排；選擇會寫入稽核紀錄。

### D. 產生候選班表

1. 先檢查缺交、時段不足、技能不足與互相矛盾規則。
2. 求解器先最大化可達覆蓋，再於相同覆蓋下產生均衡、公平或偏好取向的 1–3 個候選。
3. 候選用同一組指標比較，不標示模糊的「AI 最佳」，也不為湊數製造近乎相同的方案。
4. 若無完整解，顯示缺班方案與「足以造成衝突的規則組」；只有按需再縮減原因，絕不暗中突破安全／法律／不可上班條件。

### E. 修改與發布

- 拖放班次時即時驗證硬限制。
- 管理者可鎖定已滿意的安排後重新求解。
- 對話要求如「Amy 少一個晚班、週末不要動」會先轉成規則 diff，確認後重算。
- 發布後通知所有成員；新變更建立新版本並顯示前後差異。
- App 內最新 Published 版本是唯一真相，通知失敗不改變版本狀態。

## 3A. 整合式視覺與資訊架構

### 兩層產品模型

不要把產品做成「有 AI 的日曆」。主體應是：

1. 流程控制台：回答這一期走到哪裡、卡在哪裡、下一步是什麼。
2. 班表畫布：讓管理者看覆蓋、安排人員、處理例外與發布變更。

兩層必須讀寫同一個 schedule period 狀態機，不能像外部表單、試算表與聊天工具一樣需要人工同步。

### 主導航

- `總覽`：所有進行中週期、下一個截止時間與風險。
- `班表`：目前週期的流程控制台與編輯畫布。
- `團隊`：成員、職位、技能、常態可排模板。
- `範本與規則`：班別、人力需求、公司規則。
- `設定`：通知、權限、隱私、帳務。

### 每一期可操作流程軌

```text
設定需求 ✓
→ 收集需求 18/20
→ 檢查資料 2 項
→ 產生方案
→ 調整班表 1 個缺口
→ 發布
→ 成員確認 0/20
```

流程軌不是裝飾。每一步必須顯示：

- 狀態與數量。
- 阻礙下一步的項目。
- 截止時間。
- 唯一主要行動，例如「提醒 2 位未交成員」。
- 點擊後直接聚焦相對應的人、規則或班次。

### 三層狀態機

Schedule period：

```text
SETUP → COLLECTING → INPUT_REVIEW → READY_TO_SOLVE
→ SOLVING → DRAFT_REVIEW → PUBLISHED
→ AMENDMENT_DRAFT → SUPERSEDED
```

`SOLVE_FAILED`、`TIMED_OUT`、`CANCELLED` 是求解工作狀態，不應讓整個週期停在沒有說明的狀態。

Member submission：

```text
NOT_STARTED | DRAFT | SUBMITTED | NEEDS_CLARIFICATION | LATE_AMENDMENT
```

Assignment：

```text
GAP | PROPOSED | MANUALLY_LOCKED | PUBLISHED | CHANGED | MEMBER_CONFIRMED
```

### 週期駕駛艙

週期首頁只回答四件事：

- 現在在哪一步。
- 下一個可完成的動作。
- 哪些問題阻礙發布。
- 最近是誰改了什麼。

```text
9/7–9/13｜需求收集中｜剩 2 天
[設定 ✓] [收集 18/20] [檢查 2] [產生] [調整] [發布] [確認]

下一步：2 位成員尚未提交                    [提醒未交者]
覆蓋預估：早班足夠｜週五晚班可能缺 1 人
最近變更：Amy 於 14:32 修改週三需求
```

### 桌面／Windows 班表板

```text
上方：週期、狀態、Undo/Redo、重新求解、版本、發布
次列：已填 42/43｜缺 1｜硬限制 0｜偏好 31/34｜未知 2

左／中央：班表格狀畫布                 右：所選班次情境檢查器
```

首版提供兩個共用相同資料的視圖：

- `覆蓋視圖`，預設：日期為欄，班別＋職位為列；每格顯示 `2/3 已填`、人名與缺口。
- `人員視圖`：成員為列、日期為欄；用來檢查每人時數、連班、晚班與週末分布。

可延後獨立「覆蓋熱圖」，先在覆蓋視圖的格子和每日摘要呈現即可，避免做第三套畫布。

### 視覺語彙

- `＋缺 1`：營運缺口。
- `? 未回覆`：未知，不能等同可排。
- `★ 偏好`：希望排。
- `禁止圖示＋不可排`：不可放寬。
- `⚠ 班距／工時問題`：警告或阻擋，文字說明原因。
- `鎖頭＋已鎖定`：重新求解不可移動。
- 斜紋：尚未發布的草稿；實色：已發布。
- 外框與 `Changed` 文字：已發布後產生的未發布修改。

顏色只負責班型或職位的持續識別。錯誤、草稿、鎖定與發布狀態必須再搭配文字、圖示、形狀或紋理，避免一個顏色同時代表四種事情。

### 右側情境檢查器

點一個缺口或安排，不打開大型 modal；右側顯示：

- 班次、角色、人力 `2/3`。
- 為何成為問題的因果鏈。
- 符合資格與不符合資格的人選。
- 每位人選的可用性、技能、本週時數、公平差額與具體阻擋原因。
- 重新指派、設為公開班、鎖定、查看歷史。

人選不可只顯示 AI 排名。管理者必須知道「為什麼可排／不可排」。

### 編輯手感與安全契約

- 拖曳前：顯示合法與不合法人選的理由。
- 拖曳中：合法落點高亮；硬限制顯示禁止符號且不能放下；軟性代價用警告表示。
- 拖曳後：立即重算覆蓋、時數與公平影響，提供 Undo。
- 所有拖放都有點擊／鍵盤替代：`選取班次 → 重新指派 → 選人員`。
- 每個編輯動作都是 draft event，支援 Undo/Redo，不能只存在前端記憶體。
- 手動安排預設鎖定；重新求解尊重鎖定，除非使用者明確解除。
- 鎖定可以作用於單一 assignment；整人／整日批次鎖定可延後。

### 重新求解的三種模式

- `只補空缺`：不動已有安排。
- `重新平衡未鎖定安排`：保留人工鎖定。
- `產生全新方案`：建立另一個候選，不覆蓋目前草稿。

求解完成先顯示：

```text
將改動 4 位成員、6 個班次
新增 2｜移除 1｜換人 3｜時間變更 0
[只看差異] [放棄] [套用成新草稿]
```

### 自然語言修改

聊天是提案產生器，不是另一套隱藏編輯器：

```text
你的要求會被理解為：
- Amy 本期晚班最多 2 次
- 週六、週日現有安排保持不動
- 僅限本期、可權衡的公司偏好

預計重新計算 23 個未鎖定安排
[修改理解] [預覽結果]
```

確認理解後仍只產生預覽；按下「套用」才建立新 draft revision。

### 候選方案比較

不要並排三張完整班表，也不要提供一個無法解釋的總分：

- 上方並排 1–3 個摘要選項，下方只顯示一張大班表。
- A/B/C 快速切換，提供 `只看差異`。
- 所有候選都必須有 0 個硬限制違反，並先達到相同的最大可達覆蓋。
- 比較覆蓋 `42/43`、偏好 `31/34`、目標時數偏差、不熱門班範圍、與上一發布版本的改動數。
- 只有一或兩個實質不同方案時就如實顯示，不製造三個幾乎相同的答案。
- 初次排班命名：均衡、偏好優先、公平分配。
- 已發布後重排命名：最少改動、均衡、偏好優先。

### Issue Center

- `必須修正`：硬限制、資料錯誤、舊 snapshot。
- `需要決定`：缺班、未知、可接受的營運取捨。
- `提示`：偏好未滿足、公平差距。

衝突說明由求解器產生結構化因果鏈，AI 只負責改寫成人話：

```text
週五晚班需要 2 位具關店技能的人
→ 目前只有 Amy、Bo 符合
→ Amy 已確認不可排；Bo 已達本週上限
→ 因此仍缺 1 人
```

法律、安全、核准休假與明確不可排不能提供「一鍵放寬」。

### 成員手機端

手機不縮小桌面格狀表。採「常態模板＋本期例外」：

1. 預載上次已確認的常態可排時間。
2. 成員只修改本期不同的日期。
3. 每格明確為 `可以｜希望排｜不可以｜未填`。
4. 自由文字放在結構化輸入後方，作為補充而非主要入口。
5. AI 規則逐條顯示原句、解析結果與是否需澄清。
6. 提交前顯示 `已回答 12/14；2 個仍是未知，系統不會自動視為可排`。
7. 顯示 `尚未同步｜已同步｜已提交`；離線時只保存本機草稿，不假裝已完成。
8. 已發布班表使用 agenda，只突出自己的班與變更。

### 發布與成員確認閉環

發布前抽屜列出：

- 硬限制違反與尚未填滿班次。
- 尚未提交需求的人。
- 使用的輸入 snapshot 與截止時間。
- 相較上一版改動幾人、幾班。
- 通知所有人或只通知受影響者。
- 員工端預覽。

可以允許「含缺口的部分發布」，但需二次確認；硬限制錯誤則阻擋發布。

發布後分開紀錄：

```text
NOTIFICATION_DELIVERED | VERSION_VIEWED | MEMBER_CONFIRMED | REPORTED_PROBLEM
```

通知送達不代表看過，看過也不代表接受。成員的「這班有問題」只建立 change request；MVP 不因此擴張成完整換班市場。

### 響應式與無障礙

- `<600px`：agenda／日期卡片，不顯示壓縮格狀表。
- `600–1023px`：主要畫面＋可開關 inspector。
- `≥1024px`：完整班表＋常駐右側 inspector。
- 以 WCAG 2.2 AA 為目標，狀態不能只依賴顏色。
- 桌面排班格使用正確的互動 grid 語意、列欄標題與朗讀名稱。
- 方向鍵移動格子、Enter 編輯、Escape 離開；動態儲存與錯誤用 live region 公告。
- 支援 200% 縮放、高對比、減少動畫，並以 Narrator、VoiceOver、TalkBack 和純鍵盤實測。

## 4. 排班引擎

### 核心模型

使用 Google OR-Tools CP-SAT，而不是自行列舉所有組合。核心決策變數：

```text
x[member, shift] ∈ {0, 1}
```

### 規則分層

第一層，不可放寬：

- 班次不可重疊。
- 已核准休假與明確不可上班時段。
- 職位、技能與資格有效期限。
- 僱用期間。
- 班距、每日／每週工時等已啟用的安全或法規規則。
- 管理者鎖定的安排。

第二層，可顯示缺口但不可犧牲第一層：

- 班次最低人數與技能覆蓋。
- 開店／關店必要人員。
- 關鍵班次不可只有新進人員。

第三層，可權衡：

- 個人偏好。
- 接近目標時數。
- 平均分配晚班、週末與不熱門班。
- 避免過多連續工作日、拆班與快速換班。
- 與上一版相比盡量少改動。

### 最佳化順序

不用一組巨大權重把所有目標混在一起，而是採 lexicographic sequential solves：

1. 不可放寬限制必須為 0 違反。
2. 最小化缺班並固定最佳可達覆蓋。
3. 在不惡化覆蓋下，最小化公司規則違反與工時偏差。
4. 在前述結果固定或僅容許明示的小幅差距下，最佳化跨週期公平與個人偏好。
5. 以 assignment Hamming distance 加入差異限制；只有真的不同才回傳第二、第三個候選。

求解結果必須分開顯示：

- `OPTIMAL`：已證明最佳。
- `FEASIBLE`：時間內找到可用解，但未證明最佳。
- `INFEASIBLE`：已證明無解。
- `UNKNOWN/TIMEOUT`：時間不足，不能對使用者說「無解」。
- `MODEL_INVALID`：系統或模型錯誤，不能歸咎於使用者規則。

每次求解保存 immutable input snapshot、hash、規則版本、OR-Tools 版本、container image digest、seed、worker 數、時間限制、原始結果、分數明細與未滿足項目。目標是「可稽核、可重播」，不承諾跨硬體、版本與多執行緒的 bit-for-bit 完全相同。一般解釋來自確定性的 score breakdown；「為什麼不是某人」才按需做 counterfactual rerun，AI 不可自行猜理由。

## 5. AI 邊界與規則 DSL

### AI 可以做

- 將自然語言轉成既定 JSON Schema。
- 找出日期、時間、班別或強弱程度的歧義。
- 把管理者的對話修改轉成規則新增／修改／刪除建議。
- 用人話解釋求解器已算出的衝突與分數。

### AI 不可以做

- 直接執行它生成的 Python／SQL／Shell。
- 自動發布班表。
- 靜默放寬硬限制。
- 讀取不必要的姓名、電話、Email、健康或家庭資訊。
- 用性別、婚姻、宗教、身心狀態等受保護特徵做「可靠度」或公平評分。

### DSL 欄位

```text
rule_id
type
scope
hardness
parameters
effective_from / effective_to
source: law | company | scheduler | member | ai_parsed
source_reference
confirmed_by / confirmed_at
schema_version
```

首版白名單型別：

```text
UNAVAILABLE(member, interval)
PREFER_SHIFT(member, shift_type, weight)
MAX_WORK_MINUTES(member, period, minutes)
MIN_REST_MINUTES(member, minutes)
REQUIRES_SKILL(shift, skill, quantity)
TARGET_SHIFT_COUNT(member, shift_type, target, weight)
LOCK_ASSIGNMENT(member, shift)
```

### AI 成本與護欄

- 預設使用支援 Structured Outputs 的低成本模型；目前可用 GPT-5.6 Luna，官方牌價為每百萬 input tokens US$0.20、output tokens US$1.20。
- `reasoning_effort: none`，限制輸出約 300–400 tokens。
- 表單已能表達的需求不呼叫 AI。
- 每群組設月額度、輸入長度、速率限制與全域 spend alert。
- 不傳姓名，以 opaque member ID 與 group dictionary 代替。
- 不存在的 ID、低信心或 Schema 驗證失敗時 fail closed。

### 第三方 AI 同意與無 AI 路徑

- 第一次送出文字前，獨立說明 AI 供應商、會送出的資料、目的與保存方式，取得明確同意；不能只藏在服務條款中。
- 不同意或撤回 AI 同意的成員，仍能完整使用結構化表單，不降低基本排班權益。
- API 呼叫使用 `store: false`、關閉資料分享，不使用非必要的 conversation／file store；隱私政策仍揭露供應商可能保留的安全監測資料。
- 送出前在本機提示並遮罩非必要的健康、家庭、宗教、電話與 Email，讓使用者預覽實際傳送內容。
- opaque ID 是假名化，不是匿名化；在小團隊仍應視為個資並套用相同保護。
- 原文經確認後即刪除，僅保留短期復原窗口；業務真正需要保存的是使用者確認後、帶期限的結構化規則。

## 6. 快取設計

不用 Redis，先存在 Postgres：

- MVP 不做自然語言解析快取。「這週日」的語意會隨 reference instant、班表週期與時區改變；為省極少量 token 而重用錯誤日期不划算。
- 真正可重用的是成員已確認的結構化 availability／rules；新週期只複製常態模板，再加入本期例外。
- 若日後做解析快取，key 至少包含 schedule period、organization timezone、reference timestamp、locale、schema／prompt／model version 與 group dictionary hash。
- 求解端只做 idempotent job coalescing：`immutable_input_hash + objective_profile + solver_version`；命中時仍驗證 tenant、permission 與版本。
- 任一成員、班別、規則、法規 pack 或 solver 版本改變即失效；任何含個資結果不得跨 organization 共用。
- 權威 schedule snapshot 留在 Postgres；Storage 只存可重建的 PDF／ICS 等衍生匯出，不當作資料庫備份。

## 7. 建議技術架構

```text
Flutter：Web / Windows / iOS（後期）
        │
        ├── CRUD / Realtime
        ▼
Supabase Auth + Postgres + Storage + RLS
        │
        └── 特權操作：Edge Functions
                 │
                 ├── OpenAI：文字 → 結構化規則
                 ├── DB transactional outbox
                 ├── Cloud Tasks：at-least-once 排隊、重試、限流
                 └── Cloud Run：Python + FastAPI + OR-Tools
```

### Client

- Flutter：共用 Web、Windows、iOS／Android 程式碼。
- Riverpod：狀態管理。
- GoRouter：路由與登入保護。
- Freezed／JSON Schema 產生資料型別。
- 大畫面是管理者班表格；小畫面是成員提交與查看流程，不強迫完全相同版面。

### Backend

- Supabase Auth、Postgres、Storage、RLS。
- 所有多租戶表都有 `organization_id`，複合索引以它開頭。
- Client 不持有 service role；AI 與求解器也不拿通用資料庫管理權限。
- 特權操作經 Edge Function 驗證角色；queue payload 只送 `job_id`，worker 再讀伺服器建立的 immutable snapshot。
- `schedule_runs` 與 `job_outbox` 在同一 DB transaction 寫入；dispatcher 可重送未送達工作，reconciler 回收卡住的 lease。
- worker 用 compare-and-swap claim、bounded retry 與 dead-job 狀態；重複 task 若工作已完成就直接成功回覆，不重算、不重複扣額度。
- 每個 organization 同時最多一個 solver job；idempotency key 至少含 `(organization_id, input_hash, objective_profile)`。
- Supabase Realtime 只作「資料已變」的提示；重新連線或回到 App 時重新抓 canonical state，不能把 event stream 當唯一真相。

### Solver

- Cloud Run request-based，`min instances = 0`、`concurrency = 1`。
- 每次 30 秒硬上限；初期 `max instances = 3`，再依 queue delay 與成本調整。
- Cloud Tasks 控制重試與尖峰。
- 上線初期可大幅落在免費額度內。

### 通知

- App push 用 FCM／APNs。
- Email 只用於登入、邀請與必要 fallback；正式環境使用 custom SMTP 與 SPF／DKIM／DMARC，不依賴 Supabase 測試用寄信服務。
- 不使用付費 SMS。
- Windows 首版用 App 內通知與 Email。
- 截止提醒由單一全域 scanner 查詢 `next_deadline_at`，不可為每個 group 建一個 cron。
- `notification_outbox` 在發布 transaction 內建立，commit 後才發送；以 event／user／channel 唯一鍵去重，支援 TTL、重試、多裝置與失效 token 清理。
- Push payload 只放 version ID 與 deep link，不放班別或請假理由；App 開啟後重新抓權威版本。
- APNs／FCM 接受訊息不等於已讀；以 App 的 version fetch／ack 區分送達、查看、確認與回報問題。
- 發布後連續修改只通知受影響成員並可合併 digest；App 內 inbox 是 durable fallback。

### 離線與多人編輯契約

- 離線保證只讀最後一次成功同步的 Published 班表；管理員離線編班首版不支援。
- 成員 availability 可保存本機草稿，但 server ACK 前只能顯示「尚未同步」，不能顯示已提交。
- Deadline 以 server UTC 搭配 organization IANA timezone 判定，並在 UI 寫清楚是否包含截止分鐘。
- 截止後才同步的資料一律成為 late amendment，不能改寫已凍結的 input snapshot。
- draft／published version 都帶 revision／ETag；save 與 publish 用 compare-and-swap，`409` 顯示 server revision 與差異。
- 拖曳、多格貼上是一個 transaction；Undo／Redo 用可持久化 command log。
- 為省成本，早期限制同時僅一位 Scheduler 編輯；presence 只作提示，不能當鎖定機制。
- 本機快取依 account／organization 隔離，登出即清除；敏感 token 存 OS secure storage。

### 建議 monorepo

```text
apps/client_flutter/
services/solver_python/
supabase/migrations/
supabase/functions/
packages/contracts/
docs/
```

跨語言契約以 JSON Schema／OpenAPI 為真相，產生 Dart 與 Python 型別，避免兩端各自猜規則格式。

## 8. 核心資料模型

- `organizations`, `locations`
- `users`, `memberships`, `worker_profiles`
- `roles`, `skills`, `worker_skills`
- `shift_templates`, `schedule_periods`, `shift_instances`
- `staffing_demands`
- `availability_submissions`, `availability_intervals`
- `raw_text_requests`
- `rules`, `rule_versions`, `rule_confirmations`
- `input_snapshots`
- `schedule_runs`, `job_outbox`, `job_attempts`
- `schedules`, `schedule_versions`
- `assignments`, `violations`, `score_breakdowns`
- `change_requests`, `audit_events`
- `privacy_notices`, `consent_records`
- `retention_policies`, `deletion_requests`, `deletion_tombstones`
- `invitations`, `notification_outbox`, `notification_deliveries`
- `device_installations`, `notification_preferences`
- `billing_accounts`, `plans`, `provider_purchases`, `provider_event_inbox`
- `subscription_periods`, `entitlements`, `usage_ledger`
- `import_jobs`, `import_rows`, `calendar_feed_tokens`

所有可編輯 aggregate 都有 revision／ETag；所有 tenant child 以 `(organization_id, foreign_id)` 複合外鍵阻止跨組織關聯。`raw_text_requests` 與結構化規則分開保存；原始文字在確認後刪除，只留短期復原窗口。稽核紀錄只留必要 diff，不記錄 Token、密碼、完整 Prompt 或健康原因。需另建 retention matrix，逐項定義 raw request、草稿、solver job、通知、audit、安全紀錄、歷史班表、帳務與備份的目的、owner、TTL、刪除／匿名化方式；還原備份後要重播 deletion tombstones。

## 9. MVP 與延後項目

### MVP 必做

- 建立群組、單一據點、角色與邀請。
- 固定班別模板與每日人力／技能需求。
- 每期可點擊流程軌、週期駕駛艙與 Issue Center。
- 手機端「常態模板＋本期例外」，明確區分可以、偏好、不可排、未填。
- 繁體中文自由文字轉規則、預覽與確認。
- 缺交與矛盾檢查。
- 1–3 個有實質差異的候選班表、狀態、分數明細與衝突原因組。
- 桌面覆蓋／人員雙視圖、待補班次、右側情境檢查器。
- 鎖定、拖放、鍵盤替代、持久化 Undo／Redo、三種重新求解模式與套用前差異預覽。
- 簡化版對話修改 → 規則 diff → 確認 → 重算。
- Draft／Published immutable versions、逐班 diff、發布 transaction、通知與成員確認狀態。
- 成員／技能／班別模板 CSV 匯入；CSV、A4 橫式 PDF 與 ICS 匯出。
- RLS、outbox、備份還原、稽核、額度、entitlement 與帳務對帳。

### 為控制年費而延後

- 打卡、出勤、薪資、加班費、勞健保與政府申報。
- 二／四／八週彈性工時、84-1 與複雜產業規則。
- 醫療、航空、保全與 24 小時照護。
- 多據點、人員跨店與交通時間。
- 換班市集、即時聊天室。
- LINE Messaging API、SMS、電話客服、人工代排。
- POS／需求預測與營收整合。
- SSO、SCIM、企業報表與客製串接。
- 完整離線編輯、多語系。
- XLSX 匯入與複雜試算表公式；首版只收乾淨 CSV 或貼上表格資料。

## 10. 定價與單位經濟

### 建議方案

免費試用：

- 1 Group、最多 5 位排班成員。
- 結構化報班與手動排班。
- 2 個完整 AI 排班週期，不設永久無限免費 AI。

Team：

- NT$99／月或 NT$990／年，建議主推年繳。
- 1 Group、1 據點、最多 20 位 active members。
- 每年最多 60 個排班週期。
- 每週期最多 10 次 AI 對話修改。
- 結構化修改與求解器合理重跑不另外收費。
- 員工永遠免費。

`active member` 定義為本計費週期內可被排入班表且未停用的 worker；未接受邀請者不計、Owner／Scheduler 若同時也是 worker 才計。方案與上限必須 versioned，既有訂閱不被日後改價靜默改寫。

第二據點或第二個獨立群組可再購一份 Team；單一排班群組超過 20 人則改用未來的 Business 方案，不能要求客戶硬拆群組。Core 年費維持不超過 NT$1,000。

### 為何不能完全無限

單次 AI 與求解成本很低，但無限用量會造成惡意使用、尖峰與客服爭議。限制的應是自然語言與高成本功能，而不是結構化表單或正常重新排班。

### 保守成本模型

不要用統一的「收入扣 20%」估算所有通路。每個通路各算一份 P&L：

| 通路 | NT$990 標價的計算起點 | 另列項目 |
|---|---:|---|
| Web／Windows 自有金流 | 約 `990 × (1 − 2.8%)`，尚未扣固定費與稅 | 發票、退款、chargeback、金流固定費 |
| Apple IAP | 符合 15% 方案時約 `990 × 85%` | 稅區、匯率、退款、資格變更 |
| Google Play Billing | 適用 15% 費率時約 `990 × 85%` | 稅區、匯率、退款、方案規則 |

技術成本採公式而不是假精確單價：

```text
monthly_cost = fixed_platform
             + AI_input/output_tokens
             + solver_CPU_seconds
             + email/push
             + database/storage/egress/realtime
             + 30–50% safety buffer
```

每月都看三種情境：`base`（幾乎沒用）、`expected`（12 人、4 週期、約 35 次 AI 呼叫）、`abuse`（輸入與重試打滿）。abuse 不能只靠預算警報，必須由文字長度、組織 quota、同時 job 上限、Cloud Run max instances 與 global kill switch 封頂。

約 20–30 個付費群組可能覆蓋共享基礎雲端、網域與商店帳號等「基礎設施現金支出」，不等於商業損益打平；創辦人薪資、客服、行銷、公司、會計、稅務、退款、測試裝置與法律審查皆未包含。若需要人工代排或即時客服，NT$990／年不可行。

### 通路選擇

- Web 台灣金流：藍新牌價國內卡 2.8%，支援定期定額；年繳可減少固定交易與失敗重試。
- Microsoft Store：新註冊入口目前免註冊費；非遊戲 App 可使用自有 commerce，最適合第一波低成本上架。
- Apple：Developer Program US$99／年；符合 Small Business Program 時 IAP commission 15%。
- Google Play：US$25 一次性註冊費；符合適用條件的自動續訂方案通常以 15% 規劃。

商業模式要在寫付款前凍結，不能到上架才補：

1. 首發 Web／Windows 使用自有金流。
2. iOS 若做免費 companion，App 內不放購買或引導外部付款 CTA；若要在 iOS 內升級，從一開始做 StoreKit 與 server-side entitlement。
3. Google Play 內販售雲端數位功能，保守預設使用 Play Billing，除非屆時明確符合並加入其他適用方案。

付款成功畫面不能直接開權限；後端驗證 receipt／purchase token。provider webhook 必須驗簽、去重、容忍亂序，並以每日 reconciliation 校正。帳務狀態至少包含 `trialing｜active｜grace｜hold｜paused｜expired｜revoked｜refunded`，支援 restore purchase、取消、退款、撤銷與價格變更。payer、group owner 與登入 user 分離；Owner 轉移或付款人刪帳不能使群組孤兒化。要防止同一 group 在 Web、Apple、Google 重複訂閱；usage 以 logical request ID 寫 append-only ledger，重試不重複扣額度。

到期後仍可讀取、列印與匯出既有班表，只封鎖新週期、AI 與 solver。iOS 提供 App 內刪除帳號；Google Play 同時提供 App 內與公開 Web 刪除入口。若新增 Google／Facebook 登入，再評估 Sign in with Apple；最低成本先用自有 email OTP。Windows 發行仍需 Windows 裝置、VM 或 CI 實際編譯與驗證，列入測試成本。

### 匯入與匯出安全

- CSV 匯入先進 staging，提供欄位對應、preview、逐列錯誤、重複與時區檢查，確認後才單一 transaction 套用；匯入內容不送 AI。
- 繁中 CSV 使用 UTF-8 BOM；匯出時中和以 `= + - @` 或控制字元開頭的欄位，避免 formula injection。
- A4 橫式 PDF 帶店名、期間、時區、版本、產生時間與圖例。
- ICS assignment 使用穩定 UID、`SEQUENCE`；取消使用 `STATUS:CANCELLED`。訂閱式 calendar feed 每位成員使用可撤銷的高熵 token，且不可寫入 log。
- 組織 JSON／ZIP portability export 與個人資料 export 分開；個人下載不可包含其他成員的私人原文。

## 11. 台灣法規與隱私邊界

這一節是產品設計，不是法律意見；正式商用前需由熟悉台灣勞動法與個資法的專業人士審查。

- 一般工時、延長工時、休息日與彈性工時有多種例外；MVP 只支援一般工時制。
- 輪班更換班次原則有連續 11 小時休息，但特定工作與完成法定程序者存在例外，不能硬寫成所有公司的絕對結論。
- 連續工作 4 小時原則至少休息 30 分鐘，因此班別應能儲存工作段與休息段。
- 未成年、妊娠／哺乳、法定假別與特休有獨立保護；MVP 若不完整支援，建立群組時必須清楚提示。
- 預定班表不等於逐日到分鐘且需保存的實際出勤紀錄。
- 個資蒐集前需告知蒐集者、目的、類別、期間／地區／對象／方式、權利與不提供的影響。
- 若將文字送往境外第三方 AI，第一次傳送前另行揭露供應商、資料、目的與保存方式並取得明確同意；假名 ID 只降低風險，不等於匿名。
- 提供查閱、匯出、更正、停止利用與刪除流程。
- 分開處理「退出 group」「刪除個人帳號」「刪除 organization」，包括 owner transfer、資料保留、訂閱與其他成員資料的影響。
- 個資事件需有調查、遏止、通知與紀錄流程；不可自行聲稱法律一律要求 72 小時通知。

規則需有 `jurisdiction`、`effective_from`、`effective_to`、來源連結與遠端停用能力。法規變動後，舊 schedule 仍保留當時版本，新週期才套用新版本。

## 12. 安全與可靠性上線門檻

- 每張多租戶表採預設拒絕 RLS，並測試跨 tenant 的讀、寫、匯出皆被拒絕。
- tenant child 使用複合外鍵，涵蓋 views、RPC、exports、Storage 與 secret-key worker 的負向測試。
- 邀請 Token 至少 128-bit 隨機、DB 只存 hash、單次使用、短 TTL、可撤銷，必要時綁定 Email。
- AI、solver 與 free trial 前完成 Email 驗證，搭配 CAPTCHA、IP／user／organization rate limit、邀請與匯入上限。
- 管理者可撤銷 session；Owner／Scheduler 支援並要求 2FA，匯出、刪除與改角色使用 step-up authentication。
- secrets 只存在 server，金鑰輪替與最低權限。
- 傳輸與靜態加密；權威班表及 version snapshot 全留在 Postgres，Storage 只放可重建衍生檔。
- Supabase Pro 每日 DB 備份之外，定期建立加密的 off-provider logical backup／audit archive；每季實際還原演練並重播 deletion tombstones。
- 早期內部復原目標先定為 `RPO ≤ 24 小時、RTO ≤ 8 小時`；若試點不能接受，再付費提高備份頻率，而不是首日購買高價 PITR。
- optimistic locking、revision／ETag 與 compare-and-swap 防止兩位管理者互相覆蓋。
- 求解、通知、付款事件等所有背景工作使用 idempotency key、transactional outbox 與 bounded retry。
- Cloud Run 設 timeout、max instances 與帳務警報。
- AI 每群組 quota、全域成本警報與 kill switch。
- 法規 pack、AI parser、prompt、schema 與 solver 都能版本化及回滾。
- 發布、刪除、匯出與權限變更皆寫 audit event。
- 原始自由文字只對本人與必要管理流程可見，不作組內貼文；若未來允許成員互看內容，再補完整 UGC 檢舉與處理機制。

### 發布一致性

發布在一個 DB transaction 內完成：檢查 draft revision、建立 immutable Published version、更新 current pointer、寫 audit event、建立 notification outbox。通知只能在 commit 成功後發；任何一步失敗都不可出現「成員收到通知但 App 沒有該版本」。

### 可觀測性與營運手冊

- 全鏈路 correlation ID：Client → Edge → Cloud Tasks → Cloud Run → AI。
- API latency／error／traffic／saturation，以及 oldest pending job、queue delay、solver runtime／status、objective gap、缺班數。
- parser schema failure、低信心率、使用者修正率與敏感資料遮罩率。
- publish CAS conflict、通知 accepted／viewed、bounce、stale token。
- billing webhook age、reconciliation drift、quota、DB／egress／Realtime 與實際成本。
- backup 成功不等於可還原；記錄 restore drill 日期、時間與結果。

先定內部 SLO，不在 NT$990 方案承諾公開 SLA。上線前至少準備 DB outage、failed publish、dead job、payment drift、notification failure 與 restore runbook。

## 13. 測試策略與驗收

### 求解器

- property-based tests：隨機資料下不可出現硬限制違反。
- infeasible fixtures：確認不會靜默放寬規則。
- 跨午夜、月底、時區、休息段、技能到期、截止後修改。
- 驗證 `OPTIMAL／FEASIBLE／INFEASIBLE／UNKNOWN／MODEL_INVALID` 不會被 UI 混淆。
- 驗證 assumptions 回傳的衝突原因組確實足以造成無解，不把它誤稱為最小集合。
- 20 人、200 個班次，10 秒內先回傳至少 1 個可用候選或清楚 job status；30 秒硬上限內最多 3 個有實質差異的候選，硬限制違反必須為 0。

### AI

- 建立至少 200 條台灣常見排班語句的固定 eval dataset。
- 每條輸出都能追溯原句，未確認不得生效。
- 測試模糊日期、否定、雙重否定、相對日期、跨日、衝突句與不存在的人名／班別。
- 模型或 prompt 更新需通過回歸 eval 才能上線。

### 權限

- RLS allow/deny 測試覆蓋 Owner、Scheduler、Member 與非成員。
- 驗證自由文字與私人理由不會出現在他人 API、通知、log 或匯出。
- 驗證重複／亂序 job、付款 webhook 與通知事件不會重算、重複扣款、重複扣額度或發布錯版。
- 驗證斷線、過期 revision、截止瞬間提交、同時拖曳與舊版確認都會安全失敗並顯示可理解的修復方式。

### UX 與無障礙驗收

- 5 位第一次使用的管理者能在 15 分鐘內從建立週期走到可審核草稿，且能正確說出目前卡點。
- 8 位成員中至少 7 位能在 90 秒內提交本期例外，且不會把「未填」誤認成「可以」。
- 管理者能找到缺口、理解阻擋原因、改派、復原、只補空缺並看懂套用前差異。
- 以純鍵盤、200% 縮放、Windows Narrator 與手機螢幕閱讀器跑完提交、修正與查看班表的核心路徑。

### 產品指標

- 管理者 15 分鐘內完成首次建組與排班。
- 80% 以上成員不需教學即可提交；正式 Beta 目標提高到 90%。
- 產生後人工修改 assignment 比率逐步低於 15%。
- North Star：每月成功發布的 schedule periods／活躍付費群組。
- 觀察第二、第四與第十二個週期留存，而不只看下載量。

## 14. 開發路線圖與決策閘門

### Phase 0：需求驗證（1–2 週）

- 訪談 8–10 位真實排班管理者。
- 收集 3 份匿名真實班表與需求訊息。
- 用可點擊 prototype 測試「流程控制台＋班表畫布」，至少包含成員提交、缺口檢查、右側 inspector、改派、Undo 與發布前 diff。
- 確認人數、週期、班別、工時制、未成年比例、目前工具與願付價格。
- 成功門檻：上述 UX 任務通過，且至少 3 個團隊同意連續測試 4 個排班週期。

### Phase 1：Solver 與規則 spike（1–2 週）

- 實作 DSL validator、CP-SAT 最小模型、1–3 方案、求解狀態與 conflict report。
- 用真實匿名資料測試速度與可解釋性。
- 若 20 人典型案例無法在 10 秒內穩定求解，先縮小規則，不進 UI 大開發。

### Phase 2：可用 Alpha（3–5 週）

- Flutter responsive shell、登入、Group、成員、班別、需求收集。
- 流程軌、覆蓋／人員視圖、inspector、手動排班、候選方案、版本與 transaction publish。
- 本機 Supabase 開發、Cloud Run solver。

### Phase 3：AI 與封閉 Beta（2–4 週）

- 自然語言規則、獨立同意、無 AI 路徑、確認流程、對話 diff 與額度。
- RLS 負向測試、outbox、custom SMTP、off-provider backup、錯誤監控與隱私告知。
- 5–10 個團隊連續跑至少 4 個週期。

### Phase 4：付費與 Windows 上架（2–3 週）

- 年繳、receipt 驗證、entitlement、退款／撤銷／到期、對帳與 Microsoft Store 包裝。
- 第一位付費客戶前升 Supabase Pro。
- 只有 Beta 留存與客服量可接受才公開銷售。

### Phase 5：Mobile Store（2–4 週，審核時間另計）

- 先依試點手機使用比例決定 Google Play 或 iOS 優先。
- 若 iOS 做 companion，移除所有 App 內外部付款引導；若 App 內升級則完成 StoreKit entitlement。
- 準備 App Privacy／Data Safety、SDK privacy manifest／required-reason API、帳號刪除入口、restore/manage subscription 與審查 demo account。

以一人全程 vibe coding 粗估：10–14 週可得到可測試 Beta，18–26 週才比較接近能安全收費上架的產品；上架審核、法律文件、Windows 測試環境與真實試點另計。Vibe coding 可縮短程式撰寫，但不能省略資料隔離、求解驗證、付款生命週期、備份還原與法規測試。

## 15. 高風險情況與預先對策

| 風險 | 等級 | 對策 |
|---|---:|---|
| AI 誤解需求 | P0 | 白名單 DSL、Schema 驗證、逐條預覽、人類確認 |
| 無解時偷放寬限制 | P0 | 法律／安全／不可上班永不自動放寬，只顯示缺班與衝突 |
| 跨群組資料外洩 | P0 | RLS、最低權限、負向權限測試、服務帳號分離 |
| 誤稱法律合規 | P0 | 只稱規則檢查，標出制度、版本與未支援範圍 |
| 私人原因被同事看到 | P0 | 不收原因，原始文字隔離且短期刪除 |
| 公平模型造成歧視 | P1 | 只用工作相關變數、公開分數、法定假別不扣分 |
| 同時編輯發布錯版 | P1 | immutable versions、optimistic locking、compare-and-swap publish |
| 通知沒送達 | P1 | Published 版本為唯一真相，記錄送達與重試 |
| DB 寫入成功但 queue 未排入 | P0 | transactional outbox、dispatcher 重送、reconciler 回收 |
| Timeout 被誤報為無解 | P0 | 求解狀態分離、保留 job diagnostics、允許安全重試 |
| 付款成功卻無權限／退款仍有權限 | P0 | server receipt 驗證、event inbox、entitlement reconciliation |
| 離線草稿被誤認為已提交 | P0 | server ACK 才顯示已提交，逾期同步變 late amendment |
| AI／CPU 被濫用 | P1 | quota、rate limit、timeout、max instances、billing alerts |
| 規則過期 | P1 | effective dates、年度審查、遠端停用與版本保留 |
| 低價被客服吃光 | P1 | 自助 onboarding、App 內診斷、Email-only support、企業需求升級 |

## 16. 目前尚未決定、必須由產品負責人回答

以下五題會直接改變下一版 PRD 與資料模型，需先回答：

1. NT$1,000 是每個群組／據點、每位管理者，還是每位員工？本規劃建議每群組／據點，員工免費。
2. 第一批真實客群是餐飲、零售，還是其他產業？是否已有可測試店家？
3. 首版是否接受只支援一般工時制？
4. 第一批店家是否會排未滿 18 歲成員？
5. 首版是否確認只做預定班表，不做打卡、實際出勤與薪資？

在這五題確定前，可以做 UX prototype 與 solver spike，但不應直接把完整法規 pack、付款與資料庫 schema 寫死。

以下三題可在 prototype 測試後決定，不必阻擋第一個 UX spike：

6. 公開班／搶班是核心需求，還是 V1.1？首版建議只允許管理者把缺口設為 Open Shift，成員表示意願後仍由管理者確認。
7. 是否允許含缺口發布？建議允許但必須二次確認、清楚標記未填班次；硬限制錯誤永遠阻擋。
8. 成員「確認班表」是必要回覆還是只做已查看？建議首版提供確認與回報問題，但不因未確認自動撤銷已發布班表。

## 17. 主要查核來源

以下價格、商店政策與平台能力以 2026-09-04 查核內容為規劃基準，上架與採購前必須重新確認。

- [Google OR-Tools employee scheduling](https://developers.google.com/optimization/scheduling/employee_scheduling)
- [OR-Tools CP-SAT solver status](https://developers.google.com/optimization/cp/cp_solver)
- [OR-Tools CP-SAT troubleshooting](https://github.com/google/or-tools/blob/stable/ortools/sat/docs/troubleshooting.md)
- [Flutter supported platforms](https://docs.flutter.dev/reference/supported-platforms)
- [Flutter Windows setup](https://docs.flutter.dev/platform-integration/windows/setup)
- [Flutter Web FAQ](https://docs.flutter.dev/platform-integration/web/faq)
- [Supabase pricing](https://supabase.com/pricing)
- [Supabase Row Level Security](https://supabase.com/docs/guides/database/postgres/row-level-security)
- [Supabase production checklist](https://supabase.com/docs/guides/deployment/going-into-prod)
- [Supabase Realtime troubleshooting](https://supabase.com/docs/guides/troubleshooting/realtime-postgres-changes-troubleshooting)
- [Supabase database backups](https://supabase.com/docs/guides/platform/backups)
- [Supabase custom SMTP](https://supabase.com/docs/guides/auth/auth-smtp)
- [Cloud Run pricing](https://cloud.google.com/run/pricing)
- [Cloud Run max instances](https://cloud.google.com/run/docs/configuring/max-instances)
- [Cloud Tasks overview](https://cloud.google.com/tasks/docs/dual-overview)
- [Google Cloud budgets](https://cloud.google.com/billing/docs/how-to/budgets)
- [GPT-5.6 Luna 官方資料](https://developers.openai.com/api/docs/models/gpt-5.6-luna)
- [OpenAI API data controls](https://platform.openai.com/docs/guides/your-data)
- [Apple Developer Program](https://developer.apple.com/support/compare-memberships/)
- [Apple Small Business Program](https://developer.apple.com/app-store/small-business-program/)
- [Apple App Review Guidelines](https://developer.apple.com/app-store/review/guidelines/)
- [Apple App 內帳號刪除要求](https://developer.apple.com/support/offering-account-deletion-in-your-app/)
- [Apple third-party SDK requirements](https://developer.apple.com/support/third-party-SDK-requirements/)
- [Apple App Store Server Notifications status](https://developer.apple.com/documentation/appstoreservernotifications/status)
- [Google Play service fees](https://support.google.com/googleplay/android-developer/answer/112622)
- [Google Play Payments policy](https://support.google.com/googleplay/android-developer/answer/9858738)
- [Google Play User Data policy](https://support.google.com/googleplay/android-developer/answer/10144311)
- [Google Play Billing lifecycle](https://developer.android.com/google/play/billing/lifecycle)
- [Microsoft Store publishing](https://learn.microsoft.com/en-us/windows/apps/publish/get-started)
- [藍新金流費率](https://www.newebpay.com/website/Page/content/service_fare)
- [台灣勞動基準法](https://laws.mol.gov.tw/FLAW/FLAWDAT0201.aspx?id=FL014930)
- [個人資料保護法第 8 條告知事項](https://www.pdpc.gov.tw/News_Content/100/295/)
- [性別平等工作法](https://laws.mol.gov.tw/FLAW/FLAWDAT0201.aspx?id=FL015149)
- [台灣競品好排班價格](https://schedule.tw/pricing/)
- [Vome Sequences workflow](https://support.vomevolunteer.com/portal/en/kb/articles/how-do-i-manage-users-in-a-sequence-on-vome)
- [Better Impact scheduling guide](https://support.betterimpact.com/en/articles/13192494-comprehensive-guide-to-scheduling)
- [When I Work scheduler reference](https://help.wheniwork.com/articles/scheduler-reference-guide-computer/)
- [Deputy schedule overview](https://help.deputy.com/hc/en-au/articles/4688713423759-Schedule-overview)
- [7shifts publishing workflow](https://kb.7shifts.com/hc/en-us/articles/4417514210067-How-to-publish-a-schedule)
- [Bizimply published schedule lock](https://support.bizimply.com/en/articles/3997997-what-does-the-lock-symbol-mean-on-a-schedule)
- [W3C ARIA grid pattern](https://www.w3.org/WAI/ARIA/apg/patterns/grid/)
- [WCAG 2.2: Use of Color](https://www.w3.org/WAI/WCAG22/Understanding/use-of-color)
- [WCAG 2.2: Target Size Minimum](https://www.w3.org/WAI/WCAG22/Understanding/target-size-minimum)
- [OWASP CSV Injection](https://owasp.org/www-community/attacks/CSV_Injection)
- [RFC 5545 iCalendar](https://www.rfc-editor.org/rfc/rfc5545)
- [Google SRE: Monitoring Distributed Systems](https://sre.google/sre-book/monitoring-distributed-systems/)
