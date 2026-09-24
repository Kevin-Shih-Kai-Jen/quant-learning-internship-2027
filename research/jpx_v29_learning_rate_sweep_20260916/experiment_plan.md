# v29：五種learning rate倍率，15組validation對照

依使用者新指示，Sharpe最高為優先，Rank IC作第二順位；Rank IC不設必須大於0的門檻，MSE不參與選擇。MSE仍為既有訓練loss，保留更新核對；不因修改評估優先順序更換loss。

在既有全部953日validation上比較，不做年度外層選參數；test保留至候選選定後總體驗證，本輪不讀取或評估test。初始參數、每日可用標籤時序與原online expanding訓練不變。

基準共同eta_t=min(eta_A,t,eta_B,t)，eta_j,t=1/(2*lambda_max(X_j,t^T X_j,t/N_t))，由當時已到期批次決定。固定倍率c ∈ {0.25,0.5,1,1.5,1.9}，每組整段使用相同c，每日步長c*eta_t；同c的純g、A、B具有逐日完全相同的步長。全數15組從零訓練，c=1另外精確重現v28。

A=g+beta*new；B=g+r1*(new-old)+r2*old。訊號、來源、單季換算、只保留最新狀態及exp(-a/9)沿用v27，不改EPS、不排除極端值，原PR1/VR1，價量與財報共同訓練，每日一次MSE更新。僅調learning rate倍率。所有倍率小於2，逐批確認更新後MSE不增加，僅作數值正確性核對。

輸出全期Sharpe優先排名及Rank IC，並提供逐期數據、每個模型最佳倍率、每個倍率的三模型對照，以及原v7。等Sharpe才用Rank IC破同分。以完整validation選出的最佳為待test驗證候選，不自動取代正式v7。配對20日區塊4000次bootstrap只作探索，不作候選門檻或已經通過test的證據。

核對全部更新、預測、官方Sharpe、Rank IC、同倍率同日步長與樣本相同、c=1重現v28、歷史來源未改。新實驗defaults作快照避免日後設定更新改變本輪紀錄。
