const STORAGE_KEY = 'scheduling-app-mvp-v1';

const DAY_ALIASES = {
  mon: 0,
  monday: 0,
  星期一: 0,
  monday_: 0,
  tue: 1,
  tuesday: 1,
  星期二: 1,
  wed: 2,
  wednesday: 2,
  星期三: 2,
  thu: 3,
  thursday: 3,
  星期四: 3,
  fri: 4,
  friday: 4,
  星期五: 4,
  sat: 5,
  saturday: 5,
  星期六: 5,
  sun: 6,
  sunday: 6,
  星期日: 6
};

const DEFAULT_SHIFTS = [
  { id: 'morning', label: '早班', start: '08:00', end: '12:00', required: 1 },
  { id: 'afternoon', label: '中班', start: '13:00', end: '17:00', required: 1 },
  { id: 'evening', label: '晚班', start: '17:30', end: '21:30', required: 1 }
];

const SOLVE_PROFILES = {
  fast: {
    label: '加速',
    yieldEveryMs: 24,
    sleepMs: 0,
    heartbeatEveryMs: 400,
    maxRunMinutesDefault: 2,
    maxCandidates: 5
  },
  balanced: {
    label: '平衡',
    yieldEveryMs: 48,
    sleepMs: 4,
    heartbeatEveryMs: 650,
    maxRunMinutesDefault: 5,
    maxCandidates: 3
  },
  night: {
    label: '夜間節能（長時）',
    yieldEveryMs: 96,
    sleepMs: 12,
    heartbeatEveryMs: 900,
    maxRunMinutesDefault: 240,
    maxCandidates: 3
  }
};

const THROTTLE_LEVELS = [
  { level: 0, factor: 1 },
  { level: 1, factor: 1.35 },
  { level: 2, factor: 1.65 },
  { level: 3, factor: 2.05 }
];

const SOLVER_GUARD = {
  batteryRefreshMs: 12000,
  stopLevel: 0.05,
  heavyThrottleLevel: 0.08,
  warningLevel: 0.12,
  lowPower: 0.22,
  unsupportedStopMinutes: 360,
  visibilityHiddenPenalty: 0.45,
  maxFactor: 3.4
};

const SOLVER_RESUME = {
  checkpointIntervalMs: 15 * 60 * 1000,
  maxSavedCandidates: 20,
  maxStackFrames: 12,
  maxCheckpointCandidates: 20
};

let state = loadState();

let els;
let solveRunning = false;
let solveToken = 0;
let shouldRefreshPowerStateNow = false;

function sleep(ms) {
  if (ms <= 0) return Promise.resolve();
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function formatMs(ms) {
  if (!Number.isFinite(ms) || ms < 0) return '—';
  if (ms < 1000) return `${Math.round(ms)}ms`;
  const sec = ms / 1000;
  if (sec < 60) return `${sec.toFixed(1)}s`;
  const min = Math.floor(sec / 60);
  const leftSec = (sec % 60).toFixed(1);
  return `${min}m ${leftSec}s`;
}

function setSolveControls(isBusy) {
  if (!els?.solveBtn || !els?.stopSolveBtn) return;
  els.solveBtn.disabled = isBusy;
  els.stopSolveBtn.disabled = !isBusy;
  if (els?.solveResumeBtn) {
    els.solveResumeBtn.disabled = isBusy;
  }
}

function uid() {
  return `${Date.now().toString(36)}${Math.floor(Math.random() * 9999).toString(16)}`;
}

function todayStr() {
  const t = new Date();
  return `${t.getFullYear()}-${String(t.getMonth() + 1).padStart(2, '0')}-${String(t.getDate()).padStart(2, '0')}`;
}

function safeDateAdd(dateString, deltaDay) {
  const d = new Date(`${dateString}T00:00:00`);
  d.setDate(d.getDate() + deltaDay);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
}

function dayLabel(dateString, offset) {
  const d = new Date(`${dateString}T00:00:00`);
  d.setDate(d.getDate() + offset);
  const weekday = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
  const zh = ['日', '一', '二', '三', '四', '五', '六'];
  return `${String(d.getMonth() + 1)}/${String(d.getDate()).padStart(2, '0')} (${zh[d.getDay()]})`;
}

function ensureDefaults() {
  if (!Array.isArray(state.groups)) state.groups = [];
  if (!state.activeGroupId && state.groups.length > 0) state.activeGroupId = state.groups[0].id;
}

function loadState() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return { groups: [], activeGroupId: null, activePeriodId: null };
    }
    const parsed = JSON.parse(raw);
    return parsed || { groups: [], activeGroupId: null, activePeriodId: null };
  } catch {
    return { groups: [], activeGroupId: null, activePeriodId: null };
  }
}

function saveState() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}

function findGroupById(groupId) {
  return state.groups.find((g) => g.id === groupId) || null;
}

function findPeriodById(group, periodId) {
  if (!group) return null;
  return group.periods.find((p) => p.id === periodId) || null;
}

function getActiveGroup() {
  return findGroupById(state.activeGroupId);
}

function getActivePeriod(group) {
  if (!group) return null;
  if (!state.activePeriodId) return group.periods[0] || null;
  return findPeriodById(group, state.activePeriodId) || group.periods[0] || null;
}

function newGroup(name) {
  return {
    id: uid(),
    name,
    createdAt: Date.now(),
    shifts: JSON.parse(JSON.stringify(DEFAULT_SHIFTS)),
    members: [],
    periods: []
  };
}

function newMember(name, maxShiftsPerPeriod) {
  return {
    id: uid(),
    name,
    maxShiftsPerPeriod: Number(maxShiftsPerPeriod || 10)
  };
}

function newInputTemplate(dayCount, shifts) {
  const availability = {};
  const preference = {};
  for (let day = 0; day < dayCount; day++) {
    availability[day] = {};
    preference[day] = {};
    for (const shift of shifts) {
      availability[day][shift.id] = true;
      preference[day][shift.id] = 'neutral';
    }
  }
  return { availability, preference, notes: '' };
}

function newPeriod(name, startDate, dayCount, maxShiftsPerDay, shifts) {
  const shiftDefs = shifts.map((s) => ({
    id: s.id,
    label: s.label,
    start: s.start,
    end: s.end,
    required: Number(s.required || 1)
  }));
  return {
    id: uid(),
    name,
    startDate,
    dayCount: Number(dayCount || 7),
    maxShiftsPerDay: Number(maxShiftsPerDay || 1),
    createdAt: Date.now(),
    shifts: shiftDefs,
    memberInputs: {},
    candidates: [],
    selectedCandidateId: null,
    status: 'draft'
  };
}

function setStatus(text, level = '') {
  if (!els.statusText) return;
  els.statusText.textContent = text;
  els.statusText.className = level;
}

function getSolveProfile() {
  if (!els?.solveProfile) {
    return SOLVE_PROFILES.night;
  }
  const key = els.solveProfile.value || 'night';
  return SOLVE_PROFILES[key] || SOLVE_PROFILES.night;
}

function getSolveBudgetMs(profile) {
  const inputRaw = String(els?.solveRunMinutes?.value || '').trim();
  if (!inputRaw) return Math.max(0, (profile?.maxRunMinutesDefault || 0) * 60 * 1000);
  if (inputRaw === '0') return 0;
  const inputMinutes = Number(inputRaw);
  if (Number.isFinite(inputMinutes) && inputMinutes > 0) {
    return Math.max(0, inputMinutes * 60 * 1000);
  }
  return 0;
}

function normalizeSolverBudgetMs(inputMs) {
  return inputMs > 0 ? inputMs : Number.POSITIVE_INFINITY;
}

function makeSimpleHash(text) {
  let h = 2166136261;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return `h${(h >>> 0).toString(16)}`;
}

function makeSolverFingerprint(group, period, profile, maxCandidates) {
  const fingerprintPayload = {
    g: group.id,
    p: period.id,
    m: group.members.map((m) => ({ id: m.id, name: m.name, max: m.maxShiftsPerPeriod })),
    d: period.dayCount,
    dly: period.maxShiftsPerDay,
    shifts: period.shifts.map((s) => ({ id: s.id, r: s.required })),
    profile: profile?.label || 'night',
    maxC: maxCandidates
  };
  const memberInputs = {};
  for (const m of group.members) {
    memberInputs[m.id] = period.memberInputs?.[m.id] || {};
  }
  fingerprintPayload.inputs = memberInputs;
  return makeSimpleHash(JSON.stringify(fingerprintPayload));
}

function candidateSignature(candidate) {
  const keys = Object.keys(candidate.assignments || {}).sort();
  const packed = keys.map((k) => `${k}:${(candidate.assignments[k] || []).slice().sort().join('|')}`).join('||');
  return `${candidate.score ?? 0}|${candidate.fairness ?? 0}|${candidate.preferenceScore ?? 0}|${packed}`;
}

function mergeCandidateLists(baseCandidates, incomingCandidates, maxLen) {
  const limit = Number.isFinite(maxLen) && maxLen > 0 ? Math.max(1, Math.floor(maxLen)) : 6;
  const map = new Map();

  for (const c of baseCandidates || []) {
    const signature = candidateSignature(c);
    const existing = map.get(signature);
    if (!existing || (c.score || Infinity) < (existing.score || Infinity)) {
      map.set(signature, c);
    }
  }

  for (const c of incomingCandidates || []) {
    const signature = candidateSignature(c);
    const existing = map.get(signature);
    if (!existing || (c.score || Infinity) < (existing.score || Infinity)) {
      map.set(signature, c);
    }
  }

  const merged = [...map.values()];
  merged.sort((a, b) => (a.score || 0) - (b.score || 0));
  return merged.slice(0, limit);
}

function normalizeCandidatesForResume(candidates) {
  return (candidates || []).map((c) => ({
    ...c,
    assignments: JSON.parse(JSON.stringify(c.assignments || {}))
  }));
}

function makeResumeState(period) {
  return {
    status: 'idle',
    startedAt: Date.now(),
    lastUpdated: Date.now(),
    elapsedMs: 0,
    totalNodes: 0,
    totalSegments: 0,
    fingerprint: '',
    profile: '',
    maxCandidates: 3,
    maxMs: 0,
    stack: [],
    stateNow: null,
    assignment: null,
    totalGroups: 0,
    candidates: []
  };
}

function cloneResumeStateNow(raw) {
  if (!raw || typeof raw !== 'object') return null;
  const total = typeof raw.total === 'object' && raw.total !== null ? { ...raw.total } : {};
  const byDay = Array.isArray(raw.byDay)
    ? raw.byDay.map((row) => (Array.isArray(row) ? row.slice() : []))
    : [];
  return { total, byDay };
}

function cloneSolverAssignments(source) {
  if (!source || typeof source !== 'object') return {};
  const out = {};
  for (const [key, values] of Object.entries(source)) {
    if (!Array.isArray(values)) continue;
    out[key] = values.slice();
  }
  return out;
}

function cloneSolverStateForCheckpoint(stateNow, candidateGroups) {
  if (!stateNow) return null;
  return {
    ...stateNow,
    total: typeof stateNow.total === 'object' && stateNow.total !== null ? { ...stateNow.total } : {},
    byDay: Array.isArray(stateNow.byDay) ? stateNow.byDay.map((row) => Array.isArray(row) ? row.slice() : []) : []
  };
}

function normalizeSolverStack(rawStack) {
  if (!Array.isArray(rawStack)) return [];
  return rawStack
    .map((item) => {
      if (!item || typeof item !== 'object') return null;
      const gIndex = Number.isInteger(item.gIndex) ? item.gIndex : Number(item.gIndex);
      if (!Number.isInteger(gIndex) || gIndex < 0) return null;
      const candidateIds = Array.isArray(item.candidateIds)
        ? item.candidateIds.filter((id) => typeof id === 'string' || Number.isInteger(id) || typeof id === 'number')
        : [];
      const indices = Array.isArray(item.indices)
        ? item.indices
            .map((idx) => Number(idx))
            .filter((idx) => Number.isInteger(idx) && idx >= 0)
        : [];
      const required = Number.isFinite(item.required) ? Math.max(0, Math.floor(item.required)) : (typeof item.required === 'number' ? Math.floor(item.required) : 0);
      return {
        gIndex,
        candidateIds,
        indices,
        required,
        applied: !!item.applied
      };
    })
    .filter((item) => item !== null);
}

function sanitizeSolverResumeState(rawState) {
  if (!rawState || typeof rawState !== 'object') return null;
  if (!Array.isArray(rawState.candidates) || typeof rawState.fingerprint !== 'string') return null;
  return {
    status: 'paused',
    startedAt: Number.isFinite(rawState.startedAt) ? rawState.startedAt : Date.now(),
    lastUpdated: Number.isFinite(rawState.lastUpdated) ? rawState.lastUpdated : Date.now(),
    elapsedMs: Number.isFinite(rawState.elapsedMs) && rawState.elapsedMs >= 0 ? rawState.elapsedMs : 0,
    totalNodes: Number.isFinite(rawState.totalNodes) && rawState.totalNodes >= 0 ? rawState.totalNodes : 0,
    totalSegments: Number.isFinite(rawState.totalSegments) && rawState.totalSegments >= 0 ? rawState.totalSegments : 0,
    fingerprint: String(rawState.fingerprint || ''),
    profile: String(rawState.profile || ''),
    maxCandidates: Number.isFinite(rawState.maxCandidates) ? Math.max(1, Math.floor(rawState.maxCandidates)) : 3,
    maxMs: Number(rawState.maxMs || 0),
    stack: Array.isArray(rawState.stack) ? rawState.stack.slice(0, SOLVER_RESUME.maxStackFrames) : [],
    stateNow: rawState.stateNow && typeof rawState.stateNow === 'object' ? rawState.stateNow : null,
    assignment: rawState.assignment && typeof rawState.assignment === 'object' ? rawState.assignment : null,
    totalGroups: Number.isFinite(rawState.totalGroups) ? Math.max(0, Math.floor(rawState.totalGroups)) : 0,
    candidates: normalizeCandidatesForResume(rawState.candidates || [])
  };
}

function isResumeStateCompatible(period, group, profile, maxCandidates, rawResumeState) {
  if (!rawResumeState || !group || !period) return false;
  if (!rawResumeState.fingerprint || !rawResumeState.profile) return false;
  const expected = makeSolverFingerprint(group, period, profile, maxCandidates);
  if (rawResumeState.fingerprint !== expected) return false;
  if (rawResumeState.maxCandidates !== maxCandidates) return false;
  return rawResumeState.totalGroups > 0 ? rawResumeState.totalGroups === buildSolverGroups(period, group).length : true;
}

function buildSolverGroups(period, group) {
  const members = group?.members || [];
  const slots = buildSolverSlots(period, group || { members: [] });
  const periodInputs = {};
  for (const m of members) {
    periodInputs[m.id] = buildMemberInput(period, m.id);
  }
  const initialState = {
    total: {},
    byDay: {}
  };
  for (const m of members) {
    initialState.total[m.id] = 0;
    initialState.byDay[m.id] = Array(period.dayCount).fill(0);
  }
  const ordered = slots
    .map((slot) => {
      const candidates = members.filter((m) => canAssignWorker(m.id, slot.day, slot.shiftId, period, periodInputs, initialState, m));
      return {
        ...slot,
        candidateIds: candidates.map((m) => m.id)
      };
    })
    .filter((s) => s.candidateIds.length > 0)
    .sort((a, b) => {
      const da = a.candidateIds.length;
      const db = b.candidateIds.length;
      if (da !== db) return da - db;
      if (a.day !== b.day) return a.day - b.day;
      return a.shiftId.localeCompare(b.shiftId);
    });
  const groups = [];
  let i = 0;
  while (i < ordered.length) {
    const first = ordered[i];
    const same = [];
    while (i < ordered.length && ordered[i].day === first.day && ordered[i].shiftId === first.shiftId) {
      same.push(ordered[i]);
      i++;
    }
    groups.push({
      day: first.day,
      shiftId: first.shiftId,
      required: same.length,
      key: first.key
    });
  }
  return groups;
}

function getSolverResumeState(period) {
  return sanitizeSolverResumeState(period?.solverCheckpoint || null);
}

function clearSolverResumeState(period) {
  if (!period) return;
  delete period.solverCheckpoint;
}

if (typeof document !== 'undefined') {
  document.addEventListener('visibilitychange', () => {
    if (solveRunning) {
      shouldRefreshPowerStateNow = true;
    }
  });
}

async function detectPowerState() {
  if (typeof navigator === 'undefined' || typeof navigator.getBattery !== 'function') {
    return { lowPower: false, source: 'unsupported', level: 1, charging: true, levelPercent: 100 };
  }

  try {
    const battery = await navigator.getBattery();
    const level = Number.isFinite(Number(battery.level)) ? Number(battery.level) : 1;
    const lowPower = !battery.charging && level <= SOLVER_GUARD.lowPower;
    return {
      lowPower,
      level,
      levelPercent: Math.round(level * 100),
      charging: battery.charging,
      source: 'battery'
    };
  } catch {
    return { lowPower: false, source: 'error', level: 1, charging: true, levelPercent: 100 };
  }
}

function formatPercent(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) return '—';
  const v = Math.max(0, Math.min(100, Math.round(value * 100)));
  return `${v}%`;
}

function describePowerState(powerState) {
  if (!powerState || typeof powerState.levelPercent !== 'number' || powerState.source === 'unsupported' || powerState.source === 'error') {
    return '電池資訊不可用';
  }
  const levelText = formatPercent(powerState.level);
  if (powerState.charging) {
    return `${levelText}（充電中）`;
  }
  if (powerState.lowPower) {
    return `${levelText}（建議保護）`;
  }
  return levelText;
}

function buildAdaptiveProfile(profile, powerState) {
  let factor = 1;
  if (typeof document !== 'undefined' && document.visibilityState === 'hidden') {
    factor += SOLVER_GUARD.visibilityHiddenPenalty;
  }
  if (powerState?.source === 'unsupported' || powerState?.source === 'error') {
    factor += 0.25;
  }
  if (powerState?.lowPower) {
    factor += 0.55;
  }
  if (typeof powerState?.level === 'number' && powerState.level <= SOLVER_GUARD.warningLevel && !powerState.charging) {
    factor += 0.35;
  }
  if (factor <= 1) return profile;
  if (factor > SOLVER_GUARD.maxFactor) factor = SOLVER_GUARD.maxFactor;
  return {
    ...profile,
    label: `${profile.label}（保護模式）`,
    yieldEveryMs: Math.max(16, Math.round(profile.yieldEveryMs * factor)),
    sleepMs: Math.max(profile.sleepMs, Math.round(profile.sleepMs * factor + 1)),
    heartbeatEveryMs: Math.max(300, Math.round(profile.heartbeatEveryMs * factor))
  };
}

function mapFactorToThrottleLevel(factor) {
  if (factor < 1.35) return THROTTLE_LEVELS[0];
  if (factor < 1.8) return THROTTLE_LEVELS[1];
  if (factor < 2.2) return THROTTLE_LEVELS[2];
  return THROTTLE_LEVELS[3];
}

function buildRuntimeAdaptiveProfile(baseProfile, runtimeMs, powerState) {
  let factor = 1;
  const elapsedMin = runtimeMs / 60000;
  let guardReason = '節奏保護：正常';
  let forceStop = false;
  let stopReason = '';

  if (elapsedMin >= 30) factor += 0.7;
  else if (elapsedMin >= 15) factor += 0.5;
  else if (elapsedMin >= 8) factor += 0.25;
  else if (elapsedMin >= 3) factor += 0.1;

  if (typeof document !== 'undefined' && document.visibilityState === 'hidden') {
    factor += SOLVER_GUARD.visibilityHiddenPenalty;
    guardReason = '背景降頻';
  }
  if (powerState?.lowPower) {
    factor += 0.55;
    if (guardReason === '節奏保護：正常') guardReason = '低電量';
  }
  if (typeof powerState?.level === 'number' && powerState.level <= SOLVER_GUARD.warningLevel && !powerState.charging) {
    factor += 0.35;
    if (guardReason === '節奏保護：正常') guardReason = '電量保護';
  }
  if (typeof powerState?.level === 'number' && powerState.level <= SOLVER_GUARD.heavyThrottleLevel && !powerState.charging) {
    factor += 0.55;
    guardReason = '電量很低，已進入保守節流';
  }
  if (typeof powerState?.level === 'number' && powerState.level <= SOLVER_GUARD.stopLevel && !powerState.charging) {
    forceStop = true;
    stopReason = '低電量保護已停止：避免關機/資料中斷';
  }

  if (powerState?.source === 'unsupported' || powerState?.source === 'error') {
    factor += 0.25;
    if (elapsedMin >= SOLVER_GUARD.unsupportedStopMinutes) {
      forceStop = true;
      stopReason = '無法監測電量，超過安全時限自動停止';
    }
    if (elapsedMin >= 180) {
      factor += 0.25;
      guardReason = '電池監測不可用，已加大節流';
    } else {
      guardReason = '節奏保護：電池監測不可用';
    }
  }

  if (forceStop && !stopReason) {
    stopReason = '系統保護已停止';
  }

  if (factor > SOLVER_GUARD.maxFactor) factor = SOLVER_GUARD.maxFactor;

  const selectedLevel = mapFactorToThrottleLevel(factor);
  return {
    ...baseProfile,
    label: `${baseProfile.label}（${selectedLevel.factor.toFixed(2)}x）`,
    yieldEveryMs: Math.max(16, Math.round(baseProfile.yieldEveryMs * factor)),
    sleepMs: Math.max(baseProfile.sleepMs, Math.round(baseProfile.sleepMs * factor + 1)),
    heartbeatEveryMs: Math.max(300, Math.round(baseProfile.heartbeatEveryMs * factor)),
    throttleLevel: selectedLevel.level,
    throttleFactor: selectedLevel.factor,
    guardReason,
    forceStop,
    stopReason
  };
}

function renderSolveHint(profile) {
  if (!els?.solveModeHint) return;
  if (!profile) profile = getSolveProfile();
  const defaultMin = profile.maxRunMinutesDefault;
  const modeText = profile.label || '夜間節能';
  const sleepText = `${profile.yieldEveryMs}ms 搜尋節奏`;
  els.solveModeHint.textContent = `${modeText}：每 ${sleepText} 會讓出 CPU，並以 ~${profile.sleepMs}ms 休眠，預設最長 ${defaultMin} 分鐘；關掉上限可輸入 0。` +
    '長時求解會每 12 秒重估電量與背景狀態，低電量時自動提高節流；低於 5% 且未充電會安全停止。';
}

function renderGroupPanel() {
  const group = getActiveGroup();
  const select = els.groupSelect;

  select.innerHTML = '';
  for (const g of state.groups) {
    const opt = document.createElement('option');
    opt.value = g.id;
    opt.textContent = g.name;
    if (state.activeGroupId === g.id) opt.selected = true;
    select.appendChild(opt);
  }
  if (!state.groups.length) {
    els.memberPanel.innerHTML = '<p class="hint">尚未建立群組</p>';
  }
}

function renderMembers() {
  const group = getActiveGroup();
  if (!group) {
    els.memberList.innerHTML = '';
    return;
  }
  els.memberList.innerHTML = '';
  for (const m of group.members) {
    const card = document.createElement('div');
    card.className = 'member-card';
    card.innerHTML = `
      <div class="member-head">
        <div><strong>${m.name}</strong> <span class="small">每期上限: ${m.maxShiftsPerPeriod}</span></div>
        <div class="member-actions">
          <button type="button" class="ghost" data-action="remove-member" data-id="${m.id}">移除</button>
        </div>
      </div>
      <div class="small">可在「需求回報」逐筆調整可用性與偏好</div>
    `;
    els.memberList.appendChild(card);
  }
}

function renderShiftInputs() {
  const group = getActiveGroup();
  els.periodShifts.innerHTML = '';
  if (!group) {
    return;
  }
  for (const shift of group.shifts) {
    const row = document.createElement('div');
    row.className = 'member-row';
    row.innerHTML = `
      <div><strong>${shift.label}</strong> (${shift.start}~${shift.end})</div>
      <div class="form-line">
        <label>每人天必需人力</label>
        <input data-shift-id="${shift.id}" class="shift-required" type="number" min="0" max="20" value="${shift.required}" />
      </div>
    `;
    els.periodShifts.appendChild(row);
  }
}

function renderPeriods() {
  const group = getActiveGroup();
  els.periodList.innerHTML = '';
  if (!group) {
    return;
  }
  if (!group.periods.length) {
    els.periodList.innerHTML = '<p class="hint">尚未建立期次</p>';
    els.inputPanelEmpty.classList.remove('hidden');
    els.inputPanelBody.classList.add('hidden');
    els.solvePanelEmpty.classList.remove('hidden');
    els.solvePanelBody.classList.add('hidden');
    return;
  }

  for (const p of group.periods) {
    const card = document.createElement('div');
    card.className = 'period-card';
    const isActive = state.activePeriodId === p.id;
    card.innerHTML = `
      <div class="period-head">
        <div>
          <strong>${p.name}</strong>
          <span class="small">(${p.startDate}，${p.dayCount}天)</span>
          <span class="small"> | 每人每日最多: ${p.maxShiftsPerDay}班</span>
        </div>
        <div class="period-actions">
          <button type="button" class="ghost" data-action="select-period" data-id="${p.id}">${isActive ? '編輯中' : '選此期次'}</button>
          <button type="button" class="ghost" data-action="delete-period" data-id="${p.id}">刪除</button>
        </div>
      </div>
      <div class="small">狀態：${p.status}，候選：${p.candidates.length}組${p.selectedCandidateId ? '，已選：' + p.selectedCandidateId.slice(0, 6) : ''}</div>
      <div class="small">班別需求：${p.shifts.map((s) => `${s.label}${s.required}人`).join(' / ')}</div>
    `;
    if (isActive) card.style.background = 'var(--accent-soft)';
    els.periodList.appendChild(card);
  }
  els.inputPanelEmpty.classList.add('hidden');
  els.inputPanelBody.classList.remove('hidden');
}

function getPeriodShiftRequirementInputs() {
  const group = getActiveGroup();
  if (!group) return [];
  const periodShifts = [];
  for (const row of els.periodShifts.querySelectorAll('.member-row')) {
    const input = row.querySelector('.shift-required');
    const shiftId = row.querySelector('input[data-shift-id]')?.dataset.shiftId;
    const shift = group.shifts.find((s) => s.id === shiftId);
    if (!shift) continue;
    periodShifts.push({
      id: shift.id,
      label: shift.label,
      start: shift.start,
      end: shift.end,
      required: Number(input.value || 0)
    });
  }
  if (!periodShifts.length) {
    return group ? JSON.parse(JSON.stringify(group.shifts)) : [];
  }
  return periodShifts;
}

function renderInputPanel() {
  const group = getActiveGroup();
  const period = getActivePeriod(group);
  if (!group || !period) {
    els.inputPanelEmpty.classList.remove('hidden');
    els.inputPanelBody.classList.add('hidden');
    return;
  }
  if (!group.members.length) {
    els.inputPanelBody.innerHTML = '<p class="hint">先新增員工，再開始需求回報。</p>';
    els.inputPanelEmpty.classList.add('hidden');
    els.inputPanelBody.classList.remove('hidden');
    return;
  }
  els.inputPanelBody.classList.remove('hidden');
  els.inputPanelEmpty.classList.add('hidden');
  const title = document.createElement('h3');
  title.textContent = `期次：${period.name}`;

  const desc = document.createElement('p');
  desc.className = 'small';
  desc.textContent = `起始 ${period.startDate}，${period.dayCount} 天。`;

  const wrapper = document.createElement('div');
  wrapper.appendChild(title);
  wrapper.appendChild(desc);

  for (const m of group.members) {
    if (!period.memberInputs[m.id]) {
      period.memberInputs[m.id] = newInputTemplate(period.dayCount, period.shifts);
    }

    const input = period.memberInputs[m.id];
    const card = document.createElement('div');
    card.className = 'member-input-card';
    card.dataset.memberCard = m.id;
    card.innerHTML = `
      <div class="member-head">
        <div><strong>${m.name}</strong></div>
        <button type="button" class="ghost" data-action="save-member-input" data-member="${m.id}">儲存此人</button>
      </div>
      <div class="small">勾選「可排」表示可上班；偏好：偏好>中性>避免</div>
    `;

    const table = document.createElement('table');
    table.className = 'member-input-table';
    const header = document.createElement('tr');
    const h1 = document.createElement('th');
    h1.textContent = '日期';
    header.appendChild(h1);
    for (const s of period.shifts) {
      const th = document.createElement('th');
      th.textContent = `${s.label} (${s.start}-${s.end})`;
      header.appendChild(th);
    }
    table.appendChild(document.createElement('thead')).appendChild(header);

    const body = document.createElement('tbody');
    for (let day = 0; day < period.dayCount; day++) {
      const tr = document.createElement('tr');
      const dayCell = document.createElement('td');
      dayCell.textContent = dayLabel(period.startDate, day);
      tr.appendChild(dayCell);
      for (const s of period.shifts) {
        const td = document.createElement('td');
        td.dataset.day = String(day);
        td.dataset.shift = s.id;
        const avail = input.availability?.[day]?.[s.id] ?? true;
        const pref = input.preference?.[day]?.[s.id] ?? 'neutral';
        const line1 = document.createElement('label');
        line1.className = 'small';
        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.className = 'avail';
        cb.checked = avail;
        const cbText = document.createElement('span');
        cbText.textContent = '可排';
        line1.appendChild(cb);
        line1.appendChild(cbText);

        const prefSelect = document.createElement('select');
        prefSelect.className = 'pref';
        const options = [
          ['prefer', '偏好'],
          ['neutral', '中性'],
          ['avoid', '避免']
        ];
        for (const [val, title] of options) {
          const o = document.createElement('option');
          o.value = val;
          o.textContent = title;
          if (pref === val) o.selected = true;
          prefSelect.appendChild(o);
        }
        td.appendChild(line1);
        td.appendChild(prefSelect);
        tr.appendChild(td);
      }
      body.appendChild(tr);
    }
    table.appendChild(body);
    card.appendChild(table);

    const noteWrap = document.createElement('textarea');
    noteWrap.className = 'member-notes';
    noteWrap.placeholder = '自然語言補充（例：週三休息；週五偏好早班）';
    noteWrap.value = input.notes || '';
    noteWrap.dataset.notes = m.id;
    card.appendChild(noteWrap);
    wrapper.appendChild(card);
  }

  const applyAllBtn = document.createElement('button');
  applyAllBtn.type = 'button';
  applyAllBtn.dataset.action = 'save-all-inputs';
  applyAllBtn.textContent = '儲存全部需求回報';
  wrapper.appendChild(applyAllBtn);
  els.inputPanelBody.innerHTML = '';
  els.inputPanelBody.appendChild(wrapper);
}

function buildMemberInput(period, memberId) {
  return period.memberInputs[memberId] || newInputTemplate(period.dayCount, period.shifts);
}

function analyzeSolvability(group, period, selectedMembers) {
  const issues = [];
  if (!group || !period) return { ok: false, issues: ['缺少群組或期次資料'] };
  if (!Array.isArray(group.members) || group.members.length === 0) {
    return { ok: false, issues: ['目前沒有員工，無法排班'] };
  }
  if (!Array.isArray(period.shifts) || !period.shifts.length) {
    return { ok: false, issues: ['期次未設定班別需求'] };
  }

  const members = selectedMembers || group.members;
  const demandByDayShift = [];
  for (let day = 0; day < period.dayCount; day++) {
    for (const shift of period.shifts) {
      const required = Number(shift.required || 0);
      if (required > 0) {
        demandByDayShift.push({ day, shiftId: shift.id, required });
      }
    }
  }
  if (!demandByDayShift.length) {
    return { ok: false, issues: ['需求全為 0，請先設定每班需求人數'] };
  }

  const memberInputs = {};
  for (const m of members) {
    memberInputs[m.id] = buildMemberInput(period, m.id);
  }

  for (const item of demandByDayShift) {
    const shift = period.shifts.find((s) => s.id === item.shiftId);
    const label = shift ? shift.label : item.shiftId;
    const availableCount = members.filter((m) => memberInputs[m.id]?.availability?.[item.day]?.[item.shiftId] !== false).length;
    if (availableCount < item.required) {
      issues.push(`${dayLabel(period.startDate, item.day)} ${label} 需要 ${item.required} 人，但僅有 ${availableCount} 人可排`);
    }
  }

  const dailyNeed = {};
  for (const item of demandByDayShift) {
    dailyNeed[item.day] = (dailyNeed[item.day] || 0) + item.required;
  }
  const dailyCap = members.length * period.maxShiftsPerDay;
  for (const [day, need] of Object.entries(dailyNeed)) {
    if (need > dailyCap) {
      issues.push(`${dayLabel(period.startDate, Number(day))} 總人力需求 ${need} 人次，超過每天上限 ${dailyCap} 人次`);
    }
  }

  const totalDemand = demandByDayShift.reduce((acc, item) => acc + item.required, 0);
  const periodCap = members.reduce((acc, m) => acc + Math.max(0, Number(m.maxShiftsPerPeriod || 0)), 0);
  if (totalDemand > periodCap) {
    issues.push(`此期次總需求 ${totalDemand} 人次，但員工每期上限合計僅 ${periodCap}`);
  }

  return { ok: issues.length === 0, issues };
}

function parseFreeTextToInput(rawText, inputTemplate, period) {
  if (!rawText) return inputTemplate;
  const lines = String(rawText).split('\n');
  const out = JSON.parse(JSON.stringify(inputTemplate));

  for (const raw of lines) {
    const line = raw.trim();
    if (!line) continue;
    const lower = line.toLowerCase();
    const days = [];
    for (const [k, idx] of Object.entries(DAY_ALIASES)) {
      if (lower.includes(k)) days.push(idx);
    }

    const shiftTag = period.shifts.find((s) =>
      lower.includes(s.label.toLowerCase()) || lower.includes(s.id)
    );

    if (!days.length) continue;
    const uniqDays = [...new Set(days)];
    for (const d of uniqDays) {
      if (d >= period.dayCount) continue;
      if (shiftTag) {
        if (/(off|休|請假|不想|avoid)/.test(lower)) {
          out.availability[d][shiftTag.id] = false;
          if (/(prefer|偏好|想排|prefer)/.test(lower)) {
            out.preference[d][shiftTag.id] = 'prefer';
          }
        } else if (/(prefer|偏好|prefer|想要)/.test(lower)) {
          out.preference[d][shiftTag.id] = 'prefer';
        }
      } else {
        if (/(off|休|請假|不想|avoid)/.test(lower) || /(\ball\b|全部|全日|都不排)/.test(lower)) {
          for (const s of period.shifts) out.availability[d][s.id] = false;
        }
      }
    }
  }
  return out;
}

function collectMemberInputFromCard(group, period, memberId) {
  const card = els.inputPanelBody.querySelector(`[data-member-card="${memberId}"]`);
  if (!card) return null;
  const notes = card.querySelector(`[data-notes="${memberId}"]`)?.value || '';
  const input = newInputTemplate(period.dayCount, period.shifts);
  for (const td of card.querySelectorAll('td[data-day][data-shift]')) {
    const day = Number(td.dataset.day);
    const sid = td.dataset.shift;
    const avail = td.querySelector('.avail')?.checked ?? true;
    const pref = td.querySelector('.pref')?.value || 'neutral';
    input.availability[day][sid] = avail;
    input.preference[day][sid] = pref;
  }
  input.notes = notes;
  return parseFreeTextToInput(notes, input, period);
}

function saveAllInputs() {
  const group = getActiveGroup();
  const period = getActivePeriod(group);
  if (!group || !period) return;
  for (const m of group.members) {
    const data = collectMemberInputFromCard(group, period, m.id);
    if (data) period.memberInputs[m.id] = data;
  }
  period.status = 'collected';
  saveState();
  render();
  setStatus('需求已儲存，已可進行排班。', 'ok');
}

function memberInputExistsForPeriod(group, period) {
  if (!group || !period) return false;
  return group.members.every((m) => Object.prototype.hasOwnProperty.call(period.memberInputs, m.id));
}

function renderSolver() {
  const group = getActiveGroup();
  const period = getActivePeriod(group);
  if (!group || !period) {
    els.solvePanelEmpty.classList.remove('hidden');
    els.solvePanelBody.classList.add('hidden');
    if (els.solveConflicts) els.solveConflicts.textContent = '';
    return;
  }
  if (!group.members.length || !memberInputExistsForPeriod(group, period)) {
    els.solvePanelEmpty.textContent = '先完成需求回報後可排班。';
    els.solvePanelEmpty.classList.remove('hidden');
    els.solvePanelBody.classList.add('hidden');
    if (els.solveConflicts) els.solveConflicts.textContent = '';
    return;
  }
  els.solvePanelEmpty.classList.add('hidden');
  els.solvePanelBody.classList.remove('hidden');

  if (els.solveConflicts) {
    const checks = analyzeSolvability(group, period);
    if (!checks.ok) {
      els.solveConflicts.textContent = checks.issues.join('\n');
    } else {
      els.solveConflicts.textContent = '';
    }
  }

  els.solveStats.textContent = `候選：${period.candidates.length}組`;
  els.candidateList.innerHTML = '';
  if (!period.candidates.length) {
    const p = document.createElement('p');
    p.className = 'hint';
    p.textContent = '尚未產生候選班表，可直接按「開始排班」。';
    els.candidateList.appendChild(p);
    return;
  }

  for (const c of period.candidates) {
    const card = document.createElement('div');
    card.className = 'candidate';
    const metric = document.createElement('div');
    metric.className = 'score-line';
    metric.textContent = `分數 ${c.score.toFixed(2)}｜公平性 ${c.fairness.toFixed(2)}｜偏好 ${c.preferenceScore.toFixed(1)}｜已填 ${c.filledSlots}/${c.requiredSlots}`;
    const isSelected = period.selectedCandidateId === c.id;

    const head = document.createElement('div');
    head.className = 'candidate-head';
    head.innerHTML = `
      <div>
        <strong>候選 ${c.label}</strong> ${isSelected ? '<span class="ok">(目前選定)</span>' : ''}
        <div class="small">${metric.textContent}</div>
      </div>
      <div>
        <button type="button" class="ghost" data-action="pick-candidate" data-candidate="${c.id}">選為本期班表</button>
      </div>
    `;
    const table = document.createElement('table');
    table.className = 'candidate-grid';
    const thead = document.createElement('thead');
    const trh = document.createElement('tr');
    const dayTh = document.createElement('th');
    dayTh.textContent = '日期';
    trh.appendChild(dayTh);
    for (const s of period.shifts) {
      const th = document.createElement('th');
      th.textContent = `${s.label} (需求 ${s.required})`;
      trh.appendChild(th);
    }
    thead.appendChild(trh);
    table.appendChild(thead);
    const tb = document.createElement('tbody');
    for (let day = 0; day < period.dayCount; day++) {
      const tr = document.createElement('tr');
      const dayTd = document.createElement('td');
      dayTd.textContent = dayLabel(period.startDate, day);
      tr.appendChild(dayTd);
      for (const s of period.shifts) {
        const key = `${day}-${s.id}`;
        const members = c.assignments[key] || [];
        const slotTd = document.createElement('td');
        slotTd.className = 'slot-cell';
        const tagWrap = document.createElement('div');
        if (!members.length) {
          slotTd.classList.add('warn');
          slotTd.textContent = '未指派';
        } else {
          tagWrap.innerHTML = members.map((mId) => {
            const m = group.members.find((mm) => mm.id === mId);
            return `<span class="tag">${m ? m.name : mId.slice(0, 6)}</span>`;
          }).join(' ');
          slotTd.appendChild(tagWrap);
        }
        const editBtn = document.createElement('button');
        editBtn.type = 'button';
        editBtn.className = 'ghost';
        editBtn.dataset.action = 'edit-slot';
        editBtn.dataset.candidate = c.id;
        editBtn.dataset.slot = key;
        editBtn.textContent = '手動編輯';
        slotTd.appendChild(document.createElement('br'));
        slotTd.appendChild(editBtn);
        tr.appendChild(slotTd);
      }
      tb.appendChild(tr);
    }
    table.appendChild(tb);
    card.appendChild(head);
    card.appendChild(table);
    els.candidateList.appendChild(card);
  }

  if (period.selectedCandidateId) {
    const selected = period.candidates.find((item) => item.id === period.selectedCandidateId);
    if (selected) {
      const summary = document.createElement('div');
      summary.className = 'solve-banner';
      summary.innerHTML = `<span class="label">目前選定：</span>${selected.label}｜分數 ${selected.score.toFixed(2)}｜公平性 ${selected.fairness.toFixed(2)}｜偏好 ${selected.preferenceScore.toFixed(1)}｜已填 ${selected.filledSlots}/${selected.requiredSlots}`;
      if (period.solverStats) {
        summary.innerHTML += `｜耗時 ${period.solverStats.spentMs}ms｜搜尋 ${period.solverStats.nodes}節點`;
        if (period.solverStats.profile) {
          summary.innerHTML += `｜模式 ${period.solverStats.profile}`;
        }
        if (period.solverStats.throttleLevel !== undefined) {
          summary.innerHTML += `｜熱控等級 ${period.solverStats.throttleLevel}`;
        }
        if (period.solverStats.throttleFactor !== undefined) {
          summary.innerHTML += `｜節流倍率 ${period.solverStats.throttleFactor.toFixed(2)}x`;
        }
        if (period.solverStats.guardReason) {
          summary.innerHTML += `｜節流狀態 ${period.solverStats.guardReason}`;
        }
        if (period.solverStats.powerLevelPercent !== null && period.solverStats.powerLevelPercent !== undefined) {
          const chargingTag = period.solverStats.powerCharging ? '充電中' : '未充電';
          summary.innerHTML += `｜電量 ${period.solverStats.powerLevelPercent}%（${chargingTag}）`;
        }
        if (period.solverStats.stopReason && period.solverStats.timedOut) {
          summary.innerHTML += `｜停止原因：${period.solverStats.stopReason}`;
        }
        if (period.solverStats.timedOut) {
          if (period.solverStats.stopPolicy === '安全保護') {
            summary.innerHTML += '｜安全保護中止（未完成全量搜尋）';
          } else {
            summary.innerHTML += '｜到達時間上限（未完成全量搜尋）';
          }
        }
      }
      els.candidateList.appendChild(summary);
    }
  }
}

function getMemberById(group, memberId) {
  return group.members.find((m) => m.id === memberId) || null;
}

function buildSolverSlots(period, group) {
  const slots = [];
  for (let day = 0; day < period.dayCount; day++) {
    for (const s of period.shifts) {
      const req = Number(s.required || 0);
      for (let i = 0; i < req; i++) {
        slots.push({
          day,
          shiftId: s.id,
          key: `${day}-${s.id}`
        });
      }
    }
  }
  return slots;
}

function canAssignWorker(memberId, day, shiftId, period, periodInputs, stateObj, member) {
  const input = periodInputs[memberId];
  if (!input?.availability?.[day]?.[shiftId]) return false;
  if (stateObj.total[memberId] >= member.maxShiftsPerPeriod) return false;
  if (stateObj.byDay[memberId][day] >= period.maxShiftsPerDay) return false;
  return true;
}

function preferenceScoreForDaySlot(input, day, shiftId) {
  const pref = input?.preference?.[day]?.[shiftId] || 'neutral';
  if (pref === 'prefer') return -1;
  if (pref === 'avoid') return 1;
  return 0;
}

function evaluateCandidate(candidate, period, group) {
  const scoreById = {};
  const penalty = { preference: 0 };
  let filled = 0;
  for (const m of group.members) scoreById[m.id] = 0;

  for (let day = 0; day < period.dayCount; day++) {
    for (const s of period.shifts) {
      const key = `${day}-${s.id}`;
      const list = candidate.assignments[key] || [];
      filled += list.length;
      for (const wid of list) {
        scoreById[wid] = (scoreById[wid] || 0) + 1;
        const input = period.memberInputs[wid] || {};
        penalty.preference += preferenceScoreForDaySlot(input, day, s.id, wid);
      }
    }
  }

  const values = Object.values(scoreById);
  const avg = values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0;
  let fairness = 0;
  for (const v of values) {
    fairness += Math.abs(v - avg);
  }
  const fairnessScore = values.length ? fairness : 0;
  const base = Number((fairnessScore * 1.5 + penalty.preference * 2).toFixed(4));
  const totalSlots = period.shifts.reduce((acc, s) => acc + s.required, 0) * period.dayCount;
  return {
    score: base,
    fairness: fairnessScore,
    preferenceScore: penalty.preference,
    filledSlots: filled,
    requiredSlots: totalSlots
  };
}

async function solvePeriod(group, period, maxCandidates, token, options = {}) {
  const profile = options.profile || SOLVE_PROFILES.night;
  const powerState = options.powerState || {};
  const resumeState = sanitizeSolverResumeState(options.resumeState);
  const onCheckpoint = typeof options.onCheckpoint === 'function' ? options.onCheckpoint : null;
  const seedCandidates = normalizeCandidatesForResume(options.seedCandidates || []);
  const budgetInputMs = Number(options.maxMs || 0);
  const maxMs = normalizeSolverBudgetMs(budgetInputMs);
  const slots = buildSolverSlots(period, group);
  const members = group.members.slice();
  const totalGroups = (() => {
    const previewGroups = buildSolverGroups(period, group);
    return previewGroups.length;
  })();
  if (!members.length || !slots.length || !totalGroups) return { candidates: seedCandidates, nodes: 0, spentMs: 0, timedOut: false, aborted: false };

  const runtime = {
    startTs: performance.now() - Math.min(Number.isFinite(resumeState?.elapsedMs) && resumeState.elapsedMs > 0 ? resumeState.elapsedMs : 0, Number.isFinite(maxMs) ? maxMs : Number.POSITIVE_INFINITY),
    runningStartedAt: Number.isFinite(resumeState?.startedAt) ? resumeState.startedAt : Date.now(),
    lastYieldTs: performance.now(),
    lastHeartbeatTs: performance.now() + profile.heartbeatEveryMs,
    timedOut: false,
    timedOutTs: null,
    timedOutReason: '',
    aborted: false,
    maxMs,
    powerState,
    powerStateTs: performance.now(),
    nextPowerPollTs: performance.now() + 120,
    throttleLevel: 0,
    throttleFactor: 1,
    guardReason: '節奏保護：正常',
    maxGuardReason: '節奏保護：正常',
    stopReason: '',
    stopPolicy: '節奏保護',
    profileLabel: profile.label || '夜間節能（長時）',
    groupsCount: totalGroups,
    lastCheckpointTs: 0,
    lastFrameCount: 0,
    resumeCandidateSnapshot: 0
  };

  const shouldAbort = () => !solveRunning || solveToken !== token;

  const refreshPowerState = async () => {
    const now = performance.now();
    if (!shouldRefreshPowerStateNow && now < runtime.nextPowerPollTs) {
      return runtime.powerState;
    }
    shouldRefreshPowerStateNow = false;
    runtime.nextPowerPollTs = now + SOLVER_GUARD.batteryRefreshMs;
    const latest = await detectPowerState();
    if (latest) {
      runtime.powerState = latest;
      runtime.powerStateTs = now;
    }
    return runtime.powerState;
  };

  const currentProfile = () => buildRuntimeAdaptiveProfile(profile, performance.now() - runtime.startTs, runtime.powerState);

  const periodInputs = {};
  for (const m of members) {
    periodInputs[m.id] = buildMemberInput(period, m.id);
  }

  const slotsSorted = slots
    .map((slot) => {
      const candidates = members.filter((m) => canAssignWorker(m.id, slot.day, slot.shiftId, period, periodInputs, {
        total: {},
        byDay: members.reduce((acc, mm) => {
          acc[mm.id] = Array(period.dayCount).fill(0);
          return acc;
        }, {})
      }, m => m));
      return {
        ...slot,
        candidateCount: candidates.length
      };
    })
    .filter((s) => s.candidateCount > 0)
    .sort((a, b) => {
      if (a.candidateCount !== b.candidateCount) return a.candidateCount - b.candidateCount;
      if (a.day !== b.day) return a.day - b.day;
      return a.shiftId.localeCompare(b.shiftId);
    });

  const groups = [];
  let i = 0;
  while (i < slotsSorted.length) {
    const first = slotsSorted[i];
    const same = [];
    while (i < slotsSorted.length && slotsSorted[i].day === first.day && slotsSorted[i].shiftId === first.shiftId) {
      same.push(slotsSorted[i]);
      i++;
    }
    groups.push({
      day: first.day,
      shiftId: first.shiftId,
      required: same.length,
      key: first.key
    });
  }

  if (!groups.length) {
    return {
      candidates: seedCandidates,
      nodes: 0,
      spentMs: 0,
      timedOut: false,
      aborted: false,
      throttleLevel: 0,
      throttleFactor: 1,
      guardReason: runtime.maxGuardReason,
      stopPolicy: runtime.stopPolicy,
      stopReason: '',
      solverElapsedMin: 0,
      powerState: runtime.powerState,
      maxMs: runtime.maxMs
    };
  }

  const result = [];
  const seen = new Set();
  const normalizeInitialCandidates = (seed) => {
    const list = [];
    const seenSig = new Set();
    for (const c of seed || []) {
      if (!c || !c.assignments) continue;
      const normalized = {
        ...c,
        assignments: JSON.parse(JSON.stringify(c.assignments))
      };
      if (typeof normalized.score !== 'number') {
        const fallback = evaluateCandidate(normalized, period, group);
        normalized.score = fallback.score;
        normalized.fairness = fallback.fairness;
        normalized.preferenceScore = fallback.preferenceScore;
        normalized.filledSlots = fallback.filledSlots;
        normalized.requiredSlots = fallback.requiredSlots;
      }
      const signature = candidateSignature(normalized);
      if (seenSig.has(signature)) continue;
      seenSig.add(signature);
      list.push(normalized);
    }
    list.sort((a, b) => (a.score || 0) - (b.score || 0));
    while (list.length > maxCandidates) list.pop();
    return list;
  };

  const restoreResult = normalizeInitialCandidates(seedCandidates);
  for (const item of restoreResult) {
    result.push(item);
    seen.add(candidateSignature(item));
  }

  const stateNow = {
    total: {},
    byDay: {}
  };

  for (const m of members) {
    stateNow.total[m.id] = 0;
    stateNow.byDay[m.id] = Array(period.dayCount).fill(0);
  }

  if (resumeState?.stateNow) {
    const rs = cloneResumeStateNow(resumeState.stateNow);
    if (rs) {
      for (const m of members) {
        const t = rs.total?.[m.id];
        stateNow.total[m.id] = Number.isFinite(t) ? Math.max(0, Math.floor(t)) : 0;
        const rawDay = rs.byDay?.[m.id];
        const row = Array(period.dayCount).fill(0);
        for (let d = 0; d < period.dayCount; d++) {
          const v = rawDay?.[d];
          row[d] = Number.isFinite(v) ? Math.max(0, Math.floor(v)) : 0;
        }
        stateNow.byDay[m.id] = row;
      }
    }
  }

  const assignment = {};
  for (const g of groups) {
    assignment[g.key] = [];
  }

  if (resumeState?.assignment) {
    for (const [key, values] of Object.entries(resumeState.assignment)) {
      if (Object.prototype.hasOwnProperty.call(assignment, key) && Array.isArray(values)) {
        assignment[key] = values.slice(0, 24);
      }
    }
  }

  const visited = { count: Number.isFinite(resumeState?.totalNodes) ? Math.max(0, Math.floor(resumeState.totalNodes)) : 0 };

  const combinationToIds = (candidateIds, indices) => {
    if (!Array.isArray(indices) || !Array.isArray(candidateIds)) return [];
    const selected = [];
    for (const idx of indices) {
      const id = candidateIds[idx];
      if (typeof id === 'undefined') return [];
      selected.push(id);
    }
    return selected;
  };

  const initialCombination = (count, required) => {
    if (required <= 0) return [];
    if (count < required) return null;
    const idx = [];
    for (let i = 0; i < required; i++) {
      idx.push(i);
    }
    return idx;
  };

  const nextCombination = (indices, total, required) => {
    if (!Array.isArray(indices) || required === 0) return null;
    for (let pos = required - 1; pos >= 0; pos--) {
      if (indices[pos] < total - required + pos) {
        const next = indices.slice();
        next[pos] += 1;
        for (let j = pos + 1; j < required; j++) {
          next[j] = next[j - 1] + 1;
        }
        return next;
      }
    }
    return null;
  };

  const buildCandidateIdsForGroup = (g, stateObj) =>
    members
      .filter((m) => canAssignWorker(m.id, g.day, g.shiftId, period, periodInputs, stateObj, m))
      .sort((a, b) => {
        const da = preferenceScoreForDaySlot(periodInputs[a.id], g.day, g.shiftId, a.id);
        const db = preferenceScoreForDaySlot(periodInputs[b.id], g.day, g.shiftId, b.id);
        return da - db;
      })
      .map((m) => m.id);

  const isValidIndices = (indices, total, required) => {
    if (required === 0) return true;
    if (!Array.isArray(indices) || indices.length !== required) return false;
    for (let i = 0; i < indices.length; i++) {
      const idx = indices[i];
      if (!Number.isInteger(idx) || idx < 0 || idx >= total) return false;
      if (i > 0 && idx <= indices[i - 1]) return false;
    }
    return true;
  };

  const normalizeResumeFrameIndices = (frame, required, candidateIds) => {
    const total = candidateIds.length;
    if (required === 0) return [];
    if (isValidIndices(frame?.indices, total, required)) {
      return frame.indices.slice();
    }
    if (frame?.indices && Array.isArray(frame.indices) && frame.indices.some((v) => v > total + 1)) {
      return null;
    }
    return initialCombination(total, required);
  };

  const apply = (frame) => {
    const ids = combinationToIds(frame.candidateIds, frame.indices);
    if (ids.length !== frame.required) return false;
    for (const wid of ids) {
      if (!stateNow.byDay[wid]) return false;
      stateNow.total[wid] += 1;
      if (stateNow.byDay[wid][frame.group.day] !== undefined) {
        stateNow.byDay[wid][frame.group.day] += 1;
      }
    }
    assignment[frame.group.key] = ids.slice();
    return true;
  };

  const rollback = (frame) => {
    const ids = combinationToIds(frame.candidateIds, frame.indices);
    for (const wid of ids) {
      if (!stateNow.byDay[wid]) continue;
      stateNow.total[wid] -= 1;
      if (stateNow.byDay[wid][frame.group.day] !== undefined) {
        stateNow.byDay[wid][frame.group.day] -= 1;
      }
      if (stateNow.total[wid] < 0) stateNow.total[wid] = 0;
      if (stateNow.byDay[wid][frame.group.day] < 0) stateNow.byDay[wid][frame.group.day] = 0;
    }
    assignment[frame.group.key] = [];
  };

  const registerCandidate = () => {
    if (shouldAbort()) {
      runtime.aborted = true;
      return;
    }
    const scored = evaluateCandidate(assignment, period, group);
    const signature = groups
      .map((g) => `${g.key}:${(assignment[g.key] || []).slice().sort().join('|')}`)
      .join('||');
    if (seen.has(signature)) return;
    seen.add(signature);
    const candidate = {
      id: uid(),
      label: `${Date.now()}`,
      assignments: JSON.parse(JSON.stringify(assignment)),
      ...scored
    };
    result.push(candidate);
    result.sort((a, b) => (a.score || 0) - (b.score || 0));
    while (result.length > maxCandidates) result.pop();
  };

  const serializeFrames = (frames) => frames.map((f) => ({
    gIndex: f.gIndex,
    required: f.required,
    candidateIds: f.candidateIds,
    indices: f.indices,
    applied: !!f.applied
  }));

  const persistCheckpoint = async () => {
    if (!onCheckpoint) return;
    runtime.lastCheckpointTs = performance.now();
    runtime.lastFrameCount = Math.min(SOLVER_RESUME.maxStackFrames, frames.length);
    const framesToStore = serializeFrames(frames.slice(-runtime.lastFrameCount));

    onCheckpoint({
      status: 'paused',
      startedAt: runtime.runningStartedAt,
      lastUpdated: Date.now(),
      elapsedMs: Math.max(0, Math.round(performance.now() - runtime.startTs)),
      totalNodes: visited.count,
      totalSegments: frames.length,
      fingerprint: makeSolverFingerprint(group, period, profile, maxCandidates),
      profile: profile.label || 'night',
      maxCandidates,
      maxMs: runtime.maxMs,
      stack: framesToStore,
      stateNow: cloneSolverStateForCheckpoint(stateNow),
      assignment: cloneSolverAssignments(assignment),
      totalGroups: groups.length,
      candidates: result.map((c) => ({
        ...c,
        assignments: JSON.parse(JSON.stringify(c.assignments))
      })).slice(0, SOLVER_RESUME.maxSavedCandidates)
    });
  };

  const updateRunningStatus = () => {
    const nowProfile = currentProfile();
    if (!els.solveStats) return;
    const elapsed = Math.max(0, Math.round(performance.now() - runtime.startTs));
    const budgetText = Number.isFinite(runtime.maxMs) ? `/${formatMs(runtime.maxMs)}` : '/不限';
    runtime.throttleLevel = Math.max(runtime.throttleLevel, nowProfile.throttleLevel || 0);
    runtime.throttleFactor = Math.max(runtime.throttleFactor, nowProfile.throttleFactor || 1);
    if (nowProfile.guardReason && !runtime.maxGuardReason.includes(nowProfile.guardReason)) {
      runtime.maxGuardReason = runtime.maxGuardReason === '節奏保護：正常'
        ? nowProfile.guardReason
        : `${runtime.maxGuardReason} / ${nowProfile.guardReason}`;
    }
    const depthText = runtime.lastFrameCount ? `｜續算層級 ${runtime.lastFrameCount}/${runtime.groupsCount}` : `｜深度 ${runtime.lastFrameCount}`;
    const resumeHint = runtime.resumeCandidateSnapshot ? `｜已存續算 ${runtime.resumeCandidateSnapshot} 格式` : '';
    const statusText = `候選 ${result.length}組｜已搜 ${visited.count}節點｜耗時 ${formatMs(elapsed)} (上限${budgetText})｜${nowProfile.label}｜${describePowerState(runtime.powerState)}｜${nowProfile.guardReason}${depthText}${resumeHint}`;
    els.solveStats.textContent = statusText;
  };

  const maybeSleep = async (force = false) => {
    if (shouldAbort()) {
      runtime.aborted = true;
      return true;
    }

    await refreshPowerState();
    const nowProfile = currentProfile();
    const now = performance.now();
    if (runtime.timedOut || runtime.aborted) return true;
    if (!force && now - runtime.lastYieldTs < nowProfile.yieldEveryMs) {
      return shouldAbort();
    }

    runtime.lastYieldTs = now;
    if (runtime.maxMs < Number.POSITIVE_INFINITY && now - runtime.startTs >= runtime.maxMs) {
      runtime.timedOut = true;
      runtime.timedOutReason = '時間限制到達';
      runtime.timedOutTs = now;
      runtime.stopPolicy = '時間限制';
      return true;
    }

    if (nowProfile.forceStop) {
      runtime.timedOut = true;
      runtime.timedOutTs = now;
      runtime.stopReason = nowProfile.stopReason || '安全保護停止';
      runtime.timedOutReason = runtime.stopReason;
      runtime.stopPolicy = '安全保護';
      return true;
    }

    if (now >= runtime.lastHeartbeatTs) {
      updateRunningStatus();
      runtime.lastHeartbeatTs = now + nowProfile.heartbeatEveryMs;
    }

    if (now - runtime.lastCheckpointTs > SOLVER_RESUME.checkpointIntervalMs) {
      runtime.resumeCandidateSnapshot = Math.min(SOLVER_RESUME.maxCheckpointCandidates, result.length);
      await persistCheckpoint();
    }

    if (shouldAbort()) {
      runtime.aborted = true;
      return true;
    }

    runtime.throttleLevel = Math.max(runtime.throttleLevel, nowProfile.throttleLevel || 0);
    runtime.throttleFactor = Math.max(runtime.throttleFactor, nowProfile.throttleFactor || 1);
    await sleep(nowProfile.sleepMs);
    return runtime.aborted || runtime.timedOut;
  };

  const frames = [];
  const normalizedResumeFrames = normalizeSolverStack(resumeState?.stack || []);
  if (Array.isArray(normalizedResumeFrames) && normalizedResumeFrames.length <= SOLVER_RESUME.maxStackFrames) {
    let canUseResume = true;
    for (const item of normalizedResumeFrames) {
      if (!item || item.gIndex !== frames.length) {
        canUseResume = false;
        break;
      }
      const g = groups[item.gIndex];
      if (!g) {
        canUseResume = false;
        break;
      }
      const initial = {
        gIndex: item.gIndex,
        group: g,
        required: g.required,
        candidateIds: item.candidateIds || [],
        indices: Array.isArray(item.indices) ? item.indices.slice() : null,
        initialized: false,
        dead: false,
        applied: !!item.applied
      };
      frames.push(initial);
    }
    if (!canUseResume) {
      frames.length = 0;
    }
  }

  let level = frames.length;

  const ensureFrame = (frame) => {
    if (frame.initialized) return;
    const g = groups[frame.gIndex];
    if (!g) {
      frame.dead = true;
      frame.initialized = true;
      return;
    }
    frame.group = g;
    frame.required = g.required;
    frame.candidateIds = buildCandidateIdsForGroup(g, stateNow);
    if (frame.required === 0) {
      frame.candidateIds = [];
      frame.indices = [];
      frame.initialized = true;
      return;
    }
    if (frame.candidateIds.length < frame.required) {
      frame.dead = true;
      frame.initialized = true;
      return;
    }
    frame.indices = normalizeResumeFrameIndices(frame, frame.required, frame.candidateIds);
    if (!isValidIndices(frame.indices, frame.candidateIds.length, frame.required)) {
      frame.dead = true;
      return;
    }
    frame.initialized = true;
  };

  const clearFrameForNextLevel = () => {
    const f = frames[level - 1];
    if (!f || !f.group) return;
    const selectedIds = combinationToIds(f.candidateIds, f.indices);
    if (selectedIds.length === f.required) {
      rollback(f);
      f.applied = false;
    }
  };

  while (level >= 0 && level <= groups.length && !runtime.aborted && !runtime.timedOut) {
    if (await maybeSleep()) {
      break;
    }

    if (level === groups.length) {
      if (!runtime.timedOut && !runtime.aborted) {
        visited.count += 1;
        registerCandidate();
      }
      level--;
      continue;
    }

    let frame = frames[level];
    if (!frame) {
      frame = {
        gIndex: level,
        required: 0,
        candidateIds: [],
        indices: null,
        initialized: false,
        dead: false,
        applied: false,
        group: null
      };
      frames[level] = frame;
    }

    if (!frame.initialized) {
      ensureFrame(frame);
      if (frame.dead) {
        frame.dead = false;
        frames.pop();
        if (level > 0) {
          clearFrameForNextLevel();
        }
        level--;
        continue;
      }
      if (frame.required === 0) {
        frame.indices = [];
        frame.applied = false;
      }
    }

    if (!frame.candidateIds) {
      frame.dead = true;
      frame.initialized = true;
      continue;
    }

    if (!frame.applied) {
      const selected = combinationToIds(frame.candidateIds, frame.indices);
      const success = apply(frame);
      if (!success || selected.length !== frame.required) {
        frame.dead = true;
        if (frame.applied) {
          rollback(frame);
          frame.applied = false;
        }
        frame.initialized = true;
        continue;
      }
      frame.applied = true;
      level += 1;
      continue;
    }

    rollback(frame);
    frame.applied = false;
    const next = nextCombination(frame.indices, frame.candidateIds.length, frame.required);
    if (next) {
      frame.indices = next;
      continue;
    }
    frames.pop();
    level -= 1;
  }

  if ((runtime.aborted || runtime.timedOut) && frames.length) {
    runtime.lastFrameCount = Math.min(SOLVER_RESUME.maxStackFrames, frames.length);
    await persistCheckpoint();
  }

  const spentMs = Math.max(0, Math.round(performance.now() - runtime.startTs));
  if (!runtime.timedOut && !runtime.aborted && frames.length === 0 && !onCheckpoint) {
    // keep for compatibility with external behavior
  }
  if (els.solveStats) updateRunningStatus();

  return {
    candidates: result.map((c, idx) => ({ ...c, label: `#${idx + 1}` })),
    nodes: visited.count,
    spentMs,
    timedOut: runtime.timedOut,
    aborted: runtime.aborted,
    throttleLevel: runtime.throttleLevel,
    throttleFactor: runtime.throttleFactor,
    guardReason: runtime.maxGuardReason,
    stopPolicy: runtime.stopPolicy,
    stopReason: runtime.timedOutReason || runtime.stopReason || '',
    solverElapsedMin: Number((spentMs / 60000).toFixed(2)),
    powerState: runtime.powerState,
    maxMs: runtime.maxMs
  };
}

  function solveCurrentPeriod() {
  const group = getActiveGroup();
  const period = getActivePeriod(group);
  if (!group || !period || solveRunning) return;
  if (!group.members.length) {
    setStatus('請先新增員工', 'warn');
    return;
  }
  if (!memberInputExistsForPeriod(group, period)) {
    saveAllInputs();
  }
  const checks = analyzeSolvability(group, period);
  if (!checks.ok) {
    if (els.solveConflicts) {
      els.solveConflicts.textContent = checks.issues.join('\n');
    }
    setStatus('排班前先修正不可行條件', 'warn');
    return;
  }
  if (els.solveConflicts) {
    els.solveConflicts.textContent = '';
  }
  solveRunning = true;
  solveToken += 1;
  const currentToken = solveToken;
  setSolveControls(true);
  setStatus('排班中，請稍候...');
  setTimeout(async () => {
    const profile = getSolveProfile();
    const powerState = await detectPowerState();
    const adaptiveProfile = buildAdaptiveProfile(profile, powerState);
    const profileTag = adaptiveProfile.label || profile.label || '夜間節能（長時）';
    const maxCandidates = Math.min(6, Math.max(1, Number(els.solveMode.value || profile.maxCandidates || 3)));
    const maxMs = getSolveBudgetMs(profile);
    const result = await solvePeriod(group, period, maxCandidates, currentToken, {
      profile: adaptiveProfile,
      maxMs,
      powerState
    });
    if (!solveRunning || solveToken !== currentToken) return;

    period.candidates = result.candidates;
    period.status = result.candidates.length ? 'solved' : 'unsolved';
    period.selectedCandidateId = result.candidates[0]?.id || null;
    period.solverStats = {
      nodes: result.nodes,
      spentMs: result.spentMs,
      timedOut: result.timedOut,
      profile: profileTag,
      lowPower: !!result.powerState?.lowPower,
      throttleLevel: result.throttleLevel,
      throttleFactor: result.throttleFactor,
      guardReason: result.guardReason,
      stopPolicy: result.stopPolicy,
      stopReason: result.stopReason,
      powerLevelPercent: typeof result.powerState?.levelPercent === 'number' ? result.powerState.levelPercent : null,
      powerCharging: !!result.powerState?.charging
    };
    solveRunning = false;
    setSolveControls(false);

    if (result.aborted) {
      setStatus('排班已中斷', 'warn');
    } else if (result.timedOut) {
      if (result.stopPolicy === '安全保護') {
        setStatus(`保護停止：${result.stopReason || '安全條件觸發'}，暫時找到 ${result.candidates.length} 組候選`, result.candidates.length ? 'ok' : 'warn');
      } else {
        setStatus(`時間上限到達，暫時找到 ${result.candidates.length} 組候選`, result.candidates.length ? 'ok' : 'warn');
      }
    } else if (result.candidates.length) {
      setStatus(`完成排班，產生 ${result.candidates.length} 組候選（耗時 ${result.spentMs}ms，節點 ${result.nodes}）`, 'ok');
    } else {
      setStatus('排班完成，但未找到可行候選，請檢查衝突提示', 'warn');
    }
    saveState();
    render();
  }, 0);
}

function updateCandidateAssignment(group, period, candidateId, slotKey, rawNames) {
  const candidate = period.candidates.find((c) => c.id === candidateId);
  if (!candidate) return;
  const shiftId = slotKey.split('-')[1];
  const maxReq = period.shifts.find((s) => s.id === shiftId)?.required || 0;
  const names = rawNames
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);
  const mapped = [];
  for (const n of names) {
    const lower = n.toLowerCase();
    const exact = group.members.find((m) => m.name.toLowerCase() === lower);
    if (exact) mapped.push(exact.id);
    else {
      const fuzzy = group.members.find((m) => {
        const name = m.name.toLowerCase();
        return name.includes(lower) || lower.includes(name);
      });
      if (fuzzy && !mapped.includes(fuzzy.id)) mapped.push(fuzzy.id);
    }
  }
  if (mapped.length > maxReq) mapped.length = maxReq;
  candidate.assignments[slotKey] = mapped;
  const next = evaluateCandidate(candidate, period, group);
  Object.assign(candidate, next);
  candidate.label = candidate.label || `#${candidate.id}`;
  period.status = 'adjusted';
  saveState();
  render();
}

function pickCandidate(candidateId) {
  const group = getActiveGroup();
  const period = getActivePeriod(group);
  if (!group || !period) return;
  period.selectedCandidateId = candidateId;
  period.status = 'selected';
  period.candidates = period.candidates.map((c) => ({ ...c }));
  saveState();
  render();
  setStatus(`已選定候選 ${period.candidates.find((c) => c.id === candidateId)?.label || ''}`, 'ok');
}

function publishSelected() {
  const group = getActiveGroup();
  const period = getActivePeriod(group);
  if (!group || !period) return;
  if (!period.selectedCandidateId || !period.candidates.length) {
    setStatus('請先選擇一組候選班表', 'warn');
    return;
  }
  period.status = 'published';
  saveState();
  render();
  setStatus('本期班表已標記為已發布（後續可串接寄信/日曆）', 'ok');
}

function exportPeriodCSV() {
  const group = getActiveGroup();
  const period = getActivePeriod(group);
  if (!group || !period) return;
  if (!period.selectedCandidateId) {
    setStatus('請先選定一組候選', 'warn');
    return;
  }
  const candidate = period.candidates.find((c) => c.id === period.selectedCandidateId);
  if (!candidate) return;
  const lines = [];
  lines.push(['日期', '班別', '起訖', '指派人員'].join(','));
  for (let day = 0; day < period.dayCount; day++) {
    for (const s of period.shifts) {
      const key = `${day}-${s.id}`;
      const names = (candidate.assignments[key] || []).map((id) => {
        const mm = getMemberById(group, id);
        return mm ? mm.name : id;
      }).join('、');
      lines.push([dayLabel(period.startDate, day), s.label, `${s.start}-${s.end}`, names].join(','));
    }
  }
  const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `${group.name}-${period.name}.csv`;
  a.click();
  URL.revokeObjectURL(a.href);
}

function exportAll() {
  const blob = new Blob([JSON.stringify(state, null, 2)], { type: 'application/json;charset=utf-8' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `scheduling-app-state-${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
}

function exportPeriodJson() {
  const group = getActiveGroup();
  const period = getActivePeriod(group);
  if (!group || !period) return;
  const data = { groupName: group.name, period };
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json;charset=utf-8' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `${group.name}-${period.name}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
}

function deletePeriod(periodId) {
  const group = getActiveGroup();
  if (!group) return;
  group.periods = group.periods.filter((p) => p.id !== periodId);
  if (state.activePeriodId === periodId) state.activePeriodId = null;
  saveState();
  render();
}

function removeMember(memberId) {
  const group = getActiveGroup();
  if (!group) return;
  group.members = group.members.filter((m) => m.id !== memberId);
  for (const p of group.periods) {
    delete p.memberInputs[memberId];
    p.candidates = p.candidates.map((c) => {
      const copy = { ...c };
      const cleaned = {};
      for (const [k, v] of Object.entries(copy.assignments || {})) {
        cleaned[k] = v.filter((id) => id !== memberId);
      }
      copy.assignments = cleaned;
      return copy;
    });
  }
  saveState();
  render();
}

function attachEvents() {
  document.addEventListener('click', (ev) => {
    const action = ev.target.dataset.action;
    if (!action) return;

    if (action === 'add-group') {
      // preserved compatibility no-op
    } else if (action === 'select-period') {
      const periodId = ev.target.dataset.id;
      state.activePeriodId = periodId;
      saveState();
      render();
    } else if (action === 'delete-period') {
      const periodId = ev.target.dataset.id;
      if (confirm('確認刪除此期次？')) {
        deletePeriod(periodId);
      }
    } else if (action === 'remove-member') {
      const memberId = ev.target.dataset.id;
      if (confirm('確認刪除這位員工？')) {
        removeMember(memberId);
      }
    } else if (action === 'save-member-input') {
      const group = getActiveGroup();
      const period = getActivePeriod(group);
      const mid = ev.target.dataset.member;
      if (!group || !period) return;
      const data = collectMemberInputFromCard(group, period, mid);
      if (data) {
        period.memberInputs[mid] = data;
        period.status = 'collected';
        saveState();
        renderSolver();
        setStatus(`已儲存 ${group.members.find((m) => m.id === mid)?.name || '該員工'} 的需求`, 'ok');
      }
    } else if (action === 'save-all-inputs') {
      saveAllInputs();
    } else if (action === 'pick-candidate') {
      pickCandidate(ev.target.dataset.candidate);
    } else if (action === 'edit-slot') {
      const candidateId = ev.target.dataset.candidate;
      const slotKey = ev.target.dataset.slot;
      const group = getActiveGroup();
      const period = getActivePeriod(group);
      const candidate = period?.candidates.find((c) => c.id === candidateId);
      if (!group || !period || !candidate) return;
      const cur = (candidate.assignments[slotKey] || []).map((id) => {
        const mm = getMemberById(group, id);
        return mm ? mm.name : id;
      }).join('、');
      const maxReq = period.shifts.find((s) => `${slotKey.split('-')[0]}`) ? period.shifts.find((s) => s.id === slotKey.split('-')[1])?.required : 0;
      const raw = prompt(`請輸入人員姓名，逗號分隔（最多${maxReq}人）`, cur);
      if (raw === null) return;
      updateCandidateAssignment(group, period, candidateId, slotKey, raw);
    }
  });

  els.addGroupBtn.addEventListener('click', () => {
    const name = els.groupName.value.trim();
    if (!name) return;
    const group = newGroup(name);
    state.groups.push(group);
    state.activeGroupId = group.id;
    state.activePeriodId = null;
    saveState();
    render();
    els.groupName.value = '';
    setStatus(`已建立群組 ${name}`, 'ok');
  });

  els.groupSelect.addEventListener('change', (ev) => {
    state.activeGroupId = ev.target.value;
    state.activePeriodId = null;
    saveState();
    render();
  });

  els.addMemberBtn.addEventListener('click', () => {
    const group = getActiveGroup();
    if (!group) {
      setStatus('請先建立群組', 'warn');
      return;
    }
    const name = els.memberName.value.trim();
    const max = Number(els.memberMax.value || 10);
    if (!name) return;
    group.members.push(newMember(name, max));
    els.memberName.value = '';
    saveState();
    render();
  });

  els.addPeriodBtn.addEventListener('click', () => {
    const group = getActiveGroup();
    if (!group) {
      setStatus('請先建立群組', 'warn');
      return;
    }
    const name = els.periodName.value.trim() || `${safeDateAdd(els.periodStart.value || todayStr(), 0)} 期次`;
    const start = els.periodStart.value || todayStr();
    const dayCount = Number(els.periodDays.value || 7);
    const maxShiftsPerDay = Number(els.maxShiftsPerDay.value || 1);
    const shifts = getPeriodShiftRequirementInputs().length
      ? getPeriodShiftRequirementInputs().map((s) => ({
          ...s,
          required: Number(els.periodShifts.querySelector(`input[data-shift-id="${s.id}"]`)?.value || s.required)
        }))
      : group.shifts;
    const period = newPeriod(name, start, dayCount, maxShiftsPerDay, shifts);
    group.periods.push(period);
    state.activePeriodId = period.id;
    saveState();
    render();
    setStatus(`新增期次 ${name}`, 'ok');
  });

  els.solveBtn.addEventListener('click', solveCurrentPeriod);
  els.stopSolveBtn.addEventListener('click', () => {
    if (!solveRunning) return;
    solveToken += 1;
    solveRunning = false;
    setSolveControls(false);
    setStatus('已中止排班', 'warn');
  });
  els.solveProfile.addEventListener('change', () => {
    renderSolveHint();
  });
  els.publishBtn.addEventListener('click', publishSelected);
  els.exportPeriodJsonBtn.addEventListener('click', exportPeriodJson);
  els.exportPeriodCsvBtn.addEventListener('click', exportPeriodCSV);
  els.exportAllBtn.addEventListener('click', exportAll);
}

function render() {
  ensureDefaults();
  renderGroupPanel();
  renderMembers();
  renderShiftInputs();
  renderPeriods();
  renderInputPanel();
  renderSolver();
  renderSolveHint();
  saveState();
}

function initDefaults() {
  if (!state.groups.length) {
    const seed = newGroup('示範店鋪A');
    seed.members.push(newMember('王小明', 10), newMember('陳小華', 10), newMember('林雅婷', 10));
    seed.periods.push(newPeriod('第一期', todayStr(), 7, 1, seed.shifts));
    state.groups.push(seed);
    state.activeGroupId = seed.id;
    state.activePeriodId = seed.periods[0].id;
    for (const m of seed.members) {
      seed.periods[0].memberInputs[m.id] = newInputTemplate(seed.periods[0].dayCount, seed.periods[0].shifts);
    }
    saveState();
  }
  for (const g of state.groups) {
    if (!g.shifts || !g.shifts.length) g.shifts = JSON.parse(JSON.stringify(DEFAULT_SHIFTS));
    for (const p of g.periods) {
      if (!p.memberInputs) p.memberInputs = {};
      if (!p.candidates) p.candidates = [];
      if (!p.maxShiftsPerDay) p.maxShiftsPerDay = 1;
      if (!Array.isArray(p.shifts) || !p.shifts.length) p.shifts = JSON.parse(JSON.stringify(g.shifts));
      for (const m of g.members) {
        if (!p.memberInputs[m.id]) p.memberInputs[m.id] = newInputTemplate(p.dayCount, p.shifts);
      }
    }
  }
}

function bootstrap() {
  els = {
    groupSelect: document.getElementById('groupSelect'),
    groupName: document.getElementById('groupName'),
    addGroupBtn: document.getElementById('addGroupBtn'),
    memberPanel: document.getElementById('memberPanel'),
    memberName: document.getElementById('memberName'),
    memberMax: document.getElementById('memberMax'),
    addMemberBtn: document.getElementById('addMemberBtn'),
    memberList: document.getElementById('memberList'),
    periodName: document.getElementById('periodName'),
    periodStart: document.getElementById('periodStart'),
    periodDays: document.getElementById('periodDays'),
    maxShiftsPerDay: document.getElementById('maxShiftsPerDay'),
    addPeriodBtn: document.getElementById('addPeriodBtn'),
    periodShifts: document.getElementById('periodShifts'),
    periodList: document.getElementById('periodList'),
    inputPanelEmpty: document.getElementById('inputPanelEmpty'),
    inputPanelBody: document.getElementById('inputPanelBody'),
    solvePanelEmpty: document.getElementById('solvePanelEmpty'),
    solvePanelBody: document.getElementById('solvePanelBody'),
    solveMode: document.getElementById('solveMode'),
    solveProfile: document.getElementById('solveProfile'),
    solveRunMinutes: document.getElementById('solveRunMinutes'),
    solveModeHint: document.getElementById('solveModeHint'),
    solveBtn: document.getElementById('solveBtn'),
    stopSolveBtn: document.getElementById('stopSolveBtn'),
    solveConflicts: document.getElementById('solveConflicts'),
    solveStats: document.getElementById('solveStats'),
    candidateList: document.getElementById('candidateList'),
    publishBtn: document.getElementById('publishBtn'),
    exportPeriodJsonBtn: document.getElementById('exportPeriodJsonBtn'),
    exportPeriodCsvBtn: document.getElementById('exportPeriodCsvBtn'),
    exportAllBtn: document.getElementById('exportAllBtn'),
    statusText: document.getElementById('statusText')
  };

  els.periodStart.value = todayStr();
  if (els.solveProfile && !els.solveProfile.value) {
    els.solveProfile.value = 'night';
  }
  if (els.solveRunMinutes && !els.solveRunMinutes.value) {
    const defaults = getSolveProfile();
    els.solveRunMinutes.value = String(defaults.maxRunMinutesDefault);
  }
  initDefaults();
  attachEvents();
  setSolveControls(false);
  render();
  setStatus('系統就緒');
}

bootstrap();
