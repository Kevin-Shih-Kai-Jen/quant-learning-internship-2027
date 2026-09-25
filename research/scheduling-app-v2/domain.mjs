export const SCHEMA = 2;
export const AVAILABILITY = ['unknown', 'yes', 'prefer', 'no'];
export const clone = value => structuredClone(value);
export const uid = () => crypto.randomUUID();
export const emptyAssignments = snapshot => Object.fromEntries(snapshot.slots.map(s => [s.id, []]));
const fail = message => { throw new Error(message); };
const integer = (n, min, max) => Number.isInteger(n) && n >= min && n <= max;
const text = (value, max = 80) => typeof value === 'string' && value.trim().length > 0 && value.length <= max;
const idOK = id => typeof id === 'string' && /^[a-zA-Z0-9_-]{1,80}$/.test(id) && !['__proto__', 'constructor', 'prototype'].includes(id);
export function dateAdd(date, days) {
  const value = new Date(date + 'T00:00:00Z');
  value.setUTCDate(value.getUTCDate() + days);
  return value.toISOString().slice(0, 10);
}
export function validDate(date) {
  return typeof date === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(date) && Number.isFinite(Date.parse(date)) && dateAdd(date, 0) === date;
}
export function dayLabel(date) {
  const day = new Date(date + 'T00:00:00Z');
  return (day.getUTCMonth() + 1) + '/' + day.getUTCDate() + ' 週' + '日一二三四五六'[day.getUTCDay()];
}
export function timeMinutes(value) {
  if (typeof value !== 'string' || !/^([01]\d|2[0-3]):[0-5]\d$/.test(value)) fail('班次時間格式錯誤');
  const [hours, minutes] = value.split(':').map(Number);
  return hours * 60 + minutes;
}
export function makePeriod({ name, start, days = 7, deadline = '', shifts, rules }) {
  const p = {
    id: uid(), name, start, days, deadline, shifts: clone(shifts),
    rules: rules || { dailyMax: 1, restHours: 11, consecutiveMax: 5 },
    inputs: {}, inputRevision: 0, frozen: null, draft: null, candidates: [],
    versions: [], history: [], future: [], audit: [], lastRun: null
  };
  validateConfig(p);
  return p;
}
export function validateConfig(p) {
  if (!text(p.name) || !validDate(p.start) || !integer(p.days, 1, 14)) fail('期次名稱、日期或天數不正確（1–14 天）');
  if (p.deadline && !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$/.test(p.deadline)) fail('截止時間格式錯誤');
  if (!Array.isArray(p.shifts) || p.shifts.length < 1 || p.shifts.length > 4) fail('請設定 1–4 種班別');
  if (new Set(p.shifts.map(s => s.id)).size !== p.shifts.length) fail('班別 ID 重複');
  for (const s of p.shifts) {
    if (!idOK(s.id) || !text(s.name, 20) || !integer(s.required, 0, 20)) fail('班別名稱或人數不正確');
    if (timeMinutes(s.start) === timeMinutes(s.end)) fail('班次起訖不可相同');
  }
  if (!p.rules || !integer(p.rules.dailyMax, 1, 4) || !integer(p.rules.restHours, 0, 24) || !integer(p.rules.consecutiveMax, 1, 14)) fail('排班規則超出支援範圍');
}
export function slotsFor(p) {
  const slots = [];
  for (let day = 0; day < p.days; day++) {
    for (const shift of p.shifts) {
      const startMinute = day * 1440 + timeMinutes(shift.start);
      let endMinute = day * 1440 + timeMinutes(shift.end);
      if (endMinute <= startMinute) endMinute += 1440;
      slots.push({ ...shift, id: day + '_' + shift.id, shiftId: shift.id, day, date: dateAdd(p.start, day), startMinute, endMinute });
    }
  }
  return slots;
}
export function snapshotFor(group, period) {
  return {
    schema: SCHEMA, revision: period.inputRevision, start: period.start, days: period.days,
    members: clone(group.members), slots: slotsFor(period), rules: clone(period.rules),
    availability: Object.fromEntries(group.members.map(m => [m.id, Object.fromEntries(slotsFor(period).map(s => [s.id,
      period.inputs[m.id]?.confirmed ? (period.inputs[m.id].cells[s.id] || 'unknown') : 'unknown'
    ]))]))
  };
}
export function audit(period, message) {
  period.audit.push({ at: new Date().toISOString(), message });
  period.audit = period.audit.slice(-100);
}
export function saveInput(period, memberId, input) {
  period.inputs[memberId] = clone(input);
  period.inputRevision++;
  period.candidates = [];
  audit(period, '更新成員需求；既有凍結資料不會自動改寫');
}
export function freezeInputs(group, period) {
  if (group.members.length === 0) fail('請先新增成員');
  const missing = group.members.filter(m => !period.inputs[m.id]?.confirmed);
  if (missing.length) fail('還有 ' + missing.length + ' 位未確認需求；請先提交或明確將其全部設為不可排');
  period.frozen = snapshotFor(group, period);
  period.candidates = [];
  period.history = [];
  period.future = [];
  if (period.draft) period.draft.inputRevision = -1;
  audit(period, '凍結第 ' + period.inputRevision + ' 版需求');
}
export function currentSnapshot(period) {
  return !!period.frozen && period.frozen.revision === period.inputRevision;
}
export function assignmentReasons(snapshot, assignments, slotId, memberId) {
  const slot = snapshot.slots.find(s => s.id === slotId);
  const member = snapshot.members.find(m => m.id === memberId);
  if (!slot || !member) return ['班次或成員不存在'];
  const reasons = [];
  const available = snapshot.availability[memberId]?.[slotId];
  if (available !== 'yes' && available !== 'prefer') reasons.push(available === 'no' ? '已確認不可排' : '可排時間未確認');
  const assigned = snapshot.slots.filter(s => (assignments[s.id] || []).includes(memberId));
  const other = assigned.filter(s => s.id !== slotId);
  if (other.length >= member.maxShifts) reasons.push('已達本期 ' + member.maxShifts + ' 班上限');
  if (other.filter(s => s.day === slot.day).length >= snapshot.rules.dailyMax) reasons.push('已達每日班數上限');
  for (const s of other) {
    if (slot.startMinute < s.endMinute && s.startMinute < slot.endMinute) {
      reasons.push('與 ' + dayLabel(s.date) + ' ' + s.name + ' 時間重疊');
    } else {
      const gap = slot.startMinute >= s.endMinute ? slot.startMinute - s.endMinute : s.startMinute - slot.endMinute;
      if (gap < snapshot.rules.restHours * 60) reasons.push('與 ' + dayLabel(s.date) + ' ' + s.name + ' 班距不足 ' + snapshot.rules.restHours + ' 小時');
    }
  }
  const workDays = new Set([...other.map(s => s.day), slot.day]);
  let run = 0;
  for (let day = 0; day < snapshot.days; day++) {
    run = workDays.has(day) ? run + 1 : 0;
    if (run > snapshot.rules.consecutiveMax) { reasons.push('超過連續上班 ' + snapshot.rules.consecutiveMax + ' 天'); break; }
  }
  return [...new Set(reasons)];
}
export function validateAssignments(snapshot, assignments, { complete = false, locks = {} } = {}) {
  const issues = [];
  const known = new Set(snapshot.slots.map(s => s.id));
  for (const key of Object.keys(assignments)) if (!known.has(key)) issues.push({ slotId: key, message: '未知班次' });
  for (const key of Object.keys(locks)) if (!known.has(key)) issues.push({ slotId: key, message: '鎖定了未知班次' });
  for (const slot of snapshot.slots) {
    const ids = assignments[slot.id] || [];
    if (!Array.isArray(ids)) { issues.push({ slotId: slot.id, message: '指派資料格式錯誤' }); continue; }
    if (new Set(ids).size !== ids.length) issues.push({ slotId: slot.id, message: '同一人重複指派' });
    if (ids.length > slot.required) issues.push({ slotId: slot.id, message: '超過需求人數' });
    if (complete && ids.length < slot.required) issues.push({ slotId: slot.id, message: '尚缺 ' + (slot.required - ids.length) + ' 人' });
    for (const id of ids) for (const message of assignmentReasons(snapshot, assignments, slot.id, id)) issues.push({ slotId: slot.id, memberId: id, message });
    for (const id of locks[slot.id] || []) if (!ids.includes(id)) issues.push({ slotId: slot.id, message: '不能移除鎖定安排' });
  }
  return issues;
}
export function preflight(snapshot, locks = {}) {
  const issues = validateAssignments(snapshot, locks);
  const required = snapshot.slots.reduce((n, s) => n + s.required, 0);
  const capacity = snapshot.members.reduce((n, m) => n + m.maxShifts, 0);
  if (capacity < required) issues.push({ message: '本期需要 ' + required + ' 人次，但全員上限合計只有 ' + capacity + ' 班' });
  for (const s of snapshot.slots) {
    const eligible = snapshot.members.filter(m => !assignmentReasons(snapshot, locks, s.id, m.id).length).length;
    if (eligible < s.required) issues.push({ slotId: s.id, message: dayLabel(s.date) + ' ' + s.name + ' 需要 ' + s.required + ' 人，符合條件僅 ' + eligible + ' 人' });
  }
  return issues;
}
export function metrics(snapshot, assignments) {
  let filled = 0, preferred = 0;
  const loads = snapshot.members.map(m => {
    const work = snapshot.slots.filter(s => (assignments[s.id] || []).includes(m.id));
    filled += work.length;
    preferred += work.filter(s => snapshot.availability[m.id]?.[s.id] === 'prefer').length;
    return { id: m.id, name: m.name, count: work.length, hours: work.reduce((n, s) => n + (s.endMinute - s.startMinute) / 60, 0) };
  });
  const hours = loads.map(m => m.hours);
  return { filled, required: snapshot.slots.reduce((n, s) => n + s.required, 0), preferred, loads,
    spread: hours.length ? Math.max(...hours) - Math.min(...hours) : 0 };
}
export function difference(snapshot, before, after) {
  return snapshot.slots.flatMap(s => {
    const a = before[s.id] || [], b = after[s.id] || [];
    return a.length === b.length && a.every(id => b.includes(id)) ? [] : [{ slot: s, before: a, after: b }];
  });
}
export function changeDraft(period, next, label) {
  if (!currentSnapshot(period)) fail('需求已更新，請重新檢查並凍結');
  const issues = validateAssignments(period.frozen, next.assignments, { locks: next.locks || {} });
  if (issues.length) fail(issues[0].message);
  period.history.push(clone(period.draft));
  period.history = period.history.slice(-40);
  period.future = [];
  period.draft = { ...clone(next), inputRevision: period.inputRevision };
  audit(period, label);
}
export function travelHistory(period, direction) {
  const from = direction === 'undo' ? period.history : period.future;
  const to = direction === 'undo' ? period.future : period.history;
  if (!from.length || !currentSnapshot(period)) return;
  to.push(clone(period.draft));
  period.draft = from.pop();
  audit(period, direction === 'undo' ? '復原草稿修改' : '重做草稿修改');
}
export function publishBlockers(period) {
  if (!currentSnapshot(period) || period.draft?.inputRevision !== period.inputRevision) return ['草稿的需求版本已過期，請重新求解或逐班審核後採用'];
  if (!period.draft) return ['尚未建立草稿'];
  const issues = validateAssignments(period.frozen, period.draft.assignments, { complete: true });
  return issues.map(i => (period.frozen.slots.find(s => s.id === i.slotId)?.name || '') + '：' + i.message);
}
export function publish(period) {
  const blockers = publishBlockers(period);
  if (blockers.length) fail(blockers[0]);
  if (period.versions.length >= 200) fail('本期已達 200 個發布版本，請建立新期次');
  const last = period.versions.at(-1);
  if (last && last.snapshot.revision === period.inputRevision && !difference(period.frozen, last.assignments, period.draft.assignments).length) fail('這份班表已發布，沒有新的變更');
  const version = { number: period.versions.length + 1, at: new Date().toISOString(), snapshot: clone(period.frozen), assignments: clone(period.draft.assignments) };
  period.versions.push(version);
  audit(period, '在本機發布 v' + version.number + '（未發送通知）');
  return version;
}
// Deliberately narrow grammar: every complete line must match before it can be applied.
export function parseNotes(note, period) {
  const rules = [], unresolved = [];
  for (const line of note.split(/[\n；;]/).map(s => s.trim()).filter(Boolean)) {
    const match = line.match(/^(\d{4}-\d{2}-\d{2}|(?:週|星期)[一二三四五六日天])\s*(全部|早班|中班|晚班)\s*(不可排|可以排|希望排)$/);
    if (!match) { unresolved.push(line); continue; }
    const [, date, shift, kind] = match;
    const slots = slotsFor(period).filter(s => {
      const dateMatches = date.includes('-') ? s.date === date : new Date(s.date + 'T00:00:00Z').getUTCDay() === ('日一二三四五六'.indexOf(date.at(-1).replace('天', '日')));
      return dateMatches && (shift === '全部' || s.name === shift);
    });
    if (!slots.length) { unresolved.push(line); continue; }
    rules.push({ source: line, slotIds: slots.map(s => s.id), value: ({ 不可排: 'no', 可以排: 'yes', 希望排: 'prefer' })[kind] });
  }
  const conflicting = new Set();
  for (let i = 0; i < rules.length; i++) for (let j = i + 1; j < rules.length; j++) {
    if (rules[i].value !== rules[j].value && rules[i].slotIds.some(id => rules[j].slotIds.includes(id))) {
      conflicting.add(i); conflicting.add(j);
    }
  }
  for (const index of conflicting) unresolved.push(rules[index].source + '（同一時段有互相矛盾的要求）');
  return { rules: rules.filter((_, index) => !conflicting.has(index)), unresolved };
}
export function csvCell(value) {
  let str = String(value ?? '');
  if (/^[\s]*[=+@-]/.test(str) || /^[\t\r\n]/.test(str)) str = "'" + str;
  return '"' + str.replaceAll('"', '""') + '"';
}
export function toCSV(snapshot, assignments) {
  const rows = [['日期', '班別', '開始', '結束', '需求人數', '人員']];
  for (const slot of snapshot.slots) rows.push([slot.date, slot.name, slot.start, slot.end + (slot.endMinute >= (slot.day + 1) * 1440 ? '（次日）' : ''), slot.required, (assignments[slot.id] || []).map(id => snapshot.members.find(m => m.id === id)?.name || id).join('、')]);
  return '\uFEFF' + rows.map(r => r.map(csvCell).join(',')).join('\r\n');
}
export function createDemoState() {
  const members = ['林小夏', '陳柏宇', '王以安', '吳品蓉', '李子晴', '張宇森'].map((name, i) => ({ id: 'm' + i, name, maxShifts: 5 }));
  const now = new Date();
  const today = new Date(now.getTime() + 8 * 3600000).toISOString().slice(0, 10);
  const start = dateAdd(today, (8 - new Date(today + 'T00:00:00Z').getUTCDay()) % 7 || 7);
  const period = makePeriod({ name: '下週門市班表', start, shifts: [
    { id: 'am', name: '早班', start: '09:00', end: '13:00', required: 2 },
    { id: 'pm', name: '晚班', start: '14:00', end: '18:00', required: 2 }
  ] });
  const group = { id: uid(), name: '日和咖啡', demo: true, members, periods: [period] };
  const assignments = {};
  slotsFor(period).forEach((s, i) => { assignments[s.id] = [members[(i * 2) % 6].id, members[(i * 2 + 1) % 6].id]; });
  for (const member of members) period.inputs[member.id] = { confirmed: true, note: '', cells: Object.fromEntries(slotsFor(period).map(s => [s.id, assignments[s.id].includes(member.id) ? 'prefer' : 'yes'])) };
  period.inputRevision = 6;
  freezeInputs(group, period);
  period.draft = { assignments, locks: {}, inputRevision: 6 };
  return { schema: SCHEMA, revision: 0, activeGroupId: group.id, activePeriodId: period.id, groups: [group] };
}
// Validate the entire backup before replacing local data; never execute imported text.
export function validateState(raw) {
  if (!raw || raw.schema !== SCHEMA || !integer(raw.revision, 0, Number.MAX_SAFE_INTEGER) || !Array.isArray(raw.groups) || raw.groups.length < 1 || raw.groups.length > 20) fail('不是支援的 v2 備份檔');
  const rawText = JSON.stringify(raw);
  if (rawText.length > 4_000_000 || /"(?:__proto__|constructor|prototype)"\s*:/.test(rawText)) fail('備份含不支援的欄位或超過 4 MB');
  const unique = items => new Set(items.map(i => i.id)).size === items.length && items.every(i => idOK(i.id));
  if (!unique(raw.groups)) fail('群組 ID 無效');
  for (const group of raw.groups) {
    if (!text(group.name) || !Array.isArray(group.members) || group.members.length > 20 || !unique(group.members)) fail('群組或成員格式錯誤');
    for (const m of group.members) if (!text(m.name, 40) || !integer(m.maxShifts, 0, 56)) fail('成員格式錯誤');
    if (!Array.isArray(group.periods) || group.periods.length > 52 || !unique(group.periods)) fail('期次格式錯誤');
    for (const p of group.periods) {
      validateConfig(p);
      if (!integer(p.inputRevision, 0, Number.MAX_SAFE_INTEGER) || !p.inputs || typeof p.inputs !== 'object') fail('需求版本錯誤');
      for (const input of Object.values(p.inputs)) {
        if (typeof input.confirmed !== 'boolean' || !input.cells || typeof input.note !== 'string' || input.note.length > 2000) fail('需求格式錯誤');
        if (Object.values(input.cells).some(v => !AVAILABILITY.includes(v))) fail('可排狀態無效');
      }
      for (const key of ['candidates', 'versions', 'history', 'future', 'audit']) if (!Array.isArray(p[key]) || p[key].length > 200) fail('歷史資料格式錯誤');
      for (const event of p.audit) if (typeof event.at !== 'string' || typeof event.message !== 'string') fail('操作紀錄格式錯誤');
      const validateSnapshot = s => {
        if (!s || s.schema !== SCHEMA || !integer(s.revision, 0, p.inputRevision) || !validDate(s.start) || !integer(s.days, 1, 14) || !Array.isArray(s.members) || !unique(s.members) || s.members.length > 20 || !Array.isArray(s.slots) || !unique(s.slots) || s.slots.length > 56 || !s.availability) fail('凍結資料格式錯誤');
        if (!s.rules || !integer(s.rules.dailyMax, 1, 4) || !integer(s.rules.restHours, 0, 24) || !integer(s.rules.consecutiveMax, 1, 14)) fail('凍結規則格式錯誤');
        for (const m of s.members) if (!text(m.name, 40) || !integer(m.maxShifts, 0, 56) || !s.availability[m.id]) fail('凍結成員錯誤');
        for (const slot of s.slots) {
          if (!integer(slot.day, 0, s.days - 1) || !integer(slot.required, 0, 20) || !idOK(slot.shiftId) || !text(slot.name, 20) || slot.date !== dateAdd(s.start, slot.day)) fail('凍結班次錯誤');
          const start = slot.day * 1440 + timeMinutes(slot.start);
          const end = slot.day * 1440 + timeMinutes(slot.end) + (timeMinutes(slot.end) <= timeMinutes(slot.start) ? 1440 : 0);
          if (start !== slot.startMinute || end !== slot.endMinute || end === start) fail('凍結時間錯誤');
          for (const m of s.members) if (!AVAILABILITY.includes(s.availability[m.id][slot.id])) fail('凍結可排資料錯誤');
        }
      };
      if (p.frozen) {
        validateSnapshot(p.frozen);
        if (p.frozen.revision === p.inputRevision && JSON.stringify(p.frozen) !== JSON.stringify(snapshotFor(group, p))) fail('凍結資料與需求不一致');
      }
      const validateDraftShape = draft => {
        if (draft === null) return;
        if (!draft || !draft.assignments || !draft.locks || !integer(draft.inputRevision, -1, p.inputRevision)) fail('草稿格式錯誤');
        for (const map of [draft.assignments, draft.locks]) for (const [key, ids] of Object.entries(map)) if (!idOK(key) || !Array.isArray(ids) || ids.length > 20 || ids.some(id => !idOK(id))) fail('指派格式錯誤');
      };
      validateDraftShape(p.draft);
      p.history.forEach(validateDraftShape); p.future.forEach(validateDraftShape);
      // Candidate results are disposable; importing never trusts a solver claim.
      p.candidates = [];
      p.lastRun = null;
      for (let i = 0; i < p.versions.length; i++) {
        const v = p.versions[i];
        validateSnapshot(v.snapshot);
        if (v.number !== i + 1 || !Number.isFinite(Date.parse(v.at)) || !v.assignments || validateAssignments(v.snapshot, v.assignments, { complete: true }).length) fail('已發布版本無效');
      }
    }
  }
  const active = raw.groups.find(g => g.id === raw.activeGroupId);
  if (!active || (active.periods.length && !active.periods.some(p => p.id === raw.activePeriodId))) fail('目前期次不存在');
  return clone(raw);
}
