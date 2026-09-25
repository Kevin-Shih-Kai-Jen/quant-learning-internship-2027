import test from 'node:test';
import assert from 'node:assert/strict';
import {
  clone, createDemoState, makePeriod, slotsFor, snapshotFor, freezeInputs, currentSnapshot,
  assignmentReasons, validateAssignments, emptyAssignments, changeDraft, travelHistory,
  publish, publishBlockers, saveInput, parseNotes, toCSV, validateState
} from '../domain.mjs';
import { solveSync } from '../solver.mjs';
import { createStore, STORAGE_KEY } from '../storage.mjs';

function fixture({ days = 2, count = 3, required = 1, maxShifts = 5, shifts, rules } = {}) {
  const p = makePeriod({ name: '測試週', start: '2026-09-23', days, shifts: shifts || [{ id: 'am', name: '早班', start: '09:00', end: '13:00', required }], rules: rules || { dailyMax: 1, restHours: 11, consecutiveMax: 5 } });
  const g = { id: 'group', name: '測試', members: Array.from({ length: count }, (_, i) => ({ id: 'm' + i, name: '人員 ' + i, maxShifts })), periods: [p] };
  for (const m of g.members) p.inputs[m.id] = { confirmed: true, note: '', cells: Object.fromEntries(slotsFor(p).map(s => [s.id, 'yes'])) };
  freezeInputs(g, p);
  return { g, p, s: p.frozen };
}
test('demo is feasible and its full backup round-trips', () => {
  const state = createDemoState(), p = state.groups[0].periods[0];
  assert.deepEqual(validateAssignments(p.frozen, p.draft.assignments, { complete: true }), []);
  assert.equal(validateState(JSON.parse(JSON.stringify(state))).schema, 2);
  const result = solveSync(p.frozen);
  assert.equal(result.candidates.length, 3);
  for (const c of result.candidates) assert.deepEqual(validateAssignments(p.frozen, c.assignments, { complete: true }), []);
});
test('unknown and unconfirmed submissions cannot become availability', () => {
  const { g, p } = fixture();
  p.inputs.m0.confirmed = false;
  assert.throws(() => freezeInputs(g, p), /未確認/);
  const s = snapshotFor(g, p);
  assert.deepEqual(assignmentReasons(s, {}, '0_am', 'm0'), ['可排時間未確認']);
  p.inputs.m0.confirmed = true; p.inputs.m0.cells['0_am'] = 'unknown';
  assert.match(assignmentReasons(snapshotFor(g, p), {}, '0_am', 'm0')[0], /未確認/);
});
test('zero demand and zero member capacity remain zero', () => {
  const { s } = fixture({ required: 0, maxShifts: 0 });
  const result = solveSync(s);
  assert.equal(result.status, 'feasible');
  assert.deepEqual(result.candidates[0].assignments, emptyAssignments(s));
  const noCapacity = fixture({ count: 1, maxShifts: 0 });
  assert.equal(solveSync(noCapacity.s).status, 'infeasible');
});
test('overnight overlap and insufficient rest are enforced symmetrically', () => {
  const { s } = fixture({ days: 2, count: 1, rules: { dailyMax: 3, restHours: 11, consecutiveMax: 5 }, shifts: [
    { id: 'night', name: '晚班', start: '22:00', end: '06:00', required: 1 },
    { id: 'am', name: '早班', start: '05:00', end: '09:00', required: 1 },
    { id: 'noon', name: '中班', start: '12:00', end: '16:00', required: 1 }
  ] });
  assert.ok(assignmentReasons(s, { '0_night': ['m0'] }, '1_am', 'm0').some(x => x.includes('重疊')));
  assert.ok(assignmentReasons(s, { '1_am': ['m0'] }, '0_night', 'm0').some(x => x.includes('重疊')));
  assert.ok(assignmentReasons(s, { '0_night': ['m0'] }, '1_noon', 'm0').some(x => x.includes('班距不足')));
});
test('daily caps, period caps, consecutive days and duplicate assignments are blocked', () => {
  const { s } = fixture({ days: 4, count: 1, maxShifts: 3, rules: { dailyMax: 1, restHours: 0, consecutiveMax: 2 } });
  assert.match(assignmentReasons(s, { '0_am': ['m0'], '1_am': ['m0'] }, '2_am', 'm0').join(), /連續/);
  assert.match(assignmentReasons(s, { '0_am': ['m0'], '1_am': ['m0'], '2_am': ['m0'] }, '3_am', 'm0').join(), /上限/);
  assert.ok(validateAssignments(s, { '0_am': ['m0', 'm0'] }).some(x => x.message.includes('重複')));
});
test('all locks are honored, and contradictory locks fail closed', () => {
  const { s } = fixture();
  const locks = { '0_am': ['m2'] };
  const result = solveSync(s, { locks });
  for (const c of result.candidates) assert.deepEqual(c.assignments['0_am'], ['m2']);
  s.availability.m2['0_am'] = 'no';
  assert.equal(solveSync(s, { locks }).status, 'infeasible');
});
test('a node budget is not reported as infeasibility; depth above 12 remains valid', () => {
  const { s } = fixture({ days: 14, maxShifts: 14, rules: { dailyMax: 1, restHours: 11, consecutiveMax: 14 } });
  assert.equal(solveSync(s, { maxNodes: 1 }).status, 'limited');
  const result = solveSync(s, { maxCandidates: 1 });
  assert.equal(result.status, 'feasible');
  assert.equal(Object.values(result.candidates[0].assignments).flat().length, 14);
});
test('dead branches terminate and exhaustive search can prove global capacity conflicts', () => {
  const { s } = fixture({ days: 3, count: 3, maxShifts: 1 });
  for (const m of ['m1', 'm2']) for (const key of ['0_am', '1_am']) s.availability[m][key] = 'no';
  const result = solveSync(s);
  assert.equal(result.status, 'infeasible');
  assert.equal(result.reason, 'exhausted');
  assert.ok(result.nodes < 10);
});
// Independent small brute-force oracle compares the entire solution set. It uses
// chronological interval scans and direct counters, not the production validator.
function bruteForce(s) {
  const result = [];
  function valid(assignment) {
    for (const m of s.members) {
      const work = s.slots.filter(slot => assignment[slot.id]?.includes(m.id)).sort((a, b) => a.startMinute - b.startMinute);
      if (work.length > m.maxShifts) return false;
      for (const w of work) if (!['yes', 'prefer'].includes(s.availability[m.id][w.id])) return false;
      for (let i = 1; i < work.length; i++) if (work[i].startMinute - work[i - 1].endMinute < s.rules.restHours * 60) return false;
      const days = Array(s.days).fill(0); work.forEach(w => days[w.day]++);
      if (days.some(n => n > s.rules.dailyMax)) return false;
      let streak = 0; for (const n of days) { streak = n ? streak + 1 : 0; if (streak > s.rules.consecutiveMax) return false; }
    }
    return true;
  }
  function visit(i, a) {
    if (i === s.slots.length) { if (valid(a)) result.push(clone(a)); return; }
    const slot = s.slots[i];
    for (let mask = 0; mask < 2 ** s.members.length; mask++) {
      const ids = s.members.filter((_, index) => mask & (1 << index)).map(m => m.id);
      if (ids.length === slot.required) visit(i + 1, { ...a, [slot.id]: ids });
    }
  }
  visit(0, {});
  return result;
}
const signature = a => Object.keys(a).sort().map(k => k + ':' + a[k].slice().sort().join(',')).join('|');
test('exhaustive solutions match an independent oracle across 36 small cases', () => {
  for (let seed = 0; seed < 36; seed++) {
    const { s } = fixture({ days: 3, count: 3, required: seed % 3 === 0 ? 2 : 1, maxShifts: 1 + seed % 3, rules: { dailyMax: 1, restHours: 11, consecutiveMax: 1 + seed % 3 } });
    s.members.forEach((m, i) => s.slots.forEach((slot, j) => { if ((seed + i * 5 + j * 3) % 7 === 0) s.availability[m.id][slot.id] = 'no'; }));
    const expected = bruteForce(s).map(signature).sort();
    const result = solveSync(s, { maxCandidates: 1000, maxNodes: 10000, minDistance: 1 });
    assert.deepEqual(result.candidates.map(c => signature(c.assignments)).sort(), expected, 'seed ' + seed);
    assert.equal(result.exhausted, true);
  }
});
test('edits and undo/redo persist independently of published immutable versions', () => {
  const state = createDemoState(), p = state.groups[0].periods[0];
  publish(p); const published = JSON.stringify(p.versions[0]);
  const next = clone(p.draft); next.assignments['0_am'] = []; next.locks['0_am'] = [];
  changeDraft(p, next, '刪除');
  assert.ok(publishBlockers(p).length);
  travelHistory(p, 'undo'); assert.deepEqual(publishBlockers(p), []);
  travelHistory(p, 'redo'); assert.ok(publishBlockers(p).length);
  assert.equal(JSON.stringify(p.versions[0]), published);
  const restored = validateState(JSON.parse(JSON.stringify(state)));
  travelHistory(restored.groups[0].periods[0], 'undo');
  assert.deepEqual(publishBlockers(restored.groups[0].periods[0]), []);
});
test('stale snapshots cannot be published; revalidation and new versions are explicit', () => {
  const state = createDemoState(), g = state.groups[0], p = g.periods[0];
  publish(p);
  saveInput(p, 'm0', p.inputs.m0);
  assert.equal(currentSnapshot(p), false);
  assert.throws(() => publish(p), /過期/);
  freezeInputs(g, p);
  assert.throws(() => publish(p), /過期/);
  changeDraft(p, p.draft, '重新確認');
  assert.equal(publish(p).number, 2);
  assert.throws(() => publish(p), /已發布/);
});
test('candidate application cannot erase a new manual lock', () => {
  const { p, s } = fixture();
  const assignments = solveSync(s, { maxCandidates: 1 }).candidates[0].assignments;
  changeDraft(p, { assignments, locks: {} }, '套用');
  assert.throws(() => changeDraft(p, { assignments, locks: { '0_am': [s.members.find(m => !assignments['0_am'].includes(m.id)).id] } }, '舊候選'), /鎖定/);
});
test('weekday parsing uses actual calendar dates and refuses ambiguous extra text', () => {
  const { p } = fixture({ days: 7 });
  const r = parseNotes('週三 早班 不可排\n週五 全部 希望排\n週三 早班 不可排但晚上可以\n2026-10-01 全部 不可排', p);
  assert.deepEqual(r.rules[0].slotIds, ['0_am']);
  assert.deepEqual(r.rules[1].slotIds, ['2_am']);
  assert.equal(r.rules[1].value, 'prefer');
  assert.equal(r.unresolved.length, 2);
  const contradictory = parseNotes('週三 全部 不可排;週三 早班 可以排', p);
  assert.equal(contradictory.rules.length, 0);
  assert.equal(contradictory.unresolved.length, 2);
});
test('CSV safely quotes commas, double quotes, newlines and formula prefixes', () => {
  const { s } = fixture({ count: 1, days: 1 });
  s.members[0].name = '=HYPERLINK("bad"),\n小明';
  const csv = toCSV(s, { '0_am': ['m0'] });
  assert.ok(csv.includes('"\'=HYPERLINK(""bad""),\n小明"'));
  assert.ok(csv.startsWith('\uFEFF'));
});
test('backup validation rejects invalid state, prototype keys and tampered published schedules', () => {
  assert.throws(() => validateState({ schema: 1 }), /v2/);
  const state = createDemoState(); publish(state.groups[0].periods[0]);
  state.groups[0].periods[0].versions[0].assignments['0_am'] = [];
  assert.throws(() => validateState(state), /發布/);
  const dangerous = JSON.parse(JSON.stringify(createDemoState()));
  Object.defineProperty(dangerous, '__proto__', { value: {}, enumerable: true });
  assert.throws(() => validateState(dangerous), /欄位/);
});
function memoryStorage() {
  const map = new Map();
  return { getItem: k => map.get(k) ?? null, setItem: (k, v) => map.set(k, v) };
}
test('storage failure rolls back memory and a stale tab cannot overwrite changes', () => {
  const storage = memoryStorage(), a = createStore(storage), b = createStore(storage);
  a.commit(s => { s.groups[0].name = '新名稱'; });
  assert.throws(() => b.commit(s => { s.groups[0].name = '過期覆蓋'; }), /另一個分頁/);
  assert.equal(JSON.parse(storage.getItem(STORAGE_KEY)).groups[0].name, '新名稱');
  const before = clone(a.state);
  storage.setItem = () => { throw new Error('QuotaExceeded'); };
  assert.throws(() => a.commit(s => { s.groups[0].name = '不應保存'; }), /QuotaExceeded/);
  assert.deepEqual(a.state, before);
});
test('corrupt local data is preserved instead of silently replaced with a demo', () => {
  const storage = memoryStorage(); storage.setItem(STORAGE_KEY, '{invalid');
  const store = createStore(storage);
  assert.equal(store.state, null);
  assert.equal(store.raw, '{invalid');
  assert.equal(storage.getItem(STORAGE_KEY), '{invalid');
});
