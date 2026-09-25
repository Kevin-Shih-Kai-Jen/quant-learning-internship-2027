import {
  clone, uid, dateAdd, dayLabel, makePeriod, slotsFor, snapshotFor, freezeInputs,
  currentSnapshot, assignmentReasons, validateAssignments, preflight, metrics,
  difference, changeDraft, travelHistory, publishBlockers, publish, parseNotes,
  toCSV, validateState, emptyAssignments, saveInput, audit
} from './domain.mjs';
import { createStore, STORAGE_KEY } from './storage.mjs';

const $ = selector => document.querySelector(selector);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
const paths = {
  calendar: '<rect x="3" y="5" width="18" height="16" rx="3"/><path d="M7 3v4m10-4v4M3 11h18m-13 4h2m4 0h2m-8 3h2"/>',
  people: '<circle cx="9" cy="8" r="3"/><path d="M3 21v-3a6 6 0 0 1 12 0v3m1-16a3 3 0 0 1 0 6m2 3a5 5 0 0 1 3 4v3"/>',
  check: '<path d="m5 12 4 4L19 6"/>',
  list: '<path d="M9 6h11M9 12h11M9 18h11M4 6h.01M4 12h.01M4 18h.01"/>',
  settings: '<path d="M4 6h16M4 12h16M4 18h16"/><circle cx="8" cy="6" r="2"/><circle cx="16" cy="12" r="2"/><circle cx="10" cy="18" r="2"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  arrow: '<path d="M4 12h16m-6-6 6 6-6 6"/>',
  lock: '<rect x="5" y="10" width="14" height="11" rx="2"/><path d="M8 10V6a4 4 0 0 1 8 0v4"/>',
  download: '<path d="M12 3v12m-5-5 5 5 5-5M4 16v4h16v-4"/>',
  spark: '<path d="m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5Z"/>',
  undo: '<path d="M9 4 4 9l5 5M4 9h10a6 6 0 0 1 0 12"/>',
  redo: '<path d="m15 4 5 5-5 5m5-5H10a6 6 0 0 0 0 12"/>',
  clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
  close: '<path d="m6 6 12 12M6 18 18 6"/>'
};
const icon = name => `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths[name] || paths.calendar}</svg>`;
const btn = (action, label, cls = '', extra = '') => `<button type="button" class="${cls}" data-action="${action}" ${extra}>${label}</button>`;
const badge = (label, cls = '') => `<span class="badge ${cls}">${label}</span>`;
const names = (snapshot, ids) => ids.map(id => snapshot.members.find(m => m.id === id)?.name || '未知成員').join('、') || '未安排';
const stored = createStore(localStorage);
let tab = 'board', view = 'coverage', selectedSlot = null, candidateId = null, versionNumber = 0;
let job = null, toastTimer = null, diffOnly = false, pendingImport = null;
const state = () => stored.state;
const group = () => state()?.groups.find(g => g.id === state().activeGroupId);
const period = () => group()?.periods.find(p => p.id === state().activePeriodId);
function toast(message, error = false) {
  const node = $('#toast');
  node.textContent = message; node.className = error ? 'error' : ''; node.hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => { node.hidden = true; }, error ? 9000 : 4500);
}
function mutate(action, message) {
  try { stored.commit(action); render(); if (message) toast(message); return true; }
  catch (e) { toast(e.message, true); return false; }
}
function updatePeriod(action, message) {
  const gid = group().id, pid = period().id;
  return mutate(s => action(s.groups.find(g => g.id === gid).periods.find(p => p.id === pid)), message);
}
function selectedData() {
  const p = period(), g = group();
  const version = p?.versions.find(v => v.number === versionNumber);
  const snapshot = version?.snapshot || p?.frozen || (p && snapshotFor(g, p));
  const candidate = p?.candidates.find(c => c.id === candidateId);
  return {
    snapshot, assignments: version?.assignments || candidate?.assignments || p?.draft?.assignments || (snapshot && emptyAssignments(snapshot)),
    locks: p?.draft?.locks || {}, readOnly: !!(version || candidate), version, candidate
  };
}
function resetSelection() { selectedSlot = null; candidateId = null; versionNumber = 0; diffOnly = false; }
function workflow(g, p) {
  const submitted = g.members.filter(m => p.inputs[m.id]?.confirmed).length;
  const fresh = currentSnapshot(p);
  const steps = [
    ['01', '收集需求', submitted + ' / ' + g.members.length + ' 位已確認', 'inputs', submitted === g.members.length && submitted > 0],
    ['02', '檢查與凍結', fresh ? '本期需求已凍結' : '確認規則與缺口', 'review', fresh],
    ['03', '調整班表', p.draft ? '草稿可編輯、可復原' : '產生候選方案', 'board', !!p.draft],
    ['04', '發布版本', p.versions.length ? '已發布 v' + p.versions.length : '審核後留下版本', 'publish', p.versions.length > 0]
  ];
  return `<div class="workflow" aria-label="排班工作流程">${steps.map(([n, title, subtitle, target, complete]) => `<button class="workflow-step ${complete ? 'complete' : ''}" data-action="step" data-target="${target}"><span class="step-number">${complete ? icon('check') : n}</span><span><strong>${title}</strong><small>${subtitle}</small></span>${icon('arrow')}</button>`).join('')}</div>`;
}
function render() {
  if (!state()) {
    $('#app').innerHTML = `<main class="recovery"><img src="icon.svg" alt="" width="64"><h1>資料需要復原</h1><p>${esc(stored.error)}</p><p>原始資料保留在瀏覽器，沒有被新資料覆蓋。</p>${btn('raw-export', '下載原始資料', 'primary')}<p>可下載後保留原檔，再由開發者協助修復。</p></main>`;
    return;
  }
  const g = group(), p = period();
  const navs = [['board', 'calendar', '班表工作室'], ['inputs', 'list', '需求收集'], ['team', 'people', '團隊成員'], ['settings', 'settings', '規則與資料']];
  $('#app').innerHTML = `
    <aside class="sidebar">
      <a class="brand" href="./"><img src="icon.svg" width="40" height="40" alt=""><span>班伴<small>SHIFT STUDIO</small></span></a>
      <label class="field-label" for="group-select">我的工作空間</label>
      <select id="group-select" aria-label="切換群組">${state().groups.map(item => `<option value="${esc(item.id)}" ${item.id === g.id ? 'selected' : ''}>${esc(item.name)}</option>`).join('')}</select>
      ${btn('new-group', icon('plus') + '建立群組', 'text-button new-group')}
      <nav aria-label="主導覽">${navs.map(([key, symbol, label]) => `<button data-action="nav" data-tab="${key}" class="nav-item ${tab === key ? 'active' : ''}" ${tab === key ? 'aria-current="page"' : ''}>${icon(symbol)}${label}${key === 'inputs' && p ? '<span class="nav-count">' + g.members.filter(m => !p.inputs[m.id]?.confirmed).length + '</span>' : ''}</button>`).join('')}</nav>
      <div class="sidebar-note"><span class="local-dot"></span><strong>本機工作空間</strong><p>資料儲存在這個瀏覽器。<br>尚未連接多人同步。</p>${btn('backup', icon('download') + '匯出備份', 'text-button')}</div>
      <div class="sidebar-bottom">把複雜留給班伴<br><span>把時間留給團隊。</span></div>
    </aside>
    <main id="main" tabindex="-1">
      <div class="compact-workspace"><label for="group-select-mobile">工作空間</label><select id="group-select-mobile">${state().groups.map(item => `<option value="${esc(item.id)}" ${item.id === g.id ? 'selected' : ''}>${esc(item.name)}</option>`).join('')}</select>${btn('new-group', icon('plus'), 'icon-button', 'aria-label="建立群組"')}</div>
      <header class="page-header"><div><div class="eyebrow">${esc(g.name)} <span>/</span> ${g.demo ? '示範工作空間' : '團隊工作空間'}</div><h1>${({ board: '一起排好，下一週。', inputs: '讓每個人的時間被看見。', team: '好班表，從團隊開始。', settings: '每一條規則，都清楚。' })[tab]}</h1><p>${({ board: '收好需求、看懂安排，讓每一班都有共識。', inputs: '由管理者在本機代填；未確認的時段不會自動被安排。', team: '管理成員與本期可承擔的班數。', settings: '規則調整會保留原班表，重新確認後才生效。' })[tab]}</p></div><div class="header-actions">${badge('本機試用版', 'neutral')}${btn('new-period', icon('plus') + '新一期班表', 'primary')}</div></header>
      ${p ? `<div class="period-bar"><label for="period-select">${icon('calendar')} 排班週期</label><select id="period-select">${g.periods.map(item => `<option value="${esc(item.id)}" ${item.id === p.id ? 'selected' : ''}>${esc(item.name)} · ${esc(item.start)}</option>`).join('')}</select><span class="period-dates">${dayLabel(p.start)} — ${dayLabel(dateAdd(p.start, p.days - 1))}</span><span class="deadline">${icon('clock')}${p.deadline ? '截止 ' + esc(p.deadline.replace('T', ' ')) + '（台北）' : '尚未設定截止時間'}</span></div>${workflow(g, p)}` : ''}
      ${tab === 'team' ? renderTeam() : tab === 'settings' ? renderSettings() : !p ? renderEmpty('建立第一期班表', '先設定起始日、班別與人力需求。', 'new-period', '建立班表') : tab === 'inputs' ? renderInputs() : renderBoard()}
      <footer class="page-footer"><span>班伴 v0.2 · 設計給小團隊的排班日常</span><span>只在本機保存 · 建議定期匯出備份</span></footer>
    </main>`;
}
function renderEmpty(title, description, action, label) {
  return `<section class="empty-state">${icon('calendar')}<h2>${title}</h2><p>${description}</p>${btn(action, label, 'primary')}</section>`;
}
function renderBoard() {
  const p = period(), { snapshot, assignments, candidate, version } = selectedData();
  const score = metrics(snapshot, assignments);
  const hard = validateAssignments(snapshot, assignments);
  const fresh = currentSnapshot(p);
  const diff = difference(snapshot, p.draft?.assignments || {}, assignments);
  const latest = p.versions.at(-1);
  const publicationDiff = latest ? difference(snapshot, latest.assignments, assignments) : [];
  const stats = [[score.filled + '<small> / ' + score.required + '</small>', '已安排人次', 'coverage'], [score.required - score.filled, '待補人次', score.required > score.filled ? 'warning' : ''], [hard.length, '規則衝突', hard.length ? 'warning' : ''], [score.preferred, '排在偏好時段', '']];
  return `
      ${!fresh ? `<div class="notice warning">${icon('list')}<span><strong>${p.frozen ? '需求有更新，目前草稿使用舊資料。' : '先確認需求，再開始排班。'}</strong>請到檢查與凍結，確認此次排班要使用的資料。</span>${btn('review', '檢查需求')}</div>` : ''}
      ${fresh && (!p.draft || p.draft.inputRevision !== p.inputRevision) ? `<div class="notice"><span>${p.draft ? '需求已重新凍結。可檢查舊草稿後沿用，或清空安排建立新草稿。' : '需求已凍結，可以產生方案，也可以手動開始安排。'}</span>${p.draft ? btn('revalidate-draft', '檢查並沿用草稿') : ''}${btn('blank-draft', '建立空白草稿')}</div>` : ''}
    <section class="stats" aria-label="目前班表指標">${stats.map(([value, label, cls]) => `<div class="stat ${cls}"><span>${label}</span><strong>${value}</strong></div>`).join('')}</section>
    <section class="board-card">
      <div class="board-top"><div><div class="section-eyebrow">THE WEEK AHEAD</div><h2>${esc(p.name)} ${badge(version ? '已發布 v' + version.number : candidate ? '方案預覽' : '編輯中草稿', version ? 'green' : 'amber')}</h2></div><div class="toolbar">${btn('undo', icon('undo'), 'icon-button', 'aria-label="復原" title="復原" ' + (!p.history.length || candidate || version || !fresh ? 'disabled' : ''))}${btn('redo', icon('redo'), 'icon-button', 'aria-label="重做" title="重做" ' + (!p.future.length || candidate || version || !fresh ? 'disabled' : ''))}${btn('export-csv', icon('download') + '匯出', 'secondary')}${btn('publish', '審核發布' + icon('arrow'), 'primary', job ? 'disabled' : '')}</div></div>
      <div class="board-controls"><div class="segmented" role="group" aria-label="班表視圖">${btn('coverage', icon('calendar') + '班次視圖', view === 'coverage' ? 'selected' : '')}${btn('people', icon('people') + '人員視圖', view === 'people' ? 'selected' : '')}</div><div class="solver-controls"><label class="sr-only" for="solve-mode">重新排班方式</label><select id="solve-mode" ${job ? 'disabled' : ''}><option value="balance">保留鎖定，重新平衡</option><option value="fill">只補空缺，保留現有安排</option><option value="new">全新方案（仍保留鎖定）</option></select>${btn(job ? 'stop' : 'solve', icon(job ? 'close' : 'spark') + (job ? '取消運算' : '產生方案'), 'secondary', !fresh && !job ? 'disabled' : '')}</div></div>
      ${job ? '<div class="solve-status" role="status"><span class="spinner"></span>正在尋找符合規則的方案，最多運算 3 秒…</div>' : p.lastRun ? `<div class="run-summary">${esc(runSummary(p.lastRun))}</div>` : ''}
      ${p.candidates.length ? `<div class="candidate-strip"><span>候選方案<br><small>點選先預覽</small></span>${p.candidates.map((c, i) => `<button data-action="candidate" data-id="${esc(c.id)}" class="candidate-option ${candidateId === c.id ? 'selected' : ''}"><strong>方案 ${'ABC'[i]}</strong><span>偏好 ${c.metrics.preferred} 次 · 工時差 ${c.metrics.spread}h</span></button>`).join('')}${btn('show-draft', '回到草稿', 'text-button')}</div>` : ''}
      ${candidate ? `<div class="preview-banner"><span><strong>預覽中 · ${diff.length} 個班次將變更</strong> 套用後會建立可復原的草稿修改。</span><label><input type="checkbox" id="diff-only" ${diffOnly ? 'checked' : ''}>只看差異</label>${btn('apply-candidate', '套用此方案', 'primary')}</div>` : ''}
      ${version ? `<div class="preview-banner"><span>正在查看已發布 v${version.number}，此版本保持不變。</span>${btn('show-draft', '回到可編輯草稿')}</div>` : ''}
      <div class="canvas-layout"><div class="schedule-scroll">${view === 'coverage' ? coverageTable(snapshot, assignments, diff, candidate) : peopleTable(snapshot, assignments)}</div>${renderInspector()}</div>
      <div class="board-bottom"><div class="legend"><span><i class="legend-am"></i>早班／第一班</span><span><i class="legend-pm"></i>晚班／其他班</span><span>★ 偏好</span><span>${icon('lock')}人工鎖定</span></div><span>${latest && !version ? '較已發布 v' + latest.number + '：' + publicationDiff.length + ' 班變更' : '點選班次可查看人選與調整'}</span></div>
    </section>
    <div class="under-board"><section class="panel"><h3>發布紀錄 <span>${p.versions.length} 個版本</span></h3>${p.versions.length ? p.versions.slice().reverse().map(v => `<button class="version-row" data-action="version" data-number="${v.number}"><span>${icon('check')}已發布 v${v.number}<small>${new Date(v.at).toLocaleString('zh-TW')}</small></span><span>查看版本 →</span></button>`).join('') : '<p class="muted">草稿確認後，發布的版本會保存在這裡。</p>'}<p class="fine-print">本機發布不會發送通知；可匯出班表分享給成員。</p></section><section class="panel"><h3>最近動態</h3>${p.audit.slice(-3).reverse().map(a => `<div class="activity"><span class="activity-dot"></span><p>${esc(a.message)}<small>${new Date(a.at).toLocaleString('zh-TW')}</small></p></div>`).join('')}</section></div>`;
}
function coverageTable(snapshot, assignments, diff, candidate) {
  const p = period(), firstShift = p.shifts[0].id;
  return `<table class="schedule"><caption class="sr-only">班次覆蓋班表，點選格子編輯</caption><thead><tr><th scope="col" class="row-label">班次 / 日期</th>${Array.from({ length: snapshot.days }, (_, d) => `<th scope="col"><span>${'日一二三四五六'[new Date(dateAdd(snapshot.start, d) + 'T00:00:00Z').getUTCDay()]}</span><strong>${dateAdd(snapshot.start, d).slice(5).replace('-', '/')}</strong></th>`).join('')}</tr></thead><tbody>${[...new Set(snapshot.slots.map(s => s.shiftId))].map(shiftId => {
    const shift = snapshot.slots.find(s => s.shiftId === shiftId);
    return `<tr><th scope="row" class="row-label"><i class="shift-dot ${shiftId === firstShift ? 'am' : 'pm'}"></i><strong>${esc(shift.name)}</strong><small>${esc(shift.start)}<br>— ${esc(shift.end)}</small></th>${Array.from({ length: snapshot.days }, (_, d) => {
      const s = snapshot.slots.find(slot => slot.day === d && slot.shiftId === shiftId);
      const ids = assignments[s.id] || [], changed = diff.some(item => item.slot.id === s.id);
      return `<td><button class="shift-cell ${shiftId === firstShift ? 'am' : 'pm'} ${selectedSlot === s.id ? 'focused' : ''} ${ids.length < s.required ? 'has-gap' : ''} ${candidate && diffOnly && !changed ? 'dimmed' : ''}" data-action="slot" data-id="${esc(s.id)}" aria-label="${esc(dayLabel(s.date) + ' ' + s.name + '，' + names(snapshot, ids) + '，已排 ' + ids.length + ' 人，共需 ' + s.required + ' 人')}"><span class="cell-count">${ids.length}/${s.required} 人 ${period().draft?.locks[s.id]?.length ? icon('lock') : ''}</span>${ids.map(id => `<span class="person-chip"><span class="avatar-mini">${esc(snapshot.members.find(m => m.id === id)?.name.slice(-2))}</span>${esc(snapshot.members.find(m => m.id === id)?.name)}${snapshot.availability[id]?.[s.id] === 'prefer' ? '<span class="star" title="偏好此時段">★</span>' : ''}</span>`).join('')}${ids.length < s.required ? `<span class="gap-label">＋ 缺 ${s.required - ids.length} 人</span>` : !s.required ? '<span class="muted">無需求</span>' : ''}${candidate && changed ? '<span class="change-label">安排有變更</span>' : ''}</button></td>`;
    }).join('')}</tr>`;
  }).join('')}</tbody></table><div class="mobile-agenda">${Array.from({ length: snapshot.days }, (_, day) => `<section><h3>${dayLabel(dateAdd(snapshot.start, day))}</h3>${snapshot.slots.filter(s => s.day === day).map(s => `<button data-action="slot" data-id="${s.id}" class="agenda-shift"><span><strong>${esc(s.name)}</strong><small>${s.start}–${s.end}</small></span><span>${esc(names(snapshot, assignments[s.id] || []))}<small>${(assignments[s.id] || []).length}/${s.required} 人</small></span></button>`).join('')}</section>`).join('')}</div>`;
}
function peopleTable(snapshot, assignments) {
  const score = metrics(snapshot, assignments);
  return `<table class="schedule people-schedule"><caption class="sr-only">成員每日班次與本期工時</caption><thead><tr><th scope="col">成員</th>${Array.from({ length: snapshot.days }, (_, d) => '<th scope="col">' + dayLabel(dateAdd(snapshot.start, d)) + '</th>').join('')}<th scope="col">總計</th></tr></thead><tbody>${snapshot.members.map(m => `<tr><th scope="row">${esc(m.name)}</th>${Array.from({ length: snapshot.days }, (_, d) => {
    const work = snapshot.slots.filter(s => s.day === d && (assignments[s.id] || []).includes(m.id));
    return '<td>' + (work.length ? work.map(s => btn('slot', esc(s.name), 'mini-shift', 'data-id="' + s.id + '"')).join('') : '<span class="off-day">休</span>') + '</td>';
  }).join('')}<td><strong>${score.loads.find(l => l.id === m.id).hours}h</strong><small>${score.loads.find(l => l.id === m.id).count} 班</small></td></tr>`).join('')}</tbody></table><div class="mobile-agenda">${score.loads.map(load => `<section><h3>${esc(load.name)} · ${load.hours}h</h3>${snapshot.slots.filter(s => assignments[s.id]?.includes(load.id)).map(s => btn('slot', dayLabel(s.date) + ' · ' + esc(s.name) + ' ' + s.start + '–' + s.end, 'agenda-shift', 'data-id="' + s.id + '"')).join('') || '<p>本期沒有安排</p>'}</section>`).join('')}</div>`;
}
function renderInspector() {
  const p = period(), { snapshot, assignments, locks, readOnly } = selectedData();
  if (!selectedSlot || !snapshot.slots.some(s => s.id === selectedSlot)) selectedSlot = snapshot.slots[0]?.id;
  const slot = snapshot.slots.find(s => s.id === selectedSlot);
  if (!slot) return '';
  const assigned = assignments[slot.id] || [];
  const editable = !readOnly && currentSnapshot(p) && p.draft?.inputRevision === p.inputRevision && !job;
  return `<aside class="inspector" aria-label="班次檢查器"><div class="section-eyebrow">SHIFT DETAILS</div><h3>${dayLabel(slot.date)} · ${esc(slot.name)}</h3><p class="muted">${slot.start}–${slot.end} · 需要 ${slot.required} 人</p><div class="inspector-label">目前安排 <strong>${assigned.length} / ${slot.required}</strong></div>
    ${assigned.map(id => `<div class="assigned-person"><span class="avatar">${esc(snapshot.members.find(m => m.id === id)?.name.slice(-2))}</span><span>${esc(snapshot.members.find(m => m.id === id)?.name)}<small>${locks[slot.id]?.includes(id) ? '已鎖定 · 重排時保留' : '可重新安排'}</small></span>${btn('remove-assignment', '移除', 'text-button', `data-member="${esc(id)}" ${editable ? '' : 'disabled'}`)}</div>`).join('') || '<p class="muted">還沒有人安排在這一班。</p>'}
    ${btn('toggle-lock', icon('lock') + (locks[slot.id]?.length ? '解除此班鎖定' : '鎖定此班安排'), 'full-width secondary', !editable || !assigned.length ? 'disabled' : '')}
    <div class="inspector-divider"></div><div class="inspector-label">可安排人選<small>依本期規則檢查</small></div>
    ${snapshot.members.filter(m => !assigned.includes(m.id)).map(m => {
      const reasons = assignmentReasons(snapshot, assignments, slot.id, m.id);
      const full = assigned.length >= slot.required;
      const count = snapshot.slots.filter(s => assignments[s.id]?.includes(m.id)).length;
      return `<div class="eligible-person ${reasons.length ? 'unavailable' : ''}"><div><strong>${esc(m.name)}</strong><small>${reasons.length ? esc(reasons.join('；')) : `可排 · 已排 ${count}/${m.maxShifts} 班${snapshot.availability[m.id][slot.id] === 'prefer' ? ' · ★ 偏好' : ''}`}</small></div>${btn('add-assignment', icon('plus'), 'icon-button', `aria-label="安排${esc(m.name)}" data-member="${esc(m.id)}" ${!editable || full || reasons.length ? 'disabled' : ''}`)}</div>`;
    }).join('')}
    <p class="fine-print">${readOnly ? '目前為預覽；回到草稿或套用方案後可編輯。' : !editable ? '先凍結最新需求並採用新草稿，才能修改安排。' : '移除原人員再新增即可換班。手動新增會鎖定，所有修改可復原。'}</p></aside>`;
}
function renderInputs() {
  const p = period(), g = group(), slots = slotsFor(p);
  return `<section class="panel input-panel"><div class="section-heading"><div><h2>本期需求回報</h2><p class="muted">未知、可以、偏好、不可排，各自有清楚的意義。</p></div>${btn('review', '檢查與凍結' + icon('arrow'), 'primary')}</div><div class="submission-list">${g.members.map((m, i) => {
    const input = p.inputs[m.id], known = slots.filter(s => input?.cells[s.id] && input.cells[s.id] !== 'unknown').length;
    return `<div class="submission"><span class="avatar color-${i % 4}">${esc(m.name.slice(-2))}</span><div><strong>${esc(m.name)}</strong><small>${known} / ${slots.length} 個時段已回答 · 每期最多 ${m.maxShifts} 班</small></div>${badge(input?.confirmed ? '已確認' : '待提交', input?.confirmed ? 'green' : 'amber')}${btn('edit-input', input?.confirmed ? '查看／修改' : '填寫需求', 'secondary', 'data-member="' + m.id + '"')}</div>`;
  }).join('') || '<p>請先在團隊頁新增成員。</p>'}</div><div class="notice"><span>自由文字會先產生規則預覽，確認後才套用。本版使用有限格式辨識，沒有把文字送往 AI。</span></div></section>`;
}
function renderTeam() {
  return `<section class="panel"><div class="section-heading"><div><h2>團隊成員 <span class="muted">${group().members.length} / 20</span></h2><p class="muted">上限為班數，工時會依班次長度另外計算。</p></div>${btn('new-member', icon('plus') + '新增成員', 'primary')}</div><div class="team-grid">${group().members.map((m, i) => `<article class="member-card"><span class="avatar large color-${i % 4}">${esc(m.name.slice(-2))}</span><h3>${esc(m.name)}</h3><p>每期最多 <strong>${m.maxShifts}</strong> 班</p>${btn('edit-member', '編輯設定', 'secondary', 'data-member="' + m.id + '"')}</article>`).join('')}</div><p class="fine-print">本機版沒有登入與權限隔離；請勿把同一工作空間當作員工私人帳號使用。</p></section>`;
}
function renderSettings() {
  const p = period();
  return `<div class="settings-grid"><section class="panel"><h2>本期排班規則</h2>${p ? `<dl class="rule-list"><div><dt>每人每日最多</dt><dd>${p.rules.dailyMax} 班</dd></div><div><dt>班次間最少休息</dt><dd>${p.rules.restHours} 小時</dd></div><div><dt>最多連續工作</dt><dd>${p.rules.consecutiveMax} 天</dd></div><div><dt>需求截止時間</dt><dd>${esc(p.deadline.replace('T', ' ') || '未設定')}</dd></div></dl>${btn('edit-rules', '編輯規則與班別', 'secondary')}` : '<p>建立班表後即可設定。</p>'}<p class="fine-print">目前僅檢查本期資料；跨期班距、工時法規與技能資格尚未支援。這些設定不代表法規驗證。</p></section><section class="panel"><h2>資料由你掌握</h2><p class="muted">使用本機儲存，不需 API 費用。清除瀏覽器資料會移除班表，請保留 JSON 備份。</p><div class="stack-actions">${btn('backup', icon('download') + '下載完整 JSON 備份', 'primary')}${btn('import', '從 v2 備份還原', 'secondary')}${btn('print', '列印目前班表', 'secondary')}</div><p class="fine-print">還原前會驗證檔案，並要求先下載現有資料。舊版 v1 備份不會自動覆蓋。</p></section><section class="panel wide"><h2>目前可用與接下來要做的</h2><div class="capabilities"><div><h3>本機版已支援</h3><p>需求收集、短時求解、候選比較、雙視圖、編輯驗證、鎖定、復原／重做、版本發布、CSV 與 JSON 匯出。</p></div><div><h3>商用版本仍待完成</h3><p>成員登入與邀請、雲端同步、真正 AI 理解、截止提醒、訂閱付款，以及 Windows／App Store 打包與審核。</p></div></div></section></div>`;
}
function runSummary(run) {
  const count = run.candidates?.length ?? run.count ?? 0;
  if (run.status === 'error') return '排班器發生錯誤，草稿未改動。請調整後重試。';
  if (run.reason === 'cancelled') return '已取消；保留 ' + count + ' 個已驗證方案，草稿未改動。';
  if (run.status === 'infeasible') return '在目前規則下沒有完整解。' + (run.issues?.[0]?.message || '請檢查班數、可排時間與班距設定。');
  if (!count) return '已達搜尋上限，尚未找到完整方案；這不代表無解。可先手動鎖定部分安排再試。';
  return '找到 ' + count + ' 個完整可行方案 · 未證明最佳 · ' + Math.round(run.elapsedMs || 0) + ' ms · 套用前不會變動草稿';
}
function openDialog(title, body, cls = '') {
  $('#dialog').className = cls;
  $('#dialog').innerHTML = `<div class="dialog-heading"><h2 id="dialog-title">${title}</h2>${btn('close-dialog', icon('close'), 'icon-button', 'aria-label="關閉"')}</div>${body}`;
  $('#dialog').showModal();
}
function closeDialog() { $('#dialog').close(); }
function field(label, input) { return `<label class="form-field"><span>${label}</span>${input}</label>`; }
function formFooter(label) { return `<div class="dialog-actions">${btn('close-dialog', '取消', 'secondary')}<button type="submit" class="primary">${label}</button></div>`; }
function openPeriodForm(edit = false) {
  const p = period(), base = p || { name: '', start: new Date().toLocaleDateString('en-CA'), days: 7, deadline: '', rules: { dailyMax: 1, restHours: 11, consecutiveMax: 5 }, shifts: [{ id: 'am', name: '早班', start: '09:00', end: '13:00', required: 1 }, { id: 'pm', name: '晚班', start: '14:00', end: '18:00', required: 1 }] };
  openDialog(edit ? '編輯本期規則' : '開始新一期班表', `<form id="period-form" data-edit="${edit}"><div class="form-grid">${field('期次名稱', `<input name="name" value="${esc(edit ? base.name : '')}" placeholder="例如：10 月第一週" maxlength="80" required>`)}${field('起始日期', `<input name="start" type="date" value="${edit ? base.start : p ? dateAdd(p.start, p.days) : base.start}" ${edit ? 'readonly' : ''} required>`)}${field('天數（1–14）', `<input name="days" type="number" value="${base.days}" min="1" max="14" ${edit ? 'readonly' : ''} required>`)}${field('截止時間（台北，本機記錄）', `<input name="deadline" type="datetime-local" value="${edit ? base.deadline : ''}">`)}</div><h3>班別與每日需求</h3><div class="shift-form-header"><span>班別</span><span>開始</span><span>結束</span><span>人數</span></div>${base.shifts.map((s, i) => `<div class="shift-form-row" data-shift="${esc(s.id)}"><input name="shiftName${i}" aria-label="班別 ${i + 1} 名稱" value="${esc(s.name)}" maxlength="20" required><input name="shiftStart${i}" aria-label="班別 ${i + 1} 開始" type="time" value="${s.start}" required><input name="shiftEnd${i}" aria-label="班別 ${i + 1} 結束" type="time" value="${s.end}" required><input name="shiftRequired${i}" aria-label="班別 ${i + 1} 人數" type="number" min="0" max="20" value="${s.required}" required></div>`).join('')}<p class="fine-print">每日使用相同需求；0 人代表不開此班。結束早於開始視為跨午夜。</p><h3>安排限制</h3><div class="form-grid three">${field('每日最多班數', `<input name="dailyMax" type="number" min="1" max="4" value="${base.rules.dailyMax}" required>`)}${field('最少班距（小時）', `<input name="restHours" type="number" min="0" max="24" value="${base.rules.restHours}" required>`)}${field('最多連續工作天', `<input name="consecutiveMax" type="number" min="1" max="14" value="${base.rules.consecutiveMax}" required>`)}</div><p class="fine-print">${edit ? '儲存會使凍結資料過期。已發布版本保持不變。' : '新期次的需求預設未填，每位成員需重新確認。'}</p>${formFooter(edit ? '儲存並重新檢查' : '建立期次')}</form>`, 'wide-dialog');
}
function openInput(memberId) {
  const p = period(), m = group().members.find(m => m.id === memberId);
  const input = p.inputs[m.id] || { cells: {}, note: '', confirmed: false };
  const labels = { unknown: '？未填', yes: '可以排', prefer: '★ 希望排', no: '不可排' };
  openDialog(esc(m.name) + ' · 本期需求', `<form id="input-form" data-member="${m.id}"><p class="muted">未填的時段不會被安排。班別偏好是加分，並不保證一定排到。</p><div class="input-bulk">${btn('fill-yes', '全部設為可以排', 'secondary')}${btn('fill-no', '全部設為不可排', 'secondary')}${btn('fill-unknown', '全部設為未填', 'text-button')}</div><div class="availability-days">${Array.from({ length: p.days }, (_, day) => `<div class="availability-day"><strong>${dayLabel(dateAdd(p.start, day))}</strong>${slotsFor(p).filter(s => s.day === day).map(s => `<label><span>${esc(s.name)} <small>${s.start}–${s.end}</small></span><select name="slot:${s.id}" aria-label="${esc(dayLabel(s.date) + ' ' + s.name)}">${Object.entries(labels).map(([key, label]) => `<option value="${key}" ${(input.cells[s.id] || 'unknown') === key ? 'selected' : ''}>${label}</option>`).join('')}</select></label>`).join('')}</div>`).join('')}</div><label class="form-field"><span>補充需求（不需填私人原因）</span><textarea id="input-note" name="note" maxlength="2000" rows="3" placeholder="例如：週三 晚班 不可排&#10;週五 早班 希望排">${esc(input.note)}</textarea></label><div class="note-tools">${btn('parse-note', icon('spark') + '預覽文字規則', 'secondary')}<small>有限格式辨識 · 不是 AI</small></div><div id="note-preview"></div><label class="checkbox-line"><input type="checkbox" id="note-ack" ${input.confirmed ? 'checked' : ''}>我已確認補充文字中的要求，都已反映在上方時段中。</label><div class="dialog-actions">${btn('save-input-draft', '只儲存，稍後確認', 'secondary')}<button type="submit" class="primary">確認本期需求</button></div></form>`, 'wide-dialog');
}
function openReview() {
  const p = period(), g = group(), snapshot = snapshotFor(g, p);
  const missing = g.members.filter(m => !p.inputs[m.id]?.confirmed);
  const issues = preflight(snapshot);
  const unknown = Object.values(snapshot.availability).flatMap(c => Object.values(c)).filter(v => v === 'unknown').length;
  openDialog('確認這次排班的依據', `<p>凍結後，排班器會使用目前第 <strong>${p.inputRevision}</strong> 版需求。後續修改會要求再次檢查。</p><div class="review-stats">${badge(g.members.length - missing.length + '/' + g.members.length + ' 位已確認', 'green')}${badge(unknown + ' 個時段未知', unknown ? 'amber' : 'neutral')}</div>${missing.length ? `<div class="notice warning">尚未提交：${esc(missing.map(m => m.name).join('、'))}。請先填寫並確認需求。</div>` : '<div class="notice">全部成員已確認。未知時段仍會視為不可安排。</div>'}<h3>排班前檢查</h3>${issues.length ? '<ul class="issue-list">' + issues.map(i => '<li>' + esc(i.message) + '</li>').join('') + '</ul>' : '<p>初步人數與可排時間檢查通過；能否完成仍需實際求解。</p>'}<p class="fine-print">規則：每天最多 ${p.rules.dailyMax} 班、最少班距 ${p.rules.restHours} 小時、最多連上 ${p.rules.consecutiveMax} 天。僅檢查本期。</p><div class="dialog-actions">${btn('go-inputs', '回到需求收集', 'secondary')}${btn('freeze', '確認並凍結資料', 'primary', missing.length || !g.members.length ? 'disabled' : '')}</div>`, 'wide-dialog');
}
function openPublish() {
  const p = period();
  const blockers = publishBlockers(p);
  const last = p.versions.at(-1);
  const diffs = p.frozen && p.draft ? difference(p.frozen, last?.assignments || {}, p.draft.assignments) : [];
  openDialog('審核並發布 v' + (p.versions.length + 1), `<p>發布的是目前草稿；每個版本會獨立保存，之後修改不會覆蓋它。</p>${blockers.length ? '<div class="notice warning"><strong>目前還不能發布</strong></div><ul class="issue-list">' + blockers.map(i => '<li>' + esc(i) + '</li>').join('') + '</ul>' : '<div class="notice success">全部人力已填滿，已啟用的本期規則檢查通過。</div>'}<h3>${last ? '相較 v' + last.number : '首次發布'} · ${diffs.length} 個班次變更</h3><div class="diff-list">${diffs.map(d => `<div><strong>${dayLabel(d.slot.date)} ${esc(d.slot.name)}</strong><span>${esc(names(p.frozen, d.before))} → ${esc(names(p.frozen, d.after))}</span></div>`).join('') || '<p>班次安排沒有變更。</p>'}</div><p class="fine-print">發布只保存在本機，不會通知成員；你可以匯出 CSV 分享。</p><div class="dialog-actions">${btn('close-dialog', '繼續編輯', 'secondary')}${btn('confirm-publish', '確認發布到本機', 'primary', blockers.length || (last && !diffs.length && last.snapshot.revision === p.inputRevision) ? 'disabled' : '')}</div>`, 'wide-dialog');
}
function openMember(memberId) {
  const member = group().members.find(m => m.id === memberId);
  openDialog(member ? '編輯成員' : '新增團隊成員', `<form id="member-form" data-member="${member?.id || ''}">${field('姓名／顯示名稱', `<input name="name" maxlength="40" value="${esc(member?.name || '')}" required>`)}${field('每期最多班數（0–56）', `<input name="maxShifts" type="number" min="0" max="56" value="${member?.maxShifts ?? 5}" required>`)}<p class="fine-print">成員變更後，進行中的期次需重新凍結資料。0 班表示本期不安排。</p>${formFooter('儲存成員')}</form>`);
}
function download(name, content, type) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const link = document.createElement('a'); link.href = url; link.download = name; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 3000);
}
function backup() { download('班伴備份-' + new Date().toISOString().slice(0, 10) + '.json', JSON.stringify(state(), null, 2), 'application/json'); }
function collectInput(confirmed) {
  const form = $('#input-form'), p = period();
  const note = form.elements.note.value.trim();
  if (confirmed && note && !$('#note-ack').checked) throw new Error('請先確認補充文字已反映在時段中，或預覽並套用規則');
  const cells = Object.fromEntries(slotsFor(p).map(s => [s.id, form.elements['slot:' + s.id].value]));
  if (updatePeriod(p => saveInput(p, form.dataset.member, { cells, note, confirmed }), confirmed ? '需求已在本機確認；需重新凍結後才用於排班' : '已儲存本機需求草稿')) closeDialog();
}
function startSolve() {
  const p = period();
  if (!currentSnapshot(p) || job) return;
  const mode = $('#solve-mode').value;
  const locks = clone(p.draft?.locks || {});
  if (mode === 'fill') for (const [id, ids] of Object.entries(p.draft?.assignments || {})) locks[id] = [...new Set([...(locks[id] || []), ...ids])];
  const issues = validateAssignments(p.frozen, locks);
  if (issues.length) { toast('現有鎖定安排不符合最新規則：' + issues[0].message + '。請建立空白草稿或回復規則後重試。', true); return; }
  const worker = new Worker(new URL('./solver-worker.mjs', import.meta.url), { type: 'module' });
  const run = { worker, gid: group().id, pid: p.id, revision: p.inputRevision, candidates: [], started: performance.now(), timer: null, locks };
  job = run; resetSelection();
  worker.onmessage = ({ data }) => {
    if (job !== run) return;
    if (data.type === 'candidate') run.candidates.push(data.candidate);
    if (data.type === 'done') finishSolve(run, { ...data.result, elapsedMs: data.elapsedMs });
    if (data.type === 'error') finishSolve(run, { status: 'error', reason: 'model_error', candidates: [], message: data.message });
  };
  worker.onerror = () => finishSolve(run, { status: 'error', reason: 'worker_error', candidates: [] });
  run.timer = setTimeout(() => finishSolve(run, { status: run.candidates.length ? 'feasible' : 'limited', reason: 'time_limit', candidates: run.candidates, elapsedMs: performance.now() - run.started }), 3500);
  worker.postMessage({ snapshot: clone(p.frozen), maxMs: 2500, options: { locks, baseline: mode === 'balance' ? p.draft?.assignments : undefined, maxCandidates: 3 } });
  render();
}
function finishSolve(run, result) {
  if (job !== run) return;
  clearTimeout(run.timer); run.worker.terminate(); job = null;
  mutate(s => {
    const p = s.groups.find(g => g.id === run.gid)?.periods.find(p => p.id === run.pid);
    if (!p || p.inputRevision !== run.revision || !currentSnapshot(p)) throw new Error('需求已變更，已捨棄舊資料的求解結果');
    for (const c of result.candidates) if (validateAssignments(p.frozen, c.assignments, { complete: true, locks: run.locks }).length) throw new Error('候選驗證失敗，未套用任何結果');
    p.candidates = clone(result.candidates);
    p.lastRun = { ...result, count: result.candidates.length, candidates: undefined };
    audit(p, '完成排班搜尋：' + result.candidates.length + ' 個方案；草稿未改動');
  }, runSummary(result));
}
function stopSolve(reason = 'cancelled') {
  if (!job) return;
  finishSolve(job, { status: job.candidates.length ? 'feasible' : 'limited', reason, candidates: job.candidates, elapsedMs: performance.now() - job.started });
}

document.addEventListener('click', event => {
  const button = event.target.closest('[data-action]');
  if (!button || button.disabled) return;
  const action = button.dataset.action;
  try {
    if (action === 'close-dialog') { closeDialog(); return; }
    if (action === 'raw-export') { download('班伴原始資料.json', stored.raw || '', 'application/json'); return; }
    if (action === 'nav') { tab = button.dataset.tab; render(); return; }
    if (action === 'step') {
      const target = button.dataset.target;
      if (target === 'review') openReview();
      else if (target === 'publish') openPublish();
      else { tab = target; render(); }
      return;
    }
    if (action === 'new-group') openDialog('建立你的工作空間', `<form id="group-form">${field('群組名称', '<input name="name" required maxlength="80" placeholder="例如：日和咖啡 中山店">')}<p class="muted">建立後可以新增成員與第一期班表。</p>${formFooter('建立群組')}</form>`);
    else if (action === 'blank-draft') {
      if (period().draft) openDialog('建立空白草稿', '<p>目前安排與鎖定將清空；已發布版本不變，也可以用復原找回這份草稿。</p><div class="dialog-actions">' + btn('close-dialog', '取消', 'secondary') + btn('confirm-blank-draft', '建立空白草稿', 'primary') + '</div>');
      else updatePeriod(p => changeDraft(p, { assignments: emptyAssignments(p.frozen), locks: {} }, '建立手動編輯草稿'), '可以點選班次，開始手動安排');
    }
    else if (action === 'confirm-blank-draft') { if (updatePeriod(p => changeDraft(p, { assignments: emptyAssignments(p.frozen), locks: {} }, '建立空白草稿'), '空白草稿已建立')) closeDialog(); }
    else if (action === 'revalidate-draft') updatePeriod(p => changeDraft(p, p.draft, '依最新需求重新驗證並沿用草稿'), '現有安排已通過最新需求檢查');
    else if (action === 'new-period') openPeriodForm();
    else if (action === 'edit-rules') openPeriodForm(true);
    else if (action === 'new-member' || action === 'edit-member') openMember(button.dataset.member);
    else if (action === 'edit-input') openInput(button.dataset.member);
    else if (action === 'review') openReview();
    else if (action === 'go-inputs') { closeDialog(); tab = 'inputs'; render(); }
    else if (action === 'freeze') { if (updatePeriod(p => freezeInputs(group(), p), '需求已凍結，可以產生方案')) { closeDialog(); tab = 'board'; resetSelection(); render(); } }
    else if (action === 'coverage' || action === 'people') { view = action; render(); }
    else if (action === 'slot') { selectedSlot = button.dataset.id; render(); if (matchMedia('(max-width: 800px)').matches) $('.inspector')?.scrollIntoView({ behavior: 'smooth', block: 'start' }); else $('.shift-cell.focused')?.focus({ preventScroll: true }); }
    else if (action === 'candidate') { candidateId = button.dataset.id; versionNumber = 0; render(); }
    else if (action === 'show-draft') { candidateId = null; versionNumber = 0; diffOnly = false; render(); }
    else if (action === 'version') { versionNumber = Number(button.dataset.number); candidateId = null; render(); }
    else if (action === 'solve') startSolve();
    else if (action === 'stop') stopSolve();
    else if (action === 'apply-candidate') {
      const c = period().candidates.find(c => c.id === candidateId);
      if (!c) return;
      if (updatePeriod(p => changeDraft(p, { assignments: c.assignments, locks: p.draft?.locks || {} }, '套用候選方案為草稿'), '方案已套用，可使用復原')) { candidateId = null; render(); }
    } else if (action === 'add-assignment' || action === 'remove-assignment' || action === 'toggle-lock') {
      if (selectedData().readOnly || job) throw new Error('請先回到草稿編輯');
      updatePeriod(p => {
        if (!p.draft || p.draft.inputRevision !== p.inputRevision) throw new Error('請先採用最新需求的草稿');
        const next = clone(p.draft), id = button.dataset.member;
        if (action === 'toggle-lock') next.locks[selectedSlot] = next.locks[selectedSlot]?.length ? [] : [...next.assignments[selectedSlot]];
        if (action === 'add-assignment') {
          next.assignments[selectedSlot] = [...(next.assignments[selectedSlot] || []), id];
          next.locks[selectedSlot] = [...(next.locks[selectedSlot] || []), id];
        }
        if (action === 'remove-assignment') {
          next.assignments[selectedSlot] = (next.assignments[selectedSlot] || []).filter(m => m !== id);
          next.locks[selectedSlot] = (next.locks[selectedSlot] || []).filter(m => m !== id);
        }
        changeDraft(p, next, action === 'toggle-lock' ? '更新班次鎖定' : '手動調整班次人員');
      }, '草稿已儲存');
    } else if (action === 'undo' || action === 'redo') updatePeriod(p => travelHistory(p, action), action === 'undo' ? '已復原' : '已重做');
    else if (action === 'publish') openPublish();
    else if (action === 'confirm-publish') { if (updatePeriod(p => publish(p), '已發布本機版本；可匯出班表分享')) closeDialog(); }
    else if (action === 'export-csv') { const d = selectedData(); download(period().name + (d.version ? '-v' + d.version.number : '-草稿') + '.csv', toCSV(d.snapshot, d.assignments), 'text/csv;charset=utf-8'); }
    else if (action === 'backup') backup();
    else if (action === 'print') { tab = 'board'; render(); window.print(); }
    else if (action === 'import') $('#import-file').click();
    else if (action === 'confirm-import') {
      if (!pendingImport || !$('#backup-confirm').checked) throw new Error('請先下載現有備份並勾選確認');
      stopSolve();
      if (mutate(s => { const revision = s.revision; for (const key of Object.keys(s)) delete s[key]; Object.assign(s, clone(pendingImport), { revision }); }, '已還原備份')) { pendingImport = null; closeDialog(); resetSelection(); tab = 'board'; render(); }
    } else if (action.startsWith('fill-')) {
      const value = action.slice(5);
      $('#input-form').querySelectorAll('select[name^="slot:"]').forEach(select => { select.value = value; });
    } else if (action === 'save-input-draft') collectInput(false);
    else if (action === 'parse-note') {
      const result = parseNotes($('#input-note').value, period());
      $('#note-preview').innerHTML = `<div class="note-preview"><h3>規則預覽</h3>${result.rules.map(r => '<p>✓ ' + esc(r.source) + ' → ' + r.slotIds.length + ' 個時段</p>').join('')}${result.unresolved.map(line => '<p class="text-warning">無法完整理解：' + esc(line) + '。請直接調整上方格子。</p>').join('')}${!result.rules.length && !result.unresolved.length ? '<p>請先輸入補充需求。</p>' : ''}${result.rules.length ? btn('apply-note', '確認並套用這些規則', 'secondary') : ''}</div>`;
    } else if (action === 'apply-note') {
      const result = parseNotes($('#input-note').value, period());
      for (const rule of result.rules) for (const id of rule.slotIds) $('#input-form').elements['slot:' + id].value = rule.value;
      $('#note-ack').checked = result.unresolved.length === 0;
      $('#note-preview').textContent = '已套用 ' + result.rules.length + ' 條規則。' + (result.unresolved.length ? '仍有未理解的文字，請手動核對。' : '請確認時段後提交。');
    }
  } catch (e) { toast(e.message, true); }
});

document.addEventListener('submit', event => {
  event.preventDefault();
  const form = event.target, data = new FormData(form);
  try {
    if (form.id === 'input-form') { collectInput(true); return; }
    if (form.id === 'group-form') {
      const name = data.get('name').trim(); if (!name) throw new Error('請填寫群組名稱');
      if (state().groups.length >= 20) throw new Error('本機最多 20 個群組');
      const g = { id: uid(), name, members: [], periods: [], demo: false };
      if (mutate(s => { s.groups.push(g); s.activeGroupId = g.id; s.activePeriodId = null; }, '工作空間已建立')) { closeDialog(); resetSelection(); tab = 'team'; render(); }
    }
    if (form.id === 'member-form') {
      const name = data.get('name').trim(), maxShifts = Number(data.get('maxShifts')), gid = group().id, id = form.dataset.member;
      if (!name || !Number.isInteger(maxShifts) || maxShifts < 0 || maxShifts > 56) throw new Error('請確認姓名與班數');
      if (!id && group().members.length >= 20) throw new Error('本版每群組最多 20 人');
      if (mutate(s => {
        const g = s.groups.find(g => g.id === gid), existing = g.members.find(m => m.id === id);
        if (existing) Object.assign(existing, { name, maxShifts }); else g.members.push({ id: uid(), name, maxShifts });
        for (const p of g.periods) { p.inputRevision++; p.candidates = []; audit(p, '成員設定變更，需重新凍結需求'); }
      }, '成員已儲存')) closeDialog();
    }
    if (form.id === 'period-form') {
      const edit = form.dataset.edit === 'true', gid = group().id;
      const shifts = [...form.querySelectorAll('[data-shift]')].map((row, i) => ({ id: row.dataset.shift, name: data.get('shiftName' + i).trim(), start: data.get('shiftStart' + i), end: data.get('shiftEnd' + i), required: Number(data.get('shiftRequired' + i)) }));
      const next = makePeriod({ name: data.get('name').trim(), start: data.get('start'), days: Number(data.get('days')), deadline: data.get('deadline'), shifts, rules: { dailyMax: Number(data.get('dailyMax')), restHours: Number(data.get('restHours')), consecutiveMax: Number(data.get('consecutiveMax')) } });
      if (edit) {
        if (updatePeriod(p => { p.name = next.name; p.deadline = next.deadline; p.shifts = next.shifts; p.rules = next.rules; p.inputRevision++; p.candidates = []; audit(p, '修改班別或規則，需重新凍結'); }, '規則已儲存，請重新凍結')) closeDialog();
      } else {
        if (group().periods.length >= 52) throw new Error('本機每群組最多保留 52 期');
        if (mutate(s => { s.groups.find(g => g.id === gid).periods.push(next); s.activePeriodId = next.id; }, '新期次已建立')) { closeDialog(); resetSelection(); tab = 'inputs'; render(); }
      }
    }
  } catch (e) { toast(e.message, true); }
});
document.addEventListener('change', event => {
  const target = event.target;
  if (target.id === 'group-select' || target.id === 'group-select-mobile' || target.id === 'period-select') {
    stopSolve(); resetSelection();
    mutate(s => { if (target.id.startsWith('group-select')) { s.activeGroupId = target.value; s.activePeriodId = s.groups.find(g => g.id === target.value).periods[0]?.id || null; } else s.activePeriodId = target.value; });
  }
  if (target.id === 'diff-only') { diffOnly = target.checked; render(); }
});
document.addEventListener('input', event => { if (event.target.id === 'input-note') $('#note-ack').checked = false; });
$('#import-file').addEventListener('change', async event => {
  const file = event.target.files[0]; event.target.value = '';
  if (!file) return;
  try {
    if (file.size > 4_000_000) throw new Error('備份超過 4 MB 上限');
    pendingImport = validateState(JSON.parse(await file.text()));
    openDialog('還原備份前，保留目前資料', `<p>備份包含 ${pendingImport.groups.length} 個群組。確認後將取代目前瀏覽器的 v2 工作空間。</p>${btn('backup', '先下載目前的完整備份', 'primary')}<label class="checkbox-line"><input type="checkbox" id="backup-confirm">我已保存目前資料，同意還原選擇的備份。</label><div class="dialog-actions">${btn('close-dialog', '取消', 'secondary')}${btn('confirm-import', '還原備份', 'primary')}</div>`);
  } catch (e) { pendingImport = null; toast('無法還原：' + e.message, true); }
});
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && !$('#dialog').open) { selectedSlot = null; return; }
  if (event.target.matches('.shift-cell') && ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)) {
    event.preventDefault();
    const cells = [...document.querySelectorAll('.shift-cell')], index = cells.indexOf(event.target);
    const delta = ({ ArrowLeft: -1, ArrowRight: 1, ArrowUp: -period().days, ArrowDown: period().days })[event.key];
    cells[Math.max(0, Math.min(cells.length - 1, index + delta))]?.focus();
  }
});
window.addEventListener('storage', event => {
  if (event.key !== STORAGE_KEY) return;
  if (job) { clearTimeout(job.timer); job.worker.terminate(); job = null; }
  toast('另一個分頁已更新班表。請重新載入；本分頁的舊資料不會覆蓋新資料。', true);
});
window.addEventListener('beforeunload', () => { job?.worker.terminate(); });
render();
if (stored.error) toast(stored.error, true);
if ('serviceWorker' in navigator) navigator.serviceWorker.register('./sw.js').catch(() => {});
