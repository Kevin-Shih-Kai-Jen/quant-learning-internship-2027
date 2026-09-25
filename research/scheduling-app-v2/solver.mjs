import { clone, emptyAssignments, assignmentReasons, validateAssignments, preflight, metrics, difference } from './domain.mjs';

export const SOLVER_VERSION = 'bounded-dfs-2.0';

// A bounded generator runs in a dedicated worker. Only full, validated schedules
// become candidates; a time/node limit never proves infeasibility or optimality.
export function* search(snapshot, options = {}) {
  const maxCandidates = options.maxCandidates ?? 3;
  const maxNodes = options.maxNodes ?? 80000;
  const minDistance = options.minDistance ?? 2;
  const locks = clone(options.locks || {});
  const assignment = { ...emptyAssignments(snapshot), ...locks };
  const issues = preflight(snapshot, assignment);
  let nodes = 0, stopped = false;
  const candidates = [];
  const result = (reason, exhausted = false) => ({
    status: candidates.length ? 'feasible' : exhausted ? 'infeasible' : 'limited',
    reason, exhausted, nodes, candidates, issues, solverVersion: SOLVER_VERSION
  });
  if (issues.length) return result('preflight', true);

  function* combinations(ids, count, offset = 0, chosen = []) {
    if (count === 0) { yield chosen; return; }
    for (let i = offset; i <= ids.length - count; i++) yield* combinations(ids, count - 1, i + 1, [...chosen, ids[i]]);
  }
  function rank(member, slot) {
    const work = snapshot.slots.filter(s => assignment[s.id].includes(member.id));
    const load = work.reduce((n, s) => n + s.endMinute - s.startMinute, 0) / 60;
    const preference = snapshot.availability[member.id]?.[slot.id] === 'prefer' ? 1 : 0;
    const retained = options.baseline?.[slot.id]?.includes(member.id) ? 1 : 0;
    return load - preference * 4 - retained * 8;
  }
  function* visit() {
    if (stopped) return;
    if (nodes >= maxNodes) { stopped = true; return; }
    nodes++;
    if (nodes % 64 === 0) yield { type: 'progress', nodes, count: candidates.length };
    let next = null;
    for (const slot of snapshot.slots) {
      const need = slot.required - assignment[slot.id].length;
      if (!need) continue;
      const eligible = snapshot.members.filter(m => !assignment[slot.id].includes(m.id) && !assignmentReasons(snapshot, assignment, slot.id, m.id).length);
      if (eligible.length < need) return;
      const slack = eligible.length - need;
      if (!next || slack < next.slack) next = { slot, eligible, need, slack };
    }
    if (!next) {
      if (validateAssignments(snapshot, assignment, { complete: true, locks }).length) throw new Error('Solver returned an invalid assignment');
      if (candidates.every(c => difference(snapshot, c.assignments, assignment).length >= minDistance)) {
        const candidate = { id: 'candidate-' + (candidates.length + 1), assignments: clone(assignment), metrics: metrics(snapshot, assignment) };
        candidates.push(candidate);
        yield { type: 'candidate', candidate, nodes };
        if (candidates.length >= maxCandidates) stopped = true;
      }
      return;
    }
    const { slot, eligible, need } = next;
    eligible.sort((a, b) => rank(a, slot) - rank(b, slot) || a.id.localeCompare(b.id));
    const previous = assignment[slot.id];
    for (const ids of combinations(eligible.map(m => m.id), need)) {
      assignment[slot.id] = [...previous, ...ids];
      yield* visit();
      assignment[slot.id] = previous;
      if (stopped) break;
    }
  }
  yield* visit();
  return result(candidates.length >= maxCandidates ? 'candidate_limit' : stopped ? 'node_limit' : 'exhausted', !stopped);
}

export function solveSync(snapshot, options) {
  const iterator = search(snapshot, options);
  let step;
  do { step = iterator.next(); } while (!step.done);
  return step.value;
}
