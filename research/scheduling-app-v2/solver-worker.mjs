import { search, SOLVER_VERSION } from './solver.mjs';

self.onmessage = ({ data }) => {
  const started = performance.now();
  const iterator = search(data.snapshot, data.options);
  const candidates = [];
  let nodes = 0;
  const maxMs = Math.min(5000, Math.max(100, data.maxMs || 2500));
  function pump() {
    try {
      const sliceStart = performance.now();
      while (performance.now() - sliceStart < 12) {
        if (performance.now() - started >= maxMs) {
          postMessage({ type: 'done', result: { status: candidates.length ? 'feasible' : 'limited', reason: 'time_limit', candidates, nodes, exhausted: false, issues: [], solverVersion: SOLVER_VERSION }, elapsedMs: performance.now() - started });
          return;
        }
        const step = iterator.next();
        if (step.done) { postMessage({ type: 'done', result: step.value, elapsedMs: performance.now() - started }); return; }
        nodes = step.value.nodes;
        if (step.value.type === 'candidate') candidates.push(step.value.candidate);
        postMessage(step.value);
      }
      setTimeout(pump, 8);
    } catch (error) { postMessage({ type: 'error', message: error.message }); }
  }
  pump();
};
