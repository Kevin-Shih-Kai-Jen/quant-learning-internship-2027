import { createDemoState, validateState, clone } from './domain.mjs';
export const STORAGE_KEY = 'banban-scheduling-v2';

export function createStore(storage) {
  let state;
  let baseline;
  let error = '';
  try {
    baseline = storage.getItem(STORAGE_KEY);
    state = baseline ? validateState(JSON.parse(baseline)) : createDemoState();
  } catch (e) {
    // Preserve unreadable data. A recovery screen must offer raw download.
    return { error: '無法讀取資料：' + e.message, raw: baseline, state: null };
  }
  function commit(action) {
    const actual = storage.getItem(STORAGE_KEY);
    if (actual !== baseline) throw new Error('另一個分頁已更新資料。請重新載入後再操作；本次修改尚未儲存。');
    const next = clone(state);
    action(next);
    next.revision = state.revision + 1;
    const encoded = JSON.stringify(next);
    if (encoded.length > 4_000_000) throw new Error('本機資料已接近容量上限，請先匯出備份');
    storage.setItem(STORAGE_KEY, encoded);
    baseline = encoded;
    state = next;
    return state;
  }
  if (!baseline) {
    try { commit(() => {}); } catch (e) { error = '無法保存到瀏覽器：' + e.message; }
  }
  return { get state() { return state; }, error, commit };
}
