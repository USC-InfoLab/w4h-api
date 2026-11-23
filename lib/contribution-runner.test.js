import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import http from 'node:http';
import { PassThrough } from 'node:stream';
import test, { afterEach } from 'node:test';
import {
  monitorImporterRun, delegatedRunUnavailable, reapOrphanedRuns, runTotals, sqlRunMeta, startSqlContributionRun,
} from './contribution-runner.js';
import { startDelegatedChild, loadDelegateModules, _resetDelegateCache } from './contribution-delegate.js';

function childProcess() {
  const child = new EventEmitter();
  child.stdout = new PassThrough();
  child.stderr = new PassThrough();
  return child;
}

function gate() {
  let release;
  const promise = new Promise((resolve) => { release = resolve; });
  return { promise, release };
}

const nextTurn = () => new Promise((resolve) => setImmediate(resolve));
const RUN_ROUTE = '/contributions/:slug/run';

function runStore({ failLog = 0, failProgress = 0, failFinal = 0, holdLog, holdProgress } = {}) {
  const row = { log: '', progress: null, status: 'running', exitCode: null };
  const writes = [];
  let logAttempts = 0;
  let progressAttempts = 0;
  let finalAttempts = 0;
  const persist = async (sql, params) => {
    if (sql.includes('SET log = log ||')) {
      writes.push('log');
      logAttempts++;
      if (logAttempts === 1 && holdLog) await holdLog;
      if (logAttempts <= failLog) throw new Error('log unavailable');
      row.log += params[1];
    } else if (sql.includes('SET progress =')) {
      writes.push('progress');
      progressAttempts++;
      if (progressAttempts === 1 && holdProgress) await holdProgress;
      if (progressAttempts <= failProgress) throw new Error('progress unavailable');
      row.progress = JSON.parse(params[1]);
    } else if (sql.includes('SET status =')) {
      writes.push('final');
      finalAttempts++;
      if (finalAttempts <= failFinal) throw new Error('database unavailable');
      row.log += params[3];
      row.status = params[1];
      row.exitCode = params[2];
      return {
        rowCount: 1,
        rows: [{ started_by: 7, started_at: new Date(1000), finished_at: new Date(1500), progress: row.progress }],
      };
    } else {
      throw new Error(`unexpected query: ${sql}`);
    }
    return { rowCount: 1 };
  };
  return { row, writes, persist, get finalAttempts() { return finalAttempts; } };
}

test('serializes output and progress before final status, while cleaning uploads promptly', async () => {
  const child = childProcess();
  const logGate = gate();
  const progressGate = gate();
  const store = runStore({ holdLog: logGate.promise, holdProgress: progressGate.promise });
  let refreshed = 0;
  let cleaned = 0;
  const finished = monitorImporterRun({
    runId: 'ordered', child, persist: store.persist,
    refreshSummary: () => { refreshed++; }, cleanupUpload: () => { cleaned++; },
  });
  const utf8 = Buffer.from('café\n');
  child.stdout.write(utf8.subarray(0, 4));
  child.stdout.write(utf8.subarray(4));
  child.stdout.write('@@PROGRESS@@ {"rows_done":1,"rows_total":2}\n');
  child.stdout.write('@@PROGRESS@@ {"rows_done":2,"rows_total":2}\nlast line');
  child.stderr.write('warning\n');
  child.emit('close', 0, null);
  try {
    await nextTurn();
    assert.deepEqual(store.writes, ['log'], 'later writes must wait for the first log');
    assert.equal(store.row.status, 'running');
    assert.equal(store.row.log, '');
    assert.equal(refreshed, 0);
    assert.equal(cleaned, 1, 'closed processes release uploads even while persistence stalls');

    logGate.release();
    await nextTurn();
    assert.deepEqual(store.writes, ['log', 'progress'], 'later progress must wait for earlier progress');
    assert.equal(store.row.log, 'café\n');
    assert.equal(store.row.progress, null);
    assert.equal(store.row.status, 'running');
    assert.equal(refreshed, 0);
    assert.equal(cleaned, 1);
  } finally {
    logGate.release();
    progressGate.release();
  }
  assert.equal(await finished, true);
  assert.equal(store.row.status, 'succeeded');
  assert.equal(store.row.exitCode, 0);
  // The stdout fragment is held until close; stderr is stored as it arrives.
  assert.equal(store.row.log, 'café\nwarning\nlast line\n');
  assert.deepEqual(store.row.progress, { rows_done: 2, rows_total: 2 });
  assert.deepEqual(store.writes, ['log', 'progress', 'progress', 'log', 'log', 'final']);
  assert.equal(refreshed, 1);
  assert.equal(cleaned, 1);
});

test('failed writes do not block later output, progress or finalization', async () => {
  const child = childProcess();
  const store = runStore({ failLog: 1, failProgress: 1 });
  const finished = monitorImporterRun({
    runId: 'write-failure', child, persist: store.persist,
    refreshSummary: () => {}, cleanupUpload: () => {},
  });
  child.stdout.write('lost line\n@@PROGRESS@@ {"rows_done":1}\nretained line\n@@PROGRESS@@ {"rows_done":2}\n');
  child.emit('close', 0, null);
  assert.equal(await finished, true);
  assert.equal(store.row.status, 'succeeded');
  assert.match(store.row.log, /^retained line\n/);
  assert.match(store.row.log, /run output incomplete: one or more log writes failed/);
  assert.deepEqual(store.row.progress, { rows_done: 2 });
  assert.deepEqual(store.writes, ['log', 'progress', 'log', 'progress', 'final']);
});

test('spawn error is finalized once on close, without exposing its path', async () => {
  const child = childProcess();
  const store = runStore();
  let refreshed = 0;
  let cleaned = 0;
  const events = [];
  const finished = monitorImporterRun({
    runId: 'spawn-error', child, persist: store.persist,
    refreshSummary: () => { refreshed++; }, cleanupUpload: () => { cleaned++; },
    recordUsage: (e) => events.push(e), runMeta: { runner: 'local', mode: 'replace' },
  });
  child.emit('error', Object.assign(new Error('/private/run.sh not found'), { code: 'ENOENT' }));
  child.emit('close', null, null);
  assert.equal(await finished, true);
  assert.equal(store.row.status, 'failed');
  assert.equal(store.row.exitCode, null);
  assert.match(store.row.log, /run\.sh failed to start \(ENOENT\)/);
  assert.doesNotMatch(store.row.log, /\/private/);
  assert.equal(store.finalAttempts, 1);
  assert.equal(refreshed, 0);
  assert.equal(cleaned, 1);
  assert.deepEqual(events.map((e) => [e.eventType, e.metadata]), [
    ['contribution.run.failed', { route: RUN_ROUTE, runner: 'local', mode: 'replace', failure: 'spawn_error' }],
  ]);
});

test('final update failure is reported once and does not refresh the summary', async () => {
  const child = childProcess();
  const store = runStore({ failFinal: 1 });
  let refreshed = 0;
  let cleaned = 0;
  const finished = monitorImporterRun({
    runId: 'final-failure', child, persist: store.persist,
    refreshSummary: () => { refreshed++; }, cleanupUpload: () => { cleaned++; },
  });
  child.stdout.write('last line\n');
  child.emit('close', 0, null);
  assert.equal(await finished, false);
  assert.equal(store.finalAttempts, 1);
  assert.equal(store.row.status, 'running');
  assert.equal(refreshed, 0);
  assert.equal(cleaned, 1);
});

test('progress failure alone does not mark output incomplete', async () => {
  const child = childProcess();
  const store = runStore({ failProgress: 1 });
  const finished = monitorImporterRun({
    runId: 'progress-failure', child, persist: store.persist,
    refreshSummary: () => {}, cleanupUpload: () => {},
  });
  child.stdout.write('kept\n@@PROGRESS@@ {"rows_done":1}\n');
  child.emit('close', 0, null);
  assert.equal(await finished, true);
  assert.equal(store.row.status, 'succeeded');
  assert.equal(store.row.log, 'kept\n');
  assert.equal(store.row.progress, null);
});

test('a stalled run does not block another run', async () => {
  const held = gate();
  const first = childProcess();
  const firstStore = runStore({ holdLog: held.promise });
  const firstFinished = monitorImporterRun({
    runId: 'independent-first', child: first, persist: firstStore.persist,
    refreshSummary: () => {}, cleanupUpload: () => {},
  });
  first.stdout.write('first\n');
  first.emit('close', 0, null);
  const second = childProcess();
  const secondStore = runStore();
  const secondFinished = monitorImporterRun({
    runId: 'independent-second', child: second, persist: secondStore.persist,
    refreshSummary: () => {}, cleanupUpload: () => {},
  });
  second.stdout.write('second\n');
  second.emit('close', 0, null);
  try {
    await nextTurn();
    assert.equal(firstStore.row.status, 'running');
    assert.equal(secondStore.row.status, 'succeeded');
    assert.equal(secondStore.row.log, 'second\n');
  } finally {
    held.release();
  }
  assert.deepEqual(await Promise.all([firstFinished, secondFinished]), [true, true]);
});

test('nonzero exit and signal termination preserve output without refreshing summaries', async () => {
  for (const [code, signal] of [[3, null], [null, 'SIGTERM']]) {
    const child = childProcess();
    const store = runStore();
    let refreshed = 0;
    let cleaned = 0;
    const events = [];
    const finished = monitorImporterRun({
      runId: `failed-${code}-${signal}`, child, persist: store.persist,
      refreshSummary: () => { refreshed++; }, cleanupUpload: () => { cleaned++; },
      recordUsage: (e) => events.push(e),
    });
    child.stdout.write('last line');
    child.emit('close', code, signal);
    assert.equal(await finished, true);
    assert.equal(store.row.status, 'failed');
    assert.equal(store.row.exitCode, code);
    assert.equal(store.row.log, `last line\n${signal ? '\n[run.sh terminated by signal SIGTERM]\n' : ''}`);
    assert.equal(store.finalAttempts, 1);
    assert.equal(refreshed, 0);
    assert.equal(cleaned, 1);
    assert.deepEqual(events.map((e) => e.metadata), [
      { route: RUN_ROUTE, failure: 'importer_exit', ...(code === null ? {} : { exit_code: code }) },
    ]);
  }
});

// --- Delegated runs (CONTRIBUTION_RUNNER_URL): the runner's stream through the same monitor.

const savedRunnerUrl = process.env.CONTRIBUTION_RUNNER_URL;
afterEach(() => {
  if (savedRunnerUrl === undefined) delete process.env.CONTRIBUTION_RUNNER_URL;
  else process.env.CONTRIBUTION_RUNNER_URL = savedRunnerUrl;
  _resetDelegateCache();
});

// A node:http stand-in for the runner; `respond(req, res)` answers every request.
async function stubRunner(respond) {
  const server = http.createServer((req, res) => { req.resume(); req.on('end', () => respond(req, res)); });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const url = `http://127.0.0.1:${server.address().port}`;
  const close = () => new Promise((resolve) => { server.close(() => resolve()); server.closeAllConnections(); });
  return { url, close };
}

async function delegatedRun(url, recordUsage, runMeta) {
  process.env.CONTRIBUTION_RUNNER_URL = url;
  const store = runStore();
  let cleaned = 0;
  const finished = await monitorImporterRun({
    ...(recordUsage && { recordUsage }),
    ...(runMeta && { runMeta }),
    runId: `delegated-${url}`,
    child: startDelegatedChild({ slug: 'm', api_key: 'w4h_sk_test', dataset_id: 'd1', mode: 'append' }),
    persist: store.persist, refreshSummary: () => {}, cleanupUpload: () => { cleaned++; },
  });
  return { finished, row: store.row, cleaned };
}

test('delegated success', async () => {
  const runner = await stubRunner((req, res) => res.end('x\n@@EXITCODE@@ {"code":0}\n'));
  try {
    const { finished, row, cleaned } = await delegatedRun(runner.url);
    assert.equal(finished, true);
    assert.equal(row.status, 'succeeded');
    assert.equal(row.exitCode, 0);
    assert.match(row.log, /x/);
    assert.equal(cleaned, 1);
  } finally {
    await runner.close();
  }
});

test('delegated failure code', async () => {
  const runner = await stubRunner((req, res) => res.end('x\n@@EXITCODE@@ {"code":3}\n'));
  try {
    const { row } = await delegatedRun(runner.url);
    assert.equal(row.status, 'failed');
    assert.equal(row.exitCode, 3);
  } finally {
    await runner.close();
  }
});

test('delegated stream drop', async () => {
  const runner = await stubRunner((req, res) => res.end('x\n'));
  try {
    const { row, cleaned } = await delegatedRun(runner.url);
    assert.equal(row.status, 'failed');
    assert.match(row.log, /\[runner connection closed before the run finished\]/);
    assert.equal(cleaned, 1);
  } finally {
    await runner.close();
  }
});

test('delegated unreachable', async () => {
  const runner = await stubRunner((req, res) => res.end());
  await runner.close();
  const { row, cleaned } = await delegatedRun(runner.url);
  assert.equal(row.status, 'failed');
  assert.ok(row.log.includes('[could not reach the importer runner:'), row.log);
  assert.equal(cleaned, 1);
});

test('delegated unreachable records one failed run event', async () => {
  const runner = await stubRunner((req, res) => res.end());
  await runner.close();
  const events = [];
  await delegatedRun(runner.url, (e) => events.push(e));
  assert.equal(events.length, 1);
  assert.equal(events[0].eventType, 'contribution.run.failed');
  assert.equal(events[0].userId, 7);
  assert.equal(events[0].durationMs, 500);
  assert.deepEqual(events[0].metadata, { route: RUN_ROUTE, exit_code: 1, failure: 'runner_unreachable' });
});

test('delegated success records a succeeded run event with the inserted row count', async () => {
  const runner = await stubRunner((req, res) => res.end('@@PROGRESS@@ {"inserted":12}\n@@EXITCODE@@ {"code":0}\n'));
  try {
    const events = [];
    await delegatedRun(runner.url, (e) => events.push(e));
    assert.equal(events.length, 1);
    assert.equal(events[0].eventType, 'contribution.run.succeeded');
    assert.deepEqual(events[0].metadata, { route: RUN_ROUTE, exit_code: 0, row_count: 12, rows_skipped: 0, signals: 1 });
  } finally {
    await runner.close();
  }
});

test('delegated success sums every signal and records the runner and upload size', async () => {
  const p = (o) => `@@PROGRESS@@ ${JSON.stringify({ signal_count: 2, ...o })}\n`;
  const runner = await stubRunner((req, res) => res.end(
    p({ signal: 'steps', signal_index: 1, inserted: 10, skipped: 1 })
    + p({ signal: 'steps', signal_index: 1, inserted: 40, skipped: 2 })
    + p({ signal: 'sleep', signal_index: 2, inserted: 5, skipped: 0 })
    + '@@EXITCODE@@ {"code":0}\n'
  ));
  try {
    const events = [];
    await delegatedRun(runner.url, (e) => events.push(e), { runner: 'delegate', mode: 'append', files: 2, bytes: 300 });
    assert.equal(events.length, 1);
    assert.equal(events[0].eventType, 'contribution.run.succeeded');
    assert.deepEqual(events[0].metadata, {
      route: RUN_ROUTE, runner: 'delegate', mode: 'append', files: 2, bytes: 300,
      exit_code: 0, row_count: 45, rows_skipped: 2, signals: 2, signals_planned: 2,
    });
  } finally {
    await runner.close();
  }
});

test('delegated failures record why: refused, stream closed, importer exit, runner-side sentinel', async () => {
  const cases = [
    ['runner_refused', 1, (req, res) => { res.statusCode = 503; res.end('busy'); }],
    ['stream_closed', 1, (req, res) => res.end('x\n')],
    ['importer_exit', 3, (req, res) => res.end('x\n@@EXITCODE@@ {"code":3}\n')],
    ['upload_fetch', 1, (req, res) => res.end('x\n@@EXITCODE@@ {"code":1,"failure":"upload_fetch"}\n')],
    ['runner_spawn', 1, (req, res) => res.end('x\n@@EXITCODE@@ {"code":1,"failure":"runner_spawn"}\n')],
  ];
  for (const [failure, exitCode, respond] of cases) {
    const runner = await stubRunner(respond);
    try {
      const events = [];
      await delegatedRun(runner.url, (e) => events.push(e));
      assert.deepEqual(events.map((e) => [e.eventType, e.metadata]), [
        ['contribution.run.failed', { route: RUN_ROUTE, exit_code: exitCode, failure }],
      ], failure);
    } finally {
      await runner.close();
    }
  }
});

test('runTotals sums the last snapshot of every signal (Fitbit sample shape)', () => {
  const snap = (signal, signal_index, inserted, skipped) => ({
    signal, signal_index, signal_count: 4, rows_done: inserted, rows_total: inserted, inserted, skipped,
  });
  assert.deepEqual(runTotals([
    snap('subjects', 1, 33, 0),
    snap('weight', 2, 70, 2),
    snap('heart_rate', 3, 165588, 10),
    snap('calories', 4, 1022000, 5),
    snap('calories', 4, 1215360, 7),
  ]), { row_count: 1381051, rows_skipped: 19, signals: 4, signals_planned: 4 });
});

test('runTotals: null without a numeric inserted; keys on signal when signal_index is missing', () => {
  assert.equal(runTotals([]), null);
  assert.equal(runTotals([{ rows_done: 5 }, { inserted: '7' }, null]), null);
  assert.deepEqual(runTotals([
    { signal: 'a', inserted: 1, skipped: 1 },
    { signal: 'a', inserted: 3, skipped: 1 },
    { signal: 'b', inserted: 2 },
  ]), { row_count: 5, rows_skipped: 1, signals: 2, signals_planned: null });
});

test('sqlRunMeta: outcome, write-back vs preview, rows written or returned', () => {
  const ok = { ok: true, write_back: true, inserted: 9, row_count: 12, physical_schema: 'ds_1', physical_table: 't', signal: 'mets' };
  assert.deepEqual(sqlRunMeta(ok), { runner: 'sql', mode: 'write_back', exit_code: 0, row_count: 9 });
  // a write-back that reports no inserted count must not fall back to the rows the query returned
  assert.deepEqual(sqlRunMeta({ ok: true, write_back: true, row_count: 12 }), {
    runner: 'sql', mode: 'write_back', exit_code: 0, row_count: null,
  });
  assert.deepEqual(sqlRunMeta({ ok: true, write_back: false, row_count: 12, inserted: null }), {
    runner: 'sql', mode: 'preview', exit_code: 0, row_count: 12,
  });
  assert.deepEqual(sqlRunMeta({ ok: false, error: 'Dataset is missing required signal(s): steps' }), {
    runner: 'sql', exit_code: 1, failure: 'sql_error',
  });
});

test('a throwing recordUsage does not break finalization', async () => {
  const child = childProcess();
  const store = runStore();
  const warn = console.warn;
  console.warn = () => {};
  try {
    const finished = monitorImporterRun({
      runId: 'usage-throws', child, persist: store.persist, refreshSummary: () => {}, cleanupUpload: () => {},
      recordUsage: () => { throw new Error('usage down'); },
    });
    child.emit('close', 0, null);
    assert.equal(await finished, true);
    assert.equal(store.row.status, 'succeeded');
  } finally {
    console.warn = warn;
  }
});

test('no RETURNING row records nothing', async () => {
  const child = childProcess();
  const store = runStore();
  const events = [];
  const finished = monitorImporterRun({
    runId: 'usage-no-row', child, refreshSummary: () => {}, cleanupUpload: () => {},
    persist: async (sql, params) => { await store.persist(sql, params); return { rowCount: 1, rows: [] }; },
    recordUsage: (e) => events.push(e),
  });
  child.emit('close', 0, null);
  assert.equal(await finished, true);
  assert.deepEqual(events, []);
});

test('delegatedRunUnavailable', async () => {
  delete process.env.CONTRIBUTION_RUNNER_URL;
  assert.equal(delegatedRunUnavailable('m'), null, 'no URL: the local path decides');

  const runner = await stubRunner((req, res) => res.end(JSON.stringify([{ slug: 'm' }])));
  try {
    process.env.CONTRIBUTION_RUNNER_URL = runner.url;
    await loadDelegateModules();
    assert.equal(delegatedRunUnavailable('m'), null);
    const missing = delegatedRunUnavailable('w4h-glucose-monitor-csv-file-importer');
    assert.equal(missing.ok, false);
    assert.equal(missing.status, 503);
    assert.match(missing.error, /not included in this installation/);
  } finally {
    await runner.close();
  }
});

test('delegatedRunUnavailable: URL set, runner unreachable -> null (the run fails with a readable line)', async () => {
  const runner = await stubRunner((req, res) => res.end());
  await runner.close();
  process.env.CONTRIBUTION_RUNNER_URL = runner.url;
  const warn = console.warn;
  console.warn = () => {};
  try {
    await loadDelegateModules();
  } finally {
    console.warn = warn;
  }
  assert.equal(delegatedRunUnavailable('w4h-glucose-monitor-csv-file-importer'), null);
});

test('reapOrphanedRuns', async () => {
  const calls = [];
  const persist = async (sql, params) => {
    calls.push({ sql, params });
    const row = (id, slug, startedBy) => ({
      id, contribution_slug: slug, started_by: startedBy, started_at: new Date(1000), finished_at: new Date(4000),
    });
    return { rowCount: 2, rows: [row(1, 'a', 7), row(2, 'b', 8)] };
  };
  const events = [];
  const warn = console.warn;
  console.warn = () => {};
  let reaped;
  try {
    reaped = await reapOrphanedRuns({ persist, recordUsage: (e) => events.push(e) });
  } finally {
    console.warn = warn;
  }
  assert.equal(reaped, 2);
  assert.match(calls[0].sql, /RETURNING[^]*started_by[^]*started_at/);
  assert.deepEqual(events, [7, 8].map((userId) => ({
    userId, eventType: 'contribution.run.failed', durationMs: 3000,
    metadata: { route: RUN_ROUTE, failure: 'api_restart' },
  })));
  assert.equal(calls.length, 1);
  assert.ok(calls[0].sql.includes("status IN ('queued', 'running')"), calls[0].sql);
  assert.ok(calls[0].params[0].includes('[orphaned:'));
  assert.ok(calls[0].sql.includes('started_at <'), calls[0].sql);
  assert.ok(calls[0].params[1] instanceof Date);
});

test('reapOrphanedRuns only touches runs started before the given boot time', async () => {
  const calls = [];
  const persist = async (sql, params) => {
    calls.push({ sql, params });
    return { rowCount: 0, rows: [] };
  };
  const before = new Date('2026-01-02T03:04:05Z');
  await reapOrphanedRuns({ persist, before });
  assert.ok(calls[0].sql.includes('started_at <'), calls[0].sql);
  assert.equal(calls[0].params[1], before);
});

// --- SQL runs: exactly one event per run (F1).

function sqlStore({ failFirstSelect = false } = {}) {
  const sqls = [];
  let selects = 0;
  const persist = async (sql) => {
    sqls.push(sql);
    if (sql.includes('INSERT INTO')) return { rowCount: 1, rows: [{ id: 'sql-run', status: 'running' }] };
    if (sql.includes('UPDATE')) {
      return { rowCount: 1, rows: [{ started_by: 7, started_at: new Date(1000), finished_at: new Date(1200) }] };
    }
    if (failFirstSelect && ++selects === 1) throw new Error('read timed out');
    return { rowCount: 1, rows: [{ id: 'sql-run', status: 'succeeded' }] };
  };
  return { sqls, persist };
}

test('SQL run: a status read failing after a recorded finalize does not re-finalize it as failed', async () => {
  const store = sqlStore({ failFirstSelect: true });
  const events = [];
  const out = await startSqlContributionRun({
    slug: 's', datasetId: 'd', user: { id: 7 }, contribution: {},
    persist: store.persist, runSql: async () => ({ ok: true, write_back: true, inserted: 9 }),
    recordUsage: (e) => events.push(e), refreshSummary: () => {},
  });
  assert.deepEqual(events.map((e) => e.eventType), ['contribution.run.succeeded']);
  assert.ok(!store.sqls.some((q) => q.includes("status = 'failed'")), 'row not flipped to failed');
  assert.equal(out.run.status, 'succeeded');
});

test('SQL run: a thrown job records one failed sql_error event', async () => {
  const store = sqlStore();
  const events = [];
  await startSqlContributionRun({
    slug: 's', datasetId: 'd', user: { id: 7 }, contribution: {},
    persist: store.persist, runSql: async () => { throw new Error('relation does not exist'); },
    recordUsage: (e) => events.push(e), refreshSummary: () => {},
  });
  assert.deepEqual(events.map((e) => [e.eventType, e.durationMs, e.metadata]), [
    ['contribution.run.failed', 200, { route: RUN_ROUTE, runner: 'sql', exit_code: 1, failure: 'sql_error' }],
  ]);
});

test('a trailing progress snapshot without inserted does not hide the signal count (F3)', async () => {
  const child = childProcess();
  const store = runStore();
  const events = [];
  const finished = monitorImporterRun({
    runId: 'snapshot-no-inserted', child, persist: store.persist,
    refreshSummary: () => {}, cleanupUpload: () => {}, recordUsage: (e) => events.push(e),
  });
  child.stdout.write('@@PROGRESS@@ {"signal_index":1,"inserted":10,"skipped":1}\n@@PROGRESS@@ {"signal_index":1,"rows_done":20}\n');
  child.emit('close', 0, null);
  assert.equal(await finished, true);
  assert.deepEqual(events[0].metadata, { route: RUN_ROUTE, exit_code: 0, row_count: 10, rows_skipped: 1, signals: 1 });
});
