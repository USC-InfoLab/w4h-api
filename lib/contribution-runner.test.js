import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import http from 'node:http';
import { PassThrough } from 'node:stream';
import test, { afterEach } from 'node:test';
import { monitorImporterRun, delegatedRunUnavailable, reapOrphanedRuns } from './contribution-runner.js';
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
  const finished = monitorImporterRun({
    runId: 'spawn-error', child, persist: store.persist,
    refreshSummary: () => { refreshed++; }, cleanupUpload: () => { cleaned++; },
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
    const finished = monitorImporterRun({
      runId: `failed-${code}-${signal}`, child, persist: store.persist,
      refreshSummary: () => { refreshed++; }, cleanupUpload: () => { cleaned++; },
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

async function delegatedRun(url) {
  process.env.CONTRIBUTION_RUNNER_URL = url;
  const store = runStore();
  let cleaned = 0;
  const finished = await monitorImporterRun({
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
    return { rowCount: 2, rows: [{ id: 1, contribution_slug: 'a' }, { id: 2, contribution_slug: 'b' }] };
  };
  const warn = console.warn;
  console.warn = () => {};
  let reaped;
  try {
    reaped = await reapOrphanedRuns({ persist });
  } finally {
    console.warn = warn;
  }
  assert.equal(reaped, 2);
  assert.equal(calls.length, 1);
  assert.ok(calls[0].sql.includes("status IN ('queued', 'running')"), calls[0].sql);
  assert.ok(calls[0].params[0].includes('[orphaned:'));
});
