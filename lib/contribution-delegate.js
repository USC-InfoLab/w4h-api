import http from 'node:http';
import { EventEmitter } from 'node:events';
import { PassThrough } from 'node:stream';

/**
 * Client for the Python importer runner (w4h-docker `runner` service). In
 * Docker the api image has no Python, so it delegates importer runs over HTTP:
 *   GET  ${CONTRIBUTION_RUNNER_URL}/modules  what the runner has installed
 *   POST ${CONTRIBUTION_RUNNER_URL}/run      streams run.sh output, then a
 *                                            final `@@EXITCODE@@ {"code":N}`
 */

const EXIT_MARKER = '@@EXITCODE@@ ';
const MODULES_OK_MS = 60_000;
const MODULES_FAILED_MS = 10_000;

/** Runner base URL without a trailing slash, or '' when not delegating. */
export function delegateUrl() {
  return (process.env.CONTRIBUTION_RUNNER_URL || '').replace(/\/+$/, '');
}

let cache = { map: new Map(), expires: 0, ok: false };

/** Tests only. */
export function _resetDelegateCache() {
  cache = { map: new Map(), expires: 0, ok: false };
}

/**
 * Refreshes the installed-modules cache (60 s; an empty map for 10 s after a
 * failure, so a down runner is not hit on every request). Never throws.
 */
export async function loadDelegateModules() {
  const url = delegateUrl();
  if (!url || Date.now() < cache.expires) return;
  try {
    const res = await fetch(`${url}/modules`, { signal: AbortSignal.timeout(3000) });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const rows = await res.json();
    if (!Array.isArray(rows)) throw new Error('unexpected response');
    const map = new Map();
    for (const row of rows) {
      if (typeof row?.slug !== 'string') continue;
      map.set(row.slug, {
        version: typeof row.version === 'string' ? row.version : null,
        contribution_yaml: typeof row.contribution_yaml === 'string' ? row.contribution_yaml : null,
      });
    }
    cache = { map, expires: Date.now() + MODULES_OK_MS, ok: true };
  } catch (err) {
    console.warn(`importer runner: could not read ${url}/modules: ${err.message}`);
    cache = { map: new Map(), expires: Date.now() + MODULES_FAILED_MS, ok: false };
  }
}

/** Cached lookup; call loadDelegateModules() first. */
export function delegateModule(slug) {
  return cache.map.get(slug) ?? null;
}

/** True only when the most recent /modules load succeeded. */
export function delegateRunnerReachable() {
  return cache.ok;
}

/**
 * `{ code, failure }` from an `@@EXITCODE@@ {"code":N[,"failure":"upload_fetch"|"runner_spawn"]}`
 * line, or null for any other line. `failure` (a runner-side reason) is undefined when absent
 * or not a lower_snake word.
 */
export function exitFromLine(line) {
  if (!line.startsWith(EXIT_MARKER)) return null;
  try {
    const { code, failure } = JSON.parse(line.slice(EXIT_MARKER.length));
    if (!Number.isInteger(code)) return null;
    return { code, failure: typeof failure === 'string' && /^[a-z_]{1,32}$/.test(failure) ? failure : undefined };
  } catch {
    return null;
  }
}

/** The code from an `@@EXITCODE@@` line, or null for any other line. */
export function exitCodeFromLine(line) {
  return exitFromLine(line)?.code ?? null;
}

/**
 * Starts a run on the runner and returns a child-process look-alike for
 * monitorImporterRun: `stdout` (a PassThrough the caller must consume) and
 * 'close'(code, null), emitted once, after stdout has ended. There is no
 * stderr and no 'error': every failure is a line in stdout plus close(1), and
 * `child.failure` (runner_unreachable | runner_refused | stream_closed, or the
 * sentinel's upload_fetch | runner_spawn) is set before 'close' when the runner,
 * not the importer, ended the run.
 *
 * node:http with no timeout, not fetch: imports run for as long as they run,
 * and undici aborts a response body that idles for 300 s.
 */
export function startDelegatedChild(body) {
  const child = new EventEmitter();
  child.stdout = new PassThrough();

  let closed = false;
  let lastChar = '\n';
  const write = (text) => {
    if (!text) return;
    child.stdout.write(text);
    lastChar = text[text.length - 1];
  };
  const finish = (code, notice, failure) => {
    if (closed) return;
    closed = true;
    if (failure) child.failure = failure;
    // Keep the notice on a line of its own after a partial line.
    if (notice) write(`${lastChar === '\n' ? '' : '\n'}${notice}\n`);
    child.stdout.once('end', () => child.emit('close', code, null));
    child.stdout.end();
  };

  // The body carries the person's api_key: never log it or put it in a notice.
  let req;
  try {
    // ponytail: plain http only (compose network); add https if a runner ever sits behind TLS.
    req = http.request(new URL(`${delegateUrl()}/run`), {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
    });
  } catch (err) {
    finish(1, `[could not reach the importer runner: ${err.message}]`, 'runner_unreachable');
    return child;
  }

  req.on('error', (err) => {
    finish(1, `[could not reach the importer runner: ${err.message || err.code || 'unknown error'}]`, 'runner_unreachable');
  });

  req.on('response', (res) => {
    res.setEncoding('utf8');

    if (res.statusCode < 200 || res.statusCode >= 300) {
      let text = '';
      res.on('data', (chunk) => { text += chunk; });
      const refuse = () => finish(1, `[importer runner refused the run: ${res.statusCode} ${text}]`, 'runner_refused');
      res.on('end', refuse);
      res.on('error', refuse);
      res.on('close', refuse);
      return;
    }

    let exit = null;
    let pending = '';
    const route = (line) => {
      const sentinel = exitFromLine(line);
      if (sentinel) exit = sentinel; // last one wins
      else write(`${line}\n`);
    };
    res.on('data', (chunk) => {
      const lines = (pending + chunk).split('\n');
      pending = lines.pop();
      lines.forEach(route);
    });
    // 'end' on a clean finish; 'error' / 'close' alone when the connection drops.
    const ended = () => {
      if (closed) return;
      if (pending) {
        const sentinel = exitFromLine(pending);
        if (sentinel) exit = sentinel;
        else write(pending);
        pending = '';
      }
      if (!exit) finish(1, '[runner connection closed before the run finished]', 'stream_closed');
      else finish(exit.code, undefined, exit.failure);
    };
    res.on('end', ended);
    res.on('error', ended);
    res.on('close', ended);
  });

  req.end(JSON.stringify(body));
  return child;
}
