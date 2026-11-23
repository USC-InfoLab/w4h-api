import test, { beforeEach, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import {
  delegateUrl,
  loadDelegateModules,
  delegateModule,
  _resetDelegateCache,
  exitCodeFromLine,
  startDelegatedChild,
} from './contribution-delegate.js';

const API_KEY = 'w4h_sk_TESTKEY_must_never_appear';
const BODY = { slug: 'imp', api_key: API_KEY, dataset_id: 7, mode: 'append', upload_id: 'u1' };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const savedUrl = process.env.CONTRIBUTION_RUNNER_URL;
const stubs = [];

beforeEach(() => {
  delete process.env.CONTRIBUTION_RUNNER_URL;
  _resetDelegateCache();
});

afterEach(async () => {
  if (savedUrl === undefined) delete process.env.CONTRIBUTION_RUNNER_URL;
  else process.env.CONTRIBUTION_RUNNER_URL = savedUrl;
  _resetDelegateCache();
  while (stubs.length) await closeStub(stubs.pop());
});

// handler(req, res, bodyText) answers every request; stub.hits counts them.
async function startStub(handler) {
  const stub = { hits: 0, requests: [] };
  stub.server = http.createServer((req, res) => {
    stub.hits += 1;
    let text = '';
    req.setEncoding('utf8');
    req.on('data', (c) => { text += c; });
    req.on('end', () => {
      stub.requests.push({ method: req.method, url: req.url, headers: req.headers, text });
      handler(req, res, text);
    });
  });
  await new Promise((r) => stub.server.listen(0, '127.0.0.1', r));
  stub.url = `http://127.0.0.1:${stub.server.address().port}`;
  stubs.push(stub);
  return stub;
}

function closeStub(stub) {
  return new Promise((r) => {
    stub.server.close(() => r());
    stub.server.closeAllConnections();
  });
}

// Runs startDelegatedChild against `url` and resolves once 'close' fired.
// Also asserts what every case must satisfy: one close, after stdout ended,
// no stderr, no api key in the output.
function runChild(url, body = BODY) {
  process.env.CONTRIBUTION_RUNNER_URL = url;
  const child = startDelegatedChild(body);
  assert.equal(child.stderr, undefined);
  return new Promise((resolve) => {
    let out = '';
    let closes = 0;
    let endedAtClose = null;
    let result;
    child.stdout.setEncoding('utf8');
    child.stdout.on('data', (c) => { out += c; });
    child.on('close', async (code, signal) => {
      closes += 1;
      endedAtClose = child.stdout.readableEnded;
      result = { code, signal, out };
      await sleep(30); // a second close would land here
      assert.equal(closes, 1, 'close emitted exactly once');
      assert.equal(endedAtClose, true, 'close came after stdout ended');
      assert.ok(!out.includes(API_KEY), 'api key not in stdout');
      resolve(result);
    });
  });
}

test('exitCodeFromLine parses the sentinel line', () => {
  assert.equal(exitCodeFromLine('@@EXITCODE@@ {"code":0}'), 0);
  assert.equal(exitCodeFromLine('@@EXITCODE@@ {"code":2}'), 2);
  assert.equal(exitCodeFromLine('hello'), null);
  assert.equal(exitCodeFromLine('@@EXITCODE@@ nope'), null);
  assert.equal(exitCodeFromLine('@@EXITCODE@@ {"code":"x"}'), null);
  assert.equal(exitCodeFromLine('@@PROGRESS@@ {}'), null);
});

test('delegateUrl strips a trailing slash and is read at call time', () => {
  assert.equal(delegateUrl(), '');
  process.env.CONTRIBUTION_RUNNER_URL = 'http://runner:8080/';
  assert.equal(delegateUrl(), 'http://runner:8080');
  process.env.CONTRIBUTION_RUNNER_URL = 'http://runner:8080';
  assert.equal(delegateUrl(), 'http://runner:8080');
});

test('loadDelegateModules caches the modules, and an empty map on failure', async () => {
  const stub = await startStub((req, res) => {
    res.setHeader('content-type', 'application/json');
    res.end(JSON.stringify([{ slug: 'm', version: '1.0', contribution_yaml: 'inputs: []' }, { slug: 'bare' }]));
  });
  process.env.CONTRIBUTION_RUNNER_URL = stub.url;
  await loadDelegateModules();
  assert.equal(stub.requests[0].url, '/modules');
  assert.equal(delegateModule('m').version, '1.0');
  assert.equal(delegateModule('m').contribution_yaml, 'inputs: []');
  assert.deepEqual(delegateModule('bare'), { version: null, contribution_yaml: null });
  assert.equal(delegateModule('x'), null);

  // Fresh cache: no second request.
  await loadDelegateModules();
  assert.equal(stub.hits, 1);

  // Runner gone: after a reset the load still resolves, with an empty map.
  await closeStub(stub);
  _resetDelegateCache();
  const warn = console.warn;
  const warnings = [];
  console.warn = (...a) => warnings.push(a.join(' '));
  try {
    await loadDelegateModules();
    assert.equal(delegateModule('m'), null);
    // The failure is cached for a while: no retry, no second warning.
    await loadDelegateModules();
    assert.equal(warnings.length, 1);
  } finally {
    console.warn = warn;
  }
});

test('loadDelegateModules treats a non-2xx or non-array answer as a failure', async () => {
  const warn = console.warn;
  console.warn = () => {};
  try {
    const failing = await startStub((req, res) => { res.statusCode = 500; res.end('boom'); });
    process.env.CONTRIBUTION_RUNNER_URL = failing.url;
    await loadDelegateModules();
    assert.equal(delegateModule('m'), null);

    const wrongShape = await startStub((req, res) => res.end('{"slug":"m"}'));
    process.env.CONTRIBUTION_RUNNER_URL = wrongShape.url;
    _resetDelegateCache();
    await loadDelegateModules();
    assert.equal(delegateModule('m'), null);
  } finally {
    console.warn = warn;
  }
});

test('loadDelegateModules with no URL makes no request', async () => {
  const stub = await startStub((req, res) => res.end('[]'));
  await loadDelegateModules();
  assert.equal(stub.hits, 0);
  assert.equal(delegateModule('m'), null);
});

test('run stream: sentinel gives the exit code and is not forwarded', async () => {
  const stub = await startStub((req, res) => {
    res.write('a\n@@PROGRESS@@ {"pct":50}\n@@EXITCODE@@ {"code":2}\n');
    res.end();
  });
  const { code, signal, out } = await runChild(stub.url);
  assert.equal(code, 2);
  assert.equal(signal, null);
  assert.equal(out, 'a\n@@PROGRESS@@ {"pct":50}\n');
  assert.equal(stub.requests[0].method, 'POST');
  assert.equal(stub.requests[0].url, '/run');
  assert.match(stub.requests[0].headers['content-type'], /application\/json/);
  assert.deepEqual(JSON.parse(stub.requests[0].text), BODY);
});

test('run stream: exit code 0', async () => {
  const stub = await startStub((req, res) => res.end('done\n@@EXITCODE@@ {"code":0}\n'));
  const { code, out } = await runChild(stub.url);
  assert.equal(code, 0);
  assert.equal(out, 'done\n');
});

test('run stream: lines and the sentinel split across chunks', async () => {
  const stub = await startStub(async (req, res) => {
    res.write('hel');
    await sleep(15);
    res.write('lo\n@@EXIT');
    await sleep(15);
    res.write('CODE@@ {"co');
    await sleep(15);
    res.write('de":3}\n');
    res.end();
  });
  const { code, out } = await runChild(stub.url);
  assert.equal(code, 3);
  assert.equal(out, 'hello\n');
});

test('run stream: a multibyte character split across chunks survives', async () => {
  const stub = await startStub(async (req, res) => {
    const bytes = Buffer.from('café ✓\n');
    res.write(bytes.subarray(0, 4)); // splits the two-byte e-acute
    await sleep(15);
    res.write(bytes.subarray(4, 7)); // splits the three-byte check mark
    await sleep(15);
    res.write(bytes.subarray(7));
    res.end('@@EXITCODE@@ {"code":0}\n');
  });
  const { out } = await runChild(stub.url);
  assert.equal(out, 'café ✓\n');
});

test('run stream: a trailing partial line passes through; a sentinel without newline counts', async () => {
  const partial = await startStub((req, res) => res.end('a\n@@EXITCODE@@ {"code":0}\ntail'));
  const first = await runChild(partial.url);
  assert.equal(first.code, 0);
  assert.equal(first.out, 'a\ntail');

  const bare = await startStub((req, res) => res.end('a\n@@EXITCODE@@ {"code":4}'));
  const second = await runChild(bare.url);
  assert.equal(second.code, 4);
  assert.equal(second.out, 'a\n');
});

test('run stream: a malformed sentinel is forwarded as a normal line', async () => {
  const stub = await startStub((req, res) => res.end('@@EXITCODE@@ nope\n'));
  const { code, out } = await runChild(stub.url);
  assert.equal(code, 1);
  assert.equal(out, '@@EXITCODE@@ nope\n[runner connection closed before the run finished]\n');
});

test('run stream: closing without the sentinel is reported and fails the run', async () => {
  const stub = await startStub((req, res) => res.end('a\n'));
  const { code, signal, out } = await runChild(stub.url);
  assert.equal(code, 1);
  assert.equal(signal, null);
  assert.ok(out.endsWith('[runner connection closed before the run finished]\n'));
  assert.ok(out.startsWith('a\n'));
});

test('run stream: a partial line before an early close keeps the notice on its own line', async () => {
  const stub = await startStub((req, res) => res.end('half'));
  const { out } = await runChild(stub.url);
  assert.equal(out, 'half\n[runner connection closed before the run finished]\n');
});

test('run stream: a connection reset mid-run is the same early close', async () => {
  const stub = await startStub((req, res) => {
    res.write('a\n');
    setTimeout(() => res.destroy(), 20);
  });
  const { code, out } = await runChild(stub.url);
  assert.equal(code, 1);
  assert.ok(out.startsWith('a\n'));
  assert.ok(out.endsWith('[runner connection closed before the run finished]\n'));
});

test('run stream: an unreachable runner is reported and fails the run', async () => {
  const stub = await startStub((req, res) => res.end());
  const { url } = stub;
  await closeStub(stub);
  stubs.splice(stubs.indexOf(stub), 1);
  const { code, signal, out } = await runChild(url);
  assert.equal(code, 1);
  assert.equal(signal, null);
  assert.ok(out.includes('[could not reach the importer runner:'), out);
  assert.ok(out.endsWith(']\n'));
});

test('run stream: an unset or malformed runner URL fails the run, never throws', async () => {
  const unset = await runChild('');
  assert.equal(unset.code, 1);
  assert.ok(unset.out.includes('[could not reach the importer runner:'), unset.out);
  const bad = await runChild('not a url');
  assert.equal(bad.code, 1);
  assert.ok(bad.out.includes('[could not reach the importer runner:'), bad.out);
});

test('run stream: a non-2xx answer is reported with its status and body', async () => {
  const stub = await startStub((req, res) => {
    res.statusCode = 400;
    res.setHeader('content-type', 'application/json');
    res.end('{"error":"bad slug"}');
  });
  const { code, out } = await runChild(stub.url);
  assert.equal(code, 1);
  assert.equal(out, '[importer runner refused the run: 400 {"error":"bad slug"}]\n');
});
