import test, { before, after } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import express from 'express';

// UPLOADS_ROOT is read at import time, so set it first.
const root = fs.mkdtempSync(path.join(os.tmpdir(), 'w4h-uploads-test-'));
process.env.CONTRIBUTION_UPLOADS_ROOT = root;
const { createUpload, loadUpload, sendUploadFile } = await import('./contribution-uploads.js');

const A_BYTES = Buffer.from('subject,time,v\n1,2,3\n');
const NESTED_BYTES = Buffer.from('x,y\n4,5\n');
let server;
let base;
let uploadId;

before(async () => {
  const { id, dir } = createUpload({ user: { id: 2 }, slug: 'any', datasetId: '1' });
  uploadId = id;
  fs.writeFileSync(path.join(dir, 'a.csv'), A_BYTES);
  fs.mkdirSync(path.join(dir, 'participant_data'));
  fs.writeFileSync(path.join(dir, 'participant_data', 'a b (1).csv'), NESTED_BYTES);
  // A secret outside the upload dir, for the traversal case.
  fs.writeFileSync(path.join(root, 'secret.csv'), 'nope');

  const app = express();
  app.get('/f/:uploadId', (req, _res, next) => {
    req.user = { id: Number(req.get('x-uid')) };
    next();
  }, (req, res) => {
    const u = loadUpload(req.params.uploadId, req.user);
    if (!u.ok) return res.status(u.status).json({ error: u.error });
    sendUploadFile(res, u.dir, req.query.name);
  });
  server = app.listen(0);
  await new Promise((resolve) => server.once('listening', resolve));
  base = `http://127.0.0.1:${server.address().port}`;
});

after(() => {
  server.close();
  fs.rmSync(root, { recursive: true, force: true });
});

const get = (id, query, uid = 2) => fetch(`${base}/f/${id}${query}`, { headers: { 'x-uid': String(uid) } });

test('owner gets the bytes', async () => {
  const res = await get(uploadId, '?name=a.csv');
  assert.equal(res.status, 200);
  assert.deepEqual(Buffer.from(await res.arrayBuffer()), A_BYTES);
});

test('encoded nested name', async () => {
  const res = await get(uploadId, '?name=participant_data%2Fa%20b%20%281%29.csv');
  assert.equal(res.status, 200);
  assert.deepEqual(Buffer.from(await res.arrayBuffer()), NESTED_BYTES);
});

test('other user 403', async () => {
  const res = await get(uploadId, '?name=a.csv', 3);
  assert.equal(res.status, 403);
});

test('traversal and dotfiles 400', async () => {
  for (const q of ['?name=..%2F.owner.json', '?name=.owner.json', '?name=..%2Fsecret.csv', '']) {
    assert.equal((await get(uploadId, q)).status, 400, q);
  }
});

test('repeated name uses the first value', async () => {
  assert.equal((await get(uploadId, '?name=a.csv&name=..%2Fsecret.csv')).status, 200);
  assert.equal((await get(uploadId, '?name=..%2Fsecret.csv&name=a.csv')).status, 400);
});

test('missing 404', async () => {
  const res = await get(uploadId, '?name=b.csv');
  assert.equal(res.status, 404);
  assert.deepEqual(await res.json(), { error: 'File not found' });
});

test('bad upload id 400', async () => {
  const res = await get('..%2Fetc', '?name=a.csv');
  assert.equal(res.status, 400);
});
