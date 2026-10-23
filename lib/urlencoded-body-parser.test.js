import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { PassThrough } from 'node:stream';
import test from 'node:test';
import express from 'express';
import { saveUploadFile } from './contribution-uploads.js';
import { urlencodedBodyParser } from './urlencoded-body-parser.js';

const csv = 'Subject ID,timestamp,HR\ns1,2026-01-01T00:00:00Z,61\n';

async function request(method, url, body, contentType, parser = urlencodedBodyParser) {
  const req = new PassThrough();
  req.method = method;
  req.url = url;
  req.path = new URL(url, 'http://localhost').pathname;
  req.headers = {
    'content-type': contentType,
    'content-length': String(Buffer.byteLength(body)),
  };
  req.is = express.request.is;
  req.end(body);
  await new Promise((resolve, reject) => parser(req, {}, (err) => err ? reject(err) : resolve()));
  return req;
}

test('urlencoded CSV upload reaches the file stream on both route mounts', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'w4h-upload-parser-'));
  try {
    for (const mount of ['contributions', 'packages']) {
      const req = await request('PUT', `/${mount}/uploads/id/files/hr.csv?x=1`, csv, 'application/x-www-form-urlencoded; charset=UTF-8');
      const result = await saveUploadFile(dir, 'hr.csv', req);
      assert.deepEqual(result, { ok: true, name: 'hr.csv', size: Buffer.byteLength(csv) });
      assert.equal(fs.readFileSync(path.join(dir, 'hr.csv'), 'utf8'), csv);
    }
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('ordinary form requests and unrelated upload PUTs still parse', async () => {
  for (const [method, url] of [
    ['POST', '/users/login'],
    ['PUT', '/contributions/uploads/id'],
  ]) {
    const req = await request(method, url, 'name=Jane+Doe', 'application/x-www-form-urlencoded');
    assert.equal(req.body.name, 'Jane Doe');
  }
});

test('text/csv upload stays raw and an empty upload stays invalid', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'w4h-upload-parser-'));
  try {
    const raw = await request('PUT', '/contributions/uploads/id/files/hr.csv', csv, 'text/csv');
    assert.deepEqual(await saveUploadFile(dir, 'hr.csv', raw), {
      ok: true,
      name: 'hr.csv',
      size: Buffer.byteLength(csv),
    });
    assert.equal(fs.readFileSync(path.join(dir, 'hr.csv'), 'utf8'), csv);

    const empty = await request('PUT', '/contributions/uploads/id/files/empty.csv', '', 'application/x-www-form-urlencoded');
    const result = await saveUploadFile(dir, 'empty.csv', empty);
    assert.equal(result.status, 400);
    assert.match(result.error, /is empty/);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});
