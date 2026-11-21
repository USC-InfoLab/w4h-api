import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { forgetNotebookKey, shareKeyWithNotebooks } from './notebook-key.js';

test('shareKeyWithNotebooks: first non-Super-admin claims the file, others never overwrite', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'w4h-nbkey-'));
  const file = path.join(dir, 'api_key');
  const read = () => JSON.parse(fs.readFileSync(file, 'utf8'));

  assert.equal(shareKeyWithNotebooks({ userId: 1, role: 'super_admin', apiKey: 'k-super' }, file), false);
  assert.equal(fs.existsSync(file), false, 'Super admin key is never shared');

  assert.equal(shareKeyWithNotebooks({ userId: 2, role: 'project_user', apiKey: 'k1' }, file), true);
  assert.deepEqual(read(), { user_id: 2, api_key: 'k1' });
  assert.equal(fs.statSync(file).mode & 0o777, 0o644);

  assert.equal(shareKeyWithNotebooks({ userId: 3, role: 'project_user', apiKey: 'k-other' }, file), false);
  assert.equal(read().api_key, 'k1', 'another user does not take over');

  assert.equal(shareKeyWithNotebooks({ userId: '2', role: 'project_user', apiKey: 'k2' }, file), true);
  assert.equal(read().api_key, 'k2', 'the owner rotating replaces it');

  assert.equal(forgetNotebookKey(3, file), false, 'another user revoking keeps the file');
  assert.equal(forgetNotebookKey(2, file), true);
  assert.equal(fs.existsSync(file), false);
  assert.equal(shareKeyWithNotebooks({ userId: 3, role: 'project_user', apiKey: 'k3' }, file), true, 'released: next key claims it');
  assert.equal(read().user_id, 3);
  assert.equal(forgetNotebookKey(3, path.join(dir, 'missing')), false);

  fs.writeFileSync(file, 'not json');
  assert.equal(shareKeyWithNotebooks({ userId: 2, role: 'project_user', apiKey: 'k4' }, file), false);
  assert.equal(fs.readFileSync(file, 'utf8'), 'not json', 'unreadable owner: leave the file alone');

  assert.equal(shareKeyWithNotebooks({ userId: 2, role: 'project_user', apiKey: 'k' }, undefined), false, 'off unless configured');
  fs.rmSync(dir, { recursive: true, force: true });
});
