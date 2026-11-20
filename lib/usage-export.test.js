import { test } from 'node:test';
import assert from 'node:assert/strict';
import { ROW_KEYS, pseudonym, safeRoute, scrubRow, toCsv, clampDays } from './usage-export.js';

const INSTALL_A = '11111111-1111-4111-8111-111111111111';
const INSTALL_B = '22222222-2222-4222-8222-222222222222';

test('pseudonym is stable per install and differs across installs', () => {
  assert.equal(pseudonym(INSTALL_A, 7), pseudonym(INSTALL_A, 7));
  assert.notEqual(pseudonym(INSTALL_A, 7), pseudonym(INSTALL_A, 8));
  assert.notEqual(pseudonym(INSTALL_A, 7), pseudonym(INSTALL_B, 7));
  assert.match(pseudonym(INSTALL_A, 7), /^[0-9a-f]{16}$/);
  assert.equal(pseudonym(INSTALL_A, null), null);
});

test('safeRoute keeps templates and drops emails, digit and UUID segments', () => {
  assert.equal(safeRoute('/datasets/:id/import/csv-batch'), '/datasets/:id/import/csv-batch');
  assert.equal(safeRoute('/datasets/42/import/csv-batch'), null);
  assert.equal(safeRoute('/teams/3f2b6a10-1c2d-4e5f-8a9b-0c1d2e3f4a5b'), null);
  assert.equal(safeRoute('/users/user/a@b.org'), null);
  assert.equal(safeRoute(undefined), null);
});

test('scrubRow keeps only the allow-listed keys and no identifying values', () => {
  const raw = {
    user_id: 5,
    team_role: 'team_admin',
    event_type: 'api.request',
    timestamp: new Date('2026-11-20T10:00:00Z'),
    duration_ms: 123,
    metadata: {
      method: 'POST', route: '/datasets/:id/import/csv-batch', status: 200, row_count: 500,
      path: '/datasets/42/import/csv-batch', auth_method: 'api_key', email: 'p@x.org',
      subject: 'secret', page_url: 'http://127.0.0.1:5180/projects/9', team_id: 'abc',
    },
  };
  const row = scrubRow(raw, INSTALL_A);
  assert.deepEqual(Object.keys(row), ROW_KEYS);
  assert.equal(row.route, '/datasets/:id/import/csv-batch');
  assert.equal(row.row_count, 500);
  assert.equal(row.timestamp, '2026-11-20T10:00:00.000Z');
  const flat = JSON.stringify(row);
  assert.ok(!flat.includes('@'), 'no emails');
  assert.ok(!/\/\d+(\/|"|$)/.test(flat), 'no digits-only path segments');
  assert.ok(!flat.includes('secret'));

  // Legacy API-key row with a raw path only: route is null, path is dropped.
  const legacy = scrubRow({ ...raw, metadata: { method: 'GET', path: '/datasets/42', status: 200 } }, INSTALL_A);
  assert.equal(legacy.route, null);
  assert.ok(!JSON.stringify(legacy).includes('42'));
});

test('toCsv writes meta lines, header, quoted cells', () => {
  const csv = toCsv({ install_id: INSTALL_A, row_count: 1 }, [
    { pseudonym: 'abc', event_type: 'page.view', timestamp: 't', route: '/a,b' },
  ]);
  const lines = csv.trim().split('\n');
  assert.equal(lines[0], `# install_id=${INSTALL_A}`);
  assert.equal(lines[2], ROW_KEYS.join(','));
  assert.ok(lines[3].includes('"/a,b"'));
});

test('clampDays bounds the export window', () => {
  assert.equal(clampDays(undefined), 30);
  assert.equal(clampDays('0'), 1);
  assert.equal(clampDays('9999'), 365);
  assert.equal(clampDays('7'), 7);
});
