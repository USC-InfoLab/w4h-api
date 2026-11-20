import { test } from 'node:test';
import assert from 'node:assert/strict';
import { ROW_KEYS, pseudonym, pseudonymKey, safeRoute, scrubRow, toCsv, clampDays, clampWindowEnd } from './usage-export.js';

const SECRET = 'test-token-secret';
const INSTALL_A = pseudonymKey(SECRET, '11111111-1111-4111-8111-111111111111');
const INSTALL_B = pseudonymKey(SECRET, '22222222-2222-4222-8222-222222222222');
const INSTALL_ID_A = '11111111-1111-4111-8111-111111111111';

test('pseudonym is stable per install and differs across installs', () => {
  assert.equal(pseudonym(INSTALL_A, 7), pseudonym(INSTALL_A, 7));
  assert.notEqual(pseudonym(INSTALL_A, 7), pseudonym(INSTALL_A, 8));
  assert.notEqual(pseudonym(INSTALL_A, 7), pseudonym(INSTALL_B, 7));
  assert.match(pseudonym(INSTALL_A, 7), /^[0-9a-f]{16}$/);
  assert.equal(pseudonym(INSTALL_A, null), null);
});

test('pseudonym cannot be recomputed from the published install_id alone', () => {
  const naive = (uid) => pseudonym(Buffer.from(INSTALL_ID_A), uid);
  assert.notEqual(pseudonym(INSTALL_A, 7), naive(7));
  assert.throws(() => pseudonymKey('', INSTALL_ID_A));
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
  const csv = toCsv({ install_id: INSTALL_ID_A, row_count: 1 }, [
    { pseudonym: 'abc', event_type: 'page.view', timestamp: 't', route: '/a,b' },
  ]);
  const lines = csv.trim().split('\n');
  assert.equal(lines[0], `# install_id=${INSTALL_ID_A}`);
  assert.equal(lines[2], ROW_KEYS.join(','));
  assert.ok(lines[3].includes('"/a,b"'));
});

test('clampDays bounds the export window', () => {
  assert.equal(clampDays(undefined), 30);
  assert.equal(clampDays('0'), 1);
  assert.equal(clampDays('9999'), 365);
  assert.equal(clampDays('7'), 7);
});

test('toCsv flattens one level of meta (window) into key_subkey lines', () => {
  const csv = toCsv(
    { install_id: INSTALL_ID_A, scope: 'teams', window: { days: 7, from: 'F', to: 'T' }, row_count: 0 },
    []
  );
  const lines = csv.trim().split('\n');
  assert.deepEqual(lines.slice(0, 6), [
    `# install_id=${INSTALL_ID_A}`,
    '# scope=teams',
    '# window_days=7',
    '# window_from=F',
    '# window_to=T',
    '# row_count=0',
  ]);
  assert.equal(lines[6], ROW_KEYS.join(','));
  assert.ok(!csv.includes('[object Object]'));
});

test('clampWindowEnd keeps a preview end, defaults to now, rejects bad input with status 400', () => {
  const now = new Date('2026-11-20T12:00:00Z');
  assert.equal(clampWindowEnd(undefined, now).toISOString(), now.toISOString());
  assert.equal(clampWindowEnd('', now).toISOString(), now.toISOString());
  assert.equal(clampWindowEnd('2026-11-19T08:30:00.000Z', now).toISOString(), '2026-11-19T08:30:00.000Z');
  assert.equal(clampWindowEnd('2030-01-01T00:00:00Z', now).toISOString(), now.toISOString(), 'future end becomes now');
  for (const bad of ['nonsense', '2000-01-01T00:00:00Z', ['2026-11-19T08:30:00Z'], 0]) {
    assert.throws(() => clampWindowEnd(bad, now), (e) => e.status === 400, `rejects ${JSON.stringify(bad)}`);
  }
});

test('ROW_KEYS are the published columns, in order', () => {
  assert.deepEqual(ROW_KEYS, ['pseudonym', 'team_role', 'event_type', 'timestamp', 'duration_ms', 'status', 'method', 'route', 'row_count']);
});
