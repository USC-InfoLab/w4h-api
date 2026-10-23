#!/usr/bin/env node
/** Exercise unknown dedupe_columns through a running API using disposable DB rows. */
import assert from 'node:assert/strict';
import { randomBytes } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import dotenv from 'dotenv';
import jwt from 'jsonwebtoken';

dotenv.config({ path: fileURLToPath(new URL('../.env', import.meta.url)) });
const { query, getPool } = await import('../db.js');
const { quoteIdent } = await import('../lib/sql-ident.js');

const api = (process.env.T3_07_API || 'http://127.0.0.1:2107').replace(/\/$/, '');
const suffix = randomBytes(5).toString('hex');
const datasetId = `t307_${suffix}`;
const teamSlug = `t307-${suffix}`;
const schema = `ds_${datasetId}`;
let teamId;
let createdUserId;

async function post(token, body) {
  const response = await fetch(`${api}/datasets/${datasetId}/import/csv-batch`, {
    method: 'POST',
    headers: { authorization: `Bearer ${token}`, 'content-type': 'application/json' },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(15000),
  });
  const text = await response.text();
  return { status: response.status, body: JSON.parse(text) };
}

try {
  assert.ok(process.env.TOKEN_SECRET, 'TOKEN_SECRET must be set in .env');
  const owner = await query(
    `SELECT id, name, email, role FROM w4h.users
     WHERE COALESCE(status, 'active') NOT IN ('invited', 'discontinued') ORDER BY id LIMIT 1`
  );
  let user = owner.rows[0];
  if (!user) {
    const created = await query(
      `INSERT INTO w4h.users (name, email, password, role)
       VALUES ('T3-07 test', $1, '!test-only', 'project_user')
       RETURNING id, name, email, role`,
      [`t307-${suffix}@example.invalid`]
    );
    user = created.rows[0];
    createdUserId = user.id;
  }
  const token = jwt.sign(user, process.env.TOKEN_SECRET, { expiresIn: '10m' });

  const team = await query(
    `INSERT INTO w4h.teams (slug, name) VALUES ($1, 'T3-07 test') RETURNING id`, [teamSlug]
  );
  teamId = team.rows[0].id;
  await query(
    `INSERT INTO w4h.team_members (team_id, user_id, role) VALUES ($1, $2, 'team_admin')`,
    [teamId, user.id]
  );
  await query(
    `INSERT INTO w4h.datasets (id, name, team_id) VALUES ($1, 'T3-07 test', $2)`,
    [datasetId, teamId]
  );

  const payload = {
    signal: 'Heart Rate Test',
    mode: 'replace',
    columns: [{ name: 'Subject ID', type: 'text' }, { name: 'HR', type: 'integer' }],
    dedupe_columns: ['Subject ID'],
    rows: [
      { 'Subject ID': 's1', HR: 61 },
      { 'Subject ID': 's2', HR: 63 },
      { 'Subject ID': 's3', HR: 70 },
    ],
  };
  const seed = await post(token, payload);
  assert.equal(seed.status, 200, `seed failed: ${JSON.stringify(seed.body)}`);
  assert.equal(seed.body.inserted, 3);

  const bad = await post(token, { ...payload, dedupe_columns: ['nope'] });
  assert.equal(bad.status, 400, `unexpected response: ${JSON.stringify(bad.body)}`);
  assert.deepEqual(bad.body, { error: 'dedupe_columns: unknown column "nope"' });
  const empty = await post(token, { ...payload, dedupe_columns: [''] });
  assert.equal(empty.status, 400, `empty name response: ${JSON.stringify(empty.body)}`);
  assert.deepEqual(empty.body, { error: 'dedupe_columns: unknown column ""' });
  const count = await query(`SELECT count(*)::int AS n FROM ${quoteIdent(schema)}."Heart Rate Test"`);
  assert.equal(count.rows[0].n, 3, 'failed replace changed the existing rows');
  console.log('PASS T3-07: HTTP 400 for unknown/empty dedupe names; existing rows still 3');
} finally {
  try {
    await query('DELETE FROM w4h.tables WHERE dataset_id = $1', [datasetId]);
    await query(`DROP SCHEMA IF EXISTS ${quoteIdent(schema)} CASCADE`);
    await query('DELETE FROM w4h.datasets WHERE id = $1', [datasetId]);
    if (teamId) await query('DELETE FROM w4h.teams WHERE id = $1', [teamId]);
    if (createdUserId) await query('DELETE FROM w4h.users WHERE id = $1', [createdUserId]);
  } finally {
    await getPool()?.end();
  }
}
