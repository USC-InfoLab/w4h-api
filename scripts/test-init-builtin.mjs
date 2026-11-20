#!/usr/bin/env node
/**
 * End-to-end check of POST /config/init-builtin (ADR-031 §4) against a FRESH
 * built-in install. Talks HTTP only, so it runs on the host next to the kit:
 *
 *   W4H_API_BASE=http://127.0.0.1:5180/api \
 *   W4H_TEST_ENV_FILE=./data/.env \
 *   W4H_TEST_LOG_FILE=./api.log \        # optional: `docker compose logs api > api.log` after the run
 *   node scripts/test-init-builtin.mjs
 *
 * Row counts need W4H_TEST_PSQL (see below).
 *
 * Asserts: first call 201 with a recovery record; second call refused (409);
 * exactly 2 users (Super admin + the person); the person is team_admin of one team;
 * the recovery password signs in; .env is mode 600 with W4H_DB_MODE=builtin;
 * the plain passwords never appear in the api log.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { execSync } from 'node:child_process';

const BASE = (process.env.W4H_API_BASE || 'http://127.0.0.1:2026').replace(/\/$/, '');
const ENV_FILE = process.env.W4H_TEST_ENV_FILE || '';
const LOG_FILE = process.env.W4H_TEST_LOG_FILE || '';

async function call(method, path, body, token) {
  const res = await fetch(`${BASE}${path}`, {
    method,
    headers: {
      'content-type': 'application/json',
      ...(token ? { authorization: `Bearer ${token}` } : {}),
    },
    body: body ? JSON.stringify(body) : undefined,
  });
  const text = await res.text();
  let json = null;
  try { json = JSON.parse(text); } catch { /* not JSON */ }
  return { status: res.status, json, text };
}

const person = {
  name: 'Pilot Tester',
  email: `pilot-${Date.now()}@example.org`,
  password: `Pilot-${Date.now()}-aA1!`,
  load_samples: true,
};

const status = await call('GET', '/config/status');
assert.equal(status.status, 200);
assert.equal(status.json.builtin_available, true, 'builtin_available must be true on a built-in kit');
assert.equal(status.json.requiresAdminSetup || !status.json.configured, true, 'install must be fresh');

const first = await call('POST', '/config/init-builtin', person);
assert.equal(first.status, 201, `init-builtin failed: ${first.text}`);
const { recovery, session, samples } = first.json;
assert.equal(recovery.super_admin_email, 'superadmin@w4h.local');
assert.ok(recovery.super_admin_password?.length >= 20, 'random Super admin password');
assert.ok(recovery.db_password, 'db password in recovery record');
assert.ok(session.token, 'session token');
assert.equal(samples.notebook, 'import_fitbit_csv.ipynb');
const memberships = session.team_memberships || [];
assert.equal(memberships.length, 1, 'person is on exactly one team');
assert.equal(memberships[0].role, 'team_admin', 'person is Admin of their team');

const second = await call('POST', '/config/init-builtin', { ...person, email: `x-${person.email}` });
assert.equal(second.status, 409, `second call must be refused, got ${second.status}`);

const after = await call('GET', '/config/status');
assert.equal(after.json.configured, true);
assert.equal(after.json.requiresAdminSetup, false);
assert.equal(after.json.db_mode, 'builtin');

const login = await call('POST', '/users/login', {
  email: recovery.super_admin_email,
  password: recovery.super_admin_password,
});
assert.ok(login.json?.token, `recovery password must sign in: ${login.text}`);
assert.equal(login.json.role, 'super_admin');

// Row counts straight from Postgres (the kit's db has no host port, so go through docker).
// Kit: W4H_TEST_PSQL='docker compose exec -T db psql -U w4h -d w4h -tAc'
const PSQL = process.env.W4H_TEST_PSQL || '';
if (PSQL) {
  const sql = (q) => execSync(`${PSQL} ${JSON.stringify(q)}`, { encoding: 'utf8' }).trim();
  assert.equal(sql('SELECT count(*) FROM w4h.users'), '2', 'exactly 2 users');
  assert.equal(sql("SELECT count(*) FROM w4h.users WHERE role = 'super_admin'"), '1');
  assert.equal(sql('SELECT count(*) FROM w4h.teams'), '1', 'exactly 1 team');
  assert.equal(sql("SELECT count(*) FROM w4h.team_members WHERE role = 'team_admin'"), '1', 'exactly 1 team_admin');
  assert.equal(sql("SELECT count(*) FROM w4h.audit_log WHERE action = 'platform.bootstrap.single_user'"), '1');
} else {
  console.log('skip: W4H_TEST_PSQL not set (row counts)');
}

const personLogin = await call('POST', '/users/login', { email: person.email, password: person.password });
assert.ok(personLogin.json?.token, 'person signs in with their own password');

if (ENV_FILE) {
  const mode = fs.statSync(ENV_FILE).mode & 0o777;
  assert.equal(mode, 0o600, `.env mode ${mode.toString(8)}`);
  assert.match(fs.readFileSync(ENV_FILE, 'utf8'), /^W4H_DB_MODE=builtin$/m);
} else {
  console.log('skip: W4H_TEST_ENV_FILE not set (.env mode check)');
}

if (LOG_FILE) {
  const log = fs.readFileSync(LOG_FILE, 'utf8');
  assert.ok(!log.includes(recovery.super_admin_password), 'Super admin password leaked to log');
  assert.ok(!log.includes(person.password), 'person password leaked to log');
} else {
  console.log('skip: W4H_TEST_LOG_FILE not set (password-in-log check)');
}

console.log('OK — init-builtin: recovery once, refused twice, 2 users, team_admin, recovery login, env 600');
