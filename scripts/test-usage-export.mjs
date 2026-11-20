#!/usr/bin/env node
/**
 * End-to-end check of the ADR-031 §7 usage export against a FRESH built-in install
 * (it runs init-builtin itself). HTTP plus psql, like test-init-builtin.mjs:
 *
 *   W4H_API_BASE=http://127.0.0.1:5180/api \
 *   W4H_TEST_ENV_FILE=./data/.env \
 *   W4H_TEST_PSQL='docker compose exec -T db psql -U w4h -d w4h -tAc' \
 *   W4H_TEST_MAILPIT=http://127.0.0.1:8025 \   # optional: api has SMTP -> Mailpit + TELEMETRY_TO
 *   node scripts/test-usage-export.mjs
 *
 * Asserts: Super admin exports the whole platform; an Admin only events of members of the
 * teams they administer; a member gets 403; JSON shape (meta.scope, meta.window, columns);
 * CSV meta lines and filename; a re-export with the preview's window.to returns the same rows;
 * Send returns 503 without mail/TELEMETRY_TO, or {success,row_count,format} with an attachment
 * holding row_count data lines; audit rows carry scope + window; no '@' anywhere.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { execSync } from 'node:child_process';
import bcrypt from 'bcrypt';
import dotenv from 'dotenv';
import { pseudonym, pseudonymKey, ROW_KEYS } from '../lib/usage-export.js';

const BASE = (process.env.W4H_API_BASE || 'http://127.0.0.1:2026').replace(/\/$/, '');
const ENV_FILE = process.env.W4H_TEST_ENV_FILE || '';
const PSQL = process.env.W4H_TEST_PSQL || '';
const MAILPIT = (process.env.W4H_TEST_MAILPIT || '').replace(/\/$/, '');
assert.ok(ENV_FILE && PSQL, 'W4H_TEST_ENV_FILE and W4H_TEST_PSQL are required');

// Single-quote for the shell: bcrypt hashes contain `$`.
const shq = (v) => `'${v.replace(/'/g, `'\\''`)}'`;
const sql = (q) => execSync(`${PSQL} ${shq(q)}`, { encoding: 'utf8' }).trim();

async function call(method, path, { body, token, raw } = {}) {
  const res = await fetch(`${BASE}${path}`, {
    method,
    headers: { 'content-type': 'application/json', ...(token ? { authorization: `Bearer ${token}` } : {}) },
    body: body ? JSON.stringify(body) : undefined,
  });
  const text = await res.text();
  let json = null;
  if (!raw) { try { json = JSON.parse(text); } catch { /* not JSON */ } }
  return { status: res.status, json, text, headers: res.headers };
}

const stamp = Date.now();
const person = { name: 'Pilot Admin', email: `pa-${stamp}@example.org`, password: `Pilot-${stamp}-aA1!`, load_samples: false };
const setup = await call('POST', '/config/init-builtin', { body: person });
assert.equal(setup.status, 200, `init-builtin (needs a fresh install): ${setup.text}`);
const superAdmin = setup.json.recovery;
const personToken = setup.json.session.token;
const t1 = sql(`SELECT team_id FROM w4h.team_members WHERE role = 'team_admin' LIMIT 1`);

// Three more accounts: M member of T1; B team_admin of a new team T2; O member of T2.
const pw = `Other-${stamp}-aA1!`;
const hash = bcrypt.hashSync(pw, 4);
const mkUser = (tag) => sql(
  `INSERT INTO w4h.users (name, email, password, role, status) VALUES ('${tag}', '${tag}-${stamp}@example.org', '${hash}', 'project_user', 'active') RETURNING id`
).split('\n')[0];
const m = mkUser('m');
const b = mkUser('b');
const o = mkUser('o');
const c = mkUser('c');
const t2 = sql(`INSERT INTO w4h.teams (name, slug) VALUES ('T2', 't2-${stamp}') RETURNING id`).split('\n')[0];
sql(`INSERT INTO w4h.team_members (team_id, user_id, role) VALUES ('${t1}', ${m}, 'member'), ('${t2}', ${b}, 'team_admin'), ('${t2}', ${o}, 'member'), ('${t1}', ${c}, 'team_admin'), ('${t2}', ${c}, 'team_admin')`);

async function login(email, password) {
  const r = await call('POST', '/users/login', { body: { email, password } });
  assert.ok(r.json?.token, `login ${email}: ${r.text}`);
  return r.json.token;
}
const tokens = {
  super: await login(superAdmin.super_admin_email, superAdmin.super_admin_password),
  person: personToken,
  m: await login(`m-${stamp}@example.org`, pw),
  b: await login(`b-${stamp}@example.org`, pw),
  o: await login(`o-${stamp}@example.org`, pw),
  c: await login(`c-${stamp}@example.org`, pw),
};
// One JWT request each, so every account has api.request rows.
for (const t of Object.values(tokens)) await call('GET', '/teams', { token: t });
await new Promise((r) => setTimeout(r, 300)); // usage rows are written on response 'finish'

const ids = {
  super: sql(`SELECT id FROM w4h.users WHERE email = '${superAdmin.super_admin_email}'`),
  person: sql(`SELECT id FROM w4h.users WHERE email = '${person.email}'`),
  m, b, o, c,
};
const exportJson = async (token, extra = '') => call('GET', `/config/usage-export?days=1&format=json${extra}`, { token });

// Super admin: platform scope, shape.
const sa = await exportJson(tokens.super);
assert.equal(sa.status, 200, sa.text);
assert.equal(sa.json.meta.schema, 'w4h-usage-export/1');
assert.equal(sa.json.meta.scope, 'platform');
assert.deepEqual(Object.keys(sa.json.meta.window), ['days', 'from', 'to']);
assert.equal(sa.json.meta.window.days, 1);
assert.deepEqual(sa.json.columns, ROW_KEYS);
assert.equal(sa.json.meta.row_count, sa.json.rows.length);
for (const r of sa.json.rows) assert.deepEqual(Object.keys(r), ROW_KEYS);
assert.ok(!sa.text.includes('@'), 'no emails in the export');

const env = dotenv.parse(fs.readFileSync(ENV_FILE)); // after the first export wrote W4H_INSTALL_ID
const installId = env.W4H_INSTALL_ID;
assert.ok(installId, 'W4H_INSTALL_ID saved to .env on first export');
const key = pseudonymKey(env.TOKEN_SECRET, installId);
const ps = Object.fromEntries(Object.entries(ids).map(([k, v]) => [k, pseudonym(key, v)]));
const seen = (rows) => new Set(rows.map((r) => r.pseudonym));
const saSeen = seen(sa.json.rows);
for (const k of Object.keys(ids)) assert.ok(saSeen.has(ps[k]), `platform export has ${k}`);

// Admin of T1 (the person): only T1 members (person + m).
const pa = await exportJson(tokens.person);
assert.equal(pa.status, 200, pa.text);
assert.equal(pa.json.meta.scope, 'teams');
const paSeen = seen(pa.json.rows);
assert.ok(paSeen.has(ps.person) && paSeen.has(ps.m), 'Admin sees own team');
for (const k of ['super', 'b', 'o']) assert.ok(!paSeen.has(ps[k]), `Admin of T1 must not see ${k}`);

// Admin of T2 (b): only b + o.
const ba = await exportJson(tokens.b);
const baSeen = seen(ba.json.rows);
assert.ok(baSeen.has(ps.b) && baSeen.has(ps.o));
for (const k of ['super', 'person', 'm']) assert.ok(!baSeen.has(ps[k]), `Admin of T2 must not see ${k}`);

// Admin of both teams (c): everyone except the Super admin (exercises ANY() with 2 team ids).
const cSeen = seen((await exportJson(tokens.c)).json.rows);
for (const k of ['person', 'm', 'b', 'o', 'c']) assert.ok(cSeen.has(ps[k]), `2-team Admin sees ${k}`);
assert.ok(!cSeen.has(ps.super), '2-team Admin must not see the Super admin');

// Members: 403 on both routes, error as a string (the app reads it as one).
for (const t of [tokens.m, tokens.o]) {
  const g = await exportJson(t);
  assert.equal(g.status, 403);
  assert.equal(typeof g.json.error, 'string');
  assert.equal((await call('POST', '/config/usage-export/send', { token: t, body: { days: 1 } })).status, 403);
}

// Bad window_end: 400 with a string error, never a silently different window.
for (const bad of ['nonsense', '2000-01-01T00:00:00Z']) {
  const r = await exportJson(tokens.person, `&window_end=${encodeURIComponent(bad)}`);
  assert.equal(r.status, 400, `window_end=${bad}`);
  assert.equal(typeof r.json.error, 'string');
}

// Preview = what is sent: re-export with the preview's window.to selects the same rows,
// although more events (the exports above) happened since.
await new Promise((r) => setTimeout(r, 500)); // let earlier requests' usage rows commit (written on 'finish')
const preview = await exportJson(tokens.person);
await call('GET', '/teams', { token: tokens.person });
await new Promise((r) => setTimeout(r, 300));
const again = await exportJson(tokens.person, `&window_end=${encodeURIComponent(preview.json.meta.window.to)}`);
assert.deepEqual(again.json.rows, preview.json.rows, "same window_end -> same rows");
assert.equal(again.json.meta.window.to, preview.json.meta.window.to);
const later = await exportJson(tokens.person);
assert.ok(later.json.rows.length > preview.json.rows.length, 'without window_end newer events are included');

// CSV download.
const csv = await call('GET', '/config/usage-export?days=1&format=csv', { token: tokens.person, raw: true });
assert.equal(csv.status, 200);
assert.match(csv.headers.get('content-type'), /^text\/csv/);
assert.match(csv.headers.get('content-disposition'), /^attachment; filename="w4h-usage-[0-9a-f]{8}-\d{4}-\d{2}-\d{2}\.csv"$/);
const csvLines = csv.text.trim().split('\n');
for (const k of ['schema', 'install_id', 'version', 'db_mode', 'scope', 'window_days', 'window_from', 'window_to', 'row_count']) {
  assert.ok(csvLines.some((l) => l.startsWith(`# ${k}=`)), `CSV meta line ${k}`);
}
assert.ok(!csv.text.includes('[object Object]'));
assert.ok(csvLines.includes(ROW_KEYS.join(',')));

// Send.
const status = (await call('GET', '/config/status')).json;
const sendBody = { days: 1, format: 'csv', window_end: preview.json.meta.window.to, expected_row_count: preview.json.meta.row_count };
const noWindow = await call('POST', '/config/usage-export/send', { token: tokens.person, body: { days: 1 } });
if (!MAILPIT) {
  assert.equal(status.telemetry_configured, false, 'api can send: set W4H_TEST_MAILPIT to test the Send path');
  const send = await call('POST', '/config/usage-export/send', { token: tokens.person, body: sendBody });
  assert.equal(send.status, 503, send.text);
  assert.equal(typeof send.json.error, 'string', 'Send 503 error is a string');
  console.log('note: W4H_TEST_MAILPIT not set: only the Send 503 path was checked');
} else {
  assert.equal(status.telemetry_configured, true, 'Mailpit run needs SMTP + TELEMETRY_TO on the api');
  assert.equal(noWindow.status, 400, 'Send without window_end is refused');
  assert.equal(typeof noWindow.json.error, 'string');
  const stale = await call('POST', '/config/usage-export/send', { token: tokens.person, body: { ...sendBody, expected_row_count: sendBody.expected_row_count + 1 } });
  assert.equal(stale.status, 409, 'row count changed since preview -> 409');

  const send = await call('POST', '/config/usage-export/send', { token: tokens.person, body: sendBody });
  assert.equal(send.status, 200, send.text);
  assert.deepEqual(send.json, { success: true, row_count: preview.json.rows.length, format: 'csv' });
  await new Promise((r) => setTimeout(r, 500));
  const attachmentOf = async (subjectPart, ext) => {
    const list = await (await fetch(`${MAILPIT}/api/v1/messages`)).json();
    const msg = list.messages.find((x) => x.Subject.startsWith('[W4H usage]') && x.Subject.includes(subjectPart) && x.Attachments > 0
      && !seenMail.has(x.ID));
    assert.ok(msg, 'usage mail delivered');
    seenMail.add(msg.ID);
    const full = await (await fetch(`${MAILPIT}/api/v1/message/${msg.ID}`)).json();
    const att = full.Attachments?.[0];
    assert.ok(att && att.FileName.endsWith(ext), `${ext} attachment`);
    return (await fetch(`${MAILPIT}/api/v1/message/${msg.ID}/part/${att.PartID}`)).text();
  };
  const seenMail = new Set();
  const body = await attachmentOf(' teams ', '.csv');
  const dataLines = body.trim().split('\n').filter((l) => !l.startsWith('#')).slice(1);
  assert.equal(dataLines.length, send.json.row_count, 'attachment rows = row_count = preview rows');
  assert.ok(!body.includes('@'));
  const sent = JSON.parse(sql(`SELECT details::text FROM w4h.audit_log WHERE action = 'usage.sent' ORDER BY id DESC LIMIT 1`));
  assert.equal(sent.scope, 'teams');
  assert.equal(sent.format, 'csv');
  assert.equal(sent.window.to, preview.json.meta.window.to, 'Send used the preview window');

  // Default format is json.
  const js = await call('POST', '/config/usage-export/send', { token: tokens.person, body: { days: 1, window_end: preview.json.meta.window.to } });
  assert.equal(js.status, 200, js.text);
  assert.equal(js.json.format, 'json');
  await new Promise((r) => setTimeout(r, 500));
  const jbody = JSON.parse(await attachmentOf(' teams ', '.json'));
  assert.deepEqual(jbody.columns, ROW_KEYS);
  assert.deepEqual(jbody.rows, preview.json.rows);
}

// Audit rows: scope + window, never rows or emails.
const audit = sql(`SELECT details::text FROM w4h.audit_log WHERE action = 'usage.exported' ORDER BY id DESC LIMIT 1`);
const d = JSON.parse(audit);
assert.deepEqual(Object.keys(d).sort(), ['format', 'row_count', 'scope', 'window']);
assert.deepEqual(Object.keys(d.window).sort(), ['days', 'from', 'to']); // jsonb reorders keys
assert.equal(d.scope, 'teams');
assert.equal(d.format, 'csv'); // last export was the person's CSV download

console.log('OK — usage export: platform/teams scope, member 403, shape, CSV, preview = send, audit');
