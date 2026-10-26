#!/usr/bin/env node
/**
 * Real T2-09 integration check, using the runnable Fitbit file importer and a
 * unique disposable dataset inside the existing w4h-db database. Uses this
 * worktree's API/environment and an existing team admin; never drops the DB.
 * Requires the importer's installed .venv. See verify-t2-09.md for configuration.
 *
 *   RUNS=10 node scripts/verify-fitbit-file-run.mjs
 *   FITBIT_IMPORTER_ROOT=/path/to/w4h-fitbit-file-csv-importer RUNS=20 node ...
 */
import assert from 'node:assert/strict';
import { createHash, randomUUID } from 'node:crypto';
import { spawn, spawnSync } from 'node:child_process';
import { once } from 'node:events';
import fs from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import dotenv from 'dotenv';
import jwt from 'jsonwebtoken';
import pg from 'pg';
import { parse as parseYaml } from 'yaml';
import { quoteIdent } from '../lib/sql-ident.js';
import { envFilePath } from '../lib/env-path.js';
import { decryptUserApiKey } from '../lib/utils/apiKeyCrypto.js';

const API_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const ENV_FILE = envFilePath();
dotenv.config({ path: ENV_FILE, quiet: true });
const CATALOG_SLUG = 'w4h-fitbit-csv-file-importer';
const IMPORTER_ROOT = path.resolve(process.env.FITBIT_IMPORTER_ROOT || path.join(API_ROOT, '..', 'w4h-fitbit-file-csv-importer'));
const FIXTURES = ['heartrate_seconds_merged.csv', 'subjects.csv'];
const RUNS = Number(process.env.RUNS || 10);
const DATASET_ID = `t209_${randomUUID().replaceAll('-', '')}`;
const SCHEMA = `ds_${DATASET_ID}`;
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

if (!Number.isInteger(RUNS) || RUNS < 1 || RUNS > 50) throw new Error('RUNS must be an integer from 1 to 50');
for (const key of ['DB_HOST', 'DB_PORT', 'DB_USER', 'DB_PASSWORD', 'DB_DATABASE', 'TOKEN_SECRET']) {
  if (!process.env[key]) throw new Error(`${key} is required in the API environment`);
}
assert.equal(process.env.DB_DATABASE, 'w4h-db', 'This verifier requires DB_DATABASE=w4h-db');
const authMode = process.env.AUTH_MODE || 'mint';
assert.ok(['mint', 'token'].includes(authMode), 'AUTH_MODE must be mint or token');
const importerVenv = path.join(IMPORTER_ROOT, '.venv');
const installed = path.join(importerVenv, '.installed');
const pyproject = path.join(IMPORTER_ROOT, 'pyproject.toml');
if (!fs.existsSync(installed) || !fs.existsSync(path.join(importerVenv, 'bin', 'w4h-fitbit-file-csv-import'))) {
  throw new Error(`Install the importer's .venv first: ${IMPORTER_ROOT}/run.sh plan with W4H_DATA_DIR set`);
}
if (fs.statSync(pyproject).mtimeMs > fs.statSync(installed).mtimeMs) {
  throw new Error('Importer pyproject.toml is newer than .venv/.installed; refresh its environment before this check');
}
const packagePath = spawnSync(path.join(importerVenv, 'bin', 'python'),
  ['-c', 'import w4h_fitbit_file_csv_importer as m; print(m.__file__)'], { encoding: 'utf8' });
if (packagePath.error) throw packagePath.error;
assert.equal(packagePath.status, 0, packagePath.stderr);
assert.ok(path.resolve(packagePath.stdout.trim()).startsWith(`${IMPORTER_ROOT}${path.sep}`),
  `Importer .venv points at another checkout: ${packagePath.stdout.trim()}`);

const { Pool } = pg;
const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'w4h-t209-real-'));
const runnerDir = path.join(tempRoot, CATALOG_SLUG);
const uploadsRoot = path.join(tempRoot, 'uploads');
const dbConfig = { host: process.env.DB_HOST, port: process.env.DB_PORT,
  user: process.env.DB_USER, password: process.env.DB_PASSWORD };
const scratchPool = new Pool({ ...dbConfig, database: process.env.DB_DATABASE });
let api;
let apiStopped = false;
let apiLog = '';
let schemaCreated = false;
let datasetCreated = false;
let passed = 0;

async function localPort() {
  const listener = net.createServer();
  listener.listen(0, '127.0.0.1');
  await once(listener, 'listening');
  const port = listener.address().port;
  await new Promise((resolve) => listener.close(resolve));
  return port;
}

async function request(base, method, route, token, body, contentType) {
  const response = await fetch(`${base}${route}`, {
    method,
    headers: { authorization: `Bearer ${token}`, ...(contentType ? { 'content-type': contentType } : {}) },
    body,
    signal: AbortSignal.timeout(30_000),
  });
  const text = await response.text();
  let json;
  try { json = JSON.parse(text); } catch { json = null; }
  return { status: response.status, json, text };
}

function runPlan(uploadDir, command) {
  const result = spawnSync(path.join(runnerDir, 'run.sh'), [command], {
    cwd: runnerDir, env: { ...process.env, W4H_DATA_DIR: uploadDir }, encoding: 'utf8', timeout: 60_000,
  });
  if (result.error) throw result.error;
  assert.equal(result.status, 0, `${command}: ${(result.stdout || '') + (result.stderr || '')}`);
}

try {
  const database = (await scratchPool.query('SELECT current_database() AS name')).rows[0].name;
  assert.equal(database, 'w4h-db');
  const suppliedToken = authMode === 'token'
    ? jwt.verify(process.env.TOKEN_TA || '', process.env.TOKEN_SECRET) : null;
  const userId = suppliedToken ? suppliedToken.id : Number(process.env.TA_ID || 3);
  const user = (await scratchPool.query(
    'SELECT id, name, email, role, status, api_key_encrypted, api_key_hash FROM w4h.users WHERE id = $1', [userId]
  )).rows[0];
  assert.equal(user?.status, 'active', 'TA_ID/TOKEN_TA must identify an active user in w4h-db');
  // Refuse to rotate a real user's API key if the environment is stale.
  const apiKey = decryptUserApiKey(user.api_key_encrypted);
  assert.ok(apiKey, 'User API key cannot be decrypted: check this API environment/wrap secret');
  assert.equal(createHash('sha256').update(apiKey).digest('hex'), user.api_key_hash,
    'Wrapped API key does not match the stored key hash');
  const teams = (await scratchPool.query(
    "SELECT team_id FROM w4h.team_members WHERE user_id = $1 AND role = 'team_admin' AND ($2::uuid IS NULL OR team_id = $2)",
    [user.id, process.env.TEAM_ID || null]
  )).rows;
  assert.equal(teams.length, 1, 'Set TEAM_ID to one team administered by TA_ID/TOKEN_TA');
  const catalog = (await scratchPool.query('SELECT status, runnable FROM w4h.contributions WHERE slug = $1', [CATALOG_SLUG])).rows[0];
  assert.equal(catalog?.status, 'published', `${CATALOG_SLUG} must be published`);
  assert.equal(catalog?.runnable, true, `${CATALOG_SLUG} is not runnable in w4h-db`);
  await scratchPool.query(`CREATE SCHEMA ${quoteIdent(SCHEMA)}`);
  schemaCreated = true;
  await scratchPool.query("INSERT INTO w4h.datasets (id, name, team_id, read_only) VALUES ($1, 'T2-09 scratch', $2, false)", [DATASET_ID, teams[0].team_id]);
  datasetCreated = true;

  // The catalog slug differs from the checkout name. The installed editable
  // package still imports code from the checkout; only run.sh is copied here.
  fs.mkdirSync(runnerDir);
  fs.copyFileSync(path.join(IMPORTER_ROOT, 'run.sh'), path.join(runnerDir, 'run.sh'));
  fs.chmodSync(path.join(runnerDir, 'run.sh'), 0o755);
  fs.copyFileSync(pyproject, path.join(runnerDir, 'pyproject.toml'));
  const stat = fs.statSync(pyproject);
  fs.utimesSync(path.join(runnerDir, 'pyproject.toml'), stat.atime, stat.mtime);
  fs.symlinkSync(importerVenv, path.join(runnerDir, '.venv'), 'dir');

  const port = await localPort();
  const base = `http://127.0.0.1:${port}`;
  api = spawn(process.execPath, ['app.js'], {
    cwd: API_ROOT,
    env: { ...process.env, W4H_ENV_FILE: ENV_FILE, PORT: String(port),
      CONTRIBUTION_RUNNER_API_BASE: base,
      CONTRIBUTION_RUNNER_REPO_ROOT: tempRoot, CONTRIBUTION_UPLOADS_ROOT: uploadsRoot },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  api.on('error', (error) => { apiLog += `API spawn failed: ${error.message}\n`; apiStopped = true; });
  api.on('exit', () => { apiStopped = true; });
  api.stdout.on('data', (chunk) => { apiLog += chunk.toString(); });
  api.stderr.on('data', (chunk) => { apiLog += chunk.toString(); });
  const readyBy = Date.now() + 30_000;
  while (!apiLog.includes('Database schema ensured')) {
    if (apiStopped || apiLog.includes('Schema ensure skipped/failed') || Date.now() > readyBy) {
      throw new Error(`API did not initialize: ${apiLog.slice(-2000)}`);
    }
    await sleep(100);
  }

  const token = suppliedToken ? process.env.TOKEN_TA
    : jwt.sign({ id: user.id, name: user.name, email: user.email, role: user.role }, process.env.TOKEN_SECRET, { expiresIn: '1h' });
  console.log(`CONFIG api=${base} pid=${api.pid} node=${process.version} worktree=${API_ROOT}`);
  console.log(`CONFIG db=${database} host=${dbConfig.host} port=${dbConfig.port} auth=${authMode} user=${user.id} team=${teams[0].team_id} dataset=${DATASET_ID}`);
  console.log(`CONFIG importer=${IMPORTER_ROOT} catalog_slug=${CATALOG_SLUG} env=${ENV_FILE}`);
  for (let n = 0; n < RUNS; n++) {
    const created = await request(base, 'POST', `/contributions/${CATALOG_SLUG}/uploads`, token,
      JSON.stringify({ dataset_id: DATASET_ID }), 'application/json');
    assert.equal(created.status, 201, `upload ${n + 1}: ${created.text}`);
    const uploadId = created.json.upload_id;
    const uploadDir = path.join(uploadsRoot, uploadId);
    for (const file of FIXTURES) {
      const uploaded = await request(base, 'PUT', `/contributions/uploads/${uploadId}/files/${file}`, token,
        fs.readFileSync(path.join(IMPORTER_ROOT, 'tests', 'fixtures', file)), 'text/csv');
      assert.equal(uploaded.status, 200, `file ${file}: ${uploaded.text}`);
    }
    runPlan(uploadDir, 'plan');
    const manifest = parseYaml(fs.readFileSync(path.join(uploadDir, 'manifest.yaml'), 'utf8'));
    assert.deepEqual(manifest.signals.map((signal) => signal.slug), ['heartrate_seconds', 'subjects']);
    assert.equal(manifest.signals[0].geomts_mapping.subject, 'Id');
    assert.deepEqual(manifest.signals[0].geomts_mapping.time, { column: 'Time' });
    assert.equal(manifest.signals[1].geomts_mapping.subject, 'id');
    runPlan(uploadDir, 'validate');

    const started = await request(base, 'POST', `/contributions/${CATALOG_SLUG}/run`, token,
      JSON.stringify({ dataset_id: DATASET_ID, upload_id: uploadId, mode: 'update' }), 'application/json');
    assert.equal(started.status, 202, `run ${n + 1}: ${started.text}`);
    let final;
    const deadline = Date.now() + 60_000;
    for (;;) {
      const response = await request(base, 'GET', `/contributions/runs/${started.json.id}`, token);
      assert.equal(response.status, 200, `poll ${n + 1}: ${response.text}`);
      final = response.json;
      if (['succeeded', 'failed'].includes(final.status)) break;
      if (Date.now() > deadline) throw new Error(`run ${n + 1} did not finish`);
      await sleep(40);
    }
    assert.equal(final.status, 'succeeded', `run ${n + 1}: ${final.log}`);
    assert.equal(final.exit_code, 0);
    // import_all prints chunk lines as it runs, then the CLI prints summaries.
    const markers = ['heartrate_seconds: chunk', 'subjects: chunk', 'heartrate_seconds: inserted=', 'subjects: inserted='];
    let index = -1;
    for (const marker of markers) {
      const next = final.log.indexOf(marker, index + 1);
      assert.ok(next > index, `run ${n + 1} missing/out-of-order ${marker}: ${final.log}`);
      index = next;
    }
    assert.equal(final.progress?.signal, 'subjects');
    assert.equal(final.progress?.signal_count, 2);
    assert.equal(final.progress?.rows_done, 1);
    assert.ok(!final.log.includes('@@PROGRESS@@'), 'progress markers belong in progress, not the log');
    assert.ok(!final.log.includes('run output incomplete'), 'log persistence failed');
    assert.ok(!fs.existsSync(uploadDir), 'completed upload should already be removed');
    // Prove the HTTP response and SQL assertions use this same database/run.
    const stored = (await scratchPool.query(
      'SELECT status, log, progress FROM w4h.contribution_runs WHERE id = $1 AND dataset_id = $2',
      [started.json.id, DATASET_ID]
    )).rows[0];
    assert.deepEqual(stored, { status: final.status, log: final.log, progress: final.progress });
    passed++;
  }

  const tables = (await scratchPool.query(
    'SELECT name, physical_schema, physical_table FROM w4h.tables WHERE dataset_id = $1 ORDER BY name', [DATASET_ID]
  )).rows;
  assert.deepEqual(tables.map((row) => row.name), ['heartrate_seconds', 'subjects']);
  for (const row of tables) {
    assert.equal(row.physical_schema, SCHEMA);
    assert.ok(row.physical_table.startsWith('fitbit_'), `missing Fitbit table prefix: ${row.physical_table}`);
    const count = (await scratchPool.query(
      `SELECT count(*)::int AS n FROM ${quoteIdent(row.physical_schema)}.${quoteIdent(row.physical_table)}`
    )).rows[0].n;
    assert.equal(count, row.name === 'subjects' ? 1 : 3, `unexpected row count for ${row.name}`);
  }
  const keyAfter = (await scratchPool.query('SELECT api_key_encrypted, api_key_hash FROM w4h.users WHERE id = $1', [user.id])).rows[0];
  assert.deepEqual(keyAfter, { api_key_encrypted: user.api_key_encrypted, api_key_hash: user.api_key_hash }, 'Existing user API key changed');
  console.log(`PASS real ${CATALOG_SLUG}: ${passed}/${RUNS} runs; terminal log markers, progress, cleanup and table rows verified in ${database}`);
} catch (e) {
  console.error(`FAIL real Fitbit importer after ${passed}/${RUNS} runs: ${e.message}`);
  if (apiLog) console.error(apiLog.slice(-3000));
  process.exitCode = 1;
} finally {
  if (api && !apiStopped) {
    api.kill('SIGTERM');
    for (let i = 0; i < 50 && !apiStopped; i++) await sleep(100);
    if (!apiStopped) {
      api.kill('SIGKILL');
      for (let i = 0; i < 50 && !apiStopped; i++) await sleep(100);
    }
  }
  try {
    assert.ok(!api || apiStopped, 'API did not stop; retaining fixtures for inspection');
    if (schemaCreated) await scratchPool.query(`DROP SCHEMA ${quoteIdent(SCHEMA)} CASCADE`);
    if (datasetCreated) await scratchPool.query('DELETE FROM w4h.datasets WHERE id = $1', [DATASET_ID]);
    fs.rmSync(tempRoot, { recursive: true, force: true });
    console.log(`CLEANUP dataset=${DATASET_ID}: fixture schema, dataset/run rows and files removed; API stopped by PID; w4h-db and audit history retained`);
  } finally {
    await scratchPool.end();
  }
}
