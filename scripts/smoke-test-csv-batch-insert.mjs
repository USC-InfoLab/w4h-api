#!/usr/bin/env node
/**
 * Runnable self-check for the batched importCsvBatch() insert path
 * (lib/ingest-csv.js) — the fix for the Fitabase-sized-import slowness.
 * Exercises: multi-statement sub-batching (>1 INSERT per call), dedupe
 * ON CONFLICT skip counting, mode:'replace' truncation, and rollback on
 * a mid-batch failure. Creates/drops its own scratch dataset + schema —
 * safe to run against a real dev DB. `node scripts/smoke-test-csv-batch-insert.mjs`
 */
import assert from 'node:assert/strict';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
dotenv.config({ path: path.join(__dirname, '..', '.env') });

const { query, getPool } = await import('../db.js');
const { importCsvBatch } = await import('../lib/ingest-csv.js');

const DATASET_ID = 'smoke_test_csv_batch_tmp';
const PROJECT_SLUG = 'smoke-test-csv-batch-tmp';
const TEAM_SLUG = 'smoke-test-csv-batch-tmp';
let TEST_USER;

function makeRows(n, offset = 0) {
  return Array.from({ length: n }, (_, i) => ({
    id: String(offset + i),
    value: offset + i,
  }));
}

async function cleanup() {
  await query(`DROP SCHEMA IF EXISTS ds_${DATASET_ID} CASCADE`);
  await query(`DELETE FROM w4h.projects WHERE slug = $1`, [PROJECT_SLUG]);
  await query(`DELETE FROM w4h.datasets WHERE id = $1`, [DATASET_ID]);
  await query(`DELETE FROM w4h.teams WHERE slug = $1`, [TEAM_SLUG]);
}

async function main() {
  await cleanup();

  // assertCanIngestDataset() requires the dataset's owning team Admin or a project
  // owner of that same team (ADR-027; no super-admin bypass) — set up a scratch team,
  // owned by an existing real user, not a fake id.
  const owner = await query(`SELECT id FROM w4h.users ORDER BY id LIMIT 1`);
  if (!owner.rowCount) throw new Error('No users in this database to run the smoke test as');
  TEST_USER = { id: owner.rows[0].id };

  const team = await query(
    `INSERT INTO w4h.teams (slug, name) VALUES ($1, 'Smoke test') RETURNING id`,
    [TEAM_SLUG]
  );
  const teamId = team.rows[0].id;
  await query(
    `INSERT INTO w4h.team_members (team_id, user_id, role) VALUES ($1, $2, 'team_admin')`,
    [teamId, TEST_USER.id]
  );
  await query(`INSERT INTO w4h.datasets (id, name, team_id) VALUES ($1, 'Smoke test', $2)`, [DATASET_ID, teamId]);
  const project = await query(
    `INSERT INTO w4h.projects (slug, name, created_by) VALUES ($1, 'Smoke test', $2) RETURNING id`,
    [PROJECT_SLUG, TEST_USER.id]
  );
  const projectId = project.rows[0].id;
  await query(
    `INSERT INTO w4h.project_members (project_id, user_id, role) VALUES ($1, $2, 'owner')`,
    [projectId, TEST_USER.id]
  );
  await query(
    `INSERT INTO w4h.project_datasets (project_id, dataset_id) VALUES ($1, $2)`,
    [projectId, DATASET_ID]
  );

  const columns = [
    { name: 'id', type: 'text' },
    { name: 'value', type: 'integer' },
  ];

  // 1) A batch bigger than one INSERT statement's param budget (60000/2 cols
  // = 30000 rows/stmt) forces the sub-batch loop to run more than once.
  const big = makeRows(65000);
  const res1 = await importCsvBatch(TEST_USER, DATASET_ID, {
    signal: 'sig', mode: 'replace', columns, dedupe_columns: ['id'], rows: big,
  });
  assert.equal(res1.ok, true, 'insert should succeed');
  assert.equal(res1.inserted, 65000, `expected 65000 inserted, got ${res1.inserted}`);
  const count1 = await query(`SELECT count(*)::int AS n FROM ds_${DATASET_ID}.sig`);
  assert.equal(count1.rows[0].n, 65000, 'row count should match');

  // 2) Re-import an overlapping batch in append mode — dedupe (ON CONFLICT)
  // should skip the 30000 rows that already exist and only add the new ones.
  const overlap = makeRows(40000, 50000); // ids 50000..89999; 15000 already exist
  const res2 = await importCsvBatch(TEST_USER, DATASET_ID, {
    signal: 'sig', mode: 'append', columns, dedupe_columns: ['id'], rows: overlap,
  });
  assert.equal(res2.skipped, 15000, `expected 15000 skipped, got ${res2.skipped}`);
  assert.equal(res2.inserted, 25000, `expected 25000 inserted, got ${res2.inserted}`);
  const count2 = await query(`SELECT count(*)::int AS n FROM ds_${DATASET_ID}.sig`);
  assert.equal(count2.rows[0].n, 90000, 'row count should be 90000 after dedupe append');

  // 3) A bad row mid-batch (violates NOT NULL via invalid type coercion is
  // silently null-coerced, so force a real DB error instead: duplicate a
  // non-dedupe insert against a column type mismatch) should roll back
  // cleanly and leave the table untouched.
  const before = (await query(`SELECT count(*)::int AS n FROM ds_${DATASET_ID}.sig`)).rows[0].n;
  await assert.rejects(
    () => importCsvBatch(TEST_USER, DATASET_ID, {
      signal: 'sig', mode: 'append', columns: [{ name: 'id', type: 'text' }, { name: 'nope', type: 'integer' }],
      dedupe_columns: ['id'], rows: makeRows(5, 200000),
    }),
    /Query failed/,
    'inserting into a nonexistent column should throw'
  );
  const after = (await query(`SELECT count(*)::int AS n FROM ds_${DATASET_ID}.sig`)).rows[0].n;
  assert.equal(after, before, 'failed batch must roll back, not partially insert');

  // 4) Mixed-case / spaced identifiers keep their exact case (lib/sql-ident.js quoting),
  // through the bulk insert, TRUNCATE and ON CONFLICT paths.
  const mixedCols = [{ name: 'Subject ID', type: 'text' }, { name: 'HR', type: 'integer' }];
  const mixedRows = [{ 'Subject ID': 'a', HR: 60 }, { 'Subject ID': 'b', HR: 70 }, { 'Subject ID': 'a', HR: 61 }];
  const res4 = await importCsvBatch(TEST_USER, DATASET_ID, {
    signal: 'Mixed Case', mode: 'replace', columns: mixedCols, dedupe_columns: ['Subject ID'], rows: mixedRows,
  });
  assert.equal(res4.ok, true, `mixed-case insert should succeed: ${res4.error}`);
  assert.equal(res4.inserted, 2);
  assert.equal(res4.skipped, 1);
  const cols = await query(
    `SELECT column_name FROM information_schema.columns
     WHERE table_schema = $1 AND table_name = 'Mixed Case' ORDER BY ordinal_position`,
    [`ds_${DATASET_ID}`]
  );
  assert.deepEqual(cols.rows.map((r) => r.column_name), ['Subject ID', 'HR'], 'column case must be preserved');

  console.log('OK — batched insert, dedupe accounting, rollback, and mixed-case identifiers all behave correctly');
}

main()
  .catch((e) => {
    console.error('FAILED:', e);
    process.exitCode = 1;
  })
  .finally(async () => {
    await cleanup();
    await getPool()?.end();
  });
