import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import { after, before, test } from 'node:test';
import { fileURLToPath } from 'node:url';
import dotenv from 'dotenv';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
dotenv.config({ path: path.join(__dirname, '../.env') });
const enabled = process.env.W4H_INTEGRATION_TEST === '1';
const databaseName = process.env.DB_DATABASE || '';

if (enabled && !/(^|[_-])test($|[_-])/i.test(databaseName)) {
  throw new Error('Refusing to run dataset integration tests against a non-test database');
}

if (enabled) process.env.DB_POOL_MAX = '1';
const teamA = randomUUID();
const teamB = randomUUID();
const datasetId = `isolation-${randomUUID()}`;
const ingestDatasetId = `ingest-${randomUUID()}`;
const failedIngestDatasetId = `failed-ingest-${randomUUID()}`;
const realSampleDatasetId = `real-sample-${randomUUID()}`;
const grantDatasetId = `grant-${randomUUID()}`;
const projectAId = randomUUID();
const projectBId = randomUUID();
const fitbitMapping = {
  device_type: 'fitbit',
  subjects: {
    id: 'id',
    enrollment_start: 'start_date',
    enrollment_end: 'end_date',
  },
  signal: {
    signal_type: 'heart_rate',
    subject_id: 'id',
    timestamp: { combine: ['date', 'timestamp'] },
    columns: [{ source: 'value', target: 'bpm', type: 'integer' }],
    sampling_hz: 1 / 60,
  },
};
let db;
let projectProvisioning;
let datasetRoute;
let projectRoute;
let ingestion;

function responseRecorder() {
  return {
    statusCode: 200,
    body: null,
    status(code) {
      this.statusCode = code;
      return this;
    },
    json(body) {
      this.body = body;
      return this;
    },
  };
}

test('derives immutable safe identifiers from a team UUID', async () => {
  const { teamAdminRoleForTeam } = await import('../lib/dataset-provision.js');
  assert.equal(
    teamAdminRoleForTeam('123e4567-e89b-12d3-a456-426614174000'),
    'team_123e4567e89b12d3a456426614174000_admin'
  );
  assert.throws(() => teamAdminRoleForTeam('team-slug'), /Invalid team id/);
});

test('derives immutable safe identifiers from a dataset UUID', async () => {
  const { datasetSchemaNameFromUuid } = await import('../lib/dataset-provision.js');
  assert.equal(
    datasetSchemaNameFromUuid('123e4567-e89b-12d3-a456-426614174000'),
    'ds_123e4567e89b12d3a456426614174000'
  );
  assert.throws(() => datasetSchemaNameFromUuid('not-a-uuid'), /Invalid dataset schema uuid/);
});

test('validates and transforms generic subjects and signal CSV mappings', async () => {
  const { parseMappedCsvFiles } = await import('../lib/dataset-ingest.js');
  const subjectsBuffer = await fs.readFile(path.join(__dirname, 'fixtures/subjects.csv'));
  const signalBuffer = await fs.readFile(path.join(__dirname, 'fixtures/heart_rate.csv'));
  const parsed = parseMappedCsvFiles({ subjectsBuffer, signalBuffer, mapping: fitbitMapping });
  assert.equal(parsed.subjects.length, 2);
  assert.equal(parsed.signals.length, 3);
  assert.equal(parsed.signals[0].timestamp, '2024-01-01T00:00:00.000Z');
  assert.equal(parsed.signals[0].values[0], 72);
  assert.equal(parsed.minDate, '2024-01-01');
  assert.equal(parsed.maxDate, '2024-01-02');
  assert.throws(
    () => parseMappedCsvFiles({
      subjectsBuffer,
      signalBuffer: Buffer.from('id,date,timestamp,value\nunknown,2024-01-01,00:00:00,72\n'),
      mapping: fitbitMapping,
    }),
    /unknown subject id/
  );
});

before(async () => {
  if (!enabled) return;
  db = await import('../db.js');
  projectProvisioning = await import('../lib/project-provision.js');
  datasetRoute = await import('../routes/datasets.js');
  projectRoute = await import('../routes/projects.js');
  ingestion = await import('../lib/dataset-ingest.js');

  const schemaSql = await fs.readFile(path.join(__dirname, '../data/schema.sql'), 'utf8');
  const rolesSql = await fs.readFile(path.join(__dirname, '../data/platform_roles.sql'), 'utf8');
  await db.query(schemaSql);
  await db.query(rolesSql);
  await db.query(
    `INSERT INTO w4h.teams (id, slug, name)
     VALUES ($1::uuid, $2, 'Isolation A'), ($3::uuid, $4, 'Isolation B')`,
    [teamA, `isolation-a-${teamA}`, teamB, `isolation-b-${teamB}`]
  );
});

after(async () => {
  if (!enabled || !db) return;
  const { teamAdminRoleForTeam } = await import('../lib/dataset-provision.js');
  const adminRoles = [teamAdminRoleForTeam(teamA), teamAdminRoleForTeam(teamB)];
  const datasetSchemas = await db.query(
    `SELECT schema_name FROM w4h.dataset_registry
     WHERE dataset_id = ANY($1::text[])`,
    [[datasetId, ingestDatasetId, failedIngestDatasetId, realSampleDatasetId, grantDatasetId]]
  );
  const projectRoles = [
    projectProvisioning?.projectRoleForId(projectAId),
    projectProvisioning?.projectRoleForId(projectBId),
  ].filter(Boolean);
  await db.withTransaction(async (client) => {
    await client.query('SET LOCAL ROLE w4h_platform');
    for (const row of datasetSchemas.rows) {
      await client.query(`DROP SCHEMA IF EXISTS "${row.schema_name}" CASCADE`);
    }
    for (const roleName of [...adminRoles, ...projectRoles]) {
      await client.query(`DROP ROLE IF EXISTS "${roleName}"`);
    }
  });
  await db.query(`DELETE FROM w4h.project_datasets WHERE project_id = ANY($1::uuid[])`, [[projectAId, projectBId]]);
  await db.query(`DELETE FROM w4h.projects WHERE id = ANY($1::uuid[])`, [[projectAId, projectBId]]);
  await db.query(
    `DELETE FROM w4h.dataset_registry WHERE dataset_id = ANY($1::text[])`,
    [[datasetId, ingestDatasetId, failedIngestDatasetId, realSampleDatasetId, grantDatasetId]]
  );
  await db.query(
    `DELETE FROM w4h.datasets WHERE id = ANY($1::text[])`,
    [[datasetId, ingestDatasetId, failedIngestDatasetId, realSampleDatasetId, grantDatasetId]]
  );
  await db.query(`DELETE FROM w4h.teams WHERE id = ANY($1::uuid[])`, [[teamA, teamB]]);
  await db.query(`DELETE FROM w4h.users WHERE email LIKE 'isolation-grants-%@example.test'`);
  await db.getPool()?.end();
});

test('POST dataset handler writes catalog metadata and a dataset schema', { skip: !enabled }, async () => {
  const { teamAdminRoleForTeam } = await import('../lib/dataset-provision.js');
  const response = responseRecorder();
  await datasetRoute.createDataset(
    {
      body: {
        id: datasetId,
        name: 'Isolation dataset',
        team_id: teamA,
        schema_def: { device_type: 'test', signals: [] },
      },
      user: { id: null, role: 'super_admin' },
    },
    response
  );

  assert.equal(response.statusCode, 201);
  assert.equal(response.body.id, datasetId);
  assert.equal(response.body.table_count, 0);
  assert.match(response.body.schema_name, /^ds_[a-f0-9]{32}$/);

  const catalog = await db.query(
    `SELECT team_id::text, ingest_status, schema_def FROM w4h.datasets WHERE id = $1`,
    [datasetId]
  );
  assert.equal(catalog.rows[0].team_id, teamA);
  assert.equal(catalog.rows[0].ingest_status, 'created');
  assert.equal(catalog.rows[0].schema_def.device_type, 'test');
  const registry = await db.query(
    `SELECT schema_name, status FROM w4h.dataset_registry WHERE dataset_id = $1`,
    [datasetId]
  );
  assert.equal(registry.rowCount, 1);
  assert.equal(registry.rows[0].status, 'active');
  const schemaName = registry.rows[0].schema_name;
  const schemaExists = await db.query(
    `SELECT 1 FROM information_schema.schemata WHERE schema_name = $1`,
    [schemaName]
  );
  assert.equal(schemaExists.rowCount, 1);

  const adminA = teamAdminRoleForTeam(teamA);
  const adminB = teamAdminRoleForTeam(teamB);
  await db.withTransaction(async (client) => {
    await client.query('SET LOCAL ROLE w4h_platform');
    await client.query(`CREATE ROLE "${adminB}" NOLOGIN`);
  });
  const privileges = await db.query(
    `SELECT
       has_schema_privilege($1, $3, 'USAGE') AS admin_usage,
       has_schema_privilege($2, $3, 'USAGE') AS other_team`,
    [adminA, adminB, schemaName]
  );
  assert.equal(privileges.rows[0].admin_usage, true);
  assert.equal(privileges.rows[0].other_team, false);
});

test('ingests mapped subjects and heart rate into only the owning dataset schema', { skip: !enabled }, async () => {
  const createResponse = responseRecorder();
  await datasetRoute.createDataset(
    {
      body: { id: ingestDatasetId, name: 'Ingestion dataset', team_id: teamA },
      user: { id: null, role: 'super_admin' },
    },
    createResponse
  );
  assert.equal(createResponse.statusCode, 201);

  const result = await ingestion.ingestDatasetFromCsv({
    teamId: teamA,
    catalogId: ingestDatasetId,
    subjectsBuffer: await fs.readFile(path.join(__dirname, 'fixtures/subjects.csv')),
    signalBuffer: await fs.readFile(path.join(__dirname, 'fixtures/heart_rate.csv')),
    mapping: fitbitMapping,
  });
  assert.equal(result.status, 'ready');
  assert.equal(result.subjects_loaded, 2);
  assert.equal(result.signal_rows_loaded, 3);
  assert.equal(result.table.name, 'heart_rate');
  assert.deepEqual(
    result.tables.map((table) => table.table_name).sort(),
    ['heart_rate', 'subjects']
  );

  const inA = await db.withDatasetTransaction(ingestDatasetId, async (client, { schemaName }) => {
    const dataset = await client.query(
      `SELECT ingest_status, schema_def FROM w4h.datasets WHERE id = $1`,
      [ingestDatasetId]
    );
    const tables = await client.query(
      `SELECT table_name, signal_type FROM dataset_tables ORDER BY table_name`
    );
    const subjectCount = await client.query(`SELECT COUNT(*)::int AS count FROM subjects`);
    const signalCount = await client.query(`SELECT COUNT(*)::int AS count FROM heart_rate`);
    const signalTable = await client.query(
      `SELECT to_regclass($1) AS table_name`,
      [`${schemaName}.heart_rate`]
    );
    return { dataset, tables, subjectCount, signalCount, signalTable };
  });
  assert.equal(inA.dataset.rows[0].ingest_status, 'ready');
  assert.equal(inA.dataset.rows[0].schema_def.tables[1].name, 'heart_rate');
  assert.equal(inA.tables.rowCount, 2);
  assert.equal(inA.tables.rows[0].table_name, 'heart_rate');
  assert.equal(inA.tables.rows[1].table_name, 'subjects');
  assert.equal(inA.subjectCount.rows[0].count, 2);
  assert.equal(inA.signalCount.rows[0].count, 3);
  assert.ok(inA.signalTable.rows[0].table_name);

  const otherTeam = await db.query(
    `SELECT 1 FROM w4h.datasets WHERE id = $1 AND team_id = $2::uuid`,
    [ingestDatasetId, teamB]
  );
  assert.equal(otherTeam.rowCount, 0);

  await assert.rejects(
    ingestion.ingestDatasetFromCsv({
      teamId: teamA,
      catalogId: ingestDatasetId,
      subjectsBuffer: await fs.readFile(path.join(__dirname, 'fixtures/subjects.csv')),
      signalBuffer: await fs.readFile(path.join(__dirname, 'fixtures/heart_rate.csv')),
      mapping: fitbitMapping,
    }),
    (error) => error.code === 'INVALID_DATASET_STATUS'
  );
});

test('a post-DDL failure rolls back without orphan signal tables', { skip: !enabled }, async () => {
  const createResponse = responseRecorder();
  await datasetRoute.createDataset(
    {
      body: { id: failedIngestDatasetId, name: 'Failed ingestion dataset', team_id: teamA },
      user: { id: null, role: 'super_admin' },
    },
    createResponse
  );
  assert.equal(createResponse.statusCode, 201);

  await db.withDatasetTransaction(failedIngestDatasetId, async (client) => {
    await client.query(
      `INSERT INTO subjects (id, enrollment_start, enrollment_end)
       VALUES ('subject-a', '2024-01-01', '2024-01-02')`
    );
  });

  await assert.rejects(
    ingestion.ingestDatasetFromCsv({
      teamId: teamA,
      catalogId: failedIngestDatasetId,
      subjectsBuffer: Buffer.from(
        'id,start_date,end_date\n' +
        'subject-a,2024-02-01,2024-02-02\n' +
        'subject-b,2024-01-01,2024-01-03\n'
      ),
      signalBuffer: await fs.readFile(path.join(__dirname, 'fixtures/heart_rate.csv')),
      mapping: fitbitMapping,
    }),
    /conflicting enrollment_start/
  );

  const state = await db.withDatasetTransaction(failedIngestDatasetId, async (client, { schemaName }) => {
    const dataset = await client.query(
      `SELECT ingest_status FROM w4h.datasets WHERE id = $1`,
      [failedIngestDatasetId]
    );
    const tables = await client.query(`SELECT COUNT(*)::int AS count FROM dataset_tables`);
    const table = await client.query(
      `SELECT to_regclass($1) AS table_name`,
      [`${schemaName}.heart_rate`]
    );
    return { dataset, tables, table };
  });
  assert.equal(state.dataset.rows[0].ingest_status, 'failed');
  assert.equal(state.tables.rows[0].count, 0);
  assert.equal(state.table.rows[0].table_name, null);
});

test(
  'smoke ingests the complete Fitbit subjects and heart-rate sample',
  { skip: !enabled || process.env.W4H_REAL_SAMPLE_TEST !== '1', timeout: 120000 },
  async () => {
    const createResponse = responseRecorder();
    await datasetRoute.createDataset(
      {
        body: { id: realSampleDatasetId, name: 'Real Fitbit sample', team_id: teamA },
        user: { id: null, role: 'super_admin' },
      },
      createResponse
    );
    assert.equal(createResponse.statusCode, 201);

    const result = await ingestion.ingestDatasetFromCsv({
      teamId: teamA,
      catalogId: realSampleDatasetId,
      subjectsBuffer: await fs.readFile(
        path.join(__dirname, '../../w4h-datasets/fitbit_subjects.csv')
      ),
      signalBuffer: await fs.readFile(
        path.join(__dirname, '../../w4h-datasets/fitbit_heart_rate.csv')
      ),
      mapping: fitbitMapping,
    });
    assert.equal(result.status, 'ready');
    assert.equal(result.subjects_loaded, 5);
    assert.equal(result.signal_rows_loaded, 165588);
  }
);

test('ingestion handler denies a user who is not admin on the owning team', { skip: !enabled }, async () => {
  const response = responseRecorder();
  await datasetRoute.ingestDatasetCsv(
    {
      params: { id: ingestDatasetId },
      files: {
        subjects_csv: [{ buffer: await fs.readFile(path.join(__dirname, 'fixtures/subjects.csv')) }],
        signal_csv: [{ buffer: await fs.readFile(path.join(__dirname, 'fixtures/heart_rate.csv')) }],
      },
      body: { mapping: JSON.stringify(fitbitMapping) },
      user: { id: 999999999, role: 'project_user' },
    },
    response
  );
  assert.equal(response.statusCode, 403);
});

test('manifest counts stay dataset-scoped and search_path resets after reuse', { skip: !enabled }, async () => {
  await db.withDatasetTransaction(datasetId, async (client) => {
    await client.query(
      `INSERT INTO dataset_tables (table_name, signal_type, columns_def)
       VALUES ('weight', 'weight', '[]'::jsonb)`
    );
  });

  const [counted] = await datasetRoute.attachTableCounts([{ id: datasetId, team_id: teamA }]);
  assert.equal(counted.table_count, 1);

  const searchPath = await db.query('SHOW search_path');
  assert.ok(!searchPath.rows[0].search_path.includes('ds_'));
});

test('unknown and inactive datasets fail closed', { skip: !enabled }, async () => {
  await assert.rejects(
    db.withDatasetTransaction('missing-dataset', async () => {}),
    (error) => error.code === 'DATASET_NOT_PROVISIONED'
  );
});

test('project A and B can share dataset A grants independently', { skip: !enabled }, async () => {
  const createResponse = responseRecorder();
  await datasetRoute.createDataset(
    {
      body: { id: grantDatasetId, name: 'Grant dataset', team_id: teamA },
      user: { id: null, role: 'super_admin' },
    },
    createResponse
  );
  assert.equal(createResponse.statusCode, 201);

  const user = await db.query(
    `INSERT INTO w4h.users (name, email, password, role)
     VALUES ('Isolation grants', $1, 'x', 'super_admin')
     RETURNING id`,
    [`isolation-grants-${randomUUID()}@example.test`]
  );
  await db.query(
    `INSERT INTO w4h.projects (id, slug, name, created_by, team_id)
     VALUES ($1::uuid, $2, 'Project A', $3, $4::uuid),
            ($5::uuid, $6, 'Project B', $3, $4::uuid)`,
    [
      projectAId,
      `proj-a-${projectAId}`,
      user.rows[0].id,
      teamA,
      projectBId,
      `proj-b-${projectBId}`,
    ]
  );

  await projectRoute.replaceProjectDatasets(
    {
      params: { id: projectAId },
      body: { datasets: [{ dataset_id: grantDatasetId, access_type: 'read' }] },
      user: { id: user.rows[0].id, role: 'super_admin' },
    },
    responseRecorder()
  );
  await projectRoute.replaceProjectDatasets(
    {
      params: { id: projectBId },
      body: { datasets: [{ dataset_id: grantDatasetId, access_type: 'read' }] },
      user: { id: user.rows[0].id, role: 'super_admin' },
    },
    responseRecorder()
  );

  const schemaName = (await db.query(
    `SELECT schema_name FROM w4h.dataset_registry WHERE dataset_id = $1`,
    [grantDatasetId]
  )).rows[0].schema_name;
  const roleA = projectProvisioning.projectRoleForId(projectAId);
  const roleB = projectProvisioning.projectRoleForId(projectBId);
  const linked = await db.query(
    `SELECT
       has_schema_privilege($1, $3, 'USAGE') AS a_usage,
       has_schema_privilege($2, $3, 'USAGE') AS b_usage`,
    [roleA, roleB, schemaName]
  );
  assert.equal(linked.rows[0].a_usage, true);
  assert.equal(linked.rows[0].b_usage, true);

  const unlinkB = responseRecorder();
  await projectRoute.replaceProjectDatasets(
    {
      params: { id: projectBId },
      body: { datasets: [] },
      user: { id: user.rows[0].id, role: 'super_admin' },
    },
    unlinkB
  );
  assert.equal(unlinkB.statusCode, 200);

  const afterUnlink = await db.query(
    `SELECT
       has_schema_privilege($1, $3, 'USAGE') AS a_usage,
       has_schema_privilege($2, $3, 'USAGE') AS b_usage`,
    [roleA, roleB, schemaName]
  );
  assert.equal(afterUnlink.rows[0].a_usage, true);
  assert.equal(afterUnlink.rows[0].b_usage, false);

  const { teamAdminRoleForTeam } = await import('../lib/dataset-provision.js');
  const teamBSchema = await db.query(
    `SELECT has_schema_privilege($1, $2, 'USAGE') AS other_team`,
    [teamAdminRoleForTeam(teamB), schemaName]
  );
  assert.equal(teamBSchema.rows[0].other_team, false);
});
