/**
 * CSV ingest helpers — API-key / team-admin path for sample importers.
 * Creates isolation schema tables and registers w4h.tables metadata (ADR-003).
 */

import { query } from '../db.js';
import { isSuperAdmin, canManageTeamAsync } from './utils/user.js';

const PG_TYPE_MAP = {
  text: 'text',
  string: 'text',
  integer: 'integer',
  int: 'integer',
  bigint: 'bigint',
  double: 'double precision',
  float: 'double precision',
  'double precision': 'double precision',
  numeric: 'numeric',
  boolean: 'boolean',
  date: 'date',
  timestamp: 'timestamp with time zone',
};

function sqlIdent(name) {
  const s = String(name || '').trim();
  if (!/^[a-z_][a-z0-9_]*$/i.test(s)) {
    throw new Error(`Invalid SQL identifier: ${name}`);
  }
  return s;
}

function datasetSchemaName(datasetId) {
  const slug = String(datasetId)
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/(^_+|_+$)/g, '');
  return `ds_${slug || 'dataset'}`;
}

function pgType(col) {
  const t = String(col?.type || 'text').trim().toLowerCase();
  return PG_TYPE_MAP[t] || 'text';
}

async function assertCanIngestDataset(user, datasetId) {
  const res = await query(
    `SELECT id, team_id::text AS team_id, COALESCE(read_only, false) AS read_only, origin
     FROM w4h.datasets WHERE id = $1`,
    [datasetId]
  );
  if (!res.rowCount) {
    return { ok: false, status: 404, error: 'Dataset not found' };
  }
  const row = res.rows[0];
  if (row.read_only && !isSuperAdmin(user.role)) {
    return {
      ok: false,
      status: 403,
      error: row.origin
        ? `This dataset is read-only sample data (${row.origin}). Super admin seeds it; do not re-ingest.`
        : 'This dataset is read-only sample data. Super admin seeds it; do not re-ingest.',
    };
  }
  if (isSuperAdmin(user.role)) {
    return { ok: true, dataset: row };
  }
  if (row.team_id) {
    const teamAdmin = await query(
      `SELECT 1 FROM w4h.team_members
       WHERE user_id = $1 AND team_id = $2::uuid AND role = 'team_admin'
       LIMIT 1`,
      [user.id, row.team_id]
    );
    if (teamAdmin.rowCount) return { ok: true, dataset: row };
  }
  const projectAdmin = await query(
    `SELECT 1
     FROM w4h.project_members pm
     JOIN w4h.project_datasets pd ON pd.project_id = pm.project_id
     WHERE pm.user_id = $1 AND pm.role = 'owner' AND pd.dataset_id = $2
     LIMIT 1`,
    [user.id, datasetId]
  );
  if (projectAdmin.rowCount) return { ok: true, dataset: row };
  if (!(await canManageTeamAsync(user))) {
    return { ok: false, status: 403, error: 'Admin required to ingest this dataset' };
  }
  if (!row.team_id) {
    return { ok: false, status: 403, error: 'Dataset has no owning team' };
  }
  const onTeam = await query(
    `SELECT 1 FROM w4h.team_members WHERE user_id = $1 AND team_id = $2::uuid LIMIT 1`,
    [user.id, row.team_id]
  );
  if (!onTeam.rowCount) {
    return { ok: false, status: 403, error: 'Unauthorized for this dataset team' };
  }
  return { ok: true, dataset: row };
}

async function ensurePhysicalTable(schemaName, tableName, columns, dedupeColumns) {
  const schema = sqlIdent(schemaName);
  const table = sqlIdent(tableName);
  await query(`CREATE SCHEMA IF NOT EXISTS ${schema}`);

  const colDefs = columns.map((c) => {
    const name = sqlIdent(c.name);
    return `${name} ${pgType(c)}`;
  });
  if (!colDefs.length) {
    throw new Error('columns array is required');
  }

  await query(`CREATE TABLE IF NOT EXISTS ${schema}.${table} (${colDefs.join(', ')})`);

  if (dedupeColumns?.length) {
    const dedupe = dedupeColumns.map(sqlIdent);
    const idxName = sqlIdent(`${table}_dedupe_${dedupe.join('_')}`);
    await query(
      `CREATE UNIQUE INDEX IF NOT EXISTS ${idxName} ON ${schema}.${table} (${dedupe.join(', ')})`
    );
  }

  return { schema, table };
}

async function registerCatalogTable(datasetId, signal, meta, physicalSchema, physicalTable) {
  const name = sqlIdent(signal);
  await query(
    `INSERT INTO w4h.tables (dataset_id, name, source_name, source_url, mapping_json, physical_schema, physical_table, description)
     VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7, $8)
     ON CONFLICT (dataset_id, name) DO UPDATE SET
       source_name = EXCLUDED.source_name,
       source_url = EXCLUDED.source_url,
       mapping_json = EXCLUDED.mapping_json,
       physical_schema = EXCLUDED.physical_schema,
       physical_table = EXCLUDED.physical_table,
       description = EXCLUDED.description`,
    [
      datasetId,
      name,
      meta.source_name || null,
      meta.source_url || null,
      meta.mapping ? JSON.stringify(meta.mapping) : null,
      physicalSchema,
      physicalTable,
      meta.description || null,
    ]
  );
}

/**
 * Import a batch of CSV rows into a dataset signal table.
 */
async function importCsvBatch(user, datasetId, payload) {
  const access = await assertCanIngestDataset(user, datasetId);
  if (!access.ok) {
    return access;
  }

  const signal = String(payload?.signal || '').trim();
  const mode = String(payload?.mode || 'append').trim().toLowerCase();
  const columns = Array.isArray(payload?.columns) ? payload.columns : [];
  const rows = Array.isArray(payload?.rows) ? payload.rows : [];
  const dedupeColumns = Array.isArray(payload?.dedupe_columns)
    ? payload.dedupe_columns.map(String)
    : [];

  if (!signal) {
    return { ok: false, status: 400, error: 'signal is required' };
  }
  if (!columns.length) {
    return { ok: false, status: 400, error: 'columns is required' };
  }
  if (!rows.length) {
    return { ok: true, inserted: 0, skipped: 0 };
  }

  const schemaName = datasetSchemaName(datasetId);
  const tableName = sqlIdent(`fitbit_${signal}`);
  const { schema, table } = await ensurePhysicalTable(schemaName, tableName, columns, dedupeColumns);

  if (mode === 'replace') {
    await query(`TRUNCATE TABLE ${schema}.${table}`);
  }

  const colNames = columns.map((c) => sqlIdent(c.name));
  let inserted = 0;
  let skipped = 0;

  // Multi-row INSERT per batch instead of one round trip per row: without an
  // explicit transaction, Postgres autocommits every statement, and a
  // separate INSERT per row forces a separate WAL fsync per row — the
  // textbook worst case for insert throughput. Batching rows into one
  // VALUES list per statement removes that per-row fsync cost. Stay
  // comfortably under Postgres's 65535 bind-parameter limit per statement.
  const MAX_BIND_PARAMS = 60000;
  const rowsPerBatch = Math.max(1, Math.floor(MAX_BIND_PARAMS / colNames.length));

  const validRows = rows.filter((row) => row && typeof row === 'object');

  for (let start = 0; start < validRows.length; start += rowsPerBatch) {
    const batch = validRows.slice(start, start + rowsPerBatch);
    const values = [];
    const tuples = batch.map((row, i) => {
      const base = i * colNames.length;
      const placeholders = colNames.map((c, j) => {
        const raw = row[c];
        values.push(raw === '' || raw === undefined ? null : raw);
        return `$${base + j + 1}`;
      });
      return `(${placeholders.join(', ')})`;
    });

    const sql = dedupeColumns.length
      ? `INSERT INTO ${schema}.${table} (${colNames.join(', ')}) VALUES ${tuples.join(', ')}
         ON CONFLICT (${dedupeColumns.map(sqlIdent).join(', ')}) DO NOTHING`
      : `INSERT INTO ${schema}.${table} (${colNames.join(', ')}) VALUES ${tuples.join(', ')}`;

    const result = await query(sql, values);
    const batchInserted = typeof result.rowCount === 'number' ? result.rowCount : batch.length;
    inserted += batchInserted;
    if (dedupeColumns.length) {
      skipped += batch.length - batchInserted;
    }
  }

  await registerCatalogTable(datasetId, signal, {
    source_name: payload.source_name,
    source_url: payload.source_url,
    description: payload.description,
    mapping: payload.mapping,
  }, schema, table);

  return {
    ok: true,
    inserted,
    skipped,
    physical_schema: schema,
    physical_table: table,
  };
}

export {
  importCsvBatch,
  assertCanIngestDataset,
  datasetSchemaName,
};
