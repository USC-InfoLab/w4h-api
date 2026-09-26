/**
 * CSV ingest helpers — API-key / team-admin path for sample importers.
 * Creates isolation schema tables and registers w4h.tables metadata (ADR-003).
 */

import { query, getPool } from '../db.js';
import { canManageTeamAsync } from './utils/user.js';

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

/** Coerce a CSV cell to a Postgres-friendly JS value; invalid typed values → null. */
function coerceValue(raw, col) {
  if (raw === '' || raw === undefined || raw === null) return null;
  const t = String(col?.type || 'text').trim().toLowerCase();
  const s = String(raw).trim();
  if (t === 'integer' || t === 'int') {
    return /^-?\d+$/.test(s) ? parseInt(s, 10) : null;
  }
  if (t === 'bigint') {
    return /^-?\d+$/.test(s) ? s : null;
  }
  if (t === 'double precision' || t === 'double' || t === 'float') {
    return /^-?\d+(\.\d+)?([eE][+-]?\d+)?$/.test(s) ? parseFloat(s) : null;
  }
  if (t === 'numeric') {
    return /^-?\d+(\.\d+)?([eE][+-]?\d+)?$/.test(s) ? s : null;
  }
  if (t === 'boolean') {
    const lower = s.toLowerCase();
    if (['true', 't', '1', 'yes'].includes(lower)) return true;
    if (['false', 'f', '0', 'no'].includes(lower)) return false;
    return null;
  }
  return raw;
}

async function assertCanIngestDataset(user, datasetId) {
  const res = await query(
    `SELECT id, team_id::text AS team_id, COALESCE(read_only, false) AS read_only, origin,
            COALESCE(status, 'active') AS status
     FROM w4h.datasets WHERE id = $1`,
    [datasetId]
  );
  if (!res.rowCount) {
    return { ok: false, status: 404, error: 'Dataset not found' };
  }
  const row = res.rows[0];
  if (row.status === 'archived') {
    return {
      ok: false,
      status: 403,
      error: 'This dataset is archived. Restore it from the team catalog before ingesting or running jobs.',
    };
  }
  if (row.read_only) {
    return {
      ok: false,
      status: 403,
      error: row.origin
        ? `This dataset is read-only sample data (${row.origin}). Do not re-ingest.`
        : 'This dataset is read-only sample data. Do not re-ingest.',
    };
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

/** Run a utility statement built with format(%I/%L). COMMENT does not accept bind parameters. */
async function execFormatted(formatStr, args) {
  const placeholders = args.map((_, i) => `$${i + 2}`).join(', ');
  const formatted = await query(
    `SELECT format($1, ${placeholders}) AS sql`,
    [formatStr, ...args]
  );
  const sql = formatted.rows[0]?.sql;
  if (sql) await query(sql);
}

/**
 * Mirror signal and column descriptions as PostgreSQL comments (ADR-003).
 * Empty descriptions are left unchanged so a batch that omits them does not clear comments.
 */
async function applyRelationComments(schema, table, tableDescription, columns) {
  const tableComment = String(tableDescription || '').trim();
  if (tableComment) {
    await execFormatted('COMMENT ON TABLE %I.%I IS %L', [schema, table, tableComment]);
  }
  for (const col of columns) {
    const comment = String(col?.description || '').trim();
    if (!comment) continue;
    await execFormatted('COMMENT ON COLUMN %I.%I.%I IS %L', [
      schema,
      table,
      sqlIdent(col.name),
      comment,
    ]);
  }
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

  const tablePrefix = String(payload?.table_prefix || '').trim();
  const schemaName = datasetSchemaName(datasetId);
  const tableName = sqlIdent(tablePrefix ? `${tablePrefix}_${signal}` : signal);
  const { schema, table } = await ensurePhysicalTable(schemaName, tableName, columns, dedupeColumns);
  await applyRelationComments(schema, table, payload.description, columns);

  const colNames = columns.map((c) => sqlIdent(c.name));
  const validRows = rows.filter((row) => row && typeof row === 'object');
  let inserted = 0;
  let skipped = 0;

  // Bulk-insert path: one multi-row VALUES statement per sub-batch (capped by
  // Postgres's 65535-param limit), all under one client + one transaction —
  // replaces the old per-row query() (per-row round trip + autocommit fsync,
  // the actual bottleneck at Fitabase-sized imports of millions of rows).
  const onConflict = dedupeColumns.length
    ? ` ON CONFLICT (${dedupeColumns.map(sqlIdent).join(', ')}) DO NOTHING`
    : '';
  const rowsPerStmt = Math.max(1, Math.floor(60000 / colNames.length));

  const pool = getPool();
  if (!pool) throw new Error('Database not configured');
  const client = await pool.connect();
  let rollbackFailed = false;
  try {
    await client.query('BEGIN');
    if (mode === 'replace') {
      await client.query(`TRUNCATE TABLE ${schema}.${table}`);
    }

    for (let i = 0; i < validRows.length; i += rowsPerStmt) {
      const slice = validRows.slice(i, i + rowsPerStmt);
      const values = [];
      const tuples = slice.map((row) => {
        const cellVals = columns.map((col) => coerceValue(row[col.name] ?? row[sqlIdent(col.name)], col));
        const base = values.length;
        values.push(...cellVals);
        return `(${cellVals.map((_, j) => `$${base + j + 1}`).join(', ')})`;
      });
      const sql = `INSERT INTO ${schema}.${table} (${colNames.join(', ')}) VALUES ${tuples.join(', ')}${onConflict}`;
      let result;
      try {
        result = await client.query(sql, values);
      } catch (e) {
        // Same wrapped shape query() (db.js) gives every other call site —
        // callers/logs rely on .code/.detail/.constraint plus the
        // "Query failed: ..." message.
        const wrapped = new Error(`Query failed: ${e.message}\nSQL: ${sql.slice(0, 300)}`);
        wrapped.code = e.code;
        wrapped.detail = e.detail;
        wrapped.constraint = e.constraint;
        wrapped.cause = e;
        throw wrapped;
      }
      const rowCount = result.rowCount || 0;
      inserted += rowCount;
      if (dedupeColumns.length) skipped += slice.length - rowCount;
    }

    await client.query('COMMIT');
  } catch (e) {
    await client.query('ROLLBACK').catch((rollbackErr) => {
      console.error('csv-batch rollback failed:', rollbackErr.message);
      rollbackFailed = true;
    });
    throw e;
  } finally {
    client.release(rollbackFailed || undefined);
  }

  const mapping = { ...(payload.mapping && typeof payload.mapping === 'object' ? payload.mapping : {}) };
  const interval = payload.sampling_interval || mapping.sampling_interval;
  if (interval && !mapping.sampling_interval) mapping.sampling_interval = interval;
  if (tablePrefix && !mapping.table_prefix) mapping.table_prefix = tablePrefix;

  await registerCatalogTable(datasetId, signal, {
    source_name: payload.source_name,
    source_url: payload.source_url,
    description: payload.description,
    mapping: Object.keys(mapping).length ? mapping : payload.mapping,
  }, schema, table);

  // Optional provenance (source, local/OneDrive path, vendor, generated_at,
  // imported_by, …) — the generated CLI's `import` sends this once per run.
  // Merged into the existing `summary` jsonb column (the established
  // cached-metadata bucket — see mergeMetsSummary()) under its own
  // `provenance` key, not a new column. Uses the same permission gate this
  // whole call already passed (assertCanIngestDataset), not the
  // team-admin-only PATCH /datasets/:id route, since a
  // project-owner-but-not-team-admin can ingest but not PATCH.
  if (payload.provenance && typeof payload.provenance === 'object' && Object.keys(payload.provenance).length) {
    await query(
      `UPDATE w4h.datasets
       SET summary = jsonb_set(COALESCE(summary, '{}'::jsonb), '{provenance}', $1::jsonb, true),
           summary_updated_at = NOW()
       WHERE id = $2`,
      [JSON.stringify(payload.provenance), datasetId]
    );
  }

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
  ensurePhysicalTable,
  registerCatalogTable,
  coerceValue,
  sqlIdent,
  PG_TYPE_MAP,
};
