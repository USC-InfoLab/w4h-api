/**
 * YAML-bound SQL contributions (ADR-023).
 * Bind identifiers from w4h.tables; never execute client-supplied SQL.
 */

import fs from 'fs';
import path from 'path';
import yaml from 'js-yaml';
import { getPool, query } from '../db.js';
import { getContribution } from './contributions.js';
import {
  assertCanIngestDataset,
  datasetSchemaName,
  ensurePhysicalTable,
  registerCatalogTable,
} from './ingest-csv.js';
import { isSuperAdmin } from './utils/user.js';

const SIBLING_REPOS_ROOT = process.env.CONTRIBUTION_RUNNER_REPO_ROOT
  || path.resolve(process.cwd(), '..');

const STATEMENT_TIMEOUT_MS = 30_000;
const PREVIEW_CAP = 100;
const PREVIEW_HARD_CAP = 1000;

function sqlIdent(name) {
  const s = String(name || '').trim();
  if (!/^[a-z_][a-z0-9_]*$/i.test(s)) {
    throw new Error(`Invalid SQL identifier: ${name}`);
  }
  return s;
}

function qualify(schema, table) {
  return `${sqlIdent(schema)}.${sqlIdent(table)}`;
}

function parseMapping(raw) {
  if (!raw) return {};
  if (typeof raw === 'object') return raw;
  try {
    return JSON.parse(raw);
  } catch {
    return {};
  }
}

function parseIntervalMinutes(value) {
  if (value == null || value === '') return null;
  if (typeof value === 'number' && Number.isFinite(value) && value > 0) return value;
  const match = String(value).trim().match(/^(\d+(?:\.\d+)?)\s*(min|m|minutes?)?$/i);
  if (match) return Number(match[1]);
  return null;
}

function signalTypeOf(row) {
  const mapping = parseMapping(row.mapping_json);
  return String(mapping.signal_type || row.name || '').trim().toLowerCase();
}

function tablePrefixOf(row) {
  const mapping = parseMapping(row.mapping_json);
  if (mapping.table_prefix) return String(mapping.table_prefix);
  const phys = String(row.physical_table || '');
  const name = String(row.name || '');
  if (phys && name && phys !== name && phys.endsWith(`_${name}`)) {
    return phys.slice(0, -(name.length + 1));
  }
  return '';
}

function mappedSource(mapping, roleName, fallbacks = []) {
  const values = mapping?.values;
  if (Array.isArray(values)) {
    for (const entry of values) {
      if (typeof entry === 'string') {
        if (entry === roleName || fallbacks.includes(entry)) return entry;
        continue;
      }
      if (!entry || typeof entry !== 'object') continue;
      const name = String(entry.name || '').trim();
      const source = String(entry.source || entry.name || '').trim();
      if (name === roleName || fallbacks.includes(name) || fallbacks.includes(source)) {
        return source || name;
      }
    }
  }
  return fallbacks[0] || null;
}

function pickExisting(columns, candidates) {
  for (const name of candidates) {
    if (name && columns.has(name)) return name;
  }
  return null;
}

function timeExpr(mapping, columns, alias) {
  const a = sqlIdent(alias);
  const time = mapping?.time;
  if (time && typeof time === 'object' && Array.isArray(time.combine) && time.combine.length >= 2) {
    const dateCol = pickExisting(columns, time.combine);
    const timeCol = time.combine.map(String).find((c) => c !== dateCol && columns.has(c));
    if (dateCol && timeCol) {
      return `(${a}.${sqlIdent(dateCol)}::timestamp + ${a}.${sqlIdent(timeCol)}::time)`;
    }
    if (dateCol) return `(${a}.${sqlIdent(dateCol)}::timestamp)`;
  }
  const col = pickExisting(columns, [
    typeof time === 'string' ? time : null,
    'ts',
    'timestamp',
    'time',
    'date',
  ]);
  if (!col) return 'NULL::timestamp';
  return `(${a}.${sqlIdent(col)}::timestamp)`;
}

async function tableColumns(schema, table) {
  const res = await query(
    `SELECT column_name
     FROM information_schema.columns
     WHERE table_schema = $1 AND table_name = $2`,
    [schema, table]
  );
  return new Set(res.rows.map((r) => String(r.column_name)));
}

async function loadMappedTables(datasetId) {
  const res = await query(
    `SELECT name, mapping_json, physical_schema, physical_table, description
     FROM w4h.tables
     WHERE dataset_id = $1
     ORDER BY name ASC`,
    [datasetId]
  );
  return (res.rows || []).filter((row) => row.physical_schema && row.physical_table);
}

function describeSignal(row, columns) {
  const mapping = parseMapping(row.mapping_json);
  const intervalMinutes = parseIntervalMinutes(mapping.sampling_interval)
    || parseIntervalMinutes(mapping.interval_minutes);
  return {
    name: row.name,
    signal_type: signalTypeOf(row),
    physical_schema: row.physical_schema,
    physical_table: row.physical_table,
    table_prefix: tablePrefixOf(row) || null,
    subject: mapping.subject || null,
    time: mapping.time ?? null,
    values: mapping.values || [],
    sampling_interval: mapping.sampling_interval || mapping.interval_minutes || null,
    interval_minutes: intervalMinutes,
    columns: [...columns],
  };
}

export async function listDatasetSignals(datasetId) {
  const tables = await loadMappedTables(datasetId);
  const signals = [];
  for (const row of tables) {
    const columns = await tableColumns(row.physical_schema, row.physical_table);
    signals.push(describeSignal(row, columns));
  }
  return { dataset_id: datasetId, signals };
}

function loadContributionPackage(slug) {
  const root = path.join(SIBLING_REPOS_ROOT, slug);
  if (!fs.existsSync(root)) {
    const err = new Error(
      `SQL contribution checkout missing at ${root}. Clone or place ${slug} next to w4h-api.`
    );
    err.status = 500;
    throw err;
  }
  const manifestPath = path.join(root, 'manifest.yaml');
  if (!fs.existsSync(manifestPath)) {
    const err = new Error(`manifest.yaml not found in ${slug}`);
    err.status = 500;
    throw err;
  }
  const manifest = yaml.load(fs.readFileSync(manifestPath, 'utf8'));
  const sqlRel = String(manifest?.sql || 'sql/compute.sql');
  const sqlPath = path.join(root, sqlRel);
  if (!fs.existsSync(sqlPath)) {
    const err = new Error(`${sqlRel} not found in ${slug}`);
    err.status = 500;
    throw err;
  }
  const template = fs.readFileSync(sqlPath, 'utf8');
  return { root, manifest, template };
}

function findRequired(tables, spec) {
  const type = String(spec.signal_type || '').toLowerCase();
  return tables.find((row) => signalTypeOf(row) === type) || null;
}

function numericLiteral(value, label) {
  const n = Number(value);
  if (!Number.isFinite(n)) {
    throw new Error(`Invalid ${label}`);
  }
  return String(n);
}

function requireAlias(spec) {
  const alias = String(spec?.alias || spec?.signal_type || '').trim();
  if (!/^[a-z_][a-z0-9_]*$/i.test(alias)) {
    throw Object.assign(new Error('Each require needs a signal_type (or alias) usable as a SQL placeholder prefix'), { status: 400 });
  }
  return alias;
}

function requireSqlAlias(spec, alias) {
  const sqlAlias = String(spec?.sql_alias || alias).trim();
  return sqlIdent(sqlAlias);
}

async function bindRequiredSignal(spec, table, intervalOverride) {
  const alias = requireAlias(spec);
  const sqlAlias = requireSqlAlias(spec, alias);
  const columns = await tableColumns(table.physical_schema, table.physical_table);
  const mapping = parseMapping(table.mapping_json);
  const valueRole = String(spec.value || spec.signal_type || '').trim();
  const subjectCol = pickExisting(columns, [mapping.subject, 'id', 'subject_id', 'user_id']);
  const valueCol = pickExisting(columns, [
    mappedSource(mapping, valueRole, [valueRole, 'value']),
    'value',
    valueRole,
  ]);
  if (!subjectCol || !valueCol) {
    throw Object.assign(
      new Error(`Signal '${spec.signal_type}' is missing subject or value columns`),
      { status: 400 }
    );
  }
  const fromMapping = parseIntervalMinutes(mapping.sampling_interval)
    || parseIntervalMinutes(mapping.interval_minutes);
  const intervalMinutes = intervalOverride ?? fromMapping;
  const bindings = {
    [`${alias}_rel`]: qualify(table.physical_schema, table.physical_table),
    [`${alias}_subject_col`]: sqlIdent(subjectCol),
    [`${alias}_col`]: sqlIdent(valueCol),
    [`${alias}_value_col`]: sqlIdent(valueCol),
    [`${alias}_ts_expr`]: timeExpr(mapping, columns, sqlAlias),
  };
  if (intervalMinutes) {
    bindings[`${alias}_interval_minutes`] = numericLiteral(intervalMinutes, `${alias}_interval_minutes`);
  }
  if (spec.default_kg != null && spec.default_kg !== '') {
    bindings[`${alias}_default_kg`] = numericLiteral(spec.default_kg, `${alias}_default_kg`);
  }
  return { alias, prefix: tablePrefixOf(table), intervalMinutes, bindings };
}

function bindManifestScalars(manifest) {
  const out = {};
  const scalars = {
    ...(manifest.reference_interval_minutes != null
      ? { reference_interval_minutes: manifest.reference_interval_minutes }
      : {}),
    ...(manifest.scalars && typeof manifest.scalars === 'object' ? manifest.scalars : {}),
  };
  for (const [key, raw] of Object.entries(scalars)) {
    if (!/^[a-z_][a-z0-9_]*$/i.test(key)) continue;
    if (typeof raw === 'number' || (typeof raw === 'string' && raw.trim() !== '' && Number.isFinite(Number(raw)))) {
      out[key] = numericLiteral(raw, key);
    }
  }
  return out;
}

function unboundPlaceholders(template, bindings) {
  const keys = [...String(template).matchAll(/\{\{([a-z][a-z0-9_]*)\}\}/gi)].map((m) => m[1]);
  return [...new Set(keys)].filter((k) => !Object.prototype.hasOwnProperty.call(bindings, k));
}

function bindTemplate(template, bindings) {
  const unbound = unboundPlaceholders(template, bindings);
  if (unbound.length) {
    const intervals = unbound.filter((k) => k === 'interval_minutes' || k.endsWith('_interval_minutes'));
    if (intervals.length) {
      throw Object.assign(
        new Error(
          `Mapping has no sampling_interval (${intervals.join(', ')}). Pass interval_minutes (5 for ICDE, 1 for Fitbit).`
        ),
        { status: 400 }
      );
    }
    throw Object.assign(new Error(`Unknown SQL placeholder {{${unbound[0]}}}`), { status: 400 });
  }
  const bound = String(template).replace(/\{\{([a-z][a-z0-9_]*)\}\}/gi, (_, key) => {
    if (!Object.prototype.hasOwnProperty.call(bindings, key)) {
      throw new Error(`Unknown SQL placeholder {{${key}}}`);
    }
    return bindings[key];
  });
  if (/\{\{/.test(bound)) {
    throw new Error('Unbound SQL placeholder remaining after bind');
  }
  const stripped = bound
    .replace(/\/\*[\s\S]*?\*\//g, ' ')
    .replace(/--[^\n]*/g, ' ')
    .trim();
  if (stripped.includes(';')) {
    throw new Error('SQL contribution must be a single statement');
  }
  if (!/^\s*select\b/i.test(stripped)) {
    throw new Error('SQL contribution must be a SELECT');
  }
  return stripped;
}

function outputTableName(prefix, signalType) {
  const signal = sqlIdent(signalType);
  const p = String(prefix || '').trim();
  return p ? sqlIdent(`${p}_${signal}`) : signal;
}

async function withTimeout(fn) {
  const pool = getPool();
  if (!pool) throw new Error('Database not configured');
  const client = await pool.connect();
  try {
    await client.query('BEGIN');
    await client.query(`SET LOCAL statement_timeout = ${STATEMENT_TIMEOUT_MS}`);
    const result = await fn(client);
    await client.query('COMMIT');
    return result;
  } catch (e) {
    try { await client.query('ROLLBACK'); } catch { /* ignore */ }
    throw e;
  } finally {
    client.release();
  }
}

async function assertCanViewDataset(user, datasetId) {
  const res = await query(
    `SELECT id, team_id::text AS team_id FROM w4h.datasets WHERE id = $1`,
    [datasetId]
  );
  if (!res.rowCount) return { ok: false, status: 404, error: 'Dataset not found' };
  if (isSuperAdmin(user.role)) return { ok: true, dataset: res.rows[0] };
  const access = await query(
    `SELECT 1 FROM w4h.datasets d
     WHERE d.id = $1 AND (
       EXISTS (
         SELECT 1 FROM w4h.project_datasets pd
         JOIN w4h.project_members pm ON pm.project_id = pd.project_id
         WHERE pd.dataset_id = d.id AND pm.user_id = $2
       )
       OR EXISTS (
         SELECT 1 FROM w4h.team_members tm
         WHERE tm.user_id = $2 AND tm.team_id = d.team_id
       )
       OR EXISTS (
         SELECT 1 FROM w4h.dataset_shares ds
         JOIN w4h.team_members tm ON tm.team_id = ds.target_team_id
         WHERE ds.dataset_id = d.id AND tm.user_id = $2
       )
     )`,
    [datasetId, user.id]
  );
  if (!access.rowCount) {
    return { ok: false, status: 403, error: 'Unauthorized for this dataset' };
  }
  return { ok: true, dataset: res.rows[0] };
}

function resolveWriteBack(manifest, body = {}) {
  if (body.write_back != null && body.write_back !== '') {
    return Boolean(body.write_back);
  }
  if (manifest?.run?.materialize != null) {
    return Boolean(manifest.run.materialize);
  }
  const produce = (Array.isArray(manifest?.produces) ? manifest.produces[0] : null) || {};
  if (Array.isArray(produce.columns) && produce.columns.length) {
    return true;
  }
  return false;
}

export async function runSqlContribution(user, datasetId, slug, body = {}) {
  const contribution = await getContribution(slug, { includeUnpublished: true });
  if (!contribution) {
    return { ok: false, status: 404, error: 'Contribution not found' };
  }
  const kinds = [...(contribution.kinds || []), contribution.kind];
  if (!kinds.includes('sql')) {
    return { ok: false, status: 400, error: 'This contribution is not a SQL job (kind: sql)' };
  }

  const { manifest, template } = loadContributionPackage(slug);
  const writeBack = resolveWriteBack(manifest, body);
  if (writeBack) {
    const ingest = await assertCanIngestDataset(user, datasetId);
    if (!ingest.ok) return ingest;
  } else {
    const view = await assertCanViewDataset(user, datasetId);
    if (!view.ok) return view;
  }

  const tables = await loadMappedTables(datasetId);
  const requires = Array.isArray(manifest.requires) ? manifest.requires : [];
  if (!requires.length) {
    return { ok: false, status: 400, error: 'SQL contribution manifest has no requires' };
  }

  const intervalOverride = body.interval_minutes != null && body.interval_minutes !== ''
    ? parseIntervalMinutes(body.interval_minutes)
    : null;
  if (body.interval_minutes != null && body.interval_minutes !== '' && !intervalOverride) {
    return { ok: false, status: 400, error: 'interval_minutes must be a positive number' };
  }

  const bindings = bindManifestScalars(manifest);
  const missing = [];
  let prefix = String(body.table_prefix || '').trim();
  const intervals = {};

  try {
    for (const spec of requires) {
      const table = findRequired(tables, spec);
      if (!table) {
        missing.push(spec.signal_type || spec.alias || '?');
        continue;
      }
      const bound = await bindRequiredSignal(spec, table, intervalOverride);
      Object.assign(bindings, bound.bindings);
      if (bound.intervalMinutes) {
        intervals[bound.alias] = bound.intervalMinutes;
        if (bindings.interval_minutes == null) {
          bindings.interval_minutes = numericLiteral(bound.intervalMinutes, 'interval_minutes');
        }
      }
      if (!prefix && bound.prefix) prefix = bound.prefix;
    }
  } catch (e) {
    return { ok: false, status: e.status || 400, error: e.message };
  }

  if (missing.length) {
    return { ok: false, status: 400, error: `Dataset is missing required signal(s): ${missing.join(', ')}` };
  }

  const weightDefault = requires.find((r) => r.default_kg != null);
  if (weightDefault && bindings.default_weight_kg == null) {
    bindings.default_weight_kg = numericLiteral(weightDefault.default_kg, 'default_weight_kg');
  }

  let selectSql;
  try {
    selectSql = bindTemplate(template, bindings);
  } catch (e) {
    return { ok: false, status: e.status || 400, error: e.message };
  }

  const produce = (Array.isArray(manifest.produces) ? manifest.produces[0] : null) || {};
  const signalType = String(produce.signal_type || '').trim();
  if (!signalType) {
    return { ok: false, status: 400, error: 'SQL contribution manifest produces[0].signal_type is required' };
  }
  const outTable = outputTableName(prefix, signalType);
  const schema = datasetSchemaName(datasetId);

  if (!writeBack) {
    const limit = Math.min(
      Math.max(1, Number(body.limit) || PREVIEW_CAP),
      PREVIEW_HARD_CAP
    );
    const preview = await withTimeout((client) =>
      client.query(`SELECT * FROM (${selectSql}) q LIMIT ${limit}`)
    );
    return {
      ok: true,
      write_back: false,
      interval_minutes: intervals,
      table_prefix: prefix || null,
      row_count: preview.rowCount,
      columns: preview.fields?.map((f) => f.name) || [],
      rows: preview.rows,
    };
  }

  const columns = Array.isArray(produce.columns) ? produce.columns : [];
  if (!columns.length) {
    return { ok: false, status: 400, error: 'SQL contribution manifest produces[0].columns is required for write-back' };
  }
  const dedupe = columns.map((c) => c.name).filter((n) => n === 'id' || n === 'ts' || n === produce.subject || n === produce.time);

  await ensurePhysicalTable(schema, outTable, columns, dedupe.length ? dedupe : undefined);
  const inserted = await withTimeout(async (client) => {
    await client.query(`TRUNCATE TABLE ${qualify(schema, outTable)}`);
    const ins = await client.query(
      `INSERT INTO ${qualify(schema, outTable)} (${columns.map((c) => sqlIdent(c.name)).join(', ')})
       SELECT ${columns.map((c) => sqlIdent(c.name)).join(', ')} FROM (${selectSql}) q`
    );
    return ins.rowCount || 0;
  });

  const firstInterval = Object.values(intervals)[0];
  await registerCatalogTable(datasetId, signalType, {
    source_name: contribution.slug,
    description: `Derived by ${contribution.slug}`,
    mapping: {
      signal_type: signalType,
      subject: produce.subject || 'id',
      time: produce.time || 'ts',
      geo: null,
      values: [
        {
          source: produce.value || columns.find((c) => c.name !== 'id' && c.name !== 'ts')?.name,
          name: produce.value || columns.find((c) => c.name !== 'id' && c.name !== 'ts')?.name,
          type: 'double precision',
        },
      ].filter((v) => v.source),
      sampling_interval: firstInterval ? `${firstInterval}min` : undefined,
      table_prefix: prefix || undefined,
    },
  }, schema, outTable);

  return {
    ok: true,
    write_back: true,
    interval_minutes: intervals,
    table_prefix: prefix || null,
    physical_schema: schema,
    physical_table: outTable,
    signal: signalType,
    inserted,
  };
}

export { parseIntervalMinutes, sqlIdent, assertCanViewDataset };
