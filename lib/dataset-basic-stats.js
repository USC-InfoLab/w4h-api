/**
 * Cached vendor-agnostic dataset stats (ADR-020).
 *
 * Preference: D-02 catalog columns on w4h.datasets when populated
 * (subject_count, date_start/date_end, device_type). Otherwise derive
 * subjects and date range from mapped tables, and row counts from pg_class
 * estimates. Device type is catalog-only — do not infer from source_name.
 */

import { query } from '../db.js';

const QUALIFIED_REGCLASS = `
  CASE
    WHEN t.physical_schema IS NOT NULL AND t.physical_table IS NOT NULL
      THEN format('%I.%I', t.physical_schema, t.physical_table)
    WHEN position('.' IN t.name) > 0 THEN t.name
    ELSE format('%I', t.name)
  END
`;

function asObject(value) {
  if (!value) return {};
  if (typeof value === 'string') {
    try {
      const parsed = JSON.parse(value);
      return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : {};
    } catch {
      return {};
    }
  }
  return typeof value === 'object' && !Array.isArray(value) ? value : {};
}

function sqlIdent(name) {
  const s = String(name || '').trim();
  if (!/^[a-z_][a-z0-9_]*$/i.test(s)) return null;
  return s;
}

function isoDay(value) {
  if (value == null || value === '') return null;
  if (value instanceof Date && !Number.isNaN(value.getTime())) {
    const y = value.getFullYear();
    const month = String(value.getMonth() + 1).padStart(2, '0');
    const day = String(value.getDate()).padStart(2, '0');
    return `${y}-${month}-${day}`;
  }
  const match = String(value).match(/(\d{4}-\d{2}-\d{2})/);
  return match ? match[1] : null;
}

function isSubjectsTable(table) {
  const name = String(table?.name || '').toLowerCase();
  const mapping = asObject(table?.mapping_json);
  const signalType = String(mapping.signal_type || mapping.signal?.signal_type || '').toLowerCase();
  return name === 'subjects' || name.endsWith('_subjects') || signalType === 'subjects';
}

function mappingSpec(mappingJson) {
  const mapping = asObject(mappingJson);
  const nested = asObject(mapping.geomts_mapping);
  return nested.signal_type || nested.time || nested.subject ? nested : mapping;
}

function dateColumnFromMapping(mappingJson) {
  const mapping = mappingSpec(mappingJson);
  const time = mapping.time || mapping.signal?.time || mapping.timestamp;
  if (typeof time === 'string' && sqlIdent(time)) return sqlIdent(time);
  if (time && typeof time === 'object') {
    if (time.column && sqlIdent(time.column)) return sqlIdent(time.column);
    if (Array.isArray(time.combine)) {
      const namedDate = time.combine.find((col) => String(col).toLowerCase() === 'date');
      const picked = namedDate || time.combine[0];
      if (sqlIdent(picked)) return sqlIdent(picked);
    }
  }
  const named = sqlIdent(mapping.date);
  return named || null;
}

/** True when cached basic_stats is missing ADR-020 keys. */
export function needsBasicStatsRefresh(summary) {
  const existing = asObject(summary);
  const basic = asObject(existing.basic_stats);
  if (!basic || typeof basic !== 'object' || Array.isArray(basic)) return true;
  if (!basic.row_counts || typeof basic.row_counts !== 'object' || Array.isArray(basic.row_counts)) {
    return true;
  }
  if (!Object.prototype.hasOwnProperty.call(basic, 'subjects_total')) return true;
  if (!basic.date_range || typeof basic.date_range !== 'object' || Array.isArray(basic.date_range)) {
    return true;
  }
  // Pre-ADR-020 alias; strip on next persist so jsonb matches ADR-020.
  if (Object.prototype.hasOwnProperty.call(existing, 'participants')) return true;
  return false;
}

/** Per-table row estimates plus mapping metadata for one or more datasets. */
export async function fetchTableEstimates(datasetIds) {
  const ids = [...new Set((datasetIds || []).map((id) => String(id || '')).filter(Boolean))];
  const byDataset = new Map(ids.map((id) => [id, []]));
  if (!ids.length) return byDataset;

  try {
    const result = await query(
      `SELECT t.dataset_id,
              t.name,
              t.source_name,
              t.mapping_json,
              t.physical_schema,
              t.physical_table,
              COALESCE(s.n_live_tup, c.reltuples, 0)::bigint AS estimated_rows
       FROM w4h.tables t
       LEFT JOIN pg_class c
         ON c.oid = to_regclass(${QUALIFIED_REGCLASS})
       LEFT JOIN pg_stat_all_tables s ON s.relid = c.oid
       WHERE t.dataset_id = ANY($1::text[])
       ORDER BY t.dataset_id, t.name`,
      [ids]
    );
    for (const row of result.rows || []) {
      const id = String(row.dataset_id);
      if (!byDataset.has(id)) byDataset.set(id, []);
      byDataset.get(id).push({
        name: row.name,
        source_name: row.source_name,
        mapping_json: row.mapping_json,
        physical_schema: row.physical_schema,
        physical_table: row.physical_table,
        estimated_rows: Math.max(0, Math.round(Number(row.estimated_rows) || 0)),
      });
    }
  } catch {
    // w4h.tables may not exist on older installs.
  }
  return byDataset;
}

function qualifiedTable(schemaName, tableName) {
  const schema = sqlIdent(schemaName);
  const table = sqlIdent(tableName);
  if (!schema || !table) return null;
  return `"${schema}"."${table}"`;
}

function asStringList(value) {
  if (Array.isArray(value)) return value.map((item) => String(item || '').trim()).filter(Boolean);
  if (typeof value === 'string') {
    try {
      return asStringList(JSON.parse(value));
    } catch {
      return [];
    }
  }
  return [];
}

function physicalRel(table) {
  const fromPhysical = qualifiedTable(table?.physical_schema, table?.physical_table);
  if (fromPhysical) return fromPhysical;
  const raw = String(table?.name || '').trim();
  if (!raw) return null;
  if (raw.includes('.')) {
    const [schema, name] = raw.split('.', 2);
    return qualifiedTable(schema, name);
  }
  return qualifiedTable('public', raw);
}

function previewCell(value) {
  if (value == null) return null;
  if (value instanceof Date && !Number.isNaN(value.getTime())) return value.toISOString();
  if (typeof value === 'bigint') return value.toString();
  if (typeof value === 'object') {
    try {
      return JSON.stringify(value);
    } catch {
      return String(value);
    }
  }
  return value;
}

/** Column names per registered table, from information_schema (catalog only). */
export async function fetchTableColumnMap(datasetIds) {
  const ids = [...new Set((datasetIds || []).map((id) => String(id || '')).filter(Boolean))];
  const byDataset = new Map(ids.map((id) => [id, {}]));
  if (!ids.length) return byDataset;

  try {
    const result = await query(
      `SELECT t.dataset_id,
              t.name,
              COALESCE(
                (
                  SELECT json_agg(c.column_name ORDER BY c.ordinal_position)
                  FROM information_schema.columns c
                  WHERE c.table_schema = COALESCE(
                          NULLIF(t.physical_schema, ''),
                          CASE
                            WHEN position('.' IN t.name) > 0 THEN split_part(t.name, '.', 1)
                            ELSE 'public'
                          END
                        )
                    AND c.table_name = COALESCE(
                          NULLIF(t.physical_table, ''),
                          CASE
                            WHEN position('.' IN t.name) > 0 THEN split_part(t.name, '.', 2)
                            ELSE t.name
                          END
                        )
                ),
                '[]'::json
              ) AS columns
       FROM w4h.tables t
       WHERE t.dataset_id = ANY($1::text[])
       ORDER BY t.dataset_id, t.name`,
      [ids]
    );
    for (const row of result.rows || []) {
      const id = String(row.dataset_id);
      if (!byDataset.has(id)) byDataset.set(id, {});
      if (!row.name) continue;
      byDataset.get(id)[row.name] = asStringList(row.columns);
    }
  } catch {
    // w4h.tables or information_schema may be unavailable.
  }
  return byDataset;
}

const PREVIEW_ROW_CAP = 5;

/** First N rows plus column names. Not cached — live catalog read only. */
export async function fetchTablePreviews(datasetId, { limit = PREVIEW_ROW_CAP } = {}) {
  const id = String(datasetId || '').trim();
  const cap = Math.min(Math.max(Number(limit) || PREVIEW_ROW_CAP, 1), PREVIEW_ROW_CAP);
  if (!id) return [];

  const estimates = await fetchTableEstimates([id]);
  const tables = estimates.get(id) || [];
  const columnMap = await fetchTableColumnMap([id]);
  const columnsByTable = columnMap.get(id) || {};
  const previews = [];

  for (const table of tables) {
    const columns = asStringList(columnsByTable[table.name]);
    const rel = physicalRel(table);
    let rows = [];
    if (rel) {
      try {
        const result = await query(`SELECT * FROM ${rel} LIMIT $1`, [cap]);
        const keys = columns.length ? columns : Object.keys(result.rows[0] || {});
        rows = (result.rows || []).map((row) => {
          const out = {};
          for (const key of keys) out[key] = previewCell(row[key]);
          return out;
        });
        if (!columns.length) columns.push(...keys);
      } catch {
        // Physical table missing or unreadable.
      }
    }
    if (!table?.name) continue;
    previews.push({ name: table.name, columns, rows });
  }

  return previews;
}

async function countDistinctSubjects(tableRows) {
  const subjects = (tableRows || []).find(isSubjectsTable);
  if (!subjects) return null;
  const rel = qualifiedTable(subjects.physical_schema, subjects.physical_table);
  const mapping = mappingSpec(subjects.mapping_json);
  const col = sqlIdent(mapping.subject) || 'id';
  const estimated = Number(subjects.estimated_rows);
  if (rel && (!Number.isFinite(estimated) || estimated < 10000)) {
    try {
      const counted = await query(`SELECT COUNT(DISTINCT "${col}")::int AS n FROM ${rel}`);
      if (counted.rowCount) return Number(counted.rows[0].n);
    } catch {
      // Fall through to catalog estimate.
    }
  }
  return Number.isFinite(estimated) ? estimated : null;
}

function dateExprsForTable(table) {
  if (isSubjectsTable(table)) {
    // Enrollment window on the subjects roster (bounded, ADR-020 date range fallback).
    return { minExpr: 'start_date', maxExpr: 'end_date', fallback: 'date' };
  }
  const col = dateColumnFromMapping(table.mapping_json) || 'date';
  return { minExpr: col, maxExpr: col, fallback: null };
}

async function mappedDateRange(tableRows) {
  let start = null;
  let end = null;
  const ordered = [...(tableRows || [])].sort(
    (a, b) => (Number(a.estimated_rows) || 0) - (Number(b.estimated_rows) || 0)
  );
  for (const table of ordered) {
    const large = (Number(table.estimated_rows) || 0) > 250000;
    // Skip unindexed MIN/MAX on huge signal tables once a range exists (list-endpoint bound).
    if (large && start && end) continue;
    const rel = qualifiedTable(table.physical_schema, table.physical_table);
    if (!rel) continue;
    const { minExpr, maxExpr, fallback } = dateExprsForTable(table);
    const attempts = fallback && (minExpr !== fallback || maxExpr !== fallback)
      ? [
          [minExpr, maxExpr],
          [fallback, fallback],
        ]
      : [[minExpr, maxExpr]];
    for (const [minCol, maxCol] of attempts) {
      if (!sqlIdent(minCol) || !sqlIdent(maxCol)) continue;
      try {
        const result = await query(
          `SELECT MIN("${sqlIdent(minCol)}")::date AS dmin, MAX("${sqlIdent(maxCol)}")::date AS dmax FROM ${rel}`
        );
        const dmin = isoDay(result.rows[0]?.dmin);
        const dmax = isoDay(result.rows[0]?.dmax);
        if (dmin && (!start || dmin < start)) start = dmin;
        if (dmax && (!end || dmax > end)) end = dmax;
        break;
      } catch {
        // Table or column missing; try fallback / next mapped table.
      }
    }
  }
  return { start, end };
}

function catalogDeviceTypes(catalogRow) {
  const catalog = String(catalogRow?.device_type || '').trim();
  return catalog ? [catalog] : [];
}

export function buildBasicStats(catalogRow, tableRows = [], extras = {}) {
  const row_counts = {};
  for (const table of tableRows) {
    if (!table?.name) continue;
    row_counts[table.name] = Math.max(0, Math.round(Number(table.estimated_rows) || 0));
  }
  const observations = Object.values(row_counts).reduce((sum, n) => sum + n, 0);
  const catalogSubjects =
    catalogRow?.subject_count == null || catalogRow?.subject_count === ''
      ? null
      : Number(catalogRow.subject_count);
  const subjects =
    catalogSubjects != null && Number.isFinite(catalogSubjects)
      ? catalogSubjects
      : extras.subjects_total != null
        ? extras.subjects_total
        : null;
  const catalogRange = {
    start: isoDay(catalogRow?.date_start),
    end: isoDay(catalogRow?.date_end),
  };
  const fromExtras = extras.date_range && typeof extras.date_range === 'object'
    ? extras.date_range
    : null;
  const date_range = {
    start: catalogRange.start || isoDay(fromExtras?.start) || null,
    end: catalogRange.end || isoDay(fromExtras?.end) || null,
  };
  const device_types = catalogDeviceTypes(catalogRow);

  return {
    table_count: tableRows.length,
    observations,
    basic_stats: {
      subjects_total: subjects,
      date_range,
      device_types,
      row_counts,
      summary_updated_at: extras.summary_updated_at || new Date().toISOString(),
    },
  };
}

export function mergeSummaryCache(currentSummary, catalogRow, computed) {
  const existing = asObject(currentSummary);
  return {
    ...existing,
    table_count: computed.table_count,
    physical_table_count: computed.table_count,
    observations: computed.observations,
    observations_approx: true,
    observations_source: 'postgres_catalog_estimate',
    basic_stats: computed.basic_stats,
  };
}

export async function persistBasicStats(datasetId) {
  const id = String(datasetId || '').trim();
  if (!id) return null;

  const catalog = await query(
    `SELECT id, device_type, subject_count, date_start, date_end, summary
     FROM w4h.datasets WHERE id = $1`,
    [id]
  );
  if (!catalog.rowCount) return null;

  const row = catalog.rows[0];
  const estimates = await fetchTableEstimates([id]);
  const tableRows = estimates.get(id) || [];

  const catalogSubjects =
    row.subject_count == null || row.subject_count === '' ? null : Number(row.subject_count);
  const catalogStart = isoDay(row.date_start);
  const catalogEnd = isoDay(row.date_end);

  const subjects_total =
    catalogSubjects != null && Number.isFinite(catalogSubjects)
      ? catalogSubjects
      : await countDistinctSubjects(tableRows);
  const date_range =
    catalogStart || catalogEnd
      ? { start: catalogStart, end: catalogEnd }
      : await mappedDateRange(tableRows);

  const computed = buildBasicStats(row, tableRows, {
    subjects_total,
    date_range,
  });
  const next = mergeSummaryCache(row.summary, row, computed);
  const range = computed.basic_stats.date_range || {};
  const updated = await query(
    `UPDATE w4h.datasets
     SET summary = $2::jsonb,
         summary_updated_at = NOW(),
         subject_count = COALESCE(subject_count, $3),
         date_start = COALESCE(date_start, $4::date),
         date_end = COALESCE(date_end, $5::date)
     WHERE id = $1
     RETURNING summary, summary_updated_at, subject_count, date_start, date_end, device_type`,
    [
      id,
      JSON.stringify(next),
      computed.basic_stats.subjects_total,
      range.start || null,
      range.end || null,
    ]
  );
  const saved = updated.rows[0];
  return {
    summary: saved.summary,
    summary_updated_at: saved.summary_updated_at,
    basic_stats: computed.basic_stats,
    table_count: computed.table_count,
    observations: computed.observations,
    subject_count: saved.subject_count,
    date_start: saved.date_start,
    date_end: saved.date_end,
    device_type: saved.device_type,
  };
}
