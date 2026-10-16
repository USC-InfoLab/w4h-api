/**
 * Cohort METs summary from mapped calories + weight tables.
 *
 * Formula (ICDE demo `calculate_mets` / `w4h-compute-mets`):
 *   METs ≈ calories / (weight_kg × 0.25) per a 15-minute reference interval
 *
 * The 0.25 constant is calibrated for 15 minutes (classic 1 MET ≈ 1 kcal/kg/hour).
 * Native row intervals are scaled so 1-minute Fitbit rows and 5-minute ICDE demo
 * rows stay in MET units — see w4h-compute-mets README.
 *
 * Missing weight → latest known per subject, else DEFAULT_WEIGHT_KG (70), matching
 * the ICDE demo's default. Aggregates in SQL; does not pull per-interval rows.
 */

import { query } from '../db.js';
import { quoteIdent } from './sql-ident.js';

const KCAL_PER_KG_PER_MET_REFERENCE = 0.25;
const REFERENCE_INTERVAL_MINUTES = 15;
const DEFAULT_WEIGHT_KG = 70;
const SUBJECT_CAP = 100;

function qualify(schema, table) {
  return `${quoteIdent(schema)}.${quoteIdent(table)}`;
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

function signalTypeOf(row) {
  const mapping = parseMapping(row.mapping_json);
  return String(mapping.signal_type || row.name || '').trim().toLowerCase();
}

function mappedSource(mapping, roleName, fallbacks) {
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

function parseIntervalMinutes(value) {
  if (value == null || value === '') return null;
  if (typeof value === 'number' && Number.isFinite(value) && value > 0) return value;
  const match = String(value).trim().match(/^(\d+(?:\.\d+)?)\s*(min|m|minutes?)?$/i);
  if (match) return Number(match[1]);
  return null;
}

function inferIntervalMinutes(table) {
  const mapping = parseMapping(table?.mapping_json);
  const fromMapping =
    parseIntervalMinutes(mapping.sampling_interval) ||
    parseIntervalMinutes(mapping.interval_minutes);
  if (fromMapping) return fromMapping;
  const phys = String(table?.physical_table || table?.name || '').toLowerCase();
  if (phys.includes('icde')) return 5;
  return 1;
}

function intervalConstant(intervalMinutes) {
  return KCAL_PER_KG_PER_MET_REFERENCE * (intervalMinutes / REFERENCE_INTERVAL_MINUTES);
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

function pickExisting(columns, candidates) {
  for (const name of candidates) {
    if (name && columns.has(name)) return name;
  }
  return null;
}

async function loadMappedTables(datasetId) {
  const res = await query(
    `SELECT name, mapping_json, physical_schema, physical_table
     FROM w4h.tables
     WHERE dataset_id = $1`,
    [datasetId]
  );
  return (res.rows || []).filter((row) => row.physical_schema && row.physical_table);
}

function findSignal(tables, type) {
  return tables.find((row) => signalTypeOf(row) === type) || null;
}

export function deriveSubjectMets({
  caloriesSum,
  rows,
  days,
  weightKg,
  intervalMinutes,
  weightIsDefault = false,
  vendorMeanMets = null,
}) {
  const n = Number(rows) || 0;
  const interval = Number(intervalMinutes) || 1;
  const weight = Number(weightKg) || DEFAULT_WEIGHT_KG;
  const constant = intervalConstant(interval);
  const meanMets = n > 0 && constant > 0 ? caloriesSum / n / (weight * constant) : null;
  const dayCount = Number(days) || 1;
  const metHoursPerDay =
    meanMets != null ? (meanMets * interval * n) / 60 / dayCount : null;
  return {
    mean_mets: meanMets,
    met_hours_per_day: metHoursPerDay,
    weight_kg: weight,
    weight_is_default: Boolean(weightIsDefault),
    vendor_mean_mets: vendorMeanMets,
    rows: n,
    days: dayCount,
  };
}

function roundMetric(value, digits = 2) {
  if (value == null || !Number.isFinite(Number(value))) return null;
  return Number(Number(value).toFixed(digits));
}

function cohortFromSubjects(subjects, extras = {}) {
  if (!subjects.length) {
    return {
      available: false,
      reason: extras.reason || 'No calorie rows to derive METs from',
      ...extras,
    };
  }
  const means = subjects.map((s) => s.mean_mets).filter((v) => v != null);
  const hours = subjects.map((s) => s.met_hours_per_day).filter((v) => v != null);
  const mean = (list) => list.reduce((a, b) => a + b, 0) / list.length;
  return {
    available: true,
    signal: extras.signal || 'mets_computed',
    formula: 'calories / (weight_kg × 0.25) per 15-minute reference interval',
    interval_minutes: extras.interval_minutes,
    default_weight_kg: DEFAULT_WEIGHT_KG,
    subjects_with_data: subjects.length,
    cohort_mean_mets: roundMetric(means.length ? mean(means) : null),
    cohort_min_mets: means.length ? roundMetric(Math.min(...means)) : null,
    cohort_max_mets: means.length ? roundMetric(Math.max(...means)) : null,
    cohort_met_hours_per_day: roundMetric(hours.length ? mean(hours) : null),
    subjects: subjects.slice(0, SUBJECT_CAP).map((s) => ({
      subject_id: s.subject_id,
      mean_mets: roundMetric(s.mean_mets),
      met_hours_per_day: roundMetric(s.met_hours_per_day),
      rows: s.rows,
      weight_is_default: s.weight_is_default,
    })),
    ...extras.meta,
  };
}

async function summarizeComputedTable(table) {
  const schema = table.physical_schema;
  const name = table.physical_table;
  const mapping = parseMapping(table.mapping_json);
  const columns = await tableColumns(schema, name);
  const subjectCol = pickExisting(columns, [mapping.subject, 'id', 'subject_id', 'user_id']);
  const valueCol = pickExisting(columns, [
    mappedSource(mapping, 'mets_computed', ['value', 'mets_computed', 'mets']),
    'value',
    'mets_computed',
    'mets',
  ]);
  const dateCol = pickExisting(columns, ['date', 'timestamp']);
  if (!subjectCol || !valueCol) return null;

  const intervalMinutes = inferIntervalMinutes(table);
  const qualified = qualify(schema, name);
  const daysExpr = dateCol
    ? `COUNT(DISTINCT ${quoteIdent(dateCol)})::int`
    : '1';
  const res = await query(
    `SELECT ${quoteIdent(subjectCol)}::text AS subject_id,
            COUNT(*)::bigint AS rows,
            AVG(${quoteIdent(valueCol)})::double precision AS mean_mets,
            ${daysExpr} AS days
     FROM ${qualified}
     WHERE ${quoteIdent(valueCol)} IS NOT NULL
     GROUP BY ${quoteIdent(subjectCol)}
     ORDER BY ${quoteIdent(subjectCol)}
     LIMIT ${SUBJECT_CAP}`
  );

  const subjects = res.rows.map((row) => {
    const rows = Number(row.rows) || 0;
    const days = Number(row.days) || 1;
    const meanMets = Number(row.mean_mets);
    return {
      subject_id: row.subject_id,
      mean_mets: Number.isFinite(meanMets) ? meanMets : null,
      met_hours_per_day:
        Number.isFinite(meanMets) ? (meanMets * intervalMinutes * rows) / 60 / days : null,
      rows,
      weight_is_default: false,
    };
  });

  return cohortFromSubjects(subjects, {
    signal: 'mets_computed',
    interval_minutes: intervalMinutes,
    meta: { source: 'mets_computed' },
  });
}

async function summarizeDerived(caloriesTable, weightTable) {
  const mapping = parseMapping(caloriesTable.mapping_json);
  const weightMapping = parseMapping(weightTable?.mapping_json);
  const calCols = await tableColumns(caloriesTable.physical_schema, caloriesTable.physical_table);
  const subjectCol = pickExisting(calCols, [mapping.subject, 'id', 'subject_id', 'user_id']);
  const caloriesCol = pickExisting(calCols, [
    mappedSource(mapping, 'calories', ['value', 'calories']),
    'value',
    'calories',
  ]);
  const dateCol = pickExisting(calCols, ['date', 'timestamp']);
  const vendorMetsCol = pickExisting(calCols, [
    mappedSource(mapping, 'mets', ['mets']),
    'mets',
  ]);
  if (!subjectCol || !caloriesCol) {
    return {
      available: false,
      reason: 'Calories table is missing subject or value columns',
    };
  }

  let weightSubjectCol = null;
  let weightValueCol = null;
  let weightDateCol = null;
  if (weightTable) {
    const wCols = await tableColumns(weightTable.physical_schema, weightTable.physical_table);
    weightSubjectCol = pickExisting(wCols, [weightMapping.subject, 'id', 'subject_id', 'user_id']);
    weightValueCol = pickExisting(wCols, [
      mappedSource(weightMapping, 'weight_kg', ['weight', 'weight_kg', 'value']),
      'weight',
      'weight_kg',
      'value',
    ]);
    weightDateCol = pickExisting(wCols, ['date', 'timestamp']);
  }

  const intervalMinutes = inferIntervalMinutes(caloriesTable);
  const calRel = qualify(caloriesTable.physical_schema, caloriesTable.physical_table);
  const vendorSelect = vendorMetsCol
    ? `, AVG(${quoteIdent(vendorMetsCol)} / 10.0)::double precision AS vendor_mean_mets`
    : ', NULL::double precision AS vendor_mean_mets';
  const daysExpr = dateCol ? `COUNT(DISTINCT ${quoteIdent(dateCol)})::int` : '1';

  let sql;
  if (weightTable && weightSubjectCol && weightValueCol) {
    const wRel = qualify(weightTable.physical_schema, weightTable.physical_table);
    const wOrder = weightDateCol
      ? `${quoteIdent(weightSubjectCol)}, ${quoteIdent(weightDateCol)} DESC NULLS LAST`
      : quoteIdent(weightSubjectCol);
    sql = `
      WITH latest_weight AS (
        SELECT DISTINCT ON (${quoteIdent(weightSubjectCol)})
               ${quoteIdent(weightSubjectCol)}::text AS subject_id,
               ${quoteIdent(weightValueCol)}::double precision AS weight_kg
        FROM ${wRel}
        WHERE ${quoteIdent(weightValueCol)} IS NOT NULL
        ORDER BY ${wOrder}
      ),
      cal AS (
        SELECT ${quoteIdent(subjectCol)}::text AS subject_id,
               COUNT(*)::bigint AS rows,
               SUM(${quoteIdent(caloriesCol)})::double precision AS calories_sum,
               ${daysExpr} AS days
               ${vendorSelect}
        FROM ${calRel}
        WHERE ${quoteIdent(caloriesCol)} IS NOT NULL
        GROUP BY ${quoteIdent(subjectCol)}
      )
      SELECT cal.subject_id,
             cal.rows,
             cal.calories_sum,
             cal.days,
             cal.vendor_mean_mets,
             lw.weight_kg,
             (lw.weight_kg IS NULL) AS weight_is_default
      FROM cal
      LEFT JOIN latest_weight lw ON lw.subject_id = cal.subject_id
      ORDER BY cal.subject_id
      LIMIT ${SUBJECT_CAP}
    `;
  } else {
    sql = `
      SELECT ${quoteIdent(subjectCol)}::text AS subject_id,
             COUNT(*)::bigint AS rows,
             SUM(${quoteIdent(caloriesCol)})::double precision AS calories_sum,
             ${daysExpr} AS days
             ${vendorSelect},
             NULL::double precision AS weight_kg,
             TRUE AS weight_is_default
      FROM ${calRel}
      WHERE ${quoteIdent(caloriesCol)} IS NOT NULL
      GROUP BY ${quoteIdent(subjectCol)}
      ORDER BY ${quoteIdent(subjectCol)}
      LIMIT ${SUBJECT_CAP}
    `;
  }

  const res = await query(sql);
  const subjects = res.rows.map((row) => {
    const derived = deriveSubjectMets({
      caloriesSum: Number(row.calories_sum) || 0,
      rows: Number(row.rows) || 0,
      days: Number(row.days) || 1,
      weightKg: row.weight_kg == null ? DEFAULT_WEIGHT_KG : Number(row.weight_kg),
      intervalMinutes,
      weightIsDefault: Boolean(row.weight_is_default),
      vendorMeanMets: row.vendor_mean_mets == null ? null : Number(row.vendor_mean_mets),
    });
    return { subject_id: row.subject_id, ...derived };
  });

  const vendorMeans = subjects
    .map((s) => s.vendor_mean_mets)
    .filter((v) => v != null && Number.isFinite(v));
  return cohortFromSubjects(subjects, {
    signal: 'mets_computed',
    interval_minutes: intervalMinutes,
    meta: {
      source: weightTable ? 'calories+weight' : 'calories+default_weight',
      vendor_mean_mets: vendorMeans.length
        ? roundMetric(vendorMeans.reduce((a, b) => a + b, 0) / vendorMeans.length)
        : null,
    },
  });
}

async function summarizeVendorMets(caloriesTable) {
  const mapping = parseMapping(caloriesTable.mapping_json);
  const columns = await tableColumns(caloriesTable.physical_schema, caloriesTable.physical_table);
  const subjectCol = pickExisting(columns, [mapping.subject, 'id', 'subject_id', 'user_id']);
  const metsCol = pickExisting(columns, [mappedSource(mapping, 'mets', ['mets']), 'mets']);
  const dateCol = pickExisting(columns, ['date', 'timestamp']);
  if (!subjectCol || !metsCol) return null;

  const intervalMinutes = inferIntervalMinutes(caloriesTable);
  const qualified = qualify(caloriesTable.physical_schema, caloriesTable.physical_table);
  const daysExpr = dateCol ? `COUNT(DISTINCT ${quoteIdent(dateCol)})::int` : '1';
  const res = await query(
    `SELECT ${quoteIdent(subjectCol)}::text AS subject_id,
            COUNT(*)::bigint AS rows,
            AVG(${quoteIdent(metsCol)} / 10.0)::double precision AS mean_mets,
            ${daysExpr} AS days
     FROM ${qualified}
     WHERE ${quoteIdent(metsCol)} IS NOT NULL
     GROUP BY ${quoteIdent(subjectCol)}
     ORDER BY ${quoteIdent(subjectCol)}
     LIMIT ${SUBJECT_CAP}`
  );

  const subjects = res.rows.map((row) => {
    const rows = Number(row.rows) || 0;
    const days = Number(row.days) || 1;
    const meanMets = Number(row.mean_mets);
    return {
      subject_id: row.subject_id,
      mean_mets: Number.isFinite(meanMets) ? meanMets : null,
      met_hours_per_day:
        Number.isFinite(meanMets) ? (meanMets * intervalMinutes * rows) / 60 / days : null,
      rows,
      weight_is_default: false,
    };
  });

  return cohortFromSubjects(subjects, {
    signal: 'mets',
    interval_minutes: intervalMinutes,
    meta: { source: 'vendor_mets' },
  });
}

/**
 * Compute a cached mets_summary for one dataset. Bounded GROUP BY aggregates only.
 */
export async function computeMetsSummary(datasetId) {
  const tables = await loadMappedTables(datasetId);
  const computed = findSignal(tables, 'mets_computed');
  if (computed) {
    const summary = await summarizeComputedTable(computed);
    if (summary) return summary;
  }

  const calories = findSignal(tables, 'calories');
  const weight = findSignal(tables, 'weight');
  if (calories && weight) {
    return summarizeDerived(calories, weight);
  }
  if (calories) {
    const vendor = await summarizeVendorMets(calories);
    if (vendor) return vendor;
    return summarizeDerived(calories, null);
  }

  return {
    available: false,
    reason: 'Dataset has no calories, weight, or mets_computed signal',
  };
}

export function mergeMetsSummary(existingSummary, metsSummary) {
  const current =
    existingSummary && typeof existingSummary === 'object' && !Array.isArray(existingSummary)
      ? existingSummary
      : {};
  return {
    ...current,
    mets_summary: metsSummary,
  };
}

export {
  DEFAULT_WEIGHT_KG,
  KCAL_PER_KG_PER_MET_REFERENCE,
  REFERENCE_INTERVAL_MINUTES,
  intervalConstant,
};
