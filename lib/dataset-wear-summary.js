/**
 * Cohort wear-compliance summary from the mapped heart-rate (and optional enrollment) tables.
 *
 * Port of the protocol in `w4h-fitbit-wear-compliance` (`ComplianceAnalyzer`, ADR-019): heart
 * rate is a wear proxy, so a sample with heart_rate <> 0 inside a daily protocol window counts as
 * "worn" for that sample's interval. A day is compliant when it covers at least
 * `daily_band_wear_percentage` of the window minutes; a subject is compliant when at least
 * `overall_band_wear_percentage` of their days with data are. Aggregated in SQL, bounded, no raw
 * rows returned. This is the interim platform producer for `datasets.summary.wear_compliance`
 * until the logic has its own SQL-engine manifest (ADR-023).
 *
 * Two deliberate differences from the notebook, both documented in the repo README link above:
 *  - the interval of one sample comes from the mapping (default 1 min), not "one row = one minute";
 *  - a NULL heart rate is not a sample (pandas counted NaN as worn).
 */

import { getPool, query } from '../db.js';

const SUBJECT_CAP = 100;
const STATEMENT_TIMEOUT_MS = 20_000;
const WINDOW_CAP = 4;

export const DEFAULT_WEAR_PROTOCOL = {
  daily_band_wear_percentage: 60,
  overall_band_wear_percentage: 60,
  daily_band_wear_hours: [['08:00', '20:00']],
};

function sqlIdent(name) {
  const s = String(name || '').trim();
  if (!/^[a-z_][a-z0-9_]*$/i.test(s)) throw new Error(`Invalid SQL identifier: ${name}`);
  return s;
}

function qualify(schema, table) {
  return `${sqlIdent(schema)}.${sqlIdent(table)}`;
}

function parseMapping(raw) {
  if (!raw) return {};
  if (typeof raw === 'object') return raw;
  try { return JSON.parse(raw); } catch { return {}; }
}

function signalTypeOf(row) {
  const mapping = parseMapping(row.mapping_json);
  return String(mapping.signal_type || row.name || '').trim().toLowerCase();
}

function pickExisting(columns, candidates) {
  for (const name of candidates) {
    const lower = name && String(name).toLowerCase();
    if (lower && columns.has(lower)) return lower;
  }
  return null;
}

function valueNames(mapping) {
  return (Array.isArray(mapping.values) ? mapping.values : [])
    .map((v) => (typeof v === 'string' ? v : (v?.source || v?.name)))
    .filter(Boolean)
    .map(String);
}

function parseIntervalMinutes(value) {
  if (value == null || value === '') return null;
  if (typeof value === 'number' && Number.isFinite(value) && value > 0) return value;
  const match = String(value).trim().match(/^(\d+(?:\.\d+)?)\s*(min|m|minutes?)?$/i);
  return match ? Number(match[1]) : null;
}

function timeExpr(mapping, columns, alias) {
  const a = sqlIdent(alias);
  const time = mapping?.time;
  if (time && typeof time === 'object' && Array.isArray(time.combine) && time.combine.length >= 2) {
    const dateCol = pickExisting(columns, time.combine);
    const timeCol = time.combine.map(String).find((c) => c !== dateCol && columns.has(c));
    if (dateCol && timeCol) return `(${a}.${sqlIdent(dateCol)}::timestamp + ${a}.${sqlIdent(timeCol)}::time)`;
    if (dateCol) return `(${a}.${sqlIdent(dateCol)}::timestamp)`;
  }
  const col = pickExisting(columns, [typeof time === 'string' ? time : null, 'ts', 'timestamp', 'time', 'date']);
  return col ? `(${a}.${sqlIdent(col)}::timestamp)` : null;
}

async function tableColumns(schema, table) {
  const res = await query(
    `SELECT column_name FROM information_schema.columns WHERE table_schema = $1 AND table_name = $2`,
    [schema, table]
  );
  return new Set(res.rows.map((r) => String(r.column_name)));
}

const HM = /^([01]?\d|2[0-3]):([0-5]\d)$/;
const toMinutes = (hm) => {
  const m = String(hm).match(HM);
  return m ? Number(m[1]) * 60 + Number(m[2]) : null;
};
const pad = (hm) => {
  const [h, m] = String(hm).split(':');
  return `${String(h).padStart(2, '0')}:${m}`;
};

/** Validate a protocol override; returns { ok, protocol } or { ok: false, error }. */
export function normalizeProtocol(raw) {
  const input = raw && typeof raw === 'object' ? raw : {};
  const pct = (value, fallback, label) => {
    if (value == null || value === '') return { value: fallback };
    const n = Number(value);
    if (!Number.isFinite(n) || n <= 0 || n > 100) return { error: `${label} must be between 1 and 100` };
    return { value: n };
  };
  const daily = pct(input.daily_band_wear_percentage, DEFAULT_WEAR_PROTOCOL.daily_band_wear_percentage, 'daily_band_wear_percentage');
  if (daily.error) return { ok: false, error: daily.error };
  const overall = pct(input.overall_band_wear_percentage, DEFAULT_WEAR_PROTOCOL.overall_band_wear_percentage, 'overall_band_wear_percentage');
  if (overall.error) return { ok: false, error: overall.error };

  let windows = DEFAULT_WEAR_PROTOCOL.daily_band_wear_hours;
  if (input.daily_band_wear_hours != null) {
    if (!Array.isArray(input.daily_band_wear_hours) || !input.daily_band_wear_hours.length
      || input.daily_band_wear_hours.length > WINDOW_CAP) {
      return { ok: false, error: `daily_band_wear_hours needs 1–${WINDOW_CAP} [start, end] windows` };
    }
    windows = [];
    for (const w of input.daily_band_wear_hours) {
      const s = Array.isArray(w) ? toMinutes(w[0]) : null;
      const e = Array.isArray(w) ? toMinutes(w[1]) : null;
      if (s == null || e == null || e <= s) {
        return { ok: false, error: 'Each wear window must be [HH:MM, HH:MM] with the end after the start (same day)' };
      }
      windows.push([pad(w[0]), pad(w[1])]);
    }
  }
  return {
    ok: true,
    protocol: {
      daily_band_wear_percentage: daily.value,
      overall_band_wear_percentage: overall.value,
      daily_band_wear_hours: windows,
    },
  };
}

async function timed(fn) {
  const pool = getPool();
  if (!pool) throw new Error('Database not configured');
  const client = await pool.connect();
  try {
    await client.query('BEGIN READ ONLY');
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

const round1 = (v) => (v == null || !Number.isFinite(Number(v)) ? null : Number(Number(v).toFixed(1)));

export async function computeWearSummary(datasetId, protocolInput = null) {
  const parsed = normalizeProtocol(protocolInput);
  if (!parsed.ok) return { available: false, reason: parsed.error, invalid: true };
  const protocol = parsed.protocol;

  const tablesRes = await query(
    `SELECT name, mapping_json, physical_schema, physical_table
     FROM w4h.tables WHERE dataset_id = $1`,
    [datasetId]
  );
  const tables = (tablesRes.rows || []).filter((r) => r.physical_schema && r.physical_table);
  const hrTable = tables.find((r) => signalTypeOf(r) === 'heart_rate');
  if (!hrTable) {
    return { available: false, reason: 'No mapped heart_rate signal — heart rate is the wear proxy' };
  }

  const hrMap = parseMapping(hrTable.mapping_json);
  const hrCols = await tableColumns(hrTable.physical_schema, hrTable.physical_table);
  const subjectCol = pickExisting(hrCols, [hrMap.subject, 'id', 'subject_id', 'user_id', 'participant']);
  const valueCol = pickExisting(hrCols, [...valueNames(hrMap), 'heart_rate', 'value', 'bpm']);
  const ts = timeExpr(hrMap, hrCols, 't');
  if (!subjectCol || !valueCol || !ts) {
    return { available: false, reason: 'heart_rate is missing a subject, time or value mapping' };
  }
  const interval = parseIntervalMinutes(hrMap.sampling_interval) || parseIntervalMinutes(hrMap.interval_minutes) || 1;

  // Enrollment dates bound each subject's days, like subjects.start_date / end_date in the notebook.
  let enrollJoin = '';
  let enrollWhere = '';
  const enrollTable = tables.find((r) => signalTypeOf(r) === 'subjects' || r.name === 'subjects');
  if (enrollTable) {
    const eMap = parseMapping(enrollTable.mapping_json);
    const eCols = await tableColumns(enrollTable.physical_schema, enrollTable.physical_table);
    const eSubject = pickExisting(eCols, [eMap.subject, 'id', 'subject_id', 'user_id']);
    const eStart = pickExisting(eCols, ['start_date', 'enrollment_start', 'collection_start']);
    const eEnd = pickExisting(eCols, ['end_date', 'enrollment_end', 'collection_end']);
    if (eSubject && (eStart || eEnd)) {
      enrollJoin = `LEFT JOIN ${qualify(enrollTable.physical_schema, enrollTable.physical_table)} e
                    ON e.${sqlIdent(eSubject)}::text = h.sid`;
      const parts = [];
      if (eStart) parts.push(`(e.${sqlIdent(eStart)} IS NULL OR h.ts >= e.${sqlIdent(eStart)}::timestamp)`);
      if (eEnd) parts.push(`(e.${sqlIdent(eEnd)} IS NULL OR h.ts < (e.${sqlIdent(eEnd)}::date + 1)::timestamp)`);
      enrollWhere = `AND ${parts.join(' AND ')}`;
    }
  }

  const params = [];
  const windowPredicate = protocol.daily_band_wear_hours.map(([start, end]) => {
    params.push(start, end);
    return `(h.ts::time >= $${params.length - 1}::time AND h.ts::time <= $${params.length}::time)`;
  }).join(' OR ');
  const windowMinutes = protocol.daily_band_wear_hours
    .reduce((sum, [s, e]) => sum + (toMinutes(e) - toMinutes(s)), 0);
  params.push(interval, windowMinutes, protocol.daily_band_wear_percentage);
  const pInterval = `$${params.length - 2}::double precision`;
  const pWindow = `$${params.length - 1}::double precision`;
  const pDaily = `$${params.length}::double precision`;

  const sql = `
    WITH h AS (
      SELECT t.${sqlIdent(subjectCol)}::text AS sid, ${ts} AS ts, t.${sqlIdent(valueCol)}::double precision AS hr
      FROM ${qualify(hrTable.physical_schema, hrTable.physical_table)} t
      WHERE ${ts} IS NOT NULL AND t.${sqlIdent(subjectCol)} IS NOT NULL
    ),
    bounded AS (
      SELECT h.sid, h.ts, h.hr FROM h ${enrollJoin} WHERE TRUE ${enrollWhere}
    ),
    daily AS (
      SELECT h.sid, h.ts::date AS day,
             COUNT(*) FILTER (WHERE h.hr IS NOT NULL AND h.hr <> 0 AND (${windowPredicate})) AS active
      FROM bounded h GROUP BY h.sid, day
    ),
    pct AS (
      SELECT sid, day, LEAST(100.0, active * ${pInterval} / ${pWindow} * 100.0) AS pct FROM daily
    )
    SELECT sid,
           COUNT(*)::int AS days,
           COUNT(*) FILTER (WHERE pct >= ${pDaily})::int AS compliant_days,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY pct) AS median_pct,
           GROUPING(sid)::int AS is_total
    FROM pct
    GROUP BY GROUPING SETS ((sid), ())
    ORDER BY is_total, sid`;
  const res = await timed((c) => c.query(sql, params));

  const subjects = [];
  let cohortMedian = null;
  for (const row of res.rows) {
    if (row.is_total === 1) { cohortMedian = round1(row.median_pct); continue; }
    const days = Number(row.days) || 0;
    const compliant = Number(row.compliant_days) || 0;
    const overall = days ? (compliant / days) * 100 : 0;
    subjects.push({
      subject_id: String(row.sid),
      days_with_data: days,
      compliant_days: compliant,
      overall_compliance_percentage: round1(overall),
      is_overall_compliant: overall >= protocol.overall_band_wear_percentage,
      median_daily_wear_pct: round1(row.median_pct),
    });
  }
  if (!subjects.length) {
    return { available: false, reason: 'No heart-rate days to assess' };
  }
  const capped = subjects.length > SUBJECT_CAP;
  const kept = subjects.slice(0, SUBJECT_CAP);
  const compliantSubjects = kept.filter((s) => s.is_overall_compliant).length;
  return {
    available: true,
    source: 'platform',
    signal: 'heart_rate',
    interval_minutes: interval,
    computed_at: new Date().toISOString(),
    protocol,
    subjects_total: kept.length,
    subjects_compliant: compliantSubjects,
    compliance_pct: round1((compliantSubjects / kept.length) * 100),
    median_daily_wear_pct: cohortMedian,
    subjects_capped: capped,
    subjects: kept,
  };
}

/** Merge computed keys (`wear_compliance`, `mets_summary`) into a dataset summary. */
export function mergeSummaryKeys(existingSummary, patch) {
  const current =
    existingSummary && typeof existingSummary === 'object' && !Array.isArray(existingSummary)
      ? existingSummary
      : {};
  return { ...current, ...patch };
}
