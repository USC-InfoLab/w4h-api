/**
 * Generic visualization queries (ADR-029).
 *
 * Reads a dataset's registered tables (w4h.tables role mapping: subject, time, values) and
 * returns aggregated or downsampled data for the in-app chart component. Vendor-agnostic:
 * nothing here knows "Fitbit". No free SQL — identifiers are validated against the mapping and
 * information_schema, every value is a bind parameter, and every result is capped.
 */

import { getPool, query } from '../db.js';
import { listDatasetSignals, sqlIdent } from './sql-contribution-engine.js';
import { CHART_KINDS } from './viz-display.js';

const STATEMENT_TIMEOUT_MS = 15_000;
const SUBJECT_LIST_CAP = 200;
const SERIES_SUBJECT_CAP = 12;
const COVERAGE_SUBJECT_CAP = 40;
const MAX_POINTS_DEFAULT = 600;
const MAX_POINTS_CAP = 2000;
const SCATTER_POINTS_CAP = 5000;
const COVERAGE_CELLS_CAP = 20_000;
const EVENTS_CAP = 500;
const BINS_DEFAULT = 20;
const BINS_CAP = 60;

export const AGGREGATES = ['avg', 'min', 'max', 'sum', 'count', 'median'];

const NUMERIC_TYPES = new Set([
  'smallint', 'integer', 'bigint', 'numeric', 'real', 'double precision', 'decimal',
]);
const TEXT_TYPES = new Set(['text', 'character varying', 'character', 'varchar', 'char', 'uuid']);
const TIME_TYPES = new Set([
  'timestamp without time zone', 'timestamp with time zone', 'date',
]);

function fail(status, error) {
  return { ok: false, status, error };
}

function pickExisting(columns, candidates) {
  for (const name of candidates) {
    if (name && columns.has(name)) return name;
  }
  return null;
}

function valueNamesOf(signal) {
  return (signal.values || [])
    .map((entry) => (typeof entry === 'string' ? entry : (entry?.source || entry?.name)))
    .filter(Boolean)
    .map(String);
}

function timeExpr(time, columns, alias) {
  const a = sqlIdent(alias);
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
    'ts', 'timestamp', 'time', 'date',
  ]);
  return col ? `(${a}.${sqlIdent(col)}::timestamp)` : null;
}

function round(value, digits = 3) {
  if (value == null || !Number.isFinite(Number(value))) return null;
  return Number(Number(value).toFixed(digits));
}

// Naive timestamps leave the database as ISO text (UTC), never as JS Dates: node-pg would read a
// timestamp-without-zone in the server's timezone and shift days.
const iso = (expr) => `to_char(${expr}, 'YYYY-MM-DD"T"HH24:MI:SS"Z"')`;

function isoBucket(value) {
  if (!value) return null;
  const d = value instanceof Date ? value : new Date(value);
  return Number.isNaN(d.getTime()) ? null : d.toISOString();
}

function parseIsoDate(value) {
  const s = String(value || '').trim();
  return /^\d{4}-\d{2}-\d{2}/.test(s) ? s.slice(0, 10) : null;
}

function normalizeIds(raw, cap) {
  if (!Array.isArray(raw)) return [];
  const seen = new Set();
  const out = [];
  for (const value of raw) {
    const id = String(value ?? '').trim();
    if (!id || seen.has(id)) continue;
    seen.add(id);
    out.push(id);
    if (out.length >= cap) break;
  }
  return out;
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

async function columnTypes(schema, table) {
  const res = await query(
    `SELECT column_name, data_type
     FROM information_schema.columns
     WHERE table_schema = $1 AND table_name = $2
     ORDER BY ordinal_position`,
    [schema, table]
  );
  return new Map(res.rows.map((r) => [String(r.column_name), String(r.data_type)]));
}

function classify(dataType) {
  if (NUMERIC_TYPES.has(dataType)) return 'numeric';
  if (dataType === 'boolean') return 'boolean';
  if (TEXT_TYPES.has(dataType)) return 'categorical';
  if (TIME_TYPES.has(dataType)) return 'time';
  return 'other';
}

/** Bind one mapped signal to safe SQL fragments. Returns null when it cannot be charted. */
async function bindSignal(signal) {
  if (!signal.physical_schema || !signal.physical_table) return null;
  const types = await columnTypes(signal.physical_schema, signal.physical_table);
  const columns = new Set(types.keys());
  const subjectCol = pickExisting(columns, [signal.subject, 'id', 'subject_id', 'user_id']);
  const ts = timeExpr(signal.time, columns, 't');
  const mapped = valueNamesOf(signal).filter((c) => columns.has(c));
  const skip = new Set([subjectCol, ...(typeof signal.time === 'string' ? [signal.time] : []),
    ...(Array.isArray(signal.time?.combine) ? signal.time.combine : []), 'ts', 'timestamp', 'time', 'date']);
  const candidates = mapped.length
    ? mapped
    : [...columns].filter((c) => !skip.has(c));
  const valueColumns = candidates
    .map((name) => ({ name, type: types.get(name), role: classify(types.get(name)) }))
    .filter((c) => c.role !== 'other' && c.role !== 'time');
  return {
    name: signal.name,
    signal_type: signal.signal_type,
    rel: `${sqlIdent(signal.physical_schema)}.${sqlIdent(signal.physical_table)}`,
    subjectCol: subjectCol ? sqlIdent(subjectCol) : null,
    tsExpr: ts,
    hasTime: Boolean(ts),
    isEnrollment: signal.signal_type === 'subjects' || signal.name === 'subjects',
    interval_minutes: signal.interval_minutes ?? null,
    valueColumns,
  };
}

async function bindAll(datasetId) {
  const listed = await listDatasetSignals(datasetId);
  const bound = [];
  for (const signal of listed.signals || []) {
    const b = await bindSignal(signal);
    if (b) bound.push(b);
  }
  return bound;
}

function pickBound(bound, name) {
  const wanted = String(name || '').trim();
  if (!wanted) return null;
  return bound.find((b) => b.name === wanted) || null;
}

function pickValue(b, column, role) {
  const wanted = String(column || '').trim();
  const col = wanted
    ? b.valueColumns.find((c) => c.name === wanted)
    : b.valueColumns.find((c) => !role || c.role === role);
  if (!col) return null;
  if (role && col.role !== role) return null;
  return { ...col, sql: sqlIdent(col.name) };
}

/** Which chart suits this column, from its type and the signal's shape (ADR-029 §2). */
function defaultKind(b, col) {
  if (col.role === 'categorical' || col.role === 'boolean') return 'events';
  if (b.hasTime) return 'line';
  return 'bar';
}

/** Catalog of what can be charted for a dataset, plus its subjects and time span. */
export async function describeViz(datasetId) {
  const bound = await bindAll(datasetId);
  const subjects = new Set();
  let start = null;
  let end = null;
  const signals = [];
  for (const b of bound) {
    if (!b.subjectCol && !b.hasTime) continue;
    if (b.subjectCol && subjects.size < SUBJECT_LIST_CAP) {
      const res = await timed((c) => c.query(
        `SELECT DISTINCT t.${b.subjectCol}::text AS id FROM ${b.rel} t
         WHERE t.${b.subjectCol} IS NOT NULL ORDER BY 1 LIMIT ${SUBJECT_LIST_CAP}`));
      for (const row of res.rows) subjects.add(String(row.id));
    }
    let span = null;
    if (b.hasTime) {
      const res = await timed((c) => c.query(
        `SELECT ${iso(`MIN(${b.tsExpr})`)} AS lo, ${iso(`MAX(${b.tsExpr})`)} AS hi, COUNT(*)::bigint AS n FROM ${b.rel} t`));
      const row = res.rows[0] || {};
      span = { start: isoBucket(row.lo), end: isoBucket(row.hi), rows: Number(row.n) || 0 };
      if (span.start && (!start || span.start < start)) start = span.start;
      if (span.end && (!end || span.end > end)) end = span.end;
    }
    signals.push({
      name: b.name,
      signal_type: b.signal_type,
      is_enrollment: b.isEnrollment,
      has_time: b.hasTime,
      interval_minutes: b.interval_minutes,
      span,
      values: b.valueColumns.map((c) => ({
        name: c.name,
        type: c.type,
        role: c.role,
        default_chart: defaultKind(b, c),
      })),
    });
  }
  return {
    ok: true,
    signals,
    subjects: [...subjects].slice(0, SUBJECT_LIST_CAP),
    start,
    end,
    aggregates: AGGREGATES,
    chart_kinds: CHART_KINDS,
  };
}

function aggSql(agg, valueExpr) {
  switch (agg) {
    case 'min': return `MIN(${valueExpr})`;
    case 'max': return `MAX(${valueExpr})`;
    case 'sum': return `SUM(${valueExpr})`;
    case 'count': return `COUNT(${valueExpr})`;
    case 'median': return `percentile_cont(0.5) WITHIN GROUP (ORDER BY ${valueExpr})`;
    default: return `AVG(${valueExpr})`;
  }
}

function pickBucketSeconds(requested, startIso, endIso, maxPoints, intervalMinutes) {
  const span = Math.max(60, (Date.parse(endIso) - Date.parse(startIso)) / 1000);
  const floor = Math.max(1, Math.ceil(span / maxPoints));
  const named = { '1m': 60, '5m': 300, '15m': 900, '1h': 3600, '6h': 21600, '1d': 86400 };
  let seconds = Number(requested);
  if (named[requested]) seconds = named[requested];
  if (!Number.isFinite(seconds) || seconds <= 0) {
    // auto: the smallest tidy bucket that stays under max_points
    const tidy = [intervalMinutes ? intervalMinutes * 60 : 60, 60, 300, 900, 3600, 21600, 86400]
      .filter(Boolean).sort((a, b) => a - b);
    seconds = tidy.find((s) => s >= floor) || 86400;
  }
  return Math.max(seconds, floor);
}

async function resolveWindow(b, body) {
  let start = parseIsoDate(body.start);
  let end = parseIsoDate(body.end);
  if (!b.hasTime) return { start: null, end: null };
  if (!start || !end) {
    const res = await timed((c) => c.query(
      `SELECT ${iso(`MIN(${b.tsExpr})`)} AS lo, ${iso(`MAX(${b.tsExpr})`)} AS hi FROM ${b.rel} t`));
    const row = res.rows[0] || {};
    if (!row.lo) return { start: null, end: null, empty: true };
    start = start || isoBucket(row.lo).slice(0, 10);
    end = end || isoBucket(row.hi).slice(0, 10);
  }
  if (start > end) [start, end] = [end, start];
  return { start, end };
}

function windowSql(b, params, start, end) {
  if (!b.hasTime || !start || !end) return 'TRUE';
  params.push(start, end);
  return `${b.tsExpr} IS NOT NULL AND ${b.tsExpr} >= $${params.length - 1}::date
          AND ${b.tsExpr} < ($${params.length}::date + INTERVAL '1 day')`;
}

function subjectSql(b, params, ids) {
  if (!ids.length || !b.subjectCol) return 'TRUE';
  params.push(ids);
  return `t.${b.subjectCol}::text = ANY($${params.length}::text[])`;
}

async function pickSubjects(b, ids, cap) {
  if (ids.length) return ids.slice(0, cap);
  if (!b.subjectCol) return [];
  const res = await timed((c) => c.query(
    `SELECT DISTINCT t.${b.subjectCol}::text AS id FROM ${b.rel} t
     WHERE t.${b.subjectCol} IS NOT NULL ORDER BY 1 LIMIT ${cap}`));
  return res.rows.map((r) => String(r.id));
}

async function kindSeries(b, body) {
  const col = pickValue(b, body.value, 'numeric');
  if (!col) return fail(400, 'Choose a numeric value column for a line chart');
  if (!b.hasTime) return fail(400, 'This signal has no time column');
  const win = await resolveWindow(b, body);
  if (win.empty) return { ok: true, kind: 'series', series: [], note: 'No data' };
  const agg = AGGREGATES.includes(body.agg) ? body.agg : 'avg';
  const maxPoints = Math.min(Math.max(Number(body.max_points) || MAX_POINTS_DEFAULT, 20), MAX_POINTS_CAP);
  const seconds = pickBucketSeconds(body.bucket, `${win.start}T00:00:00Z`, `${win.end}T23:59:59Z`, maxPoints, b.interval_minutes);
  const groupBySubject = body.group_by !== 'none';
  let ids = normalizeIds(body.subjects, 200);
  if (groupBySubject) ids = await pickSubjects(b, ids, SERIES_SUBJECT_CAP);

  const params = [seconds];
  const where = [
    subjectSql(b, params, ids),
    windowSql(b, params, win.start, win.end),
    `t.${col.sql} IS NOT NULL`,
  ].join(' AND ');
  const bucket = `to_timestamp(FLOOR(EXTRACT(EPOCH FROM ${b.tsExpr}) / $1) * $1)`;
  const valueExpr = `t.${col.sql}::double precision`;
  const subjectSelect = groupBySubject && b.subjectCol ? `t.${b.subjectCol}::text AS subject_id,` : '';
  const subjectGroup = groupBySubject && b.subjectCol ? 'subject_id, ' : '';
  const res = await timed((c) => c.query(
    `SELECT ${subjectSelect} ${bucket} AS bucket, ${aggSql(agg, valueExpr)}::double precision AS v
     FROM ${b.rel} t
     WHERE ${where}
     GROUP BY ${subjectGroup}bucket
     ORDER BY ${subjectGroup}bucket`,
    params));

  const byKey = new Map();
  for (const row of res.rows) {
    const key = groupBySubject && b.subjectCol ? String(row.subject_id) : 'All subjects';
    const t = isoBucket(row.bucket);
    if (!t) continue;
    if (!byKey.has(key)) byKey.set(key, []);
    byKey.get(key).push([t, round(row.v)]);
  }
  return {
    ok: true,
    kind: 'series',
    signal: b.name,
    value: col.name,
    agg,
    bucket_seconds: seconds,
    start: win.start,
    end: win.end,
    max_points: maxPoints,
    series: [...byKey.entries()].map(([key, points]) => ({ key, points })),
  };
}

async function kindDistribution(b, body) {
  const col = pickValue(b, body.value, 'numeric');
  if (!col) return fail(400, 'Choose a numeric value column for a distribution');
  const win = await resolveWindow(b, body);
  const bins = Math.min(Math.max(Number(body.bins) || BINS_DEFAULT, 4), BINS_CAP);
  const ids = normalizeIds(body.subjects, 200);
  const params = [];
  const where = [
    subjectSql(b, params, ids),
    windowSql(b, params, win.start, win.end),
    `t.${col.sql} IS NOT NULL`,
  ].join(' AND ');
  const valueExpr = `t.${col.sql}::double precision`;

  const range = await timed((c) => c.query(
    `SELECT MIN(${valueExpr}) AS lo, MAX(${valueExpr}) AS hi, COUNT(*)::bigint AS n
     FROM ${b.rel} t WHERE ${where}`, params));
  const lo = Number(range.rows[0]?.lo);
  const hi = Number(range.rows[0]?.hi);
  const n = Number(range.rows[0]?.n) || 0;
  if (!n || !Number.isFinite(lo)) return { ok: true, kind: 'distribution', signal: b.name, value: col.name, n: 0, histogram: [], box: [] };
  const width = hi > lo ? (hi - lo) / bins : 1;

  const hist = await timed((c) => c.query(
    `SELECT LEAST(GREATEST(FLOOR((${valueExpr} - $${params.length + 1}::double precision) / $${params.length + 2}::double precision), 0), $${params.length + 3}::int - 1)::int AS bin,
            COUNT(*)::bigint AS n
     FROM ${b.rel} t WHERE ${where}
     GROUP BY bin ORDER BY bin`, [...params, lo, width, bins]));
  const counts = new Map(hist.rows.map((r) => [Number(r.bin), Number(r.n)]));
  const histogram = Array.from({ length: bins }, (_, i) => ({
    x0: round(lo + i * width), x1: round(lo + (i + 1) * width), count: counts.get(i) || 0,
  }));

  let box = [];
  if (b.subjectCol) {
    const subjectIds = await pickSubjects(b, ids, SERIES_SUBJECT_CAP);
    const bp = [];
    const bwhere = [subjectSql(b, bp, subjectIds), windowSql(b, bp, win.start, win.end), `t.${col.sql} IS NOT NULL`].join(' AND ');
    const res = await timed((c) => c.query(
      `SELECT t.${b.subjectCol}::text AS subject_id,
              MIN(${valueExpr}) AS min,
              percentile_cont(0.25) WITHIN GROUP (ORDER BY ${valueExpr}) AS q1,
              percentile_cont(0.5) WITHIN GROUP (ORDER BY ${valueExpr}) AS median,
              percentile_cont(0.75) WITHIN GROUP (ORDER BY ${valueExpr}) AS q3,
              MAX(${valueExpr}) AS max, COUNT(*)::bigint AS n
       FROM ${b.rel} t WHERE ${bwhere}
       GROUP BY subject_id ORDER BY subject_id`, bp));
    box = res.rows.map((r) => ({
      key: String(r.subject_id), min: round(r.min), q1: round(r.q1), median: round(r.median),
      q3: round(r.q3), max: round(r.max), n: Number(r.n),
    }));
  }
  return { ok: true, kind: 'distribution', signal: b.name, value: col.name, n, min: round(lo), max: round(hi), histogram, box };
}

async function kindCoverage(b, body) {
  if (!b.hasTime || !b.subjectCol) return fail(400, 'Coverage needs a subject and a time column');
  const win = await resolveWindow(b, body);
  if (win.empty) return { ok: true, kind: 'coverage', cells: [], subjects: [], days: [] };
  const hourly = body.resolution === 'hour';
  const ids = await pickSubjects(b, normalizeIds(body.subjects, COVERAGE_SUBJECT_CAP), COVERAGE_SUBJECT_CAP);
  const params = [];
  const where = [subjectSql(b, params, ids), windowSql(b, params, win.start, win.end)].join(' AND ');
  const unit = hourly ? 'hour' : 'day';
  const res = await timed((c) => c.query(
    `SELECT t.${b.subjectCol}::text AS subject_id, ${iso(`date_trunc('${unit}', ${b.tsExpr})`)} AS bucket, COUNT(*)::bigint AS n
     FROM ${b.rel} t WHERE ${where}
     GROUP BY subject_id, bucket ORDER BY subject_id, bucket
     LIMIT ${COVERAGE_CELLS_CAP}`, params));
  const subjects = [...new Set(res.rows.map((r) => String(r.subject_id)))];
  const days = [...new Set(res.rows.map((r) => isoBucket(r.bucket)))].sort();
  return {
    ok: true, kind: 'coverage', signal: b.name, resolution: unit,
    subjects, days,
    cells: res.rows.map((r) => [isoBucket(r.bucket), String(r.subject_id), Number(r.n)]),
    truncated: res.rows.length >= COVERAGE_CELLS_CAP,
  };
}

async function kindScatter(bound, body) {
  const bx = pickBound(bound, body.x?.signal);
  const by = pickBound(bound, body.y?.signal);
  if (!bx || !by) return fail(400, 'Choose two signals for a scatter plot');
  const cx = pickValue(bx, body.x?.value, 'numeric');
  const cy = pickValue(by, body.y?.value, 'numeric');
  if (!cx || !cy) return fail(400, 'Both axes need a numeric value column');
  if (!bx.hasTime || !by.hasTime || !bx.subjectCol || !by.subjectCol) {
    return fail(400, 'Both signals need a subject and a time column');
  }
  const win = await resolveWindow(bx, body);
  if (win.empty) return { ok: true, kind: 'scatter', points: [] };
  const maxPoints = Math.min(Math.max(Number(body.max_points) || 1500, 50), SCATTER_POINTS_CAP);
  const seconds = pickBucketSeconds(body.bucket, `${win.start}T00:00:00Z`, `${win.end}T23:59:59Z`, maxPoints, Math.max(bx.interval_minutes || 0, by.interval_minutes || 0) || null);
  const ids = normalizeIds(body.subjects, 200);
  const params = [seconds];
  const side = (b, c, alias) => {
    const w = [subjectSql(b, params, ids), windowSql(b, params, win.start, win.end), `t.${c.sql} IS NOT NULL`].join(' AND ');
    return `(SELECT t.${b.subjectCol}::text AS s,
                    to_timestamp(FLOOR(EXTRACT(EPOCH FROM ${b.tsExpr}) / $1) * $1) AS b,
                    AVG(t.${c.sql}::double precision) AS v
             FROM ${b.rel} t WHERE ${w} GROUP BY s, b) ${alias}`;
  };
  const sx = side(bx, cx, 'x');
  const sy = side(by, cy, 'y');
  const res = await timed((c) => c.query(
    `SELECT x.s AS subject_id, x.b AS bucket, x.v AS xv, y.v AS yv
     FROM ${sx} JOIN ${sy} ON x.s = y.s AND x.b = y.b
     ORDER BY x.b LIMIT ${SCATTER_POINTS_CAP}`, params));
  return {
    ok: true, kind: 'scatter',
    x: { signal: bx.name, value: cx.name }, y: { signal: by.name, value: cy.name },
    bucket_seconds: seconds,
    points: res.rows.map((r) => [round(r.xv), round(r.yv), String(r.subject_id)]),
    truncated: res.rows.length >= SCATTER_POINTS_CAP,
  };
}

async function kindEvents(b, body) {
  if (!b.hasTime) return fail(400, 'Events need a time column');
  const col = pickValue(b, body.value);
  if (!col) return fail(400, 'Choose a value column to label the events');
  const win = await resolveWindow(b, body);
  if (win.empty) return { ok: true, kind: 'events', events: [], subjects: [] };
  const ids = normalizeIds(body.subjects, 200);
  const params = [];
  const where = [subjectSql(b, params, ids), windowSql(b, params, win.start, win.end), `t.${col.sql} IS NOT NULL`].join(' AND ');
  const limit = Math.min(Math.max(Number(body.limit) || EVENTS_CAP, 10), EVENTS_CAP);
  const subjectSelect = b.subjectCol ? `t.${b.subjectCol}::text` : `'All'`;
  const res = await timed((c) => c.query(
    `SELECT ${subjectSelect} AS subject_id, ${iso(b.tsExpr)} AS at, t.${col.sql}::text AS label
     FROM ${b.rel} t WHERE ${where} ORDER BY ${b.tsExpr} LIMIT ${limit}`, params));
  return {
    ok: true, kind: 'events', signal: b.name, value: col.name,
    events: res.rows.map((r) => [isoBucket(r.at), String(r.subject_id), String(r.label)]),
    subjects: [...new Set(res.rows.map((r) => String(r.subject_id)))],
    truncated: res.rows.length >= limit,
  };
}

/** Entry point for POST /datasets/:id/analytics/series (ADR-029 §3). */
export async function runViz(datasetId, body = {}) {
  const kind = String(body.kind || 'series');
  const bound = await bindAll(datasetId);
  if (kind === 'scatter') return kindScatter(bound, body);
  const b = pickBound(bound, body.signal);
  if (!b) return fail(404, 'Unknown signal for this dataset');
  if (!b.subjectCol && !b.hasTime) return fail(400, 'This signal has no subject or time mapping');
  if (kind === 'series') return kindSeries(b, body);
  if (kind === 'distribution') return kindDistribution(b, body);
  if (kind === 'coverage') return kindCoverage(b, body);
  if (kind === 'events') return kindEvents(b, body);
  return fail(400, `Unknown kind "${kind}"`);
}
