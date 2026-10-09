/**
 * Filter-based cohort comparison (ADR-021 v1 / ICDE demo offline analytics).
 *
 * Aggregates mapped heart_rate, calories, and mets/mets_computed tables.
 * Does not own a MET formula — derived METs only appear when a mapped
 * mets or mets_computed signal already exists.
 */

import { query } from '../db.js';
import { listDatasetSignals, sqlIdent } from './sql-contribution-engine.js';

const SUBJECT_LIST_CAP = 200;
const SELECT_CAP = 12;
const SERIES_POINT_TARGET = 360;
const WINDOW_CAP = 10;
const COMPARABLE_ORDER = ['heart_rate', 'calories', 'mets_computed', 'mets'];
const SIGNAL_LABELS = {
  heart_rate: 'Heart rate',
  calories: 'Calories',
  mets: 'METs',
  mets_computed: 'METs',
};
const NOTE = 'Filter-based groups; not a matched cohort.';

function qualify(schema, table) {
  return `${sqlIdent(schema)}.${sqlIdent(table)}`;
}

function pickExisting(columns, candidates) {
  for (const name of candidates) {
    if (name && columns.has(name)) return name;
  }
  return null;
}

function mappedSource(values, roleName, fallbacks = []) {
  if (!Array.isArray(values)) return fallbacks[0] || null;
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
  return fallbacks[0] || null;
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
    'ts',
    'timestamp',
    'time',
    'date',
  ]);
  if (!col) return null;
  return `(${a}.${sqlIdent(col)}::timestamp)`;
}

function roundMetric(value, digits = 2) {
  if (value == null || !Number.isFinite(Number(value))) return null;
  return Number(Number(value).toFixed(digits));
}

function fail(status, error) {
  return { ok: false, status, error };
}

function parseIsoDate(value) {
  const s = String(value || '').trim();
  if (!/^\d{4}-\d{2}-\d{2}/.test(s)) return null;
  return s.slice(0, 10);
}

function parseHm(value) {
  const match = String(value || '').trim().match(/^(\d{1,2}):(\d{2})(?::\d{2})?$/);
  if (!match) return null;
  const hour = Number(match[1]);
  const minute = Number(match[2]);
  if (hour > 23 || minute > 59) return null;
  return `${String(hour).padStart(2, '0')}:${String(minute).padStart(2, '0')}:00`;
}

function parseTimeWindows(raw) {
  if (!Array.isArray(raw)) return [];
  const out = [];
  for (const row of raw.slice(0, WINDOW_CAP)) {
    const start = parseHm(row?.start);
    const end = parseHm(row?.end);
    if (!start || !end) continue;
    out.push({
      label: String(row?.label || `Window ${out.length + 1}`).slice(0, 80),
      start,
      end,
    });
  }
  return out;
}

function normalizeIds(raw, cap = SELECT_CAP) {
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

function bindSignal(signal) {
  const columns = new Set(signal.columns || []);
  const type = String(signal.signal_type || '').toLowerCase();
  const valueNames = (signal.values || []).map((entry) => (
    typeof entry === 'string' ? entry : (entry?.source || entry?.name)
  )).filter(Boolean);
  const subjectCol = pickExisting(columns, [signal.subject, 'id', 'subject_id', 'user_id']);
  const valueCol = pickExisting(columns, [
    mappedSource(signal.values, type, ['value', type, ...valueNames]),
    ...valueNames,
    'value',
    type,
  ]);
  const ts = timeExpr(signal.time, columns, 't');
  if (!subjectCol || !valueCol || !ts) return null;
  if (!signal.physical_schema || !signal.physical_table) return null;
  return {
    type,
    label: SIGNAL_LABELS[type] || signal.name || type,
    name: signal.name,
    rel: qualify(signal.physical_schema, signal.physical_table),
    subjectCol: sqlIdent(subjectCol),
    valueCol: sqlIdent(valueCol),
    tsExpr: ts,
  };
}

function pickComparable(signals) {
  const byType = new Map();
  for (const signal of signals || []) {
    const type = String(signal.signal_type || '').toLowerCase();
    if (!COMPARABLE_ORDER.includes(type)) continue;
    if (byType.has(type)) continue;
    const bound = bindSignal(signal);
    if (bound) byType.set(type, bound);
  }
  const out = [];
  for (const type of ['heart_rate', 'calories']) {
    if (byType.has(type)) out.push(byType.get(type));
  }
  if (byType.has('mets_computed')) out.push(byType.get('mets_computed'));
  else if (byType.has('mets')) out.push(byType.get('mets'));
  return out;
}

function windowPredicate(tsExpr, windows, params) {
  if (!windows.length) return 'TRUE';
  const parts = windows.map((window) => {
    params.push(window.start, window.end);
    const startPh = `$${params.length - 1}::time`;
    const endPh = `$${params.length}::time`;
    return `((${startPh} <= ${endPh} AND (${tsExpr})::time >= ${startPh} AND (${tsExpr})::time < ${endPh})
      OR (${startPh} > ${endPh} AND ((${tsExpr})::time >= ${startPh} OR (${tsExpr})::time < ${endPh})))`;
  });
  return `(${parts.join(' OR ')})`;
}

function bucketSeconds(startDate, endDate) {
  const start = Date.parse(`${startDate}T00:00:00Z`);
  const end = Date.parse(`${endDate}T23:59:59Z`);
  const spanSec = Math.max(60, (end - start) / 1000);
  const raw = spanSec / SERIES_POINT_TARGET;
  if (raw <= 300) return 300;
  if (raw <= 900) return 900;
  if (raw <= 3600) return 3600;
  if (raw <= 3 * 3600) return 3 * 3600;
  return 86400;
}

function statsShape(row) {
  const n = Number(row?.n) || 0;
  const avg = roundMetric(row?.avg);
  const std = roundMetric(row?.std);
  return {
    n,
    min: roundMetric(row?.min),
    max: roundMetric(row?.max),
    avg,
    std,
  };
}

function safeBand(control) {
  if (control.avg == null || control.std == null) return null;
  return {
    min: roundMetric(control.avg - 2 * control.std),
    max: roundMetric(control.avg + 2 * control.std),
    source: 'control',
  };
}

async function queryStats(bound, ids, startDate, endDate, windows) {
  if (!ids.length) {
    return statsShape({ n: 0, min: null, max: null, avg: null, std: null });
  }
  const params = [ids, startDate, endDate];
  const winSql = windowPredicate(bound.tsExpr, windows, params);
  const res = await query(
    `SELECT COUNT(*)::bigint AS n,
            MIN(q.val)::double precision AS min,
            MAX(q.val)::double precision AS max,
            AVG(q.val)::double precision AS avg,
            STDDEV_SAMP(q.val)::double precision AS std
     FROM (
       SELECT t.${bound.valueCol}::double precision AS val
       FROM ${bound.rel} t
       WHERE t.${bound.subjectCol}::text = ANY($1::text[])
         AND ${bound.tsExpr} IS NOT NULL
         AND ${bound.tsExpr} >= $2::date
         AND ${bound.tsExpr} < ($3::date + INTERVAL '1 day')
         AND t.${bound.valueCol} IS NOT NULL
         AND ${winSql}
     ) q`,
    params
  );
  return statsShape(res.rows[0] || {});
}

function isoBucket(value) {
  if (!value) return null;
  const d = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(d.getTime())) return null;
  return d.toISOString();
}

async function querySeries(bound, selectedIds, controlIds, startDate, endDate, windows, seconds) {
  const empty = { bucket_seconds: seconds, control: [], subjects: {} };
  if (!selectedIds.length && !controlIds.length) return empty;

  async function grouped(ids, withSubject) {
    if (!ids.length) return [];
    const params = [ids, startDate, endDate, seconds];
    const winSql = windowPredicate(bound.tsExpr, windows, params);
    const subjectSelect = withSubject
      ? `t.${bound.subjectCol}::text AS subject_id,`
      : '';
    const subjectGroup = withSubject ? 'subject_id, ' : '';
    const res = await query(
      `SELECT ${subjectSelect}
              to_timestamp(FLOOR(EXTRACT(EPOCH FROM ${bound.tsExpr}) / $4) * $4) AS bucket,
              AVG(t.${bound.valueCol}::double precision)::double precision AS avg
       FROM ${bound.rel} t
       WHERE t.${bound.subjectCol}::text = ANY($1::text[])
         AND ${bound.tsExpr} IS NOT NULL
         AND ${bound.tsExpr} >= $2::date
         AND ${bound.tsExpr} < ($3::date + INTERVAL '1 day')
         AND t.${bound.valueCol} IS NOT NULL
         AND ${winSql}
       GROUP BY ${subjectGroup}bucket
       ORDER BY bucket`,
      params
    );
    return res.rows;
  }

  const [subjectRows, controlRows] = await Promise.all([
    grouped(selectedIds, true),
    grouped(controlIds, false),
  ]);

  const subjects = {};
  for (const row of subjectRows) {
    const id = String(row.subject_id || '');
    if (!id) continue;
    if (!subjects[id]) subjects[id] = [];
    const t = isoBucket(row.bucket);
    if (!t) continue;
    subjects[id].push({ t, avg: roundMetric(row.avg) });
  }

  return {
    bucket_seconds: seconds,
    control: controlRows
      .map((row) => ({ t: isoBucket(row.bucket), avg: roundMetric(row.avg) }))
      .filter((row) => row.t),
    subjects,
  };
}

async function loadBoundSignals(datasetId) {
  const listed = await listDatasetSignals(datasetId);
  return {
    listed,
    comparable: pickComparable(listed.signals || []),
  };
}

async function listSubjectsAndRange(comparable) {
  const subjects = new Set();
  let start = null;
  let end = null;
  for (const bound of comparable) {
    const idRes = await query(
      `SELECT DISTINCT t.${bound.subjectCol}::text AS id
       FROM ${bound.rel} t
       WHERE t.${bound.subjectCol} IS NOT NULL
       ORDER BY 1
       LIMIT ${SUBJECT_LIST_CAP}`,
      []
    );
    for (const row of idRes.rows) {
      if (row.id) subjects.add(String(row.id));
    }
    const rangeRes = await query(
      // Dates as text: a timestamp without a zone must not be shifted by the server's timezone.
      `SELECT MIN(${bound.tsExpr})::date::text AS start_date, MAX(${bound.tsExpr})::date::text AS end_date
       FROM ${bound.rel} t
       WHERE ${bound.tsExpr} IS NOT NULL`,
      []
    );
    const row = rangeRes.rows[0] || {};
    if (row.start_date && (!start || row.start_date < start)) start = row.start_date;
    if (row.end_date && (!end || row.end_date > end)) end = row.end_date;
  }
  return {
    subjects: [...subjects].sort((a, b) => a.localeCompare(b)).slice(0, SUBJECT_LIST_CAP),
    date_range: start && end ? { start, end } : null,
  };
}

export async function getCohortAnalyticsSetup(datasetId) {
  const { comparable } = await loadBoundSignals(datasetId);
  if (!comparable.length) {
    return {
      ok: true,
      dataset_id: datasetId,
      note: NOTE,
      subjects: [],
      date_range: null,
      signals: [],
      reason: 'No mapped heart_rate, calories, or METs signals to compare',
    };
  }
  const { subjects, date_range } = await listSubjectsAndRange(comparable);
  return {
    ok: true,
    dataset_id: datasetId,
    note: NOTE,
    subjects,
    date_range,
    signals: comparable.map((s) => ({ type: s.type, label: s.label, name: s.name })),
    select_cap: SELECT_CAP,
  };
}

export async function runCohortComparison(datasetId, body = {}) {
  const { comparable } = await loadBoundSignals(datasetId);
  if (!comparable.length) {
    return fail(400, 'No mapped heart_rate, calories, or METs signals to compare');
  }
  const roster = await listSubjectsAndRange(comparable);
  const allIds = roster.subjects;
  const selectedIds = normalizeIds(body.subject_ids, SELECT_CAP);
  if (!selectedIds.length) {
    return fail(400, 'Select at least one subject');
  }
  const unknown = selectedIds.filter((id) => !allIds.includes(id));
  if (unknown.length) {
    return fail(400, `Unknown subject id: ${unknown[0]}`);
  }

  const controlSpec = body.control && typeof body.control === 'object' ? body.control : {};
  const controlMode = String(controlSpec.mode || 'all').toLowerCase();
  let controlIds;
  if (controlMode === 'ids') {
    controlIds = normalizeIds(controlSpec.ids, SUBJECT_LIST_CAP).filter((id) => allIds.includes(id));
  } else if (controlMode === 'rest') {
    const selected = new Set(selectedIds);
    controlIds = allIds.filter((id) => !selected.has(id));
  } else {
    controlIds = [...allIds];
  }
  if (!controlIds.length) {
    return fail(400, 'Control group is empty');
  }

  const startDate = parseIsoDate(body.start) || roster.date_range?.start;
  const endDate = parseIsoDate(body.end) || roster.date_range?.end;
  if (!startDate || !endDate) {
    return fail(400, 'Dataset has no time range to query');
  }
  if (startDate > endDate) {
    return fail(400, 'Start date must be on or before end date');
  }

  const windows = parseTimeWindows(body.time_windows);
  const seconds = bucketSeconds(startDate, endDate);
  const signals = {};

  for (const bound of comparable) {
    const [selected, control, series] = await Promise.all([
      queryStats(bound, selectedIds, startDate, endDate, windows),
      queryStats(bound, controlIds, startDate, endDate, windows),
      querySeries(bound, selectedIds, controlIds, startDate, endDate, windows, seconds),
    ]);
    const deltaAvg =
      selected.avg != null && control.avg != null
        ? roundMetric(selected.avg - control.avg)
        : null;
    signals[bound.type] = {
      label: bound.label,
      selected,
      control,
      delta_avg: deltaAvg,
      safe_band: bound.type === 'heart_rate' ? safeBand(control) : null,
      series,
    };
  }

  const windowRows = [];
  for (const window of windows) {
    const perSignal = {};
    for (const bound of comparable) {
      const [selected, control] = await Promise.all([
        queryStats(bound, selectedIds, startDate, endDate, [window]),
        queryStats(bound, controlIds, startDate, endDate, [window]),
      ]);
      perSignal[bound.type] = {
        label: bound.label,
        selected,
        control,
        delta_avg:
          selected.avg != null && control.avg != null
            ? roundMetric(selected.avg - control.avg)
            : null,
      };
    }
    windowRows.push({
      label: window.label,
      start: window.start.slice(0, 5),
      end: window.end.slice(0, 5),
      signals: perSignal,
    });
  }

  return {
    ok: true,
    dataset_id: datasetId,
    note: NOTE,
    selected_ids: selectedIds,
    control_ids: controlIds,
    control_mode: controlMode === 'ids' || controlMode === 'rest' ? controlMode : 'all',
    start: startDate,
    end: endDate,
    signals,
    windows: windowRows,
  };
}
