/**
 * Dataset metadata summary — computed on read from w4h.datasets + w4h.tables.
 * Observed cadence uses bounded reads on physical signal tables.
 * Wear cohort tiles aggregate cached per-subject rows from datasets.summary
 * (CLI-aligned); does not re-scan heart-rate for wear.
 */

import { query } from '../db.js';
import { quoteIdent } from './sql-ident.js';

const OBSERVED_GAP_CAP = 5000;

function parseMapping(raw) {
  if (!raw) return {};
  if (typeof raw === 'object') return raw;
  try {
    return JSON.parse(raw);
  } catch {
    return {};
  }
}

function parseSummary(raw) {
  if (!raw) return {};
  if (typeof raw === 'object') return raw;
  try {
    return JSON.parse(raw);
  } catch {
    return {};
  }
}

function formatTimeMapping(time) {
  if (time == null) return null;
  if (typeof time === 'string') return { column: time };
  if (typeof time === 'object' && Array.isArray(time.combine)) {
    return { combine: time.combine.map(String) };
  }
  return time;
}

function formatValues(values) {
  if (!Array.isArray(values)) return [];
  return values.map((entry) => {
    if (typeof entry === 'string') {
      return {
        source: entry,
        name: entry,
        type: null,
        description: null,
        unit: null,
        ontology_code: null,
      };
    }
    if (!entry || typeof entry !== 'object') {
      return {
        source: null,
        name: null,
        type: null,
        description: null,
        unit: null,
        ontology_code: null,
      };
    }
    return {
      source: entry.source ?? null,
      name: entry.name ?? null,
      type: entry.type ?? null,
      description: entry.description ?? null,
      unit: entry.unit ?? null,
      ontology_code: entry.ontology_code ?? entry.ontology?.code ?? null,
    };
  });
}

function pickExisting(columns, candidates) {
  for (const name of candidates) {
    if (name && columns.has(name)) return name;
  }
  return null;
}

function qualify(schema, table) {
  return `${quoteIdent(schema)}.${quoteIdent(table)}`;
}

function timeExpr(mapping, columns, alias) {
  const a = quoteIdent(alias);
  const time = mapping?.time;
  if (time && typeof time === 'object' && Array.isArray(time.combine) && time.combine.length >= 2) {
    const dateCol = pickExisting(columns, time.combine);
    const timeCol = time.combine.map(String).find((c) => c !== dateCol && columns.has(c));
    if (dateCol && timeCol) {
      return `(${a}.${quoteIdent(dateCol)}::timestamp + ${a}.${quoteIdent(timeCol)}::time)`;
    }
    if (dateCol) return `(${a}.${quoteIdent(dateCol)}::timestamp)`;
  }
  const col = pickExisting(columns, [
    typeof time === 'string' ? time : null,
    'ts',
    'timestamp',
    'time',
    'date',
  ]);
  if (!col) return null;
  return `(${a}.${quoteIdent(col)}::timestamp)`;
}

function subjectExpr(mapping, columns, alias) {
  const a = quoteIdent(alias);
  const col = pickExisting(columns, [
    mapping?.subject,
    'id',
    'subject_id',
    'user_id',
  ]);
  if (!col) return null;
  return `${a}.${quoteIdent(col)}`;
}

function toDateIso(value) {
  if (value == null) return null;
  const s = String(value).trim();
  const match = s.match(/^(\d{4}-\d{2}-\d{2})/);
  return match ? match[1] : null;
}

function studyLengthDays(dateStart, dateEnd) {
  if (dateStart == null || dateEnd == null) return null;
  const start = new Date(dateStart);
  const end = new Date(dateEnd);
  if (Number.isNaN(start.getTime()) || Number.isNaN(end.getTime())) return null;
  const ms = end.getTime() - start.getTime();
  if (!Number.isFinite(ms)) return null;
  return Math.floor(ms / 86400000) + 1;
}

/** Merge per-table MIN/MAX ranges into one dataset collection span. */
export function mergeCollectionDateRanges(ranges) {
  let dateStart = null;
  let dateEnd = null;
  for (const range of ranges || []) {
    if (!range) continue;
    const start = toDateIso(range.date_start);
    const end = toDateIso(range.date_end);
    if (start && (!dateStart || start < dateStart)) dateStart = start;
    if (end && (!dateEnd || end > dateEnd)) dateEnd = end;
  }
  return dateStart || dateEnd ? { date_start: dateStart, date_end: dateEnd } : null;
}

/** Catalog dates win; fill gaps from computed range or cached summary. */
export function resolveScopeDates(dataset, collectionRange = null) {
  const cachedSummary = parseSummary(dataset?.summary);
  return {
    date_start:
      toDateIso(dataset?.date_start)
      ?? toDateIso(collectionRange?.date_start)
      ?? toDateIso(cachedSummary.date_start)
      ?? null,
    date_end:
      toDateIso(dataset?.date_end)
      ?? toDateIso(collectionRange?.date_end)
      ?? toDateIso(cachedSummary.date_end)
      ?? null,
  };
}

function median(numbers) {
  if (!numbers.length) return null;
  const sorted = [...numbers].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 0) {
    return (sorted[mid - 1] + sorted[mid]) / 2;
  }
  return sorted[mid];
}

function roundMetric(value, digits = 1) {
  if (value == null || !Number.isFinite(Number(value))) return null;
  return Number(Number(value).toFixed(digits));
}

/** Days with recorded data per subject id, used when `days_with_data` is absent. */
function countDailyRowsBySubject(daily) {
  const counts = new Map();
  for (const row of daily) {
    const id = String(row?.subject_id ?? '');
    if (!id) continue;
    counts.set(id, (counts.get(id) || 0) + 1);
  }
  return counts;
}

/** Null when neither source can tell days worn apart from unknown. */
function subjectDaysWorn(subject, dailyCounts) {
  const reported = Number(subject?.days_with_data);
  if (Number.isFinite(reported)) return reported;
  const counted = dailyCounts.get(String(subject?.subject_id ?? ''));
  return counted == null ? null : counted;
}

/**
 * Aggregate CLI-aligned per-subject wear rows into cohort display tiles.
 * Source: datasets.summary.wear_compliance (proposed cache envelope).
 * @param {object|null} wearSource
 * @returns {object|null}
 */
export function aggregateWearCohort(wearSource) {
  if (!wearSource || typeof wearSource !== 'object') return null;

  const subjects = Array.isArray(wearSource.subjects) ? wearSource.subjects : [];
  if (!subjects.length) return null;

  const subjectsTotal = subjects.length;
  const subjectsCompliant = subjects.filter((s) => Boolean(s?.is_overall_compliant)).length;
  const compliancePct = subjectsTotal > 0
    ? roundMetric((subjectsCompliant / subjectsTotal) * 100, 1)
    : null;

  const daily = Array.isArray(wearSource.daily) ? wearSource.daily : [];
  const dailyPcts = daily
    .map((d) => Number(d?.compliance_percentage))
    .filter((n) => Number.isFinite(n));
  // The platform producer stores per-subject rows plus a precomputed pooled median (no `daily`).
  const medianDailyWearPct = dailyPcts.length
    ? roundMetric(median(dailyPcts), 1)
    : roundMetric(wearSource.median_daily_wear_pct, 1);

  const dailyCounts = countDailyRowsBySubject(daily);
  const daysWorn = subjects
    .map((s) => subjectDaysWorn(s, dailyCounts))
    .filter((n) => Number.isFinite(n));
  const medianDaysWorn = daysWorn.length ? roundMetric(median(daysWorn), 1) : null;

  const protocol = wearSource.protocol && typeof wearSource.protocol === 'object'
    ? wearSource.protocol
    : {};
  const dailyThreshold = Number(protocol.daily_band_wear_percentage);
  const overallThreshold = Number(protocol.overall_band_wear_percentage);
  const dailyPct = Number.isFinite(dailyThreshold) ? dailyThreshold : 60;
  const overallPct = Number.isFinite(overallThreshold) ? overallThreshold : 60;

  return {
    subjects_compliant: subjectsCompliant,
    subjects_total: subjectsTotal,
    compliance_pct: compliancePct,
    median_daily_wear_pct: medianDailyWearPct,
    median_days_worn: medianDaysWorn,
    protocol: {
      daily_band_wear_percentage: dailyPct,
      overall_band_wear_percentage: overallPct,
      daily_band_wear_hours: Array.isArray(protocol.daily_band_wear_hours)
        ? protocol.daily_band_wear_hours
        : [['08:00', '20:00']],
    },
  };
}

/** Enrollment registry (e.g. fitbit subjects.csv) — not a measurement signal. */
function isEnrollmentTable(tableRow) {
  const mapping = parseMapping(tableRow.mapping_json);
  const slug = String(tableRow.name || '').toLowerCase();
  const signalType = String(mapping.signal_type || '').toLowerCase();
  return signalType === 'subjects' || slug === 'subjects';
}

function buildSignalSummary(tableRow) {
  const mapping = parseMapping(tableRow.mapping_json);
  return {
    slug: tableRow.name,
    name: tableRow.name,
    is_enrollment: isEnrollmentTable(tableRow),
    description: tableRow.description ?? null,
    source_name: tableRow.source_name ?? null,
    source_url: tableRow.source_url ?? null,
    signal_type: mapping.signal_type ?? tableRow.name ?? null,
    sampling_interval: mapping.sampling_interval ?? null,
    native_sensor_rate: mapping.native_sensor_rate ?? null,
    timezone: mapping.timezone ?? null,
    mapping: {
      subject: mapping.subject ?? null,
      time: formatTimeMapping(mapping.time),
      geo: mapping.geo ?? null,
      values: formatValues(mapping.values),
    },
    physical_schema: tableRow.physical_schema ?? null,
    physical_table: tableRow.physical_table ?? null,
  };
}

function uniqueVendors(measurementSignals, deviceType) {
  const fromSignals = [
    ...new Set(measurementSignals.map((s) => s.source_name).filter(Boolean)),
  ];
  if (fromSignals.length === 1) return fromSignals[0];
  if (fromSignals.length > 1) return fromSignals.join(', ');
  return deviceType ?? null;
}

function parseProvenanceCache(summary) {
  const cached = parseSummary(summary);
  const block =
    cached.provenance && typeof cached.provenance === 'object' && !Array.isArray(cached.provenance)
      ? cached.provenance
      : {};
  return {
    package_vendor: block.package_vendor ?? null,
    package_format: block.package_format ?? null,
    package_version: block.package_version ?? null,
    ingestion_source: block.ingestion_source ?? null,
  };
}

/**
 * Latest in-app import contribution per dataset (contribution_runs, then audit_log).
 * @param {string[]} datasetIds
 * @returns {Promise<Map<string, { slug: string, label: string }>>}
 */
export async function fetchIngestionSourcesForDatasets(datasetIds) {
  const ids = [...new Set((datasetIds || []).map(String).filter(Boolean))];
  const map = new Map();
  if (!ids.length) return map;

  try {
    const runsRes = await query(
      `SELECT DISTINCT ON (cr.dataset_id)
         cr.dataset_id,
         cr.contribution_slug,
         c.title AS contribution_title
       FROM w4h.contribution_runs cr
       LEFT JOIN w4h.contributions c ON c.slug = cr.contribution_slug
       WHERE cr.dataset_id = ANY($1::text[])
       ORDER BY cr.dataset_id,
         CASE cr.status
           WHEN 'succeeded' THEN 0
           WHEN 'running' THEN 1
           WHEN 'queued' THEN 2
           ELSE 3
         END,
         cr.finished_at DESC NULLS LAST,
         cr.started_at DESC`,
      [ids]
    );
    for (const row of runsRes.rows || []) {
      const id = String(row.dataset_id);
      map.set(id, {
        slug: row.contribution_slug,
        label: row.contribution_title || row.contribution_slug,
      });
    }
  } catch {
    // contribution_runs may be missing on older installs.
  }

  const missing = ids.filter((id) => !map.has(id));
  if (missing.length) {
    try {
      const auditRes = await query(
        `SELECT DISTINCT ON (resource_id)
           resource_id,
           details->>'contribution_slug' AS contribution_slug
         FROM w4h.audit_log
         WHERE resource_type = 'dataset'
           AND resource_id = ANY($1::text[])
           AND action IN ('dataset.import.update', 'dataset.import.full_reimport')
           AND details->>'contribution_slug' IS NOT NULL
         ORDER BY resource_id, timestamp DESC`,
        [missing]
      );
      const slugs = [...new Set((auditRes.rows || []).map((r) => r.contribution_slug).filter(Boolean))];
      let titlesBySlug = new Map();
      if (slugs.length) {
        const titleRes = await query(
          `SELECT slug, title FROM w4h.contributions WHERE slug = ANY($1::text[])`,
          [slugs]
        );
        titlesBySlug = new Map(
          (titleRes.rows || []).map((r) => [String(r.slug), r.title || r.slug])
        );
      }
      for (const row of auditRes.rows || []) {
        const id = String(row.resource_id);
        if (map.has(id)) continue;
        const slug = row.contribution_slug;
        map.set(id, {
          slug,
          label: titlesBySlug.get(slug) || slug,
        });
      }
    } catch {
      // audit_log may be missing on older installs.
    }
  }

  return map;
}

/**
 * Build normalized metadata_summary for one dataset + registry rows.
 * Pass observedCadenceBySignal (Map slug → observed block) when available.
 * @param {object} dataset
 * @param {object[]} tableRows
 * @param {Map<string, object|null>} [observedCadenceBySignal]
 * @param {{ slug: string, label: string }|null} [ingestionContext]
 * @param {{ date_start: string|null, date_end: string|null }|null} [collectionRange]
 * @param {number|null} [enrolledParticipantCount]
 */
export function buildMetadataSummary(
  dataset,
  tableRows = [],
  observedCadenceBySignal = null,
  ingestionContext = null,
  collectionRange = null,
  enrolledParticipantCount = null
) {
  const signals = [...(tableRows || [])]
    .sort((a, b) => String(a.name).localeCompare(String(b.name)))
    .map(buildSignalSummary);

  const measurementSignals = signals.filter((s) => !s.is_enrollment);

  const sources = measurementSignals
    .filter((s) => s.source_name || s.source_url)
    .map((s) => ({
      signal: s.slug,
      source_name: s.source_name,
      source_url: s.source_url,
    }));

  const cachedSummary = parseSummary(dataset.summary);
  const provenanceCache = parseProvenanceCache(dataset.summary);
  const wearCohort = aggregateWearCohort(cachedSummary.wear_compliance);

  const cadenceSignals = measurementSignals.map((s) => {
    const observed = observedCadenceBySignal?.get(String(s.slug)) ?? null;
    return {
      signal: s.slug,
      description: s.description ?? null,
      native_sensor_rate: s.native_sensor_rate ?? null,
      declared_interval: s.sampling_interval ?? null,
      observed_cadence: observed,
      timezone: s.timezone ?? null,
    };
  });

  const semanticSignals = measurementSignals.map((s) => ({
    signal: s.slug,
    description: s.description ?? null,
    variables: (s.mapping?.values || []).map((v) => ({
      name: v.name || v.source || null,
      unit: v.unit ?? null,
      type: v.type ?? null,
      description: v.description ?? null,
      ontology_code: v.ontology_code ?? null,
    })),
  }));

  const observations = cachedSummary.observations ?? null;
  const observationsApprox = cachedSummary.observations_approx !== false;
  const scopeDates = resolveScopeDates(dataset, collectionRange);

  return {
    scope: {
      table_count: dataset.table_count != null ? Number(dataset.table_count) : null,
      observations,
      observations_approx: observations != null ? observationsApprox : null,
      date_start: scopeDates.date_start,
      date_end: scopeDates.date_end,
      study_length_days: studyLengthDays(scopeDates.date_start, scopeDates.date_end),
    },
    subjects: {
      enrolled_participants:
        enrolledParticipantCount ?? dataset.subject_count ?? null,
      participant_age: null,
    },
    device: {
      vendor: uniqueVendors(measurementSignals, dataset.device_type ?? null),
      signal_types: measurementSignals
        .map((s) => s.signal_type || s.slug)
        .filter(Boolean),
      model: null,
      sensor_specifications: null,
      firmware_os: null,
      anatomical_placement: null,
    },
    cadence: {
      signals: cadenceSignals,
    },
    provenance: {
      origin: dataset.origin ?? null,
      description: dataset.description ?? null,
      ingestion_source:
        ingestionContext?.label ?? provenanceCache.ingestion_source ?? null,
      package_vendor: provenanceCache.package_vendor ?? null,
      package_format: provenanceCache.package_format ?? null,
      package_version: provenanceCache.package_version ?? null,
      api_payload_scheme: null,
      api_payload_version: null,
      data_state: null,
      sources,
    },
    compliance: {
      irb_reference: dataset.irb_reference ?? null,
      read_only: Boolean(dataset.read_only),
      wear_cohort: wearCohort,
    },
    semantic: {
      signals: semanticSignals,
    },
  };
}

async function computeTableDateRange(tableRow) {
  const schema = tableRow.physical_schema;
  const table = tableRow.physical_table;
  if (!schema || !table) return null;

  let columns;
  try {
    columns = await tableColumns(schema, table);
  } catch {
    return null;
  }
  if (!columns.size) return null;

  const mapping = parseMapping(tableRow.mapping_json);
  const alias = 't';
  const rel = qualify(schema, table);
  const a = quoteIdent(alias);

  try {
    if (isEnrollmentTable(tableRow)) {
      const startCol = pickExisting(columns, ['start_date', 'enrollment_start', 'collection_start']);
      const endCol = pickExisting(columns, ['end_date', 'enrollment_end', 'collection_end']);
      if (!startCol && !endCol) return null;
      const select = [];
      // ::text — node-pg turns a DATE into a local-midnight JS Date, which toDateIso cannot read.
      if (startCol) select.push(`MIN(${a}.${quoteIdent(startCol)}::date)::text AS date_start`);
      else select.push('NULL::text AS date_start');
      if (endCol) select.push(`MAX(${a}.${quoteIdent(endCol)}::date)::text AS date_end`);
      else select.push('NULL::text AS date_end');
      const res = await query(`SELECT ${select.join(', ')} FROM ${rel} ${alias}`);
      const row = res.rows?.[0];
      if (!row) return null;
      const dateStart = toDateIso(row.date_start);
      const dateEnd = toDateIso(row.date_end);
      return dateStart || dateEnd ? { date_start: dateStart, date_end: dateEnd } : null;
    }

    const ts = timeExpr(mapping, columns, alias);
    if (!ts) return null;
    const res = await query(
      `SELECT MIN((${ts})::date)::text AS date_start, MAX((${ts})::date)::text AS date_end
       FROM ${rel} ${alias}
       WHERE ${ts} IS NOT NULL`
    );
    const row = res.rows?.[0];
    if (!row) return null;
    const dateStart = toDateIso(row.date_start);
    const dateEnd = toDateIso(row.date_end);
    return dateStart || dateEnd ? { date_start: dateStart, date_end: dateEnd } : null;
  } catch (err) {
    console.warn(
      `computeTableDateRange failed for ${schema}.${table}:`,
      err?.message || err
    );
    return null;
  }
}

/**
 * Derive collection start/end from enrolled subjects and time-mapped signal tables.
 * @param {object[]} tableRows
 */
export async function computeCollectionDateRange(tableRows) {
  const physical = (tableRows || []).filter((t) => t.physical_schema && t.physical_table);
  if (!physical.length) return null;
  const ranges = await Promise.all(physical.map((tableRow) => computeTableDateRange(tableRow)));
  return mergeCollectionDateRanges(ranges);
}

async function countDistinctEnrollmentSubjects(tableRow) {
  if (!isEnrollmentTable(tableRow)) return null;
  const schema = tableRow.physical_schema;
  const table = tableRow.physical_table;
  if (!schema || !table) return null;

  let columns;
  try {
    columns = await tableColumns(schema, table);
  } catch {
    return null;
  }
  if (!columns.size) return null;

  const mapping = parseMapping(tableRow.mapping_json);
  const alias = 't';
  const rel = qualify(schema, table);
  const a = quoteIdent(alias);
  const idCol = pickExisting(columns, [
    mapping?.subject,
    'id',
    'subject_id',
    'user_id',
  ]);
  if (!idCol) return null;

  try {
    const res = await query(
      `SELECT COUNT(DISTINCT ${a}.${quoteIdent(idCol)})::int AS enrolled
       FROM ${rel} ${alias}
       WHERE ${a}.${quoteIdent(idCol)} IS NOT NULL`
    );
    const n = res.rows?.[0]?.enrolled;
    return n != null ? Number(n) : null;
  } catch (err) {
    console.warn(
      `countDistinctEnrollmentSubjects failed for ${schema}.${table}:`,
      err?.message || err
    );
    return null;
  }
}

/**
 * Count enrolled participants from the physical subjects / enrollment table.
 * @param {object[]} tableRows
 * @returns {Promise<number|null>}
 */
export async function computeEnrolledParticipants(tableRows) {
  const enrollment = (tableRows || []).filter(
    (t) => isEnrollmentTable(t) && t.physical_schema && t.physical_table
  );
  if (!enrollment.length) return null;
  return countDistinctEnrollmentSubjects(enrollment[0]);
}

async function tableColumns(schema, table) {
  const res = await query(
    `SELECT column_name
     FROM information_schema.columns
     WHERE table_schema = $1 AND table_name = $2`,
    [schema, table]
  );
  return new Set((res.rows || []).map((r) => String(r.column_name)));
}

/**
 * Bounded gap query: mode of consecutive timestamp gaps + max gap.
 * Caps at OBSERVED_GAP_CAP most recent gaps; sets approx when capped.
 */
export async function computeObservedCadence(tableRow) {
  const mapping = parseMapping(tableRow.mapping_json);
  const schema = tableRow.physical_schema;
  const table = tableRow.physical_table;
  if (!schema || !table || !mapping.time) return null;

  let columns;
  try {
    columns = await tableColumns(schema, table);
  } catch {
    return null;
  }
  if (!columns.size) return null;

  const alias = 't';
  const ts = timeExpr(mapping, columns, alias);
  const subject = subjectExpr(mapping, columns, alias);
  if (!ts || !subject) return null;

  const rel = qualify(schema, table);

  try {
    const sql = `
      WITH ordered AS (
        SELECT
          ${subject} AS subject_id,
          ${ts} AS ts
        FROM ${rel} ${alias}
        WHERE ${ts} IS NOT NULL
      ),
      gaps AS (
        SELECT
          subject_id,
          ts,
          EXTRACT(EPOCH FROM (ts - LAG(ts) OVER (PARTITION BY subject_id ORDER BY ts))) / 60.0
            AS gap_minutes
        FROM ordered
      ),
      positive AS (
        SELECT ROUND(gap_minutes)::int AS gap_bucket
        FROM gaps
        WHERE gap_minutes IS NOT NULL AND gap_minutes > 0
        ORDER BY ts DESC
        LIMIT ${OBSERVED_GAP_CAP}
      ),
      stats AS (
        SELECT
          COUNT(*)::int AS gaps_sampled,
          MAX(gap_bucket)::int AS max_gap_minutes
        FROM positive
      ),
      mode_row AS (
        SELECT gap_bucket, COUNT(*)::int AS mode_count
        FROM positive
        GROUP BY gap_bucket
        ORDER BY mode_count DESC, gap_bucket ASC
        LIMIT 1
      )
      SELECT
        m.gap_bucket AS mode_minutes,
        m.mode_count,
        s.gaps_sampled,
        s.max_gap_minutes
      FROM stats s
      LEFT JOIN mode_row m ON TRUE
    `;
    const res = await query(sql);
    const row = res.rows?.[0];
    if (!row || row.gaps_sampled == null || Number(row.gaps_sampled) === 0) return null;

    const gapsSampled = Number(row.gaps_sampled);
    const modeCount = Number(row.mode_count) || 0;
    const modeMinutes = row.mode_minutes != null ? Number(row.mode_minutes) : null;
    const maxGap = row.max_gap_minutes != null ? Number(row.max_gap_minutes) : null;
    const modePct = gapsSampled > 0
      ? roundMetric((modeCount / gapsSampled) * 100, 1)
      : null;

    // Cap check: if we hit the limit, result is approximate.
    // We cannot perfectly detect "exactly 5000 and more exist" without a second query;
    // treat reaching the cap as approx.
    const approx = gapsSampled >= OBSERVED_GAP_CAP;

    return {
      mode_minutes: modeMinutes,
      mode_count: modeCount,
      mode_pct: modePct,
      max_gap_minutes: maxGap,
      gaps_sampled: gapsSampled,
      approx,
    };
  } catch (err) {
    console.warn(
      `computeObservedCadence failed for ${schema}.${table}:`,
      err?.message || err
    );
    return null;
  }
}

/**
 * Batch-fetch w4h.tables registry rows for multiple datasets.
 * @param {string[]} datasetIds
 * @returns {Promise<Map<string, object[]>>}
 */
export async function fetchRegisteredTablesForDatasets(datasetIds) {
  const map = new Map();
  const ids = [...new Set((datasetIds || []).map(String).filter(Boolean))];
  if (!ids.length) return map;

  try {
    const res = await query(
      `SELECT dataset_id, name, description, source_name, source_url,
              mapping_json, physical_schema, physical_table
       FROM w4h.tables
       WHERE dataset_id = ANY($1::text[])
       ORDER BY dataset_id ASC, name ASC`,
      [ids]
    );
    for (const row of res.rows || []) {
      const id = String(row.dataset_id);
      if (!map.has(id)) map.set(id, []);
      map.get(id).push(row);
    }
  } catch {
    // Registry not provisioned on older installs.
  }

  return map;
}

function hasTimeMapping(tableRow) {
  const mapping = parseMapping(tableRow.mapping_json);
  return Boolean(mapping.time);
}

/**
 * Attach metadata_summary to dataset row objects (project overview, catalog detail).
 * Runs bounded observed-cadence queries per time-series signal.
 * @param {object[]} rows
 */
export async function attachMetadataSummaries(rows) {
  if (!Array.isArray(rows) || !rows.length) return rows || [];

  const ids = rows.map((r) => String(r.dataset_id || r.id || '')).filter(Boolean);
  const [tablesByDataset, ingestionByDataset] = await Promise.all([
    fetchRegisteredTablesForDatasets(ids),
    fetchIngestionSourcesForDatasets(ids),
  ]);

  const out = [];
  for (const row of rows) {
    const id = String(row.dataset_id || row.id || '');
    const tables = tablesByDataset.get(id) || [];
    const observedCadenceBySignal = new Map();

    const timed = tables.filter(
      (t) => t.physical_schema && t.physical_table && hasTimeMapping(t)
    );
    const [collectionRange, enrolledParticipantCount] = await Promise.all([
      computeCollectionDateRange(tables),
      computeEnrolledParticipants(tables),
      Promise.all(
        timed.map(async (tableRow) => {
          const observed = await computeObservedCadence(tableRow);
          observedCadenceBySignal.set(String(tableRow.name), observed);
        })
      ),
    ]);

    out.push({
      ...row,
      metadata_summary: buildMetadataSummary(
        row,
        tables,
        observedCadenceBySignal,
        ingestionByDataset.get(id) || null,
        collectionRange,
        enrolledParticipantCount
      ),
    });
  }

  return out;
}
