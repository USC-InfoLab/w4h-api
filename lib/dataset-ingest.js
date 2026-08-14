import { parse } from 'csv-parse/sync';
import { query, withDatasetTransaction } from '../db.js';

const SUPPORTED_TYPES = new Set([
  'text',
  'integer',
  'double precision',
  'boolean',
  'date',
  'timestamptz',
]);
const RESERVED_SIGNAL_COLUMNS = new Set(['subject_id', 'ts']);
const IDENTIFIER_PATTERN = /^[a-z][a-z0-9_]{0,31}$/;
const INSERT_BATCH_SIZE = 1000;

export class IngestionError extends Error {
  constructor(message, statusCode = 400, code = 'INVALID_INGESTION') {
    super(message);
    this.name = 'IngestionError';
    this.statusCode = statusCode;
    this.code = code;
  }
}

function quoteIdentifier(identifier) {
  if (!/^[a-z_][a-z0-9_]*$/.test(identifier)) {
    throw new IngestionError(`Unsafe database identifier: ${identifier}`);
  }
  return `"${identifier}"`;
}

function requireSourceName(value, label) {
  const name = String(value || '').trim();
  if (!name) throw new IngestionError(`${label} is required`);
  return name;
}

function parseCsv(buffer, label) {
  if (!buffer?.length) throw new IngestionError(`${label} is required`);
  try {
    return parse(buffer, {
      bom: true,
      columns: true,
      skip_empty_lines: true,
      trim: true,
    });
  } catch (error) {
    throw new IngestionError(`${label} is not valid CSV: ${error.message}`);
  }
}

function assertHeaders(rows, requiredHeaders, label) {
  if (!rows.length) throw new IngestionError(`${label} must contain at least one data row`);
  const headers = new Set(Object.keys(rows[0]));
  for (const header of requiredHeaders) {
    if (!headers.has(header)) {
      throw new IngestionError(`${label} is missing mapped column "${header}"`);
    }
  }
}

function parseDate(value, label, { optional = false } = {}) {
  const raw = String(value ?? '').trim();
  if (!raw && optional) return null;
  if (!/^\d{4}-\d{2}-\d{2}$/.test(raw)) {
    throw new IngestionError(`${label} must use YYYY-MM-DD`);
  }
  const date = new Date(`${raw}T00:00:00.000Z`);
  if (Number.isNaN(date.getTime()) || date.toISOString().slice(0, 10) !== raw) {
    throw new IngestionError(`${label} is not a valid date`);
  }
  return raw;
}

function convertValue(value, type, label) {
  const raw = String(value ?? '').trim();
  if (type === 'text') {
    if (!raw) throw new IngestionError(`${label} cannot be empty`);
    return raw;
  }
  if (type === 'integer') {
    if (!/^[+-]?\d+$/.test(raw)) throw new IngestionError(`${label} must be an integer`);
    const number = Number(raw);
    if (!Number.isSafeInteger(number) || number < -2147483648 || number > 2147483647) {
      throw new IngestionError(`${label} is outside the PostgreSQL integer range`);
    }
    return number;
  }
  if (type === 'double precision') {
    const number = Number(raw);
    if (!raw || !Number.isFinite(number)) {
      throw new IngestionError(`${label} must be a finite number`);
    }
    return number;
  }
  if (type === 'boolean') {
    if (['true', 't', '1', 'yes'].includes(raw.toLowerCase())) return true;
    if (['false', 'f', '0', 'no'].includes(raw.toLowerCase())) return false;
    throw new IngestionError(`${label} must be a boolean`);
  }
  if (type === 'date') return parseDate(raw, label);
  if (type === 'timestamptz') {
    const normalized = /^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?$/.test(raw)
      ? `${raw.replace(' ', 'T')}Z`
      : raw;
    const date = new Date(normalized);
    if (!raw || Number.isNaN(date.getTime())) {
      throw new IngestionError(`${label} must be a valid timestamp`);
    }
    return date.toISOString();
  }
  throw new IngestionError(`Unsupported mapped type: ${type}`);
}

export function normalizeMapping(input) {
  if (!input || typeof input !== 'object' || Array.isArray(input)) {
    throw new IngestionError('mapping must be a JSON object');
  }
  const subjects = input.subjects;
  const signal = input.signal;
  if (!subjects || typeof subjects !== 'object' || !signal || typeof signal !== 'object') {
    throw new IngestionError('mapping.subjects and mapping.signal are required');
  }

  const signalType = String(signal.signal_type || '').trim().toLowerCase();
  if (!IDENTIFIER_PATTERN.test(signalType)) {
    throw new IngestionError('signal.signal_type must be a lowercase SQL-safe slug');
  }
  const timestamp = signal.timestamp;
  if (!timestamp || typeof timestamp !== 'object') {
    throw new IngestionError('signal.timestamp is required');
  }
  let timestampMapping;
  if (timestamp.column) {
    timestampMapping = { column: requireSourceName(timestamp.column, 'signal.timestamp.column') };
  } else if (Array.isArray(timestamp.combine) && timestamp.combine.length >= 2) {
    timestampMapping = {
      combine: timestamp.combine.map((item, index) =>
        requireSourceName(item, `signal.timestamp.combine[${index}]`)
      ),
    };
  } else {
    throw new IngestionError('signal.timestamp requires column or combine');
  }

  if (!Array.isArray(signal.columns) || !signal.columns.length || signal.columns.length > 20) {
    throw new IngestionError('signal.columns must contain between 1 and 20 mapped columns');
  }
  const seenTargets = new Set();
  const columns = signal.columns.map((column, index) => {
    const source = requireSourceName(column?.source, `signal.columns[${index}].source`);
    const target = String(column?.target || '').trim().toLowerCase();
    const type = String(column?.type || '').trim().toLowerCase();
    if (!IDENTIFIER_PATTERN.test(target) || RESERVED_SIGNAL_COLUMNS.has(target)) {
      throw new IngestionError(`Invalid signal target column: ${target || '(empty)'}`);
    }
    if (seenTargets.has(target)) throw new IngestionError(`Duplicate signal target column: ${target}`);
    if (!SUPPORTED_TYPES.has(type)) throw new IngestionError(`Unsupported type for ${target}: ${type}`);
    seenTargets.add(target);
    return { source, target, type };
  });

  const samplingHz = signal.sampling_hz == null ? null : Number(signal.sampling_hz);
  if (samplingHz != null && (!Number.isFinite(samplingHz) || samplingHz <= 0)) {
    throw new IngestionError('signal.sampling_hz must be a positive number');
  }

  return {
    version: 1,
    device_type: input.device_type == null ? null : String(input.device_type).trim() || null,
    subjects: {
      id: requireSourceName(subjects.id, 'subjects.id'),
      enrollment_start: subjects.enrollment_start
        ? requireSourceName(subjects.enrollment_start, 'subjects.enrollment_start')
        : null,
      enrollment_end: subjects.enrollment_end
        ? requireSourceName(subjects.enrollment_end, 'subjects.enrollment_end')
        : null,
    },
    signal: {
      signal_type: signalType,
      subject_id: requireSourceName(signal.subject_id, 'signal.subject_id'),
      timestamp: timestampMapping,
      columns,
      sampling_hz: samplingHz,
      geospatial: signal.geospatial === true,
    },
  };
}

export function parseMappedCsvFiles({ subjectsBuffer, signalBuffer, mapping }) {
  const normalizedMapping = normalizeMapping(mapping);
  const subjectRows = parseCsv(subjectsBuffer, 'subjects_csv');
  const signalRows = parseCsv(signalBuffer, 'signal_csv');

  const subjectHeaders = [
    normalizedMapping.subjects.id,
    normalizedMapping.subjects.enrollment_start,
    normalizedMapping.subjects.enrollment_end,
  ].filter(Boolean);
  assertHeaders(subjectRows, subjectHeaders, 'subjects_csv');

  const signalTimestampHeaders = normalizedMapping.signal.timestamp.column
    ? [normalizedMapping.signal.timestamp.column]
    : normalizedMapping.signal.timestamp.combine;
  assertHeaders(
    signalRows,
    [
      normalizedMapping.signal.subject_id,
      ...signalTimestampHeaders,
      ...normalizedMapping.signal.columns.map((column) => column.source),
    ],
    'signal_csv'
  );

  const subjectIds = new Set();
  const subjects = subjectRows.map((row, index) => {
    const id = String(row[normalizedMapping.subjects.id] ?? '').trim();
    if (!id) throw new IngestionError(`subjects_csv row ${index + 2}: subject id is required`);
    if (subjectIds.has(id)) {
      throw new IngestionError(`subjects_csv row ${index + 2}: duplicate subject id "${id}"`);
    }
    subjectIds.add(id);
    const enrollmentStart = normalizedMapping.subjects.enrollment_start
      ? parseDate(
        row[normalizedMapping.subjects.enrollment_start],
        `subjects_csv row ${index + 2}: enrollment_start`,
        { optional: true }
      )
      : null;
    const enrollmentEnd = normalizedMapping.subjects.enrollment_end
      ? parseDate(
        row[normalizedMapping.subjects.enrollment_end],
        `subjects_csv row ${index + 2}: enrollment_end`,
        { optional: true }
      )
      : null;
    if (enrollmentStart && enrollmentEnd && enrollmentStart > enrollmentEnd) {
      throw new IngestionError(`subjects_csv row ${index + 2}: enrollment_start is after enrollment_end`);
    }
    return { id, enrollmentStart, enrollmentEnd };
  });

  const seenObservations = new Set();
  let minTimestamp = null;
  let maxTimestamp = null;
  const signals = signalRows.map((row, index) => {
    const subjectId = String(row[normalizedMapping.signal.subject_id] ?? '').trim();
    if (!subjectIds.has(subjectId)) {
      throw new IngestionError(
        `signal_csv row ${index + 2}: unknown subject id "${subjectId}"`
      );
    }
    const rawTimestamp = normalizedMapping.signal.timestamp.column
      ? row[normalizedMapping.signal.timestamp.column]
      : normalizedMapping.signal.timestamp.combine.map((header) => row[header]).join(' ');
    const timestamp = convertValue(
      rawTimestamp,
      'timestamptz',
      `signal_csv row ${index + 2}: timestamp`
    );
    const observationKey = `${subjectId}\u0000${timestamp}`;
    if (seenObservations.has(observationKey)) {
      throw new IngestionError(
        `signal_csv row ${index + 2}: duplicate subject/timestamp observation`
      );
    }
    seenObservations.add(observationKey);
    if (!minTimestamp || timestamp < minTimestamp) minTimestamp = timestamp;
    if (!maxTimestamp || timestamp > maxTimestamp) maxTimestamp = timestamp;
    return {
      subjectId,
      timestamp,
      values: normalizedMapping.signal.columns.map((column) =>
        convertValue(
          row[column.source],
          column.type,
          `signal_csv row ${index + 2}: ${column.source}`
        )
      ),
    };
  });

  return {
    mapping: normalizedMapping,
    subjects,
    signals,
    minDate: minTimestamp?.slice(0, 10) || null,
    maxDate: maxTimestamp?.slice(0, 10) || null,
  };
}

async function insertSubjects(client, subjects) {
  for (const subject of subjects) {
    const existing = await client.query(
      `SELECT enrollment_start, enrollment_end FROM subjects WHERE id = $1`,
      [subject.id]
    );
    if (existing.rowCount) {
      const row = existing.rows[0];
      if (
        row.enrollment_start &&
        subject.enrollmentStart &&
        String(row.enrollment_start).slice(0, 10) !== subject.enrollmentStart
      ) {
        throw new IngestionError(`Subject ${subject.id} has a conflicting enrollment_start`, 409);
      }
      if (
        row.enrollment_end &&
        subject.enrollmentEnd &&
        String(row.enrollment_end).slice(0, 10) !== subject.enrollmentEnd
      ) {
        throw new IngestionError(`Subject ${subject.id} has a conflicting enrollment_end`, 409);
      }
    }
    await client.query(
      `INSERT INTO subjects (id, enrollment_start, enrollment_end)
       VALUES ($1, $2, $3)
       ON CONFLICT (id) DO UPDATE
       SET enrollment_start = COALESCE(subjects.enrollment_start, EXCLUDED.enrollment_start),
           enrollment_end = COALESCE(subjects.enrollment_end, EXCLUDED.enrollment_end),
           updated_at = CURRENT_TIMESTAMP`,
      [subject.id, subject.enrollmentStart, subject.enrollmentEnd]
    );
  }
}

async function insertSignalRows(client, tableName, columns, signals) {
  const quotedTable = quoteIdentifier(tableName);
  const quotedColumns = [
    quoteIdentifier('subject_id'),
    quoteIdentifier('ts'),
    ...columns.map((column) => quoteIdentifier(column.target)),
  ];
  const width = quotedColumns.length;
  for (let offset = 0; offset < signals.length; offset += INSERT_BATCH_SIZE) {
    const batch = signals.slice(offset, offset + INSERT_BATCH_SIZE);
    const params = [];
    const values = batch.map((row, rowIndex) => {
      params.push(row.subjectId, row.timestamp, ...row.values);
      const base = rowIndex * width;
      return `(${Array.from({ length: width }, (_, index) => `$${base + index + 1}`).join(', ')})`;
    });
    await client.query(
      `INSERT INTO ${quotedTable} (${quotedColumns.join(', ')})
       VALUES ${values.join(', ')}`,
      params
    );
  }
}

function schemaDefinition(mapping) {
  return {
    version: mapping.version,
    device_type: mapping.device_type,
    tables: [
      { name: 'subjects' },
      {
        name: mapping.signal.signal_type,
        columns: mapping.signal.columns.map(({ target: name, type }) => ({ name, type })),
        sampling_hz: mapping.signal.sampling_hz,
      },
    ],
    signals: [{
      name: mapping.signal.signal_type,
      columns: mapping.signal.columns.map(({ target: name, type }) => ({ name, type })),
      sampling_hz: mapping.signal.sampling_hz,
    }],
    geospatial: mapping.signal.geospatial,
    source_mapping: {
      subjects: mapping.subjects,
      signal: {
        subject_id: mapping.signal.subject_id,
        timestamp: mapping.signal.timestamp,
        columns: mapping.signal.columns,
      },
    },
  };
}

async function markDatasetFailed(catalogId) {
  try {
    await query(
      `UPDATE w4h.datasets
       SET ingest_status = 'failed', updated_at = CURRENT_TIMESTAMP
       WHERE id = $1 AND ingest_status IN ('created', 'ingesting', 'failed')`,
      [catalogId]
    );
  } catch (error) {
    console.warn('Failed to mark dataset ingestion as failed:', error.message);
  }
}

export async function ingestDatasetFromCsv({
  teamId,
  catalogId,
  subjectsBuffer,
  signalBuffer,
  mapping,
}) {
  let parsed;
  try {
    parsed = parseMappedCsvFiles({ subjectsBuffer, signalBuffer, mapping });
    const result = await withDatasetTransaction(catalogId, async (client, { schemaName, teamId: registryTeamId }) => {
      if (String(registryTeamId) !== String(teamId)) {
        throw new IngestionError('Dataset does not belong to this team', 409, 'DATASET_TEAM_MISMATCH');
      }
      const datasetResult = await client.query(
        `SELECT id, ingest_status
         FROM w4h.datasets
         WHERE id = $1
         FOR UPDATE`,
        [catalogId]
      );
      if (!datasetResult.rowCount) {
        throw new IngestionError('Dataset was not found', 409, 'DATASET_MISSING');
      }
      const dataset = datasetResult.rows[0];
      if (!['created', 'failed'].includes(dataset.ingest_status)) {
        throw new IngestionError(
          `Dataset cannot be ingested from status "${dataset.ingest_status}"`,
          409,
          'INVALID_DATASET_STATUS'
        );
      }

      const tableName = parsed.mapping.signal.signal_type;
      if (!IDENTIFIER_PATTERN.test(tableName)) {
        throw new IngestionError('signal_type must be a lowercase SQL identifier');
      }
      const subjectsColumnsDef = [
        { name: 'id', type: 'text' },
        { name: 'enrollment_start', type: 'date' },
        { name: 'enrollment_end', type: 'date' },
      ];
      const columnsDef = [
        { name: 'subject_id', type: 'text' },
        { name: 'ts', type: 'timestamptz' },
        ...parsed.mapping.signal.columns.map(({ target: name, type }) => ({ name, type })),
      ];
      const schemaDef = schemaDefinition(parsed.mapping);

      await client.query(
        `UPDATE w4h.datasets SET ingest_status = 'ingesting', updated_at = CURRENT_TIMESTAMP WHERE id = $1`,
        [dataset.id]
      );

      await client.query('SET LOCAL ROLE w4h_platform');
      if (dataset.ingest_status === 'failed') {
        await client.query(
          `DELETE FROM dataset_tables WHERE signal_type = ANY($1::text[])`,
          [['subjects', tableName]]
        );
        await client.query(`DROP TABLE IF EXISTS ${quoteIdentifier(schemaName)}.${quoteIdentifier(tableName)}`);
      }
      const valueColumnsSql = parsed.mapping.signal.columns
        .map((column) => `${quoteIdentifier(column.target)} ${column.type} NOT NULL`)
        .join(',\n');
      await client.query(
        `CREATE TABLE ${quoteIdentifier(schemaName)}.${quoteIdentifier(tableName)} (
           subject_id text NOT NULL REFERENCES ${quoteIdentifier(schemaName)}.${quoteIdentifier('subjects')}(id) ON DELETE RESTRICT,
           ts timestamptz NOT NULL,
           ${valueColumnsSql},
           UNIQUE (subject_id, ts)
         )`
      );
      const indexName = `${tableName.slice(0, 55)}_ts_idx`;
      await client.query(
        `CREATE INDEX ${quoteIdentifier(indexName)} ON ${quoteIdentifier(schemaName)}.${quoteIdentifier(tableName)} (ts)`
      );
      await client.query('RESET ROLE');

      await insertSubjects(client, parsed.subjects);
      await insertSignalRows(
        client,
        tableName,
        parsed.mapping.signal.columns,
        parsed.signals
      );
      await client.query(
        `INSERT INTO dataset_tables (table_name, signal_type, columns_def)
         VALUES ('subjects', 'subjects', $1::jsonb),
                ($2, $3, $4::jsonb)
         ON CONFLICT (signal_type) DO UPDATE
           SET table_name = EXCLUDED.table_name,
               columns_def = EXCLUDED.columns_def`,
        [
          JSON.stringify(subjectsColumnsDef),
          tableName,
          tableName,
          JSON.stringify(columnsDef),
        ]
      );
      await client.query(
        `UPDATE w4h.datasets
         SET schema_def = $2::jsonb,
             ingest_status = 'ready',
             device_type = COALESCE($3, device_type),
             sampling_rate = COALESCE($4, sampling_rate),
             subject_count = $5,
             date_start = $6,
             date_end = $7,
             updated_at = CURRENT_TIMESTAMP
         WHERE id = $1`,
        [
          catalogId,
          JSON.stringify(schemaDef),
          parsed.mapping.device_type,
          parsed.mapping.signal.sampling_hz == null
            ? null
            : String(parsed.mapping.signal.sampling_hz),
          parsed.subjects.length,
          parsed.minDate,
          parsed.maxDate,
        ]
      );

      const tables = await client.query(
        `SELECT table_name AS name, table_name, signal_type, columns_def
         FROM dataset_tables
         ORDER BY table_name`
      );

      return {
        dataset_id: catalogId,
        status: 'ready',
        schema_def: schemaDef,
        subjects_loaded: parsed.subjects.length,
        signal_rows_loaded: parsed.signals.length,
        table: {
          name: tableName,
          signal_type: tableName,
          columns_def: columnsDef,
        },
        tables: tables.rows,
      };
    });
    return result;
  } catch (error) {
    await markDatasetFailed(catalogId);
    throw error;
  }
}
