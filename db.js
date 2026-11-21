import pkg from 'pg';
const { Pool, types } = pkg;

types.setTypeParser(1700, d => parseFloat(d));

export const SCHEMA_NAME = 'w4h';
export const USERS_TABLE_NAME = 'users';
export const DOCS_TABLE_NAME = 'docs';
export const DOC_TYPES_TABLE_NAME = 'doc_types';
export const USERS_USER_PROJECTS = 'user_projects';

export const DOCS_TABLE = `${SCHEMA_NAME}.${DOCS_TABLE_NAME}`;
export const DOC_TYPES_TABLE = `${SCHEMA_NAME}.${DOC_TYPES_TABLE_NAME}`;
export const USER_TABLE = `${SCHEMA_NAME}.${USERS_TABLE_NAME}`;
export const USER_PROJECTS_TABLE = `${SCHEMA_NAME}.${USERS_USER_PROJECTS}`;

export const SQL_TYPES = {
  INTEGER: 'integer',
  DOUBLE_PRECISION: 'double precision',
  NUMERIC: 'numeric',
  BOOLEAN: 'boolean',
  VARCHAR: 'character varying',
  USER_DEFINED: 'USER-DEFINED',
  GEOGRAPHY: 'geography',
  JSONB: 'jsonb',
  TIMESTAMP_WITHOUT_TIME_ZONE: 'timestamp without time zone',
  TIMESTAMP_WITH_TIME_ZONE: 'timestamp with time zone',
  SERIAL: 'serial',
};

let pool = null;

export function isConfigured() {
  return !!(
    process.env.DB_HOST &&
    process.env.DB_PORT &&
    process.env.DB_DATABASE &&
    process.env.DB_USER &&
    process.env.DB_PASSWORD
  );
}

function createPool() {
  if (!isConfigured()) {
    console.warn('Database not configured. Set DB_* environment variables.');
    return null;
  }

  const poolConfig = {
    host: process.env.DB_HOST,
    port: process.env.DB_PORT,
    database: process.env.DB_DATABASE,
    user: process.env.DB_USER,
    password: process.env.DB_PASSWORD,
  };

  if (process.env.W4H_DEBUG) {
    console.log('Database connection config:', {
      host: poolConfig.host,
      port: poolConfig.port,
      database: poolConfig.database,
      user: poolConfig.user,
      password: poolConfig.password ? '***' : 'NOT SET'
    });
  }

  const newPool = new Pool(poolConfig);

  newPool.on('error', (err) => {
    console.error('POOL ERROR:', err.message || err);
  });

  newPool.query('SELECT NOW()')
    .then(() => console.log('✓ Database pool connected successfully'))
    .catch(err => {
      console.error('✗ Database pool connection FAILED:', err.message);
      console.error('  Host:', poolConfig.host, ' Port:', poolConfig.port, ' DB:', poolConfig.database);
    });

  return newPool;
}

export function getPool() {
  if (!pool && isConfigured()) {
    pool = createPool();
  }
  return pool;
}

export async function resetPool() {
  if (pool) {
    await pool.end();
    pool = null;
  }
  pool = createPool();
  return pool !== null;
}

export async function query(sql, params) {
  const currentPool = getPool();

  if (!currentPool) {
    throw new Error('Database not configured. Please configure database credentials first.');
  }

  if (process.env.W4H_DEBUG) {
    console.log(`Pool state: idle=${currentPool.idleCount}, total=${currentPool.totalCount}`);
  }

  try {
    const result = params
      ? await currentPool.query(sql, params)
      : await currentPool.query(sql);
    result.sql = sql;
    return result;
  } catch (e) {
    if (process.env.W4H_DEBUG) {
      console.log('Database Error:', {
        code: e.code, message: e.message, detail: e.detail, hint: e.hint, sql, params,
        poolState: { idle: currentPool.idleCount, total: currentPool.totalCount, waiting: currentPool.waitingCount }
      });
    }
    const wrapped = new Error(`Query failed: ${e.message}\nSQL: ${sql}`);
    wrapped.code = e.code;
    wrapped.detail = e.detail;
    wrapped.constraint = e.constraint;
    wrapped.cause = e;
    throw wrapped;
  }
}

function parseTable(table) {
  const dot = table.indexOf('.');
  return dot >= 0
    ? [table.slice(0, dot), table.slice(dot + 1)]
    : [SCHEMA_NAME, table];
}

export async function getColumnType(column, table) {
  const [schema, tableName] = parseTable(table);
  const res = await query(
    `SELECT data_type FROM information_schema.columns
     WHERE table_schema = $1 AND table_name = $2 AND column_name = $3`,
    [schema, tableName, column]
  );
  return res.rowCount === 1 ? res.rows[0].data_type : null;
}

export async function columnExists(column, table) {
  const [schema, tableName] = parseTable(table);
  const result = await query(
    `SELECT EXISTS (SELECT 1 FROM information_schema.columns
     WHERE table_schema = $1 AND table_name = $2 AND column_name = $3)`,
    [schema, tableName, column]
  );
  return result.rows[0].exists;
}

export async function tableExists(table) {
  const [schema, tableName] = parseTable(table);
  const result = await query(
    `SELECT EXISTS (SELECT 1 FROM information_schema.tables
     WHERE table_schema = $1 AND table_name = $2)`,
    [schema, tableName]
  );
  return result.rows[0].exists;
}

export async function getUserTablesInSchema(schema) {
  const result = await query(
    `SELECT table_name FROM information_schema.tables
     WHERE table_schema = $1 AND table_type = 'BASE TABLE'`,
    [schema]
  );
  return result.rows.map(row => row.table_name);
}

export default { query, resetPool, isConfigured, getPool };
