import pg from 'pg';
const { Pool, types } = pg;

// Register a parser for the 'numeric' data type that parses values as floats.
types.setTypeParser(1700, d => parseFloat(d));

let pool = null;


// XXX: check if this is needed.
// const requiredEnv = ['DB_HOST', 'DB_PORT', 'DB_DATABASE', 'DB_USER', 'DB_PASSWORD'];
// for (const key of requiredEnv) {
//   if (!process.env[key]) {
//     throw new Error(`Missing required environment variable: ${key}`);
//   }
// }

function getPool() {
  if (!pool) {
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

    pool = new Pool(poolConfig);

    pool.on('error', (err, client) => {
      console.error('POOL ERROR:', err.message || err);
      console.error('Connection refused likely means: PostgreSQL not running, wrong host/port, or firewall blocking');
    });

    // Test pool connection at startup
    pool.query('SELECT NOW() as current_time')
      .then(() => {
        console.log('✓ Database pool connected successfully');
      })
      .catch(err => {
        console.error('✗ Database pool connection FAILED:', err.message);
        console.error('  Host:', poolConfig.host);
        console.error('  Port:', poolConfig.port);
        console.error('  Database:', poolConfig.database);
      });
  }

  return pool;
}

async function query(sql, params) {
  try {
    const currentPool = getPool();
    
    if (process.env.W4H_DEBUG) {
      console.log(`Pool state: idle=${currentPool.idleCount}, total=${currentPool.totalCount}`);
    }
    
    const result = params
      ? await currentPool.query(sql, params)
      : await currentPool.query(sql); // plain SQL
    result.sql = sql;
    return result;
  } catch (e) {
    const currentPool = getPool();
    const errorDetails = {
      code: e.code,
      message: e.message,
      detail: e.detail,
      hint: e.hint,
      sql: sql,
      params: params,
      poolState: {
        idle: currentPool.idleCount,
        total: currentPool.totalCount,
        waiting: currentPool.waitingCount
      }
    };
    
    if (process.env.W4H_DEBUG) {
      console.log('Database Error:', errorDetails);
    }

    throw new Error(`Query failed: ${e.message}\nSQL: ${sql}`);
  }
}

// function to return the type of args.value_column
const getColumnType = async function (column, table) {
  const [schemaName, tableName] = table.split('.');
  const queryString = `
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_schema = '${schemaName}'
      AND table_name = '${tableName}'
      AND column_name = '${column}'
  `;
  const res = await query(queryString);
  return res.rowCount === 1 ? res.rows[0].data_type : null;
}

const columnExists = async function (column, table) {
  const [schemaName, tableName] = table.split('.');
  const result = await query(`SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = '${schemaName}' AND table_name = '${tableName}' AND column_name = '${column}')`);
  return result.rows[0].exists;
}

const tableExists = async (table) => {
  const [schemaName, tableName] = table.split('.');
  const result = await query(`SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = '${schemaName}' AND table_name = '${tableName}')`);
  return result.rows[0].exists;
}

const getUserTablesInSchema = async function (schema) {
  const result = await query(`
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = '${schema}'
      AND table_type = 'BASE TABLE'
  `);
  
  return result.rows.map(row => row.table_name);
}

// XXX: This should be the only def -- see also utils/def.js
// We only track types for injested data.
const SQL_TYPES = {
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
  SERIAL: 'serial', // auto-incrementing integer for CSV tables without unike primary key
};

export { 
  query,
  getColumnType,
  columnExists,
  tableExists,
  getUserTablesInSchema,
  SQL_TYPES
};
