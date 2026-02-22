import pkg from 'pg';
const {Pool} = pkg;

let pool = null;

function isConfigured() {
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

  const newPool = new Pool({
    host: process.env.DB_HOST,
    port: process.env.DB_PORT,
    database: process.env.DB_DATABASE,
    user: process.env.DB_USER,
    password: process.env.DB_PASSWORD
  });

  newPool.on('error', (err) => {
    console.error('Unexpected error on idle client', err);
  });

  return newPool;
}

function getPool() {
  if (!pool && isConfigured()) {
    pool = createPool();
  }
  return pool;
}

async function resetPool() {
  if (pool) {
    await pool.end();
    pool = null;
  }
  pool = createPool();
  return pool !== null;
}

export default {
  async query(text, params) {
    const currentPool = getPool();
    
    if (!currentPool) {
      throw new Error('Database not configured. Please configure database credentials first.');
    }

    let result;
    try {
      result = await currentPool.query(text, params);
    } catch (err) {
      console.error('Database query error:', err);
      throw err;
    }
    return result;
  },
  
  resetPool,
  isConfigured,
  getPool
};