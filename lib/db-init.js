import db from '../db.js';
import bcrypt from 'bcrypt';
import fs from 'fs-extra';
import path from 'path';
import { fileURLToPath } from 'url';
import { validatePassword } from './utils/password-validator.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// IMPORTANT: Initial setup requires CREATEDB privilege
// Recommended approach:
// 1. Setup: Use privileged user (postgres superuser, or user with CREATEDB privilege)
//    - Database will be created if it doesn't exist
//    - Schema and tables will be created automatically
// 2. Runtime: Admin can later switch to restricted user with only DML privileges
//    (INSERT, SELECT, UPDATE, DELETE on w4h.* tables)
// All operations use IF NOT EXISTS - safe and idempotent
const SCHEMA_NAME = 'w4h';
const TABLE_NAME = 'users';

/**
 * Load schema SQL file and execute it
 */
async function loadSchema() {
  try {
    const schemaSqlPath = path.join(__dirname, '../data/schema.sql');

    if (!await fs.pathExists(schemaSqlPath)) {
      throw new Error(`Schema SQL file not found at ${schemaSqlPath}`);
    }

    const schemaSql = await fs.readFile(schemaSqlPath, 'utf8');
    await db.query(schemaSql);
    console.log('Schema SQL executed successfully');
  } catch (error) {
    console.error('Error loading schema SQL:', error);
    throw error;
  }
}

/**
 * Load initial docs from SQL file and execute it
 */
async function loadInitialDocs() {
  try {
    const docsSqlPath = path.join(__dirname, '../data/docs.sql');
    
    if (!await fs.pathExists(docsSqlPath)) {
      console.log('No initial docs SQL file found at', docsSqlPath);
      return;
    }

    const docsSql = await fs.readFile(docsSqlPath, 'utf8');
    
    // Execute the SQL file
    await db.query(docsSql);
    console.log('Initial docs loaded successfully');
  } catch (error) {
    console.error('Error loading initial docs:', error);
    throw error;
  }
}

/**
 * Initialize database - creates w4h schema and users table
 * Note: Database creation is handled by the config endpoint
 * Requires CREATE SCHEMA and CREATE TABLE privileges
 * After setup, credentials can be updated to a less privileged user
 */
async function initializeDatabase(options = {}) {
  try {
    // Create schema if it doesn't exist (requires CREATE SCHEMA privilege)
    console.log(`Creating schema ${SCHEMA_NAME} if not exists...`);
    await db.query(`CREATE SCHEMA IF NOT EXISTS ${SCHEMA_NAME};`);
    console.log(`Schema ${SCHEMA_NAME} created or already exists`);

    // Check if users table exists in w4h schema
    const tableCheck = await db.query(`
      SELECT EXISTS (
        SELECT FROM information_schema.tables 
        WHERE table_schema = $1
        AND table_name = $2
      );
    `, [SCHEMA_NAME, TABLE_NAME]);

    const tableExists = tableCheck.rows[0].exists;

    // Check if docs table exists in w4h schema
    const docsTableCheck = await db.query(`
      SELECT EXISTS (
        SELECT FROM information_schema.tables 
        WHERE table_schema = $1
        AND table_name = 'docs'
      );
    `, [SCHEMA_NAME]);

    const docsTableExists = docsTableCheck.rows[0].exists;

    // Create all core tables/indexes from SQL file (idempotent)
    await loadSchema();

    if (!tableExists) {
      console.log(`Creating ${SCHEMA_NAME}.${TABLE_NAME} table...`);

      console.log(`Users table created successfully in ${SCHEMA_NAME} schema`);
    } else {
      console.log(`Users table already exists in ${SCHEMA_NAME} schema`);
    }

    if (!docsTableExists) {
      console.log(`Creating ${SCHEMA_NAME}.docs table...`);

      console.log(`Docs table created successfully in ${SCHEMA_NAME} schema`);
    } else {
      console.log(`Docs table already exists in ${SCHEMA_NAME} schema`);
    }

    // Load initial docs from file if docs table was just created
    if (!docsTableExists) {
      console.log('Loading initial docs...');
      await loadInitialDocs();
    }

    return { success: true, message: 'Database initialized successfully' };
  } catch (error) {
    console.error('Database initialization error:', error);
    return { success: false, error: error.message };
  }
}

async function hasAnyUsers() {
  const result = await db.query(`SELECT COUNT(*)::int AS count FROM ${SCHEMA_NAME}.${TABLE_NAME};`);
  return result.rows[0].count > 0;
}

async function createInitialAdmin(options = {}) {
  try {
    const { username, email, password, name } = options;

    if (!username || !email || !password) {
      return { success: false, error: 'username, email, and password are required' };
    }

    const passwordValidation = validatePassword(password);
    if (!passwordValidation.valid) {
      return { success: false, error: passwordValidation.message };
    }

    const usersExist = await hasAnyUsers();
    if (usersExist) {
      return { success: false, error: 'Initial admin already configured' };
    }

    const hashedPassword = bcrypt.hashSync(password, 8);
    const displayName = name || 'Administrator';

    await db.query(
      `INSERT INTO ${SCHEMA_NAME}.${TABLE_NAME} (username, name, email, password, role)
       VALUES ($1, $2, $3, $4, $5)`,
      [username, displayName, email, hashedPassword, 'admin']
    );

    return {
      success: true,
      user: {
        username,
        email,
        name: displayName,
        role: 'admin'
      }
    };
  } catch (error) {
    if (error.code === '23505') {
      return { success: false, error: 'Username or email already exists' };
    }

    console.error('Initial admin creation error:', error);
    return { success: false, error: error.message };
  }
}

/**
 * Check if database is initialized
 */
async function isDatabaseInitialized() {
  try {
    const tableCheck = await db.query(`
      SELECT EXISTS (
        SELECT FROM information_schema.tables 
        WHERE table_schema = $1
        AND table_name = $2
      );
    `, [SCHEMA_NAME, TABLE_NAME]);
    return tableCheck.rows[0].exists;
  } catch (error) {
    return false;
  }
}

export {
  initializeDatabase,
  isDatabaseInitialized,
  loadInitialDocs,
  loadSchema,
  createInitialAdmin,
  hasAnyUsers
};
