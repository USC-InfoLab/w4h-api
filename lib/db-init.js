import {query, SCHEMA_NAME, USERS_TABLE_NAME, tableExists} from '../db.js';
import bcrypt from 'bcrypt';
import fs from 'fs-extra';
import path from 'path';
import { fileURLToPath } from 'url';
import { importDocs } from './docs-repo-import.js';
import { validatePassword } from './utils/password-validator.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const SCHEMA_FILE_PATH = path.join(__dirname, '../data/schema.sql');

/**
 * Load static schema SQL file and execute it
 */
async function loadSchema() {
  try {
    if (!await fs.pathExists(SCHEMA_FILE_PATH)) {
      throw new Error(`Schema SQL file not found at ${SCHEMA_FILE_PATH}`);
    }

    const schemaSql = await fs.readFile(SCHEMA_FILE_PATH, 'utf8');
    await query(schemaSql);
    console.log('Schema SQL executed successfully');
  } catch (error) {
    console.error('Error loading schema SQL:', error);
    throw error;
  }
}

/**
 * Initialize database - creates w4h schema and set up tables
 * Requires CREATE SCHEMA and CREATE TABLE privileges
 */
async function initializeDatabase(options = {}) {
  // Create schema if it doesn't exist (requires CREATE SCHEMA privilege)
  await query(`CREATE SCHEMA IF NOT EXISTS ${SCHEMA_NAME};`);

  // Create all core tables/indexes
  await loadSchema();

  // Load docs
  await importDocs();
}

async function hasAnyUsers() {
  const result = await query(`SELECT COUNT(*)::int AS count FROM ${USERS_TABLE};`);
  return result.rows[0].count > 0;
}

async function createInitialAdmin(options = {}) {
  try {
    const { email, password, name } = options;

    if (!email || !password) {
      return { success: false, error: 'email and password are required' };
    }

    const normalizedEmail = email.trim();

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

    const insertResult = await query(
      `INSERT INTO ${USERS_TABLE} (name, email, password, role)
       VALUES ($1, $2, $3, $4)
       RETURNING id`,
      [displayName, normalizedEmail, hashedPassword, 'admin']
    );

    await query(
      `INSERT INTO ${SCHEMA_NAME}.user_studies (user_id, study_id)
       SELECT $1, s.id
       FROM ${SCHEMA_NAME}.studies s
       ON CONFLICT (user_id, study_id) DO NOTHING`,
      [insertResult.rows[0].id]
    );

    return {
      success: true,
      user: {
        email: normalizedEmail,
        name: displayName,
        role: 'admin'
      }
    };
  } catch (error) {
    if (error.code === '23505') {
      return { success: false, error: 'Email already exists' };
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
    return await tableExists(USERS_TABLE_NAME);
  } catch (error) {
    return false;
  }
}

export {
  initializeDatabase,
  createInitialAdmin,
  hasAnyUsers
};
