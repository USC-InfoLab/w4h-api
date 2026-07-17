import express from 'express';
const router = express.Router();
import fs from 'fs-extra';
import path from 'path';
import { fileURLToPath } from 'url';
import { isSuperAdmin } from '../lib/utils/user.js';
import crypto from 'crypto';
import pkg from 'pg';
const { Pool } = pkg;
import db, { tableExists, USERS_TABLE_NAME } from '../db.js';
import * as auth from '../auth.js';
import { initializeDatabase, createInitialAdmin, hasAnyUsers } from '../lib/db-init.js';
import { getMailerConfig, isMailerConfigured, sendMail } from '../lib/mailer.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const isAdmin = user => !!user && isSuperAdmin(user.role);

// Keep `.env` formatting consistent with `.env.example` by rewriting `.env`
// from the template (preserves ordering + comments).
const ENV_PATH = path.join(__dirname, '..', '.env');
const ENV_EXAMPLE_PATH = path.join(__dirname, '..', '.env.example');

function parseEnvValues(envContent = '') {
  const values = {};
  const lines = String(envContent).split(/\r?\n/);

  for (const line of lines) {
    // Match `KEY=value` only (non-comment lines)
    const m = /^\s*([A-Z0-9_]+)\s*=(.*)\s*$/.exec(line);
    if (!m) continue;
    values[m[1]] = m[2];
  }

  return values;
}

function extractTemplateKeys(templateContent) {
  const keys = [];
  const lines = String(templateContent).split(/\r?\n/);
  const keyLineRe = /^\s*([A-Z0-9_]+)\s*=/;

  for (const line of lines) {
    const m = keyLineRe.exec(line);
    if (m) keys.push(m[1]);
  }
  return keys;
}

function applyValuesToTemplate(templateContent, mergedValues, templateKeys) {
  let out = String(templateContent);

  for (const key of templateKeys) {
    if (!Object.prototype.hasOwnProperty.call(mergedValues, key)) continue;
    const value = mergedValues[key] ?? '';
    const lineRe = new RegExp(`^${key}=.*$`, 'm');
    out = out.replace(lineRe, `${key}=${value}`);
  }

  return out;
}

async function writeEnvValues(keyValues) {
  if (!(await fs.pathExists(ENV_EXAMPLE_PATH))) {
    throw new Error(`Missing ${path.basename(ENV_EXAMPLE_PATH)} — cannot preserve formatting`);
  }

  const template = await fs.readFile(ENV_EXAMPLE_PATH, 'utf8');
  const templateKeys = extractTemplateKeys(template);

  const existingValues = (await fs.pathExists(ENV_PATH))
    ? parseEnvValues(await fs.readFile(ENV_PATH, 'utf8'))
    : {};

  const mergedValues = { ...existingValues, ...keyValues };

  // Rewrite `.env` from `.env.example` to guarantee stable ordering/comments
  let out = applyValuesToTemplate(template, mergedValues, templateKeys);

  // If callers pass keys not present in the template, append them.
  const missingKeys = Object.keys(keyValues).filter(k => !templateKeys.includes(k));
  if (missingKeys.length) {
    const extra = missingKeys.map(k => `${k}=${keyValues[k] ?? ''}`).join('\n');
    out = out.trimEnd() + `\n${extra}\n`;
  }

  await fs.writeFile(ENV_PATH, out.trimEnd() + '\n');

  // Keep runtime env in sync for keys we just changed
  Object.keys(keyValues).forEach((key) => {
    process.env[key] = String(keyValues[key] ?? '');
  });
}

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

router.get('/status', getStatus);
router.get('/database-settings', auth.authenticate, getDatabaseSettings);
router.get('/mailer-settings', auth.authenticate, getMailerSettings);
router.post('/mailer-settings', auth.authenticate, postMailerSettings);
router.post('/mailer-test', auth.authenticate, postMailerTest);
router.get('/app-settings', auth.authenticate, getAppSettings);
router.post('/app-settings', auth.authenticate, postAppSettings);
router.post('/init-database', postInitDatabase);
router.post('/admin-setup', postAdminSetup);
router.get('/test', auth.authenticate, getTest);
router.post('/reset', auth.authenticate, postReset);

export default router;

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/** GET /config/status — Check if database is configured. */
async function getStatus(req, res) {
  const envConfigured = !!(
    process.env.DB_HOST &&
    process.env.DB_PORT &&
    process.env.DB_DATABASE &&
    process.env.DB_USER &&
    process.env.DB_PASSWORD
  );

  const details = {
    host: process.env.DB_HOST || '',
    port: process.env.DB_PORT || '5432',
    database: process.env.DB_DATABASE || '',
    user: process.env.DB_USER || ''
  };

  if (!envConfigured) {
    return res.json({ configured: false, details });
  }

  // Env vars are set — verify the DB is actually reachable and schema is initialized
  try {
    const initialized = await tableExists(USERS_TABLE_NAME);
    if (!initialized) {
      return res.json({ configured: false, details });
    }
    return res.json({ configured: true, details });
  } catch {
    return res.json({ configured: false, details });
  }
}

/** GET /config/database-settings — Get current database settings (even if not fully configured). */
function getDatabaseSettings(req, res) {
  if (!isAdmin(req.user)) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  res.json({
    host: process.env.DB_HOST || '',
    port: process.env.DB_PORT || '5432',
    database: process.env.DB_DATABASE || '',
    user: process.env.DB_USER || ''
  });
}

/** GET /config/mailer-settings — Get current SMTP settings (without password). */
function getMailerSettings(req, res) {
  if (!isAdmin(req.user)) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  const cfg = getMailerConfig();
  res.json({
    host: cfg.host,
    port: cfg.port,
    secure: cfg.secure,
    user: cfg.user,
    from: cfg.from,
    replyTo: cfg.replyTo,
    configured: isMailerConfigured(),
    hasPassword: !!cfg.pass
  });
}

/** POST /config/mailer-settings — Save SMTP settings to .env. Admin only. */
async function postMailerSettings(req, res) {
  if (!isAdmin(req.user)) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  const host = String(req.body?.host || '').trim();
  const port = String(req.body?.port || '').trim();
  const secure = !!req.body?.secure;
  const user = String(req.body?.user || '').trim();
  const from = String(req.body?.from || '').trim();
  const replyTo = String(req.body?.replyTo || '').trim();
  const incomingPassword = req.body?.password;

  if (!host || !port || !from) {
    return res.status(400).json({
      success: false,
      error: 'Required SMTP fields: host, port, from'
    });
  }

  try {
    const effectivePassword = incomingPassword === undefined || incomingPassword === null || incomingPassword === ''
      ? (process.env.SMTP_PASS || '')
      : String(incomingPassword);

    await writeEnvValues({
      SMTP_HOST: host,
      SMTP_PORT: port,
      SMTP_SECURE: secure ? 'true' : 'false',
      SMTP_USER: user,
      SMTP_PASS: effectivePassword,
      SMTP_FROM: from,
      SMTP_REPLY_TO: replyTo
    });

    return res.json({
      success: true,
      message: 'SMTP settings saved',
      configured: isMailerConfigured()
    });
  } catch (error) {
    return res.status(500).json({
      success: false,
      error: 'Failed to save SMTP settings',
      details: error.message
    });
  }
}

/** POST /config/mailer-test — Send a test email. Admin only.
 *  Optional SMTP fields in the body are used for this send only (no Save required).
 *  Omitted password falls back to the saved SMTP_PASS.
 */
async function postMailerTest(req, res) {
  if (!isAdmin(req.user)) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  const to = String(req.body?.to || '').trim();
  if (!to) {
    return res.status(400).json({ success: false, error: 'Test recipient email is required' });
  }

  const smtpOverrides = {
    host: req.body?.host,
    port: req.body?.port,
    secure: req.body?.secure,
    user: req.body?.user,
    password: req.body?.password,
    from: req.body?.from,
    replyTo: req.body?.replyTo
  };

  // If the client sent any SMTP field, treat this as an override payload
  const hasOverride = ['host', 'port', 'from', 'user', 'password', 'replyTo'].some(
    (k) => req.body?.[k] !== undefined && req.body?.[k] !== null && String(req.body[k]).trim() !== ''
  ) || req.body?.secure !== undefined;

  try {
    await sendMail({
      to,
      subject: 'W4H SMTP Test',
      text: 'SMTP test email from W4H settings.',
      html: '<p>SMTP test email from <strong>W4H</strong> settings.</p>',
      smtp: hasOverride ? smtpOverrides : null
    });

    return res.json({
      success: true,
      message: hasOverride
        ? `Test email sent to ${to} using the form values (not saved yet).`
        : `Test email sent to ${to}`
    });
  } catch (error) {
    return res.status(500).json({
      success: false,
      error: 'SMTP test failed',
      details: error.message
    });
  }
}

/** GET /config/app-settings — Get general app settings. Admin only. */
function getAppSettings(req, res) {
  if (!isAdmin(req.user)) {
    return res.status(403).json({ error: 'Unauthorized' });
  }
  res.json({
    helpEmail: process.env.HELP_EMAIL || '',
    usageTracking: process.env.USAGE_TRACKING === 'true',
  });
}

/** POST /config/app-settings — Save general app settings. Admin only. */
async function postAppSettings(req, res) {
  if (!isAdmin(req.user)) {
    return res.status(403).json({ error: 'Unauthorized' });
  }
  const helpEmail = String(req.body?.helpEmail ?? '').trim();
  const usageTracking = !!req.body?.usageTracking;

  try {
    await writeEnvValues({ HELP_EMAIL: helpEmail, USAGE_TRACKING: usageTracking ? 'true' : 'false' });
    return res.json({ success: true, message: 'App settings saved.' });
  } catch (error) {
    return res.status(500).json({ success: false, error: 'Failed to save app settings.', details: error.message });
  }
}

/**
 * POST /config/database
 * Configure database credentials
 * Body: { host, port, database, user, password }
 */
async function postInitDatabase(req, res) {
  const { host, port, database, user, password } = req.body;
  const effectivePassword = password || process.env.DB_PASSWORD;

  // Validate required fields
  if (!host || !port || !database || !user) {
    return res.status(400).json({
      success: false,
      error: 'Required database fields: host, port, database, user'
    });
  }

  if (!effectivePassword) {
    return res.status(400).json({
      success: false,
      error: 'Password is required when no existing database password is configured'
    });
  }

  try {
    // Test the connection first
    const testPool = new Pool({
      host,
      port: parseInt(port),
      database,
      user,
      password: effectivePassword,
      connectionTimeoutMillis: 5000
    });

    let databaseCreated = false;
    let testPoolEnded = false;
    try {
      await testPool.query('SELECT 1');
    } catch (connError) {
      if (process.env.W4H_DEBUG) console.log(`[config] Connection error code: ${connError.code}, message: ${connError.message}`);
      
      // If database doesn't exist, try to create it
      if (connError.code === '3D000' || connError.message?.includes('does not exist')) { 
        if (process.env.W4H_DEBUG) console.info(`[config] Database '${database}' does not exist. Attempting to create it...`);
        await testPool.end();
        testPoolEnded = true;
        
        // Connect to 'postgres' database to create the target database
        const adminPool = new Pool({
          host,
          port: parseInt(port),
          database: 'postgres', // Connect to default postgres database
          user,
          password: effectivePassword,
          connectionTimeoutMillis: 5000
        });

        try {
          // Test connection to postgres database first
          await adminPool.query('SELECT 1');
        } catch (postgresError) {
          await adminPool.end();
          throw new Error(`Cannot connect to PostgreSQL server with provided credentials: ${postgresError.message}. Make sure host, port, username, and password are correct, and PostgreSQL is running.`);
        }

        try {
          // First check if user has CREATEDB privilege
          const privCheck = await adminPool.query(
            "SELECT usecreatedb FROM pg_user WHERE usename = $1",
            [user]
          );
          
          if (privCheck.rows.length === 0) {
            throw new Error(`User '${user}' not found on PostgreSQL server. Check that the username exists and is correct.`);
          }
          
          if (!privCheck.rows[0].usecreatedb) {
            throw new Error(`User '${user}' does not have CREATEDB privilege. Your DBA needs to run: ALTER USER "${user}" CREATEDB;`);
          }

          await adminPool.query(`CREATE DATABASE "${database}" TEMPLATE template1 ENCODING 'UTF8'`);
          if (process.env.W4H_DEBUG) console.info(`[config] Database '${database}' created successfully`);
          databaseCreated = true;
        } catch (createError) {
          await adminPool.end();
          throw createError;
        }

        await adminPool.end();

        // Reconnect to the newly created database to verify
        const newTestPool = new Pool({
          host,
          port: parseInt(port),
          database,
          user,
          password: effectivePassword,
          connectionTimeoutMillis: 5000
        });

        try {
          await newTestPool.query('SELECT 1');
          await newTestPool.end();
        } catch (verifyError) {
          throw new Error(`Database created but cannot connect to it: ${verifyError.message}`);
        }
      } else {
        // Not a "database doesn't exist" error, so it's a connection problem
        throw new Error(`Cannot connect to database '${database}': ${connError.message}. Check that PostgreSQL is running, the host/port are correct, and the username/password are valid.`);
      }
    }

    if (!testPoolEnded) {
      await testPool.end();
    }

    // Generate TOKEN_SECRET if it doesn't exist
    if (!process.env.TOKEN_SECRET) {
      process.env.TOKEN_SECRET = crypto.randomBytes(64).toString('hex');
    }

    // Update or add database config
    const dbConfig = {
      DB_HOST: host,
      DB_PORT: port,
      DB_DATABASE: database,
      DB_USER: user,
      DB_PASSWORD: effectivePassword,
      TOKEN_SECRET: process.env.TOKEN_SECRET
    };

    // Connection successful — update `.env` in a stable, template-driven way
    await writeEnvValues(dbConfig);

    // Reset the database pool with new credentials
    await db.resetPool();

    // Initialize database schema (create tables if needed)
    const initResult = await initializeDatabase();
    if (!initResult.success) {
      console.error('Database initialization failed:', initResult.error);
      return res.status(500).json({
        success: false,
        error: 'Database configured but schema initialization failed',
        details: initResult.error
      });
    }

    let needsInitialAdmin = false;
    try {
      needsInitialAdmin = !(await hasAnyUsers());
    } catch (userCheckError) {
      console.warn('Unable to verify initial admin status:', userCheckError.message);
    }

    const message = databaseCreated 
      ? `Database '${database}' created and configured successfully.`
      : 'Database configuration saved and applied successfully.';

    res.json({
      success: true,
      message,
      databaseCreated,
      requiresRestart: false,
      initialized: initResult.success,
      requiresAdminSetup: needsInitialAdmin
    });

  } catch (error) {
    console.error('Database configuration error:', error);
    
    // Provide specific error messages based on error type
    let errorMessage = error.message || 'Unknown error';
    let userMessage = 'Failed to configure database';
    
    if (error.message?.includes('ECONNREFUSED')) {
      userMessage = 'Connection refused - PostgreSQL may not be running or accepting connections';
    } else if (error.message?.includes('ENOTFOUND')) {
      userMessage = 'Database host not found - check the hostname/IP address';
    } else if (error.message?.includes('ETIMEDOUT')) {
      userMessage = 'Connection timed out - network or firewall issue';
    } else if (error.message?.includes('does not have CREATEDB')) {
      userMessage = 'User does not have CREATEDB privilege - ask DBA to grant it';
    } else if (error.message?.includes('not found on PostgreSQL')) {
      userMessage = 'User does not exist on PostgreSQL server - check username';
    } else if (error.message?.includes('password authentication failed')) {
      userMessage = 'Incorrect password - check credentials';
    } else if (error.message?.includes('role')) {
      userMessage = 'Role/user error - check PostgreSQL user exists and has correct privileges';
    } else if (error.message?.includes('database')) {
      userMessage = 'Database error - ' + error.message;
    }
    
    res.status(500).json({
      success: false,
      error: userMessage,
      details: errorMessage,
      code: error.code
    });
  }
}

/** POST /config/admin-setup — Create initial admin account. Public only when no users exist yet. */
async function postAdminSetup(req, res) {
  const { email, password, name } = req.body;

  if (!email || !password) {
    return res.status(400).json({
      success: false,
      error: 'email and password are required'
    });
  }

  try {
    const usersExist = await hasAnyUsers();
    if (usersExist) {
      return res.status(400).json({
        success: false,
        error: 'Initial admin has already been configured'
      });
    }

    const adminResult = await createInitialAdmin({
      email: email.trim(),
      password,
      name: name ? name.trim() : undefined
    });

    if (!adminResult.success) {
      return res.status(400).json({
        success: false,
        error: adminResult.error
      });
    }

    const token = auth.sign({
      id: adminResult.user.id,
      email: adminResult.user.email,
      name: adminResult.user.name,
      role: 'super_admin',
      firstTimeLogin: true
    });

    let mailWarning = null;
    if (isMailerConfigured()) {
      try {
        await sendMail({
          to: adminResult.user.email,
          subject: 'Your W4H admin account is ready',
          text: `Hello ${adminResult.user.name},\n\nYour W4H admin account has been created successfully.\nYou can now sign in to complete setup.\n`,
          html: `<p>Hello ${adminResult.user.name},</p><p>Your <strong>W4H admin account</strong> has been created successfully.</p><p>You can now sign in to complete setup.</p>`
        });
      } catch (mailError) {
        mailWarning = mailError.message;
      }
    }

    res.json({
      success: true,
      message: 'Initial admin account created successfully',
      token,
      ...(mailWarning ? { mailWarning: `Admin created but email failed: ${mailWarning}` } : {}),
      user: {
        id: adminResult.user.id,
        email: adminResult.user.email,
        name: adminResult.user.name,
        role: 'super_admin',
        firstTimeLogin: true
      }
    });
  } catch (error) {
    console.error('Initial admin setup error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to create initial admin account',
      details: error.message
    });
  }
}

/** GET /config/test — Test database connection. Admin only. */
async function getTest(req, res) {
  if (!isAdmin(req.user)) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  if (!process.env.DB_HOST || !process.env.DB_PORT || !process.env.DB_DATABASE || 
      !process.env.DB_USER || !process.env.DB_PASSWORD) {
    return res.status(400).json({
      success: false,
      error: 'Database not configured'
    });
  }

  try {
    const testPool = new Pool({
      host: process.env.DB_HOST,
      port: parseInt(process.env.DB_PORT),
      database: process.env.DB_DATABASE,
      user: process.env.DB_USER,
      password: process.env.DB_PASSWORD,
      connectionTimeoutMillis: 5000
    });

    const result = await testPool.query('SELECT NOW() as current_time, version() as version');
    await testPool.end();

    res.json({
      success: true,
      message: 'Database connection successful',
      serverTime: result.rows[0].current_time,
      version: result.rows[0].version
    });

  } catch (error) {
    console.error('Database test error:', error);
    res.status(500).json({
      success: false,
      error: 'Database connection failed',
      details: error.message
    });
  }
}

/** POST /config/reset — Reset database connection pool. Admin only. */
async function postReset(req, res) {
  if (!isAdmin(req.user)) {
    return res.status(403).json({ error: 'Unauthorized' });
  }
  
  try {
    const success = await db.resetPool();
    
    if (success) {
      res.json({
        success: true,
        message: 'Database pool reset successfully'
      });
    } else {
      res.status(400).json({
        success: false,
        error: 'Database not configured'
      });
    }
  } catch (error) {
    console.error('Pool reset error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to reset pool',
      details: error.message
    });
  }
}
