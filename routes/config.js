import express from 'express';
const router = express.Router();
import fs from 'fs-extra';
import path from 'path';
import { fileURLToPath } from 'url';
import crypto from 'crypto';
import dayjs from 'dayjs';
import pkg from 'pg';
const { Pool } = pkg;
import db from '../db.js';
import * as auth from '../auth.js';
import { initializeDatabase, createInitialAdmin, hasAnyUsers } from '../lib/db-init.js';
import { getMailerConfig, isMailerConfigured, sendMail } from '../lib/mailer.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function sqlString(value) {
  if (value === null || value === undefined) return 'NULL';
  return `'${String(value).replace(/'/g, "''")}'`;
}

function sqlBool(value) {
  if (value === null || value === undefined) return 'NULL';
  return value ? 'true' : 'false';
}

function sqlNumber(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return 'NULL';
  return String(value);
}

async function writeEnvValues(keyValues) {
  const envPath = path.join(__dirname, '..', '.env');
  let envContent = '';

  if (await fs.pathExists(envPath)) {
    envContent = await fs.readFile(envPath, 'utf8');
  }

  Object.keys(keyValues).forEach((key) => {
    const value = keyValues[key] ?? '';
    const regex = new RegExp(`^${key}=.*$`, 'm');
    if (regex.test(envContent)) {
      envContent = envContent.replace(regex, `${key}=${value}`);
    } else {
      envContent += `\n${key}=${value}`;
    }
    process.env[key] = String(value);
  });

  await fs.writeFile(envPath, envContent.trim() + '\n');
}

function requireAdminIfConfigured(req, res, next) {
  if (!db.isConfigured()) {
    return next();
  }

  return auth.authenticate(req, res, () => auth.requireAdmin(req, res, next));
}

/**
 * GET /config/status
 * Check if database is configured
 * Public endpoint - no auth required
 */
router.get('/status', (req, res) => {
  const isConfigured = !!(
    process.env.DB_HOST &&
    process.env.DB_PORT &&
    process.env.DB_DATABASE &&
    process.env.DB_USER &&
    process.env.DB_PASSWORD
  );

  res.json({
    configured: isConfigured,
    details: isConfigured ? {
      host: process.env.DB_HOST,
      port: process.env.DB_PORT,
      database: process.env.DB_DATABASE,
      user: process.env.DB_USER
    } : null
  });
});

/**
 * GET /config/database-settings
 * Get current database settings (even if not fully configured)
 * Public endpoint - no auth required
 */
router.get('/database-settings', (req, res) => {
  res.json({
    host: process.env.DB_HOST || '',
    port: process.env.DB_PORT || '5432',
    database: process.env.DB_DATABASE || '',
    user: process.env.DB_USER || ''
  });
});

/**
 * GET /config/mailer-settings
 * Get current SMTP settings (without returning password)
 * ADMIN ONLY
 */
router.get('/mailer-settings', requireAdminIfConfigured, (req, res) => {
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
});

/**
 * POST /config/mailer-settings
 * Save SMTP settings to .env
 * ADMIN ONLY
 */
router.post('/mailer-settings', requireAdminIfConfigured, async (req, res) => {
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
});

/**
 * POST /config/mailer-test
 * Send test email to verify SMTP settings
 * ADMIN ONLY
 */
router.post('/mailer-test', requireAdminIfConfigured, async (req, res) => {
  const to = String(req.body?.to || '').trim();
  if (!to) {
    return res.status(400).json({ success: false, error: 'Test recipient email is required' });
  }

  try {
    await sendMail({
      to,
      subject: 'W4H SMTP Test',
      text: 'SMTP test email from W4H settings.',
      html: '<p>SMTP test email from <strong>W4H</strong> settings.</p>'
    });

    return res.json({ success: true, message: `Test email sent to ${to}` });
  } catch (error) {
    return res.status(500).json({
      success: false,
      error: 'SMTP test failed',
      details: error.message
    });
  }
});

/**
 * POST /config/database
 * Configure database credentials
 * Body: { host, port, database, user, password }
 * ADMIN ONLY (after initial setup)
 */
router.post('/database', requireAdminIfConfigured, async (req, res) => {
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
      console.log(`Connection error code: ${connError.code}, message: ${connError.message}`);
      
      // If database doesn't exist, try to create it
      if (connError.code === '3D000' || connError.message?.includes('does not exist')) { 
        console.log(`Database '${database}' does not exist. Attempting to create it...`);
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
          console.log(`Database '${database}' created successfully`);
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

    // Connection successful, update .env file
    const envPath = path.join(__dirname, '..', '.env');
    let envContent = '';

    if (await fs.pathExists(envPath)) {
      envContent = await fs.readFile(envPath, 'utf8');
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

    Object.keys(dbConfig).forEach(key => {
      const regex = new RegExp(`^${key}=.*$`, 'm');
      if (regex.test(envContent)) {
        envContent = envContent.replace(regex, `${key}=${dbConfig[key]}`);
      } else {
        envContent += `\n${key}=${dbConfig[key]}`;
      }
    });

    await fs.writeFile(envPath, envContent.trim() + '\n');

    // Update process.env
    Object.keys(dbConfig).forEach(key => {
      process.env[key] = dbConfig[key];
    });

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
});

/**
 * POST /config/admin-setup
 * Create initial admin account after database setup
 * Public endpoint only when no users exist yet
 */
router.post('/admin-setup', async (req, res) => {
  const { email, password, name } = req.body;

  if (!email || !password) {
    return res.status(400).json({
      success: false,
      error: 'email and password are required'
    });
  }

  try {
    const initResult = await initializeDatabase();
    if (!initResult.success) {
      return res.status(500).json({
        success: false,
        error: 'Failed to initialize database schema',
        details: initResult.error
      });
    }

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
      email: adminResult.user.email,
      role: 'admin',
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
        email: adminResult.user.email,
        name: adminResult.user.name,
        role: 'admin',
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
});

/**
 * GET /config/test
 * Test database connection with current config
 * ADMIN ONLY (after initial setup)
 */
router.get('/test', requireAdminIfConfigured, async (req, res) => {
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
});

/**
 * POST /config/reset
 * Reset database connection pool (useful after manual .env changes)
 * ADMIN ONLY (after initial setup)
 */
router.post('/reset', requireAdminIfConfigured, async (req, res) => {
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
});

/**
 * GET /config/docs-backup
 * Generate docs SQL backup and return filename + SQL payload
 * ADMIN ONLY (after initial setup)
 */
router.get('/docs-backup', requireAdminIfConfigured, async (req, res) => {
  try {
    const result = await db.query(
      `SELECT
        slug,
        title,
        card_title,
        description,
        action_label,
        doc_type,
        sort_order,
        hidden,
        is_feature_card,
        show_in_docs,
        card_order,
        md,
        html,
        toc,
        updated_at
       FROM w4h.docs
       ORDER BY sort_order ASC NULLS LAST, slug ASC`
    );

    const header = [
      '-- Docs backup generated from current database content',
      '-- Generated by GET /config/docs-backup',
      '-- This script is restore-ready: creates schema/table and replaces docs data',
      ''
    ].join('\n');

    const valuesSql = result.rows
      .map((doc) => {
        const tocJson = JSON.stringify(doc.toc ?? []);
        return `  (${[
          sqlString(doc.slug),
          sqlString(doc.title),
          sqlString(doc.card_title),
          sqlString(doc.description),
          sqlString(doc.action_label),
          sqlString(doc.doc_type),
          sqlNumber(doc.sort_order),
          sqlBool(doc.hidden),
          sqlBool(doc.is_feature_card),
          sqlBool(doc.show_in_docs),
          sqlNumber(doc.card_order),
          sqlString(doc.md),
          sqlString(doc.html),
          `${sqlString(tocJson)}::jsonb`,
          'CURRENT_TIMESTAMP'
        ].join(', ')})`;
      })
      .join(',\n');

    const insertSql = valuesSql
      ? `INSERT INTO w4h.docs (slug, title, card_title, description, action_label, doc_type, sort_order, hidden, is_feature_card, show_in_docs, card_order, md, html, toc, updated_at)\nVALUES\n${valuesSql};\n`
      : '-- No docs rows found to insert.\n';

    const sql = `${header}BEGIN;\n\nCREATE SCHEMA IF NOT EXISTS w4h;\nCREATE EXTENSION IF NOT EXISTS pgcrypto;\n\nCREATE TABLE IF NOT EXISTS w4h.docs (\n  id uuid NOT NULL DEFAULT gen_random_uuid(),\n  slug text COLLATE pg_catalog."default" NOT NULL,\n  title text COLLATE pg_catalog."default",\n  card_title text COLLATE pg_catalog."default",\n  description text COLLATE pg_catalog."default",\n  action_label text COLLATE pg_catalog."default",\n  doc_type text COLLATE pg_catalog."default",\n  sort_order integer,\n  hidden boolean,\n  updated_at timestamp without time zone,\n  is_feature_card boolean,\n  show_in_docs boolean,\n  md text COLLATE pg_catalog."default",\n  html text COLLATE pg_catalog."default",\n  toc jsonb,\n  card_order integer,\n  CONSTRAINT docs_pkey PRIMARY KEY (id),\n  CONSTRAINT docs_slug_key UNIQUE (slug)\n);\n\nCREATE INDEX IF NOT EXISTS idx_docs_slug ON w4h.docs(slug);\n\nTRUNCATE TABLE w4h.docs;\n\n${insertSql}\nCOMMIT;\n`;

    const ts = dayjs().format('YYYY-MM-DD_HH-mm-ss');
    const filename = `docs-backup-${ts}.sql`;

    res.json({
      success: true,
      filename,
      sql
    });
  } catch (error) {
    console.error('Docs backup generation error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to generate docs backup',
      details: error.message
    });
  }
});

export default router;
