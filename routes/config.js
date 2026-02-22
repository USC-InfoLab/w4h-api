import express from 'express';
const router = express.Router();
import fs from 'fs-extra';
import path from 'path';
import crypto from 'crypto';
import pkg from 'pg';
const { Pool } = pkg;
import db from '../db.js';
import * as auth from '../auth.js';
import { initializeDatabase, createInitialAdmin, hasAnyUsers } from '../lib/db-init.js';

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
 * POST /config/database
 * Configure database credentials
 * Body: { host, port, database, user, password }
 * ADMIN ONLY (after initial setup)
 */
router.post('/database', requireAdminIfConfigured, async (req, res) => {
  const { host, port, database, user, password } = req.body;

  // Validate required fields
  if (!host || !port || !database || !user || !password) {
    return res.status(400).json({
      success: false,
      error: 'All database fields are required: host, port, database, user, password'
    });
  }

  try {
    // Test the connection first
    const testPool = new Pool({
      host,
      port: parseInt(port),
      database,
      user,
      password,
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
          password,
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
          password,
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
      DB_PASSWORD: password,
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
      console.warn('Database initialization warning:', initResult.error);
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
  const { username, email, password, name } = req.body;

  if (!username || !email || !password) {
    return res.status(400).json({
      success: false,
      error: 'username, email, and password are required'
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
      username: username.trim(),
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
      username: adminResult.user.username,
      role: 'admin',
      firstTimeLogin: true
    });

    res.json({
      success: true,
      message: 'Initial admin account created successfully',
      token,
      user: {
        username: adminResult.user.username,
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

export default router;
