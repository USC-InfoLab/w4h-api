import express from 'express';
const router = express.Router();
import bcrypt from 'bcrypt';
import crypto from 'crypto';
import { query } from '../db.js';
import * as auth from '../auth.js';
import jwt from 'jsonwebtoken';
import { validatePassword } from '../lib/utils/password-validator.js';
import { isMailerConfigured, sendMail } from '../lib/mailer.js';

const SCHEMA = 'w4h';
const TABLE = 'users';

router.post('/login', login);
router.post('/send-code', sendLoginCode);
router.post('/verify-code', verifyLoginCode);
router.post('/token-valid', tokenValid);
router.get('/me/api-key', auth.authenticate, getMyApiKeyMeta);
router.post('/me/api-key', auth.authenticate, rotateMyApiKey);
router.delete('/me/api-key', auth.authenticate, revokeMyApiKey);
router.delete('/user/:email/api-key', auth.authenticate, revokeUserApiKeyByEmail);
router.get('/', auth.authenticate, list);
router.get('/user/:email', auth.authenticate, getByEmail);
router.post('/profile', auth.authenticate, updateProfile);
router.post('/add', auth.authenticate, addUser);
router.post('/set', auth.authenticate, setUser);
router.delete('/:email', auth.authenticate, del);
router.get('/roles', auth.authenticate, roles);


function pickProjectIds(user) {
  if (!user || typeof user !== 'object') return [];
  return normalizeProjectIds(user.project_ids ?? user.study_ids);
}

function normalizeProjectIds(input) {
  if (!Array.isArray(input)) {
    return [];
  }

  return [...new Set(
    input
      .map(id => (id == null ? '' : String(id).trim()))
      .filter(Boolean)
  )];
}

async function getAllProjectIds() {
  const result = await query(`SELECT id FROM ${SCHEMA}.projects ORDER BY name ASC`);
  return result.rows.map(row => row.id);
}

async function syncUserProjects(userId, role, requestedProjectIds = []) {
  const projectIds = role === 'admin'
    ? await getAllProjectIds()
    : normalizeProjectIds(requestedProjectIds);

  await query(`DELETE FROM ${SCHEMA}.user_projects WHERE user_id = $1`, [userId]);

  if (projectIds.length > 0) {
    await query(
      `INSERT INTO ${SCHEMA}.user_projects (user_id, project_id)
       SELECT $1, UNNEST($2::text[])
       ON CONFLICT (user_id, project_id) DO NOTHING`,
      [userId, projectIds]
    );
  }

  return projectIds;
}

async function ensureAdminAssignedToAllProjects(userId) {
  await query(
    `INSERT INTO ${SCHEMA}.user_projects (user_id, project_id)
     SELECT $1, s.id
     FROM ${SCHEMA}.projects s
     ON CONFLICT (user_id, project_id) DO NOTHING`,
    [userId]
  );
}

function roles(req, res) {
  if (req.user.role !== 'admin') {
    return res.json(null);
  }
  res.json(['clinician', 'researcher', 'admin']);
}

function tokenValid(req, res) {
  try {
    const { token } = req.body;
    if (!token) {
      return res.json({ tokenExpired: true });
    }
    
    jwt.verify(token, process.env.TOKEN_SECRET);
    res.json({ tokenExpired: false });
  } catch (error) {
    res.json({ tokenExpired: true });
  }
}

async function del(req, res) {
  if (req.user.role !== 'admin') {
    return res.json(null);
  }
  try {
    const targetResult = await query(
      `SELECT id, role, email FROM ${SCHEMA}.${TABLE} WHERE email = $1`,
      [req.params.email]
    );

    if (targetResult.rowCount === 0) {
      return res.status(404).json({ error: 'User not found' });
    }

    if (targetResult.rows[0].role === 'admin') {
      return res.status(400).json({ error: 'Admin users cannot be deleted' });
    }

    await query(`DELETE FROM ${SCHEMA}.${TABLE} WHERE email = $1`, [req.params.email]);
    res.json({ success: true, message: 'User deleted' });
  } catch (error) {
    console.error('Delete user error:', error);
    res.status(500).json({ error: error.message });
  }
}

async function getMyApiKeyMeta(req, res) {
  try {
    const result = await query(
      `SELECT api_key_prefix, api_key_created_at, (api_key_hash IS NOT NULL) AS api_key_active
       FROM ${SCHEMA}.${TABLE} WHERE id = $1`,
      [req.user.id]
    );
    if (!result.rowCount) {
      return res.status(404).json({ error: 'User not found' });
    }
    const row = result.rows[0];
    res.json({
      active: row.api_key_active,
      api_key_prefix: row.api_key_prefix,
      api_key_created_at: row.api_key_created_at
    });
  } catch (error) {
    console.error('getMyApiKeyMeta:', error);
    res.status(500).json({ error: error.message });
  }
}

async function rotateMyApiKey(req, res) {
  try {
    const raw = `${auth.USER_API_KEY_PREFIX}${crypto.randomBytes(32).toString('base64url')}`;
    const tokenHash = crypto.createHash('sha256').update(raw).digest('hex');
    const prefix = raw.slice(0, 24);
    const result = await query(
      `UPDATE ${SCHEMA}.${TABLE}
       SET api_key_hash = $1, api_key_prefix = $2, api_key_created_at = NOW()
       WHERE id = $3
       RETURNING api_key_created_at`,
      [tokenHash, prefix, req.user.id]
    );
    if (!result.rowCount) {
      return res.status(404).json({ error: 'User not found' });
    }
    res.status(201).json({
      api_key: raw,
      api_key_prefix: prefix,
      api_key_created_at: result.rows[0].api_key_created_at
    });
  } catch (error) {
    console.error('rotateMyApiKey:', error);
    res.status(500).json({ error: error.message });
  }
}

async function revokeMyApiKey(req, res) {
  try {
    await query(
      `UPDATE ${SCHEMA}.${TABLE}
       SET api_key_hash = NULL, api_key_prefix = NULL, api_key_created_at = NULL
       WHERE id = $1`,
      [req.user.id]
    );
    res.json({ success: true });
  } catch (error) {
    console.error('revokeMyApiKey:', error);
    res.status(500).json({ error: error.message });
  }
}

async function revokeUserApiKeyByEmail(req, res) {
  if (req.user.role !== 'admin') {
    return res.status(403).json({ error: 'Admin privileges required' });
  }
  try {
    const rawEmail = Array.isArray(req.params.email) ? req.params.email[0] : req.params.email;
    const email = decodeURIComponent(String(rawEmail || '').trim());
    if (!email) {
      return res.status(400).json({ error: 'Email is required' });
    }
    const result = await query(
      `UPDATE ${SCHEMA}.${TABLE}
       SET api_key_hash = NULL, api_key_prefix = NULL, api_key_created_at = NULL
       WHERE email = $1
       RETURNING id`,
      [email]
    );
    if (!result.rowCount) {
      return res.status(404).json({ error: 'User not found' });
    }
    res.json({ success: true });
  } catch (error) {
    console.error('revokeUserApiKeyByEmail:', error);
    res.status(500).json({ error: error.message });
  }
}

async function setUser(req, res) {
  if (req.user.role !== 'admin') {
    return res.json(null);
  }
  try {
    const { user } = req.body;
    const targetId = user.id || null;
    const targetEmail = (user.original_email || user.email || '').trim();

    if (!targetId && !targetEmail) {
      return res.status(400).json({ error: 'User id or email is required' });
    }

    const identifyValue = targetId || targetEmail;
    const identifyClause = targetId ? 'id = $1' : 'email = $1';
    const existingResult = await query(
      `SELECT id, role FROM ${SCHEMA}.${TABLE} WHERE ${identifyClause}`,
      [identifyValue]
    );

    if (existingResult.rowCount === 0) {
      return res.status(404).json({ error: 'User not found' });
    }

    const existingUser = existingResult.rows[0];
    if (existingUser.role === 'admin' && user.role !== 'admin') {
      return res.status(400).json({ error: 'Admin role cannot be changed' });
    }

    const effectiveRole = existingUser.role === 'admin' ? 'admin' : user.role;
    
    if (user.password) {
      const passwordValidation = validatePassword(user.password);
      if (!passwordValidation.valid) {
        return res.status(400).json({ error: passwordValidation.message });
      }
    }
    
    const fields = [`name = $1`, `email = $2`, `role = $3`];
    const params = [user.name, user.email, effectiveRole];

    if (user.password) {
      const hashedPassword = bcrypt.hashSync(user.password, process.env.BCRYPT_ROUNDS);
      params.push(hashedPassword);
      fields.push(`password = $${params.length}`);
    }

    params.push(targetId || targetEmail);
    const whereClause = targetId ? `id = $${params.length}` : `email = $${params.length}`;

    const result = await query(
      `UPDATE ${SCHEMA}.${TABLE}
       SET ${fields.join(', ')}
       WHERE ${whereClause}
       RETURNING id, role`,
      params
    );

    if (result.rowCount > 0) {
      await syncUserProjects(result.rows[0].id, result.rows[0].role, pickProjectIds(user));
      res.json({ success: true, message: 'User updated' });
    } else {
      res.json({ error: 'User not found' });
    }
  } catch (error) {
    console.error('Set user error:', error);
    res.status(500).json({ error: error.message });
  }
}

async function addUser(req, res) {
  if (req.user.role !== 'admin') {
    return res.json(null);
  }
  try {
    const { user } = req.body;

    const email = (user.email || '').trim();
    const name = (user.name || email || '').trim();
    const role = user.role;
    const password = user.password;

    if (!email || !role || !password) {
      return res.status(400).json({ error: 'email, role, and password are required' });
    }

    const passwordValidation = validatePassword(password);
    if (!passwordValidation.valid) {
      return res.status(400).json({ error: passwordValidation.message });
    }

    const hashedPassword = bcrypt.hashSync(password, process.env.BCRYPT_ROUNDS);

    const insertResult = await query(
      `INSERT INTO ${SCHEMA}.${TABLE} (name, email, password, role)
       VALUES ($1, $2, $3, $4)
       RETURNING id, role`,
      [name, email, hashedPassword, role]
    );

    await syncUserProjects(insertResult.rows[0].id, insertResult.rows[0].role, pickProjectIds(user));

    let mailWarning = null;
    if (isMailerConfigured()) {
      try {
        await sendMail({
          to: email,
          subject: 'Your W4H account is ready',
          text: `Hello ${name},\n\nAn account has been created for you in W4H with role: ${role}.\nYou can now sign in to the W4H portal.\n`,
          html: `<p>Hello ${name},</p><p>An account has been created for you in <strong>W4H</strong> with role: <strong>${role}</strong>.</p><p>You can now sign in to the W4H portal.</p>`
        });
      } catch (mailError) {
        mailWarning = mailError.message;
      }
    }

    res.status(201).json({
      success: true,
      message: 'User created',
      ...(mailWarning ? { mailWarning: `User created but email failed: ${mailWarning}` } : {})
    });
  } catch (error) {
    if (error.code === '23505') {
      return res.status(409).json({ error: 'Email already exists' });
    }

    console.error('Add user error:', error);
    res.status(500).json({ error: error.message });
  }
}

async function updateProfile(req, res) {
  try {
    const { user } = req.body;
    const updates = [];
    const params = [];

    if (user?.name !== undefined) {
      params.push(user.name);
      updates.push(`name = $${params.length}`);
    }

    if (user?.password) {
      const passwordValidation = validatePassword(user.password);
      if (!passwordValidation.valid) {
        return res.status(400).json({ error: passwordValidation.message });
      }

      const hashedPassword = bcrypt.hashSync(user.password, process.env.BCRYPT_ROUNDS);
      params.push(hashedPassword);
      updates.push(`password = $${params.length}`);
    }

    if (updates.length === 0) {
      return res.status(400).json({ error: 'No profile changes provided' });
    }

    params.push(req.user.id);
    const result = await query(
      `UPDATE ${SCHEMA}.${TABLE}
       SET ${updates.join(', ')}
       WHERE id = $${params.length}
       RETURNING id, name, email, role`,
      params
    );

    if (result.rowCount === 0) {
      return res.status(404).json({ error: 'User not found' });
    }

    const updatedUser = result.rows[0];
    const token = auth.sign(updatedUser);
    return res.json({ success: true, user: { ...updatedUser, token } });
  } catch (error) {
    console.error('Update profile error:', error);
    return res.status(500).json({ error: error.message });
  }
}

async function getByEmail(req, res) {
  if (req.user.role !== 'admin') {
    return res.json(null);
  }
  try {
    const result = await query(
      `SELECT
         u.id,
         u.name,
         u.email,
         u.role,
         u.created_at,
         u.api_key_prefix,
         u.api_key_created_at,
         (u.api_key_hash IS NOT NULL) AS api_key_active,
         CASE
           WHEN u.role = 'admin' THEN COALESCE(
             (SELECT ARRAY_AGG(s.id ORDER BY s.name) FROM ${SCHEMA}.projects s),
             '{}'
           )
           ELSE COALESCE(
             (
               SELECT ARRAY_AGG(us.project_id ORDER BY s.name)
               FROM ${SCHEMA}.user_projects us
               LEFT JOIN ${SCHEMA}.projects s ON s.id = us.project_id
               WHERE us.user_id = u.id
             ),
             '{}'
           )
         END AS project_ids,
         CASE
           WHEN u.role = 'admin' THEN COALESCE(
             (SELECT STRING_AGG(s.name, ', ' ORDER BY s.name) FROM ${SCHEMA}.projects s),
             ''
           )
           ELSE COALESCE(
             (
               SELECT STRING_AGG(s.name, ', ' ORDER BY s.name)
               FROM ${SCHEMA}.user_projects us
               LEFT JOIN ${SCHEMA}.projects s ON s.id = us.project_id
               WHERE us.user_id = u.id
             ),
             ''
           )
         END AS projects
       FROM ${SCHEMA}.${TABLE} u
       WHERE u.email = $1`,
      [decodeURIComponent(String(req.params.email || ''))]
    );
    if (result.rows.length > 0) {
      res.json(result.rows[0]);
    } else {
      res.json(null);
    }
  } catch (error) {
    console.error('Get user error:', error);
    res.status(500).json({ error: error.message });
  }
}

async function list(req, res) {
  if (req.user.role !== 'admin') {
    return res.json(null);
  }

  try {
    const result = await query(
      `SELECT
         u.id,
         u.name,
         u.email,
         u.role,
         u.created_at,
         u.api_key_prefix,
         u.api_key_created_at,
         (u.api_key_hash IS NOT NULL) AS api_key_active,
         CASE
           WHEN u.role = 'admin' THEN COALESCE(
             (SELECT ARRAY_AGG(s.id ORDER BY s.name) FROM ${SCHEMA}.projects s),
             '{}'
           )
           ELSE COALESCE(
             (
               SELECT ARRAY_AGG(us.project_id ORDER BY s.name)
               FROM ${SCHEMA}.user_projects us
               LEFT JOIN ${SCHEMA}.projects s ON s.id = us.project_id
               WHERE us.user_id = u.id
             ),
             '{}'
           )
         END AS project_ids,
         CASE
           WHEN u.role = 'admin' THEN COALESCE(
             (SELECT STRING_AGG(s.name, ', ' ORDER BY s.name) FROM ${SCHEMA}.projects s),
             ''
           )
           ELSE COALESCE(
             (
               SELECT STRING_AGG(s.name, ', ' ORDER BY s.name)
               FROM ${SCHEMA}.user_projects us
               LEFT JOIN ${SCHEMA}.projects s ON s.id = us.project_id
               WHERE us.user_id = u.id
             ),
             ''
           )
         END AS projects
       FROM ${SCHEMA}.${TABLE} u
       ORDER BY u.email ASC`
    );

    res.json(result.rows);
  } catch (error) {
    console.error('List users error:', error);
    res.status(500).json({ error: error.message });
  }
}

const LOGIN_CODE_TTL_MINUTES = 10;
const LOGIN_CODE_MAX_SENDS = 3;
const LOGIN_CODE_SEND_WINDOW_MINUTES = 15;
const LOGIN_CODE_MAX_ATTEMPTS = 5;

function hashLoginCode(code) {
  return crypto.createHash('sha256').update(String(code).trim()).digest('hex');
}

function generateLoginCode() {
  return String(crypto.randomInt(0, 1_000_000)).padStart(6, '0');
}

async function writeAuthAudit(req, { userId, action, details }) {
  try {
    await query(
      `INSERT INTO ${SCHEMA}.audit_log (user_id, action, resource_type, ip_address, user_agent, details)
       VALUES ($1, $2, 'user', $3, $4, $5)`,
      [
        userId ?? null,
        action,
        req.ip || null,
        req.get('user-agent') || null,
        details ? JSON.stringify(details) : null
      ]
    );
  } catch (err) {
    console.warn('Auth audit log write failed:', err.message);
  }
}

async function sendLoginCode(req, res) {
  try {
    const email = (req.body.email || '').trim().toLowerCase();
    if (!email) {
      return res.status(400).json({ error: { message: 'Email is required.' } });
    }

    if (!isMailerConfigured()) {
      return res.status(503).json({
        error: {
          message: 'Email is not configured. Ask your administrator to set up SMTP, or sign in with your password.'
        }
      });
    }

    const userResult = await query(
      `SELECT id, name, email FROM ${SCHEMA}.${TABLE} WHERE lower(trim(email)) = $1`,
      [email]
    );

    if (userResult.rowCount === 0) {
      return res.status(404).json({ error: { message: 'No account found for that email address.' } });
    }

    const user = userResult.rows[0];

    const recentSends = await query(
      `SELECT COUNT(*)::int AS count
       FROM ${SCHEMA}.login_codes
       WHERE user_id = $1
         AND created_at > now() - ($2 || ' minutes')::interval`,
      [user.id, String(LOGIN_CODE_SEND_WINDOW_MINUTES)]
    );

    if (recentSends.rows[0].count >= LOGIN_CODE_MAX_SENDS) {
      return res.status(429).json({
        error: { message: 'Too many code requests. Please wait a few minutes and try again.' }
      });
    }

    await query(
      `DELETE FROM ${SCHEMA}.login_codes
       WHERE user_id = $1 AND (used_at IS NOT NULL OR expires_at < now())`,
      [user.id]
    );

    const code = generateLoginCode();
    const codeHash = hashLoginCode(code);

    await query(
      `INSERT INTO ${SCHEMA}.login_codes (user_id, code_hash, expires_at)
       VALUES ($1, $2, now() + ($3 || ' minutes')::interval)`,
      [user.id, codeHash, String(LOGIN_CODE_TTL_MINUTES)]
    );

    try {
      await sendMail({
        to: user.email,
        subject: 'Your W4H sign-in code',
        text: `Hello ${user.name || user.email},\n\nYour W4H sign-in code is: ${code}\n\nThis code expires in ${LOGIN_CODE_TTL_MINUTES} minutes. If you did not request it, ignore this email.\n`,
        html: `<p>Hello ${user.name || user.email},</p><p>Your W4H sign-in code is:</p><p style="font-size:1.5em;font-weight:bold;letter-spacing:0.15em">${code}</p><p>This code expires in ${LOGIN_CODE_TTL_MINUTES} minutes. If you did not request it, you can ignore this email.</p>`
      });
    } catch (mailError) {
      console.error('sendLoginCode mail error:', mailError);
      return res.status(502).json({
        error: { message: 'Could not send email. Check SMTP settings or try signing in with your password.' }
      });
    }

    await writeAuthAudit(req, {
      userId: user.id,
      action: 'user.login_code_sent',
      details: { email: user.email }
    });

    res.json({
      success: true,
      message: `A sign-in code was sent to ${user.email}. It expires in ${LOGIN_CODE_TTL_MINUTES} minutes.`
    });
  } catch (error) {
    console.error('sendLoginCode error:', error);
    res.status(500).json({ error: { message: 'Failed to send sign-in code.' } });
  }
}

async function verifyLoginCode(req, res) {
  try {
    const email = (req.body.email || '').trim().toLowerCase();
    const code = String(req.body.code || '').trim();

    if (!email || !code) {
      return res.status(400).json({ error: { message: 'Email and verification code are required.' } });
    }

    if (!/^\d{6}$/.test(code)) {
      return res.status(400).json({ error: { message: 'Enter the 6-digit code from your email.' } });
    }

    const userResult = await query(
      `SELECT id, name, email, role FROM ${SCHEMA}.${TABLE} WHERE lower(trim(email)) = $1`,
      [email]
    );

    if (userResult.rowCount === 0) {
      return res.status(401).json({ error: { message: 'Invalid email or verification code.' } });
    }

    const user = userResult.rows[0];
    const codeHash = hashLoginCode(code);

    const codeResult = await query(
      `SELECT id, attempts
       FROM ${SCHEMA}.login_codes
       WHERE user_id = $1
         AND code_hash = $2
         AND used_at IS NULL
         AND expires_at > now()
       ORDER BY created_at DESC
       LIMIT 1`,
      [user.id, codeHash]
    );

    if (codeResult.rowCount === 0) {
      await query(
        `UPDATE ${SCHEMA}.login_codes
         SET attempts = attempts + 1
         WHERE user_id = $1 AND used_at IS NULL AND expires_at > now()`,
        [user.id]
      );
      await writeAuthAudit(req, {
        userId: user.id,
        action: 'user.login_code_failed',
        details: { email: user.email, reason: 'invalid_code' }
      });
      return res.status(401).json({ error: { message: 'Invalid or expired verification code.' } });
    }

    const codeRow = codeResult.rows[0];
    if (codeRow.attempts >= LOGIN_CODE_MAX_ATTEMPTS) {
      await query(`UPDATE ${SCHEMA}.login_codes SET used_at = now() WHERE id = $1`, [codeRow.id]);
      return res.status(429).json({
        error: { message: 'Too many failed attempts. Request a new code.' }
      });
    }

    await query(`UPDATE ${SCHEMA}.login_codes SET used_at = now() WHERE id = $1`, [codeRow.id]);

    if (user.role === 'admin') {
      await ensureAdminAssignedToAllProjects(user.id);
    }

    const userInfo = {
      id: user.id,
      name: user.name,
      email: user.email,
      role: user.role
    };
    const token = auth.sign(userInfo);

    await writeAuthAudit(req, {
      userId: user.id,
      action: 'user.login_code_success',
      details: { email: user.email }
    });

    res.json({ user: userInfo, token });
  } catch (error) {
    console.error('verifyLoginCode error:', error);
    res.status(500).json({ error: { message: 'Login failed.' } });
  }
}

async function login(req, res) {
  try {
    const email = (req.body.email || req.body.username || '').trim();
    const { password } = req.body;
    
    if (!email || !password) {
      return res.json({ error: { message: 'Email and password are required' } });
    }

    const result = await query(
      `SELECT id, name, email, password, role FROM ${SCHEMA}.${TABLE} WHERE email = $1`,
      [email]
    );

    if (result.rows.length === 0) {
      return res.json({ error: { message: 'Invalid Email or Password!' } });
    }

    const user = result.rows[0];
    
    if (!bcrypt.compareSync(password, user.password)) {
      return res.json({ error: { message: 'Invalid Email or Password!' } });
    }

    if (user.role === 'admin') {
      await ensureAdminAssignedToAllProjects(user.id);
    }

    const userInfo = {
      id: user.id,
      name: user.name,
      email: user.email,
      role: user.role
    };

    const token = auth.sign(userInfo);
    userInfo.token = token;

    res.json(userInfo);
  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({ error: error.message });
  }
}

export default router;