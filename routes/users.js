import express from 'express';
const router = express.Router();
import bcrypt from 'bcrypt';
import crypto from 'crypto';
import { query } from '../db.js';
import * as auth from '../auth.js';
import jwt from 'jsonwebtoken';
import { validatePassword } from '../lib/utils/password-validator.js';
import { isMailerConfigured, sendMail } from '../lib/mailer.js';
import {
  USER_ROLES,
  USER_ROLES_NAMES,
  isSuperAdmin,
  isTeamAdmin,
  canManageUsers,
  isAssignablePlatformRole,
  rolesAssignableBy,
} from '../lib/utils/user.js';

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

function normalizeTeamIds(input) {
  const list = Array.isArray(input) ? input : (input == null || input === '' ? [] : [input]);
  return [...new Set(
    list
      .map(id => (id == null ? '' : String(id).trim()))
      .filter(Boolean)
  )];
}

async function getAllProjectIds() {
  const result = await query(`SELECT id FROM ${SCHEMA}.projects ORDER BY name ASC`);
  return result.rows.map(row => row.id);
}

async function getTeamIdsForUser(userId) {
  const result = await query(
    `SELECT team_id::text AS team_id FROM ${SCHEMA}.team_members WHERE user_id = $1`,
    [userId]
  );
  return result.rows.map(r => r.team_id);
}

async function syncUserTeams(userId, teamIds = []) {
  const ids = normalizeTeamIds(teamIds);
  await query(`DELETE FROM ${SCHEMA}.team_members WHERE user_id = $1`, [userId]);
  if (ids.length > 0) {
    await query(
      `INSERT INTO ${SCHEMA}.team_members (user_id, team_id)
       SELECT $1, UNNEST($2::uuid[])
       ON CONFLICT (team_id, user_id) DO NOTHING`,
      [userId, ids]
    );
  }
  return ids;
}

async function syncUserProjects(userId, role, requestedProjectIds = []) {
  const projectIds = isSuperAdmin(role)
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

async function buildUserSessionInfo(user) {
  const teamIds = await getTeamIdsForUser(user.id);
  return {
    id: user.id,
    name: user.name,
    email: user.email,
    role: user.role,
    team_ids: teamIds,
    team_id: teamIds[0] || null,
  };
}

function roles(req, res) {
  const allowed = rolesAssignableBy(req.user.role);
  if (!allowed.length) {
    return res.status(403).json({ error: 'Unauthorized' });
  }
  res.json(allowed.map(value => ({
    value,
    text: USER_ROLES_NAMES[value] || value,
  })));
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
  if (!isSuperAdmin(req.user.role)) {
    return res.status(403).json({ error: 'Unauthorized' });
  }
  try {
    const targetResult = await query(
      `SELECT id, role, email FROM ${SCHEMA}.${TABLE} WHERE email = $1`,
      [req.params.email]
    );

    if (targetResult.rowCount === 0) {
      return res.status(404).json({ error: 'User not found' });
    }

    if (isSuperAdmin(targetResult.rows[0].role)) {
      return res.status(400).json({ error: 'Super admin users cannot be deleted' });
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
  if (!isSuperAdmin(req.user.role)) {
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
  if (!canManageUsers(req.user.role)) {
    return res.status(403).json({ error: 'Unauthorized' });
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
    if (isSuperAdmin(existingUser.role) && !isSuperAdmin(user.role)) {
      return res.status(400).json({ error: 'Super admin role cannot be changed' });
    }

    const allowed = rolesAssignableBy(req.user.role);
    const requestedRole = user.role;
    if (requestedRole && !isSuperAdmin(existingUser.role) && !allowed.includes(requestedRole)) {
      return res.status(403).json({ error: 'Cannot assign this role' });
    }

    const effectiveRole = isSuperAdmin(existingUser.role) ? existingUser.role : (requestedRole || existingUser.role);
    
    if (user.password) {
      const passwordValidation = validatePassword(user.password);
      if (!passwordValidation.valid) {
        return res.status(400).json({ error: passwordValidation.message });
      }
    }
    
    const fields = [`name = $1`, `email = $2`, `role = $3`];
    const params = [user.name, user.email, effectiveRole];

    if (user.password) {
      const hashedPassword = bcrypt.hashSync(user.password, +process.env.BCRYPT_ROUNDS || 12);
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
      const userId = result.rows[0].id;
      const role = result.rows[0].role;
      const teamIds = normalizeTeamIds(user.team_ids ?? user.team_id);

      if (isTeamAdmin(req.user.role) && !isSuperAdmin(req.user.role)) {
        const myTeams = await getTeamIdsForUser(req.user.id);
        if (teamIds.length && !teamIds.every((id) => myTeams.includes(id))) {
          return res.status(403).json({ error: 'Can only assign users to your own team(s)' });
        }
      }

      if (isSuperAdmin(role)) {
        await syncUserTeams(userId, []);
        await ensureAdminAssignedToAllProjects(userId);
      } else if (Object.prototype.hasOwnProperty.call(user, 'team_ids')
        || Object.prototype.hasOwnProperty.call(user, 'team_id')) {
        if (!teamIds.length) {
          return res.status(400).json({ error: 'At least one team is required for this role' });
        }
        await syncUserTeams(userId, teamIds);
      }

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
  if (!canManageUsers(req.user.role)) {
    return res.status(403).json({ error: 'Unauthorized' });
  }
  try {
    const { user } = req.body;

    const email = (user.email || '').trim();
    const name = (user.name || email || '').trim();
    const role = user.role;
    const password = user.password;
    const teamIds = normalizeTeamIds(user.team_ids ?? user.team_id);

    if (!email || !role || !password) {
      return res.status(400).json({ error: 'email, role, and password are required' });
    }

    if (!isAssignablePlatformRole(role)) {
      return res.status(400).json({ error: 'Invalid role' });
    }

    const allowed = rolesAssignableBy(req.user.role);
    if (!allowed.includes(role)) {
      return res.status(403).json({ error: 'Cannot assign this role' });
    }

    if (role !== USER_ROLES.SUPER_ADMIN && teamIds.length === 0) {
      return res.status(400).json({ error: 'At least one team is required for this role' });
    }

    if (isTeamAdmin(req.user.role)) {
      const myTeams = await getTeamIdsForUser(req.user.id);
      if (!teamIds.length || !teamIds.every(id => myTeams.includes(id))) {
        return res.status(403).json({ error: 'Can only invite users to your own team(s)' });
      }
    }

    const passwordValidation = validatePassword(password);
    if (!passwordValidation.valid) {
      return res.status(400).json({ error: passwordValidation.message });
    }

    const hashedPassword = bcrypt.hashSync(password, +process.env.BCRYPT_ROUNDS || 12);

    const insertResult = await query(
      `INSERT INTO ${SCHEMA}.${TABLE} (name, email, password, role)
       VALUES ($1, $2, $3, $4)
       RETURNING id, role`,
      [name, email, hashedPassword, role]
    );

    const newUserId = insertResult.rows[0].id;
    await syncUserTeams(newUserId, teamIds);

    // Project membership is managed on the project, not at user create —
    // except Super admins who retain access to all projects.
    if (isSuperAdmin(role)) {
      await ensureAdminAssignedToAllProjects(newUserId);
    }

    const roleLabel = USER_ROLES_NAMES[role] || role;
    let mailWarning = null;
    if (isMailerConfigured()) {
      try {
        await sendMail({
          to: email,
          subject: 'Your W4H account is ready',
          text: `Hello ${name},\n\nAn account has been created for you in W4H with role: ${roleLabel}.\nYou can now sign in to the W4H portal.\n`,
          html: `<p>Hello ${name},</p><p>An account has been created for you in <strong>W4H</strong> with role: <strong>${roleLabel}</strong>.</p><p>You can now sign in to the W4H portal.</p>`
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

      const hashedPassword = bcrypt.hashSync(user.password, +process.env.BCRYPT_ROUNDS || 12);
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
  if (!canManageUsers(req.user.role)) {
    return res.status(403).json({ error: 'Unauthorized' });
  }
  try {
    const email = decodeURIComponent(String(req.params.email || ''));
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
         COALESCE((
           SELECT ARRAY_AGG(t.id::text ORDER BY t.name)
           FROM ${SCHEMA}.team_members tm
           JOIN ${SCHEMA}.teams t ON t.id = tm.team_id
           WHERE tm.user_id = u.id
         ), '{}') AS team_ids,
         COALESCE((
           SELECT STRING_AGG(t.name, ', ' ORDER BY t.name)
           FROM ${SCHEMA}.team_members tm
           JOIN ${SCHEMA}.teams t ON t.id = tm.team_id
           WHERE tm.user_id = u.id
         ), '') AS teams
       FROM ${SCHEMA}.${TABLE} u
       WHERE u.email = $1`,
      [email]
    );
    if (!result.rowCount) {
      return res.json(null);
    }

    const row = result.rows[0];
    if (isTeamAdmin(req.user.role) && !isSuperAdmin(req.user.role)) {
      const myTeams = await getTeamIdsForUser(req.user.id);
      const theirTeams = Array.isArray(row.team_ids) ? row.team_ids.map(String) : [];
      const overlap = theirTeams.some((id) => myTeams.includes(id));
      if (!overlap && row.id !== req.user.id) {
        return res.status(403).json({ error: 'Unauthorized' });
      }
    }

    res.json(row);
  } catch (error) {
    console.error('Get user error:', error);
    res.status(500).json({ error: error.message });
  }
}

async function list(req, res) {
  if (!canManageUsers(req.user.role)) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  try {
    const params = [];
    let where = '';

    if (isTeamAdmin(req.user.role) && !isSuperAdmin(req.user.role)) {
      const myTeams = await getTeamIdsForUser(req.user.id);
      if (!myTeams.length) {
        return res.json([]);
      }
      params.push(myTeams);
      where = `
        WHERE EXISTS (
          SELECT 1 FROM ${SCHEMA}.team_members tm
          WHERE tm.user_id = u.id AND tm.team_id = ANY($1::uuid[])
        )
      `;
    }

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
         COALESCE((
           SELECT ARRAY_AGG(t.id::text ORDER BY t.name)
           FROM ${SCHEMA}.team_members tm
           JOIN ${SCHEMA}.teams t ON t.id = tm.team_id
           WHERE tm.user_id = u.id
         ), '{}') AS team_ids,
         COALESCE((
           SELECT STRING_AGG(t.name, ', ' ORDER BY t.name)
           FROM ${SCHEMA}.team_members tm
           JOIN ${SCHEMA}.teams t ON t.id = tm.team_id
           WHERE tm.user_id = u.id
         ), '') AS teams,
         COALESCE((
           SELECT STRING_AGG(p.name, ', ' ORDER BY p.name)
           FROM ${SCHEMA}.project_members pm
           JOIN ${SCHEMA}.projects p ON p.id = pm.project_id
           WHERE pm.user_id = u.id
         ), '') AS projects
       FROM ${SCHEMA}.${TABLE} u
       ${where}
       ORDER BY u.name ASC NULLS LAST, u.email ASC`,
      params
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

    if (isSuperAdmin(user.role)) {
      await ensureAdminAssignedToAllProjects(user.id);
    }

    const userInfo = await buildUserSessionInfo(user);
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

    if (isSuperAdmin(user.role)) {
      await ensureAdminAssignedToAllProjects(user.id);
    }

    const userInfo = await buildUserSessionInfo(user);

    const token = auth.sign(userInfo);
    userInfo.token = token;

    res.json(userInfo);
  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({ error: error.message });
  }
}

export default router;