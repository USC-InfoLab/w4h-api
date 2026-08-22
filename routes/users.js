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
  appBaseUrl,
  createInvitation,
  hashInviteToken,
  inviteAcceptUrl,
  sendInviteEmail,
} from '../lib/invitations.js';
import { recordUsageEventSafe } from '../lib/utils/usage.js';
import { writeAuditLogSafe } from '../lib/utils/audit.js';
import { decryptUserApiKey } from '../lib/utils/apiKeyCrypto.js';
import { issueApiKeyForUser } from '../lib/utils/apiKeys.js';
import {
  USER_ROLES,
  USER_ROLES_NAMES,
  TEAM_MEMBER_ROLES,
  isSuperAdmin,
  isTeamAdmin,
  isAssignablePlatformRole,
  rolesAssignableBy,
  authCapabilities,
  normalizeTeamMemberRole,
  canManageTeamAsync,
} from '../lib/utils/user.js';

const SCHEMA = 'w4h';
const TABLE = 'users';

router.post('/login', login);
router.post('/send-code', sendLoginCode);
router.post('/verify-code', verifyLoginCode);
router.post('/token-valid', tokenValid);
router.get('/invite/:token', getInvite);
router.post('/invite/accept', acceptInvite);
router.post('/invite/resend', auth.authenticate, auth.requireTeamAdmin, resendInvite);
router.post('/invite/revoke', auth.authenticate, auth.requireTeamAdmin, revokeInvite);
router.get('/me/api-key', auth.authenticate, getMyApiKeyMeta);
router.post('/me/api-key/reveal', auth.authenticate, revealMyApiKey);
router.post('/me/api-key', auth.authenticate, rotateMyApiKey);
router.delete('/me/api-key', auth.authenticate, revokeMyApiKey);
router.delete('/user/:email/api-key', auth.authenticate, auth.requirePlatformAdmin, revokeUserApiKeyByEmail);
router.get('/', auth.authenticate, auth.requireTeamAdmin, list);
router.get('/user/:email', auth.authenticate, auth.requireTeamAdmin, getByEmail);
router.post('/profile', auth.authenticate, updateProfile);
router.post('/add', auth.authenticate, auth.requireTeamAdmin, addUser);
router.post('/set', auth.authenticate, auth.requireTeamAdmin, setUser);
router.post('/discontinue', auth.authenticate, auth.requireTeamAdmin, discontinueUser);
router.post('/resume', auth.authenticate, auth.requireTeamAdmin, resumeUser);
router.delete('/:email', auth.authenticate, auth.requirePlatformAdmin, del);
router.get('/roles', auth.authenticate, auth.requireTeamAdmin, roles);

function assertUserCanAuthenticate(user, { requirePassword = true } = {}) {
  if (!user) return { ok: false, message: 'Invalid Email or Password!' };
  if (user.status === 'invited') {
    return { ok: false, message: 'Account not activated. Check your email for the invitation link.' };
  }
  if (user.status === 'discontinued') {
    return { ok: false, message: 'This account has been discontinued.' };
  }
  // Password login needs a stored hash; email-code login does not.
  if (requirePassword && !user.password) {
    return { ok: false, message: 'Invalid Email or Password!' };
  }
  return { ok: true };
}


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

/** Accept team_ids[] or teams: [{ team_id|id, role }] and return [{ team_id, role }]. */
function normalizeTeamsWithRoles(input, defaultRole = TEAM_MEMBER_ROLES.MEMBER) {
  if (!Array.isArray(input)) {
    return normalizeTeamIds(input).map((team_id) => ({
      team_id,
      role: normalizeTeamMemberRole(defaultRole),
    }));
  }
  if (!input.length) return [];

  if (typeof input[0] === 'object' && input[0] != null) {
    const out = [];
    const seen = new Set();
    for (const row of input) {
      const team_id = String(row.team_id ?? row.id ?? '').trim();
      if (!team_id || seen.has(team_id)) continue;
      seen.add(team_id);
      out.push({
        team_id,
        role: normalizeTeamMemberRole(row.role || defaultRole),
      });
    }
    return out;
  }

  return normalizeTeamIds(input).map((team_id) => ({
    team_id,
    role: normalizeTeamMemberRole(defaultRole),
  }));
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

async function getTeamMembershipsForUser(userId) {
  const result = await query(
    `SELECT tm.team_id::text AS team_id, tm.role AS team_role, t.name AS team_name
     FROM ${SCHEMA}.team_members tm
     JOIN ${SCHEMA}.teams t ON t.id = tm.team_id
     WHERE tm.user_id = $1
     ORDER BY t.name ASC`,
    [userId]
  );
  return result.rows.map((r) => ({
    team_id: r.team_id,
    role: r.team_role || TEAM_MEMBER_ROLES.MEMBER,
    name: r.team_name,
  }));
}

async function syncUserTeams(userId, teamsInput = [], defaultRole = TEAM_MEMBER_ROLES.MEMBER) {
  const teams = normalizeTeamsWithRoles(teamsInput, defaultRole);
  await query(`DELETE FROM ${SCHEMA}.team_members WHERE user_id = $1`, [userId]);
  for (const t of teams) {
    await query(
      `INSERT INTO ${SCHEMA}.team_members (user_id, team_id, role)
       VALUES ($1, $2::uuid, $3)
       ON CONFLICT (team_id, user_id) DO UPDATE SET role = EXCLUDED.role`,
      [userId, t.team_id, t.role]
    );
  }
  return teams;
}

/** Keep legacy users.role in sync with whether the user is team_admin on any team. */
async function syncPlatformRoleFromMemberships(userId, currentRole) {
  if (isSuperAdmin(currentRole)) return currentRole;
  const hasAdmin = await query(
    `SELECT 1 FROM ${SCHEMA}.team_members WHERE user_id = $1 AND role = 'team_admin' LIMIT 1`,
    [userId]
  );
  const nextRole = hasAdmin.rowCount ? USER_ROLES.TEAM_ADMIN : USER_ROLES.PROJECT_USER;
  if (nextRole !== currentRole) {
    await query(`UPDATE ${SCHEMA}.${TABLE} SET role = $1 WHERE id = $2`, [nextRole, userId]);
  }
  return nextRole;
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
  const memberships = await getTeamMembershipsForUser(user.id);
  const teamIds = memberships.map((m) => m.team_id);
  return {
    id: user.id,
    name: user.name,
    email: user.email,
    role: user.role,
    team_ids: teamIds,
    team_id: teamIds[0] || null,
    team_memberships: memberships,
    caps: await authCapabilities(user),
  };
}

function roles(req, res) {
  Promise.resolve(canManageTeamAsync(req.user)).then(async (ok) => {
    if (!ok) {
      return res.status(403).json({ error: 'Unauthorized' });
    }
    let allowed = rolesAssignableBy(req.user.role);
    if (!allowed.length && await canManageTeamAsync(req.user) && !isSuperAdmin(req.user.role)) {
      allowed = [USER_ROLES.PROJECT_USER];
    }
    res.json(allowed.map((value) => ({
      value,
      text: USER_ROLES_NAMES[value] || value,
    })));
  }).catch((e) => {
    console.error('roles:', e);
    res.status(500).json({ error: e.message });
  });
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

/** Hard delete is intentionally unsupported (ADR-010 / U-01). */
async function del(req, res) {
  return res.status(405).json({
    error: 'Users cannot be deleted. Use discontinue to suspend access (POST /users/discontinue).'
  });
}

async function loadManageableUser(req, { email, userId }) {
  if (!(await canManageTeamAsync(req.user))) {
    return { error: { status: 403, message: 'Unauthorized' } };
  }

  const identifyClause = userId ? 'id = $1' : 'email = $1';
  const identifyValue = userId || email;
  if (!identifyValue) {
    return { error: { status: 400, message: 'email or user_id is required' } };
  }

  const result = await query(
    `SELECT id, name, email, role, status, (password IS NOT NULL) AS has_password
     FROM ${SCHEMA}.${TABLE}
     WHERE ${identifyClause}`,
    [identifyValue]
  );

  if (!result.rowCount) {
    return { error: { status: 404, message: 'User not found' } };
  }

  const target = result.rows[0];

  if (isSuperAdmin(target.role)) {
    return { error: { status: 400, message: 'Super admin accounts cannot be discontinued or resumed here' } };
  }

  if (Number(target.id) === Number(req.user.id)) {
    return { error: { status: 400, message: 'You cannot discontinue your own account' } };
  }

  if (!isSuperAdmin(req.user.role) && (await canManageTeamAsync(req.user))) {
    const myTeams = await getTeamIdsForUser(req.user.id);
    const theirTeams = await getTeamIdsForUser(target.id);
    if (!theirTeams.some((id) => myTeams.includes(id))) {
      return { error: { status: 403, message: 'Unauthorized' } };
    }
    // Team admins may only manage Project users in their team.
    if (!rolesAssignableBy(req.user.role).includes(target.role) && target.role !== 'researcher' && target.role !== 'clinician') {
      return { error: { status: 403, message: 'Team admins can only discontinue Project users in their team' } };
    }
  }

  return { target };
}

async function discontinueUser(req, res) {
  try {
    const email = (req.body?.email || '').trim();
    const userId = req.body?.user_id || null;
    const loaded = await loadManageableUser(req, { email, userId });
    if (loaded.error) {
      return res.status(loaded.error.status).json({ error: loaded.error.message });
    }

    const target = loaded.target;
    if (target.status === 'discontinued') {
      return res.json({ success: true, message: 'User is already discontinued', status: 'discontinued' });
    }

    await query(
      `UPDATE ${SCHEMA}.${TABLE}
       SET status = 'discontinued',
           api_key_hash = NULL,
           api_key_prefix = NULL,
           api_key_encrypted = NULL,
           api_key_created_at = NULL,
           updated_at = CURRENT_TIMESTAMP
       WHERE id = $1`,
      [target.id]
    );

    // Invalidate any unused invite tokens so the account cannot be activated while discontinued.
    await query(
      `UPDATE ${SCHEMA}.invitations
       SET revoked_at = now()
       WHERE user_id = $1 AND used_at IS NULL AND revoked_at IS NULL`,
      [target.id]
    );

    await writeAuthAudit(req, {
      userId: req.user.id,
      action: 'user.discontinued',
      details: { target_user_id: target.id, email: target.email }
    });

    res.json({
      success: true,
      message: 'User discontinued. Login and API access are suspended; the account record is retained.',
      status: 'discontinued'
    });
  } catch (error) {
    console.error('Discontinue user error:', error);
    res.status(500).json({ error: error.message });
  }
}

async function resumeUser(req, res) {
  try {
    const email = (req.body?.email || '').trim();
    const userId = req.body?.user_id || null;
    const loaded = await loadManageableUser(req, { email, userId });
    if (loaded.error) {
      return res.status(loaded.error.status).json({ error: loaded.error.message });
    }

    const target = loaded.target;
    if (target.status !== 'discontinued') {
      return res.status(400).json({ error: 'Only discontinued users can be resumed' });
    }

    const nextStatus = target.has_password ? 'active' : 'invited';
    await query(
      `UPDATE ${SCHEMA}.${TABLE}
       SET status = $1, updated_at = CURRENT_TIMESTAMP
       WHERE id = $2`,
      [nextStatus, target.id]
    );

    await writeAuthAudit(req, {
      userId: req.user.id,
      action: 'user.resumed',
      details: { target_user_id: target.id, email: target.email, status: nextStatus }
    });

    res.json({
      success: true,
      message: nextStatus === 'active'
        ? 'User resumed. They can sign in again; new API keys must be issued if needed.'
        : 'User resumed as invited. Resend an invitation so they can activate their account.',
      status: nextStatus
    });
  } catch (error) {
    console.error('Resume user error:', error);
    res.status(500).json({ error: error.message });
  }
}

async function getMyApiKeyMeta(req, res) {
  try {
    const result = await query(
      `SELECT api_key_prefix, api_key_created_at, api_key_last_used_at,
              (api_key_hash IS NOT NULL) AS api_key_active,
              (api_key_encrypted IS NOT NULL AND api_key_encrypted <> '') AS revealable
       FROM ${SCHEMA}.${TABLE} WHERE id = $1`,
      [req.user.id]
    );
    if (!result.rowCount) {
      return res.status(404).json({ error: 'User not found' });
    }
    const row = result.rows[0];
    res.json({
      active: row.api_key_active,
      revealable: !!row.revealable,
      api_key_prefix: row.api_key_prefix,
      api_key_created_at: row.api_key_created_at,
      api_key_last_used_at: row.api_key_last_used_at
    });
  } catch (error) {
    console.error('getMyApiKeyMeta:', error);
    res.status(500).json({ error: error.message });
  }
}

async function verifyCurrentUserPassword(userId, password) {
  if (!password || typeof password !== 'string') {
    return { ok: false, status: 400, error: 'Password is required' };
  }
  const result = await query(
    `SELECT password FROM ${SCHEMA}.${TABLE} WHERE id = $1`,
    [userId]
  );
  if (!result.rowCount) {
    return { ok: false, status: 404, error: 'User not found' };
  }
  const hash = result.rows[0].password;
  if (!hash || !bcrypt.compareSync(password, hash)) {
    return { ok: false, status: 401, error: 'Incorrect password' };
  }
  return { ok: true };
}

async function revealMyApiKey(req, res) {
  try {
    const verified = await verifyCurrentUserPassword(req.user.id, req.body?.password);
    if (!verified.ok) {
      writeAuditLogSafe(req, {
        action: 'user.api_key.reveal_denied',
        resourceType: 'user',
        resourceId: req.user.id,
        details: { reason: verified.error }
      });
      return res.status(verified.status).json({ error: verified.error });
    }

    const result = await query(
      `SELECT api_key_encrypted, api_key_prefix, api_key_created_at,
              (api_key_hash IS NOT NULL) AS api_key_active
       FROM ${SCHEMA}.${TABLE} WHERE id = $1`,
      [req.user.id]
    );
    if (!result.rowCount) {
      return res.status(404).json({ error: 'User not found' });
    }
    const row = result.rows[0];
    if (!row.api_key_active) {
      return res.status(404).json({ error: 'No API credential on file. Generate one first.' });
    }

    const plaintext = decryptUserApiKey(row.api_key_encrypted);
    if (!plaintext) {
      writeAuditLogSafe(req, {
        action: 'user.api_key.reveal_unavailable',
        resourceType: 'user',
        resourceId: req.user.id,
        details: { reason: 'not_encrypted' }
      });
      return res.status(409).json({
        error:
          'This credential was created before password-protected reveal was enabled. Rotate it to create a new revealable credential.'
      });
    }

    writeAuditLogSafe(req, {
      action: 'user.api_key.reveal',
      resourceType: 'user',
      resourceId: req.user.id,
      details: { api_key_prefix: row.api_key_prefix || null }
    });

    res.json({
      api_key: plaintext,
      api_key_prefix: row.api_key_prefix,
      api_key_created_at: row.api_key_created_at
    });
  } catch (error) {
    console.error('revealMyApiKey:', error);
    res.status(500).json({ error: error.message });
  }
}

async function rotateMyApiKey(req, res) {
  try {
    const verified = await verifyCurrentUserPassword(req.user.id, req.body?.password);
    if (!verified.ok) {
      writeAuditLogSafe(req, {
        action: 'user.api_key.rotate_denied',
        resourceType: 'user',
        resourceId: req.user.id,
        details: { reason: verified.error }
      });
      return res.status(verified.status).json({ error: verified.error });
    }

    let issued;
    try {
      issued = await issueApiKeyForUser(req.user.id);
    } catch (err) {
      if (err.message === 'User not found') {
        return res.status(404).json({ error: 'User not found' });
      }
      throw err;
    }

    writeAuditLogSafe(req, {
      action: 'user.api_key.rotate',
      resourceType: 'user',
      resourceId: req.user.id,
      details: { api_key_prefix: issued.api_key_prefix }
    });

    res.status(201).json(issued);
  } catch (error) {
    console.error('rotateMyApiKey:', error);
    res.status(500).json({ error: error.message });
  }
}

async function revokeMyApiKey(req, res) {
  try {
    await query(
      `UPDATE ${SCHEMA}.${TABLE}
       SET api_key_hash = NULL, api_key_prefix = NULL, api_key_encrypted = NULL, api_key_created_at = NULL
       WHERE id = $1`,
      [req.user.id]
    );
    writeAuditLogSafe(req, {
      action: 'user.api_key.revoke',
      resourceType: 'user',
      resourceId: req.user.id
    });
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
       SET api_key_hash = NULL, api_key_prefix = NULL, api_key_encrypted = NULL, api_key_created_at = NULL
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
  if (!(await canManageTeamAsync(req.user))) {
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
      // Admin-set password activates an invited account (escape hatch when email is unavailable).
      fields.push(`status = 'active'`);
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
      let role = result.rows[0].role;
      const defaultMemberRole = isTeamAdmin(role)
        ? TEAM_MEMBER_ROLES.TEAM_ADMIN
        : TEAM_MEMBER_ROLES.MEMBER;
      const teamsInput = user.teams ?? user.team_memberships ?? user.team_ids ?? user.team_id;
      const teams = normalizeTeamsWithRoles(teamsInput, defaultMemberRole);
      const teamIds = teams.map((t) => t.team_id);

      if (!isSuperAdmin(req.user.role) && (await canManageTeamAsync(req.user))) {
        const myTeams = await getTeamIdsForUser(req.user.id);
        if (teamIds.length && !teamIds.every((id) => myTeams.includes(id))) {
          return res.status(403).json({ error: 'Can only assign users to your own team(s)' });
        }
      }

      if (isSuperAdmin(role)) {
        await syncUserTeams(userId, []);
        await ensureAdminAssignedToAllProjects(userId);
      } else if (Object.prototype.hasOwnProperty.call(user, 'teams')
        || Object.prototype.hasOwnProperty.call(user, 'team_memberships')
        || Object.prototype.hasOwnProperty.call(user, 'team_ids')
        || Object.prototype.hasOwnProperty.call(user, 'team_id')) {
        if (!teams.length) {
          return res.status(400).json({ error: 'At least one team is required for this role' });
        }
        await syncUserTeams(userId, teams, defaultMemberRole);
        role = await syncPlatformRoleFromMemberships(userId, role);
      }

      res.json({ success: true, message: 'User updated', role });
    } else {
      res.json({ error: 'User not found' });
    }
  } catch (error) {
    console.error('Set user error:', error);
    res.status(500).json({ error: error.message });
  }
}

async function addUser(req, res) {
  if (!(await canManageTeamAsync(req.user))) {
    return res.status(403).json({ error: 'Unauthorized' });
  }
  try {
    const { user } = req.body;

    const email = (user.email || '').trim();
    const name = (user.name || email || '').trim();
    const role = user.role;
    const defaultMemberRole = isTeamAdmin(role)
      ? TEAM_MEMBER_ROLES.TEAM_ADMIN
      : TEAM_MEMBER_ROLES.MEMBER;
    const teams = normalizeTeamsWithRoles(
      user.teams ?? user.team_memberships ?? user.team_ids ?? user.team_id,
      defaultMemberRole
    );
    const teamIds = teams.map((t) => t.team_id);

    if (!email || !role) {
      return res.status(400).json({ error: 'email and role are required' });
    }

    if (!appBaseUrl()) {
      return res.status(503).json({
        error: 'APP_URL is not configured. Set APP_URL to the public app base URL so invitation links can be generated.'
      });
    }

    if (!isAssignablePlatformRole(role)) {
      return res.status(400).json({ error: 'Invalid role' });
    }

    let allowed = rolesAssignableBy(req.user.role);
    if (!allowed.length && !isSuperAdmin(req.user.role) && (await canManageTeamAsync(req.user))) {
      allowed = [USER_ROLES.PROJECT_USER];
    }
    if (!allowed.includes(role)) {
      return res.status(403).json({ error: 'Cannot assign this role' });
    }

    if (role !== USER_ROLES.SUPER_ADMIN && teamIds.length === 0) {
      return res.status(400).json({ error: 'At least one team is required for this role' });
    }

    if (!isSuperAdmin(req.user.role) && (await canManageTeamAsync(req.user))) {
      const myTeams = await getTeamIdsForUser(req.user.id);
      if (!teamIds.length || !teamIds.every(id => myTeams.includes(id))) {
        return res.status(403).json({ error: 'Can only invite users to your own team(s)' });
      }
    }

    const insertResult = await query(
      `INSERT INTO ${SCHEMA}.${TABLE} (name, email, password, role, status)
       VALUES ($1, $2, NULL, $3, 'invited')
       RETURNING id, role`,
      [name, email, role]
    );

    const newUserId = insertResult.rows[0].id;
    await syncUserTeams(newUserId, teams, defaultMemberRole);
    if (!isSuperAdmin(role)) {
      await syncPlatformRoleFromMemberships(newUserId, role);
    }

    // Project membership is managed on the project, not at user create —
    // except Super admins who retain access to all projects.
    if (isSuperAdmin(role)) {
      await ensureAdminAssignedToAllProjects(newUserId);
    }

    const roleLabel = USER_ROLES_NAMES[role] || role;
    const rawToken = await createInvitation(newUserId, req.user.id);
    const inviteUrl = inviteAcceptUrl(rawToken);

    let mailWarning = null;
    if (isMailerConfigured()) {
      try {
        await sendInviteEmail({ email, name, roleLabel, rawToken });
      } catch (mailError) {
        mailWarning = mailError.message;
      }
    } else {
      mailWarning = 'SMTP is not configured; share the invitation link securely with the user.';
    }

    await writeAuthAudit(req, {
      userId: req.user.id,
      action: 'user.invite_sent',
      details: { invited_user_id: newUserId, email, role }
    });

    res.status(201).json({
      success: true,
      message: 'Invitation created',
      status: 'invited',
      inviteUrl,
      ...(mailWarning ? { mailWarning } : {})
    });
  } catch (error) {
    if (error.code === '23505' || /users_email_key|duplicate key/i.test(error.message || '')) {
      return res.status(409).json({ error: 'A user with this email already exists' });
    }

    console.error('Add user error:', error);
    res.status(500).json({ error: error.message });
  }
}

async function getInvite(req, res) {
  try {
    const rawToken = String(req.params.token || '').trim();
    if (!rawToken) {
      return res.status(400).json({ error: 'Invitation token is required' });
    }

    const result = await query(
      `SELECT i.id, i.expires_at, i.used_at, i.revoked_at,
              u.id AS user_id, u.name, u.email, u.status, u.role
       FROM ${SCHEMA}.invitations i
       JOIN ${SCHEMA}.${TABLE} u ON u.id = i.user_id
       WHERE i.token_hash = $1`,
      [hashInviteToken(rawToken)]
    );

    if (!result.rowCount) {
      return res.status(404).json({ error: 'Invitation not found' });
    }

    const row = result.rows[0];
    if (row.used_at) {
      return res.status(410).json({ error: 'This invitation has already been used' });
    }
    if (row.revoked_at) {
      return res.status(410).json({ error: 'This invitation has been revoked' });
    }
    if (new Date(row.expires_at) <= new Date()) {
      return res.status(410).json({ error: 'This invitation has expired' });
    }
    if (row.status === 'discontinued') {
      return res.status(403).json({ error: 'This account has been discontinued' });
    }

    res.json({
      email: row.email,
      name: row.name,
      role: row.role,
      expires_at: row.expires_at
    });
  } catch (error) {
    console.error('getInvite error:', error);
    res.status(500).json({ error: error.message });
  }
}

async function acceptInvite(req, res) {
  try {
    const rawToken = String(req.body?.token || '').trim();
    const password = req.body?.password;
    const name = (req.body?.name || '').trim();

    if (!rawToken || !password) {
      return res.status(400).json({ error: 'token and password are required' });
    }

    const passwordValidation = validatePassword(password);
    if (!passwordValidation.valid) {
      return res.status(400).json({ error: passwordValidation.message });
    }

    const result = await query(
      `SELECT i.id AS invite_id, i.expires_at, i.used_at, i.revoked_at,
              u.id AS user_id, u.email, u.name, u.status, u.role
       FROM ${SCHEMA}.invitations i
       JOIN ${SCHEMA}.${TABLE} u ON u.id = i.user_id
       WHERE i.token_hash = $1`,
      [hashInviteToken(rawToken)]
    );

    if (!result.rowCount) {
      return res.status(404).json({ error: 'Invitation not found' });
    }

    const row = result.rows[0];
    if (row.used_at) {
      return res.status(410).json({ error: 'This invitation has already been used' });
    }
    if (row.revoked_at) {
      return res.status(410).json({ error: 'This invitation has been revoked' });
    }
    if (new Date(row.expires_at) <= new Date()) {
      return res.status(410).json({ error: 'This invitation has expired' });
    }
    if (row.status === 'discontinued') {
      return res.status(403).json({ error: 'This account has been discontinued' });
    }

    const hashedPassword = bcrypt.hashSync(password, +process.env.BCRYPT_ROUNDS || 12);
    const displayName = name || row.name || row.email;

    await query(
      `UPDATE ${SCHEMA}.${TABLE}
       SET password = $1, name = $2, status = 'active', updated_at = CURRENT_TIMESTAMP
       WHERE id = $3`,
      [hashedPassword, displayName, row.user_id]
    );
    await query(
      `UPDATE ${SCHEMA}.invitations SET used_at = now() WHERE id = $1`,
      [row.invite_id]
    );
    await query(
      `UPDATE ${SCHEMA}.invitations
       SET revoked_at = now()
       WHERE user_id = $1 AND id <> $2 AND used_at IS NULL AND revoked_at IS NULL`,
      [row.user_id, row.invite_id]
    );

    if (isSuperAdmin(row.role)) {
      await ensureAdminAssignedToAllProjects(row.user_id);
    }

    // D-16a: auto-issue a personal API key at activation so first-time users
    // don't hit a manual "Generate new key" step. Best-effort — a failure here
    // must not block account activation; the user can still generate one later.
    let issuedApiKey = null;
    try {
      issuedApiKey = await issueApiKeyForUser(row.user_id);
      writeAuditLogSafe(req, {
        action: 'user.api_key.auto_issued',
        resourceType: 'user',
        resourceId: row.user_id,
        details: { api_key_prefix: issuedApiKey.api_key_prefix, reason: 'invite_accepted' }
      });
    } catch (err) {
      console.error('acceptInvite: auto-issue API key failed:', err.message);
    }

    const userInfo = await buildUserSessionInfo({
      id: row.user_id,
      name: displayName,
      email: row.email,
      role: row.role
    });
    const token = auth.sign(userInfo);

    await writeAuthAudit(req, {
      userId: row.user_id,
      action: 'user.invite_accepted',
      details: { email: row.email }
    });

    res.json({
      success: true,
      user: { ...userInfo, token },
      token,
      ...(issuedApiKey ? { api_key: issuedApiKey.api_key, api_key_prefix: issuedApiKey.api_key_prefix } : {})
    });
  } catch (error) {
    console.error('acceptInvite error:', error);
    res.status(500).json({ error: error.message });
  }
}

async function resendInvite(req, res) {
  if (!(await canManageTeamAsync(req.user))) {
    return res.status(403).json({ error: 'Unauthorized' });
  }
  try {
    if (!appBaseUrl()) {
      return res.status(503).json({
        error: 'APP_URL is not configured. Set APP_URL to the public app base URL so invitation links can be generated.'
      });
    }

    const email = (req.body?.email || '').trim();
    const userId = req.body?.user_id || null;
    if (!email && !userId) {
      return res.status(400).json({ error: 'email or user_id is required' });
    }

    const identifyClause = userId ? 'id = $1' : 'email = $1';
    const identifyValue = userId || email;
    const userResult = await query(
      `SELECT id, name, email, role, status FROM ${SCHEMA}.${TABLE} WHERE ${identifyClause}`,
      [identifyValue]
    );

    if (!userResult.rowCount) {
      return res.status(404).json({ error: 'User not found' });
    }

    const target = userResult.rows[0];
    if (target.status !== 'invited') {
      return res.status(400).json({ error: 'Only invited (not yet activated) users can receive a new invitation' });
    }

    if (!isSuperAdmin(req.user.role) && (await canManageTeamAsync(req.user))) {
      const myTeams = await getTeamIdsForUser(req.user.id);
      const theirTeams = await getTeamIdsForUser(target.id);
      if (!theirTeams.some((id) => myTeams.includes(id))) {
        return res.status(403).json({ error: 'Unauthorized' });
      }
    }

    const rawToken = await createInvitation(target.id, req.user.id);
    const inviteUrl = inviteAcceptUrl(rawToken);
    const roleLabel = USER_ROLES_NAMES[target.role] || target.role;

    let mailWarning = null;
    if (isMailerConfigured()) {
      try {
        await sendInviteEmail({
          email: target.email,
          name: target.name,
          roleLabel,
          rawToken
        });
      } catch (mailError) {
        mailWarning = mailError.message;
      }
    } else {
      mailWarning = 'SMTP is not configured; share the invitation link securely with the user.';
    }

    await writeAuthAudit(req, {
      userId: req.user.id,
      action: 'user.invite_resent',
      details: { invited_user_id: target.id, email: target.email }
    });

    res.json({
      success: true,
      message: 'Invitation resent',
      inviteUrl,
      ...(mailWarning ? { mailWarning } : {})
    });
  } catch (error) {
    console.error('resendInvite error:', error);
    res.status(500).json({ error: error.message });
  }
}

async function revokeInvite(req, res) {
  if (!(await canManageTeamAsync(req.user))) {
    return res.status(403).json({ error: 'Unauthorized' });
  }
  try {
    const email = (req.body?.email || '').trim();
    const userId = req.body?.user_id || null;
    if (!email && !userId) {
      return res.status(400).json({ error: 'email or user_id is required' });
    }

    const identifyClause = userId ? 'id = $1' : 'email = $1';
    const identifyValue = userId || email;
    const userResult = await query(
      `SELECT id, email, status FROM ${SCHEMA}.${TABLE} WHERE ${identifyClause}`,
      [identifyValue]
    );

    if (!userResult.rowCount) {
      return res.status(404).json({ error: 'User not found' });
    }

    const target = userResult.rows[0];
    if (target.status !== 'invited') {
      return res.status(400).json({ error: 'Only pending invitations can be revoked' });
    }

    if (!isSuperAdmin(req.user.role) && (await canManageTeamAsync(req.user))) {
      const myTeams = await getTeamIdsForUser(req.user.id);
      const theirTeams = await getTeamIdsForUser(target.id);
      if (!theirTeams.some((id) => myTeams.includes(id))) {
        return res.status(403).json({ error: 'Unauthorized' });
      }
    }

    await query(
      `UPDATE ${SCHEMA}.invitations
       SET revoked_at = now()
       WHERE user_id = $1 AND used_at IS NULL AND revoked_at IS NULL`,
      [target.id]
    );

    await writeAuthAudit(req, {
      userId: req.user.id,
      action: 'user.invite_revoked',
      details: { invited_user_id: target.id, email: target.email }
    });

    res.json({ success: true, message: 'Invitation revoked' });
  } catch (error) {
    console.error('revokeInvite error:', error);
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
  if (!(await canManageTeamAsync(req.user))) {
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
         u.status,
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
    if (!isSuperAdmin(req.user.role) && (await canManageTeamAsync(req.user))) {
      const myTeams = await getTeamIdsForUser(req.user.id);
      const theirTeams = Array.isArray(row.team_ids) ? row.team_ids.map(String) : [];
      const overlap = theirTeams.some((id) => myTeams.includes(id));
      if (!overlap && row.id !== req.user.id) {
        return res.status(403).json({ error: 'Unauthorized' });
      }
    }

    const memberships = await getTeamMembershipsForUser(row.id);
    res.json({
      ...row,
      team_memberships: memberships,
    });
  } catch (error) {
    console.error('Get user error:', error);
    res.status(500).json({ error: error.message });
  }
}

async function list(req, res) {
  if (!(await canManageTeamAsync(req.user))) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  try {
    const params = [];
    let where = '';

    if (!isSuperAdmin(req.user.role) && (await canManageTeamAsync(req.user))) {
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
         u.status,
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
           SELECT json_agg(
             json_build_object(
               'team_id', t.id::text,
               'name', t.name,
               'role', tm.role
             )
             ORDER BY t.name
           )
           FROM ${SCHEMA}.team_members tm
           JOIN ${SCHEMA}.teams t ON t.id = tm.team_id
           WHERE tm.user_id = u.id
         ), '[]'::json) AS team_memberships,
         COALESCE((
           SELECT STRING_AGG(
             p.name || ' (' || pm.role || ')',
             ', ' ORDER BY p.name
           )
           FROM ${SCHEMA}.project_members pm
           JOIN ${SCHEMA}.projects p ON p.id = pm.project_id
           WHERE pm.user_id = u.id
         ), '') AS projects,
         COALESCE((
           SELECT json_agg(
             json_build_object(
               'project_id', p.id::text,
               'name', p.name,
               'role', pm.role
             )
             ORDER BY p.name
           )
           FROM ${SCHEMA}.project_members pm
           JOIN ${SCHEMA}.projects p ON p.id = pm.project_id
           WHERE pm.user_id = u.id
         ), '[]'::json) AS project_memberships
       FROM ${SCHEMA}.${TABLE} u
       ${where}
       ORDER BY u.name ASC NULLS LAST, u.email ASC`,
      params
    );

    res.json(result.rows.map((row) => ({
      ...row,
      team_memberships: Array.isArray(row.team_memberships) ? row.team_memberships : [],
      project_memberships: Array.isArray(row.project_memberships) ? row.project_memberships : [],
    })));
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
      `SELECT id, name, email, status FROM ${SCHEMA}.${TABLE} WHERE lower(trim(email)) = $1`,
      [email]
    );

    if (userResult.rowCount === 0) {
      return res.status(404).json({ error: { message: 'No account found for that email address.' } });
    }

    const user = userResult.rows[0];
    if (user.status === 'invited') {
      return res.status(403).json({
        error: { message: 'Account not activated. Check your email for the invitation link.' }
      });
    }
    if (user.status === 'discontinued') {
      return res.status(403).json({
        error: { message: 'This account has been discontinued.' }
      });
    }

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
      `SELECT id, name, email, role, status FROM ${SCHEMA}.${TABLE} WHERE lower(trim(email)) = $1`,
      [email]
    );

    if (userResult.rowCount === 0) {
      return res.status(401).json({ error: { message: 'Invalid email or verification code.' } });
    }

    const user = userResult.rows[0];
    const authCheck = assertUserCanAuthenticate(user, { requirePassword: false });
    if (!authCheck.ok) {
      return res.status(403).json({ error: { message: authCheck.message } });
    }

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

    recordUsageEventSafe({
      userId: user.id,
      eventType: 'user.login',
      metadata: { method: 'email_code', auth_method: 'jwt' },
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
      `SELECT id, name, email, password, role, status FROM ${SCHEMA}.${TABLE} WHERE email = $1`,
      [email]
    );

    if (result.rows.length === 0) {
      return res.json({ error: { message: 'Invalid Email or Password!' } });
    }

    const user = result.rows[0];
    const authCheck = assertUserCanAuthenticate(user);
    if (!authCheck.ok) {
      return res.json({ error: { message: authCheck.message } });
    }

    if (!bcrypt.compareSync(password, user.password)) {
      return res.json({ error: { message: 'Invalid Email or Password!' } });
    }

    if (isSuperAdmin(user.role)) {
      await ensureAdminAssignedToAllProjects(user.id);
    }

    const userInfo = await buildUserSessionInfo(user);

    const token = auth.sign(userInfo);
    userInfo.token = token;

    recordUsageEventSafe({
      userId: user.id,
      eventType: 'user.login',
      metadata: { method: 'password', auth_method: 'jwt' },
    });

    res.json(userInfo);
  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({ error: error.message });
  }
}

export default router;