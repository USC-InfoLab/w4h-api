import os from 'os';
import { query } from '../../db.js';
import crypto from 'crypto';

/** Canonical MVP platform roles (ADR-004). */
const USER_ROLES = {
  SUPER_ADMIN: 'super_admin',
  TEAM_ADMIN: 'team_admin',
  PROJECT_USER: 'project_user',
  // Legacy stored values — treat as aliases until data migrated
  ADMIN: 'admin',
  RESEARCHER: 'researcher',
  CLINICIAN: 'clinician',
  GUEST: 'guest',
};

/** Per-team membership roles (ADR-004 / MVP v0.11). */
const TEAM_MEMBER_ROLES = {
  TEAM_ADMIN: 'team_admin',
  MEMBER: 'member',
};

const PLATFORM_ROLES = [
  USER_ROLES.SUPER_ADMIN,
  USER_ROLES.TEAM_ADMIN,
  USER_ROLES.PROJECT_USER,
];

const USER_ROLES_NAMES = {
  [USER_ROLES.SUPER_ADMIN]: 'Super admin',
  [USER_ROLES.TEAM_ADMIN]: 'Admin',
  [USER_ROLES.PROJECT_USER]: 'User',
  [USER_ROLES.ADMIN]: 'Super admin',
  [USER_ROLES.RESEARCHER]: 'User',
  [USER_ROLES.CLINICIAN]: 'User',
  [USER_ROLES.GUEST]: 'Guest',
};

const TEAM_MEMBER_ROLE_NAMES = {
  [TEAM_MEMBER_ROLES.TEAM_ADMIN]: 'Admin',
  [TEAM_MEMBER_ROLES.MEMBER]: 'User',
};

function isSuperAdmin(role) {
  return role === USER_ROLES.SUPER_ADMIN || role === USER_ROLES.ADMIN;
}

/**
 * Input alias only: callers may pass role `team_admin` when creating/updating a user to mean
 * "Admin on the teams given". It is never stored on `users.role` (R-09 — Team admin is
 * membership-scoped, `team_members.role`).
 */
function roleImpliesTeamAdmin(role) {
  return role === USER_ROLES.TEAM_ADMIN;
}

function normalizeTeamMemberRole(role) {
  return role === TEAM_MEMBER_ROLES.TEAM_ADMIN
    ? TEAM_MEMBER_ROLES.TEAM_ADMIN
    : TEAM_MEMBER_ROLES.MEMBER;
}

async function userHasTeamAdminMembership(userId) {
  if (userId == null) return false;
  const result = await query(
    `SELECT 1 FROM w4h.team_members WHERE user_id = $1 AND role = 'team_admin' LIMIT 1`,
    [userId]
  );
  return result.rowCount > 0;
}

/** Effective Team admin on a specific team: Super admin, or a `team_admin` membership on that team. */
async function isTeamAdminOnTeam(userId, teamId, platformRole) {
  if (isSuperAdmin(platformRole)) return true;
  if (userId == null || !teamId) return false;
  const result = await query(
    `SELECT role FROM w4h.team_members WHERE team_id = $1::uuid AND user_id = $2`,
    [teamId, userId]
  );
  return result.rowCount > 0 && result.rows[0].role === TEAM_MEMBER_ROLES.TEAM_ADMIN;
}

/**
 * Platform Admin capability — Super admin only.
 * Enforced by auth.requirePlatformAdmin and canAccessAdminUi().
 * Team admins must NEVER receive this capability.
 */
function canAccessAdminUi(role) {
  return isSuperAdmin(role);
}

/**
 * Synchronous platform check: Super admin only. For "is a Team admin anywhere" use
 * canManageTeamAsync (membership-based) or the `caps.team_admin` claim.
 */
function canManageTeam(role) {
  // Super admin only; Team admin authority lives on team_members (use canManageTeamAsync).
  return isSuperAdmin(role);
}

async function canManageTeamAsync(user) {
  if (!user) return false;
  if (canManageTeam(user.role)) return true;
  return userHasTeamAdminMembership(user.id);
}

/**
 * May create teams: an Admin on any team, or an approved Admin who has not created their first
 * team yet (`users.can_create_teams`). A Super admin never creates teams — they only approve
 * access requests (ADR-028); the Admin defines their own teams.
 */
async function canCreateTeamsAsync(user) {
  if (!user || user.id == null || isSuperAdmin(user.role)) return false;
  if (await userHasTeamAdminMembership(user.id)) return true;
  const res = await query(`SELECT can_create_teams FROM w4h.users WHERE id = $1`, [user.id]);
  return res.rowCount > 0 && res.rows[0].can_create_teams === true;
}

function canManageUsers(role) {
  return canManageTeam(role);
}

/** Sync claims for JWTs — pass user id when possible so membership Team admins get caps. */
async function authCapabilities(roleOrUser, maybeUserId) {
  let role = roleOrUser;
  let userId = maybeUserId;
  if (roleOrUser && typeof roleOrUser === 'object') {
    role = roleOrUser.role;
    userId = roleOrUser.id;
  }
  const platform_admin = canAccessAdminUi(role);
  let team_admin = canManageTeam(role);
  if (!team_admin && userId != null) {
    team_admin = await userHasTeamAdminMembership(userId);
  }
  const can_create_teams = userId != null && !platform_admin
    ? await canCreateTeamsAsync({ id: userId, role })
    : false;
  return { platform_admin, team_admin, can_create_teams };
}

function isAssignablePlatformRole(role) {
  return PLATFORM_ROLES.includes(role);
}

function rolesAssignableBy(actorRole) {
  if (isSuperAdmin(actorRole)) {
    return [...PLATFORM_ROLES];
  }
  // Callers are already gated by canManageTeamAsync (a Team admin on some team).
  return [USER_ROLES.PROJECT_USER];
}

const getUser = async function (username) {
	if (username) {
		const result = await query(`SELECT * FROM w4h.users WHERE username = '${username}'`);
		return result.rowCount == 1 ? result.rows[0] : { name: os.userInfo().username };
	} else {
    return { name: os.userInfo().username };
  }
}

function generateApiKey(type = 'pk') {
  const payload = crypto.randomBytes(24).toString('base64url');
  const signature = crypto.randomBytes(24).toString('base64url');
  return `${type}.${payload}.${signature}`;
}

async function updateApiKey(id) {
  const apiKey = generateApiKey();
  const res = await query(
    `UPDATE w4h.accounts
     SET api_key = $1
     WHERE id = $2`,
    [apiKey, id]
  );

  return res.rowCount !== 0 ? apiKey : null;
}

async function createAccount(accountId, accountName, studyId) {
  const apiKey = generateApiKey();
  const accRes = await query(
    `INSERT INTO w4h.accounts (id, name, study_id, api_key)
   VALUES ($1, $2, $3, $4)
   ON CONFLICT (id) DO UPDATE
   SET name = EXCLUDED.name,
       study_id = EXCLUDED.city_id,
       api_key = EXCLUDED.api_key
   RETURNING id, api_key`,
    [accountId, accountName || accountId, studyId || null, apiKey]
  );

  if (accRes.rowCount === 0) {
    throw new Error('Failed to create or update account');
  }

  // --- get first theme (crosstown) theme ---
  const res = await query(`SELECT id, spec, name FROM w4h.vega_themes WHERE id = 1 LIMIT 1`);
  if (res.rowCount === 0) {
    throw new Error('No themes found to link to account');
  }
  const theme = res.rows[0];
  const themeRes = await query(
    `INSERT INTO w4h.vega_themes (slug, name, account_id, spec)
      VALUES ($1, $2, $3, $4)
      RETURNING id`,
    [accountId, theme.name, accountId, JSON.stringify(theme.spec)]
  );

  await query(
    `UPDATE w4h.accounts SET vega_theme_id = $1 WHERE id = $2`,
    [themeRes.rows[0].id, accountId]
  );

  return accRes.rows[0];
}

async function createCity(id, name) {
  const res = await query(
    `INSERT INTO w4h.cities (id, name)
   VALUES ($1, $2)
   ON CONFLICT (id) DO UPDATE
   SET name = EXCLUDED.name
   RETURNING id`,
    [id, name || id]
  );

  return res.rows[0];
}

async function getAccount(id) {
  const res = await query(
    `SELECT
       a.id,
       a.name,
       a.city_id,
       c.name AS study_name,
       a.api_key
     FROM w4h.accounts a
     LEFT JOIN w4h.projects c ON a.study_id = c.id
     WHERE a.id = $1`,
    [id]
  );

  return res.rows[0];
}

async function getUsers() {
  const res = await query(`
      SELECT 
        u.email,
        u.name,
        u.role,
        u.account_id,
        a.study_id,
      FROM w4h.users u
      LEFT JOIN w4h.accounts a ON a.id = u.account_id
      GROUP BY account_id, u.email, u.role, u.name, a.study_id
  `);
  return res.rows;
}

async function getAccounts() {
  const res = await query(`SELECT
      a.id,
      a.name,
      a.city_id,
      c.name AS study_name,
      api_key
    FROM w4h.accounts a
    LEFT JOIN w4h.projects c ON a.study_id = c.id
    ORDER BY a.id ASC`);

  return res.rows;
}

export {
  USER_ROLES,
  USER_ROLES_NAMES,
  PLATFORM_ROLES,
  TEAM_MEMBER_ROLES,
  TEAM_MEMBER_ROLE_NAMES,
  isSuperAdmin,
  roleImpliesTeamAdmin,
  normalizeTeamMemberRole,
  userHasTeamAdminMembership,
  isTeamAdminOnTeam,
  canAccessAdminUi,
  canManageTeam,
  canManageTeamAsync,
  canCreateTeamsAsync,
  canManageUsers,
  authCapabilities,
  isAssignablePlatformRole,
  rolesAssignableBy,
  getUser,
  getUsers,
  getAccount,
  getAccounts,
  updateApiKey,
  createAccount,
  createCity
};
