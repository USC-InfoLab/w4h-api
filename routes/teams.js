import express from 'express';
import { authenticate, requireTeamAdmin, requirePlatformAdmin } from '../auth.js';
import { query } from '../db.js';
import {
  isSuperAdmin,
  isTeamAdmin,
  canManageTeamAsync,
  isTeamAdminOnTeam,
  normalizeTeamMemberRole,
  TEAM_MEMBER_ROLES,
} from '../lib/utils/user.js';

const router = express.Router();
const SCHEMA = 'w4h';

router.get('/', authenticate, requireTeamAdmin, listTeams);
router.post('/', authenticate, requireTeamAdmin, createTeam);
router.get('/:id', authenticate, getTeam);
router.patch('/:id', authenticate, requireTeamAdmin, updateTeam);
router.post('/:id/members', authenticate, requireTeamAdmin, addTeamMember);
router.patch('/:id/members/:userId', authenticate, requireTeamAdmin, updateTeamMember);
router.delete('/:id/members/:userId', authenticate, requireTeamAdmin, removeTeamMember);

async function assertCanAccessTeam(req, teamId) {
  return isTeamAdminOnTeam(req.user.id, teamId, req.user.role);
}

async function actorIsScopedTeamAdmin(user) {
  if (isSuperAdmin(user.role)) return false;
  return canManageTeamAsync(user);
}

async function listTeamMembers(teamId) {
  const members = await query(
    `SELECT u.id, u.name, u.email, u.role AS platform_role, u.status,
            tm.role AS team_role
     FROM ${SCHEMA}.team_members tm
     JOIN ${SCHEMA}.users u ON u.id = tm.user_id
     WHERE tm.team_id = $1::uuid
     ORDER BY u.name ASC`,
    [teamId]
  );
  return members.rows.map((m) => ({
    id: m.id,
    name: m.name,
    email: m.email,
    status: m.status,
    role: m.platform_role,
    platform_role: m.platform_role,
    team_role: m.team_role || TEAM_MEMBER_ROLES.MEMBER,
  }));
}

function slugify(input) {
  return String(input || '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/(^-|-$)/g, '');
}

async function listTeams(req, res) {
  if (!(await canManageTeamAsync(req.user))) {
    return res.status(403).json({ error: 'Unauthorized' });
  }
  try {
    let sql = `
      SELECT
        t.id,
        t.slug,
        t.name,
        t.status,
        t.created_at,
        t.updated_at,
        COUNT(DISTINCT tm.user_id)::int AS member_count
      FROM ${SCHEMA}.teams t
      LEFT JOIN ${SCHEMA}.team_members tm ON tm.team_id = t.id
    `;
    const params = [];

    if (await actorIsScopedTeamAdmin(req.user)) {
      sql += `
        JOIN ${SCHEMA}.team_members mine
          ON mine.team_id = t.id AND mine.user_id = $1
      `;
      params.push(req.user.id);
    }

    sql += `
      GROUP BY t.id
      ORDER BY t.name ASC
    `;

    const result = await query(sql, params);
    res.json(result.rows);
  } catch (error) {
    console.error('listTeams:', error);
    res.status(500).json({ error: error.message });
  }
}

async function createTeam(req, res) {
  if (!(await canManageTeamAsync(req.user))) {
    return res.status(403).json({ error: 'Unauthorized' });
  }
  try {
    const name = String(req.body?.name || '').trim();
    if (!name) {
      return res.status(400).json({ error: 'name is required' });
    }
    const slug = slugify(req.body?.slug || name);
    if (!slug) {
      return res.status(400).json({ error: 'A valid team slug is required' });
    }
    const duplicate = await query(
      `SELECT id
       FROM ${SCHEMA}.teams
       WHERE slug = $1 OR lower(name) = lower($2)
       LIMIT 1`,
      [slug, name]
    );
    if (duplicate.rowCount) {
      return res.status(409).json({ error: 'A team with this name or slug already exists' });
    }

    const result = await query(
      `INSERT INTO ${SCHEMA}.teams (slug, name)
       VALUES ($1, $2)
       RETURNING id, slug, name, status, created_at, updated_at`,
      [slug, name]
    );

    const team = result.rows[0];

    // Creator membership: Team admin owns new teams; Super admin is listed as member for visibility only.
    if (req.user?.id != null) {
      const membershipRole = isSuperAdmin(req.user.role)
        ? TEAM_MEMBER_ROLES.MEMBER
        : TEAM_MEMBER_ROLES.TEAM_ADMIN;
      await query(
        `INSERT INTO ${SCHEMA}.team_members (team_id, user_id, role)
         VALUES ($1::uuid, $2, $3)
         ON CONFLICT (team_id, user_id) DO UPDATE SET role = EXCLUDED.role`,
        [team.id, req.user.id, membershipRole]
      );
    }

    res.status(201).json(team);
  } catch (error) {
    if (error.code === '23505') {
      return res.status(409).json({ error: 'A team with this name or slug already exists' });
    }
    console.error('createTeam:', error);
    res.status(500).json({ error: error.message });
  }
}

async function userIsOnTeam(userId, teamId) {
  if (userId == null || !teamId) return false;
  const res = await query(
    `SELECT 1 FROM ${SCHEMA}.team_members WHERE team_id = $1::uuid AND user_id = $2`,
    [teamId, userId]
  );
  return res.rowCount > 0;
}

async function getTeam(req, res) {
  try {
    const teamId = req.params.id;
    const allowed = isSuperAdmin(req.user.role) || (await userIsOnTeam(req.user.id, teamId));
    if (!allowed) {
      return res.status(403).json({ error: 'Unauthorized' });
    }

    const team = await query(
      `SELECT id, slug, name, status, created_at, updated_at
       FROM ${SCHEMA}.teams WHERE id = $1::uuid`,
      [teamId]
    );
    if (!team.rowCount) {
      return res.status(404).json({ error: 'Team not found' });
    }

    const members = await listTeamMembers(teamId);
    res.json({ ...team.rows[0], members });
  } catch (error) {
    console.error('getTeam:', error);
    res.status(500).json({ error: error.message });
  }
}

async function addTeamMember(req, res) {
  if (!(await canManageTeamAsync(req.user))) {
    return res.status(403).json({ error: 'Unauthorized' });
  }
  try {
    const teamId = req.params.id;
    if (!(await assertCanAccessTeam(req, teamId))) {
      return res.status(403).json({ error: 'Unauthorized' });
    }

    const team = await query(`SELECT id FROM ${SCHEMA}.teams WHERE id = $1::uuid`, [teamId]);
    if (!team.rowCount) {
      return res.status(404).json({ error: 'Team not found' });
    }

    const userIdRaw = req.body?.user_id;
    const emailRaw = req.body?.email != null ? String(req.body.email).trim().toLowerCase() : '';
    const teamRole = normalizeTeamMemberRole(req.body?.role || TEAM_MEMBER_ROLES.MEMBER);

    // Scoped team admins cannot grant team_admin on a team unless they are team_admin there
    // (assertCanAccessTeam already requires that for non-super). Super admin can grant either.
    if (teamRole === TEAM_MEMBER_ROLES.TEAM_ADMIN
      && !isSuperAdmin(req.user.role)
      && !(await isTeamAdminOnTeam(req.user.id, teamId, req.user.role))) {
      return res.status(403).json({ error: 'Only a Team admin on this team can assign Team admin' });
    }

    let target;
    if (userIdRaw != null && userIdRaw !== '') {
      const found = await query(
        `SELECT id, name, email, role, status FROM ${SCHEMA}.users WHERE id = $1`,
        [Number(userIdRaw)]
      );
      target = found.rows[0];
    } else if (emailRaw) {
      const found = await query(
        `SELECT id, name, email, role, status FROM ${SCHEMA}.users WHERE lower(email) = $1`,
        [emailRaw]
      );
      target = found.rows[0];
    } else {
      return res.status(400).json({ error: 'user_id or email is required' });
    }

    if (!target) {
      return res.status(404).json({ error: 'User not found' });
    }
    if (target.status === 'discontinued') {
      return res.status(400).json({ error: 'Cannot add a discontinued user' });
    }
    if (isSuperAdmin(target.role)) {
      return res.status(400).json({
        error: 'Super admins are not added as team members for role management; they already have platform-wide access'
      });
    }

    // Team admins may only attach users they already manage (shared team membership),
    // unless Super admin.
    if (await actorIsScopedTeamAdmin(req.user)) {
      const overlap = await query(
        `SELECT 1
         FROM ${SCHEMA}.team_members mine
         JOIN ${SCHEMA}.team_members theirs
           ON theirs.team_id = mine.team_id AND theirs.user_id = $2
         WHERE mine.user_id = $1
         LIMIT 1`,
        [req.user.id, target.id]
      );
      if (!overlap.rowCount && Number(target.id) !== Number(req.user.id)) {
        return res.status(403).json({
          error: 'Can only add users who already belong to one of your teams'
        });
      }
    }

    const inserted = await query(
      `INSERT INTO ${SCHEMA}.team_members (team_id, user_id, role)
       VALUES ($1::uuid, $2, $3)
       ON CONFLICT (team_id, user_id) DO NOTHING
       RETURNING team_id, user_id, role`,
      [teamId, target.id, teamRole]
    );

    if (!inserted.rowCount) {
      return res.status(409).json({ error: 'User is already a member of this team' });
    }

    // Keep legacy users.role aligned with any team_admin membership
    if (teamRole === TEAM_MEMBER_ROLES.TEAM_ADMIN && !isTeamAdmin(target.role)) {
      await query(
        `UPDATE ${SCHEMA}.users SET role = 'team_admin' WHERE id = $1 AND role NOT IN ('super_admin', 'admin')`,
        [target.id]
      );
    }

    const members = await listTeamMembers(teamId);
    res.status(201).json({
      success: true,
      message: 'Member added',
      member: members.find((m) => Number(m.id) === Number(target.id)),
      members
    });
  } catch (error) {
    console.error('addTeamMember:', error);
    res.status(500).json({ error: error.message });
  }
}

async function updateTeamMember(req, res) {
  if (!(await canManageTeamAsync(req.user))) {
    return res.status(403).json({ error: 'Unauthorized' });
  }
  try {
    const teamId = req.params.id;
    const userId = Number(req.params.userId);
    if (!(await assertCanAccessTeam(req, teamId))) {
      return res.status(403).json({ error: 'Unauthorized' });
    }
    if (!Number.isFinite(userId)) {
      return res.status(400).json({ error: 'Invalid user id' });
    }

    const teamRole = normalizeTeamMemberRole(req.body?.role);
    if (req.body?.role == null) {
      return res.status(400).json({ error: 'role is required' });
    }

    const result = await query(
      `UPDATE ${SCHEMA}.team_members
       SET role = $3
       WHERE team_id = $1::uuid AND user_id = $2
       RETURNING user_id, role`,
      [teamId, userId, teamRole]
    );
    if (!result.rowCount) {
      return res.status(404).json({ error: 'Member not found on this team' });
    }

    if (teamRole === TEAM_MEMBER_ROLES.TEAM_ADMIN) {
      await query(
        `UPDATE ${SCHEMA}.users SET role = 'team_admin'
         WHERE id = $1 AND role NOT IN ('super_admin', 'admin')`,
        [userId]
      );
    } else {
      // Demote platform role only if they are not team_admin on any other team
      await query(
        `UPDATE ${SCHEMA}.users u
         SET role = 'project_user'
         WHERE u.id = $1
           AND u.role = 'team_admin'
           AND NOT EXISTS (
             SELECT 1 FROM ${SCHEMA}.team_members tm
             WHERE tm.user_id = u.id AND tm.role = 'team_admin'
           )`,
        [userId]
      );
    }

    const members = await listTeamMembers(teamId);
    res.json({ success: true, message: 'Member role updated', members });
  } catch (error) {
    console.error('updateTeamMember:', error);
    res.status(500).json({ error: error.message });
  }
}

async function removeTeamMember(req, res) {
  if (!(await canManageTeamAsync(req.user))) {
    return res.status(403).json({ error: 'Unauthorized' });
  }
  try {
    const teamId = req.params.id;
    const userId = Number(req.params.userId);
    if (!(await assertCanAccessTeam(req, teamId))) {
      return res.status(403).json({ error: 'Unauthorized' });
    }
    if (!Number.isFinite(userId)) {
      return res.status(400).json({ error: 'Invalid user id' });
    }

    const result = await query(
      `DELETE FROM ${SCHEMA}.team_members
       WHERE team_id = $1::uuid AND user_id = $2
       RETURNING user_id`,
      [teamId, userId]
    );
    if (!result.rowCount) {
      return res.status(404).json({ error: 'Member not found on this team' });
    }

    await query(
      `UPDATE ${SCHEMA}.users u
       SET role = 'project_user'
       WHERE u.id = $1
         AND u.role = 'team_admin'
         AND NOT EXISTS (
           SELECT 1 FROM ${SCHEMA}.team_members tm
           WHERE tm.user_id = u.id AND tm.role = 'team_admin'
         )`,
      [userId]
    );

    const members = await listTeamMembers(teamId);
    res.json({ success: true, message: 'Member removed', members });
  } catch (error) {
    console.error('removeTeamMember:', error);
    res.status(500).json({ error: error.message });
  }
}

async function updateTeam(req, res) {
  if (!(await canManageTeamAsync(req.user))) {
    return res.status(403).json({ error: 'Unauthorized' });
  }
  try {
    const teamId = req.params.id;
    if (!(await assertCanAccessTeam(req, teamId))) {
      return res.status(403).json({ error: 'Unauthorized' });
    }

    const name = req.body?.name != null ? String(req.body.name).trim() : null;
    const status = req.body?.status != null ? String(req.body.status).trim() : null;

    if (status && !isSuperAdmin(req.user.role)) {
      return res.status(403).json({ error: 'Only Super admins can change team status' });
    }

    const fields = [];
    const params = [];
    if (name) {
      params.push(name);
      fields.push(`name = $${params.length}`);
    }
    if (status) {
      params.push(status);
      fields.push(`status = $${params.length}`);
    }
    if (!fields.length) {
      return res.status(400).json({ error: 'No updates provided' });
    }
    fields.push('updated_at = CURRENT_TIMESTAMP');
    params.push(teamId);

    const result = await query(
      `UPDATE ${SCHEMA}.teams
       SET ${fields.join(', ')}
       WHERE id = $${params.length}::uuid
       RETURNING id, slug, name, status, created_at, updated_at`,
      params
    );
    if (!result.rowCount) {
      return res.status(404).json({ error: 'Team not found' });
    }
    res.json(result.rows[0]);
  } catch (error) {
    if (error.code === '23505') {
      return res.status(409).json({ error: 'A team with this name already exists' });
    }
    console.error('updateTeam:', error);
    res.status(500).json({ error: error.message });
  }
}

export default router;
