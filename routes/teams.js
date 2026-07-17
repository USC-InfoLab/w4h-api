import express from 'express';
import { authenticate } from '../auth.js';
import { query } from '../db.js';
import { isSuperAdmin, isTeamAdmin, canAccessAdminUi } from '../lib/utils/user.js';

const router = express.Router();
const SCHEMA = 'w4h';

router.get('/', authenticate, listTeams);
router.post('/', authenticate, createTeam);
router.get('/:id', authenticate, getTeam);
router.patch('/:id', authenticate, updateTeam);

function slugify(input) {
  return String(input || '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/(^-|-$)/g, '');
}

async function ensureUniqueSlug(base) {
  const baseSlug = slugify(base) || `team-${Date.now()}`;
  let slug = baseSlug;
  let i = 1;
  while (true) {
    const check = await query(`SELECT id FROM ${SCHEMA}.teams WHERE slug = $1`, [slug]);
    if (!check.rowCount) return slug;
    slug = `${baseSlug}-${i}`;
    i += 1;
  }
}

async function listTeams(req, res) {
  if (!canAccessAdminUi(req.user.role)) {
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

    if (isTeamAdmin(req.user.role) && !isSuperAdmin(req.user.role)) {
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
  if (!isSuperAdmin(req.user.role)) {
    return res.status(403).json({ error: 'Only Super admins can create teams' });
  }
  try {
    const name = String(req.body?.name || '').trim();
    if (!name) {
      return res.status(400).json({ error: 'name is required' });
    }
    const slug = await ensureUniqueSlug(req.body?.slug || name);

    const result = await query(
      `INSERT INTO ${SCHEMA}.teams (slug, name)
       VALUES ($1, $2)
       RETURNING id, slug, name, status, created_at, updated_at`,
      [slug, name]
    );

    const team = result.rows[0];

    // Add the creating Super admin as a team member so they appear on the team
    // (platform role stays super_admin — membership is separate).
    if (req.user?.id != null) {
      await query(
        `INSERT INTO ${SCHEMA}.team_members (team_id, user_id)
         VALUES ($1::uuid, $2)
         ON CONFLICT (team_id, user_id) DO NOTHING`,
        [team.id, req.user.id]
      );
    }

    res.status(201).json(team);
  } catch (error) {
    if (error.code === '23505') {
      return res.status(409).json({ error: 'Team slug already exists' });
    }
    console.error('createTeam:', error);
    res.status(500).json({ error: error.message });
  }
}

async function getTeam(req, res) {
  if (!canAccessAdminUi(req.user.role)) {
    return res.status(403).json({ error: 'Unauthorized' });
  }
  try {
    const teamId = req.params.id;
    if (isTeamAdmin(req.user.role) && !isSuperAdmin(req.user.role)) {
      const mine = await query(
        `SELECT 1 FROM ${SCHEMA}.team_members WHERE team_id = $1::uuid AND user_id = $2`,
        [teamId, req.user.id]
      );
      if (!mine.rowCount) {
        return res.status(403).json({ error: 'Unauthorized' });
      }
    }

    const team = await query(
      `SELECT id, slug, name, status, created_at, updated_at
       FROM ${SCHEMA}.teams WHERE id = $1::uuid`,
      [teamId]
    );
    if (!team.rowCount) {
      return res.status(404).json({ error: 'Team not found' });
    }

    const members = await query(
      `SELECT u.id, u.name, u.email, u.role
       FROM ${SCHEMA}.team_members tm
       JOIN ${SCHEMA}.users u ON u.id = tm.user_id
       WHERE tm.team_id = $1::uuid
       ORDER BY u.name ASC`,
      [teamId]
    );

    res.json({ ...team.rows[0], members: members.rows });
  } catch (error) {
    console.error('getTeam:', error);
    res.status(500).json({ error: error.message });
  }
}

async function updateTeam(req, res) {
  if (!isSuperAdmin(req.user.role) && !isTeamAdmin(req.user.role)) {
    return res.status(403).json({ error: 'Unauthorized' });
  }
  try {
    const teamId = req.params.id;
    if (isTeamAdmin(req.user.role) && !isSuperAdmin(req.user.role)) {
      const mine = await query(
        `SELECT 1 FROM ${SCHEMA}.team_members WHERE team_id = $1::uuid AND user_id = $2`,
        [teamId, req.user.id]
      );
      if (!mine.rowCount) {
        return res.status(403).json({ error: 'Unauthorized' });
      }
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
    console.error('updateTeam:', error);
    res.status(500).json({ error: error.message });
  }
}

export default router;
