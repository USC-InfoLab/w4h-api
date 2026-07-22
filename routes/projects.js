import express from 'express';
import crypto from 'crypto';

import { authenticateJwtOrApiKey, requireTeamAdmin } from '../auth.js';
import { authenticateJwtApiKeyOrProjectDbToken } from '../lib/middleware/projectDbTokenAuth.js';
import { query } from '../db.js';
import { isSuperAdmin, isTeamAdmin, isTeamAdminOnTeam } from '../lib/utils/user.js';
import { resolveDatasetProjectLinkColumn } from '../lib/utils/datasetProjectLinkColumn.js';
import { writeAuditLogSafe } from '../lib/utils/audit.js';

const router = express.Router();

router.get('/', authenticateJwtOrApiKey, listProjects);
router.post('/', authenticateJwtOrApiKey, requireTeamAdmin, createProject);

router.get('/:id/db-tokens', authenticateJwtOrApiKey, listProjectDbTokens);
router.delete('/:id/db-tokens/purge-inactive', authenticateJwtOrApiKey, purgeInactiveProjectDbTokens);
router.delete('/:id/db-tokens/:tokenId', authenticateJwtOrApiKey, revokeProjectDbToken);
router.get('/:id/acl', authenticateJwtOrApiKey, getProjectAclRules);
router.put('/:id/acl', authenticateJwtOrApiKey, replaceProjectAclRules);
router.get('/:id/linkable-datasets', authenticateJwtOrApiKey, listLinkableDatasets);

router.get('/:id', authenticateJwtApiKeyOrProjectDbToken, getProjectOverview);

router.patch('/:id', authenticateJwtOrApiKey, updateProject);
router.post('/:id/members', authenticateJwtOrApiKey, addProjectMember);
router.patch('/:id/members/:userId', authenticateJwtOrApiKey, updateProjectMemberRole);
router.delete('/:id/members/:userId', authenticateJwtOrApiKey, removeProjectMember);
router.put('/:id/members', authenticateJwtOrApiKey, replaceProjectMembers);
router.put('/:id/organizations', authenticateJwtOrApiKey, replaceProjectOrganizations);
router.put('/:id/datasets', authenticateJwtOrApiKey, replaceProjectDatasets);
router.put('/:id/notebooks', authenticateJwtOrApiKey, replaceProjectNotebooks);
router.put('/:id/docs', authenticateJwtOrApiKey, replaceProjectDocs);
router.post('/:id/db-token', authenticateJwtOrApiKey, issueProjectDbToken);

const PROJECT_MEMBER_ROLES = new Set(['owner', 'editor', 'viewer']);

function canCreateProject(user) {
  return !!user && (isSuperAdmin(user.role) || isTeamAdmin(user.role));
}

async function getTeamIdsForUser(userId) {
  const res = await query(
    `SELECT team_id::text AS team_id FROM w4h.team_members WHERE user_id = $1`,
    [userId]
  );
  return res.rows.map((r) => r.team_id);
}

async function isUserOnTeam(userId, teamId) {
  if (!userId || !teamId) return false;
  const res = await query(
    `SELECT 1 FROM w4h.team_members WHERE team_id = $1::uuid AND user_id = $2`,
    [teamId, userId]
  );
  return res.rowCount > 0;
}

async function getProjectTeamId(projectId) {
  const res = await query(`SELECT team_id FROM w4h.projects WHERE id = $1::uuid`, [projectId]);
  return res.rowCount ? res.rows[0].team_id : null;
}

async function canAccessTeam(user, teamId) {
  if (!user || !teamId) return false;
  if (isSuperAdmin(user.role)) return true;
  return isUserOnTeam(user.id, teamId);
}

async function getMembership(projectId, userId) {
  const res = await query(
    `SELECT role
     FROM w4h.project_members
     WHERE project_id = $1::uuid AND user_id = $2`,
    [projectId, userId]
  );
  return res.rowCount ? res.rows[0] : null;
}

async function canReadProject(projectId, user) {
  if (!user) return false;
  if (isSuperAdmin(user.role)) return true;
  const member = await getMembership(projectId, user.id);
  if (member) return true;
  if (isTeamAdmin(user.role)) {
    const teamId = await getProjectTeamId(projectId);
    if (teamId && (await isUserOnTeam(user.id, teamId))) return true;
  }
  return false;
}

async function canManageProject(projectId, user) {
  if (!user) return false;
  if (isSuperAdmin(user.role)) return true;
  if (isTeamAdmin(user.role)) {
    const teamId = await getProjectTeamId(projectId);
    if (teamId && (await isUserOnTeam(user.id, teamId))) return true;
  }
  const member = await getMembership(projectId, user.id);
  return !!member && ['owner', 'editor'].includes(member.role);
}

/** Team ids where the user is an effective Team admin (null = Super admin → all teams). */
async function getAdminTeamIdsForUser(user) {
  if (!user?.id) return [];
  if (isSuperAdmin(user.role)) return null;
  const adminRes = await query(
    `SELECT team_id::text AS team_id
     FROM w4h.team_members
     WHERE user_id = $1 AND role = 'team_admin'`,
    [user.id]
  );
  let ids = adminRes.rows.map((r) => r.team_id);
  // Legacy platform team_admin: treat all memberships as admin-scoped
  if (!ids.length && isTeamAdmin(user.role)) {
    const all = await query(
      `SELECT team_id::text AS team_id FROM w4h.team_members WHERE user_id = $1`,
      [user.id]
    );
    ids = all.rows.map((r) => r.team_id);
  }
  return ids;
}

/**
 * Linkable set (ADR-012 + dual-hat Team admin UX):
 * 1) Owned by the project's team (local)
 * 2) Inbound shares to the project's team (shared)
 * 3) Owned by other teams the actor admins — listed so they can link; linking auto-shares to the project team
 */
async function fetchLinkableDatasets(projectId, teamId, user = null) {
  if (!teamId) return [];

  const owned = await query(
    `SELECT d.id, d.name, d.description, d.team_id,
            t.name AS owner_team_name,
            'local'::text AS source_kind,
            NULL::text AS shared_by_team_id,
            NULL::text AS shared_by_team_name,
            'read_write'::text AS max_access_type
     FROM w4h.datasets d
     LEFT JOIN w4h.teams t ON t.id = d.team_id
     WHERE d.team_id = $1::uuid
        OR d.team_id IS NULL
     ORDER BY d.name ASC NULLS LAST, d.id ASC`,
    [teamId]
  );

  const shared = await query(
    `SELECT d.id, d.name, d.description, d.team_id,
            ot.name AS owner_team_name,
            'shared'::text AS source_kind,
            ot.id::text AS shared_by_team_id,
            ot.name AS shared_by_team_name,
            ds.access_type AS max_access_type
     FROM w4h.dataset_shares ds
     JOIN w4h.datasets d ON d.id = ds.dataset_id
     LEFT JOIN w4h.teams ot ON ot.id = d.team_id
     WHERE ds.target_team_id = $1::uuid
       AND (ds.target_project_id IS NULL OR ds.target_project_id = $2::uuid)
     ORDER BY d.name ASC NULLS LAST, d.id ASC`,
    [teamId, projectId]
  );

  const byId = new Map();
  for (const row of owned.rows) {
    byId.set(row.id, {
      ...row,
      source_label: 'This team'
    });
  }
  for (const row of shared.rows) {
    if (byId.has(row.id) && byId.get(row.id).source_kind === 'local') continue;
    byId.set(row.id, {
      ...row,
      source_label: row.shared_by_team_name
        ? `Shared by ${row.shared_by_team_name}`
        : 'Shared'
    });
  }

  if (user) {
    const adminTeamIds = await getAdminTeamIdsForUser(user);
    let managed;
    if (adminTeamIds === null) {
      // Super admin: every other team's datasets
      managed = await query(
        `SELECT d.id, d.name, d.description, d.team_id,
                t.name AS owner_team_name,
                'managed'::text AS source_kind,
                t.id::text AS shared_by_team_id,
                t.name AS shared_by_team_name,
                'read_write'::text AS max_access_type
         FROM w4h.datasets d
         LEFT JOIN w4h.teams t ON t.id = d.team_id
         WHERE d.team_id IS NOT NULL
           AND d.team_id IS DISTINCT FROM $1::uuid
         ORDER BY d.name ASC NULLS LAST, d.id ASC`,
        [teamId]
      );
    } else if (adminTeamIds.length) {
      managed = await query(
        `SELECT d.id, d.name, d.description, d.team_id,
                t.name AS owner_team_name,
                'managed'::text AS source_kind,
                t.id::text AS shared_by_team_id,
                t.name AS shared_by_team_name,
                'read_write'::text AS max_access_type
         FROM w4h.datasets d
         LEFT JOIN w4h.teams t ON t.id = d.team_id
         WHERE d.team_id = ANY($1::uuid[])
           AND d.team_id IS DISTINCT FROM $2::uuid
         ORDER BY d.name ASC NULLS LAST, d.id ASC`,
        [adminTeamIds, teamId]
      );
    }
    if (managed) {
      for (const row of managed.rows) {
        if (byId.has(row.id)) continue;
        byId.set(row.id, {
          ...row,
          source_label: row.owner_team_name
            ? `Your team · ${row.owner_team_name}`
            : 'Your other team'
        });
      }
    }
  }

  return [...byId.values()].sort((a, b) =>
    String(a.name || a.id).localeCompare(String(b.name || b.id), undefined, { sensitivity: 'base' })
  );
}

/** Ensure a team-wide share exists when linking a dataset owned by another team (dual-hat / Super admin). */
async function ensureInboundShareForLink(req, datasetMeta, projectTeamId) {
  if (!datasetMeta?.team_id || String(datasetMeta.team_id) === String(projectTeamId)) {
    return;
  }
  if (datasetMeta.source_kind === 'shared') return;

  const existing = await query(
    `SELECT id FROM w4h.dataset_shares
     WHERE dataset_id = $1
       AND target_team_id = $2::uuid
       AND target_project_id IS NULL
     LIMIT 1`,
    [datasetMeta.id, projectTeamId]
  );
  if (existing.rowCount) return;

  const canSource = await isTeamAdminOnTeam(
    req.user.id,
    datasetMeta.team_id,
    req.user.role
  );
  if (!canSource && !isSuperAdmin(req.user.role)) {
    const err = new Error('Unauthorized to share this dataset into the project team');
    err.status = 403;
    throw err;
  }

  const inserted = await query(
    `INSERT INTO w4h.dataset_shares
       (dataset_id, target_team_id, target_project_id, access_type, shared_by)
     VALUES ($1, $2::uuid, NULL, 'read', $3)
     RETURNING id::text AS id`,
    [datasetMeta.id, projectTeamId, req.user.id]
  );
  writeAuditLogSafe(req, {
    action: 'sharing.create',
    resourceType: 'dataset_share',
    resourceId: inserted.rows[0]?.id,
    details: {
      dataset_id: datasetMeta.id,
      target_team_id: projectTeamId,
      reason: 'auto_share_on_project_link',
    },
  });
}

async function listEligibleTeamMembers(teamId) {
  if (!teamId) return [];
  const res = await query(
    `SELECT u.id AS user_id, u.name, u.email, u.role AS user_role
     FROM w4h.team_members tm
     JOIN w4h.users u ON u.id = tm.user_id
     WHERE tm.team_id = $1::uuid
       AND u.role NOT IN ('admin', 'super_admin')
     ORDER BY u.name ASC NULLS LAST, u.email ASC`,
    [teamId]
  );
  return res.rows;
}

function slugify(input) {
  return String(input || '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/(^-|-$)/g, '');
}

function normalizeOrgWebsiteUrl(row) {
  if (!row || typeof row !== 'object') return null;
  const raw = row.website_url ?? row.websiteUrl ?? row.url ?? row.link;
  if (raw == null) return null;
  const s = String(raw).trim();
  return s || null;
}

async function resolveProjectId(projectRef) {
  const ref = String(projectRef || '').trim();
  if (!ref) return null;
  const found = await query(
    `SELECT id
     FROM w4h.projects
     WHERE id::text = $1 OR slug = $1
     LIMIT 1`,
    [ref]
  );
  return found.rowCount ? found.rows[0].id : null;
}

async function ensureUniqueSlug(base) {
  const baseSlug = slugify(base) || `project-${Date.now()}`;
  let slug = baseSlug;
  let i = 1;
  while (true) {
    const check = await query(`SELECT id FROM w4h.projects WHERE slug = $1`, [slug]);
    if (!check.rowCount) return slug;
    slug = `${baseSlug}-${i}`;
    i += 1;
  }
}

async function listProjects(req, res) {
  try {
    let sql = `
      SELECT p.id, p.slug, p.name, p.description, p.status, p.visibility,
             p.irb_protocol_id, p.irb_expiry_date, p.team_id,
             t.name AS team_name, t.slug AS team_slug,
             p.created_at, p.updated_at
      FROM w4h.projects p
      LEFT JOIN w4h.teams t ON t.id = p.team_id
    `;
    const params = [];

    if (!isSuperAdmin(req.user.role)) {
      const teamIds = await getTeamIdsForUser(req.user.id);
      params.push(req.user.id);
      if (teamIds.length) {
        params.push(teamIds);
        sql += `
          WHERE (
            EXISTS (
              SELECT 1 FROM w4h.project_members pm
              WHERE pm.project_id = p.id AND pm.user_id = $1
            )
            OR p.team_id = ANY($2::uuid[])
          )
        `;
      } else {
        sql += `
          WHERE EXISTS (
            SELECT 1 FROM w4h.project_members pm
            WHERE pm.project_id = p.id AND pm.user_id = $1
          )
        `;
      }
    }

    sql += ` ORDER BY p.updated_at DESC, p.created_at DESC`;

    const projects = await query(sql, params);
    res.json(projects.rows);
  } catch (error) {
    res.status(500).json({ error: error.message || 'Failed to list projects' });
  }
}

async function listLinkableDatasets(req, res) {
  const projectId = await resolveProjectId(req.params.id);
  if (!projectId) return res.status(404).json({ error: 'Project not found' });
  if (!(await canManageProject(projectId, req.user))) {
    return res.status(403).json({ error: 'Unauthorized' });
  }
  try {
    const teamId = await getProjectTeamId(projectId);
    res.json(await fetchLinkableDatasets(projectId, teamId, req.user));
  } catch (error) {
    res.status(500).json({ error: error.message || 'Failed to list linkable datasets' });
  }
}

async function getProjectOverview(req, res) {
  const projectId = await resolveProjectId(req.params.id);
  if (!projectId) return res.status(404).json({ error: 'Project not found' });
  try {
    if (!(await canReadProject(projectId, req.user))) {
      return res.status(403).json({ error: 'Unauthorized' });
    }

    const projectRes = await query(
      `SELECT p.*,
              u.name AS created_by_name,
              u.email AS created_by_email,
              t.name AS team_name,
              t.slug AS team_slug
       FROM w4h.projects p
       LEFT JOIN w4h.users u ON u.id = p.created_by
       LEFT JOIN w4h.teams t ON t.id = p.team_id
       WHERE p.id = $1::uuid`,
      [projectId]
    );
    if (!projectRes.rowCount) {
      return res.status(404).json({ error: 'Project not found' });
    }

    const project = projectRes.rows[0];
    const pc = await resolveDatasetProjectLinkColumn();

    const [membersRes, datasetsRes, notebooksRes, docsRes, orgsRes, derivedOrgsRes, aclRes, linkableDatasets, eligibleMembers] =
      await Promise.all([
        Promise.resolve({ rows: await listMembers(projectId) }),
        query(
          `SELECT pd.dataset_id, pd.access_type, pd.approved_at,
                  d.name, d.description, d.${pc} AS dataset_project_id, d.team_id AS dataset_team_id
           FROM w4h.project_datasets pd
           JOIN w4h.datasets d ON d.id = pd.dataset_id
           WHERE pd.project_id = $1::uuid
           ORDER BY d.name ASC NULLS LAST, d.id ASC`,
          [projectId]
        ),
        query(
          `SELECT id, title, description, purpose, repo_url, repo_ref,
                  notebook_path, runtime_type, launchers, created_by, created_at, updated_at
           FROM w4h.project_notebooks
           WHERE project_id = $1::uuid
           ORDER BY updated_at DESC, created_at DESC`,
          [projectId]
        ),
        query(
          `SELECT d.slug, d.title, d.updated_at, pd.linked_at
           FROM w4h.project_docs pd
           JOIN w4h.docs d ON d.slug = pd.doc_slug
           WHERE pd.project_id = $1::uuid
           ORDER BY d.updated_at DESC, pd.linked_at DESC`,
          [projectId]
        ),
        query(
          `SELECT id, short, name, website_url, sort_order
           FROM w4h.project_organizations
           WHERE project_id = $1::uuid
           ORDER BY sort_order ASC, name ASC`,
          [projectId]
        ),
        query(
          `SELECT DISTINCT s.id AS dataset_project_id, s.name AS dataset_project_name
           FROM w4h.project_datasets pd
           JOIN w4h.datasets d ON d.id = pd.dataset_id
           LEFT JOIN w4h.projects s ON s.id::text = d.${pc}::text
           WHERE pd.project_id = $1::uuid
           ORDER BY s.name ASC NULLS LAST`,
          [projectId]
        ),
        query(
          `SELECT id, subject_type, subject_key, resource_type, resource_key, permission, effect, created_at
           FROM w4h.project_acl_rules
           WHERE project_id = $1::uuid
           ORDER BY created_at ASC`,
          [projectId]
        ),
        fetchLinkableDatasets(projectId, project.team_id, req.user),
        listEligibleTeamMembers(project.team_id)
      ]);

    const canManage = !req.authViaProjectDbToken && (await canManageProject(projectId, req.user));
    const canIssueDbToken = !req.authViaProjectDbToken && (await canReadProject(projectId, req.user));
    const manualOrgRows = Array.isArray(orgsRes.rows) ? orgsRes.rows : [];
    const organizations =
      manualOrgRows.length > 0
        ? manualOrgRows
        : derivedOrgsRes.rows.map((o, idx) => ({
            id: `derived-${idx}`,
            short: String(o.dataset_project_id || '').slice(0, 4).toUpperCase() || 'ORG',
            name: o.dataset_project_name || o.dataset_project_id || 'Project',
            website_url: null,
            sort_order: idx
          }));

    const linkableById = new Map(linkableDatasets.map((d) => [String(d.id), d]));
    const datasets = datasetsRes.rows.map((row) => {
      const meta = linkableById.get(String(row.dataset_id));
      return {
        ...row,
        source_kind: meta?.source_kind || 'local',
        source_label: meta?.source_label || 'Local',
        shared_by_team_name: meta?.shared_by_team_name || null
      };
    });

    res.json({
      project,
      members: membersRes.rows,
      datasets,
      linkable_datasets: linkableDatasets,
      eligible_members: eligibleMembers,
      notebooks: notebooksRes.rows,
      docs: docsRes.rows,
      organizations,
      acl_rules: aclRes.rows,
      capabilities: {
        can_manage: canManage,
        can_issue_db_token: canIssueDbToken,
        auth_via_project_token: Boolean(req.authViaProjectDbToken)
      }
    });
  } catch (error) {
    res.status(500).json({ error: error.message || 'Failed to load project overview' });
  }
}

async function createProject(req, res) {
  if (!canCreateProject(req.user)) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  try {
    if (req.user?.id == null) {
      return res.status(401).json({
        error: 'Session is missing user id. Sign out and sign in again, then retry.'
      });
    }

    const name = String(req.body?.name || '').trim();
    const description = req.body?.description == null ? null : String(req.body.description).trim();
    const status = ['active', 'archived'].includes(req.body?.status) ? req.body.status : 'active';
    const visibility = ['org-private', 'project-private', 'internal'].includes(req.body?.visibility)
      ? req.body.visibility
      : 'org-private';
    const irbProtocolId = req.body?.irb_protocol_id == null ? null : String(req.body.irb_protocol_id).trim();
    const irbExpiryDate = req.body?.irb_expiry_date || null;
    let teamId = req.body?.team_id == null ? null : String(req.body.team_id).trim();

    if (!name) return res.status(400).json({ error: 'name is required' });

    if (!teamId && isTeamAdmin(req.user.role) && !isSuperAdmin(req.user.role)) {
      const myTeams = await getTeamIdsForUser(req.user.id);
      if (myTeams.length === 1) teamId = myTeams[0];
    }

    if (!teamId) {
      return res.status(400).json({ error: 'team_id is required' });
    }

    const teamCheck = await query(`SELECT id FROM w4h.teams WHERE id = $1::uuid`, [teamId]);
    if (!teamCheck.rowCount) {
      return res.status(400).json({ error: 'Unknown team_id' });
    }

    if (!(await canAccessTeam(req.user, teamId))) {
      return res.status(403).json({ error: 'Unauthorized for this team' });
    }

    const slug = await ensureUniqueSlug(req.body?.slug || name);
    const inserted = await query(
      `INSERT INTO w4h.projects
         (slug, name, description, status, visibility, irb_protocol_id, irb_expiry_date, created_by, team_id)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::uuid)
       RETURNING *`,
      [slug, name, description, status, visibility, irbProtocolId || null, irbExpiryDate, req.user.id, teamId]
    );

    const project = inserted.rows[0];

    // Team admins (and other non–super-admin creators on the team) become project owner
    if (!(await isPlatformAdminUserId(req.user.id))) {
      await query(
        `INSERT INTO w4h.project_members (project_id, user_id, role, invited_by)
         VALUES ($1::uuid, $2, 'owner', $2)
         ON CONFLICT (project_id, user_id) DO NOTHING`,
        [project.id, req.user.id]
      );
    }

    res.status(201).json(project);
  } catch (error) {
    res.status(500).json({ error: error.message || 'Failed to create project' });
  }
}

async function replaceProjectOrganizationsData(projectId, organizations) {
  const rows = Array.isArray(organizations) ? organizations : [];
  await query(`DELETE FROM w4h.project_organizations WHERE project_id = $1::uuid`, [projectId]);
  let sortOrder = 0;
  for (const row of rows) {
    const name = String(row?.name || '').trim();
    if (!name) continue;
    const short = row?.short == null ? null : String(row.short).trim();
    const websiteUrl = normalizeOrgWebsiteUrl(row);
    await query(
      `INSERT INTO w4h.project_organizations (project_id, short, name, website_url, sort_order)
       VALUES ($1::uuid, $2, $3, $4, $5)`,
      [projectId, short || null, name, websiteUrl, sortOrder]
    );
    sortOrder += 1;
  }
}

async function updateProject(req, res) {
  const projectId = await resolveProjectId(req.params.id);
  if (!projectId) return res.status(404).json({ error: 'Project not found' });
  if (!(await canManageProject(projectId, req.user))) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  try {
    const fields = [];
    const params = [];
    const allowed = [
      ['name', 'name'],
      ['description', 'description'],
      ['status', 'status'],
      ['visibility', 'visibility'],
      ['irb_protocol_id', 'irb_protocol_id'],
      ['irb_expiry_date', 'irb_expiry_date']
    ];

    for (const [bodyKey, column] of allowed) {
      if (Object.prototype.hasOwnProperty.call(req.body || {}, bodyKey)) {
        params.push(req.body[bodyKey] === '' ? null : req.body[bodyKey]);
        fields.push(`${column} = $${params.length}`);
      }
    }

    if (Object.prototype.hasOwnProperty.call(req.body || {}, 'team_id')) {
      if (!isSuperAdmin(req.user.role)) {
        return res.status(403).json({ error: 'Only Super admins can change the owning team' });
      }
      const teamId = req.body.team_id == null || req.body.team_id === ''
        ? null
        : String(req.body.team_id).trim();
      if (teamId) {
        const teamCheck = await query(`SELECT id FROM w4h.teams WHERE id = $1::uuid`, [teamId]);
        if (!teamCheck.rowCount) {
          return res.status(400).json({ error: 'Unknown team_id' });
        }
      }
      params.push(teamId);
      fields.push(`team_id = $${params.length}::uuid`);
    }

    // Launchers unused in Flexible mode / TRE — ignore payload if present

    const orgPayload = req.body?.organizations;
    const shouldSyncOrgs = Array.isArray(orgPayload);
    if (!fields.length && !shouldSyncOrgs) {
      return res.status(400).json({ error: 'No updatable fields provided' });
    }

    let projectRow = null;
    if (fields.length) {
      params.push(projectId);
      const updated = await query(
        `UPDATE w4h.projects
         SET ${fields.join(', ')}, updated_at = NOW()
         WHERE id = $${params.length}::uuid
         RETURNING *`,
        params
      );
      if (!updated.rowCount) return res.status(404).json({ error: 'Project not found' });
      projectRow = updated.rows[0];
    }

    let organizationsOut = null;
    if (shouldSyncOrgs) {
      await replaceProjectOrganizationsData(projectId, orgPayload);
      const orgsFresh = await query(
        `SELECT id, short, name, website_url, sort_order
         FROM w4h.project_organizations
         WHERE project_id = $1::uuid
         ORDER BY sort_order ASC, name ASC`,
        [projectId]
      );
      organizationsOut = orgsFresh.rows;
    }

    if (!projectRow) {
      const cur = await query(`SELECT * FROM w4h.projects WHERE id = $1::uuid`, [projectId]);
      if (!cur.rowCount) return res.status(404).json({ error: 'Project not found' });
      projectRow = cur.rows[0];
    }

    if (organizationsOut) {
      res.json({ ...projectRow, organizations: organizationsOut });
    } else {
      res.json(projectRow);
    }
  } catch (error) {
    res.status(500).json({ error: error.message || 'Failed to update project' });
  }
}

async function resolveMemberUserId(row) {
  let userId = Number(row?.user_id);
  if (Number.isInteger(userId)) return userId;
  const email = row?.email == null ? '' : String(row.email).trim();
  if (!email) return null;
  const found = await query(`SELECT id FROM w4h.users WHERE lower(trim(email)) = lower(trim($1))`, [email]);
  return found.rowCount ? found.rows[0].id : null;
}

async function isPlatformAdminUserId(userId) {
  const res = await query(`SELECT role FROM w4h.users WHERE id = $1`, [userId]);
  return res.rowCount > 0 && isSuperAdmin(res.rows[0].role);
}

async function listMembers(projectId) {
  const fresh = await query(
    `SELECT pm.user_id, pm.role, pm.joined_at, u.name, u.email, u.role AS user_role
     FROM w4h.project_members pm
     JOIN w4h.users u ON u.id = pm.user_id
     WHERE pm.project_id = $1::uuid
       AND u.role NOT IN ('admin', 'super_admin')
     ORDER BY pm.joined_at ASC`,
    [projectId]
  );
  return fresh.rows;
}

async function ownerCount(projectId) {
  const result = await query(
    `SELECT COUNT(*)::int AS count
     FROM w4h.project_members
     WHERE project_id = $1::uuid AND role = 'owner'`,
    [projectId]
  );
  return result.rows[0]?.count || 0;
}

async function addProjectMember(req, res) {
  const projectId = await resolveProjectId(req.params.id);
  if (!projectId) return res.status(404).json({ error: 'Project not found' });
  if (!(await canManageProject(projectId, req.user))) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  try {
    const role = String(req.body?.role || 'viewer');
    if (!PROJECT_MEMBER_ROLES.has(role)) {
      return res.status(400).json({ error: 'Invalid role' });
    }
    const userId = await resolveMemberUserId(req.body || {});
    if (!Number.isInteger(userId)) {
      const email = String(req.body?.email || '').trim();
      return res.status(400).json({ error: email ? `Unknown user email: ${email}` : 'Member user_id or email is required' });
    }
    if (await isPlatformAdminUserId(userId)) {
      return res.status(400).json({
        error: 'Platform admin accounts are not project members. Use Created by for the provisioning admin; add team members here.'
      });
    }

    const teamId = await getProjectTeamId(projectId);
    if (!teamId) {
      return res.status(400).json({ error: 'Project has no owning team; set team_id before adding members' });
    }
    if (!(await isUserOnTeam(userId, teamId))) {
      return res.status(400).json({ error: 'Member must belong to the project owning team' });
    }

    await query(
      `INSERT INTO w4h.project_members (project_id, user_id, role, invited_by)
       VALUES ($1::uuid, $2, $3, $4)
       ON CONFLICT (project_id, user_id) DO UPDATE SET role = EXCLUDED.role`,
      [projectId, userId, role, req.user.id]
    );

    res.json(await listMembers(projectId));
  } catch (error) {
    res.status(500).json({ error: error.message || 'Failed to add project member' });
  }
}

async function updateProjectMemberRole(req, res) {
  const projectId = await resolveProjectId(req.params.id);
  if (!projectId) return res.status(404).json({ error: 'Project not found' });
  const userId = Number(req.params.userId);
  if (!(await canManageProject(projectId, req.user))) {
    return res.status(403).json({ error: 'Unauthorized' });
  }
  if (!Number.isInteger(userId)) {
    return res.status(400).json({ error: 'Invalid user id' });
  }
  const role = String(req.body?.role || '');
  if (!PROJECT_MEMBER_ROLES.has(role)) {
    return res.status(400).json({ error: 'Invalid role' });
  }

  try {
    const current = await query(
      `SELECT role FROM w4h.project_members WHERE project_id = $1::uuid AND user_id = $2`,
      [projectId, userId]
    );
    if (!current.rowCount) {
      return res.status(404).json({ error: 'Member not found' });
    }
    if (current.rows[0].role === 'owner' && role !== 'owner' && (await ownerCount(projectId)) <= 1) {
      return res.status(400).json({ error: 'Project must have at least one owner' });
    }

    await query(
      `UPDATE w4h.project_members
       SET role = $1
       WHERE project_id = $2::uuid AND user_id = $3`,
      [role, projectId, userId]
    );
    res.json(await listMembers(projectId));
  } catch (error) {
    res.status(500).json({ error: error.message || 'Failed to update member role' });
  }
}

async function removeProjectMember(req, res) {
  const projectId = await resolveProjectId(req.params.id);
  if (!projectId) return res.status(404).json({ error: 'Project not found' });
  const userId = Number(req.params.userId);
  if (!(await canManageProject(projectId, req.user))) {
    return res.status(403).json({ error: 'Unauthorized' });
  }
  if (!Number.isInteger(userId)) {
    return res.status(400).json({ error: 'Invalid user id' });
  }

  try {
    const current = await query(
      `SELECT role FROM w4h.project_members WHERE project_id = $1::uuid AND user_id = $2`,
      [projectId, userId]
    );
    if (!current.rowCount) {
      return res.status(404).json({ error: 'Member not found' });
    }
    if (current.rows[0].role === 'owner' && (await ownerCount(projectId)) <= 1) {
      return res.status(400).json({ error: 'Project must have at least one owner' });
    }

    await query(
      `DELETE FROM w4h.project_members
       WHERE project_id = $1::uuid AND user_id = $2`,
      [projectId, userId]
    );
    res.json(await listMembers(projectId));
  } catch (error) {
    res.status(500).json({ error: error.message || 'Failed to remove member' });
  }
}

async function replaceProjectMembers(req, res) {
  const projectId = await resolveProjectId(req.params.id);
  if (!projectId) return res.status(404).json({ error: 'Project not found' });
  if (!(await canManageProject(projectId, req.user))) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  try {
    const members = Array.isArray(req.body?.members) ? req.body.members : [];
    const resolved = [];
    const unresolvedEmails = [];
    const teamId = await getProjectTeamId(projectId);
    if (!teamId) {
      return res.status(400).json({ error: 'Project has no owning team; set team_id before managing members' });
    }

    for (const row of members) {
      const role = String(row?.role || 'viewer');
      if (!PROJECT_MEMBER_ROLES.has(role)) continue;
      const userId = await resolveMemberUserId(row);
      if (!Number.isInteger(userId)) {
        const email = row?.email == null ? '' : String(row.email).trim();
        if (email) unresolvedEmails.push(email);
        continue;
      }
      if (await isPlatformAdminUserId(userId)) continue;
      if (!(await isUserOnTeam(userId, teamId))) {
        return res.status(400).json({
          error: `User must belong to the project owning team (user_id ${userId})`
        });
      }
      resolved.push({ userId, role });
    }

    if (unresolvedEmails.length) {
      return res.status(400).json({
        error: `Unknown user email(s): ${[...new Set(unresolvedEmails)].join(', ')}`
      });
    }

    const byUser = new Map();
    for (const { userId, role } of resolved) {
      byUser.set(userId, role);
    }
    const finalList = [...byUser.entries()].map(([userId, role]) => ({ userId, role }));

    if (!finalList.some((m) => m.role === 'owner')) {
      return res.status(400).json({ error: 'At least one member must have the owner role' });
    }

    await query(`DELETE FROM w4h.project_members WHERE project_id = $1::uuid`, [projectId]);

    for (const { userId, role } of finalList) {
      await query(
        `INSERT INTO w4h.project_members (project_id, user_id, role, invited_by)
         VALUES ($1::uuid, $2, $3, $4)
         ON CONFLICT (project_id, user_id) DO UPDATE SET role = EXCLUDED.role`,
        [projectId, userId, role, req.user.id]
      );
    }

    const fresh = await query(
      `SELECT pm.user_id, pm.role, pm.joined_at, u.name, u.email, u.role AS user_role
       FROM w4h.project_members pm
       JOIN w4h.users u ON u.id = pm.user_id
       WHERE pm.project_id = $1::uuid
       ORDER BY pm.joined_at ASC`,
      [projectId]
    );
    res.json(fresh.rows);
  } catch (error) {
    res.status(500).json({ error: error.message || 'Failed to update project members' });
  }
}

async function replaceProjectDatasets(req, res) {
  const projectId = await resolveProjectId(req.params.id);
  if (!projectId) return res.status(404).json({ error: 'Project not found' });
  if (!(await canManageProject(projectId, req.user))) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  try {
    const datasets = Array.isArray(req.body?.datasets) ? req.body.datasets : [];
    const teamId = await getProjectTeamId(projectId);
    const linkable = await fetchLinkableDatasets(projectId, teamId, req.user);
    const linkableById = new Map(linkable.map((d) => [String(d.id), d]));

    await query(`DELETE FROM w4h.project_datasets WHERE project_id = $1::uuid`, [projectId]);

    for (const row of datasets) {
      const datasetId = String(row?.dataset_id || '').trim();
      if (!datasetId) continue;
      const meta = linkableById.get(datasetId);
      if (!meta) {
        return res.status(400).json({
          error: `Dataset is not linkable to this project: ${datasetId}`
        });
      }
      try {
        await ensureInboundShareForLink(req, meta, teamId);
      } catch (shareErr) {
        return res.status(shareErr.status || 400).json({
          error: shareErr.message || 'Failed to share dataset into project team'
        });
      }
      let accessType = row?.access_type === 'read_write' ? 'read_write' : 'read';
      if (accessType === 'read_write' && meta.max_access_type !== 'read_write') {
        accessType = 'read';
      }
      await query(
        `INSERT INTO w4h.project_datasets (project_id, dataset_id, access_type, approved_by)
         VALUES ($1::uuid, $2, $3, $4)
         ON CONFLICT (project_id, dataset_id)
         DO UPDATE SET access_type = EXCLUDED.access_type, approved_by = EXCLUDED.approved_by, approved_at = NOW()`,
        [projectId, datasetId, accessType, req.user.id]
      );
    }

    const pc = await resolveDatasetProjectLinkColumn();
    const fresh = await query(
      `SELECT pd.dataset_id, pd.access_type, pd.approved_at, d.name, d.description, d.${pc} AS dataset_project_id
       FROM w4h.project_datasets pd
       JOIN w4h.datasets d ON d.id = pd.dataset_id
       WHERE pd.project_id = $1::uuid
       ORDER BY d.name ASC NULLS LAST, d.id ASC`,
      [projectId]
    );
    res.json(fresh.rows);
  } catch (error) {
    res.status(500).json({ error: error.message || 'Failed to update project datasets' });
  }
}

async function replaceProjectOrganizations(req, res) {
  const projectId = await resolveProjectId(req.params.id);
  if (!projectId) return res.status(404).json({ error: 'Project not found' });
  if (!(await canManageProject(projectId, req.user))) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  try {
    await replaceProjectOrganizationsData(projectId, req.body?.organizations);
    const fresh = await query(
      `SELECT id, short, name, website_url, sort_order
       FROM w4h.project_organizations
       WHERE project_id = $1::uuid
       ORDER BY sort_order ASC, name ASC`,
      [projectId]
    );
    res.json(fresh.rows);
  } catch (error) {
    res.status(500).json({ error: error.message || 'Failed to update organizations' });
  }
}

async function replaceProjectNotebooks(req, res) {
  const projectId = await resolveProjectId(req.params.id);
  if (!projectId) return res.status(404).json({ error: 'Project not found' });
  if (!(await canManageProject(projectId, req.user))) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  try {
    const notebooks = Array.isArray(req.body?.notebooks) ? req.body.notebooks : [];
    await query(`DELETE FROM w4h.project_notebooks WHERE project_id = $1::uuid`, [projectId]);

    for (const row of notebooks) {
      const title = String(row?.title || '').trim();
      const notebookPath = String(row?.notebook_path || '').trim();
      if (!title || !notebookPath) continue;
      const purpose = ['exploration', 'preprocessing', 'analysis', 'visualization', 'report'].includes(row?.purpose)
        ? row.purpose
        : 'analysis';
      const runtimeType = ['jupyter', 'rstudio', 'vscode', 'quarto', 'other'].includes(row?.runtime_type)
        ? row.runtime_type
        : 'jupyter';
      const launchers = Array.isArray(row?.launchers) ? row.launchers : ['jupyter'];
      const optionalId =
        row?.id && /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(String(row.id))
          ? String(row.id)
          : null;
      await query(
        `INSERT INTO w4h.project_notebooks
           (id, project_id, title, description, purpose, repo_url, repo_ref, notebook_path, runtime_type, launchers, created_by, updated_at)
         VALUES (COALESCE($1::uuid, gen_random_uuid()), $2::uuid, $3, $4, $5, $6, $7, $8, $9, $10::jsonb, $11, NOW())`,
        [
          optionalId,
          projectId,
          title,
          row?.description == null ? null : String(row.description),
          purpose,
          row?.repo_url == null ? null : String(row.repo_url),
          row?.repo_ref == null ? null : String(row.repo_ref),
          notebookPath,
          runtimeType,
          JSON.stringify(launchers),
          req.user.id
        ]
      );
    }

    const fresh = await query(
      `SELECT id, title, description, purpose, repo_url, repo_ref, notebook_path, runtime_type,
              launchers, created_at, updated_at
       FROM w4h.project_notebooks
       WHERE project_id = $1::uuid
       ORDER BY updated_at DESC, created_at DESC`,
      [projectId]
    );
    res.json(fresh.rows);
  } catch (error) {
    res.status(500).json({ error: error.message || 'Failed to update project notebooks' });
  }
}

async function replaceProjectDocs(req, res) {
  const projectId = await resolveProjectId(req.params.id);
  if (!projectId) return res.status(404).json({ error: 'Project not found' });
  if (!(await canManageProject(projectId, req.user))) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  try {
    const docSlugs = Array.isArray(req.body?.doc_slugs)
      ? [...new Set(req.body.doc_slugs.map(v => String(v || '').trim()).filter(Boolean))]
      : [];

    await query(`DELETE FROM w4h.project_docs WHERE project_id = $1::uuid`, [projectId]);

    for (const slug of docSlugs) {
      await query(
        `INSERT INTO w4h.project_docs (project_id, doc_slug, linked_by)
         VALUES ($1::uuid, $2, $3)
         ON CONFLICT (project_id, doc_slug) DO NOTHING`,
        [projectId, slug, req.user.id]
      );
    }

    const fresh = await query(
      `SELECT d.slug, d.title, d.updated_at, pd.linked_at
       FROM w4h.project_docs pd
       JOIN w4h.docs d ON d.slug = pd.doc_slug
       WHERE pd.project_id = $1::uuid
       ORDER BY d.updated_at DESC, pd.linked_at DESC`,
      [projectId]
    );
    res.json(fresh.rows);
  } catch (error) {
    res.status(500).json({ error: error.message || 'Failed to update project docs' });
  }
}

async function listProjectDbTokens(req, res) {
  const projectId = await resolveProjectId(req.params.id);
  if (!projectId) return res.status(404).json({ error: 'Project not found' });
  if (!(await canManageProject(projectId, req.user))) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  try {
    const historyRaw = req.query?.history;
    const history =
      historyRaw === 1 ||
      historyRaw === true ||
      String(historyRaw ?? '').trim() === '1' ||
      String(historyRaw ?? '').trim().toLowerCase() === 'true';
    let sql = `
      SELECT t.id, t.user_id, t.issued_at, t.expires_at, t.revoked_at,
             u.email, u.name
       FROM w4h.project_db_tokens t
       JOIN w4h.users u ON u.id = t.user_id
       WHERE t.project_id = $1::uuid`;
    const params = [projectId];
    if (!history) {
      sql += ` AND t.revoked_at IS NULL AND t.expires_at > NOW()`;
    }
    sql += ` ORDER BY t.issued_at DESC`;
    const rows = await query(sql, params);
    res.json(rows.rows);
  } catch (error) {
    res.status(500).json({ error: error.message || 'Failed to list tokens' });
  }
}

async function purgeInactiveProjectDbTokens(req, res) {
  const projectId = await resolveProjectId(req.params.id);
  if (!projectId) return res.status(404).json({ error: 'Project not found' });
  if (!(await canManageProject(projectId, req.user))) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  try {
    const result = await query(
      `DELETE FROM w4h.project_db_tokens
       WHERE project_id = $1::uuid
         AND (revoked_at IS NOT NULL OR expires_at < NOW())
       RETURNING id`,
      [projectId]
    );
    res.json({ deleted: result.rowCount });
  } catch (error) {
    res.status(500).json({ error: error.message || 'Failed to purge tokens' });
  }
}

async function revokeProjectDbToken(req, res) {
  const projectId = await resolveProjectId(req.params.id);
  if (!projectId) return res.status(404).json({ error: 'Project not found' });
  const tokenId = req.params.tokenId;
  if (!(await canManageProject(projectId, req.user))) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  try {
    const updated = await query(
      `UPDATE w4h.project_db_tokens
       SET revoked_at = NOW()
       WHERE id = $1::uuid AND project_id = $2::uuid AND revoked_at IS NULL
       RETURNING id`,
      [tokenId, projectId]
    );
    if (!updated.rowCount) {
      return res.status(404).json({ error: 'Token not found or already revoked' });
    }
    res.json({ success: true, id: updated.rows[0].id });
  } catch (error) {
    res.status(500).json({ error: error.message || 'Failed to revoke token' });
  }
}

async function replaceProjectAclRules(req, res) {
  const projectId = await resolveProjectId(req.params.id);
  if (!projectId) return res.status(404).json({ error: 'Project not found' });
  if (!(await canManageProject(projectId, req.user))) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  const rules = Array.isArray(req.body?.rules) ? req.body.rules : [];
  const SUBJECT = new Set(['role', 'user']);
  const RESOURCE = new Set(['project', 'dataset', 'doc', 'derived_table', 'notebook']);
  const PERM = new Set(['view', 'edit', 'manage_members', 'link_dataset', 'create_derived', 'launch_notebook']);
  const EFFECT = new Set(['allow', 'deny']);

  const cleaned = [];
  for (const row of rules) {
    const subjectType = String(row?.subject_type || '');
    const subjectKey = String(row?.subject_key || '').trim();
    const resourceType = String(row?.resource_type || '');
    const resourceKey =
      row?.resource_key == null || row.resource_key === '' ? null : String(row.resource_key).trim();
    const permission = String(row?.permission || '');
    const effect = EFFECT.has(row?.effect) ? row.effect : 'allow';
    if (!SUBJECT.has(subjectType) || !subjectKey || !RESOURCE.has(resourceType) || !PERM.has(permission)) {
      return res.status(400).json({ error: 'Each rule needs valid subject_type, subject_key, resource_type, and permission' });
    }
    cleaned.push({
      subjectType,
      subjectKey,
      resourceType,
      resourceKey,
      permission,
      effect
    });
  }

  try {
    await query(`DELETE FROM w4h.project_acl_rules WHERE project_id = $1::uuid`, [projectId]);
    for (const r of cleaned) {
      await query(
        `INSERT INTO w4h.project_acl_rules
         (project_id, subject_type, subject_key, resource_type, resource_key, permission, effect, created_by)
         VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8)`,
        [
          projectId,
          r.subjectType,
          r.subjectKey,
          r.resourceType,
          r.resourceKey,
          r.permission,
          r.effect,
          req.user.id
        ]
      );
    }
    const fresh = await query(
      `SELECT id, subject_type, subject_key, resource_type, resource_key, permission, effect, created_at
       FROM w4h.project_acl_rules
       WHERE project_id = $1::uuid
       ORDER BY created_at ASC`,
      [projectId]
    );
    res.json(fresh.rows);
  } catch (error) {
    res.status(500).json({ error: error.message || 'Failed to save ACL rules' });
  }
}

async function getProjectAclRules(req, res) {
  const projectId = await resolveProjectId(req.params.id);
  if (!projectId) return res.status(404).json({ error: 'Project not found' });
  if (!(await canManageProject(projectId, req.user))) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  try {
    const rows = await query(
      `SELECT id, subject_type, subject_key, resource_type, resource_key, permission, effect, created_at
       FROM w4h.project_acl_rules
       WHERE project_id = $1::uuid
       ORDER BY created_at ASC`,
      [projectId]
    );
    res.json(rows.rows);
  } catch (error) {
    res.status(500).json({ error: error.message || 'Failed to load ACL rules' });
  }
}

async function issueProjectDbToken(req, res) {
  const projectId = await resolveProjectId(req.params.id);
  if (!projectId) return res.status(404).json({ error: 'Project not found' });
  if (req.authViaProjectDbToken) {
    return res.status(403).json({ error: 'Issue tokens using app login, not a project token' });
  }
  if (!(await canReadProject(projectId, req.user))) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  try {
    const expiresRes = await query(
      `SELECT NOW() + interval '2 days' AS expires_at`
    );
    const expiresAt = expiresRes.rows[0].expires_at;

    const token = crypto.randomBytes(24).toString('base64url');
    const tokenHash = crypto.createHash('sha256').update(token).digest('hex');

    await query(
      `INSERT INTO w4h.project_db_tokens (project_id, user_id, token_hash, expires_at)
       VALUES ($1::uuid, $2, $3, $4)`,
      [projectId, req.user.id, tokenHash, expiresAt]
    );

    res.json({
      token,
      token_type: 'project_db_access',
      project_id: projectId,
      expires_at: expiresAt
    });
  } catch (error) {
    res.status(500).json({ error: error.message || 'Failed to issue DB token' });
  }
}

export default router;
