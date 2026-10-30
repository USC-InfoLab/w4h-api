import express from 'express';
import crypto from 'crypto';

import { authenticateJwtOrApiKey, requireTeamAdmin } from '../auth.js';
import { authenticateJwtApiKeyOrProjectDbToken } from '../lib/middleware/projectDbTokenAuth.js';
import { query } from '../db.js';
import { isSuperAdmin, canManageTeamAsync, isTeamAdminOnTeam, USER_ROLES } from '../lib/utils/user.js';
import { deliverInvitation } from '../lib/invitations.js';
import { resolveDatasetProjectLinkColumn } from '../lib/utils/datasetProjectLinkColumn.js';
import { writeAuditLogSafe } from '../lib/utils/audit.js';
import { REPORT_MAX_BYTES, injectReportCsp, looksLikeHtml, sha256 } from '../lib/reports.js';
import { renderDoc, stripLeadingTitleHeading } from '../lib/utils/md-render.js';
import { attachMetadataSummaries } from '../lib/dataset-metadata-summary.js';

const router = express.Router();

function auditProjectAction(req, action, projectId, extra = {}) {
  if (!projectId) return;
  writeAuditLogSafe(req, {
    action,
    resourceType: extra.resourceType || 'project',
    resourceId: extra.resourceId != null ? extra.resourceId : projectId,
    details: {
      project_id: String(projectId),
      ...(extra.details || {}),
    },
  });
}

router.get('/', authenticateJwtOrApiKey, listProjects);
router.post('/', authenticateJwtOrApiKey, requireTeamAdmin, createProject);

router.get('/collaboration-invites', authenticateJwtOrApiKey, listCollaborationInvites);
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
router.post('/:id/datasets/:datasetId/link', authenticateJwtOrApiKey, linkProjectDataset);
router.delete('/:id/datasets/:datasetId', authenticateJwtOrApiKey, unlinkProjectDataset);
router.get('/:id/invitable-teams', authenticateJwtOrApiKey, listInvitableTeams);
router.post('/:id/teams', authenticateJwtOrApiKey, inviteProjectTeam);
router.post('/:id/teams/:teamId/accept', authenticateJwtOrApiKey, acceptProjectTeam);
router.delete('/:id/teams/:teamId', authenticateJwtOrApiKey, removeProjectTeam);
router.put('/:id/docs', authenticateJwtOrApiKey, replaceProjectDocs);
router.get('/:id/pages', authenticateJwtOrApiKey, listProjectPages);
router.post('/:id/pages', authenticateJwtOrApiKey, createProjectPage);
router.get('/:id/pages/:slug', authenticateJwtOrApiKey, getProjectPage);
router.put('/:id/pages/:slug', authenticateJwtOrApiKey, updateProjectPage);
router.delete('/:id/pages/:slug', authenticateJwtOrApiKey, deleteProjectPage);
router.post('/:id/db-token', authenticateJwtOrApiKey, issueProjectDbToken);
router.get('/:id/reports', authenticateJwtOrApiKey, listProjectReports);
router.post('/:id/reports', authenticateJwtOrApiKey, publishProjectReport);
router.get('/:id/reports/:reportId/content', authenticateJwtOrApiKey, getProjectReportContent);
router.delete('/:id/reports/:reportId', authenticateJwtOrApiKey, deleteProjectReport);

const PROJECT_MEMBER_ROLES = new Set(['owner', 'editor', 'viewer']);

async function canCreateProject(user) {
  return canManageTeamAsync(user);
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

/** Owning team plus active collaborating teams (ADR-027). */
async function getProjectTeamIds(projectId) {
  const res = await query(
    `SELECT p.team_id::text AS team_id FROM w4h.projects p WHERE p.id = $1::uuid AND p.team_id IS NOT NULL
     UNION
     SELECT pt.team_id::text FROM w4h.project_teams pt
     WHERE pt.project_id = $1::uuid AND pt.status = 'active'`,
    [projectId]
  );
  return res.rows.map((r) => r.team_id);
}

async function isUserOnProjectTeams(userId, projectId) {
  const teamIds = await getProjectTeamIds(projectId);
  if (!teamIds.length || userId == null) return false;
  const res = await query(
    `SELECT 1 FROM w4h.team_members WHERE user_id = $1 AND team_id = ANY($2::uuid[]) LIMIT 1`,
    [userId, teamIds]
  );
  return res.rowCount > 0;
}

/** Team admin by per-team membership role (never Super admin). */
async function isTeamAdminOfTeam(user, teamId) {
  if (!user || !teamId) return false;
  if (isSuperAdmin(user.role)) return false;
  const res = await query(
    `SELECT 1 FROM w4h.team_members WHERE team_id = $1::uuid AND user_id = $2 AND role = 'team_admin'`,
    [teamId, user.id]
  );
  return res.rowCount > 0;
}

/** Active collaborating teams (not the owning team) of this project that `user` administers. */
async function adminCollaboratingTeamIds(projectId, user) {
  if (!user) return [];
  const res = await query(
    `SELECT pt.team_id::text AS team_id
     FROM w4h.project_teams pt
     JOIN w4h.team_members tm ON tm.team_id = pt.team_id AND tm.user_id = $2 AND tm.role = 'team_admin'
     WHERE pt.project_id = $1::uuid AND pt.status = 'active'`,
    [projectId, user.id]
  );
  return res.rows.map((r) => r.team_id);
}

/** True when the user belongs to the project's owning team (collaborators are not). */
async function isOnOwningTeam(userId, projectId) {
  const teamId = await getProjectTeamId(projectId);
  return teamId ? isUserOnTeam(userId, teamId) : false;
}

async function listProjectTeams(projectId) {
  const res = await query(
    `SELECT pt.team_id::text AS team_id, t.name AS team_name, pt.status,
            pt.created_at, pt.accepted_at
     FROM w4h.project_teams pt
     JOIN w4h.teams t ON t.id = pt.team_id
     WHERE pt.project_id = $1::uuid
     ORDER BY t.name ASC`,
    [projectId]
  );
  return res.rows;
}

async function canAccessTeam(user, teamId) {
  if (!user || !teamId) return false;
  if (isSuperAdmin(user.role)) return false;
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

/** Project participant: project member, or team Admin on the owning team (not Super admin). */
async function canReadProject(projectId, user) {
  if (!user) return false;
  if (isSuperAdmin(user.role)) return false;
  const member = await getMembership(projectId, user.id);
  if (member) return true;
  const teamId = await getProjectTeamId(projectId);
  if (!teamId) return false;
  // Per-team membership role only (ADR-004) — a global users.role = team_admin must not leak across teams.
  const teamRole = await query(
    `SELECT role FROM w4h.team_members WHERE team_id = $1::uuid AND user_id = $2`,
    [teamId, user.id]
  );
  return teamRole.rowCount > 0 && teamRole.rows[0].role === 'team_admin';
}

/** Can see the project workspace (P-06): participant, or any member of the owning team. */
async function canViewProject(projectId, user) {
  if (await canReadProject(projectId, user)) return true;
  if (!user) return false;
  return isUserOnTeam(user.id, await getProjectTeamId(projectId));
}

async function canManageProject(projectId, user) {
  if (!user) return false;
  const teamId = await getProjectTeamId(projectId);
  if (teamId) {
    const teamRole = await query(
      `SELECT role FROM w4h.team_members WHERE team_id = $1::uuid AND user_id = $2`,
      [teamId, user.id]
    );
    if (teamRole.rowCount && teamRole.rows[0].role === 'team_admin') return true;
  }
  const member = await getMembership(projectId, user.id);
  return !!member && member.role === 'owner';
}

/** Doc-11 (product): anyone with access to the project can create/edit/delete its pages. */
async function canEditProjectPages(projectId, user) {
  return canReadProject(projectId, user);
}

function slugifyPageTitle(input) {
  return String(input || '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/(^-|-$)/g, '')
    .slice(0, 80);
}

async function uniquePageSlug(projectId, title, excludeSlug = null) {
  const base = slugifyPageTitle(title) || `page-${Date.now()}`;
  let n = 0;
  while (n < 50) {
    const trySlug = n === 0 ? base : `${base}-${n}`;
    const exists = await query(
      `SELECT 1 FROM w4h.project_pages
       WHERE project_id = $1::uuid AND slug = $2
         AND ($3::text IS NULL OR slug <> $3)`,
      [projectId, trySlug, excludeSlug]
    );
    if (!exists.rowCount) return trySlug;
    n += 1;
  }
  return `${base}-${crypto.randomBytes(3).toString('hex')}`;
}

function normalizePageDocType(value) {
  const t = String(value || '')
    .trim()
    .toLowerCase()
    .replace(/s$/, '');
  if (['guide', 'workflow', 'reference'].includes(t)) return t;
  return 'guide';
}

function mapPageRow(row) {
  if (!row) return null;
  return {
    id: row.id,
    project_id: row.project_id,
    slug: row.slug,
    title: row.title,
    doc_type: row.doc_type,
    sort_order: row.sort_order,
    md: row.md,
    html: row.html,
    toc: row.toc,
    created_by: row.created_by,
    updated_by: row.updated_by,
    created_at: row.created_at,
    updated_at: row.updated_at
  };
}

/**
 * Attach cheap dataset-table metadata and populate a missing cached summary.
 *
 * Uses PostgreSQL catalog estimates (not COUNT(*)) so opening a project never scans
 * large wearable tables. A zero-observation summary is retried on later loads so a
 * dataset registered before ingestion can update automatically after data arrives.
 */
async function hydrateDatasetSummaries(rows) {
  if (!Array.isArray(rows) || !rows.length) return rows || [];

  const ids = rows.map((r) => String(r.dataset_id || r.id || '')).filter(Boolean);
  if (!ids.length) return rows;

  let metadata = [];
  try {
    const result = await query(
      `SELECT t.dataset_id,
              COUNT(*)::int AS table_count,
              COUNT(c.oid)::int AS physical_table_count,
              COALESCE(
                ROUND(SUM(GREATEST(COALESCE(s.n_live_tup, c.reltuples, 0), 0)))::bigint,
                0
              ) AS estimated_observations
       FROM w4h.tables t
       LEFT JOIN pg_class c
         ON c.oid = to_regclass(
              CASE
                WHEN t.physical_schema IS NOT NULL AND t.physical_table IS NOT NULL
                  THEN t.physical_schema || '.' || t.physical_table
                WHEN position('.' IN t.name) > 0 THEN t.name
                ELSE 'public.' || t.name
              END
            )
       LEFT JOIN pg_stat_all_tables s ON s.relid = c.oid
       WHERE t.dataset_id = ANY($1::text[])
       GROUP BY t.dataset_id`,
      [ids]
    );
    metadata = result.rows || [];
  } catch {
    // Older installations may not have the registered-table catalog yet.
  }

  const byDataset = new Map(metadata.map((m) => [String(m.dataset_id), m]));

  return Promise.all(
    rows.map(async (row) => {
      const datasetId = String(row.dataset_id || row.id || '');
      const meta = byDataset.get(datasetId);
      const tableCount = Number(meta?.table_count || 0);
      const physicalTableCount = Number(meta?.physical_table_count || 0);
      const estimatedObservations = Number(meta?.estimated_observations || 0);
      const currentSummary =
        row.summary && typeof row.summary === 'object' && !Array.isArray(row.summary)
          ? row.summary
          : {};

      const statsMissing =
        !row.summary ||
        Object.keys(currentSummary).length === 0 ||
        currentSummary.observations == null ||
        Number(currentSummary.observations) === 0;

      const generatedSummary = {
        ...currentSummary,
        table_count: tableCount,
        physical_table_count: physicalTableCount,
        ...(row.device_type && !currentSummary.modality
          ? { modality: row.device_type }
          : {}),
        ...(row.subject_count != null && currentSummary.participants == null
          ? { participants: Number(row.subject_count) }
          : {}),
        ...(row.date_start && !currentSummary.date_start
          ? { date_start: row.date_start }
          : {}),
        ...(row.date_end && !currentSummary.date_end
          ? { date_end: row.date_end }
          : {}),
        ...(tableCount > 0 && statsMissing
          ? {
              observations: estimatedObservations,
              observations_approx: true,
              observations_source: 'postgres_catalog_estimate'
            }
          : {})
      };

      let updatedAt = row.summary_updated_at || null;
      if (tableCount > 0 && statsMissing) {
        const updated = await query(
          `UPDATE w4h.datasets
           SET summary = $2::jsonb, summary_updated_at = NOW()
           WHERE id = $1
             AND (
               summary IS NULL
               OR summary = '{}'::jsonb
               OR (summary->>'observations') IS NULL
               OR NULLIF(summary->>'observations', '')::numeric = 0
             )
           RETURNING summary_updated_at`,
          [datasetId, JSON.stringify(generatedSummary)]
        );
        updatedAt = updated.rows[0]?.summary_updated_at || updatedAt;
      }

      return {
        ...row,
        summary: generatedSummary,
        summary_updated_at: updatedAt,
        table_count: tableCount,
        physical_table_count: physicalTableCount
      };
    })
  );
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
  return adminRes.rows.map((r) => r.team_id);
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
            COALESCE(d.read_only, false) AS read_only, d.origin,
            t.name AS owner_team_name,
            'local'::text AS source_kind,
            NULL::text AS shared_by_team_id,
            NULL::text AS shared_by_team_name,
            'read_write'::text AS max_access_type
     FROM w4h.datasets d
     LEFT JOIN w4h.teams t ON t.id = d.team_id
     WHERE (d.team_id = $1::uuid OR d.team_id IS NULL)
       AND COALESCE(d.status, 'active') = 'active'
     ORDER BY d.name ASC NULLS LAST, d.id ASC`,
    [teamId]
  );

  const shared = await query(
    `SELECT d.id, d.name, d.description, d.team_id,
            COALESCE(d.read_only, false) AS read_only, d.origin,
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
       AND COALESCE(d.status, 'active') = 'active'
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

  // ADR-027: datasets already linked (e.g. contributed by a collaborating team) stay in the
  // set so a full-list save by the project Admin round-trips them.
  const linked = await query(
    `SELECT d.id, d.name, d.description, d.team_id,
            COALESCE(d.read_only, false) AS read_only, d.origin,
            t.name AS owner_team_name,
            'linked'::text AS source_kind,
            t.id::text AS shared_by_team_id,
            t.name AS shared_by_team_name,
            pd.access_type AS max_access_type
     FROM w4h.project_datasets pd
     JOIN w4h.datasets d ON d.id = pd.dataset_id
     LEFT JOIN w4h.teams t ON t.id = d.team_id
     WHERE pd.project_id = $1::uuid
       AND COALESCE(d.status, 'active') = 'active'`,
    [projectId]
  );
  for (const row of linked.rows) {
    if (byId.has(row.id)) continue;
    byId.set(row.id, {
      ...row,
      source_label: row.owner_team_name ? `Contributed by ${row.owner_team_name}` : 'Contributed',
    });
  }

  if (user) {
    const adminTeamIds = await getAdminTeamIdsForUser(user);
    let managed;
    if (adminTeamIds === null) {
      // Super admin: every other team's datasets
      managed = await query(
        `SELECT d.id, d.name, d.description, d.team_id,
                COALESCE(d.read_only, false) AS read_only, d.origin,
                t.name AS owner_team_name,
                'managed'::text AS source_kind,
                t.id::text AS shared_by_team_id,
                t.name AS shared_by_team_name,
                'read_write'::text AS max_access_type
         FROM w4h.datasets d
         LEFT JOIN w4h.teams t ON t.id = d.team_id
         WHERE d.team_id IS NOT NULL
           AND d.team_id IS DISTINCT FROM $1::uuid
           AND COALESCE(d.status, 'active') = 'active'
         ORDER BY d.name ASC NULLS LAST, d.id ASC`,
        [teamId]
      );
    } else if (adminTeamIds.length) {
      managed = await query(
        `SELECT d.id, d.name, d.description, d.team_id,
                COALESCE(d.read_only, false) AS read_only, d.origin,
                t.name AS owner_team_name,
                'managed'::text AS source_kind,
                t.id::text AS shared_by_team_id,
                t.name AS shared_by_team_name,
                'read_write'::text AS max_access_type
         FROM w4h.datasets d
         LEFT JOIN w4h.teams t ON t.id = d.team_id
         WHERE d.team_id = ANY($1::uuid[])
           AND d.team_id IS DISTINCT FROM $2::uuid
           AND COALESCE(d.status, 'active') = 'active'
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

/**
 * Linking a dataset owned by another team needs that owner's consent (ADR-027). Consent is
 * (a) a team-wide share to the project's team (ADR-002), (b) an existing project-scoped
 * share, or (c) the actor being an Admin of the dataset's owning team (or Super admin),
 * in which case a project-scoped share is recorded.
 */
async function ensureInboundShareForLink(req, datasetMeta, projectTeamId, projectId) {
  if (!datasetMeta?.team_id || String(datasetMeta.team_id) === String(projectTeamId)) {
    return;
  }
  if (datasetMeta.source_kind === 'shared' || datasetMeta.source_kind === 'linked') return;

  const existing = await query(
    `SELECT id FROM w4h.dataset_shares
     WHERE dataset_id = $1
       AND target_team_id = $2::uuid
       AND (target_project_id IS NULL OR target_project_id = $3::uuid)
     LIMIT 1`,
    [datasetMeta.id, projectTeamId, projectId]
  );
  if (existing.rowCount) return;

  if (!(await isTeamAdminOfTeam(req.user, datasetMeta.team_id))) {
    const err = new Error('Only an Admin of the dataset\'s owning team can link it into this project');
    err.status = 403;
    throw err;
  }

  await recordProjectShare(req, datasetMeta.id, projectTeamId, projectId, 'auto_share_on_project_link');
}

async function recordProjectShare(req, datasetId, projectTeamId, projectId, reason) {
  const inserted = await query(
    `INSERT INTO w4h.dataset_shares
       (dataset_id, target_team_id, target_project_id, access_type, shared_by)
     VALUES ($1, $2::uuid, $3::uuid, 'read', $4)
     ON CONFLICT (dataset_id, target_team_id, target_project_id) DO NOTHING
     RETURNING id::text AS id`,
    [datasetId, projectTeamId, projectId, req.user.id]
  );
  if (inserted.rowCount) {
    writeAuditLogSafe(req, {
      action: 'sharing.create',
      resourceType: 'dataset_share',
      resourceId: inserted.rows[0].id,
      details: {
        dataset_id: datasetId,
        target_team_id: projectTeamId,
        target_project_id: projectId,
        reason,
      },
    });
  }
}

async function listEligibleTeamMembers(teamId) {
  if (!teamId) return [];
  const res = await query(
    `SELECT u.id AS user_id, u.name, u.email, u.role AS user_role,
            tm.role AS team_role
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
    if (isSuperAdmin(req.user.role)) {
      return res.json([]);
    }
    const userId = req.user.id;
    let sql = `
      SELECT p.id, p.slug, p.name, p.description, p.status, p.visibility,
             p.irb_protocol_id, p.irb_expiry_date, p.team_id,
             t.name AS team_name, t.slug AS team_slug,
             p.created_at, p.updated_at,
             (
               SELECT pm.role
               FROM w4h.project_members pm
               WHERE pm.project_id = p.id AND pm.user_id = $1
               LIMIT 1
             ) AS my_role
      FROM w4h.projects p
      LEFT JOIN w4h.teams t ON t.id = p.team_id
    `;
    const params = [userId];
    const conditions = [];

    if (!isSuperAdmin(req.user.role)) {
      const teamIds = await getTeamIdsForUser(userId);
      if (teamIds.length) {
        params.push(teamIds);
        conditions.push(`
          (
            EXISTS (
              SELECT 1 FROM w4h.project_members pm
              WHERE pm.project_id = p.id AND pm.user_id = $1
            )
            OR p.team_id = ANY($${params.length}::uuid[])
          )
        `);
      } else {
        conditions.push(`
          EXISTS (
            SELECT 1 FROM w4h.project_members pm
            WHERE pm.project_id = p.id AND pm.user_id = $1
          )
        `);
      }
    }

    // ADR-010: archived projects are hidden by default; ?status=archived or ?status=all opts in.
    const statusFilter = ['active', 'archived', 'all'].includes(req.query?.status)
      ? req.query.status
      : 'active';
    if (statusFilter !== 'all') {
      params.push(statusFilter);
      conditions.push(`p.status = $${params.length}`);
    }

    if (conditions.length) {
      sql += ` WHERE ${conditions.join(' AND ')}`;
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
    if (!(await canViewProject(projectId, req.user))) {
      return res.status(403).json({ error: 'Unauthorized' });
    }
    const isParticipant = await canReadProject(projectId, req.user);

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
    const canManage = !req.authViaProjectDbToken && (await canManageProject(projectId, req.user));
    const canEditPages =
      !req.authViaProjectDbToken && (await canEditProjectPages(projectId, req.user));

    const collaboratingTeams = await listProjectTeams(projectId);
    const [membersRes, datasetsRes, pagesRes, orgsRes, derivedOrgsRes, aclRes, linkableDatasets, eligibleMembers] =
      await Promise.all([
        Promise.resolve({ rows: await listMembers(projectId) }),
        query(
          `SELECT pd.dataset_id, pd.access_type, pd.approved_at,
                  d.name, d.description, d.${pc} AS dataset_project_id, d.team_id AS dataset_team_id,
                  d.device_type, d.sampling_rate, d.subject_count, d.date_start, d.date_end,
                  d.irb_reference, d.description,
                  d.summary, d.summary_updated_at, d.read_only, d.origin
           FROM w4h.project_datasets pd
           JOIN w4h.datasets d ON d.id = pd.dataset_id
           WHERE pd.project_id = $1::uuid
             AND COALESCE(d.status, 'active') = 'active'
           ORDER BY d.name ASC NULLS LAST, d.id ASC`,
          [projectId]
        ),
        query(
          `SELECT id, project_id, slug, title, doc_type, sort_order, updated_by, created_at, updated_at
           FROM w4h.project_pages
           WHERE project_id = $1::uuid
           ORDER BY sort_order ASC, title ASC`,
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
        canManage ? fetchLinkableDatasets(projectId, project.team_id, req.user) : Promise.resolve([]),
        listEligibleTeamMembers(project.team_id)
      ]);

    // A project with no roster still has its Admins on the owning team.
    if (!membersRes.rows.length && project.team_id) {
      for (const m of eligibleMembers) {
        if (m.team_role !== 'team_admin') continue;
        await query(
          `INSERT INTO w4h.project_members (project_id, user_id, role, invited_by)
           VALUES ($1::uuid, $2, 'owner', $2)
           ON CONFLICT (project_id, user_id) DO NOTHING`,
          [projectId, m.user_id]
        );
      }
      membersRes.rows = await listMembers(projectId);
    }

    const canIssueDbToken = !req.authViaProjectDbToken && isParticipant;
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

    const hydratedDatasetRows = await attachMetadataSummaries(
      await hydrateDatasetSummaries(datasetsRes.rows)
    );
    const hydratedLinkable = canManage
      ? await attachMetadataSummaries(
          await hydrateDatasetSummaries(
            linkableDatasets.map((d) => ({ ...d, dataset_id: d.id }))
          )
        )
      : [];
    const linkableById = new Map(hydratedLinkable.map((d) => [String(d.id), d]));
    const datasets = hydratedDatasetRows.map((row) => {
      const meta = linkableById.get(String(row.dataset_id));
      return {
        ...row,
        source_kind: row.read_only ? 'sample' : (meta?.source_kind || 'local'),
        source_label: row.read_only
          ? `${row.origin || 'W4H Samples'} · read-only`
          : (meta?.source_label || 'Local'),
        shared_by_team_name: meta?.shared_by_team_name || null
      };
    });

    res.json({
      project,
      members: membersRes.rows,
      collaborating_teams: collaboratingTeams,
      datasets,
      linkable_datasets: hydratedLinkable,
      eligible_members: canManage ? eligibleMembers : [],
      // Project pages are project-private: team-only viewers see the project, not its pages.
      pages: isParticipant ? pagesRes.rows : [],
      docs: [],
      organizations,
      acl_rules: isParticipant ? aclRes.rows : [],
      capabilities: {
        can_manage: canManage,
        can_edit_pages: canEditPages,
        // Project member (or owning-team Admin) vs. a team member who can only see the overview.
        is_participant: isParticipant,
        can_issue_db_token: canIssueDbToken,
        auth_via_project_token: Boolean(req.authViaProjectDbToken)
      }
    });
  } catch (error) {
    res.status(500).json({ error: error.message || 'Failed to load project overview' });
  }
}

async function createProject(req, res) {
  if (!(await canCreateProject(req.user))) {
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

    if (!teamId) {
      const adminTeams = await getAdminTeamIdsForUser(req.user);
      if (adminTeams.length === 1) teamId = adminTeams[0];
    }

    if (!teamId) {
      return res.status(400).json({ error: 'team_id is required' });
    }

    const teamCheck = await query(`SELECT id FROM w4h.teams WHERE id = $1::uuid`, [teamId]);
    if (!teamCheck.rowCount) {
      return res.status(400).json({ error: 'Unknown team_id' });
    }

    // Creating a project needs Team admin on *that* team (R-08 dual-hat), not just membership.
    if (!(await isTeamAdminOfTeam(req.user, teamId))) {
      return res.status(403).json({ error: 'Unauthorized for this team' });
    }

    const isDemonstration = Boolean(req.body?.is_demonstration);
    if (isDemonstration) {
      return res.status(403).json({
        error: 'Demonstration projects are created by platform seed scripts, not the API.',
      });
    }

    const slug = await ensureUniqueSlug(req.body?.slug || name);
    const inserted = await query(
      `INSERT INTO w4h.projects
         (slug, name, description, status, visibility, irb_protocol_id, irb_expiry_date, created_by, team_id, is_demonstration)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::uuid, $10)
       RETURNING *`,
      [slug, name, description, status, visibility, irbProtocolId || null, irbExpiryDate, req.user.id, teamId, isDemonstration]
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

    const seedFrom = String(req.body?.seed_from_project_id || '').trim();
    if (seedFrom) {
      const seed = await query(
        `SELECT id, team_id FROM w4h.projects WHERE id::text = $1 OR slug = $1`,
        [seedFrom]
      );
      if (seed.rowCount && (await canAccessTeam(req.user, seed.rows[0].team_id))) {
        const seedMembers = await query(
          `SELECT user_id, role FROM w4h.project_members WHERE project_id = $1::uuid`,
          [seed.rows[0].id]
        );
        for (const m of seedMembers.rows) {
          if (!(await isUserOnTeam(m.user_id, teamId))) continue;
          await query(
            `INSERT INTO w4h.project_members (project_id, user_id, role, invited_by)
             VALUES ($1::uuid, $2, $3, $4)
             ON CONFLICT (project_id, user_id) DO NOTHING`,
            [project.id, m.user_id, m.role, req.user.id]
          );
        }
      }
    }

    if (isDemonstration) {
      await linkSampleDatasetsToProject(project.id, teamId, req.user.id);
    }

    res.status(201).json(project);
  } catch (error) {
    res.status(500).json({ error: error.message || 'Failed to create project' });
  }
}

async function linkSampleDatasetsToProject(projectId, teamId, userId) {
  let samples = { rows: [] };
  try {
    samples = await query(
      `SELECT id, team_id FROM w4h.datasets WHERE read_only = true ORDER BY name ASC NULLS LAST, id ASC`
    );
  } catch {
    return;
  }
  for (const row of samples.rows) {
    if (row.team_id && teamId && String(row.team_id) !== String(teamId)) {
      await query(
        `INSERT INTO w4h.dataset_shares
           (dataset_id, target_team_id, target_project_id, access_type, shared_by)
         VALUES ($1, $2::uuid, $3::uuid, 'read', $4)
         ON CONFLICT DO NOTHING`,
        [row.id, teamId, projectId, userId ?? null]
      ).catch(() => {});
    }
    await query(
      `INSERT INTO w4h.project_datasets (project_id, dataset_id, access_type)
       VALUES ($1::uuid, $2, 'read')
       ON CONFLICT (project_id, dataset_id) DO NOTHING`,
      [projectId, row.id]
    ).catch(() => {});
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
    auditProjectAction(req, 'project.update', projectId);
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
    `SELECT pm.user_id, pm.role, pm.joined_at, u.name, u.email, u.role AS user_role,
            COALESCE((
              SELECT json_agg(json_build_object('id', t.id::text, 'name', t.name) ORDER BY t.name)
              FROM w4h.team_members tm
              JOIN w4h.teams t ON t.id = tm.team_id
              WHERE tm.user_id = pm.user_id
                AND (t.id = p.team_id
                  OR t.id IN (SELECT pt.team_id FROM w4h.project_teams pt
                              WHERE pt.project_id = pm.project_id AND pt.status = 'active'))
            ), '[]'::json) AS home_teams,
            NOT EXISTS (
              SELECT 1 FROM w4h.team_members tm2
              WHERE tm2.user_id = pm.user_id AND tm2.team_id = p.team_id
            ) AS is_collaborator
     FROM w4h.project_members pm
     JOIN w4h.projects p ON p.id = pm.project_id
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
  // Project Admin/owning-team Admin, or (ADR-027) a Team admin of an active collaborating
  // team adding people from their own team.
  const isManager = await canManageProject(projectId, req.user);
  const collabAdminTeamIds = isManager ? [] : await adminCollaboratingTeamIds(projectId, req.user);
  if (!isManager && !collabAdminTeamIds.length) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  try {
    const role = String(req.body?.role || 'editor');
    if (!PROJECT_MEMBER_ROLES.has(role)) {
      return res.status(400).json({ error: 'Invalid role' });
    }

    const teamId = await getProjectTeamId(projectId);
    if (!teamId) {
      return res.status(400).json({ error: 'Project has no owning team; set team_id before adding members' });
    }

    const email = String(req.body?.email || '').trim().toLowerCase();
    const name = String(req.body?.name || '').trim();
    let userId = await resolveMemberUserId(req.body || {});
    let invited = false;
    let inviteUrl = null;
    let mailWarning = null;

    if (!Number.isInteger(userId) && !email) {
      return res.status(400).json({ error: 'Email is required' });
    }

    if (!Number.isInteger(userId) && email) {
      const found = await query(
        `SELECT id, role, status FROM w4h.users WHERE lower(trim(email)) = $1`,
        [email]
      );
      if (found.rowCount) {
        userId = found.rows[0].id;
      }
    }

    if (Number.isInteger(userId) && (await isPlatformAdminUserId(userId))) {
      return res.status(400).json({
        error: 'Platform admin accounts are not project members.'
      });
    }

    if (Number.isInteger(userId)) {
      const already = await query(
        `SELECT 1 FROM w4h.project_members WHERE project_id = $1::uuid AND user_id = $2`,
        [projectId, userId]
      );
      if (already.rowCount) {
        return res.status(409).json({ error: 'That person is already on this project' });
      }
    }

    if (!isManager && !Number.isInteger(userId)) {
      return res.status(403).json({
        error: 'Collaborating-team Admins can only add people who already have an account on their team'
      });
    }

    if (Number.isInteger(userId) && !invited) {
      const onProjectTeam = await isUserOnProjectTeams(userId, projectId);
      if (!onProjectTeam) {
        return res.status(400).json({
          error: 'That person is not on the owning team or an active collaborating team. Invite their team to the project first.'
        });
      }
      if (!isManager) {
        const theirTeams = await getTeamIdsForUser(userId);
        if (!theirTeams.some((t) => collabAdminTeamIds.includes(t))) {
          return res.status(403).json({ error: 'You can only add people from the team you administer' });
        }
      }
      if (role === 'owner' && !(await isOnOwningTeam(userId, projectId))) {
        return res.status(400).json({
          error: 'Only owning-team members can be project Admins; collaborators join as Users'
        });
      }
    }

    if (!Number.isInteger(userId)) {
      const displayName = name || email;
      const inserted = await query(
        `INSERT INTO w4h.users (name, email, password, role, status)
         VALUES ($1, $2, NULL, $3, 'invited')
         RETURNING id`,
        [displayName, email, USER_ROLES.PROJECT_USER]
      );
      userId = inserted.rows[0].id;
      invited = true;
      const delivered = await deliverInvitation({
        email,
        name: displayName,
        roleLabel: role === 'owner' ? 'Admin' : 'User',
        userId,
        invitedBy: req.user.id,
      });
      inviteUrl = delivered.inviteUrl;
      mailWarning = delivered.mailWarning;
    }

    if (invited && !(await isUserOnTeam(userId, teamId))) {
      await query(
        `INSERT INTO w4h.team_members (team_id, user_id, role)
         VALUES ($1::uuid, $2, 'member')
         ON CONFLICT (team_id, user_id) DO NOTHING`,
        [teamId, userId]
      );
    }

    await query(
      `INSERT INTO w4h.project_members (project_id, user_id, role, invited_by)
       VALUES ($1::uuid, $2, $3, $4)
       ON CONFLICT (project_id, user_id) DO UPDATE SET role = EXCLUDED.role`,
      [projectId, userId, role, req.user.id]
    );

    auditProjectAction(req, invited ? 'project.member.invite' : 'project.member.add', projectId, {
      resourceType: 'project_member',
      resourceId: userId,
      details: { user_id: userId, role, email: email || undefined, invited },
    });

    res.status(invited ? 201 : 200).json({
      members: await listMembers(projectId),
      invited,
      ...(inviteUrl ? { inviteUrl } : {}),
      ...(mailWarning ? { mailWarning } : {}),
    });
  } catch (error) {
    if (error.code === '23505' || /users_email_key|duplicate key/i.test(error.message || '')) {
      return res.status(409).json({ error: 'That person is already on this project' });
    }
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
      return res.status(400).json({ error: 'Project must have at least one Admin' });
    }
    if (role === 'owner' && !(await isOnOwningTeam(userId, projectId))) {
      return res.status(400).json({
        error: 'Only owning-team members can be project Admins; collaborators stay Users'
      });
    }

    await query(
      `UPDATE w4h.project_members
       SET role = $1
       WHERE project_id = $2::uuid AND user_id = $3`,
      [role, projectId, userId]
    );
    auditProjectAction(req, 'project.member.role', projectId, {
      resourceType: 'project_member',
      resourceId: userId,
      details: { user_id: userId, role },
    });
    res.json(await listMembers(projectId));
  } catch (error) {
    res.status(500).json({ error: error.message || 'Failed to update member role' });
  }
}

async function removeProjectMember(req, res) {
  const projectId = await resolveProjectId(req.params.id);
  if (!projectId) return res.status(404).json({ error: 'Project not found' });
  const userId = Number(req.params.userId);
  if (!Number.isInteger(userId)) {
    return res.status(400).json({ error: 'Invalid user id' });
  }
  if (!(await canManageProject(projectId, req.user))) {
    // ADR-027: a collaborating team's Admin may remove their own team's people.
    const collabTeams = await adminCollaboratingTeamIds(projectId, req.user);
    const theirTeams = collabTeams.length ? await getTeamIdsForUser(userId) : [];
    if (!theirTeams.some((t) => collabTeams.includes(t))) {
      return res.status(403).json({ error: 'Unauthorized' });
    }
  }

  try {
    const current = await query(
      `SELECT role FROM w4h.project_members WHERE project_id = $1::uuid AND user_id = $2`,
      [projectId, userId]
    );
    if (!current.rowCount) {
      return res.status(404).json({ error: 'Member not found' });
    }
    if (Number(userId) === Number(req.user.id)) {
      return res.status(400).json({
        error: 'You cannot remove yourself from the project. Ask another Admin, or a Super admin, to remove you.'
      });
    }
    if (current.rows[0].role === 'owner' && (await ownerCount(projectId)) <= 1) {
      return res.status(400).json({ error: 'Project must have at least one Admin' });
    }

    await query(
      `DELETE FROM w4h.project_members
       WHERE project_id = $1::uuid AND user_id = $2`,
      [projectId, userId]
    );
    auditProjectAction(req, 'project.member.remove', projectId, {
      resourceType: 'project_member',
      resourceId: userId,
      details: { user_id: userId },
    });
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
      if (!(await isUserOnProjectTeams(userId, projectId))) {
        return res.status(400).json({
          error: `User must belong to the project's owning team or an active collaborating team (user_id ${userId})`
        });
      }
      if (role === 'owner' && !(await isUserOnTeam(userId, teamId))) {
        return res.status(400).json({
          error: `Only owning-team members can be project Admins (user_id ${userId})`
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

    // Remove links that are no longer listed — but a collaborator's dataset is only the
    // owner's to unlink (ADR-027), so keep those unless the actor administers that team.
    const submittedIds = new Set(datasets.map((d) => String(d?.dataset_id || '').trim()).filter(Boolean));
    const existingLinks = await query(
      `SELECT pd.dataset_id, d.team_id::text AS team_id
       FROM w4h.project_datasets pd
       JOIN w4h.datasets d ON d.id = pd.dataset_id
       WHERE pd.project_id = $1::uuid`,
      [projectId]
    );
    for (const link of existingLinks.rows) {
      if (submittedIds.has(String(link.dataset_id))) continue;
      const foreign = link.team_id && String(link.team_id) !== String(teamId);
      if (foreign && !(await isTeamAdminOfTeam(req.user, link.team_id))) continue;
      await removeProjectDatasetLink(projectId, link.dataset_id);
    }

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
        await ensureInboundShareForLink(req, meta, teamId, projectId);
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
    auditProjectAction(req, 'project.datasets.update', projectId, {
      details: { dataset_ids: datasets.map((d) => d?.dataset_id).filter(Boolean) },
    });
    res.json(fresh.rows);
  } catch (error) {
    res.status(500).json({ error: error.message || 'Failed to update project datasets' });
  }
}

/** Drop a dataset link and withdraw any project-scoped share that granted it (ADR-027). */
async function removeProjectDatasetLink(projectId, datasetId) {
  await query(
    `DELETE FROM w4h.project_datasets WHERE project_id = $1::uuid AND dataset_id = $2`,
    [projectId, datasetId]
  );
  await query(
    `DELETE FROM w4h.dataset_shares WHERE dataset_id = $1 AND target_project_id = $2::uuid`,
    [datasetId, projectId]
  );
  // Reports embed the data they plot (ADR-029): revoking a dataset revokes reports built from it.
  await query(
    `DELETE FROM w4h.project_reports WHERE project_id = $1::uuid AND $2 = ANY(dataset_ids)`,
    [projectId, datasetId]
  );
}

/**
 * POST /projects/:id/datasets/:datasetId/link — the dataset owner's Admin contributes a
 * dataset to a project owned by their team or by a team that invited theirs (ADR-027 §2).
 */
async function linkProjectDataset(req, res) {
  const projectId = await resolveProjectId(req.params.id);
  if (!projectId) return res.status(404).json({ error: 'Project not found' });
  try {
    const datasetId = String(req.params.datasetId || '').trim();
    const ds = await query(
      `SELECT id, team_id::text AS team_id, COALESCE(status, 'active') AS status
       FROM w4h.datasets WHERE id = $1`,
      [datasetId]
    );
    if (!ds.rowCount) return res.status(404).json({ error: 'Dataset not found' });
    const dataset = ds.rows[0];
    if (!dataset.team_id) {
      return res.status(400).json({ error: 'Dataset has no owning team' });
    }
    if (dataset.status !== 'active') {
      return res.status(400).json({ error: 'Archived datasets cannot be linked' });
    }
    if (!(await isTeamAdminOfTeam(req.user, dataset.team_id))) {
      return res.status(403).json({ error: 'Only an Admin of the dataset\'s owning team can link it' });
    }
    const projectTeamIds = await getProjectTeamIds(projectId);
    if (!projectTeamIds.includes(dataset.team_id)) {
      return res.status(403).json({
        error: 'The dataset\'s team is not the owning team or an active collaborating team of this project'
      });
    }
    const owningTeamId = await getProjectTeamId(projectId);
    if (String(dataset.team_id) !== String(owningTeamId)) {
      await recordProjectShare(req, datasetId, owningTeamId, projectId, 'collaborator_contribution');
    }
    await query(
      `INSERT INTO w4h.project_datasets (project_id, dataset_id, access_type, approved_by)
       VALUES ($1::uuid, $2, 'read', $3)
       ON CONFLICT (project_id, dataset_id)
       DO UPDATE SET approved_by = EXCLUDED.approved_by, approved_at = NOW()`,
      [projectId, datasetId, req.user.id]
    );
    auditProjectAction(req, 'project.dataset.link', projectId, {
      resourceType: 'dataset',
      resourceId: datasetId,
      details: { dataset_id: datasetId, dataset_team_id: dataset.team_id },
    });
    res.json({ ok: true, dataset_id: datasetId });
  } catch (error) {
    res.status(500).json({ error: error.message || 'Failed to link dataset' });
  }
}

/** DELETE /projects/:id/datasets/:datasetId — owner's Admin, project Admin, or Super admin. */
async function unlinkProjectDataset(req, res) {
  const projectId = await resolveProjectId(req.params.id);
  if (!projectId) return res.status(404).json({ error: 'Project not found' });
  try {
    const datasetId = String(req.params.datasetId || '').trim();
    const link = await query(
      `SELECT d.team_id::text AS team_id
       FROM w4h.project_datasets pd JOIN w4h.datasets d ON d.id = pd.dataset_id
       WHERE pd.project_id = $1::uuid AND pd.dataset_id = $2`,
      [projectId, datasetId]
    );
    if (!link.rowCount) return res.status(404).json({ error: 'Dataset is not linked to this project' });
    const allowed =
      (await isTeamAdminOfTeam(req.user, link.rows[0].team_id)) ||
      (await canManageProject(projectId, req.user));
    if (!allowed) return res.status(403).json({ error: 'Unauthorized' });
    await removeProjectDatasetLink(projectId, datasetId);
    auditProjectAction(req, 'project.dataset.unlink', projectId, {
      resourceType: 'dataset',
      resourceId: datasetId,
      details: { dataset_id: datasetId },
    });
    res.json({ ok: true });
  } catch (error) {
    res.status(500).json({ error: error.message || 'Failed to unlink dataset' });
  }
}

/** GET /projects/:id/invitable-teams — active teams not yet on the project (id + name only). */
async function listInvitableTeams(req, res) {
  const projectId = await resolveProjectId(req.params.id);
  if (!projectId) return res.status(404).json({ error: 'Project not found' });
  if (!(await canManageProject(projectId, req.user))) {
    return res.status(403).json({ error: 'Unauthorized' });
  }
  try {
    const rows = await query(
      `SELECT t.id::text AS id, t.name
       FROM w4h.teams t
       WHERE COALESCE(t.status, 'active') = 'active'
         AND t.id <> (SELECT team_id FROM w4h.projects WHERE id = $1::uuid)
         AND NOT EXISTS (
           SELECT 1 FROM w4h.project_teams pt WHERE pt.project_id = $1::uuid AND pt.team_id = t.id
         )
       ORDER BY t.name ASC`,
      [projectId]
    );
    res.json(rows.rows);
  } catch (error) {
    res.status(500).json({ error: error.message || 'Failed to list teams' });
  }
}

/** POST /projects/:id/teams { team_id } — invite a collaborating team (ADR-027 §1–2). */
async function inviteProjectTeam(req, res) {
  const projectId = await resolveProjectId(req.params.id);
  if (!projectId) return res.status(404).json({ error: 'Project not found' });
  if (!(await canManageProject(projectId, req.user))) {
    return res.status(403).json({ error: 'Unauthorized' });
  }
  try {
    const teamId = req.body?.team_id == null ? '' : String(req.body.team_id).trim();
    if (!teamId) return res.status(400).json({ error: 'team_id is required' });
    const team = await query(
      `SELECT id::text AS id, name, COALESCE(status, 'active') AS status FROM w4h.teams WHERE id = $1::uuid`,
      [teamId]
    );
    if (!team.rowCount) return res.status(400).json({ error: 'Unknown team_id' });
    if (team.rows[0].status !== 'active') {
      return res.status(400).json({ error: 'That team is deactivated' });
    }
    if (String(await getProjectTeamId(projectId)) === teamId) {
      return res.status(400).json({ error: 'That is already the owning team' });
    }
    // The invited team's own Admin (or a Super admin) accepts; an Admin of both teams accepts at once.
    const selfAccept = await isTeamAdminOfTeam(req.user, teamId);
    const inserted = await query(
      `INSERT INTO w4h.project_teams (project_id, team_id, status, invited_by, accepted_by, accepted_at)
       VALUES ($1::uuid, $2::uuid, $3, $4, $5, $6)
       ON CONFLICT (project_id, team_id) DO NOTHING
       RETURNING status`,
      [
        projectId,
        teamId,
        selfAccept ? 'active' : 'pending',
        req.user.id,
        selfAccept ? req.user.id : null,
        selfAccept ? new Date() : null,
      ]
    );
    if (!inserted.rowCount) {
      return res.status(409).json({ error: 'That team is already invited to this project' });
    }
    auditProjectAction(req, 'project.team.invite', projectId, {
      resourceType: 'team',
      resourceId: teamId,
      details: { team_id: teamId, status: inserted.rows[0].status },
    });
    res.status(201).json({ teams: await listProjectTeams(projectId) });
  } catch (error) {
    res.status(500).json({ error: error.message || 'Failed to invite team' });
  }
}

/** POST /projects/:id/teams/:teamId/accept — the invited team's Admin (or Super admin). */
async function acceptProjectTeam(req, res) {
  const projectId = await resolveProjectId(req.params.id);
  if (!projectId) return res.status(404).json({ error: 'Project not found' });
  const teamId = String(req.params.teamId || '').trim();
  if (!(await isTeamAdminOfTeam(req.user, teamId))) {
    return res.status(403).json({ error: 'Only an Admin of the invited team can accept' });
  }
  try {
    const updated = await query(
      `UPDATE w4h.project_teams
       SET status = 'active', accepted_by = $3, accepted_at = NOW()
       WHERE project_id = $1::uuid AND team_id = $2::uuid AND status = 'pending'
       RETURNING team_id`,
      [projectId, teamId, req.user.id]
    );
    if (!updated.rowCount) {
      return res.status(404).json({ error: 'No pending invitation for that team' });
    }
    auditProjectAction(req, 'project.team.accept', projectId, {
      resourceType: 'team',
      resourceId: teamId,
      details: { team_id: teamId },
    });
    res.json({ teams: await listProjectTeams(projectId) });
  } catch (error) {
    res.status(500).json({ error: error.message || 'Failed to accept invitation' });
  }
}

/**
 * DELETE /projects/:id/teams/:teamId — decline/leave (that team's Admin) or remove
 * (owning side). Its people and its datasets leave the project immediately.
 */
async function removeProjectTeam(req, res) {
  const projectId = await resolveProjectId(req.params.id);
  if (!projectId) return res.status(404).json({ error: 'Project not found' });
  const teamId = String(req.params.teamId || '').trim();
  const allowed =
    (await canManageProject(projectId, req.user)) || (await isTeamAdminOfTeam(req.user, teamId));
  if (!allowed) return res.status(403).json({ error: 'Unauthorized' });
  try {
    const removed = await query(
      `DELETE FROM w4h.project_teams WHERE project_id = $1::uuid AND team_id = $2::uuid RETURNING team_id`,
      [projectId, teamId]
    );
    if (!removed.rowCount) return res.status(404).json({ error: 'That team is not a collaborator' });

    // Datasets the leaving team contributed.
    const contributed = await query(
      `SELECT pd.dataset_id FROM w4h.project_datasets pd
       JOIN w4h.datasets d ON d.id = pd.dataset_id
       WHERE pd.project_id = $1::uuid AND d.team_id = $2::uuid`,
      [projectId, teamId]
    );
    for (const row of contributed.rows) {
      await removeProjectDatasetLink(projectId, row.dataset_id);
    }
    // People who no longer belong to any team of the project.
    const remaining = await getProjectTeamIds(projectId);
    await query(
      `DELETE FROM w4h.project_members pm
       WHERE pm.project_id = $1::uuid
         AND NOT EXISTS (
           SELECT 1 FROM w4h.team_members tm
           WHERE tm.user_id = pm.user_id AND tm.team_id = ANY($2::uuid[])
         )`,
      [projectId, remaining]
    );
    auditProjectAction(req, 'project.team.remove', projectId, {
      resourceType: 'team',
      resourceId: teamId,
      details: { team_id: teamId, datasets_unlinked: contributed.rowCount },
    });
    res.json({ teams: await listProjectTeams(projectId), members: await listMembers(projectId) });
  } catch (error) {
    res.status(500).json({ error: error.message || 'Failed to remove team' });
  }
}

/** GET /projects/collaboration-invites — pending and active collaborations for teams the caller administers. */
async function listCollaborationInvites(req, res) {
  try {
    const adminTeamIds = await getAdminTeamIdsForUser(req.user);
    const where = adminTeamIds === null ? '' : 'AND pt.team_id = ANY($1::uuid[])';
    const params = adminTeamIds === null ? [] : [adminTeamIds];
    if (adminTeamIds !== null && !adminTeamIds.length) return res.json([]);
    const rows = await query(
      `SELECT p.id::text AS project_id, p.name AS project_name,
              ot.name AS owning_team_name,
              pt.team_id::text AS team_id, it.name AS team_name, pt.status, pt.created_at
       FROM w4h.project_teams pt
       JOIN w4h.projects p ON p.id = pt.project_id
       LEFT JOIN w4h.teams ot ON ot.id = p.team_id
       JOIN w4h.teams it ON it.id = pt.team_id
       WHERE TRUE ${where}
       ORDER BY pt.status ASC, pt.created_at DESC`,
      params
    );
    res.json(rows.rows);
  } catch (error) {
    res.status(500).json({ error: error.message || 'Failed to list invitations' });
  }
}

// ---------------------------------------------------------------------------
// Published notebook reports (ADR-029 §5)
// ---------------------------------------------------------------------------

const REPORT_META_COLS = `r.id, r.title, r.description, r.size_bytes, r.sha256, r.dataset_ids,
  r.source, r.snapshot_at, r.created_at, r.created_by, u.name AS created_by_name`;

async function listProjectReports(req, res) {
  const projectId = await resolveProjectId(req.params.id);
  if (!projectId) return res.status(404).json({ error: 'Project not found' });
  if (!(await canReadProject(projectId, req.user))) {
    return res.status(403).json({ error: 'Unauthorized' });
  }
  try {
    const rows = await query(
      `SELECT ${REPORT_META_COLS}
       FROM w4h.project_reports r LEFT JOIN w4h.users u ON u.id = r.created_by
       WHERE r.project_id = $1::uuid ORDER BY r.created_at DESC`,
      [projectId]
    );
    res.json(rows.rows);
  } catch (error) {
    res.status(500).json({ error: error.message || 'Failed to list reports' });
  }
}

async function publishProjectReport(req, res) {
  const projectId = await resolveProjectId(req.params.id);
  if (!projectId) return res.status(404).json({ error: 'Project not found' });
  if (!(await canManageProject(projectId, req.user))) {
    return res.status(403).json({ error: 'Only a project Admin can publish a report' });
  }
  try {
    const title = String(req.body?.title || '').trim().slice(0, 200);
    const description = req.body?.description == null ? null : String(req.body.description).trim().slice(0, 1000);
    const html = String(req.body?.html ?? '').replace(/^﻿/, '');
    if (!title) return res.status(400).json({ error: 'title is required' });
    if (!html.trim() || !looksLikeHtml(html)) {
      return res.status(400).json({ error: 'html must be the report page (an HTML document)' });
    }
    const bytes = Buffer.byteLength(html, 'utf8');
    if (bytes > REPORT_MAX_BYTES) {
      return res.status(413).json({
        error: `Report is ${(bytes / 1048576).toFixed(1)} MB; the limit is ${(REPORT_MAX_BYTES / 1048576).toFixed(0)} MB`,
      });
    }
    // The HTML embeds whatever the notebook plotted — make the publisher say they know.
    if (req.body?.acknowledge_embedded_data !== true) {
      return res.status(400).json({
        error: 'Confirm that you understand the report embeds the data it plots (acknowledge_embedded_data).',
      });
    }

    // Provenance, and the cross-org rule (ADR-027): only datasets the publisher's own team owns,
    // because another team could not revoke data embedded in a file.
    const datasetIds = [...new Set((Array.isArray(req.body?.dataset_ids) ? req.body.dataset_ids : [])
      .map((d) => String(d || '').trim()).filter(Boolean))];
    if (!datasetIds.length) {
      return res.status(400).json({ error: 'dataset_ids is required: list the datasets the report was built from' });
    }
    const found = await query(
      `SELECT d.id, d.name, d.team_id::text AS team_id, t.name AS team_name,
              EXISTS (SELECT 1 FROM w4h.project_datasets pd
                      WHERE pd.project_id = $2::uuid AND pd.dataset_id = d.id) AS linked,
              EXISTS (SELECT 1 FROM w4h.team_members tm
                      WHERE tm.team_id = d.team_id AND tm.user_id = $3) AS mine
       FROM w4h.datasets d LEFT JOIN w4h.teams t ON t.id = d.team_id
       WHERE d.id = ANY($1::text[])`,
      [datasetIds, projectId, req.user.id]
    );
    const byId = new Map(found.rows.map((r) => [r.id, r]));
    for (const id of datasetIds) {
      const row = byId.get(id);
      if (!row) return res.status(400).json({ error: `Unknown dataset: ${id}` });
      if (!row.linked) return res.status(400).json({ error: `Dataset "${row.name || id}" is not linked to this project` });
      if (!row.mine) {
        return res.status(403).json({
          error: `Dataset "${row.name || id}" belongs to ${row.team_name || 'another team'}. A report may only use datasets your own team owns.`,
        });
      }
    }
    const source = {
      contribution: req.body?.contribution && typeof req.body.contribution === 'object'
        ? {
            slug: String(req.body.contribution.slug || '').slice(0, 120) || null,
            version: String(req.body.contribution.version || '').slice(0, 60) || null,
          }
        : null,
      datasets: datasetIds.map((id) => ({
        id, name: byId.get(id).name, team_id: byId.get(id).team_id, team_name: byId.get(id).team_name,
      })),
    };
    const snapshotAt = req.body?.snapshot_at && !Number.isNaN(Date.parse(req.body.snapshot_at))
      ? new Date(req.body.snapshot_at) : new Date();

    const inserted = await query(
      `INSERT INTO w4h.project_reports
         (project_id, title, description, html, size_bytes, sha256, dataset_ids, source, snapshot_at, created_by)
       VALUES ($1::uuid, $2, $3, $4, $5, $6, $7::text[], $8::jsonb, $9, $10)
       RETURNING id`,
      [projectId, title, description, html, bytes, sha256(html), datasetIds, JSON.stringify(source), snapshotAt, req.user.id]
    );
    auditProjectAction(req, 'project.report.publish', projectId, {
      resourceType: 'project_report',
      resourceId: inserted.rows[0].id,
      details: { title, size_bytes: bytes, dataset_ids: datasetIds, contribution: source.contribution?.slug || null },
    });
    const meta = await query(
      `SELECT ${REPORT_META_COLS}
       FROM w4h.project_reports r LEFT JOIN w4h.users u ON u.id = r.created_by
       WHERE r.id = $1::uuid`,
      [inserted.rows[0].id]
    );
    res.status(201).json(meta.rows[0]);
  } catch (error) {
    res.status(500).json({ error: error.message || 'Failed to publish report' });
  }
}

/** The report page itself, with a strict CSP injected. Rendered by the app in a sandboxed iframe. */
async function getProjectReportContent(req, res) {
  const projectId = await resolveProjectId(req.params.id);
  if (!projectId) return res.status(404).json({ error: 'Project not found' });
  if (!(await canReadProject(projectId, req.user))) {
    return res.status(403).json({ error: 'Unauthorized' });
  }
  try {
    const row = await query(
      `SELECT ${REPORT_META_COLS}, r.html
       FROM w4h.project_reports r LEFT JOIN w4h.users u ON u.id = r.created_by
       WHERE r.id = $1::uuid AND r.project_id = $2::uuid`,
      [req.params.reportId, projectId]
    );
    if (!row.rowCount) return res.status(404).json({ error: 'Report not found' });
    const { html, ...meta } = row.rows[0];
    auditProjectAction(req, 'project.report.view', projectId, {
      resourceType: 'project_report',
      resourceId: meta.id,
      details: { title: meta.title },
    });
    res.json({ report: meta, html: injectReportCsp(html) });
  } catch (error) {
    res.status(500).json({ error: error.message || 'Failed to load report' });
  }
}

async function deleteProjectReport(req, res) {
  const projectId = await resolveProjectId(req.params.id);
  if (!projectId) return res.status(404).json({ error: 'Project not found' });
  try {
    const row = await query(
      `SELECT id, title, created_by FROM w4h.project_reports WHERE id = $1::uuid AND project_id = $2::uuid`,
      [req.params.reportId, projectId]
    );
    if (!row.rowCount) return res.status(404).json({ error: 'Report not found' });
    const allowed = (await canManageProject(projectId, req.user))
      || Number(row.rows[0].created_by) === Number(req.user.id);
    if (!allowed) return res.status(403).json({ error: 'Unauthorized' });
    await query(`DELETE FROM w4h.project_reports WHERE id = $1::uuid`, [row.rows[0].id]);
    auditProjectAction(req, 'project.report.delete', projectId, {
      resourceType: 'project_report',
      resourceId: row.rows[0].id,
      details: { title: row.rows[0].title },
    });
    res.json({ ok: true });
  } catch (error) {
    res.status(500).json({ error: error.message || 'Failed to delete report' });
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

async function listProjectPages(req, res) {
  const projectId = await resolveProjectId(req.params.id);
  if (!projectId) return res.status(404).json({ error: 'Project not found' });
  if (!(await canReadProject(projectId, req.user))) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  try {
    const result = await query(
      `SELECT id, project_id, slug, title, doc_type, sort_order, updated_by, created_at, updated_at
       FROM w4h.project_pages
       WHERE project_id = $1::uuid
       ORDER BY sort_order ASC, title ASC`,
      [projectId]
    );
    res.json(result.rows.map(mapPageRow));
  } catch (error) {
    res.status(500).json({ error: error.message || 'Failed to list project pages' });
  }
}

async function getProjectPage(req, res) {
  const projectId = await resolveProjectId(req.params.id);
  if (!projectId) return res.status(404).json({ error: 'Project not found' });
  if (!(await canReadProject(projectId, req.user))) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  try {
    const slug = String(req.params.slug || '').trim();
    const result = await query(
      `SELECT *
       FROM w4h.project_pages
       WHERE project_id = $1::uuid AND slug = $2`,
      [projectId, slug]
    );
    if (!result.rowCount) return res.status(404).json({ error: 'Page not found' });
    res.json(mapPageRow(result.rows[0]));
  } catch (error) {
    res.status(500).json({ error: error.message || 'Failed to load project page' });
  }
}

async function createProjectPage(req, res) {
  const projectId = await resolveProjectId(req.params.id);
  if (!projectId) return res.status(404).json({ error: 'Project not found' });
  if (!(await canEditProjectPages(projectId, req.user))) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  try {
    const title = String(req.body?.title || '').trim() || 'Untitled document';
    const docType = normalizePageDocType(req.body?.doc_type);
    const slug = await uniquePageSlug(projectId, req.body?.slug || title);
    const mdRaw = req.body?.md != null ? String(req.body.md) : `# ${title}\n\n`;
    const md = stripLeadingTitleHeading(mdRaw, title);
    const { html, toc } = renderDoc(md);
    const sortOrder = Number.isFinite(Number(req.body?.sort_order))
      ? Number(req.body.sort_order)
      : 0;

    const inserted = await query(
      `INSERT INTO w4h.project_pages (
         project_id, slug, title, doc_type, sort_order, md, html, toc, created_by, updated_by
       )
       VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8::jsonb, $9, $10)
       RETURNING *`,
      [
        projectId,
        slug,
        title,
        docType,
        sortOrder,
        md,
        html,
        JSON.stringify(toc || []),
        req.user.id,
        req.user.email || null
      ]
    );

    auditProjectAction(req, 'project.page.create', projectId, {
      resourceType: 'doc',
      resourceId: slug,
      details: { slug, title, doc_type: docType }
    });

    res.status(201).json(mapPageRow(inserted.rows[0]));
  } catch (error) {
    res.status(500).json({ error: error.message || 'Failed to create project page' });
  }
}

async function updateProjectPage(req, res) {
  const projectId = await resolveProjectId(req.params.id);
  if (!projectId) return res.status(404).json({ error: 'Project not found' });
  if (!(await canEditProjectPages(projectId, req.user))) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  try {
    const currentSlug = String(req.params.slug || '').trim();
    const existing = await query(
      `SELECT * FROM w4h.project_pages WHERE project_id = $1::uuid AND slug = $2`,
      [projectId, currentSlug]
    );
    if (!existing.rowCount) return res.status(404).json({ error: 'Page not found' });

    const row = existing.rows[0];
    const title =
      req.body?.title != null ? String(req.body.title).trim() || row.title : row.title;
    const docType =
      req.body?.doc_type != null ? normalizePageDocType(req.body.doc_type) : row.doc_type;
    const sortOrder =
      req.body?.sort_order != null && Number.isFinite(Number(req.body.sort_order))
        ? Number(req.body.sort_order)
        : row.sort_order;
    const mdRaw = req.body?.md != null ? String(req.body.md) : row.md;
    const md = stripLeadingTitleHeading(mdRaw, title);
    const { html, toc } = renderDoc(md);

    let nextSlug = currentSlug;
    if (req.body?.slug != null && String(req.body.slug).trim()) {
      nextSlug = await uniquePageSlug(projectId, String(req.body.slug).trim(), currentSlug);
    }

    const updated = await query(
      `UPDATE w4h.project_pages
       SET slug = $3,
           title = $4,
           doc_type = $5,
           sort_order = $6,
           md = $7,
           html = $8,
           toc = $9::jsonb,
           updated_by = $10,
           updated_at = NOW()
       WHERE project_id = $1::uuid AND slug = $2
       RETURNING *`,
      [
        projectId,
        currentSlug,
        nextSlug,
        title,
        docType,
        sortOrder,
        md,
        html,
        JSON.stringify(toc || []),
        req.user.email || null
      ]
    );

    auditProjectAction(req, 'project.page.update', projectId, {
      resourceType: 'doc',
      resourceId: nextSlug,
      details: { slug: nextSlug, title, doc_type: docType }
    });

    res.json(mapPageRow(updated.rows[0]));
  } catch (error) {
    res.status(500).json({ error: error.message || 'Failed to update project page' });
  }
}

async function deleteProjectPage(req, res) {
  const projectId = await resolveProjectId(req.params.id);
  if (!projectId) return res.status(404).json({ error: 'Project not found' });
  if (!(await canEditProjectPages(projectId, req.user))) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  try {
    const slug = String(req.params.slug || '').trim();
    const deleted = await query(
      `DELETE FROM w4h.project_pages
       WHERE project_id = $1::uuid AND slug = $2
       RETURNING slug, title`,
      [projectId, slug]
    );
    if (!deleted.rowCount) return res.status(404).json({ error: 'Page not found' });

    auditProjectAction(req, 'project.page.delete', projectId, {
      resourceType: 'doc',
      resourceId: slug,
      details: { slug, title: deleted.rows[0].title }
    });

    res.json({ success: true, slug });
  } catch (error) {
    res.status(500).json({ error: error.message || 'Failed to delete project page' });
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
  const RESOURCE = new Set(['project', 'dataset', 'doc', 'derived_table']);
  const PERM = new Set(['view', 'edit', 'manage_members', 'link_dataset', 'create_derived']);
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
