import express from 'express';
const router = express.Router();

import { authenticateJwtOrApiKey, requireTeamAdmin, requirePlatformAdmin } from '../auth.js';
import {
  isSuperAdmin,
  canManageTeamAsync,
  isTeamAdminOnTeam,
} from '../lib/utils/user.js';
import { query } from '../db.js';
import { resolveDatasetProjectLinkColumn } from '../lib/utils/datasetProjectLinkColumn.js';
import { describeViz, runViz } from '../lib/viz-queries.js';
import { writeAuditLogSafe } from '../lib/utils/audit.js';
import { importCsvBatch, assertCanIngestDataset } from '../lib/ingest-csv.js';
import { refreshDatasetSummaryData } from '../lib/dataset-summary-refresh.js';
import { listDatasetSignals, runSqlContribution, assertCanViewDataset } from '../lib/sql-contribution-engine.js';
import { getCohortAnalyticsSetup, runCohortComparison } from '../lib/cohort-analytics.js';
import { attachMetadataSummaries } from '../lib/dataset-metadata-summary.js';

router.get('/projects', authenticateJwtOrApiKey, listProjects);
// Legacy projects-admin CRUD is platform-wide — Super admin only
router.get('/projects-admin', authenticateJwtOrApiKey, requirePlatformAdmin, listProjectsAdmin);
router.get('/projects-admin/:id', authenticateJwtOrApiKey, requirePlatformAdmin, getProjectAdmin);
router.post('/projects-admin', authenticateJwtOrApiKey, requirePlatformAdmin, addProjectAdmin);
router.put('/projects-admin/:id', authenticateJwtOrApiKey, requirePlatformAdmin, setProjectAdmin);
router.get('/tableinfo', authenticateJwtOrApiKey, tableInfo);
router.get('/', authenticateJwtOrApiKey, listDatasets);
router.post('/', authenticateJwtOrApiKey, createDataset);
// Share registry (ADR-002 / ADR-012) — register before /:id
router.get('/shares', authenticateJwtOrApiKey, requireTeamAdmin, listDatasetShares);
router.get('/share-targets', authenticateJwtOrApiKey, requireTeamAdmin, listShareTargets);
router.delete('/shares/:shareId', authenticateJwtOrApiKey, requireTeamAdmin, revokeDatasetShare);
router.get('/:id/shares', authenticateJwtOrApiKey, requireTeamAdmin, listSharesForDataset);
router.post('/:id/shares', authenticateJwtOrApiKey, requireTeamAdmin, createDatasetShare);
router.delete('/:id', authenticateJwtOrApiKey, requireTeamAdmin, deleteDataset);
router.patch('/:id', authenticateJwtOrApiKey, requireTeamAdmin, updateDataset);
router.post('/:id/import/csv-batch', authenticateJwtOrApiKey, importCsvBatchHandler);
router.post('/:id/summary/refresh', authenticateJwtOrApiKey, refreshDatasetSummary);
router.post('/:id/contributions/:slug/sql', authenticateJwtOrApiKey, runDatasetSqlContribution);
router.get('/:id/dependents', authenticateJwtOrApiKey, getDatasetDependents);
router.get('/:id/signals', authenticateJwtOrApiKey, getDatasetSignals);
router.get('/:id/analytics/cohort', authenticateJwtOrApiKey, getDatasetCohortAnalytics);
router.post('/:id/analytics/cohort', authenticateJwtOrApiKey, postDatasetCohortAnalytics);
router.get('/:id/analytics/catalog', authenticateJwtOrApiKey, getDatasetVizCatalog);
router.post('/:id/analytics/series', authenticateJwtOrApiKey, postDatasetViz);
router.get('/:id', authenticateJwtOrApiKey, getDataset);

const STATUS_TABLE_NAME = 'w4h.status_log';
const DATASET_ALLOWED_ROLES = new Set([
  'admin', 'super_admin', 'team_admin', 'project_user', 'researcher', 'clinician'
]);

function canAccessDatasets(user) {
  return !!user && DATASET_ALLOWED_ROLES.has(user.role);
}

async function canManageDatasets(user) {
  return canManageTeamAsync(user);
}

function isAdmin(user) {
  return !!user && isSuperAdmin(user.role);
}

function slugifyDatasetId(input) {
  return String(input || '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/(^-|-$)/g, '');
}

async function getTeamIdsForUser(userId) {
  const res = await query(
    `SELECT team_id::text AS team_id FROM w4h.team_members WHERE user_id = $1`,
    [userId]
  );
  return res.rows.map((r) => r.team_id);
}

async function canAccessTeam(user, teamId) {
  if (!user || !teamId) return false;
  if (isSuperAdmin(user.role)) return false;
  return (await getTeamIdsForUser(user.id)).includes(String(teamId));
}

async function actorCanManageCatalogOnTeam(user, teamId) {
  if (!user || !teamId) return false;
  if (isSuperAdmin(user.role)) return false;
  if (await isTeamAdminOnTeam(user.id, teamId, user.role)) return true;
  const owner = await query(
    `SELECT 1
     FROM w4h.project_members pm
     JOIN w4h.projects p ON p.id = pm.project_id
     WHERE pm.user_id = $1 AND pm.role = 'owner' AND p.team_id = $2::uuid
     LIMIT 1`,
    [user.id, teamId]
  );
  return owner.rowCount > 0;
}

function parseCatalogStatusFilter(raw) {
  return ['active', 'archived', 'all'].includes(raw) ? raw : 'active';
}

/** SQL fragment excluding archived datasets (COALESCE for pre-migration rows). */
function sqlDatasetActiveOnly(alias = 'd') {
  return `COALESCE(${alias}.status, 'active') = 'active'`;
}

function sqlOwnedDatasetStatusFilter(alias, statusFilter, params) {
  if (statusFilter === 'all') return '';
  if (statusFilter === 'archived') {
    params.push('archived');
    return ` AND COALESCE(${alias}.status, 'active') = $${params.length}`;
  }
  return ` AND ${sqlDatasetActiveOnly(alias)}`;
}

/**
 * Access via project link (ADR-012): datasets.project_id is legacy text and must not be
 * compared to user_projects.project_id (uuid). Use project_datasets instead.
 */
function sqlDatasetAccessibleToUser(userParam, datasetAlias = 'd') {
  return `(
    EXISTS (
      SELECT 1
      FROM w4h.project_datasets pd
      JOIN w4h.project_members pm ON pm.project_id = pd.project_id
      WHERE pd.dataset_id = ${datasetAlias}.id AND pm.user_id = ${userParam}
    )
    OR EXISTS (
      SELECT 1
      FROM w4h.project_datasets pd
      JOIN w4h.user_projects us ON us.project_id = pd.project_id
      WHERE pd.dataset_id = ${datasetAlias}.id AND us.user_id = ${userParam}
    )
  )`;
}

async function ensureUniqueDatasetId(base) {
  const baseId = slugifyDatasetId(base) || `dataset-${Date.now()}`;
  let id = baseId;
  let i = 1;
  while (true) {
    const check = await query(`SELECT id FROM w4h.datasets WHERE id = $1`, [id]);
    if (!check.rowCount) return id;
    id = `${baseId}-${i}`;
    i += 1;
  }
}

const CONTACT_EMAILS_SQL = `
         COALESCE((
           SELECT STRING_AGG(u.email, ', ' ORDER BY u.email)
           FROM w4h.team_members tm
           JOIN w4h.users u ON u.id = tm.user_id
           WHERE tm.team_id = d.team_id
             AND tm.role = 'team_admin'
             AND COALESCE(u.status, 'active') = 'active'
         ), '') AS contact_emails,
         COALESCE((
           SELECT STRING_AGG(COALESCE(NULLIF(TRIM(u.name), ''), u.email), ', ' ORDER BY u.email)
           FROM w4h.team_members tm
           JOIN w4h.users u ON u.id = tm.user_id
           WHERE tm.team_id = d.team_id
             AND tm.role = 'team_admin'
             AND COALESCE(u.status, 'active') = 'active'
         ), '') AS contact_names`;

const CATALOG_SELECT = `
  SELECT d.id, d.name, d.description, d.team_id, d.project_id,
         d.device_type, d.sampling_rate, d.subject_count,
         d.date_start, d.date_end, d.irb_reference,
         d.discoverable, d.read_only, d.origin, d.status,
         d.created_by, d.created_at, d.updated_at,
         t.name AS team_name, t.slug AS team_slug,
         ${CONTACT_EMAILS_SQL},
         COALESCE((
           SELECT STRING_AGG(p.name, ', ' ORDER BY p.name)
           FROM w4h.project_datasets pd
           JOIN w4h.projects p ON p.id = pd.project_id
           WHERE pd.dataset_id = d.id
         ), '') AS project_names,
         COALESCE((
           SELECT json_agg(
             json_build_object('id', p.id::text, 'name', p.name)
             ORDER BY p.name
           )
           FROM w4h.project_datasets pd
           JOIN w4h.projects p ON p.id = pd.project_id
           WHERE pd.dataset_id = d.id
         ), '[]'::json) AS projects,
         'local'::text AS source_kind,
         'Local'::text AS source_label
  FROM w4h.datasets d
  LEFT JOIN w4h.teams t ON t.id = d.team_id
`;

/** Shared catalog column list (owned / shared / listed variants differ in source_kind + projects). */
const CATALOG_COLS = `
             d.id, d.name, d.description, d.team_id, d.project_id,
             d.device_type, d.sampling_rate, d.subject_count,
             d.date_start, d.date_end, d.irb_reference,
             d.discoverable, d.read_only, d.origin, d.status,
             d.created_by, d.created_at, d.updated_at
`;

/** Signal-table count for catalog list/detail (0 when w4h.tables is not provisioned). */
async function attachTableCounts(rows) {
  if (!rows?.length) return rows || [];
  let tableMap = {};
  try {
    const tablesRes = await query(
      `SELECT dataset_id, COUNT(*)::int AS n
       FROM w4h.tables
       WHERE dataset_id = ANY($1::text[])
       GROUP BY dataset_id`,
      [rows.map((r) => r.id)]
    );
    tableMap = Object.fromEntries((tablesRes.rows || []).map((r) => [r.dataset_id, r.n]));
  } catch {
    // Signal-table catalog not provisioned yet.
  }
  return rows.map((r) => ({
    ...r,
    project_names: r.project_names || '',
    projects: Array.isArray(r.projects) ? r.projects : [],
    table_count: tableMap[r.id] ?? 0,
  }));
}

async function listDatasetsCatalog(req, res) {
  try {
    const statusFilter = parseCatalogStatusFilter(req.query?.status);

    if (isSuperAdmin(req.user.role)) {
      return res.status(403).json({ error: 'Super admin does not browse team dataset catalogs' });
    }

    const teamIds = await getTeamIdsForUser(req.user.id);
    if (!teamIds.length) return res.json([]);

    const ownedParams = [teamIds];
    const ownedStatusSql = sqlOwnedDatasetStatusFilter('d', statusFilter, ownedParams);
    const owned = await query(
      `
      SELECT ${CATALOG_COLS},
             t.name AS team_name, t.slug AS team_slug,
             ${CONTACT_EMAILS_SQL},
             COALESCE((
               SELECT STRING_AGG(p.name, ', ' ORDER BY p.name)
               FROM w4h.project_datasets pd
               JOIN w4h.projects p ON p.id = pd.project_id
               WHERE pd.dataset_id = d.id
             ), '') AS project_names,
             COALESCE((
               SELECT json_agg(
                 json_build_object('id', p.id::text, 'name', p.name)
                 ORDER BY p.name
               )
               FROM w4h.project_datasets pd
               JOIN w4h.projects p ON p.id = pd.project_id
               WHERE pd.dataset_id = d.id
             ), '[]'::json) AS projects,
             'local'::text AS source_kind,
             'Local'::text AS source_label
      FROM w4h.datasets d
      LEFT JOIN w4h.teams t ON t.id = d.team_id
      WHERE d.team_id = ANY($1::uuid[])${ownedStatusSql}
      ORDER BY d.name ASC NULLS LAST, d.id ASC
      `,
      ownedParams
    );

    // Inbound shares and discoverable listings are active-only (archived datasets revoke shares).
    const shared = statusFilter === 'archived' ? { rows: [] } : await query(
      `
      SELECT DISTINCT ON (d.id)
             ${CATALOG_COLS},
             ot.name AS team_name, ot.slug AS team_slug,
             ${CONTACT_EMAILS_SQL},
             COALESCE((
               SELECT STRING_AGG(p.name, ', ' ORDER BY p.name)
               FROM w4h.project_datasets pd
               JOIN w4h.projects p ON p.id = pd.project_id
               WHERE pd.dataset_id = d.id
             ), '') AS project_names,
             COALESCE((
               SELECT json_agg(
                 json_build_object('id', p.id::text, 'name', p.name)
                 ORDER BY p.name
               )
               FROM w4h.project_datasets pd
               JOIN w4h.projects p ON p.id = pd.project_id
               WHERE pd.dataset_id = d.id
             ), '[]'::json) AS projects,
             'shared'::text AS source_kind,
             CASE
               WHEN ot.name IS NOT NULL THEN 'Shared by ' || ot.name
               ELSE 'Shared'
             END AS source_label
      FROM w4h.dataset_shares ds
      JOIN w4h.datasets d ON d.id = ds.dataset_id
      LEFT JOIN w4h.teams ot ON ot.id = d.team_id
      WHERE ds.target_team_id = ANY($1::uuid[])
        AND (d.team_id IS NULL OR NOT (d.team_id = ANY($1::uuid[])))
        AND ${sqlDatasetActiveOnly('d')}
      ORDER BY d.id, d.name ASC NULLS LAST
      `,
      [teamIds]
    );

    const listed = statusFilter === 'archived' ? { rows: [] } : await query(
      `
      SELECT ${CATALOG_COLS},
             ot.name AS team_name, ot.slug AS team_slug,
             ${CONTACT_EMAILS_SQL},
             ''::text AS project_names,
             '[]'::json AS projects,
             'listed'::text AS source_kind,
             CASE
               WHEN ot.name IS NOT NULL THEN 'Listed · ' || ot.name
               ELSE 'Listed'
             END AS source_label
      FROM w4h.datasets d
      LEFT JOIN w4h.teams ot ON ot.id = d.team_id
      WHERE d.discoverable = true
        AND ${sqlDatasetActiveOnly('d')}
        AND d.team_id IS NOT NULL
        AND NOT (d.team_id = ANY($1::uuid[]))
        AND NOT EXISTS (
          SELECT 1 FROM w4h.dataset_shares ds
          WHERE ds.dataset_id = d.id
            AND ds.target_team_id = ANY($1::uuid[])
        )
      ORDER BY d.name ASC NULLS LAST, d.id ASC
      `,
      [teamIds]
    );

    const byId = new Map();
    for (const row of owned.rows) byId.set(row.id, row);
    for (const row of shared.rows) {
      if (!byId.has(row.id)) byId.set(row.id, row);
    }
    for (const row of listed.rows) {
      if (!byId.has(row.id)) byId.set(row.id, row);
    }
    const merged = [...byId.values()].sort((a, b) =>
      String(a.name || a.id).localeCompare(String(b.name || b.id), undefined, { sensitivity: 'base' })
    );
    res.json(await attachTableCounts(merged));
  } catch (error) {
    console.error('listDatasetsCatalog:', error);
    res.status(500).json({ error: error.message || 'Failed to list datasets' });
  }
}

async function createDataset(req, res) {
  try {
    const name = String(req.body?.name || '').trim();
    if (!name) return res.status(400).json({ error: 'name is required' });

    let teamId = req.body?.team_id == null ? null : String(req.body.team_id).trim();
    if (!teamId) {
      const adminTeams = await query(
        `SELECT team_id::text AS team_id FROM w4h.team_members WHERE user_id = $1 AND role = 'team_admin'`,
        [req.user.id]
      );
      if (adminTeams.rowCount === 1) teamId = adminTeams.rows[0].team_id;
    }
    if (!teamId) return res.status(400).json({ error: 'team_id is required' });

    const teamCheck = await query(`SELECT id FROM w4h.teams WHERE id = $1::uuid`, [teamId]);
    if (!teamCheck.rowCount) return res.status(400).json({ error: 'Unknown team_id' });
    if (!(await actorCanManageCatalogOnTeam(req.user, teamId))) {
      return res.status(403).json({ error: 'Unauthorized for this team' });
    }

    const id = await ensureUniqueDatasetId(req.body?.id || name);
    const description = req.body?.description == null ? null : String(req.body.description).trim();
    const deviceType = req.body?.device_type == null ? null : String(req.body.device_type).trim();
    const samplingRate = req.body?.sampling_rate == null ? null : String(req.body.sampling_rate).trim();
    const subjectCount =
      req.body?.subject_count === '' || req.body?.subject_count == null
        ? null
        : Number(req.body.subject_count);
    const dateStart = req.body?.date_start || null;
    const dateEnd = req.body?.date_end || null;
    const irbReference = req.body?.irb_reference == null ? null : String(req.body.irb_reference).trim();
    const discoverable = req.body?.discoverable === false || req.body?.discoverable === 'false'
      ? false
      : true;

    if (subjectCount != null && !Number.isFinite(subjectCount)) {
      return res.status(400).json({ error: 'subject_count must be a number' });
    }

    const inserted = await query(
      `INSERT INTO w4h.datasets
         (id, name, description, team_id, device_type, sampling_rate, subject_count,
          date_start, date_end, irb_reference, created_by, discoverable)
       VALUES ($1, $2, $3, $4::uuid, $5, $6, $7, $8, $9, $10, $11, $12)
       RETURNING id`,
      [
        id,
        name,
        description || null,
        teamId,
        deviceType || null,
        samplingRate || null,
        subjectCount,
        dateStart,
        dateEnd,
        irbReference || null,
        req.user.id ?? null,
        discoverable,
      ]
    );

    const fresh = await query(
      `${CATALOG_SELECT} WHERE d.id = $1`,
      [inserted.rows[0].id]
    );
    res.status(201).json(fresh.rows[0]);
  } catch (error) {
    if (error.code === '23505') {
      return res.status(409).json({ error: 'Dataset id already exists' });
    }
    res.status(500).json({ error: error.message || 'Failed to create dataset' });
  }
}

async function deleteDataset(req, res) {
  if (!(await canManageDatasets(req.user))) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  try {
    const datasetId = String(req.params.id || '').trim();
    const existing = await query(
      `SELECT id, team_id, COALESCE(read_only, false) AS read_only
       FROM w4h.datasets WHERE id = $1`,
      [datasetId]
    );
    if (!existing.rowCount) {
      return res.status(404).json({ error: 'Dataset not found' });
    }

    const currentTeamId = existing.rows[0].team_id;
    const isReadOnly = Boolean(existing.rows[0].read_only);
    if (currentTeamId && !(await canAccessTeam(req.user, currentTeamId))) {
      return res.status(403).json({ error: 'Unauthorized' });
    }
    if (!currentTeamId) {
      return res.status(403).json({ error: 'Unauthorized' });
    }
    if (!(await actorCanManageCatalogOnTeam(req.user, currentTeamId))) {
      return res.status(403).json({ error: 'Only the owning team admin can delete this dataset' });
    }
    if (isReadOnly) {
      return res.status(403).json({ error: 'Platform sample datasets cannot be deleted via the API' });
    }

    const tableCountRes = await query(
      `SELECT COUNT(*)::int AS n FROM w4h.tables WHERE dataset_id = $1`,
      [datasetId]
    );
    if ((tableCountRes.rows[0]?.n || 0) > 0) {
      return res.status(409).json({ error: 'Only empty datasets can be deleted completely' });
    }

    await query(`DELETE FROM w4h.dataset_shares WHERE dataset_id = $1`, [datasetId]);
    await query(`DELETE FROM w4h.project_datasets WHERE dataset_id = $1`, [datasetId]);
    await query(`DELETE FROM w4h.datasets WHERE id = $1`, [datasetId]);
    writeAuditLogSafe(req, {
      action: 'dataset.delete',
      resourceType: 'dataset',
      resourceId: datasetId,
      details: { deleted: true },
    });

    res.json({ ok: true, deleted: true, dataset_id: datasetId });
  } catch (error) {
    res.status(500).json({ error: error.message || 'Failed to delete dataset' });
  }
}

async function updateDataset(req, res) {
  if (!(await canManageDatasets(req.user))) {
    return res.status(403).json({ error: 'Unauthorized' });
  }
  try {
    const datasetId = String(req.params.id || '').trim();
    const existing = await query(
      `SELECT id, team_id, COALESCE(read_only, false) AS read_only,
              COALESCE(status, 'active') AS status
       FROM w4h.datasets WHERE id = $1`,
      [datasetId]
    );
    if (!existing.rowCount) return res.status(404).json({ error: 'Dataset not found' });

    const currentTeamId = existing.rows[0].team_id;
    const priorStatus = existing.rows[0].status || 'active';
    const isReadOnly = Boolean(existing.rows[0].read_only);
    if (currentTeamId && !(await canAccessTeam(req.user, currentTeamId))) {
      return res.status(403).json({ error: 'Unauthorized' });
    }
    if (!currentTeamId) {
      return res.status(403).json({ error: 'Unauthorized' });
    }

    const fields = [];
    const params = [];
    const allowed = [
      ['name', 'name'],
      ['description', 'description'],
      ['device_type', 'device_type'],
      ['sampling_rate', 'sampling_rate'],
      ['date_start', 'date_start'],
      ['date_end', 'date_end'],
      ['irb_reference', 'irb_reference']
    ];

    for (const [bodyKey, column] of allowed) {
      if (Object.prototype.hasOwnProperty.call(req.body || {}, bodyKey)) {
        const val = req.body[bodyKey];
        params.push(val === '' || val == null ? null : val);
        fields.push(`${column} = $${params.length}`);
      }
    }

    if (Object.prototype.hasOwnProperty.call(req.body || {}, 'subject_count')) {
      const raw = req.body.subject_count;
      const subjectCount = raw === '' || raw == null ? null : Number(raw);
      if (subjectCount != null && !Number.isFinite(subjectCount)) {
        return res.status(400).json({ error: 'subject_count must be a number' });
      }
      params.push(subjectCount);
      fields.push(`subject_count = $${params.length}`);
    }

    if (Object.prototype.hasOwnProperty.call(req.body || {}, 'discoverable')) {
      const raw = req.body.discoverable;
      const on = raw === true || raw === 'true' || raw === 1 || raw === '1';
      params.push(!!on);
      fields.push(`discoverable = $${params.length}`);
    }

    if (Object.prototype.hasOwnProperty.call(req.body || {}, 'read_only')) {
      return res.status(403).json({
        error: 'read_only sample datasets are set by platform seed/migrations, not the API.',
      });
    }

    if (Object.prototype.hasOwnProperty.call(req.body || {}, 'origin')) {
      return res.status(403).json({
        error: 'Dataset origin is set by platform seed/migrations, not the API.',
      });
    }

    if (Object.prototype.hasOwnProperty.call(req.body || {}, 'team_id')) {
      return res.status(403).json({
        error: 'Owning team is fixed at dataset creation; Super admin does not reassign catalogs.',
      });
    }

    if (Object.prototype.hasOwnProperty.call(req.body || {}, 'status')) {
      const nextStatus = req.body.status === 'archived' ? 'archived' : 'active';
      if (!currentTeamId || !(await actorCanManageCatalogOnTeam(req.user, currentTeamId))) {
        return res.status(403).json({ error: 'Only the owning team admin can archive or restore this dataset' });
      }
      if (nextStatus === 'archived' && isReadOnly) {
        return res.status(403).json({ error: 'Platform sample datasets cannot be archived via the API' });
      }
      params.push(nextStatus);
      fields.push(`status = $${params.length}`);
    }

    if (!fields.length) {
      return res.status(400).json({ error: 'No updatable fields provided' });
    }

    fields.push('updated_at = NOW()');
    params.push(datasetId);
    await query(
      `UPDATE w4h.datasets SET ${fields.join(', ')} WHERE id = $${params.length}`,
      params
    );

    const nextStatus = Object.prototype.hasOwnProperty.call(req.body || {}, 'status')
      ? (req.body.status === 'archived' ? 'archived' : 'active')
      : priorStatus;
    if (nextStatus === 'archived' && priorStatus !== 'archived') {
      await query(`DELETE FROM w4h.dataset_shares WHERE dataset_id = $1`, [datasetId]);
      await query(`DELETE FROM w4h.project_datasets WHERE dataset_id = $1`, [datasetId]);
    }

    const fresh = await query(`${CATALOG_SELECT} WHERE d.id = $1`, [datasetId]);
    res.json(fresh.rows[0]);
  } catch (error) {
    res.status(500).json({ error: error.message || 'Failed to update dataset' });
  }
}

async function listProjectsAdmin(req, res) {
  if (!isAdmin(req.user)) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  try {
    const result = await query(
      `SELECT
         s.id,
         s.name,
         s.description,
         COUNT(DISTINCT pd.dataset_id)::int AS dataset_count,
         COUNT(DISTINCT us.user_id)::int AS user_count
       FROM w4h.projects s
       LEFT JOIN w4h.project_datasets pd ON pd.project_id = s.id
       LEFT JOIN w4h.user_projects us ON us.project_id = s.id
       GROUP BY s.id, s.name, s.description
       ORDER BY s.name ASC`
    );

    return res.json(result.rows);
  } catch (error) {
    return res.status(500).json({ error: error.message });
  }
}

async function getProjectAdmin(req, res) {
  if (!isAdmin(req.user)) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  try {
    const result = await query(
      `SELECT id, name, description FROM w4h.projects WHERE id = $1`,
      [req.params.id]
    );

    if (result.rowCount === 0) {
      return res.status(404).json({ error: 'Study not found' });
    }

    return res.json(result.rows[0]);
  } catch (error) {
    return res.status(500).json({ error: error.message });
  }
}

async function addProjectAdmin(req, res) {
  if (!isAdmin(req.user)) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  try {
    const id = String(req.body?.id || '').trim();
    const name = String(req.body?.name || '').trim();
    const description = req.body?.description == null ? null : String(req.body.description).trim();

    if (!id || !name) {
      return res.status(400).json({ error: 'id and name are required' });
    }

    await query(
      `INSERT INTO w4h.projects (id, name, description)
       VALUES ($1, $2, $3)`,
      [id, name, description || null]
    );

    await query(
      `INSERT INTO w4h.user_projects (user_id, project_id)
       SELECT u.id, $1
       FROM w4h.users u
       WHERE u.role = 'admin'
       ON CONFLICT (user_id, project_id) DO NOTHING`,
      [id]
    );

    return res.status(201).json({ success: true, message: 'Project created' });
  } catch (error) {
    if (error.code === '23505') {
      return res.status(409).json({ error: 'Project id already exists' });
    }
    return res.status(500).json({ error: error.message });
  }
}

async function setProjectAdmin(req, res) {
  if (!isAdmin(req.user)) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  try {
    const name = String(req.body?.name || '').trim();
    const description = req.body?.description == null ? null : String(req.body.description).trim();

    if (!name) {
      return res.status(400).json({ error: 'name is required' });
    }

    const result = await query(
      `UPDATE w4h.projects
       SET name = $1, description = $2
       WHERE id = $3`,
      [name, description || null, req.params.id]
    );

    if (result.rowCount === 0) {
      return res.status(404).json({ error: 'Project not found' });
    }

    return res.json({ success: true, message: 'Project updated' });
  } catch (error) {
    return res.status(500).json({ error: error.message });
  }
}

async function listProjects(req, res) {
  if (!canAccessDatasets(req.user)) {
    return res.status(403).json({ error: 'Unauthorized' });
  }
  if (isSuperAdmin(req.user.role)) {
    return res.status(403).json({ error: 'Super admin does not browse trial projects' });
  }

  let sql = `
    SELECT id, name
    FROM w4h.projects
  `;
  const params = [];

  sql += `
    WHERE EXISTS (
      SELECT 1
      FROM w4h.user_projects us
      WHERE us.project_id = w4h.projects.id
        AND us.user_id = $1
    )
  `;
  params.push(req.user.id);

  sql += ` ORDER BY name ASC`;
  const result = await query(sql, params);
  res.json(result.rows);
}

async function tableInfo(req, res) {
  if (!canAccessDatasets(req.user)) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  let columnIds = [];
  let columns = [];
  let values = [];
  let partitions = [];

  try {
    const pc = await resolveDatasetProjectLinkColumn();
    if (isSuperAdmin(req.user.role)) {
      return res.status(403).json({ error: 'Super admin does not query trial dataset tables' });
    }
    const datasetAccessResult = await query(
      `SELECT d.id, d.${pc} AS dataset_project_id
       FROM w4h.datasets d
       WHERE d.id = $1 AND ${sqlDatasetAccessibleToUser('$2')}`,
      [req.query.dataset_id, req.user.id]
    );

    if (datasetAccessResult.rows.length === 0) {
      return res.status(403).json({ error: 'Unauthorized for requested dataset' });
    }

    const catalogProjectId =
      req.query.project_id || datasetAccessResult.rows[0].dataset_project_id;
    const queryMap = await getQueryMap(catalogProjectId);
    const dataset = queryMap.get(req.query.dataset_id);

    if (!dataset) {
      return res.status(400).json({ error: 'Invalid dataset_id' });
    }

    const table = dataset.datasetInfo.tables[0];
    if (!table) {
      return res.status(400).json({ error: 'Missing table for dataset' });
    }

    const [schemaName, tableName] = table.includes('.')
      ? table.split('.')
      : ['public', table];

    const colSql = `
      SELECT column_name AS name, data_type AS type
      FROM information_schema.columns
      WHERE table_schema = $1 AND table_name = $2
      ORDER BY ordinal_position;
    `;
    const colResult = await query(colSql, [schemaName, tableName]);
    columnIds = colResult.rows.map(r => r.name);

    const textColumns = colResult.rows
      .filter(r => ['character varying', 'varchar', 'text'].includes(r.type))
      .map(r => r.name);

    if (columnIds.includes(XT_PARTITION_ID)) {
      const partitionTable = dataset.datasetInfo.partitionTableInfo.table;
      const partitionSql = `
  SELECT id, name
  FROM ${partitionTable}
  ORDER BY name;
`;
      const partitionResult = await query(partitionSql);
      partitions = partitionResult.rows;
    }

    const sampleSql = `
      SELECT *
      FROM ${table}
      TABLESAMPLE SYSTEM (1)
      LIMIT 200;
    `;
    const sampleResult = await query(sampleSql);
    const sampleRows = sampleResult.rows || [];

    for (const col of textColumns) {
      const sampleValues = sampleRows
        .map(r => r[col])
        .filter(v => v != null && v !== '');

      if (sampleValues.length === 0) continue;

      const unique = [...new Set(sampleValues)];

      if (unique.length >= 190) {
        if (process.env.W4H_DEBUG) console.log(`⚙️ Skipping noisy column: ${col}`);
        continue;
      }

      values[col] = unique;
    }

    const columns = colResult.rows;
    columns.forEach(col => {
      if (values[col.name]) {
        col.values = values[col.name];
      }
    });

    return res.json({ columns, partitions });
  } catch (err) {
    console.error('tableInfo error:', err);
    return res.status(500).json({
      error: 'Failed to load table info',
      message: err.message,
      details: process.env.NODE_ENV === 'development' ? err.stack : undefined
    });
  }
}

async function processDataset(dataset, user, QUERY_MAP, tableConfigs) {
  const datasetId = dataset.id;
  const mixins = [];
  for (const [key, entry] of QUERY_MAP.entries()) {
    if (entry.datasetInfo && entry.datasetInfo.id === datasetId) {
      mixins.push({ id: key, ...entry });
    }
  }

  mixins.sort((a, b) => {
    const aHasColon = a.id.includes(':');
    const bHasColon = b.id.includes(':');
    if (aHasColon !== bHasColon) {
      return aHasColon ? 1 : -1;
    }
    return (a.description || '').localeCompare(b.description || '');
  });

  const datasetInfo = (mixins[0] && mixins[0].datasetInfo) || {};
  mixins.forEach(mixin => { delete mixin.datasetInfo; });

  const processedDataset = {
    ...dataset,
    ...datasetInfo,
    mixins,
  };

  const allUniqueTableNames = new Set();
  processedDataset.tables?.forEach(table => {
    allUniqueTableNames.add(table);
    allUniqueTableNames.add(table + '_staging');
  });

  const datasetTables = tableConfigs
    .filter(d => d.dataset_id === datasetId);

  datasetTables.forEach(d => {
    allUniqueTableNames.add(d.name);
    allUniqueTableNames.add(d.name + '_staging');
  });

  const tableNamesArray = Array.from(allUniqueTableNames);
  let statusMap = new Map();

  if (tableNamesArray.length > 0) {
    const placeholders = tableNamesArray.map((_, i) => `$${i + 1}`).join(', ');
    const statusResult = await query(`
        SELECT *
        FROM ${STATUS_TABLE_NAME}
        WHERE table_name IN (${placeholders})
      `, tableNamesArray);

    statusResult.rows.forEach(row => statusMap.set(row.table_name, row));
  }

  if (processedDataset.tables) {
    processedDataset.tables = processedDataset.tables?.map((table, index) => ({
      name: table,
      status: statusMap.get(table)
        || statusMap.get(table + '_staging')
        || null,
      source_name: datasetInfo.sources?.[index]?.name || null,
      source_url: datasetInfo.sources?.[index]?.url || null
    })) || [];
    delete processedDataset.metabase_urls;
    delete processedDataset.sources;
  } else {
    processedDataset.tables = datasetTables;

    processedDataset.tables = processedDataset.tables.map(tableInfo => ({
      name: tableInfo.name,
      source_name: tableInfo.source_name || null,
      source_url: tableInfo.source_url || null,
      status: statusMap.get(tableInfo.name)
        || statusMap.get(tableInfo.name + '_staging')
        || null,
    }));
  }

  return processedDataset;
}

async function listDatasets(req, res) {
  try {
    if (!canAccessDatasets(req.user)) {
      return res.status(403).json({ error: 'Unauthorized' });
    }

    // Team catalog for Team admin / Super admin (admin UI).
    // Default managers to catalog when not filtering by project_id — avoids the legacy
    // datasets.project_id (text) = user_projects.project_id (uuid) join crash.
    if ((await canManageTeamAsync(req.user)) && (req.query.catalog === '1' || !req.query.project_id)) {
      return listDatasetsCatalog(req, res);
    }

    const pc = await resolveDatasetProjectLinkColumn();
    let sql = `
      SELECT DISTINCT d.id, d.${pc} AS dataset_project_id, d.name, d.description
      FROM w4h.datasets d
    `;
    const params = [];
    const where = [];

    if (isSuperAdmin(req.user.role)) {
      return res.status(403).json({ error: 'Super admin does not list trial datasets' });
    }
    params.push(req.user.id);
    where.push(sqlDatasetAccessibleToUser(`$${params.length}`));

    const filterProjectId = req.query.project_id;
    if (filterProjectId) {
      params.push(filterProjectId);
      where.push(`EXISTS (
        SELECT 1 FROM w4h.project_datasets pd
        WHERE pd.dataset_id = d.id AND pd.project_id = $${params.length}::uuid
      )`);
    }

    if (where.length) {
      sql += ` WHERE ${where.join(' AND ')}`;
    }

    sql += ` ORDER BY name ASC`;

    const result = await query(sql, params);

    if (result.rows.length === 0) {
      return res.json([]);
    }

    const datasetIds = result.rows.map(d => d.id);
    const tableConfigs = await query(`
      SELECT dataset_id, name, source_name, source_url
      FROM w4h.tables
      WHERE dataset_id = ANY($1)
    `, [datasetIds]);

    const datasets = [];
    const queryMapByCatalogProject = new Map();

    for (const d of result.rows) {
      const cp = d.dataset_project_id;
      if (!queryMapByCatalogProject.has(cp)) {
        queryMapByCatalogProject.set(cp, await getQueryMap(cp));
      }

      datasets.push(await processDataset(d, req.user, queryMapByCatalogProject.get(cp), tableConfigs.rows));
    }

    res.json(datasets);
  } catch (error) {
    console.error('listDatasets error:', error);
    res.status(500).json({ error: error.message || 'Failed to list datasets' });
  }
}

/**
 * What a **full** (destructive, truncate + reload) re-import would put at risk —
 * used by the Run import dialog to decide whether to warn before a full re-import.
 * Excludes the requesting project (linking one project and running against it is
 * not a "dependent"). Does not (yet) check derived datasets / model runs — there is
 * no lineage table for those (MVP D-05, M-02 are still Todo).
 */
async function getDatasetDependents(req, res) {
  if (!canAccessDatasets(req.user)) {
    return res.status(403).json({ error: 'Unauthorized' });
  }
  try {
    const datasetId = String(req.params.id || '').trim();
    const excludeProjectId = req.query.exclude_project_id
      ? String(req.query.exclude_project_id)
      : null;

    const projectsRes = await query(
      `SELECT p.id::text AS project_id, p.name
       FROM w4h.project_datasets pd
       JOIN w4h.projects p ON p.id = pd.project_id
       WHERE pd.dataset_id = $1
       ORDER BY p.name ASC`,
      [datasetId]
    );
    const otherProjects = projectsRes.rows.filter((r) => r.project_id !== excludeProjectId);

    const sharesRes = await query(
      `SELECT COUNT(*)::int AS n FROM w4h.dataset_shares WHERE dataset_id = $1`,
      [datasetId]
    );
    const shareCount = sharesRes.rows[0]?.n || 0;

    res.json({
      linkedProjectCount: projectsRes.rowCount,
      otherProjects: otherProjects.map((r) => ({ id: r.project_id, name: r.name })),
      shareCount,
      hasDependents: otherProjects.length > 0 || shareCount > 0,
    });
  } catch (e) {
    res.status(500).json({ error: e.message || 'Failed to check dataset dependents' });
  }
}

async function getDataset(req, res) {
  if (!canAccessDatasets(req.user)) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  const { id } = req.params;

  // Team catalog detail — Team admin / Super admin
  if (await canManageTeamAsync(req.user)) {
    try {
      const result = await query(`${CATALOG_SELECT} WHERE d.id = $1`, [id]);
      if (!result.rowCount) {
        return res.status(404).json({ error: 'Dataset not found' });
      }
      const row = result.rows[0];
      if (!isSuperAdmin(req.user.role)) {
        const teamIds = await getTeamIdsForUser(req.user.id);
        const owned = row.team_id && teamIds.includes(String(row.team_id));
        if (!owned) {
          const shared = await query(
            `SELECT 1 FROM w4h.dataset_shares
             WHERE dataset_id = $1 AND target_team_id = ANY($2::uuid[])
             LIMIT 1`,
            [id, teamIds]
          );
          if (!shared.rowCount) {
            if (!row.discoverable) {
              return res.status(403).json({ error: 'Unauthorized' });
            }
            // Discoverable listing: metadata only (D-23 / S-10)
            const [enriched] = await attachMetadataSummaries(await attachTableCounts([{
              ...row,
              project_names: '',
              projects: [],
              tables: [],
              source_kind: 'listed',
              source_label: row.team_name ? `Listed · ${row.team_name}` : 'Listed',
              table_count: 0,
            }]));
            return res.json(enriched);
          }
        }
      }
      let tables = [];
      try {
        const tablesRes = await query(
          `SELECT name, source_name, source_url
           FROM w4h.tables
           WHERE dataset_id = $1
           ORDER BY name ASC`,
          [id]
        );
        tables = tablesRes.rows;
      } catch {
        tables = [];
      }
      const [enriched] = await attachMetadataSummaries(await attachTableCounts([{ ...row, tables }]));
      return res.json(enriched);
    } catch (error) {
      return res.status(500).json({ error: error.message || 'Failed to load dataset' });
    }
  }

  const pc = await resolveDatasetProjectLinkColumn();
  if (isSuperAdmin(req.user.role)) {
    return res.status(403).json({ error: 'Super admin does not open trial dataset records' });
  }
  const result = await query(
    `SELECT d.id, d.${pc} AS dataset_project_id, d.name, d.description
     FROM w4h.datasets d
     WHERE d.id = $1 AND ${sqlDatasetAccessibleToUser('$2')}`,
    [id, req.user.id]
  );

  if (result.rows.length === 0) {
    return res.status(404).json({ error: 'Dataset not found' });
  }

  const dataset = result.rows[0];

  const tableConfigs = await query(`
    SELECT dataset_id, name, source_name, source_url
    FROM w4h.tables
    WHERE dataset_id = $1
  `, [dataset.id]);

  const QUERY_MAP = await getQueryMap(dataset.dataset_project_id);

  const processedDataset = await processDataset(dataset, req.user, QUERY_MAP, tableConfigs.rows);

  const catalogRes = await query(
    `SELECT device_type, sampling_rate, subject_count, date_start, date_end,
            irb_reference, origin, description, COALESCE(read_only, false) AS read_only
     FROM w4h.datasets WHERE id = $1`,
    [dataset.id]
  );
  const catalogRow = catalogRes.rows[0] || {};
  const [withMeta] = await attachMetadataSummaries([
    {
      ...catalogRow,
      id: dataset.id,
      name: dataset.name || catalogRow.name,
      description: dataset.description ?? catalogRow.description,
    },
  ]);
  processedDataset.metadata_summary = withMeta.metadata_summary;

  res.json(processedDataset);
}

const SHARE_SELECT = `
  SELECT ds.id::text AS id,
         ds.dataset_id,
         d.name AS dataset_name,
         d.team_id::text AS owner_team_id,
         ot.name AS owner_team_name,
         ds.target_team_id::text AS target_team_id,
         tt.name AS target_team_name,
         ds.target_project_id::text AS target_project_id,
         p.name AS target_project_name,
         ds.access_type,
         ds.shared_at,
         ds.shared_by,
         u.name AS shared_by_name,
         u.email AS shared_by_email
  FROM w4h.dataset_shares ds
  JOIN w4h.datasets d ON d.id = ds.dataset_id
  LEFT JOIN w4h.teams ot ON ot.id = d.team_id
  JOIN w4h.teams tt ON tt.id = ds.target_team_id
  LEFT JOIN w4h.projects p ON p.id = ds.target_project_id
  LEFT JOIN w4h.users u ON u.id = ds.shared_by
`;

async function assertCanManageDatasetShares(user, datasetId) {
  const ds = await query(
    `SELECT id, name, team_id::text AS team_id FROM w4h.datasets WHERE id = $1`,
    [datasetId]
  );
  if (!ds.rowCount) return { error: 'Dataset not found', status: 404 };
  const row = ds.rows[0];
  if (!row.team_id) {
    return { error: 'Dataset has no owning team; assign a team before sharing', status: 400 };
  }
  if (!(await isTeamAdminOnTeam(user.id, row.team_id, user.role))) {
    return { error: 'Unauthorized', status: 403 };
  }
  return { dataset: row };
}

async function listShareTargets(req, res) {
  try {
    if (!(await canManageTeamAsync(req.user))) {
      return res.status(403).json({ error: 'Unauthorized' });
    }
    const teamsRes = await query(
      `SELECT id::text AS id, name, slug
       FROM w4h.teams
       WHERE status = 'active'
       ORDER BY name ASC`
    );
    let projects = [];
    const teamId = req.query.team_id ? String(req.query.team_id).trim() : '';
    if (teamId) {
      const projectsRes = await query(
        `SELECT id::text AS id, name, slug, status
         FROM w4h.projects
         WHERE team_id = $1::uuid AND status = 'active'
         ORDER BY name ASC`,
        [teamId]
      );
      projects = projectsRes.rows;
    }
    res.json({ teams: teamsRes.rows, projects });
  } catch (error) {
    console.error('listShareTargets error:', error);
    res.status(500).json({ error: error.message || 'Failed to list share targets' });
  }
}

async function listDatasetShares(req, res) {
  try {
    if (!(await canManageTeamAsync(req.user))) {
      return res.status(403).json({ error: 'Unauthorized' });
    }

    let outboundSql = `${SHARE_SELECT}`;
    let inboundSql = `${SHARE_SELECT}`;
    const outboundParams = [];
    const inboundParams = [];

    if (!isSuperAdmin(req.user.role)) {
      const teamIds = await getTeamIdsForUser(req.user.id);
      if (!teamIds.length) {
        return res.json({ outbound: [], inbound: [] });
      }
      outboundParams.push(teamIds);
      outboundSql += ` WHERE d.team_id = ANY($1::uuid[])`;
      inboundParams.push(teamIds);
      inboundSql += ` WHERE ds.target_team_id = ANY($1::uuid[])`;
    }

    outboundSql += ` ORDER BY ds.shared_at DESC`;
    inboundSql += ` ORDER BY ds.shared_at DESC`;

    const [outbound, inbound] = await Promise.all([
      query(outboundSql, outboundParams),
      query(inboundSql, inboundParams),
    ]);

    res.json({
      outbound: outbound.rows.map((r) => ({ ...r, direction: 'outbound' })),
      inbound: inbound.rows.map((r) => ({ ...r, direction: 'inbound' })),
    });
  } catch (error) {
    console.error('listDatasetShares error:', error);
    res.status(500).json({ error: error.message || 'Failed to list shares' });
  }
}

async function listSharesForDataset(req, res) {
  try {
    const check = await assertCanManageDatasetShares(req.user, req.params.id);
    if (check.error) return res.status(check.status).json({ error: check.error });

    const result = await query(
      `${SHARE_SELECT} WHERE ds.dataset_id = $1 ORDER BY ds.shared_at DESC`,
      [req.params.id]
    );
    res.json(result.rows.map((r) => ({ ...r, direction: 'outbound' })));
  } catch (error) {
    console.error('listSharesForDataset error:', error);
    res.status(500).json({ error: error.message || 'Failed to list shares' });
  }
}

async function createDatasetShare(req, res) {
  try {
    const check = await assertCanManageDatasetShares(req.user, req.params.id);
    if (check.error) return res.status(check.status).json({ error: check.error });
    const dataset = check.dataset;

    const targetTeamId = req.body?.target_team_id == null
      ? ''
      : String(req.body.target_team_id).trim();
    if (!targetTeamId) {
      return res.status(400).json({ error: 'target_team_id is required' });
    }
    if (targetTeamId === dataset.team_id) {
      return res.status(400).json({ error: 'Cannot share a dataset with its owning team' });
    }

    const teamRes = await query(
      `SELECT id::text AS id, name, status FROM w4h.teams WHERE id = $1::uuid`,
      [targetTeamId]
    );
    if (!teamRes.rowCount) {
      return res.status(400).json({ error: 'Unknown target_team_id' });
    }
    if (teamRes.rows[0].status !== 'active') {
      return res.status(400).json({ error: 'Target team is not active' });
    }

    let targetProjectId = null;
    if (req.body?.target_project_id != null && String(req.body.target_project_id).trim() !== '') {
      targetProjectId = String(req.body.target_project_id).trim();
      const proj = await query(
        `SELECT id::text AS id, team_id::text AS team_id, name
         FROM w4h.projects WHERE id = $1::uuid`,
        [targetProjectId]
      );
      if (!proj.rowCount) {
        return res.status(400).json({ error: 'Unknown target_project_id' });
      }
      if (proj.rows[0].team_id !== targetTeamId) {
        return res.status(400).json({ error: 'Target project must belong to the target team' });
      }
    }

    const accessType = req.body?.access_type === 'read_write' ? 'read_write' : 'read';

    const dup = await query(
      `SELECT id FROM w4h.dataset_shares
       WHERE dataset_id = $1
         AND target_team_id = $2::uuid
         AND (
           ($3::uuid IS NULL AND target_project_id IS NULL)
           OR target_project_id = $3::uuid
         )
       LIMIT 1`,
      [dataset.id, targetTeamId, targetProjectId]
    );
    if (dup.rowCount) {
      return res.status(409).json({ error: 'Share already exists for this team/project' });
    }

    const inserted = await query(
      `INSERT INTO w4h.dataset_shares
         (dataset_id, target_team_id, target_project_id, access_type, shared_by)
       VALUES ($1, $2::uuid, $3::uuid, $4, $5)
       RETURNING id::text AS id`,
      [dataset.id, targetTeamId, targetProjectId, accessType, req.user.id]
    );

    writeAuditLogSafe(req, {
      action: 'sharing.create',
      resourceType: 'dataset_share',
      resourceId: inserted.rows[0].id,
      details: {
        dataset_id: dataset.id,
        target_team_id: targetTeamId,
        target_project_id: targetProjectId,
        access_type: accessType,
      },
    });

    const row = await query(
      `${SHARE_SELECT} WHERE ds.id = $1::uuid`,
      [inserted.rows[0].id]
    );
    res.status(201).json({ ...row.rows[0], direction: 'outbound' });
  } catch (error) {
    console.error('createDatasetShare error:', error);
    res.status(500).json({ error: error.message || 'Failed to create share' });
  }
}

async function revokeDatasetShare(req, res) {
  try {
    const shareId = req.params.shareId;
    const shareRes = await query(
      `SELECT ds.id::text AS id,
              ds.dataset_id,
              d.team_id::text AS owner_team_id,
              ds.target_team_id::text AS target_team_id
       FROM w4h.dataset_shares ds
       JOIN w4h.datasets d ON d.id = ds.dataset_id
       WHERE ds.id = $1::uuid`,
      [shareId]
    );
    if (!shareRes.rowCount) {
      return res.status(404).json({ error: 'Share not found' });
    }
    const share = shareRes.rows[0];

    const canOwner = share.owner_team_id
      && await isTeamAdminOnTeam(req.user.id, share.owner_team_id, req.user.role);
    const canRecipient = await isTeamAdminOnTeam(req.user.id, share.target_team_id, req.user.role);
    if (!canOwner && !canRecipient) {
      return res.status(403).json({ error: 'Unauthorized' });
    }

    await query(`DELETE FROM w4h.dataset_shares WHERE id = $1::uuid`, [shareId]);

    writeAuditLogSafe(req, {
      action: 'sharing.revoke',
      resourceType: 'dataset_share',
      resourceId: shareId,
      details: {
        dataset_id: share.dataset_id,
        target_team_id: share.target_team_id,
      },
    });

    res.json({ ok: true });
  } catch (error) {
    console.error('revokeDatasetShare error:', error);
    res.status(500).json({ error: error.message || 'Failed to revoke share' });
  }
}

/** GET /datasets/:id/analytics/cohort — subject roster + comparable signals (ADR-021). */
async function getDatasetCohortAnalytics(req, res) {
  try {
    const datasetId = String(req.params.id || '').trim();
    const access = await assertCanViewDataset(req.user, datasetId);
    if (!access.ok) {
      return res.status(access.status || 403).json({ error: access.error || 'Unauthorized' });
    }
    const payload = await getCohortAnalyticsSetup(datasetId);
    return res.json(payload);
  } catch (error) {
    console.error('getDatasetCohortAnalytics error:', error);
    return res.status(500).json({ error: error.message || 'Failed to load cohort analytics' });
  }
}

/** POST /datasets/:id/analytics/cohort — selected vs control aggregates (ADR-021). */
async function postDatasetCohortAnalytics(req, res) {
  try {
    const datasetId = String(req.params.id || '').trim();
    const access = await assertCanViewDataset(req.user, datasetId);
    if (!access.ok) {
      return res.status(access.status || 403).json({ error: access.error || 'Unauthorized' });
    }
    const result = await runCohortComparison(datasetId, req.body || {});
    if (!result.ok) {
      return res.status(result.status || 400).json({ error: result.error || 'Cohort comparison failed' });
    }
    writeAuditLogSafe(req, {
      action: 'dataset.query',
      resourceType: 'dataset',
      resourceId: datasetId,
      details: {
        analytics: 'cohort',
        selected_count: result.selected_ids?.length ?? 0,
        control_count: result.control_ids?.length ?? 0,
        control_mode: result.control_mode || null,
        signals: Object.keys(result.signals || {}),
      },
    });
    return res.json(result);
  } catch (error) {
    console.error('postDatasetCohortAnalytics error:', error);
    return res.status(500).json({ error: error.message || 'Failed to run cohort comparison' });
  }
}

/** GET /datasets/:id/analytics/catalog — what the chart component can show (ADR-029). */
async function getDatasetVizCatalog(req, res) {
  try {
    const datasetId = String(req.params.id || '').trim();
    const access = await assertCanViewDataset(req.user, datasetId);
    if (!access.ok) {
      return res.status(access.status || 403).json({ error: access.error || 'Unauthorized' });
    }
    return res.json(await describeViz(datasetId));
  } catch (error) {
    console.error('getDatasetVizCatalog error:', error);
    return res.status(500).json({ error: error.message || 'Failed to describe chartable signals' });
  }
}

/** POST /datasets/:id/analytics/series — aggregated data for one chart (ADR-029 §3). */
async function postDatasetViz(req, res) {
  try {
    const datasetId = String(req.params.id || '').trim();
    const access = await assertCanViewDataset(req.user, datasetId);
    if (!access.ok) {
      return res.status(access.status || 403).json({ error: access.error || 'Unauthorized' });
    }
    const body = req.body || {};
    const result = await runViz(datasetId, body);
    if (!result.ok) {
      return res.status(result.status || 400).json({ error: result.error || 'Chart query failed' });
    }
    writeAuditLogSafe(req, {
      action: 'dataset.query',
      resourceType: 'dataset',
      resourceId: datasetId,
      details: {
        analytics: 'viz',
        kind: result.kind,
        signal: body.signal || body.x?.signal || null,
        subjects: Array.isArray(body.subjects) ? body.subjects.length : 0,
      },
    });
    return res.json(result);
  } catch (error) {
    console.error('postDatasetViz error:', error);
    return res.status(500).json({ error: error.message || 'Failed to run chart query' });
  }
}

/** GET /datasets/:id/signals — mapped signals from w4h.tables (ADR-023). */
async function getDatasetSignals(req, res) {
  try {
    const datasetId = String(req.params.id || '').trim();
    const access = await assertCanViewDataset(req.user, datasetId);
    if (!access.ok) {
      return res.status(access.status || 403).json({ error: access.error || 'Unauthorized' });
    }
    const payload = await listDatasetSignals(datasetId);
    return res.json(payload);
  } catch (error) {
    console.error('getDatasetSignals error:', error);
    return res.status(500).json({ error: error.message || 'Failed to list signals' });
  }
}

/** POST /datasets/:id/contributions/:slug/sql — bind + execute a kind:sql contribution (ADR-023). */
async function runDatasetSqlContribution(req, res) {
  try {
    const datasetId = String(req.params.id || '').trim();
    const slug = String(req.params.slug || '').trim();
    const result = await runSqlContribution(req.user, datasetId, slug, req.body || {});
    if (!result.ok) {
      return res.status(result.status || 400).json({ error: result.error || 'SQL contribution failed' });
    }
    writeAuditLogSafe(req, {
      action: result.write_back ? 'dataset.derive' : 'dataset.query',
      resourceType: 'dataset',
      resourceId: datasetId,
      details: {
        contribution_slug: slug,
        write_back: result.write_back,
        interval_minutes: result.interval_minutes,
        physical_table: result.physical_table || null,
        inserted: result.inserted ?? null,
        row_count: result.row_count ?? null,
      },
    });
    return res.json(result);
  } catch (error) {
    console.error('runDatasetSqlContribution error:', error);
    return res.status(500).json({ error: error.message || 'Failed to run SQL contribution' });
  }
}

/**
 * POST /datasets/:id/summary/refresh — recompute the cached cohort summaries: `mets_summary` and
 * `wear_compliance` (ADR-019). Body (optional): `wear_protocol` { daily_band_wear_percentage,
 * overall_band_wear_percentage, daily_band_wear_hours: [["08:00","20:00"]] }.
 */
async function refreshDatasetSummary(req, res) {
  try {
    const datasetId = String(req.params.id || '').trim();
    const access = await assertCanIngestDataset(req.user, datasetId);
    if (!access.ok) {
      return res.status(access.status || 403).json({ error: access.error || 'Unauthorized' });
    }
    const result = await refreshDatasetSummaryData(datasetId, { wearProtocol: req.body?.wear_protocol || null });
    if (!result.ok) {
      return res.status(result.status || 400).json({ error: result.error });
    }
    writeAuditLogSafe(req, {
      action: 'dataset.summary.refresh',
      resourceType: 'dataset',
      resourceId: datasetId,
      details: {
        mets_available: Boolean(result.mets?.available),
        mets_source: result.mets?.source || null,
        subjects_with_data: result.mets?.subjects_with_data ?? 0,
        wear_available: Boolean(result.wear?.available),
        wear_subjects: result.wear?.subjects_total ?? 0,
      },
    });
    return res.json({
      ok: true,
      summary: result.summary,
      summary_updated_at: result.summary_updated_at,
    });
  } catch (error) {
    console.error('refreshDatasetSummary error:', error);
    return res.status(500).json({ error: error.message || 'Failed to refresh dataset summary' });
  }
}

/** POST /datasets/:id/import/csv-batch — sample importer / team-admin CSV ingest */
async function importCsvBatchHandler(req, res) {
  try {
    const result = await importCsvBatch(req.user, req.params.id, req.body || {});
    if (!result.ok) {
      return res.status(result.status || 400).json({ error: result.error || 'Import failed' });
    }
    writeAuditLogSafe(req, {
      action: 'dataset.import.csv',
      resourceType: 'dataset',
      resourceId: req.params.id,
      details: {
        signal: req.body?.signal,
        mode: req.body?.mode,
        inserted: result.inserted,
        skipped: result.skipped,
      },
    });
    res.json({
      success: true,
      inserted: result.inserted,
      skipped: result.skipped,
      physical_schema: result.physical_schema,
      physical_table: result.physical_table,
    });
  } catch (error) {
    console.error('importCsvBatch error:', error);
    res.status(500).json({ error: error.message || 'Import failed' });
  }
}

export default router;