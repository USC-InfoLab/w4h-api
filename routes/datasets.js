import express from 'express';
const router = express.Router();

import { authenticateJwtOrApiKey } from '../auth.js';
import { isSuperAdmin, isTeamAdmin, canAccessAdminUi } from '../lib/utils/user.js';
import { query } from '../db.js';
import { resolveDatasetProjectLinkColumn } from '../lib/utils/datasetProjectLinkColumn.js';

router.get('/projects', authenticateJwtOrApiKey, listProjects);
router.get('/projects-admin', authenticateJwtOrApiKey, listProjectsAdmin);
router.get('/projects-admin/:id', authenticateJwtOrApiKey, getProjectAdmin);
router.post('/projects-admin', authenticateJwtOrApiKey, addProjectAdmin);
router.put('/projects-admin/:id', authenticateJwtOrApiKey, setProjectAdmin);
router.get('/tableinfo', authenticateJwtOrApiKey, tableInfo);
router.get('/', authenticateJwtOrApiKey, listDatasets);
router.post('/', authenticateJwtOrApiKey, createDataset);
router.patch('/:id', authenticateJwtOrApiKey, updateDataset);
router.get('/:id', authenticateJwtOrApiKey, getDataset);

const STATUS_TABLE_NAME = 'w4h.status_log';
const DATASET_ALLOWED_ROLES = new Set([
  'admin', 'super_admin', 'team_admin', 'project_user', 'researcher', 'clinician'
]);

function canAccessDatasets(user) {
  return !!user && DATASET_ALLOWED_ROLES.has(user.role);
}

function canManageDatasets(user) {
  return !!user && canAccessAdminUi(user.role);
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
  if (isSuperAdmin(user.role)) return true;
  return (await getTeamIdsForUser(user.id)).includes(String(teamId));
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

const CATALOG_SELECT = `
  SELECT d.id, d.name, d.description, d.team_id, d.project_id,
         d.device_type, d.sampling_rate, d.subject_count,
         d.date_start, d.date_end, d.irb_reference,
         d.created_by, d.created_at, d.updated_at,
         t.name AS team_name, t.slug AS team_slug,
         'local'::text AS source_kind,
         'Local'::text AS source_label
  FROM w4h.datasets d
  LEFT JOIN w4h.teams t ON t.id = d.team_id
`;

async function listDatasetsCatalog(req, res) {
  try {
    if (isSuperAdmin(req.user.role)) {
      const result = await query(`
        ${CATALOG_SELECT}
        ORDER BY d.name ASC NULLS LAST, d.id ASC
      `);
      return res.json(
        result.rows.map((row) => ({
          ...row,
          source_kind: 'local',
          source_label: 'Local'
        }))
      );
    }

    const teamIds = await getTeamIdsForUser(req.user.id);
    if (!teamIds.length) return res.json([]);

    const owned = await query(
      `
      SELECT d.id, d.name, d.description, d.team_id, d.project_id,
             d.device_type, d.sampling_rate, d.subject_count,
             d.date_start, d.date_end, d.irb_reference,
             d.created_by, d.created_at, d.updated_at,
             t.name AS team_name, t.slug AS team_slug,
             'local'::text AS source_kind,
             'Local'::text AS source_label
      FROM w4h.datasets d
      LEFT JOIN w4h.teams t ON t.id = d.team_id
      WHERE d.team_id = ANY($1::uuid[])
      ORDER BY d.name ASC NULLS LAST, d.id ASC
      `,
      [teamIds]
    );

    const shared = await query(
      `
      SELECT DISTINCT ON (d.id)
             d.id, d.name, d.description, d.team_id, d.project_id,
             d.device_type, d.sampling_rate, d.subject_count,
             d.date_start, d.date_end, d.irb_reference,
             d.created_by, d.created_at, d.updated_at,
             ot.name AS team_name, ot.slug AS team_slug,
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
      ORDER BY d.id, d.name ASC NULLS LAST
      `,
      [teamIds]
    );

    const byId = new Map();
    for (const row of owned.rows) byId.set(row.id, row);
    for (const row of shared.rows) {
      if (!byId.has(row.id)) byId.set(row.id, row);
    }
    res.json(
      [...byId.values()].sort((a, b) =>
        String(a.name || a.id).localeCompare(String(b.name || b.id), undefined, { sensitivity: 'base' })
      )
    );
  } catch (error) {
    res.status(500).json({ error: error.message || 'Failed to list datasets' });
  }
}

async function createDataset(req, res) {
  if (!canManageDatasets(req.user)) {
    return res.status(403).json({ error: 'Unauthorized' });
  }
  try {
    const name = String(req.body?.name || '').trim();
    if (!name) return res.status(400).json({ error: 'name is required' });

    let teamId = req.body?.team_id == null ? null : String(req.body.team_id).trim();
    if (!teamId && isTeamAdmin(req.user.role) && !isSuperAdmin(req.user.role)) {
      const myTeams = await getTeamIdsForUser(req.user.id);
      if (myTeams.length === 1) teamId = myTeams[0];
    }
    if (!teamId) return res.status(400).json({ error: 'team_id is required' });

    const teamCheck = await query(`SELECT id FROM w4h.teams WHERE id = $1::uuid`, [teamId]);
    if (!teamCheck.rowCount) return res.status(400).json({ error: 'Unknown team_id' });
    if (!(await canAccessTeam(req.user, teamId))) {
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

    if (subjectCount != null && !Number.isFinite(subjectCount)) {
      return res.status(400).json({ error: 'subject_count must be a number' });
    }

    const inserted = await query(
      `INSERT INTO w4h.datasets
         (id, name, description, team_id, device_type, sampling_rate, subject_count,
          date_start, date_end, irb_reference, created_by)
       VALUES ($1, $2, $3, $4::uuid, $5, $6, $7, $8, $9, $10, $11)
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
        req.user.id ?? null
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

async function updateDataset(req, res) {
  if (!canManageDatasets(req.user)) {
    return res.status(403).json({ error: 'Unauthorized' });
  }
  try {
    const datasetId = String(req.params.id || '').trim();
    const existing = await query(
      `SELECT id, team_id FROM w4h.datasets WHERE id = $1`,
      [datasetId]
    );
    if (!existing.rowCount) return res.status(404).json({ error: 'Dataset not found' });

    const currentTeamId = existing.rows[0].team_id;
    if (currentTeamId && !(await canAccessTeam(req.user, currentTeamId))) {
      return res.status(403).json({ error: 'Unauthorized' });
    }
    if (!currentTeamId && !isSuperAdmin(req.user.role)) {
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

    if (Object.prototype.hasOwnProperty.call(req.body || {}, 'team_id')) {
      if (!isSuperAdmin(req.user.role)) {
        return res.status(403).json({ error: 'Only Super admins can change the owning team' });
      }
      const teamId = req.body.team_id == null || req.body.team_id === ''
        ? null
        : String(req.body.team_id).trim();
      if (teamId) {
        const teamCheck = await query(`SELECT id FROM w4h.teams WHERE id = $1::uuid`, [teamId]);
        if (!teamCheck.rowCount) return res.status(400).json({ error: 'Unknown team_id' });
      }
      params.push(teamId);
      fields.push(`team_id = $${params.length}::uuid`);
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
    const pc = await resolveDatasetProjectLinkColumn();
    const result = await query(
      `SELECT
         s.id,
         s.name,
         s.description,
         COUNT(DISTINCT d.id)::int AS dataset_count,
         COUNT(DISTINCT us.user_id)::int AS user_count
       FROM w4h.projects s
       LEFT JOIN w4h.datasets d ON d.${pc} = s.id
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

  let sql = `
    SELECT id, name
    FROM w4h.projects
  `;
  const params = [];

  if (!isSuperAdmin(req.user.role)) {
    sql += `
      WHERE EXISTS (
        SELECT 1
        FROM w4h.user_projects us
        WHERE us.project_id = w4h.projects.id
          AND us.user_id = $1
      )
    `;
    params.push(req.user.id);
  }

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
    const datasetAccessResult = isSuperAdmin(req.user.role)
      ? await query(
        `SELECT id, ${pc} AS dataset_project_id FROM w4h.datasets WHERE id = $1`,
        [req.query.dataset_id]
      )
      : await query(
        `SELECT d.id, d.${pc} AS dataset_project_id
         FROM w4h.datasets d
         JOIN w4h.user_projects us
           ON us.project_id = d.${pc}
         WHERE d.id = $1 AND us.user_id = $2`,
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
      metabase_url: datasetInfo.metabase_urls?.[index] || null,
      source_name: datasetInfo.sources?.[index]?.name || null,
      source_url: datasetInfo.sources?.[index]?.url || null
    })) || [];
    delete processedDataset.metabase_urls;
    delete processedDataset.sources;
  } else {
    processedDataset.tables = datasetTables;

    processedDataset.tables = processedDataset.tables.map(tableInfo => ({
      name: tableInfo.name,
      metabase_url: tableInfo.metabase_url || null,
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
  if (!canAccessDatasets(req.user)) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  // Admin catalog (team-scoped) — used by /admin/datasets
  if (canAccessAdminUi(req.user.role) || req.query.catalog === '1') {
    return listDatasetsCatalog(req, res);
  }

  const pc = await resolveDatasetProjectLinkColumn();
  let sql = `
    SELECT d.id, d.${pc} AS dataset_project_id, d.name, d.description
    FROM w4h.datasets d
  `;
  const params = [];

  if (!isSuperAdmin(req.user.role)) {
    sql += `
      JOIN w4h.user_projects us
        ON us.project_id = d.${pc}
    `;
    params.push(req.user.id);
    sql += ` WHERE us.user_id = $${params.length}`;
  }

  const filterProjectId = req.query.project_id;
  if (filterProjectId) {
    params.push(filterProjectId);
    sql += params.length === 1 && isSuperAdmin(req.user.role)
      ? ` WHERE d.${pc} = $${params.length}`
      : ` AND d.${pc} = $${params.length}`;
  }

  sql += ` ORDER BY name ASC`;

  const result = await query(sql, params);

  if (result.rows.length === 0) {
    return res.json([]);
  }

  const datasetIds = result.rows.map(d => d.id);
  const tableConfigs = await query(`
    SELECT dataset_id, name, metabase_url, source_name, source_url
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
}

async function getDataset(req, res) {
  if (!canAccessDatasets(req.user)) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  const { id } = req.params;

  // Admin catalog detail
  if (canAccessAdminUi(req.user.role) || req.query.catalog === '1') {
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
            return res.status(403).json({ error: 'Unauthorized' });
          }
        }
      }
      let tables = [];
      try {
        const tablesRes = await query(
          `SELECT name, metabase_url, source_name, source_url
           FROM w4h.tables
           WHERE dataset_id = $1
           ORDER BY name ASC`,
          [id]
        );
        tables = tablesRes.rows;
      } catch {
        tables = [];
      }
      return res.json({ ...row, tables });
    } catch (error) {
      return res.status(500).json({ error: error.message || 'Failed to load dataset' });
    }
  }

  const pc = await resolveDatasetProjectLinkColumn();
  const result = isSuperAdmin(req.user.role)
    ? await query(
      `SELECT id, ${pc} AS dataset_project_id, name, description
       FROM w4h.datasets
       WHERE id = $1`,
      [id]
    )
    : await query(
      `SELECT d.id, d.${pc} AS dataset_project_id, d.name, d.description
       FROM w4h.datasets d
       JOIN w4h.user_projects us
         ON us.project_id = d.${pc}
       WHERE d.id = $1 AND us.user_id = $2`,
      [id, req.user.id]
    );

  if (result.rows.length === 0) {
    return res.status(404).json({ error: 'Dataset not found' });
  }

  const dataset = result.rows[0];

  const tableConfigs = await query(`
    SELECT dataset_id, name, metabase_url, source_name, source_url
    FROM w4h.tables
    WHERE dataset_id = $1
  `, [dataset.id]);

  const QUERY_MAP = await getQueryMap(dataset.dataset_project_id);

  const processedDataset = await processDataset(dataset, req.user, QUERY_MAP, tableConfigs.rows);

  res.json(processedDataset);
}

export default router;