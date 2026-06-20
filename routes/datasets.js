import express from 'express';
const router = express.Router();

import { authenticateJwtOrApiKey } from '../auth.js';
import { USER_ROLES } from '../lib/utils/user.js';
import { query } from '../db.js';
import { resolveDatasetProjectLinkColumn } from '../lib/utils/datasetProjectLinkColumn.js';

router.get('/projects', authenticateJwtOrApiKey, listProjects);
router.get('/projects-admin', authenticateJwtOrApiKey, listProjectsAdmin);
router.get('/projects-admin/:id', authenticateJwtOrApiKey, getProjectAdmin);
router.post('/projects-admin', authenticateJwtOrApiKey, addProjectAdmin);
router.put('/projects-admin/:id', authenticateJwtOrApiKey, setProjectAdmin);
router.get('/tableinfo', authenticateJwtOrApiKey, tableInfo);
router.get('/', authenticateJwtOrApiKey, listDatasets);
router.get('/:id', authenticateJwtOrApiKey, getDataset);

const STATUS_TABLE_NAME = 'w4h.status_log';
const DATASET_ALLOWED_ROLES = new Set(['admin', 'researcher', 'clinician']);

function canAccessDatasets(user) {
  return !!user && DATASET_ALLOWED_ROLES.has(user.role);
}

function isAdmin(user) {
  return !!user && user.role === USER_ROLES.ADMIN;
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

  if (req.user.role !== USER_ROLES.ADMIN) {
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
    const datasetAccessResult = req.user.role === USER_ROLES.ADMIN
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

  const pc = await resolveDatasetProjectLinkColumn();
  let sql = `
    SELECT d.id, d.${pc} AS dataset_project_id, d.name, d.description
    FROM w4h.datasets d
  `;
  const params = [];

  if (req.user.role !== USER_ROLES.ADMIN) {
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
    sql += params.length === 1 && req.user.role === USER_ROLES.ADMIN
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
  const pc = await resolveDatasetProjectLinkColumn();
  const result = req.user.role === USER_ROLES.ADMIN
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