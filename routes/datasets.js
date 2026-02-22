import express from 'express';
const router = express.Router();

import { authenticate } from '../auth.js';
import { USER_ROLES } from '../lib/utils/user.js';
import { query } from '../lib/utils/db.js';

router.get('/studies', authenticate, listStudies);
router.get('/tableinfo', authenticate, tableInfo);
router.get('/', authenticate, listDatasets);
router.get('/:id', authenticate, getDataset);

const STATUS_TABLE_NAME = 'w4h.status_log';

async function listStudies(req, res) {
  if (req.user.role !== USER_ROLES.ADMIN) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  const sql = `SELECT id, name FROM w4h.studies ORDER BY name ASC`;
  const result = await query(sql);
  res.json(result.rows);
}

async function tableInfo(req, res) {
  if (req.user.role !== USER_ROLES.ADMIN && req.user.role !== USER_ROLES.EDITOR) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  const city_id = req.user.role === USER_ROLES.ADMIN
    ? (req.query.city_id || req.user.city_id)
    : req.user.city_id;

  let columnIds = [];
  let columns = [];
  let values = [];
  let partitions = [];

  try {
    const queryMap = await getQueryMap(city_id);
    const dataset = queryMap.get(req.query.dataset_id);

    if (!dataset) {
      return res.status(400).json({ error: 'Invalid dataset_id' });
    }

    const table = dataset.datasetInfo.tables[0];
    if (!table) {
      return res.status(400).json({ error: 'Missing table for dataset' });
    }

    // Parse schema + table name
    const [schemaName, tableName] = table.includes('.')
      ? table.split('.')
      : ['public', table];

    // 1️⃣ Get column metadata (names + types)
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
      // query partitions
      // Assume 'schemaName' holds the city's schema (e.g., 'city')
      const partitionTable = dataset.datasetInfo.partitionTableInfo.table;
      const partitionSql = `
  SELECT id, name
  FROM ${partitionTable}
  ORDER BY name;
`;
      const partitionResult = await query(partitionSql);
      partitions = partitionResult.rows;
    }

    // 2️⃣ Pull one small sample of data
    const sampleSql = `
      SELECT *
      FROM ${table}
      TABLESAMPLE SYSTEM (1)
      LIMIT 200;
    `;
    const sampleResult = await query(sampleSql);
    const sampleRows = sampleResult.rows || [];

    // 3️⃣ Compute distinct values from the sample
    for (const col of textColumns) {
      const sampleValues = sampleRows
        .map(r => r[col])
        .filter(v => v != null && v !== '');

      if (sampleValues.length === 0) continue;

      const unique = [...new Set(sampleValues)];

      if (unique.length >= 190) { // 95%
        if (process.env.W4H_DEBUG) console.log(`⚙️ Skipping noisy column: ${col}`);
        continue;
      }

      values[col] = unique; //.slice(0, 200);      // keep up to 200 unique values
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

// Helper function to process a single dataset
async function processDataset(dataset, user, QUERY_MAP, tableConfigs) {
  const datasetId = dataset.id;
  // Add mixins to dataset
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
      return aHasColon ? 1 : -1; // mixins WITHOUT ':' come first
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

  // Query status logs for tables in this dataset
  const allUniqueTableNames = new Set();
  processedDataset.tables?.forEach(table => {
    allUniqueTableNames.add(table);
    allUniqueTableNames.add(table + '_staging');   // add staging variant
  });

  const datasetTables = tableConfigs
    .filter(d => d.dataset_id === datasetId);

  datasetTables.forEach(d => {
    allUniqueTableNames.add(d.name);
    allUniqueTableNames.add(d.name + '_staging'); // add staging variant
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
    // Assign status and metabase_url to each table
    processedDataset.tables = processedDataset.tables?.map((table, index) => ({
      name: table,
      status: statusMap.get(table)
        || statusMap.get(table + '_staging')
        || null,
      metabase_url: datasetInfo.metabase_urls?.[index] || null,
      source_name: datasetInfo.sources?.[index]?.name || null,
      source_url: datasetInfo.sources?.[index]?.url || null
    })) || [];
    // Remove redundant top-level metabase_urls
    delete processedDataset.metabase_urls;
    delete processedDataset.sources;
  } else {
    processedDataset.tables = datasetTables;

    // Assign status to each table
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
  if (req.user.role !== USER_ROLES.ADMIN && req.user.role !== USER_ROLES.EDITOR) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  const cityId = req.user.role === USER_ROLES.ADMIN
    ? (req.query.city_id || req.user.city_id)
    : req.user.city_id;

  const QUERY_MAP = await getQueryMap(cityId);

  // 2. Fetch Datasets List
  let sql = `
    SELECT id, name, description, hidden
    FROM w4h.datasets
    WHERE city_id = $1
  `;
  const params = [cityId];

  if (req.user.role !== USER_ROLES.ADMIN) {
    sql += ` AND hidden = false`;
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

  // Process datasets
  const datasets = [];
  for (const d of result.rows) {
    datasets.push(await processDataset(d, req.user, QUERY_MAP, tableConfigs.rows));
  }

  res.json(datasets);
}

async function getDataset(req, res) {
  if (req.user.role !== USER_ROLES.ADMIN && req.user.role !== USER_ROLES.EDITOR) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  const { id } = req.params;
  const cityId = req.user.role === USER_ROLES.ADMIN
    ? (req.query.city_id || req.user.city_id)
    : req.user.city_id;

  // 1. Fetch the core dataset record
  const result = await query(`SELECT id, name, description, hidden
    FROM w4h.datasets
    WHERE id = $1 AND city_id = $2 ${req.user.role === USER_ROLES.ADMIN ? '' : 'AND hidden = false'}`,
    [id, cityId]);

  if (result.rows.length === 0) {
    return res.status(404).json({ error: 'Dataset not found' });
  }

  const dataset = result.rows[0];

  const tableConfigs = await query(`
    SELECT dataset_id, name, metabase_url, source_name, source_url
    FROM w4h.tables
    WHERE dataset_id = $1
  `, [dataset.id]);

  // 2. Load the hardcoded map (Backward Compatibility Base)
  const QUERY_MAP = await getQueryMap(cityId);

  const processedDataset = await processDataset(dataset, req.user, QUERY_MAP, tableConfigs.rows);

  res.json(processedDataset);
}

export default router;