import { query, withTransaction } from '../db.js';
import {
  grantDatasetSchemaToRole,
  quoteIdentifier,
  revokeDatasetSchemaFromRole,
} from './dataset-provision.js';

const PLATFORM_ROLE = 'w4h_platform';
const PROJECT_ROLE = /^project_[a-f0-9]{32}_role$/;

export function projectRoleForId(projectId) {
  const compact = String(projectId || '').toLowerCase().replaceAll('-', '');
  if (!/^[a-f0-9]{32}$/.test(compact)) {
    throw new Error('Invalid project id for role provisioning');
  }
  return `project_${compact}_role`;
}

export async function provisionProjectRole(projectId) {
  const id = String(projectId || '').trim();
  if (!id) throw new Error('project id is required');

  const existing = await query(
    `SELECT pg_role FROM w4h.projects WHERE id = $1::uuid`,
    [id]
  );
  if (!existing.rowCount) {
    throw new Error('Project not found');
  }

  let roleName = existing.rows[0].pg_role;
  if (roleName) {
    if (!PROJECT_ROLE.test(roleName)) {
      throw new Error('Project pg_role is not a valid identifier');
    }
  } else {
    roleName = projectRoleForId(id);
  }

  await withTransaction(async (client) => {
    await client.query(`SET LOCAL ROLE ${PLATFORM_ROLE}`);
    const exists = await client.query('SELECT 1 FROM pg_roles WHERE rolname = $1', [roleName]);
    if (!exists.rowCount) {
      await client.query(`CREATE ROLE ${quoteIdentifier(roleName)} NOLOGIN`);
    }
    await client.query('RESET ROLE');
    await client.query(
      `UPDATE w4h.projects
       SET pg_role = $2, updated_at = CURRENT_TIMESTAMP
       WHERE id = $1::uuid`,
      [id, roleName]
    );
  });

  return roleName;
}

export async function syncProjectDatasetGrants(projectId, previousDatasetIds, linkedDatasets) {
  const roleName = await provisionProjectRole(projectId);
  const previous = new Set((previousDatasetIds || []).map((id) => String(id)));
  const nextIds = new Set((linkedDatasets || []).map((row) => String(row.dataset_id)));

  const removed = [...previous].filter((id) => !nextIds.has(id));
  if (removed.length) {
    const schemas = await query(
      `SELECT schema_name
       FROM w4h.dataset_registry
       WHERE dataset_id = ANY($1::text[]) AND status = 'active'`,
      [removed]
    );
    for (const row of schemas.rows) {
      await revokeDatasetSchemaFromRole(row.schema_name, roleName);
    }
  }

  for (const link of linkedDatasets || []) {
    const registry = await query(
      `SELECT schema_name, status
       FROM w4h.dataset_registry
       WHERE dataset_id = $1`,
      [String(link.dataset_id)]
    );
    if (!registry.rowCount || registry.rows[0].status !== 'active') continue;
    await grantDatasetSchemaToRole(
      registry.rows[0].schema_name,
      roleName,
      link.access_type === 'read_write' ? 'read_write' : 'read'
    );
  }

  return roleName;
}
