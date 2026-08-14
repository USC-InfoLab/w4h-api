import { randomUUID } from 'node:crypto';
import fs from 'fs-extra';
import path from 'path';
import { fileURLToPath } from 'url';
import { query, withTransaction } from '../db.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DATASET_SCHEMA_PATH = path.join(__dirname, '../data/dataset_schema.sql');
const PLATFORM_ROLE = 'w4h_platform';
export const DATASET_SCHEMA_VERSION = 4;
const DATASET_SCHEMA_NAME = /^ds_[a-f0-9]{32}$/;
const ROLE_NAME = /^[a-z_][a-z0-9_]*$/;

export function datasetSchemaNameFromUuid(id) {
  const compact = String(id || '').toLowerCase().replaceAll('-', '');
  if (!/^[a-f0-9]{32}$/.test(compact)) {
    throw new Error('Invalid dataset schema uuid');
  }
  return `ds_${compact}`;
}

export function teamAdminRoleForTeam(teamId) {
  const compact = String(teamId || '').toLowerCase().replaceAll('-', '');
  if (!/^[a-f0-9]{32}$/.test(compact)) {
    throw new Error('Invalid team id');
  }
  return `team_${compact}_admin`;
}

export function quoteIdentifier(identifier) {
  if (!ROLE_NAME.test(identifier)) {
    throw new Error('Unsafe PostgreSQL identifier');
  }
  return `"${identifier}"`;
}

async function loadDatasetSchema() {
  return fs.readFile(DATASET_SCHEMA_PATH, 'utf8');
}

function renderDatasetSchema(sql, schemaName) {
  return sql.replaceAll('{{schema}}', quoteIdentifier(schemaName));
}

async function ensureRole(client, roleName) {
  const exists = await client.query('SELECT 1 FROM pg_roles WHERE rolname = $1', [roleName]);
  if (!exists.rowCount) {
    await client.query(`CREATE ROLE ${quoteIdentifier(roleName)} NOLOGIN`);
  }
}

function sanitizedProvisioningError(error) {
  const message = String(error?.message || 'Dataset provisioning failed')
    .replace(/\s+/g, ' ')
    .trim();
  return message.slice(0, 500);
}

export async function provisionDatasetSchema(datasetId, teamId) {
  const catalogId = String(datasetId || '').trim();
  if (!catalogId) {
    throw new Error('dataset id is required');
  }
  const adminRole = teamAdminRoleForTeam(teamId);

  const existing = await query(
    `SELECT schema_name, team_id::text AS team_id, status, schema_version
     FROM w4h.dataset_registry
     WHERE dataset_id = $1`,
    [catalogId]
  );

  let schemaName;
  if (existing.rowCount) {
    const row = existing.rows[0];
    if (row.team_id !== String(teamId)) {
      throw new Error('Existing dataset registry team does not match');
    }
    if (!DATASET_SCHEMA_NAME.test(row.schema_name)) {
      throw new Error('Existing dataset registry contains an invalid schema name');
    }
    if (row.status === 'archived') {
      throw new Error('Archived dataset schemas cannot be provisioned');
    }
    schemaName = row.schema_name;
    if (row.status === 'active' && row.schema_version >= DATASET_SCHEMA_VERSION) {
      return { schemaName, status: 'active', schemaVersion: row.schema_version };
    }
  } else {
    schemaName = datasetSchemaNameFromUuid(randomUUID());
  }

  await query(
    `INSERT INTO w4h.dataset_registry
       (dataset_id, team_id, schema_name, status, schema_version)
     VALUES ($1, $2::uuid, $3, 'provisioning', 0)
     ON CONFLICT (dataset_id) DO UPDATE
       SET status = 'provisioning',
           last_error = NULL,
           updated_at = CURRENT_TIMESTAMP`,
    [catalogId, teamId, schemaName]
  );

  try {
    const datasetSchemaSql = await loadDatasetSchema();
    const appliedVersion = await withTransaction(async (client) => {
      await client.query('SELECT pg_advisory_xact_lock(hashtext($1))', [catalogId]);
      await client.query(
        `SELECT 1 FROM w4h.dataset_registry
         WHERE dataset_id = $1 FOR UPDATE`,
        [catalogId]
      );

      await client.query(`SET LOCAL ROLE ${PLATFORM_ROLE}`);
      await ensureRole(client, adminRole);

      const schema = quoteIdentifier(schemaName);
      const quotedAdmin = quoteIdentifier(adminRole);
      await client.query(`CREATE SCHEMA IF NOT EXISTS ${schema} AUTHORIZATION ${PLATFORM_ROLE}`);
      await client.query(`ALTER SCHEMA ${schema} OWNER TO ${PLATFORM_ROLE}`);
      await client.query(renderDatasetSchema(datasetSchemaSql, schemaName));

      await client.query(`GRANT USAGE ON SCHEMA ${schema} TO ${quotedAdmin}`);
      await client.query(`GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA ${schema} TO ${quotedAdmin}`);
      await client.query(`GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA ${schema} TO ${quotedAdmin}`);
      await client.query(
        `ALTER DEFAULT PRIVILEGES FOR ROLE ${PLATFORM_ROLE} IN SCHEMA ${schema}
         GRANT ALL PRIVILEGES ON TABLES TO ${quotedAdmin}`
      );
      await client.query(
        `ALTER DEFAULT PRIVILEGES FOR ROLE ${PLATFORM_ROLE} IN SCHEMA ${schema}
         GRANT ALL PRIVILEGES ON SEQUENCES TO ${quotedAdmin}`
      );

      await client.query('RESET ROLE');
      await client.query(
        `UPDATE w4h.dataset_registry
         SET schema_version = $2,
             status = 'active',
             provisioned_at = CURRENT_TIMESTAMP,
             last_error = NULL,
             updated_at = CURRENT_TIMESTAMP
         WHERE dataset_id = $1`,
        [catalogId, DATASET_SCHEMA_VERSION]
      );
      return DATASET_SCHEMA_VERSION;
    });

    return { schemaName, status: 'active', schemaVersion: appliedVersion };
  } catch (error) {
    await query(
      `UPDATE w4h.dataset_registry
       SET status = 'failed',
           last_error = $2,
           updated_at = CURRENT_TIMESTAMP
       WHERE dataset_id = $1`,
      [catalogId, sanitizedProvisioningError(error)]
    );
    throw error;
  }
}

export async function grantDatasetSchemaToRole(schemaName, roleName, accessType = 'read') {
  if (!DATASET_SCHEMA_NAME.test(schemaName) || !ROLE_NAME.test(roleName)) {
    throw new Error('Unsafe PostgreSQL identifier');
  }
  const write = accessType === 'read_write';
  await withTransaction(async (client) => {
    await client.query(`SET LOCAL ROLE ${PLATFORM_ROLE}`);
    await ensureRole(client, roleName);
    const schema = quoteIdentifier(schemaName);
    const role = quoteIdentifier(roleName);
    await client.query(`GRANT USAGE ON SCHEMA ${schema} TO ${role}`);
    if (write) {
      await client.query(`GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA ${schema} TO ${role}`);
      await client.query(`GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA ${schema} TO ${role}`);
      await client.query(
        `ALTER DEFAULT PRIVILEGES FOR ROLE ${PLATFORM_ROLE} IN SCHEMA ${schema}
         GRANT ALL PRIVILEGES ON TABLES TO ${role}`
      );
      await client.query(
        `ALTER DEFAULT PRIVILEGES FOR ROLE ${PLATFORM_ROLE} IN SCHEMA ${schema}
         GRANT ALL PRIVILEGES ON SEQUENCES TO ${role}`
      );
    } else {
      await client.query(`REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA ${schema} FROM ${role}`);
      await client.query(`REVOKE ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA ${schema} FROM ${role}`);
      await client.query(`GRANT SELECT ON ALL TABLES IN SCHEMA ${schema} TO ${role}`);
      await client.query(
        `ALTER DEFAULT PRIVILEGES FOR ROLE ${PLATFORM_ROLE} IN SCHEMA ${schema}
         REVOKE ALL PRIVILEGES ON TABLES FROM ${role}`
      );
      await client.query(
        `ALTER DEFAULT PRIVILEGES FOR ROLE ${PLATFORM_ROLE} IN SCHEMA ${schema}
         GRANT SELECT ON TABLES TO ${role}`
      );
    }
    await client.query('RESET ROLE');
  });
}

export async function revokeDatasetSchemaFromRole(schemaName, roleName) {
  if (!DATASET_SCHEMA_NAME.test(schemaName) || !ROLE_NAME.test(roleName)) {
    throw new Error('Unsafe PostgreSQL identifier');
  }
  await withTransaction(async (client) => {
    await client.query(`SET LOCAL ROLE ${PLATFORM_ROLE}`);
    const schema = quoteIdentifier(schemaName);
    const role = quoteIdentifier(roleName);
    const roleExists = await client.query('SELECT 1 FROM pg_roles WHERE rolname = $1', [roleName]);
    const schemaExists = await client.query(
      `SELECT 1 FROM information_schema.schemata WHERE schema_name = $1`,
      [schemaName]
    );
    if (roleExists.rowCount && schemaExists.rowCount) {
      await client.query(
        `ALTER DEFAULT PRIVILEGES FOR ROLE ${PLATFORM_ROLE} IN SCHEMA ${schema}
         REVOKE ALL PRIVILEGES ON TABLES FROM ${role}`
      );
      await client.query(
        `ALTER DEFAULT PRIVILEGES FOR ROLE ${PLATFORM_ROLE} IN SCHEMA ${schema}
         REVOKE ALL PRIVILEGES ON SEQUENCES FROM ${role}`
      );
      await client.query(`REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA ${schema} FROM ${role}`);
      await client.query(`REVOKE ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA ${schema} FROM ${role}`);
      await client.query(`REVOKE USAGE, CREATE ON SCHEMA ${schema} FROM ${role}`);
    }
    await client.query('RESET ROLE');
  });
}
