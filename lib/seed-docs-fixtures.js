import bcrypt from 'bcrypt';
import pkg from 'pg';
import { query, resetPool } from '../db.js';
import { initializeDatabase } from './db-init.js';
import { validatePassword } from './utils/password-validator.js';

const { Client } = pkg;

/** Dedicated Postgres database for docs screenshots. Never the working `DB_DATABASE`. */
export const SAMPLE_DATABASE = 'w4h-sample';

/**
 * Synthetic identities for local docs screenshots (not real people).
 * Keep emails in sync with w4h-docs/dev/docs-fixtures.md.
 */
export const DOCS_FIXTURES = {
  passwordDefault: 'DocsOnly!2026',
  team: { slug: 'docs-demo', name: 'Demo Trial' },
  project: {
    slug: 'docs-pilot',
    name: 'Wearable Pilot',
    description: 'Synthetic project used only for local documentation screenshots.',
  },
  dataset: {
    id: 'docs-demo-heart-rate',
    name: 'Demo heart rate',
    description: 'Catalog placeholder for docs screenshots (no PHI).',
  },
  accessRequest: {
    name: 'Riley Morgan',
    email: 'researcher@example.com',
    affiliation: 'Example Cardiology Lab',
    purpose: 'Pilot study of resting heart-rate trends from wearables (synthetic, docs only).',
  },
  users: {
    superadmin: {
      name: 'Alex Rivera',
      email: 'superadmin@example.com',
      role: 'super_admin',
    },
    admin: {
      name: 'Jordan Lee',
      email: 'admin@example.com',
      role: 'project_user',
    },
    user: {
      name: 'Sam Chen',
      email: 'user@example.com',
      role: 'project_user',
    },
  },
};

function quoteIdent(name) {
  if (!/^[A-Za-z0-9_-]+$/.test(name)) {
    throw new Error(`Refusing unsafe database name "${name}"`);
  }
  return `"${name}"`;
}

function assertLocalHost() {
  const host = String(process.env.DB_HOST || '').trim().toLowerCase();
  const allowRemote = process.env.W4H_SEED_DOCS_ALLOW_REMOTE === '1';
  if (allowRemote) return;
  if (!['localhost', '127.0.0.1', '::1'].includes(host)) {
    throw new Error(
      `seed-docs refuses non-local DB_HOST="${process.env.DB_HOST}". ` +
        'It only creates w4h-sample on the local Postgres. ' +
        'Set W4H_SEED_DOCS_ALLOW_REMOTE=1 only if you really mean to.'
    );
  }
}

function serverConfig() {
  const host = process.env.DB_HOST;
  const port = process.env.DB_PORT;
  const user = process.env.DB_USER;
  const password = process.env.DB_PASSWORD;
  if (!host || !port || !user || !password) {
    throw new Error(
      'Need DB_HOST, DB_PORT, DB_USER, and DB_PASSWORD in w4h-api/.env to reach Postgres. ' +
        'seed-docs does not write to DB_DATABASE; it creates a separate w4h-sample database.'
    );
  }
  return { host, port: parseInt(port, 10), user, password };
}

async function withAdminClient(fn) {
  const client = new Client({ ...serverConfig(), database: 'postgres' });
  await client.connect();
  try {
    return await fn(client);
  } finally {
    await client.end();
  }
}

async function ensureSampleDatabase({ name, reset }) {
  await withAdminClient(async (client) => {
    const found = await client.query('SELECT 1 FROM pg_database WHERE datname = $1', [name]);
    if (reset && found.rowCount) {
      await client.query(
        `SELECT pg_terminate_backend(pid)
           FROM pg_stat_activity
          WHERE datname = $1 AND pid <> pg_backend_pid()`,
        [name]
      );
      await client.query(`DROP DATABASE ${quoteIdent(name)}`);
    }
    const existsAfter = reset
      ? { rowCount: 0 }
      : await client.query('SELECT 1 FROM pg_database WHERE datname = $1', [name]);
    if (!existsAfter.rowCount) {
      const priv = await client.query(
        'SELECT usecreatedb FROM pg_user WHERE usename = $1',
        [process.env.DB_USER]
      );
      if (!priv.rowCount) {
        throw new Error(`Postgres user '${process.env.DB_USER}' not found.`);
      }
      if (!priv.rows[0].usecreatedb) {
        throw new Error(
          `User '${process.env.DB_USER}' cannot CREATE DATABASE. ` +
            `Ask a DBA: ALTER USER "${process.env.DB_USER}" CREATEDB;`
        );
      }
      await client.query(`CREATE DATABASE ${quoteIdent(name)} TEMPLATE template1 ENCODING 'UTF8'`);
    }
  });
}

async function upsertUser({ name, email, role, passwordHash }) {
  const existing = await query('SELECT id FROM w4h.users WHERE email = $1', [email]);
  if (existing.rowCount) {
    await query(
      `UPDATE w4h.users
          SET name = $1, password = $2, role = $3, status = 'active', updated_at = NOW()
        WHERE email = $4`,
      [name, passwordHash, role, email]
    );
    return existing.rows[0].id;
  }
  const inserted = await query(
    `INSERT INTO w4h.users (name, email, password, role, status)
     VALUES ($1, $2, $3, $4, 'active')
     RETURNING id`,
    [name, email, passwordHash, role]
  );
  return inserted.rows[0].id;
}

async function upsertTeam() {
  const { slug, name } = DOCS_FIXTURES.team;
  const existing = await query('SELECT id FROM w4h.teams WHERE slug = $1', [slug]);
  if (existing.rowCount) {
    await query(
      `UPDATE w4h.teams SET name = $1, status = 'active', updated_at = NOW() WHERE slug = $2`,
      [name, slug]
    );
    return existing.rows[0].id;
  }
  const inserted = await query(
    `INSERT INTO w4h.teams (slug, name, status) VALUES ($1, $2, 'active') RETURNING id`,
    [slug, name]
  );
  return inserted.rows[0].id;
}

async function upsertTeamMember(teamId, userId, role) {
  await query(
    `INSERT INTO w4h.team_members (team_id, user_id, role)
     VALUES ($1, $2, $3)
     ON CONFLICT (team_id, user_id) DO UPDATE SET role = EXCLUDED.role`,
    [teamId, userId, role]
  );
}

async function upsertProject(teamId, createdBy) {
  const { slug, name, description } = DOCS_FIXTURES.project;
  const existing = await query('SELECT id FROM w4h.projects WHERE slug = $1', [slug]);
  if (existing.rowCount) {
    await query(
      `UPDATE w4h.projects
          SET name = $1, description = $2, team_id = $3, status = 'active', updated_at = NOW()
        WHERE slug = $4`,
      [name, description, teamId, slug]
    );
    return existing.rows[0].id;
  }
  const inserted = await query(
    `INSERT INTO w4h.projects (slug, name, description, status, visibility, team_id, created_by)
     VALUES ($1, $2, $3, 'active', 'org-private', $4, $5)
     RETURNING id`,
    [slug, name, description, teamId, createdBy]
  );
  return inserted.rows[0].id;
}

async function upsertProjectMember(projectId, userId, role, invitedBy) {
  await query(
    `INSERT INTO w4h.project_members (project_id, user_id, role, invited_by)
     VALUES ($1, $2, $3, $4)
     ON CONFLICT (project_id, user_id) DO UPDATE SET role = EXCLUDED.role`,
    [projectId, userId, role, invitedBy]
  );
}

async function upsertDataset(teamId, createdBy, projectId) {
  const { id, name, description } = DOCS_FIXTURES.dataset;
  const existing = await query('SELECT id FROM w4h.datasets WHERE id = $1', [id]);
  if (existing.rowCount) {
    await query(
      `UPDATE w4h.datasets
          SET name = $1, description = $2, team_id = $3, created_by = $4, updated_at = NOW()
        WHERE id = $5`,
      [name, description, teamId, createdBy, id]
    );
  } else {
    await query(
      `INSERT INTO w4h.datasets (
         id, name, description, team_id, created_by, discoverable,
         subject_count, device_type, sampling_rate
       ) VALUES ($1, $2, $3, $4, $5, true, 12, 'synthetic', '1 Hz')`,
      [id, name, description, teamId, createdBy]
    );
  }
  await query(
    `INSERT INTO w4h.project_datasets (project_id, dataset_id, access_type, approved_by)
     VALUES ($1, $2, 'read_write', $3)
     ON CONFLICT (project_id, dataset_id) DO NOTHING`,
    [projectId, id, createdBy]
  );
}

/** One pending, verified request so the Super admin "Requests" page has something to show. */
async function upsertAccessRequest() {
  const r = DOCS_FIXTURES.accessRequest;
  const existing = await query(
    `SELECT 1 FROM w4h.access_requests WHERE lower(email) = lower($1) AND status IN ('unverified', 'pending')`,
    [r.email]
  );
  if (existing.rowCount) return;
  await query(
    `INSERT INTO w4h.access_requests (name, email, affiliation, purpose, status, verified_at)
     VALUES ($1, $2, $3, $4, 'pending', now())`,
    [r.name, r.email, r.affiliation, r.purpose]
  );
}

async function seedIntoCurrentPool(password) {
  const passwordHash = bcrypt.hashSync(password, +process.env.BCRYPT_ROUNDS || 12);
  const { superadmin, admin, user } = DOCS_FIXTURES.users;

  const superadminId = await upsertUser({ ...superadmin, passwordHash });
  const adminId = await upsertUser({ ...admin, passwordHash });
  const userId = await upsertUser({ ...user, passwordHash });

  const teamId = await upsertTeam();
  await upsertTeamMember(teamId, adminId, 'team_admin');
  await upsertTeamMember(teamId, userId, 'member');

  const projectId = await upsertProject(teamId, adminId);
  await upsertProjectMember(projectId, adminId, 'owner', adminId);
  await upsertProjectMember(projectId, userId, 'editor', adminId);

  await upsertDataset(teamId, adminId, projectId);
  await upsertAccessRequest();

  return {
    users: {
      superadmin: { id: superadminId, ...superadmin },
      admin: { id: adminId, ...admin },
      user: { id: userId, ...user },
    },
    team: { id: teamId, ...DOCS_FIXTURES.team },
    project: { id: projectId, ...DOCS_FIXTURES.project },
    dataset: DOCS_FIXTURES.dataset,
  };
}

/**
 * Create (or refresh) the dedicated `w4h-sample` database and seed synthetic
 * Super admin / Admin / User plus a demo team, project, and catalog row.
 *
 * Uses DB_HOST/USER/PASSWORD only as the Postgres server login. It never
 * INSERT/UPDATEs the working `DB_DATABASE` from .env.
 */
export async function seedDocsFixtures(options = {}) {
  assertLocalHost();

  const sampleName = process.env.W4H_SAMPLE_DATABASE || SAMPLE_DATABASE;
  const workingName = process.env.DB_DATABASE || '';

  const password = options.password || process.env.W4H_DOCS_PASSWORD || DOCS_FIXTURES.passwordDefault;
  const passwordValidation = validatePassword(password);
  if (!passwordValidation.valid) {
    throw new Error(passwordValidation.message);
  }

  await ensureSampleDatabase({ name: sampleName, reset: !!options.reset });

  const previousDb = process.env.DB_DATABASE;
  process.env.DB_DATABASE = sampleName;
  try {
    await resetPool();
    const init = await initializeDatabase();
    if (!init.success) {
      throw new Error(init.error || 'Failed to initialize w4h-sample schema');
    }
    const seeded = await seedIntoCurrentPool(password);
    return {
      database: sampleName,
      workingDatabase: workingName || null,
      reset: !!options.reset,
      passwordWasDefault: password === DOCS_FIXTURES.passwordDefault,
      ...seeded,
    };
  } finally {
    if (previousDb === undefined) delete process.env.DB_DATABASE;
    else process.env.DB_DATABASE = previousDb;
    try {
      await resetPool();
    } catch {
      // CLI exits next; ignore reconnect to the working DB.
    }
  }
}
