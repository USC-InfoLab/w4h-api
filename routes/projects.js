import express from 'express';
import crypto from 'crypto';

import { authenticate } from '../auth.js';
import { query } from '../lib/utils/db.js';
import { USER_ROLES } from '../lib/utils/user.js';

const router = express.Router();

router.get('/', authenticate, listProjects);
router.get('/:id', authenticate, getProjectOverview);
router.post('/', authenticate, createProject);
router.patch('/:id', authenticate, updateProject);
router.put('/:id/members', authenticate, replaceProjectMembers);
router.put('/:id/datasets', authenticate, replaceProjectDatasets);
router.put('/:id/notebooks', authenticate, replaceProjectNotebooks);
router.put('/:id/docs', authenticate, replaceProjectDocs);
router.post('/:id/db-token', authenticate, issueProjectDbToken);

function canCreateProject(user) {
  return !!user && [USER_ROLES.ADMIN, USER_ROLES.RESEARCHER, 'clinician'].includes(user.role);
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
  if (user.role === USER_ROLES.ADMIN) return true;
  const member = await getMembership(projectId, user.id);
  return !!member;
}

async function canManageProject(projectId, user) {
  if (!user) return false;
  if (user.role === USER_ROLES.ADMIN) return true;
  const member = await getMembership(projectId, user.id);
  return !!member && ['owner', 'admin', 'editor'].includes(member.role);
}

function slugify(input) {
  return String(input || '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/(^-|-$)/g, '');
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
             p.irb_protocol_id, p.irb_expiry_date, p.created_at, p.updated_at
      FROM w4h.projects p
    `;
    const params = [];

    if (req.user.role !== USER_ROLES.ADMIN) {
      sql += `
        JOIN w4h.project_members pm ON pm.project_id = p.id
        WHERE pm.user_id = $1
      `;
      params.push(req.user.id);
    }

    sql += ` ORDER BY p.updated_at DESC, p.created_at DESC`;

    const projects = await query(sql, params);
    res.json(projects.rows);
  } catch (error) {
    res.status(500).json({ error: error.message || 'Failed to list projects' });
  }
}

async function getProjectOverview(req, res) {
  const projectId = req.params.id;
  try {
    if (!(await canReadProject(projectId, req.user))) {
      return res.status(403).json({ error: 'Unauthorized' });
    }

    const projectRes = await query(
      `SELECT p.*,
              u.name AS created_by_name,
              u.email AS created_by_email
       FROM w4h.projects p
       LEFT JOIN w4h.users u ON u.id = p.created_by
       WHERE p.id = $1::uuid`,
      [projectId]
    );
    if (!projectRes.rowCount) {
      return res.status(404).json({ error: 'Project not found' });
    }

    const [membersRes, datasetsRes, notebooksRes, docsRes, orgsRes] = await Promise.all([
      query(
        `SELECT pm.user_id, pm.role, pm.joined_at,
                u.name, u.email, u.role AS user_role
         FROM w4h.project_members pm
         JOIN w4h.users u ON u.id = pm.user_id
         WHERE pm.project_id = $1::uuid
         ORDER BY pm.joined_at ASC`,
        [projectId]
      ),
      query(
        `SELECT pd.dataset_id, pd.access_type, pd.approved_at,
                d.name, d.description, d.study_id
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
        `SELECT DISTINCT s.id AS study_id, s.name AS study_name
         FROM w4h.project_datasets pd
         JOIN w4h.datasets d ON d.id = pd.dataset_id
         LEFT JOIN w4h.studies s ON s.id = d.study_id
         WHERE pd.project_id = $1::uuid
         ORDER BY s.name ASC NULLS LAST`,
        [projectId]
      )
    ]);

    res.json({
      project: projectRes.rows[0],
      members: membersRes.rows,
      datasets: datasetsRes.rows,
      notebooks: notebooksRes.rows,
      docs: docsRes.rows,
      organizations: orgsRes.rows
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
    const name = String(req.body?.name || '').trim();
    const description = req.body?.description == null ? null : String(req.body.description).trim();
    const status = ['active', 'archived'].includes(req.body?.status) ? req.body.status : 'active';
    const visibility = ['org-private', 'project-private', 'internal'].includes(req.body?.visibility)
      ? req.body.visibility
      : 'org-private';
    const irbProtocolId = req.body?.irb_protocol_id == null ? null : String(req.body.irb_protocol_id).trim();
    const irbExpiryDate = req.body?.irb_expiry_date || null;

    if (!name) return res.status(400).json({ error: 'name is required' });

    const slug = await ensureUniqueSlug(req.body?.slug || name);
    const inserted = await query(
      `INSERT INTO w4h.projects
         (slug, name, description, status, visibility, irb_protocol_id, irb_expiry_date, created_by)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
       RETURNING *`,
      [slug, name, description, status, visibility, irbProtocolId || null, irbExpiryDate, req.user.id]
    );

    const project = inserted.rows[0];
    await query(
      `INSERT INTO w4h.project_members (project_id, user_id, role, invited_by)
       VALUES ($1::uuid, $2, 'owner', $2)
       ON CONFLICT (project_id, user_id) DO UPDATE SET role = EXCLUDED.role`,
      [project.id, req.user.id]
    );

    res.status(201).json(project);
  } catch (error) {
    res.status(500).json({ error: error.message || 'Failed to create project' });
  }
}

async function updateProject(req, res) {
  const projectId = req.params.id;
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

    if (!fields.length) return res.status(400).json({ error: 'No updatable fields provided' });
    params.push(projectId);

    const updated = await query(
      `UPDATE w4h.projects
       SET ${fields.join(', ')}, updated_at = NOW()
       WHERE id = $${params.length}::uuid
       RETURNING *`,
      params
    );

    if (!updated.rowCount) return res.status(404).json({ error: 'Project not found' });
    res.json(updated.rows[0]);
  } catch (error) {
    res.status(500).json({ error: error.message || 'Failed to update project' });
  }
}

async function replaceProjectMembers(req, res) {
  const projectId = req.params.id;
  if (!(await canManageProject(projectId, req.user))) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  try {
    const members = Array.isArray(req.body?.members) ? req.body.members : [];
    const allowedRoles = new Set(['owner', 'editor', 'viewer', 'admin']);

    await query(`DELETE FROM w4h.project_members WHERE project_id = $1::uuid`, [projectId]);

    for (const row of members) {
      const userId = Number(row?.user_id);
      const role = String(row?.role || 'viewer');
      if (!Number.isInteger(userId) || !allowedRoles.has(role)) continue;
      await query(
        `INSERT INTO w4h.project_members (project_id, user_id, role, invited_by)
         VALUES ($1::uuid, $2, $3, $4)
         ON CONFLICT (project_id, user_id) DO UPDATE SET role = EXCLUDED.role`,
        [projectId, userId, role, req.user.id]
      );
    }

    const fresh = await query(
      `SELECT pm.user_id, pm.role, pm.joined_at, u.name, u.email
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
  const projectId = req.params.id;
  if (!(await canManageProject(projectId, req.user))) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  try {
    const datasets = Array.isArray(req.body?.datasets) ? req.body.datasets : [];
    await query(`DELETE FROM w4h.project_datasets WHERE project_id = $1::uuid`, [projectId]);

    for (const row of datasets) {
      const datasetId = String(row?.dataset_id || '').trim();
      const accessType = row?.access_type === 'read_write' ? 'read_write' : 'read';
      if (!datasetId) continue;
      await query(
        `INSERT INTO w4h.project_datasets (project_id, dataset_id, access_type, approved_by)
         VALUES ($1::uuid, $2, $3, $4)
         ON CONFLICT (project_id, dataset_id)
         DO UPDATE SET access_type = EXCLUDED.access_type, approved_by = EXCLUDED.approved_by, approved_at = NOW()`,
        [projectId, datasetId, accessType, req.user.id]
      );
    }

    const fresh = await query(
      `SELECT pd.dataset_id, pd.access_type, pd.approved_at, d.name, d.description, d.study_id
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

async function replaceProjectNotebooks(req, res) {
  const projectId = req.params.id;
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
      await query(
        `INSERT INTO w4h.project_notebooks
           (project_id, title, description, purpose, repo_url, repo_ref, notebook_path, runtime_type, launchers, created_by)
         VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10)`,
        [
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
  const projectId = req.params.id;
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

async function issueProjectDbToken(req, res) {
  const projectId = req.params.id;
  if (!(await canReadProject(projectId, req.user))) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  try {
    const days = Math.max(1, Math.min(7, Number(req.body?.days || 2)));
    const expiresRes = await query(
      `SELECT NOW() + ($1::int || ' days')::interval AS expires_at`,
      [days]
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
