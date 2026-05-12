import crypto from 'crypto';

import { authenticate } from '../../auth.js';
import { query } from '../../db.js';
import { readUserApiKeyFromRequest, loadUserFromApiKeyRaw } from './userApiKeyAuth.js';

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

async function resolveProjectId(projectRef) {
  const ref = String(projectRef || '').trim();
  if (!ref) return null;
  if (UUID_RE.test(ref)) return ref;
  const found = await query(`SELECT id FROM w4h.projects WHERE slug = $1`, [ref]);
  return found.rowCount ? found.rows[0].id : null;
}

async function canReadProjectForMiddleware(projectId, user) {
  if (!user || !projectId) return false;
  if (user.role === 'admin') return true;
  const member = await query(
    `SELECT role
     FROM w4h.project_members
     WHERE project_id = $1::uuid AND user_id = $2`,
    [projectId, user.id]
  );
  return member.rowCount > 0;
}

async function authenticateJwtOrProjectDbToken(req, res, next) {
  const projectId = await resolveProjectId(req.params?.id);

  const apiKeyRaw = readUserApiKeyFromRequest(req);
  if (apiKeyRaw) {
    try {
      const user = await loadUserFromApiKeyRaw(apiKeyRaw);
      if (!user) {
        return res.sendStatus(401);
      }
      if (!projectId || !(await canReadProjectForMiddleware(projectId, user))) {
        return res.status(403).json({ error: 'Invalid API key or no access to this project' });
      }
      req.user = user;
      req.authViaUserApiKey = true;
      return next();
    } catch (err) {
      console.error('Project route user API key auth error:', err);
      return res.status(500).json({ error: err.message || 'Authentication failed' });
    }
  }

  const authHeader = req.headers.authorization;
  if (authHeader) {
    return authenticate(req, res, next);
  }

  const raw = req.headers['x-w4h-project-db-token'];
  if (!raw || !projectId) {
    return res.sendStatus(401);
  }

  const tokenHash = crypto.createHash('sha256').update(String(raw)).digest('hex');

  query(
    `SELECT t.id AS token_row_id, t.user_id, u.email, u.name, u.role
     FROM w4h.project_db_tokens t
     JOIN w4h.users u ON u.id = t.user_id
     WHERE t.token_hash = $1
       AND t.project_id = $2::uuid
       AND t.revoked_at IS NULL
       AND t.expires_at > NOW()`,
    [tokenHash, projectId]
  )
    .then((result) => {
      if (!result.rowCount) {
        return res.status(403).json({ error: 'Invalid, expired, or revoked project token' });
      }
      const row = result.rows[0];
      req.user = {
        id: row.user_id,
        email: row.email,
        name: row.name,
        role: row.role
      };
      req.projectDbTokenRowId = row.token_row_id;
      req.authViaProjectDbToken = true;
      next();
    })
    .catch((err) => {
      console.error('Project DB token auth error:', err);
      res.status(500).json({ error: err.message || 'Authentication failed' });
    });
}

export { authenticateJwtOrProjectDbToken, resolveProjectId };
