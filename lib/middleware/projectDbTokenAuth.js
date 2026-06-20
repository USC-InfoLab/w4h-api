import crypto from 'crypto';

import {
  authenticateJwtOrApiKey,
  extractSessionCredential,
  credentialLooksLikeJwt,
  USER_API_KEY_PREFIX
} from '../../auth.js';
import { query } from '../../db.js';

async function resolveProjectId(projectRef) {
  const ref = String(projectRef || '').trim();
  if (!ref) return null;
  const found = await query(
    `SELECT id FROM w4h.projects WHERE id::text = $1 OR slug = $1 LIMIT 1`,
    [ref]
  );
  return found.rowCount ? found.rows[0].id : null;
}

async function authenticateJwtApiKeyOrProjectDbToken(req, res, next) {
  const cred = extractSessionCredential(req);
  if (cred && (credentialLooksLikeJwt(cred) || cred.startsWith(USER_API_KEY_PREFIX))) {
    return authenticateJwtOrApiKey(req, res, next);
  }

  const raw = req.headers['x-w4h-project-db-token'];
  const projectId = await resolveProjectId(req.params?.id);
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

export { authenticateJwtApiKeyOrProjectDbToken };
