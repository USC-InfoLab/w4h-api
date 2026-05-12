import crypto from 'crypto';
import { query } from '../../db.js';

function hashApiKey(raw) {
  return crypto.createHash('sha256').update(String(raw).trim(), 'utf8').digest('hex');
}

function readUserApiKeyFromRequest(req) {
  const x = req.headers['x-w4h-api-key'];
  if (x != null && String(x).trim()) {
    return String(x).trim();
  }
  const auth = req.headers.authorization;
  if (auth == null) return null;
  const s = String(auth).trim();
  if (/^W4H-Api-Key\s+/i.test(s)) {
    return s.replace(/^W4H-Api-Key\s+/i, '').trim() || null;
  }
  return null;
}

async function loadUserFromApiKeyRaw(rawKey) {
  if (!rawKey || !String(rawKey).trim()) return null;
  const tokenHash = hashApiKey(rawKey);
  const result = await query(
    `SELECT u.id, u.name, u.email, u.role
     FROM w4h.user_api_keys k
     JOIN w4h.users u ON u.id = k.user_id
     WHERE k.token_hash = $1 AND k.revoked_at IS NULL`,
    [tokenHash]
  );
  if (!result.rowCount) return null;
  const row = result.rows[0];
  return { id: row.id, name: row.name, email: row.email, role: row.role };
}

export { readUserApiKeyFromRequest, loadUserFromApiKeyRaw, hashApiKey };
