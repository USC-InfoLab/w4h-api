import crypto from 'crypto';
import { query } from '../../db.js';
import * as auth from '../../auth.js';
import { encryptUserApiKey } from './apiKeyCrypto.js';

const SCHEMA = 'w4h';
const TABLE = 'users';

/**
 * Issue (or rotate) a personal API key for a user. Overwrites any existing key.
 * Returns the plaintext key once — callers must surface it to the client immediately;
 * only the hash/wrapped copy is retained server-side.
 */
export async function issueApiKeyForUser(userId) {
  const raw = `${auth.USER_API_KEY_PREFIX}${crypto.randomBytes(32).toString('base64url')}`;
  const tokenHash = crypto.createHash('sha256').update(raw).digest('hex');
  const prefix = raw.slice(0, 24);
  const wrapped = encryptUserApiKey(raw);

  const result = await query(
    `UPDATE ${SCHEMA}.${TABLE}
     SET api_key_hash = $1, api_key_prefix = $2, api_key_encrypted = $3, api_key_created_at = NOW()
     WHERE id = $4
     RETURNING api_key_created_at`,
    [tokenHash, prefix, wrapped, userId]
  );
  if (!result.rowCount) {
    throw new Error('User not found');
  }

  return {
    api_key: raw,
    api_key_prefix: prefix,
    api_key_created_at: result.rows[0].api_key_created_at
  };
}
