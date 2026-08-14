import crypto from 'crypto';

const ALGO = 'aes-256-gcm';
const IV_LEN = 12;

function wrapSecret() {
  const raw =
    process.env.API_KEY_WRAP_SECRET ||
    process.env.TOKEN_SECRET ||
    '';
  if (!raw) {
    throw new Error('API_KEY_WRAP_SECRET or TOKEN_SECRET is required to wrap API keys');
  }
  return crypto.createHash('sha256').update(String(raw)).digest();
}

/** Encrypt plaintext API key for step-up reveal (HIPAA: never log the result). */
export function encryptUserApiKey(plaintext) {
  const key = wrapSecret();
  const iv = crypto.randomBytes(IV_LEN);
  const cipher = crypto.createCipheriv(ALGO, key, iv);
  const enc = Buffer.concat([cipher.update(String(plaintext), 'utf8'), cipher.final()]);
  const tag = cipher.getAuthTag();
  return Buffer.concat([iv, tag, enc]).toString('base64url');
}

/** Decrypt a wrapped API key. Returns null if payload is missing/invalid. */
export function decryptUserApiKey(payload) {
  if (!payload) return null;
  try {
    const buf = Buffer.from(String(payload), 'base64url');
    if (buf.length <= IV_LEN + 16) return null;
    const iv = buf.subarray(0, IV_LEN);
    const tag = buf.subarray(IV_LEN, IV_LEN + 16);
    const data = buf.subarray(IV_LEN + 16);
    const key = wrapSecret();
    const decipher = crypto.createDecipheriv(ALGO, key, iv);
    decipher.setAuthTag(tag);
    return Buffer.concat([decipher.update(data), decipher.final()]).toString('utf8');
  } catch {
    return null;
  }
}
