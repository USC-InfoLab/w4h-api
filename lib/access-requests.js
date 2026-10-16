import crypto from 'crypto';

/** Escape text for HTML email bodies (requesters are unauthenticated guests). */
export function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

export const VERIFY_TTL_HOURS = 24;

/** Exact-match domain allowlist (never suffix matching: `usc.edu.evil.com` must fail). */
export function allowedDomains() {
  const raw = process.env.ACCESS_REQUEST_ALLOWED_DOMAINS || 'usc.edu';
  return raw
    .split(',')
    .map((d) => d.trim().toLowerCase().replace(/^@/, ''))
    .filter(Boolean);
}

/** Returns the normalized email, or null when it is malformed. */
export function normalizeEmail(value) {
  const email = String(value ?? '').trim().toLowerCase();
  if (!email || email.length > 254 || /\s/.test(email)) return null;
  const parts = email.split('@');
  if (parts.length !== 2 || !parts[0] || !parts[1] || !parts[1].includes('.')) return null;
  return email;
}

export function isAllowedEmailDomain(email) {
  const domain = String(email).split('@')[1] || '';
  return allowedDomains().includes(domain);
}

export function hashToken(token) {
  return crypto.createHash('sha256').update(String(token)).digest('hex');
}

export function newVerifyToken() {
  return crypto.randomBytes(32).toString('base64url');
}

/** Tiny in-memory sliding-window limiter (single API process; fine for a low-volume form). */
export function createLimiter({ windowMs, max }) {
  const hits = new Map();
  return function hit(key) {
    const now = Date.now();
    const recent = (hits.get(key) || []).filter((t) => now - t < windowMs);
    if (recent.length >= max) {
      hits.set(key, recent);
      return false;
    }
    recent.push(now);
    hits.set(key, recent);
    if (hits.size > 5000) {
      for (const [k, v] of hits) {
        if (!v.some((t) => now - t < windowMs)) hits.delete(k);
      }
    }
    return true;
  };
}
