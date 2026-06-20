import jwt from 'jsonwebtoken';
import crypto from 'crypto';
import { query } from './db.js';

export const USER_API_KEY_PREFIX = 'w4h_sk_';

export function extractSessionCredential(req) {
  const h = req.headers.authorization;
  if (h) {
    const t = String(h).trim();
    const m = /^Bearer\s+(.+)$/i.exec(t);
    return (m ? m[1] : t).trim();
  }
  const k = req.headers['x-w4h-api-key'];
  return k ? String(k).trim() : '';
}

export function credentialLooksLikeJwt(token) {
  if (!token || typeof token !== 'string') return false;
  const parts = token.split('.');
  return parts.length === 3 && parts.every(p => p.length > 0);
}

const authenticate = function (req, res, next) {
  const authorization = req.headers.authorization;
  if (!authorization) {
    return res.sendStatus(401);
  }

  const cred = /^Bearer\s+(.+)$/i.test(authorization.trim())
    ? authorization.replace(/^Bearer\s+/i, '').trim()
    : authorization.trim();

  jwt.verify(cred, process.env.TOKEN_SECRET, (err, decoded) => {
    if (err) {
      return res.sendStatus(403);
    }

    req.user = decoded;
    next();
  });
};

function authenticateJwtOrApiKey(req, res, next) {
  const cred = extractSessionCredential(req);
  if (!cred) {
    return res.sendStatus(401);
  }

  if (credentialLooksLikeJwt(cred)) {
    return jwt.verify(cred, process.env.TOKEN_SECRET, (err, decoded) => {
      if (err) {
        return res.sendStatus(403);
      }
      req.user = decoded;
      req.authVia = 'jwt';
      next();
    });
  }

  if (cred.startsWith(USER_API_KEY_PREFIX)) {
    const tokenHash = crypto.createHash('sha256').update(cred).digest('hex');
    return query(
      `SELECT id, name, email, role FROM w4h.users WHERE api_key_hash = $1`,
      [tokenHash]
    )
      .then((result) => {
        if (!result.rowCount) {
          return res.status(401).json({ error: 'Invalid or revoked API key' });
        }
        const u = result.rows[0];
        req.user = { id: u.id, name: u.name, email: u.email, role: u.role };
        req.authVia = 'api_key';
        next();
      })
      .catch((err) => {
        console.error('API key auth error:', err);
        res.status(500).json({ error: err.message || 'Authentication failed' });
      });
  }

  return res.status(401).json({ error: 'Expected JWT or w4h_sk_ API key' });
}

const sign = function (user) {
  return jwt.sign(user, process.env.TOKEN_SECRET, { expiresIn: 86400 });
};

const requireAdmin = function (req, res, next) {
  if (!req.user || req.user.role !== 'admin') {
    return res.status(403).json({
      success: false,
      error: 'Access denied. Admin privileges required.'
    });
  }
  next();
};

export { authenticate, authenticateJwtOrApiKey, sign, requireAdmin };
