import jwt from 'jsonwebtoken';
import crypto from 'crypto';
import { query } from './db.js';
import { isSuperAdmin, canAccessAdminUi, canManageTeamAsync } from './lib/utils/user.js';

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

/** Ensure JWT payloads always expose a numeric user id (older tokens may omit it). */
async function ensureUserIdOnPayload(decoded) {
  if (!decoded || typeof decoded !== 'object') return decoded;
  if (decoded.id != null && decoded.id !== '') return decoded;

  const email = typeof decoded.email === 'string' ? decoded.email.trim() : '';
  if (!email) return decoded;

  const result = await query(
    `SELECT id, name, email, role, status FROM w4h.users WHERE lower(trim(email)) = lower(trim($1))`,
    [email]
  );
  if (!result.rowCount) return decoded;

  const u = result.rows[0];
  return {
    ...decoded,
    id: u.id,
    name: decoded.name || u.name,
    email: decoded.email || u.email,
    role: decoded.role || u.role,
    status: u.status,
  };
}

async function assertUserAccountActive(user) {
  if (!user?.id) {
    return { ok: false, status: 403 };
  }
  const result = await query(
    `SELECT status FROM w4h.users WHERE id = $1`,
    [user.id]
  );
  if (!result.rowCount) {
    return { ok: false, status: 403 };
  }
  const status = result.rows[0].status;
  if (status === 'discontinued' || status === 'invited') {
    return { ok: false, status: 403, reason: status };
  }
  return { ok: true, status };
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

    ensureUserIdOnPayload(decoded)
      .then(async (user) => {
        const active = await assertUserAccountActive(user);
        if (!active.ok) {
          return res.sendStatus(403);
        }
        req.user = user;
        next();
      })
      .catch((e) => {
        console.error('authenticate ensureUserId:', e);
        res.sendStatus(500);
      });
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
      ensureUserIdOnPayload(decoded)
        .then(async (user) => {
          const active = await assertUserAccountActive(user);
          if (!active.ok) {
            return res.status(403).json({ error: 'Account is not active' });
          }
          req.user = user;
          req.authVia = 'jwt';
          next();
        })
        .catch((e) => {
          console.error('authenticateJwtOrApiKey ensureUserId:', e);
          res.status(500).json({ error: e.message || 'Authentication failed' });
        });
    });
  }

  if (cred.startsWith(USER_API_KEY_PREFIX)) {
    const tokenHash = crypto.createHash('sha256').update(cred).digest('hex');
    return query(
      `SELECT id, name, email, role, status FROM w4h.users WHERE api_key_hash = $1`,
      [tokenHash]
    )
      .then((result) => {
        if (!result.rowCount) {
          return res.status(401).json({ error: 'Invalid or revoked API key' });
        }
        const u = result.rows[0];
        if (u.status !== 'active') {
          return res.status(403).json({ error: 'Account is not active' });
        }
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

/** Platform Admin only (Super admin). Team admins must not pass this gate. */
const requirePlatformAdmin = function (req, res, next) {
  if (!req.user || !canAccessAdminUi(req.user.role)) {
    return res.status(403).json({
      success: false,
      error: 'Access denied. Platform admin privileges required.'
    });
  }
  next();
};

/** Team-scoped management (team_members.role = team_admin on at least one team — not Super admin). */
const requireTeamAdmin = function (req, res, next) {
  if (!req.user) {
    return res.status(403).json({
      success: false,
      error: 'Access denied. Team admin privileges required.'
    });
  }
  Promise.resolve(canManageTeamAsync(req.user))
    .then((ok) => {
      if (!ok) {
        return res.status(403).json({
          success: false,
          error: 'Access denied. Team admin privileges required.'
        });
      }
      next();
    })
    .catch((e) => {
      console.error('requireTeamAdmin:', e);
      res.sendStatus(500);
    });
};

/** @deprecated Use requirePlatformAdmin */
const requireAdmin = requirePlatformAdmin;
/** @deprecated Use requirePlatformAdmin */
const requireAdminUi = requirePlatformAdmin;

export {
  authenticate,
  authenticateJwtOrApiKey,
  sign,
  requireAdmin,
  requireAdminUi,
  requirePlatformAdmin,
  requireTeamAdmin,
};
