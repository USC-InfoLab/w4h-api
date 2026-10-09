import express from 'express';
import { authenticate, requirePlatformAdmin } from '../auth.js';
import { query } from '../db.js';
import { isMailerConfigured, sendMail } from '../lib/mailer.js';
import { appBaseUrl, deliverInvitation } from '../lib/invitations.js';
import { writeAuditLogSafe } from '../lib/utils/audit.js';
import { isSuperAdmin, USER_ROLES } from '../lib/utils/user.js';
import {
  VERIFY_TTL_HOURS,
  allowedDomains,
  createLimiter,
  escapeHtml,
  hashToken,
  isAllowedEmailDomain,
  newVerifyToken,
  normalizeEmail,
} from '../lib/access-requests.js';

// ADR-028 — platform access requests: public USC-only form → verified → Super admin queue.
const router = express.Router();
const SCHEMA = 'w4h';

const LIMITS = { name: 120, affiliation: 200, purpose: 1000, note: 500 };
const perIp = createLimiter({
  windowMs: 60 * 60 * 1000,
  max: Number(process.env.ACCESS_REQUEST_MAX_PER_IP) || 5,
});
const perEmail = createLimiter({
  windowMs: 60 * 60 * 1000,
  max: Number(process.env.ACCESS_REQUEST_MAX_PER_EMAIL) || 3,
});
const verifyPerIp = createLimiter({ windowMs: 60 * 60 * 1000, max: 30 });

const GENERIC_RECEIVED = {
  ok: true,
  message:
    'Thanks — check your USC inbox for a verification link. A person will review your request once you confirm your address.',
};

router.post('/', submitRequest);
router.post('/verify', verifyRequest);
router.get('/count', authenticate, requirePlatformAdmin, countPending);
router.get('/', authenticate, requirePlatformAdmin, listRequests);
router.post('/:id/fulfill', authenticate, requirePlatformAdmin, fulfillRequest);
router.post('/:id/decline', authenticate, requirePlatformAdmin, declineRequest);
router.post('/:id/spam', authenticate, requirePlatformAdmin, markSpam);

function clean(value, max) {
  return String(value ?? '').trim().slice(0, max);
}

async function sendVerificationEmail({ email, name, rawToken }) {
  const link = `${appBaseUrl()}/request-access/verify?token=${encodeURIComponent(rawToken)}`;
  await sendMail({
    to: email,
    subject: 'Confirm your W4H access request',
    text:
      `Hello ${name},\n\n` +
      `We received a request for W4H platform access from this address.\n` +
      `Confirm it using this link (expires in ${VERIFY_TTL_HOURS} hours):\n${link}\n\n` +
      `A person will review your request after you confirm. If you did not ask for access, ignore this email.\n`,
    html:
      `<p>Hello ${escapeHtml(name)},</p>` +
      `<p>We received a request for <strong>W4H</strong> platform access from this address.</p>` +
      `<p><a href="${link}">Confirm your request</a> (expires in ${VERIFY_TTL_HOURS} hours).</p>` +
      `<p>A person will review your request after you confirm. If you did not ask for access, ignore this email.</p>`,
  });
}

/** Best-effort: a mail failure must never block the request (T-18). */
async function notifySuperAdmins(request) {
  try {
    if (!isMailerConfigured()) return;
    const admins = await query(
      `SELECT email FROM ${SCHEMA}.users
       WHERE role IN ('super_admin', 'admin') AND COALESCE(status, 'active') = 'active'`
    );
    const queue = `${appBaseUrl()}/admin/access-requests`;
    for (const { email } of admins.rows) {
      await sendMail({
        to: email,
        subject: `W4H access request: ${request.name}`,
        text:
          `A verified USC researcher requested W4H access.\n\n` +
          `Name: ${request.name}\nEmail: ${request.email}\n` +
          `Affiliation: ${request.affiliation || '-'}\nPurpose: ${request.purpose || '-'}\n\n` +
          `Review: ${queue}\n`,
        html:
          `<p>A verified USC researcher requested W4H access.</p>` +
          `<ul><li><strong>Name:</strong> ${escapeHtml(request.name)}</li>` +
          `<li><strong>Email:</strong> ${escapeHtml(request.email)}</li>` +
          `<li><strong>Affiliation:</strong> ${escapeHtml(request.affiliation || '-')}</li>` +
          `<li><strong>Purpose:</strong> ${escapeHtml(request.purpose || '-')}</li></ul>` +
          `<p><a href="${queue}">Open the access requests queue</a></p>`,
      }).catch((e) => console.warn('access request notify failed:', e.message));
    }
  } catch (e) {
    console.warn('access request notify failed:', e.message);
  }
}

async function submitRequest(req, res) {
  try {
    // Honeypot: real people leave the hidden field empty. Answer like a success, store nothing.
    if (String(req.body?.website ?? '').trim()) return res.status(202).json(GENERIC_RECEIVED);

    const name = clean(req.body?.name, LIMITS.name);
    const email = normalizeEmail(req.body?.email);
    const affiliation = clean(req.body?.affiliation, LIMITS.affiliation);
    const purpose = clean(req.body?.purpose, LIMITS.purpose);

    if (!name || !email || !affiliation || !purpose) {
      return res.status(400).json({ error: 'Name, email, affiliation and purpose are required.' });
    }
    if (!isAllowedEmailDomain(email)) {
      return res.status(400).json({
        error: `Access requests are limited to USC email addresses for now (${allowedDomains().map((d) => '@' + d).join(', ')}).`,
      });
    }
    if (!isMailerConfigured() || !appBaseUrl()) {
      return res.status(503).json({ error: 'Access requests are temporarily unavailable. Please try again later.' });
    }
    if (!perIp(req.ip || 'unknown') || !perEmail(email)) {
      return res.status(429).json({ error: 'Too many requests. Please try again later.' });
    }

    // Drop stale unverified rows so the same address can try again.
    await query(
      `DELETE FROM ${SCHEMA}.access_requests
       WHERE status = 'unverified' AND verify_expires_at < now()`
    );

    // Never reveal whether an account or request already exists.
    const existingUser = await query(`SELECT 1 FROM ${SCHEMA}.users WHERE lower(email) = $1`, [email]);
    if (existingUser.rowCount) return res.status(202).json(GENERIC_RECEIVED);
    const blocked = await query(
      `SELECT 1 FROM ${SCHEMA}.access_requests WHERE lower(email) = $1 AND status IN ('pending', 'spam')`,
      [email]
    );
    if (blocked.rowCount) return res.status(202).json(GENERIC_RECEIVED);

    const rawToken = newVerifyToken();
    const saved = await query(
      `INSERT INTO ${SCHEMA}.access_requests
         (name, email, affiliation, purpose, status, verify_token_hash, verify_expires_at)
       VALUES ($1, $2, $3, $4, 'unverified', $5, now() + ($6 || ' hours')::interval)
       ON CONFLICT (lower(email)) WHERE status IN ('unverified', 'pending')
       DO UPDATE SET name = EXCLUDED.name, affiliation = EXCLUDED.affiliation,
                     purpose = EXCLUDED.purpose, verify_token_hash = EXCLUDED.verify_token_hash,
                     verify_expires_at = EXCLUDED.verify_expires_at
         WHERE ${SCHEMA}.access_requests.status = 'unverified'
       RETURNING id`,
      [name, email, affiliation, purpose, hashToken(rawToken), String(VERIFY_TTL_HOURS)]
    );
    if (!saved.rowCount) return res.status(202).json(GENERIC_RECEIVED);

    try {
      await sendVerificationEmail({ email, name, rawToken });
    } catch (mailError) {
      console.error('access request verification mail failed:', mailError.message);
      return res.status(502).json({ error: 'We could not send the verification email. Please try again later.' });
    }
    writeAuditLogSafe(req, {
      action: 'access_request.create',
      resourceType: 'access_request',
      resourceId: saved.rows[0].id,
      details: { domain: email.split('@')[1] },
    });
    return res.status(202).json(GENERIC_RECEIVED);
  } catch (error) {
    console.error('submitRequest:', error);
    return res.status(500).json({ error: 'Something went wrong. Please try again later.' });
  }
}

async function verifyRequest(req, res) {
  try {
    if (!verifyPerIp(req.ip || 'unknown')) {
      return res.status(429).json({ error: 'Too many attempts. Please try again later.' });
    }
    const token = String(req.body?.token ?? '').trim();
    if (!token) return res.status(400).json({ error: 'Missing token.' });
    const tokenHash = hashToken(token);

    const updated = await query(
      `UPDATE ${SCHEMA}.access_requests
       SET status = 'pending', verified_at = now(), verify_token_hash = NULL, verify_expires_at = NULL
       WHERE verify_token_hash = $1 AND status = 'unverified' AND verify_expires_at > now()
       RETURNING id, name, email, affiliation, purpose`,
      [tokenHash]
    );
    if (!updated.rowCount) {
      return res.status(400).json({ error: 'This link is invalid or has expired. Please submit the form again.' });
    }
    const request = updated.rows[0];
    writeAuditLogSafe(req, {
      action: 'access_request.verify',
      resourceType: 'access_request',
      resourceId: request.id,
      details: { domain: request.email.split('@')[1] },
    });
    notifySuperAdmins(request);
    return res.json({ ok: true, message: 'Thanks — your request is confirmed and waiting for review.' });
  } catch (error) {
    console.error('verifyRequest:', error);
    return res.status(500).json({ error: 'Something went wrong. Please try again later.' });
  }
}

async function countPending(req, res) {
  try {
    const r = await query(`SELECT COUNT(*)::int AS n FROM ${SCHEMA}.access_requests WHERE status = 'pending'`);
    res.json({ pending: r.rows[0]?.n ?? 0 });
  } catch (error) {
    res.status(500).json({ error: error.message || 'Failed to count requests' });
  }
}

const REQUEST_COLS = `r.id, r.name, r.email, r.affiliation, r.purpose, r.status, r.created_at,
  r.verified_at, r.handled_at, r.note, r.user_id, u.name AS handled_by_name`;

async function listRequests(req, res) {
  try {
    const status = ['pending', 'fulfilled', 'declined', 'spam', 'all'].includes(req.query.status)
      ? req.query.status
      : 'pending';
    const where = status === 'all' ? `r.status <> 'unverified'` : 'r.status = $1';
    const rows = await query(
      `SELECT ${REQUEST_COLS}
       FROM ${SCHEMA}.access_requests r
       LEFT JOIN ${SCHEMA}.users u ON u.id = r.handled_by
       WHERE ${where}
       ORDER BY r.created_at DESC
       LIMIT 200`,
      status === 'all' ? [] : [status]
    );
    res.json(rows.rows);
  } catch (error) {
    res.status(500).json({ error: error.message || 'Failed to list requests' });
  }
}

async function loadPending(id) {
  const r = await query(
    `SELECT id, name, email, affiliation, purpose, status FROM ${SCHEMA}.access_requests WHERE id = $1::uuid`,
    [id]
  );
  return r.rowCount ? r.rows[0] : null;
}

/**
 * Approve (a.k.a. fulfill): the Super admin's whole job. It makes the requester an Admin who
 * may create teams — it does NOT create a team. The Admin creates and defines their own
 * teams after signing in (ADR-028, T-15).
 */
async function fulfillRequest(req, res) {
  try {
    const request = await loadPending(req.params.id);
    if (!request) return res.status(404).json({ error: 'Request not found' });
    if (request.status !== 'pending') {
      return res.status(409).json({ error: 'Only a verified, pending request can be approved' });
    }

    const existing = await query(
      `SELECT id, role FROM ${SCHEMA}.users WHERE lower(email) = $1`,
      [request.email.toLowerCase()]
    );
    if (existing.rowCount && isSuperAdmin(existing.rows[0].role)) {
      return res.status(400).json({ error: 'That email belongs to a Super admin account' });
    }

    let userId;
    let invited = false;
    if (existing.rowCount) {
      userId = existing.rows[0].id;
      await query(`UPDATE ${SCHEMA}.users SET can_create_teams = true WHERE id = $1`, [userId]);
    } else {
      const created = await query(
        `INSERT INTO ${SCHEMA}.users (name, email, password, role, status, can_create_teams)
         VALUES ($1, $2, NULL, $3, 'invited', true) RETURNING id`,
        [request.name, request.email, USER_ROLES.PROJECT_USER]
      );
      userId = created.rows[0].id;
      invited = true;
    }

    let inviteUrl = null;
    let mailWarning = null;
    if (invited) {
      const delivered = await deliverInvitation({
        email: request.email,
        name: request.name,
        roleLabel: 'Admin',
        userId,
        invitedBy: req.user.id,
      });
      inviteUrl = delivered.inviteUrl;
      mailWarning = delivered.mailWarning;
    } else if (isMailerConfigured()) {
      sendMail({
        to: request.email,
        subject: 'Your W4H access request was approved',
        text: `Hello ${request.name},\n\nYour W4H access request was approved. Sign in at ${appBaseUrl()} and create your team.\n`,
        html: `<p>Hello ${escapeHtml(request.name)},</p><p>Your W4H access request was approved. <a href="${appBaseUrl()}">Sign in</a> and create your team.</p>`,
      }).catch((e) => console.warn('access request approve mail failed:', e.message));
    }

    await query(
      `UPDATE ${SCHEMA}.access_requests
       SET status = 'fulfilled', handled_by = $2, handled_at = now(), user_id = $3
       WHERE id = $1::uuid`,
      [request.id, req.user.id, userId]
    );
    writeAuditLogSafe(req, {
      action: 'access_request.fulfill',
      resourceType: 'access_request',
      resourceId: request.id,
      details: { user_id: userId, invited },
    });
    res.json({
      ok: true,
      user_id: userId,
      invited,
      ...(inviteUrl ? { inviteUrl } : {}),
      ...(mailWarning ? { mailWarning } : {}),
    });
  } catch (error) {
    console.error('fulfillRequest:', error);
    res.status(500).json({ error: error.message || 'Failed to approve request' });
  }
}

async function closeRequest(req, res, status, { notify = false } = {}) {
  try {
    const request = await loadPending(req.params.id);
    if (!request) return res.status(404).json({ error: 'Request not found' });
    const open = status === 'spam' ? ['pending', 'unverified'] : ['pending'];
    if (!open.includes(request.status)) {
      return res.status(409).json({ error: `Cannot mark a ${request.status} request as ${status}` });
    }
    const note = clean(req.body?.note, LIMITS.note) || null;
    await query(
      `UPDATE ${SCHEMA}.access_requests
       SET status = $2, handled_by = $3, handled_at = now(), note = $4,
           verify_token_hash = NULL, verify_expires_at = NULL
       WHERE id = $1::uuid`,
      [request.id, status, req.user.id, note]
    );
    if (notify && req.body?.notify && isMailerConfigured()) {
      sendMail({
        to: request.email,
        subject: 'Your W4H access request',
        text: `Hello ${request.name},\n\nWe are not able to approve your W4H access request at this time.${note ? `\n\n${note}` : ''}\n`,
        html: `<p>Hello ${escapeHtml(request.name)},</p><p>We are not able to approve your W4H access request at this time.</p>${note ? `<p>${escapeHtml(note)}</p>` : ''}`,
      }).catch((e) => console.warn('access request decline mail failed:', e.message));
    }
    writeAuditLogSafe(req, {
      action: `access_request.${status === 'declined' ? 'decline' : 'spam'}`,
      resourceType: 'access_request',
      resourceId: request.id,
      details: { had_note: Boolean(note) },
    });
    res.json({ ok: true });
  } catch (error) {
    console.error('closeRequest:', error);
    res.status(500).json({ error: error.message || 'Failed to update request' });
  }
}

function declineRequest(req, res) {
  return closeRequest(req, res, 'declined', { notify: true });
}

function markSpam(req, res) {
  return closeRequest(req, res, 'spam');
}

export default router;
