import crypto from 'crypto';
import { query } from '../db.js';
import { isMailerConfigured, sendMail } from './mailer.js';
import { escapeHtml } from './access-requests.js';

const SCHEMA = 'w4h';
const INVITE_TTL_HOURS = Math.max(1, parseInt(process.env.INVITE_TTL_HOURS, 10) || 72);

export function hashInviteToken(token) {
  return crypto.createHash('sha256').update(String(token)).digest('hex');
}

export function appBaseUrl() {
  return String(process.env.APP_URL || '').replace(/\/+$/, '');
}

export function inviteAcceptUrl(rawToken) {
  const base = appBaseUrl();
  if (!base) return null;
  return `${base}/invite/${encodeURIComponent(rawToken)}`;
}

export async function createInvitation(userId, invitedBy) {
  const rawToken = crypto.randomBytes(32).toString('base64url');
  const tokenHash = hashInviteToken(rawToken);
  await query(
    `UPDATE ${SCHEMA}.invitations
     SET revoked_at = now()
     WHERE user_id = $1 AND used_at IS NULL AND revoked_at IS NULL`,
    [userId]
  );
  await query(
    `INSERT INTO ${SCHEMA}.invitations (user_id, token_hash, invited_by, expires_at)
     VALUES ($1, $2, $3, now() + ($4 || ' hours')::interval)`,
    [userId, tokenHash, invitedBy ?? null, String(INVITE_TTL_HOURS)]
  );
  return rawToken;
}

export async function sendInviteEmail({ email, name, roleLabel, rawToken }) {
  const link = inviteAcceptUrl(rawToken);
  if (!link) {
    throw new Error('APP_URL is not configured; cannot build invitation link');
  }
  const greeting = name || email;
  await sendMail({
    to: email,
    subject: 'Activate your W4H account',
    text:
      `Hello ${greeting},\n\n` +
      `You have been invited to W4H` +
      (roleLabel ? ` as ${roleLabel}` : '') +
      `.\n\n` +
      `Set your password and activate your account using this link (expires in ${INVITE_TTL_HOURS} hours):\n` +
      `${link}\n\n` +
      `If you did not expect this invitation, you can ignore this email.\n`,
    html:
      `<p>Hello ${escapeHtml(greeting)},</p>` +
      `<p>You have been invited to <strong>W4H</strong>` +
      (roleLabel ? ` as <strong>${escapeHtml(roleLabel)}</strong>` : '') +
      `.</p>` +
      `<p><a href="${link}">Set your password and activate your account</a></p>` +
      `<p>This link expires in ${INVITE_TTL_HOURS} hours. If you did not expect this invitation, you can ignore this email.</p>`
  });
  return link;
}

export async function deliverInvitation({ email, name, roleLabel, userId, invitedBy }) {
  if (!appBaseUrl()) {
    return {
      inviteUrl: null,
      mailWarning: 'APP_URL is not configured. Set APP_URL so invitation links can be generated.',
    };
  }
  const rawToken = await createInvitation(userId, invitedBy);
  const inviteUrl = inviteAcceptUrl(rawToken);
  let mailWarning = null;
  if (isMailerConfigured()) {
    try {
      await sendInviteEmail({ email, name, roleLabel, rawToken });
    } catch (mailError) {
      mailWarning = mailError.message;
    }
  } else {
    mailWarning = 'SMTP is not configured; share the invitation link securely with the user.';
  }
  return { inviteUrl, mailWarning };
}
