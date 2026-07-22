import { query } from '../../db.js';

/**
 * Append-only HIPAA audit write (ADR-011).
 * Best-effort by default for non-PHI paths; callers that need fail-closed
 * should await and handle errors themselves.
 */
export async function writeAuditLog(req, {
  userId = null,
  action,
  resourceType = null,
  resourceId = null,
  details = null,
} = {}) {
  if (!action) return;
  await query(
    `INSERT INTO w4h.audit_log
       (user_id, action, resource_type, resource_id, ip_address, user_agent, details)
     VALUES ($1, $2, $3, $4, $5, $6, $7)`,
    [
      userId ?? req?.user?.id ?? null,
      action,
      resourceType,
      resourceId == null ? null : String(resourceId),
      req?.ip || null,
      req?.get?.('user-agent') || null,
      details ? JSON.stringify(details) : null,
    ]
  );
}

/** Fire-and-forget wrapper for non-critical audit paths (auth failures, UI views). */
export function writeAuditLogSafe(req, opts) {
  writeAuditLog(req, opts).catch((err) => {
    console.warn('Audit log write failed:', err.message);
  });
}
