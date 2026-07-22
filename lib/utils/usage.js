import { query } from '../../db.js';

const ALLOWED_CLIENT_EVENTS = new Set(['session.start', 'page.view']);

/**
 * Best-effort insert into usage_events (D-18 / D-19*).
 * Never throws to callers that use recordUsageEventSafe.
 */
export async function recordUsageEvent({
  userId = null,
  eventType,
  durationMs = null,
  sessionId = null,
  metadata = null,
} = {}) {
  if (!eventType) return;
  await query(
    `INSERT INTO w4h.usage_events (user_id, event_type, duration_ms, session_id, metadata)
     VALUES ($1, $2, $3, $4, $5::jsonb)`,
    [
      userId,
      eventType,
      durationMs,
      sessionId,
      metadata ? JSON.stringify(metadata) : null,
    ]
  );
}

export function recordUsageEventSafe(opts) {
  recordUsageEvent(opts).catch((err) => {
    console.warn('Usage event log failed:', err.message);
  });
}

/**
 * Best-effort API-key usage recording (D-18 / D-19*).
 * Never throws to the request path — usage logging must not break API access.
 */
export function attachApiKeyUsageLogger(req, res, next) {
  const started = Date.now();
  res.on('finish', () => {
    if (req.authVia !== 'api_key' || !req.user?.id) return;

    const userId = req.user.id;
    const durationMs = Math.max(0, Date.now() - started);
    const pathOnly = String(req.originalUrl || req.url || '').split('?')[0];
    const metadata = {
      method: req.method,
      path: pathOnly,
      status: res.statusCode,
      auth_method: 'api_key',
    };

    Promise.all([
      query(`UPDATE w4h.users SET api_key_last_used_at = now() WHERE id = $1`, [userId]),
      recordUsageEvent({
        userId,
        eventType: 'api.request',
        durationMs,
        metadata,
      }),
    ]).catch((err) => {
      console.warn('API key usage log failed:', err.message);
    });
  });
  next();
}

/**
 * Validate and normalize client-reported UI usage events (session / page views).
 */
export function normalizeClientUsageEvent(body = {}) {
  const eventType = String(body.event_type || '').trim();
  if (!ALLOWED_CLIENT_EVENTS.has(eventType)) {
    return { ok: false, error: `event_type must be one of: ${[...ALLOWED_CLIENT_EVENTS].join(', ')}` };
  }
  const path = String(body.path || '').trim().slice(0, 500) || null;
  const sessionId = String(body.session_id || '').trim().slice(0, 80) || null;
  const metadata = {
    path,
    auth_method: 'jwt',
    ...(body.route_name ? { route_name: String(body.route_name).slice(0, 120) } : {}),
  };
  return { ok: true, eventType, sessionId, metadata };
}

export { ALLOWED_CLIENT_EVENTS };
