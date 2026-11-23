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

/** Requests that are polling or telemetry plumbing, not user work. */
const SKIPPED_REQUESTS = new Set([
  'GET /config/status',
  'POST /config/client-usage',
  'POST /users/token-valid',
  'GET /contributions/uploads/:uploadId/file',
]);

/**
 * Express route template for a finished request (`/datasets/:id/import/csv-batch`),
 * never the raw path, so usage rows carry no IDs. `null` when no route matched.
 */
export function routeTemplate(req) {
  const routePath = req?.route?.path;
  if (typeof routePath !== 'string') return null;
  const joined = `${req.baseUrl || ''}${routePath}`;
  return joined.length > 1 ? joined.replace(/\/$/, '') : joined;
}

/** False for polling/telemetry endpoints that would drown the real work in usage rows. */
export function shouldRecordRequest(req) {
  const route = routeTemplate(req);
  if (!route) return false;
  return !SKIPPED_REQUESTS.has(`${req.method} ${route}`);
}

/**
 * Best-effort per-request usage recording (D-18 / D-19*): API-key and JWT requests,
 * with server duration and the route template. Never throws to the request path.
 * Handlers may set `res.locals.usageRowCount` (CSV ingest) to add `row_count`.
 */
export function attachApiKeyUsageLogger(req, res, next) {
  const started = Date.now();
  res.on('finish', () => {
    if (!req.user?.id || !shouldRecordRequest(req)) return;

    const userId = req.user.id;
    // `authenticate` (JWT-only) does not set authVia; only API-key auth sets 'api_key'.
    const authMethod = req.authVia || (req.authViaProjectDbToken ? 'project_db_token' : 'jwt');
    const metadata = {
      method: req.method,
      route: routeTemplate(req),
      status: res.statusCode,
      auth_method: authMethod,
      ...(Number.isFinite(res.locals?.usageRowCount) ? { row_count: res.locals.usageRowCount } : {}),
    };

    Promise.all([
      authMethod === 'api_key'
        ? query(`UPDATE w4h.users SET api_key_last_used_at = now() WHERE id = $1`, [userId])
        : null,
      recordUsageEvent({
        userId,
        eventType: 'api.request',
        durationMs: Math.max(0, Date.now() - started),
        metadata,
      }),
    ]).catch((err) => {
      console.warn('API usage log failed:', err.message);
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
