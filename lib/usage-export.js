/**
 * Pseudonymous usage export for pilot evaluation (ADR-031 §7, D-19f).
 * Rows carry an HMAC pseudonym instead of the user id, the route template instead
 * of the raw path, and nothing else: no emails, names, dataset names or IDs.
 */
import crypto from 'crypto';
import { query } from '../db.js';

export const ROW_KEYS = [
  'pseudonym', 'team_role', 'event_type', 'timestamp', 'duration_ms',
  'status', 'method', 'route', 'row_count',
];

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

/** Stable per install, unlinkable across installs. `null` user stays `null`. */
export function pseudonym(installId, userId) {
  if (userId == null) return null;
  return crypto.createHmac('sha256', String(installId)).update(String(userId)).digest('hex').slice(0, 16);
}

/** A route is exportable only if it is a template: no emails, digits-only or UUID segments. */
export function safeRoute(route) {
  if (typeof route !== 'string' || !route.startsWith('/') || route.includes('@')) return null;
  const segments = route.split('/').filter(Boolean);
  if (segments.some((s) => /^\d+$/.test(s) || UUID_RE.test(s))) return null;
  return route.slice(0, 200);
}

const intOrNull = (v) => (v == null || v === '' || !Number.isFinite(Number(v)) ? null : Math.trunc(Number(v)));

/** Raw usage_events row (+ team_role) → allow-listed export row. Everything else is dropped. */
export function scrubRow(row, installId) {
  const m = row.metadata && typeof row.metadata === 'object' ? row.metadata : {};
  const ts = row.timestamp instanceof Date ? row.timestamp : new Date(row.timestamp);
  return {
    pseudonym: pseudonym(installId, row.user_id),
    team_role: row.team_role || null,
    event_type: String(row.event_type || '').slice(0, 100),
    timestamp: Number.isNaN(ts.getTime()) ? null : ts.toISOString(),
    duration_ms: intOrNull(row.duration_ms),
    status: intOrNull(m.status),
    method: typeof m.method === 'string' && /^[A-Z]{3,7}$/.test(m.method) ? m.method : null,
    route: safeRoute(m.route),
    row_count: intOrNull(m.row_count),
  };
}

function csvCell(v) {
  if (v == null) return '';
  const s = String(v);
  return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

/** CSV with `# key=value` meta lines, then a header row, then data rows. */
export function toCsv(meta, rows) {
  const head = Object.entries(meta).map(([k, v]) => `# ${k}=${v ?? ''}`);
  const lines = rows.map((r) => ROW_KEYS.map((k) => csvCell(r[k])).join(','));
  return [...head, ROW_KEYS.join(','), ...lines].join('\n') + '\n';
}

export function clampDays(value) {
  const n = Number.parseInt(value, 10);
  if (!Number.isFinite(n)) return 30;
  return Math.min(365, Math.max(1, n));
}

/**
 * Build the export for the last `days` days.
 * @returns {{ meta: object, rows: object[] }}
 */
export async function buildUsageExport({ days, installId, version, dbMode }) {
  const result = await query(
    `SELECT e.user_id, e.event_type, e.timestamp, e.duration_ms, e.metadata,
            CASE
              WHEN u.id IS NULL THEN NULL
              WHEN u.role = 'super_admin' THEN 'super_admin'
              WHEN EXISTS (SELECT 1 FROM w4h.team_members tm
                           WHERE tm.user_id = u.id AND tm.role = 'team_admin') THEN 'team_admin'
              WHEN EXISTS (SELECT 1 FROM w4h.team_members tm WHERE tm.user_id = u.id) THEN 'member'
              ELSE 'none'
            END AS team_role
       FROM w4h.usage_events e
       LEFT JOIN w4h.users u ON u.id = e.user_id
      WHERE e.timestamp > now() - ($1 || ' days')::interval
      ORDER BY e.timestamp, e.id`,
    [String(days)]
  );
  const end = new Date();
  const meta = {
    schema: 'w4h-usage-export/1',
    install_id: installId,
    version,
    db_mode: dbMode || null,
    window_start: new Date(end.getTime() - days * 86400000).toISOString(),
    window_end: end.toISOString(),
    row_count: result.rowCount,
  };
  return { meta, rows: result.rows.map((r) => scrubRow(r, installId)) };
}
