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
  // contribution.run.* (ADR-031 §7): counts and fixed enums only
  'rows_skipped', 'signals', 'signals_planned', 'files', 'bytes', 'mode', 'runner', 'exit_code', 'failure',
];

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

/**
 * Secret per-install HMAC key. ADR-031 says HMAC(install_id, user_id), but install_id is
 * published in the export header and user ids are small integers, so keying on it alone
 * is reversible by brute force. Derive the key from TOKEN_SECRET (never exported, stable
 * across updates) bound to install_id.
 */
export function pseudonymKey(tokenSecret, installId) {
  if (!tokenSecret) throw new Error('TOKEN_SECRET is required for usage pseudonyms');
  return crypto.createHmac('sha256', String(tokenSecret)).update(`w4h-usage-pseudonym:${installId}`).digest();
}

/** Stable per install, unlinkable across installs. `null` user stays `null`. */
export function pseudonym(key, userId) {
  if (userId == null) return null;
  return crypto.createHmac('sha256', key).update(String(userId)).digest('hex').slice(0, 16);
}

/** A route is exportable only if it is a template: no emails, digits-only or UUID segments. */
export function safeRoute(route) {
  if (typeof route !== 'string' || !route.startsWith('/') || route.includes('@')) return null;
  const segments = route.split('/').filter(Boolean);
  if (segments.some((s) => /^\d+$/.test(s) || UUID_RE.test(s))) return null;
  return route.slice(0, 200);
}

const intOrNull = (v) => (v == null || v === '' || !Number.isFinite(Number(v)) ? null : Math.trunc(Number(v)));
const enumOrNull = (v) => (typeof v === 'string' && /^[a-z_]{1,32}$/.test(v) ? v : null);

/** Raw usage_events row (+ team_role) → allow-listed export row. Everything else is dropped. */
export function scrubRow(row, key) {
  const m = row.metadata && typeof row.metadata === 'object' ? row.metadata : {};
  const ts = row.timestamp instanceof Date ? row.timestamp : new Date(row.timestamp);
  return {
    pseudonym: pseudonym(key, row.user_id),
    team_role: row.team_role || null,
    event_type: String(row.event_type || '').slice(0, 100),
    timestamp: Number.isNaN(ts.getTime()) ? null : ts.toISOString(),
    duration_ms: intOrNull(row.duration_ms),
    status: intOrNull(m.status),
    method: typeof m.method === 'string' && /^[A-Z]{3,7}$/.test(m.method) ? m.method : null,
    route: safeRoute(m.route),
    row_count: intOrNull(m.row_count),
    rows_skipped: intOrNull(m.rows_skipped),
    signals: intOrNull(m.signals),
    signals_planned: intOrNull(m.signals_planned),
    files: intOrNull(m.files),
    bytes: intOrNull(m.bytes),
    mode: enumOrNull(m.mode),
    runner: enumOrNull(m.runner),
    exit_code: intOrNull(m.exit_code),
    failure: enumOrNull(m.failure),
  };
}

function csvCell(v) {
  if (v == null) return '';
  const s = String(v);
  return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

/** CSV with `# key=value` meta lines (nested objects flattened one level: `window_from=`), then a header row, then data rows. */
export function toCsv(meta, rows) {
  const flat = Object.entries(meta).flatMap(([k, v]) =>
    v && typeof v === 'object' ? Object.entries(v).map(([k2, v2]) => [`${k}_${k2}`, v2]) : [[k, v]]
  );
  const head = flat.map(([k, v]) => `# ${k}=${v ?? ''}`);
  const lines = rows.map((r) => ROW_KEYS.map((k) => csvCell(r[k])).join(','));
  return [...head, ROW_KEYS.join(','), ...lines].join('\n') + '\n';
}

export function clampDays(value) {
  const n = Number.parseInt(value, 10);
  if (!Number.isFinite(n)) return 30;
  return Math.min(365, Math.max(1, n));
}

/**
 * Export window end: a preview's `window.to` (so Send selects the same rows); default now.
 * An unparseable time, or one more than 365 days back, throws (status 400) instead of
 * silently exporting a different window. A slightly future end (clock skew) becomes now.
 */
export function clampWindowEnd(value, now = new Date()) {
  if (value == null || value === '') return now;
  const t = typeof value === 'string' ? new Date(value) : new Date(NaN);
  if (Number.isNaN(t.getTime()) || t.getTime() < now.getTime() - 365 * 86400000) {
    throw Object.assign(new Error('window_end must be an ISO time within the last 365 days'), { status: 400 });
  }
  return t.getTime() > now.getTime() ? now : t;
}

/**
 * Build the export for the `days` days ending at `end`. `teamIds` (Admin scope) limits rows
 * to events by members of those teams; omitted = whole platform (Super admin).
 * @returns {{ meta: object, columns: string[], rows: object[] }}
 */
export async function buildUsageExport({ days, end = new Date(), installId, version, dbMode, tokenSecret, teamIds }) {
  const key = pseudonymKey(tokenSecret, installId);
  const from = new Date(end.getTime() - days * 86400000);
  const params = [from.toISOString(), end.toISOString()];
  let teamFilter = '';
  if (teamIds) {
    params.push(teamIds);
    teamFilter = `AND e.user_id IN (SELECT tm.user_id FROM w4h.team_members tm WHERE tm.team_id = ANY($3::uuid[]))`;
  }
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
      WHERE e.timestamp > $1 AND e.timestamp <= $2 ${teamFilter}
      ORDER BY e.timestamp, e.id`,
    params
  );
  const meta = {
    schema: 'w4h-usage-export/1',
    install_id: installId,
    version,
    db_mode: dbMode || null,
    scope: teamIds ? 'teams' : 'platform',
    window: { days, from: params[0], to: params[1] },
    row_count: result.rowCount,
  };
  return { meta, columns: ROW_KEYS, rows: result.rows.map((r) => scrubRow(r, key)) };
}
