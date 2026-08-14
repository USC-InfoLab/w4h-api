import express from 'express';
const router = express.Router();
import fs from 'fs-extra';
import path from 'path';
import { fileURLToPath } from 'url';
import { isSuperAdmin, canManageTeamAsync, isTeamAdmin } from '../lib/utils/user.js';
import crypto from 'crypto';
import pkg from 'pg';
const { Pool } = pkg;
import db, { tableExists, USERS_TABLE_NAME, query } from '../db.js';
import * as auth from '../auth.js';
import { initializeDatabase, createInitialAdmin, hasAnyUsers } from '../lib/db-init.js';
import { getMailerConfig, isMailerConfigured, sendMail } from '../lib/mailer.js';
import { normalizeClientUsageEvent, recordUsageEventSafe } from '../lib/utils/usage.js';
import { writeAuditLog, writeAuditLogSafe } from '../lib/utils/audit.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const isAdmin = user => !!user && isSuperAdmin(user.role);

// Keep `.env` formatting consistent with `.env.example` by rewriting `.env`
// from the template (preserves ordering + comments).
const ENV_PATH = path.join(__dirname, '..', '.env');
const ENV_EXAMPLE_PATH = path.join(__dirname, '..', '.env.example');

function parseEnvValues(envContent = '') {
  const values = {};
  const lines = String(envContent).split(/\r?\n/);

  for (const line of lines) {
    // Match `KEY=value` only (non-comment lines)
    const m = /^\s*([A-Z0-9_]+)\s*=(.*)\s*$/.exec(line);
    if (!m) continue;
    values[m[1]] = m[2];
  }

  return values;
}

function extractTemplateKeys(templateContent) {
  const keys = [];
  const lines = String(templateContent).split(/\r?\n/);
  const keyLineRe = /^\s*([A-Z0-9_]+)\s*=/;

  for (const line of lines) {
    const m = keyLineRe.exec(line);
    if (m) keys.push(m[1]);
  }
  return keys;
}

function applyValuesToTemplate(templateContent, mergedValues, templateKeys) {
  let out = String(templateContent);

  for (const key of templateKeys) {
    if (!Object.prototype.hasOwnProperty.call(mergedValues, key)) continue;
    const value = mergedValues[key] ?? '';
    const lineRe = new RegExp(`^${key}=.*$`, 'm');
    out = out.replace(lineRe, `${key}=${value}`);
  }

  return out;
}

async function writeEnvValues(keyValues) {
  if (!(await fs.pathExists(ENV_EXAMPLE_PATH))) {
    throw new Error(`Missing ${path.basename(ENV_EXAMPLE_PATH)} — cannot preserve formatting`);
  }

  const template = await fs.readFile(ENV_EXAMPLE_PATH, 'utf8');
  const templateKeys = extractTemplateKeys(template);

  const existingValues = (await fs.pathExists(ENV_PATH))
    ? parseEnvValues(await fs.readFile(ENV_PATH, 'utf8'))
    : {};

  const mergedValues = { ...existingValues, ...keyValues };

  // Rewrite `.env` from `.env.example` to guarantee stable ordering/comments
  let out = applyValuesToTemplate(template, mergedValues, templateKeys);

  // If callers pass keys not present in the template, append them.
  const missingKeys = Object.keys(keyValues).filter(k => !templateKeys.includes(k));
  if (missingKeys.length) {
    const extra = missingKeys.map(k => `${k}=${keyValues[k] ?? ''}`).join('\n');
    out = out.trimEnd() + `\n${extra}\n`;
  }

  await fs.writeFile(ENV_PATH, out.trimEnd() + '\n');

  // Keep runtime env in sync for keys we just changed
  Object.keys(keyValues).forEach((key) => {
    process.env[key] = String(keyValues[key] ?? '');
  });
}

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

router.get('/status', getStatus);
router.get('/admin-home', auth.authenticate, auth.requirePlatformAdmin, getAdminHome);
router.get('/team-home', auth.authenticate, auth.requireTeamAdmin, getTeamHome);
router.get('/usage', auth.authenticate, auth.requirePlatformAdmin, getUsage);
router.get('/api-usage', auth.authenticate, auth.requirePlatformAdmin, getUsage); // legacy alias
router.post('/client-usage', auth.authenticate, postClientUsage);
router.post('/report-issue', auth.authenticate, postReportIssue);
router.get('/audit-log', auth.authenticate, getAuditLog);
router.get('/database-settings', auth.authenticate, auth.requirePlatformAdmin, getDatabaseSettings);
router.get('/mailer-settings', auth.authenticate, auth.requirePlatformAdmin, getMailerSettings);
router.post('/mailer-settings', auth.authenticate, auth.requirePlatformAdmin, postMailerSettings);
router.post('/mailer-test', auth.authenticate, auth.requirePlatformAdmin, postMailerTest);
router.get('/app-settings', auth.authenticate, auth.requirePlatformAdmin, getAppSettings);
router.post('/app-settings', auth.authenticate, auth.requirePlatformAdmin, postAppSettings);
router.post('/init-database', postInitDatabase);
router.post('/admin-setup', postAdminSetup);
router.get('/test', auth.authenticate, auth.requirePlatformAdmin, getTest);
router.post('/reset', auth.authenticate, auth.requirePlatformAdmin, postReset);

export default router;

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/** GET /config/status — Check if database is configured. */
async function getStatus(req, res) {
  const envConfigured = !!(
    process.env.DB_HOST &&
    process.env.DB_PORT &&
    process.env.DB_DATABASE &&
    process.env.DB_USER &&
    process.env.DB_PASSWORD
  );

  const details = {
    host: process.env.DB_HOST || '',
    port: process.env.DB_PORT || '5432',
    database: process.env.DB_DATABASE || '',
    user: process.env.DB_USER || ''
  };

  if (!envConfigured) {
    return res.json({ configured: false, details });
  }

  // Env vars are set — verify the DB is actually reachable and schema is initialized
  try {
    const initialized = await tableExists(USERS_TABLE_NAME);
    if (!initialized) {
      return res.json({ configured: false, details });
    }
    return res.json({ configured: true, details });
  } catch {
    return res.json({ configured: false, details });
  }
}

/**
 * GET /config/admin-home — Super admin Platform Admin home snapshot (MVP §15.1).
 * KPIs, health, usage, and attention items. Full ledgers stay on dedicated pages.
 */
async function getAdminHome(req, res) {
  if (!isAdmin(req.user)) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  try {
    const [
      teamsRes,
      usersRes,
      projectsRes,
      datasetsRes,
      sharesRes,
      invitesRes,
      orphanDatasetsRes,
      teamsWithoutAdminRes,
      activeKeysRes,
      failedLoginsRes,
      breakGlassRes,
      auditProbe,
      usage24hRes,
      usage7dRes,
      docsStatsRes,
    ] = await Promise.all([
      query(`SELECT COUNT(*)::int AS n FROM w4h.teams WHERE status = 'active'`).catch(() => ({ rows: [{ n: 0 }] })),
      query(`
        SELECT
          COUNT(*) FILTER (WHERE status = 'active')::int AS active,
          COUNT(*) FILTER (WHERE status = 'invited')::int AS invited,
          COUNT(*) FILTER (WHERE status = 'discontinued')::int AS discontinued,
          COUNT(*)::int AS total
        FROM w4h.users
      `).catch(() => ({ rows: [{ active: 0, invited: 0, discontinued: 0, total: 0 }] })),
      query(`SELECT COUNT(*)::int AS n FROM w4h.projects WHERE status = 'active'`).catch(() => ({ rows: [{ n: 0 }] })),
      query(`SELECT COUNT(*)::int AS n FROM w4h.datasets`).catch(() => ({ rows: [{ n: 0 }] })),
      query(`SELECT COUNT(*)::int AS n FROM w4h.dataset_shares`).catch(() => ({ rows: [{ n: 0 }] })),
      query(`
        SELECT COUNT(*)::int AS n
        FROM w4h.users
        WHERE status = 'invited'
      `).catch(() => ({ rows: [{ n: 0 }] })),
      query(`
        SELECT COUNT(*)::int AS n
        FROM w4h.datasets
        WHERE team_id IS NULL
      `).catch(() => ({ rows: [{ n: 0 }] })),
      query(`
        SELECT COUNT(*)::int AS n
        FROM w4h.teams t
        WHERE t.status = 'active'
          AND NOT EXISTS (
            SELECT 1
            FROM w4h.team_members tm
            JOIN w4h.users u ON u.id = tm.user_id
            WHERE tm.team_id = t.id
              AND (
                tm.role = 'team_admin'
                OR u.role IN ('team_admin', 'admin')
              )
              AND COALESCE(u.status, 'active') = 'active'
          )
      `).catch(() => ({ rows: [{ n: 0 }] })),
      query(`
        SELECT COUNT(*)::int AS n
        FROM w4h.users
        WHERE api_key_hash IS NOT NULL
          AND COALESCE(status, 'active') = 'active'
      `).catch(() => ({ rows: [{ n: 0 }] })),
      query(`
        SELECT COUNT(*)::int AS n
        FROM w4h.audit_log
        WHERE timestamp > now() - interval '24 hours'
          AND (
            action ILIKE '%login%fail%'
            OR action IN ('user.login_code_failed', 'user.login_failed')
          )
      `).catch(() => ({ rows: [{ n: 0 }] })),
      query(`
        SELECT COUNT(*)::int AS n
        FROM w4h.audit_log
        WHERE timestamp > now() - interval '30 days'
          AND (
            action ILIKE '%break_glass%'
            OR action ILIKE '%BREAK_GLASS%'
          )
      `).catch(() => ({ rows: [{ n: 0 }] })),
      (async () => {
        try {
          await query(`SELECT 1 FROM w4h.audit_log LIMIT 1`);
          return { ok: true, message: 'Audit log readable' };
        } catch (e) {
          if (e?.cause?.code === '42P01' || String(e.message || '').includes('does not exist')) {
            return { ok: false, message: 'Audit log table missing' };
          }
          return { ok: false, message: e.message || 'Audit log probe failed' };
        }
      })(),
      query(`
        SELECT COUNT(*)::int AS n
        FROM w4h.usage_events
        WHERE timestamp > now() - interval '24 hours'
      `).catch(() => ({ rows: [{ n: 0 }] })),
      query(`
        SELECT COUNT(*)::int AS n
        FROM w4h.usage_events
        WHERE timestamp > now() - interval '7 days'
      `).catch(() => ({ rows: [{ n: 0 }] })),
      query(`
        SELECT
          (SELECT COUNT(*)::int FROM w4h.docs) AS platform_docs,
          (SELECT COUNT(*)::int FROM w4h.docs WHERE COALESCE(is_feature_card, false)) AS feature_cards,
          (SELECT COUNT(*)::int FROM w4h.project_docs) AS project_doc_links,
          (SELECT COUNT(DISTINCT project_id)::int FROM w4h.project_docs) AS projects_with_linked_docs,
          (
            SELECT COUNT(*)::int
            FROM w4h.projects p
            WHERE p.status = 'active'
              AND NOT EXISTS (
                SELECT 1 FROM w4h.project_docs pd WHERE pd.project_id = p.id
              )
          ) AS projects_without_linked_docs
      `).catch(() => ({
        rows: [{
          platform_docs: 0,
          feature_cards: 0,
          project_doc_links: 0,
          projects_with_linked_docs: 0,
          projects_without_linked_docs: 0,
        }],
      })),
    ]);

    const users = usersRes.rows[0] || {};
    let dbOk = false;
    let dbMessage = 'Database unreachable';
    try {
      await query('SELECT 1');
      dbOk = true;
      dbMessage = 'Connected';
    } catch (e) {
      dbMessage = e.message || 'Database error';
    }

    const smtpConfigured = isMailerConfigured();

    res.json({
      kpis: {
        teams: teamsRes.rows[0]?.n ?? 0,
        users_active: users.active ?? 0,
        users_invited: users.invited ?? 0,
        users_discontinued: users.discontinued ?? 0,
        users_total: users.total ?? 0,
        projects: projectsRes.rows[0]?.n ?? 0,
        datasets: datasetsRes.rows[0]?.n ?? 0,
        active_shares: sharesRes.rows[0]?.n ?? 0,
        // Until Doc-08 private pages ship, surface legacy project↔doc links as the private-pages KPI.
        private_pages: docsStatsRes.rows[0]?.project_doc_links ?? 0,
        private_pages_ready: false,
      },
      health: {
        database: { ok: dbOk, message: dbMessage },
        smtp: {
          ok: smtpConfigured,
          message: smtpConfigured ? 'Configured' : 'Not configured — invitations and login codes need SMTP',
        },
        audit_log: auditProbe,
      },
      usage: {
        events_24h: usage24hRes.rows[0]?.n ?? 0,
        events_7d: usage7dRes.rows[0]?.n ?? 0,
        active_api_keys: activeKeysRes.rows[0]?.n ?? 0,
        note: 'Thin snapshot only — full UI vs API adoption is on Usage.',
      },
      attention: {
        invites_pending: invitesRes.rows[0]?.n ?? 0,
        datasets_without_team: orphanDatasetsRes.rows[0]?.n ?? 0,
        teams_without_admin: teamsWithoutAdminRes.rows[0]?.n ?? 0,
        projects_without_docs: docsStatsRes.rows[0]?.projects_without_linked_docs ?? 0,
        failed_logins_24h: failedLoginsRes.rows[0]?.n ?? 0,
        break_glass_30d: breakGlassRes.rows[0]?.n ?? 0,
        backup: {
          status: 'unknown',
          message:
            'No automated backup probe yet. Confirm Postgres backups are scheduled and tested for HIPAA retention.',
        },
      },
      links: {
        hipaa_checklist_slug: 'hipaa-compliance',
        audit_log: '/admin/audit-log',
        usage: '/admin/usage',
        api_usage: '/admin/usage', // legacy alias key
      },
    });
  } catch (error) {
    console.error('getAdminHome error:', error);
    res.status(500).json({ error: error.message || 'Failed to load admin home snapshot' });
  }
}

/**
 * Teams where the caller is Team admin (membership role), or all memberships
 * when legacy users.role = team_admin.
 */
async function getAdministeredTeamIds(user) {
  if (isSuperAdmin(user.role)) {
    const all = await query(`SELECT id::text AS id FROM w4h.teams WHERE status = 'active'`);
    return all.rows.map((r) => r.id);
  }

  if (isTeamAdmin(user.role)) {
    const legacy = await query(
      `SELECT team_id::text AS id FROM w4h.team_members WHERE user_id = $1`,
      [user.id]
    );
    if (legacy.rowCount) return legacy.rows.map((r) => r.id);
  }

  const res = await query(
    `SELECT team_id::text AS id
     FROM w4h.team_members
     WHERE user_id = $1 AND role = 'team_admin'`,
    [user.id]
  );
  return res.rows.map((r) => r.id);
}

/**
 * GET /config/team-home — Team admin operator home (MVP §15.1, normal app shell).
 * Scoped to teams the caller administers.
 */
async function getTeamHome(req, res) {
  if (!(await canManageTeamAsync(req.user))) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  try {
    const teamIds = await getAdministeredTeamIds(req.user);
    if (!teamIds.length) {
      return res.json({
        teams: [],
        kpis: {
          teams: 0,
          members: 0,
          members_invited: 0,
          projects: 0,
          datasets: 0,
          shares_out: 0,
          shares_in: 0,
          projects_without_docs: 0,
        },
        attention: {
          invites_pending: 0,
          projects_without_docs: 0,
        },
        recent_projects: [],
        links: { hipaa_checklist_slug: 'hipaa-compliance' },
        note: 'You are not Team admin on any team yet. Ask a Super admin to assign you.',
      });
    }

    const [
      teamsRes,
      membersRes,
      projectsRes,
      datasetsRes,
      sharesOutRes,
      sharesInRes,
      invitesRes,
      docsGapRes,
      recentProjectsRes,
    ] = await Promise.all([
      query(
        `SELECT id::text AS id, name, slug, status
         FROM w4h.teams
         WHERE id = ANY($1::uuid[])
         ORDER BY name ASC`,
        [teamIds]
      ),
      query(
        `SELECT
           COUNT(DISTINCT tm.user_id)::int AS members,
           COUNT(DISTINCT tm.user_id) FILTER (WHERE u.status = 'invited')::int AS invited
         FROM w4h.team_members tm
         JOIN w4h.users u ON u.id = tm.user_id
         WHERE tm.team_id = ANY($1::uuid[])`,
        [teamIds]
      ).catch(() => ({ rows: [{ members: 0, invited: 0 }] })),
      query(
        `SELECT COUNT(*)::int AS n
         FROM w4h.projects
         WHERE team_id = ANY($1::uuid[]) AND status = 'active'`,
        [teamIds]
      ).catch(() => ({ rows: [{ n: 0 }] })),
      query(
        `SELECT COUNT(*)::int AS n
         FROM w4h.datasets
         WHERE team_id = ANY($1::uuid[])`,
        [teamIds]
      ).catch(() => ({ rows: [{ n: 0 }] })),
      query(
        `SELECT COUNT(*)::int AS n
         FROM w4h.dataset_shares ds
         JOIN w4h.datasets d ON d.id = ds.dataset_id
         WHERE d.team_id = ANY($1::uuid[])`,
        [teamIds]
      ).catch(() => ({ rows: [{ n: 0 }] })),
      query(
        `SELECT COUNT(*)::int AS n
         FROM w4h.dataset_shares
         WHERE target_team_id = ANY($1::uuid[])`,
        [teamIds]
      ).catch(() => ({ rows: [{ n: 0 }] })),
      query(
        `SELECT COUNT(DISTINCT u.id)::int AS n
         FROM w4h.users u
         JOIN w4h.team_members tm ON tm.user_id = u.id
         WHERE tm.team_id = ANY($1::uuid[])
           AND u.status = 'invited'`,
        [teamIds]
      ).catch(() => ({ rows: [{ n: 0 }] })),
      query(
        `SELECT COUNT(*)::int AS n
         FROM w4h.projects p
         WHERE p.team_id = ANY($1::uuid[])
           AND p.status = 'active'
           AND NOT EXISTS (
             SELECT 1 FROM w4h.project_docs pd WHERE pd.project_id = p.id
           )`,
        [teamIds]
      ).catch(() => ({ rows: [{ n: 0 }] })),
      query(
        `SELECT p.id::text AS id, p.name, p.status, t.name AS team_name
         FROM w4h.projects p
         LEFT JOIN w4h.teams t ON t.id = p.team_id
         WHERE p.team_id = ANY($1::uuid[])
         ORDER BY p.updated_at DESC NULLS LAST, p.name ASC
         LIMIT 8`,
        [teamIds]
      ).catch(() => ({ rows: [] })),
    ]);

    const members = membersRes.rows[0] || {};
    res.json({
      teams: teamsRes.rows,
      kpis: {
        teams: teamsRes.rows.length,
        members: members.members ?? 0,
        members_invited: members.invited ?? 0,
        projects: projectsRes.rows[0]?.n ?? 0,
        datasets: datasetsRes.rows[0]?.n ?? 0,
        shares_out: sharesOutRes.rows[0]?.n ?? 0,
        shares_in: sharesInRes.rows[0]?.n ?? 0,
        projects_without_docs: docsGapRes.rows[0]?.n ?? 0,
      },
      attention: {
        invites_pending: invitesRes.rows[0]?.n ?? 0,
        projects_without_docs: docsGapRes.rows[0]?.n ?? 0,
      },
      recent_projects: recentProjectsRes.rows,
      irb: {
        available: false,
        message: 'IRB access reports for your team’s datasets and projects are coming soon (H-09a).',
      },
      links: {
        hipaa_checklist_slug: 'hipaa-compliance',
      },
    });
  } catch (error) {
    console.error('getTeamHome error:', error);
    res.status(500).json({ error: error.message || 'Failed to load team home snapshot' });
  }
}

/**
 * POST /config/client-usage — JWT UI session / page-view events for study adoption (D-19e).
 * Body: { event_type: 'session.start'|'page.view', path?, route_name?, session_id? }
 */
async function postClientUsage(req, res) {
  try {
    const parsed = normalizeClientUsageEvent(req.body || {});
    if (!parsed.ok) {
      return res.status(400).json({ error: parsed.error });
    }
    recordUsageEventSafe({
      userId: req.user?.id ?? null,
      eventType: parsed.eventType,
      sessionId: parsed.sessionId,
      metadata: parsed.metadata,
    });
    res.json({ ok: true });
  } catch (error) {
    console.error('postClientUsage error:', error);
    res.status(500).json({ error: error.message || 'Failed to record usage' });
  }
}

/**
 * POST /config/report-issue — unprompted feedback emailed to HELP_EMAIL (or SMTP from).
 * Body: { subject, message, page_url? }
 */
async function postReportIssue(req, res) {
  try {
    if (!isMailerConfigured()) {
      return res.status(503).json({
        error: 'Email is not configured. Ask a Super admin to set SMTP under Settings.',
      });
    }

    const subject = String(req.body?.subject || '').trim().slice(0, 200);
    const message = String(req.body?.message || '').trim().slice(0, 8000);
    const pageUrl = String(req.body?.page_url || '').trim().slice(0, 500);

    if (!subject || !message) {
      return res.status(400).json({ error: 'Subject and message are required.' });
    }

    const to = (process.env.HELP_EMAIL || '').trim() || getMailerConfig().from;
    if (!to) {
      return res.status(503).json({
        error: 'No HELP_EMAIL configured. Ask a Super admin to set it under Settings.',
      });
    }

    const reporter = req.user || {};
    const text = [
      `Report from: ${reporter.name || '—'} <${reporter.email || 'unknown'}>`,
      `User id: ${reporter.id ?? '—'}`,
      `Role: ${reporter.role || '—'}`,
      pageUrl ? `Page: ${pageUrl}` : null,
      '',
      message,
    ]
      .filter((line) => line != null)
      .join('\n');

    await sendMail({
      to,
      subject: `[W4H feedback] ${subject}`,
      text,
      html: `<pre style="font-family: system-ui, sans-serif; white-space: pre-wrap;">${escapeHtml(text)}</pre>`,
    });

    recordUsageEventSafe({
      userId: reporter.id ?? null,
      eventType: 'feedback.report',
      metadata: { subject, page_url: pageUrl || null },
    });

    res.json({ ok: true, message: 'Thanks — your report was sent to the platform administrators.' });
  } catch (error) {
    console.error('postReportIssue error:', error);
    res.status(500).json({ error: error.message || 'Failed to send report' });
  }
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/**
 * GET /config/usage — Super admin platform usage & SC-CTSI study metrics (D-18 / D-19*).
 * Legacy alias: GET /config/api-usage
 * Query: days (1–90, default 14), limit (recent events, default 100).
 */
async function getUsage(req, res) {
  if (!isAdmin(req.user)) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  try {
    const days = Math.min(90, Math.max(1, Number.parseInt(String(req.query.days || '14'), 10) || 14));
    const limit = Math.min(500, Math.max(10, Number.parseInt(String(req.query.limit || '100'), 10) || 100));
    const daysParam = String(days);

    const [
      summaryRes,
      dailyRes,
      byTypeRes,
      byRoleRes,
      statusRes,
      topUsersRes,
      keysRes,
      recentRes,
      neverUsedRes,
      knowledgeRes,
      surveyRes,
    ] = await Promise.all([
      query(
        `
        SELECT
          (SELECT COUNT(*)::int FROM w4h.usage_events
            WHERE timestamp > now() - interval '24 hours') AS events_24h,
          (SELECT COUNT(*)::int FROM w4h.usage_events
            WHERE timestamp > now() - ($1 || ' days')::interval) AS events_period,
          (SELECT COUNT(DISTINCT user_id)::int FROM w4h.usage_events
            WHERE timestamp > now() - ($1 || ' days')::interval
              AND user_id IS NOT NULL) AS unique_users_period,
          (SELECT COUNT(DISTINCT user_id)::int FROM w4h.usage_events
            WHERE timestamp > now() - ($1 || ' days')::interval
              AND user_id IS NOT NULL
              AND event_type = 'api.request') AS unique_users_api,
          (SELECT COUNT(DISTINCT user_id)::int FROM w4h.usage_events
            WHERE timestamp > now() - ($1 || ' days')::interval
              AND user_id IS NOT NULL
              AND event_type IN ('user.login', 'session.start', 'page.view')) AS unique_users_ui,
          (SELECT COUNT(*)::int FROM w4h.usage_events
            WHERE timestamp > now() - ($1 || ' days')::interval
              AND event_type = 'user.login') AS logins_period,
          (SELECT COUNT(*)::int FROM w4h.usage_events
            WHERE timestamp > now() - ($1 || ' days')::interval
              AND event_type = 'page.view') AS page_views_period,
          (SELECT COUNT(*)::int FROM w4h.usage_events
            WHERE timestamp > now() - ($1 || ' days')::interval
              AND event_type = 'session.start') AS sessions_period,
          (SELECT COUNT(*)::int FROM w4h.users
            WHERE api_key_hash IS NOT NULL AND COALESCE(status, 'active') = 'active') AS active_keys,
          (SELECT COUNT(*)::int FROM w4h.users
            WHERE api_key_hash IS NOT NULL
              AND COALESCE(status, 'active') = 'active'
              AND api_key_last_used_at > now() - interval '7 days') AS keys_used_7d,
          (SELECT COALESCE(AVG(duration_ms), 0)::int FROM w4h.usage_events
            WHERE timestamp > now() - ($1 || ' days')::interval
              AND duration_ms IS NOT NULL
              AND event_type = 'api.request') AS avg_duration_ms,
          (SELECT COALESCE(
              PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY duration_ms), 0
            )::int FROM w4h.usage_events
            WHERE timestamp > now() - ($1 || ' days')::interval
              AND duration_ms IS NOT NULL
              AND event_type = 'api.request') AS duration_p50_ms,
          (SELECT COALESCE(
              PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms), 0
            )::int FROM w4h.usage_events
            WHERE timestamp > now() - ($1 || ' days')::interval
              AND duration_ms IS NOT NULL
              AND event_type = 'api.request') AS duration_p95_ms
        `,
        [daysParam]
      ),
      query(
        `
        SELECT d::date AS day, COUNT(e.id)::int AS count
        FROM generate_series(
          CURRENT_DATE - (($1::int - 1) * INTERVAL '1 day'),
          CURRENT_DATE,
          INTERVAL '1 day'
        ) AS d
        LEFT JOIN w4h.usage_events e
          ON e.timestamp >= d AND e.timestamp < d + INTERVAL '1 day'
        GROUP BY d
        ORDER BY d ASC
        `,
        [days]
      ),
      query(
        `
        SELECT event_type, COUNT(*)::int AS count
        FROM w4h.usage_events
        WHERE timestamp > now() - ($1 || ' days')::interval
        GROUP BY event_type
        ORDER BY count DESC
        LIMIT 15
        `,
        [daysParam]
      ),
      query(
        `
        SELECT COALESCE(u.role, 'unknown') AS role, COUNT(e.id)::int AS count
        FROM w4h.usage_events e
        LEFT JOIN w4h.users u ON u.id = e.user_id
        WHERE e.timestamp > now() - ($1 || ' days')::interval
        GROUP BY COALESCE(u.role, 'unknown')
        ORDER BY count DESC
        `,
        [daysParam]
      ),
      query(
        `
        SELECT
          COUNT(*) FILTER (
            WHERE (metadata->>'status') ~ '^[0-9]+$'
              AND (metadata->>'status')::int >= 200
              AND (metadata->>'status')::int < 400
          )::int AS success_2xx_3xx,
          COUNT(*) FILTER (
            WHERE (metadata->>'status') ~ '^[0-9]+$'
              AND (metadata->>'status')::int >= 400
              AND (metadata->>'status')::int < 500
          )::int AS error_4xx,
          COUNT(*) FILTER (
            WHERE (metadata->>'status') ~ '^[0-9]+$'
              AND (metadata->>'status')::int >= 500
          )::int AS error_5xx,
          COUNT(*) FILTER (
            WHERE metadata->>'status' IS NULL
              OR metadata->>'status' = ''
              OR NOT ((metadata->>'status') ~ '^[0-9]+$')
          )::int AS status_unknown,
          COUNT(*)::int AS total
        FROM w4h.usage_events
        WHERE timestamp > now() - ($1 || ' days')::interval
          AND event_type = 'api.request'
        `,
        [daysParam]
      ),
      query(
        `
        SELECT
          u.id AS user_id,
          u.name,
          u.email,
          u.role,
          COUNT(e.id)::int AS event_count
        FROM w4h.usage_events e
        JOIN w4h.users u ON u.id = e.user_id
        WHERE e.timestamp > now() - ($1 || ' days')::interval
        GROUP BY u.id, u.name, u.email, u.role
        ORDER BY event_count DESC
        LIMIT 20
        `,
        [daysParam]
      ),
      query(
        `
        SELECT
          u.id AS user_id,
          u.name,
          u.email,
          u.role,
          u.status,
          u.api_key_prefix,
          u.api_key_created_at,
          u.api_key_last_used_at,
          (u.api_key_hash IS NOT NULL) AS active,
          COALESCE((
            SELECT STRING_AGG(t.name, ', ' ORDER BY t.name)
            FROM w4h.team_members tm
            JOIN w4h.teams t ON t.id = tm.team_id
            WHERE tm.user_id = u.id
          ), '') AS teams,
          (
            SELECT COUNT(*)::int
            FROM w4h.usage_events e
            WHERE e.user_id = u.id
              AND e.timestamp > now() - ($1 || ' days')::interval
          ) AS events_period
        FROM w4h.users u
        WHERE u.api_key_hash IS NOT NULL OR u.api_key_prefix IS NOT NULL
        ORDER BY u.api_key_last_used_at DESC NULLS LAST, u.api_key_created_at DESC NULLS LAST
        `,
        [daysParam]
      ),
      query(
        `
        SELECT
          e.id,
          e.timestamp,
          e.event_type,
          e.duration_ms,
          e.metadata,
          u.id AS user_id,
          u.name AS user_name,
          u.email AS user_email,
          u.api_key_prefix
        FROM w4h.usage_events e
        LEFT JOIN w4h.users u ON u.id = e.user_id
        ORDER BY e.timestamp DESC
        LIMIT $1
        `,
        [limit]
      ),
      query(
        `
        SELECT COUNT(*)::int AS n
        FROM w4h.users
        WHERE api_key_hash IS NOT NULL
          AND COALESCE(status, 'active') = 'active'
          AND api_key_last_used_at IS NULL
        `
      ),
      query(
        `
        SELECT
          (SELECT COUNT(*)::int FROM w4h.users
            WHERE COALESCE(status, 'active') = 'invited') AS invites_pending,
          (SELECT COUNT(*)::int FROM w4h.users
            WHERE api_key_hash IS NOT NULL
              AND COALESCE(status, 'active') = 'active') AS users_with_key,
          (SELECT COUNT(*)::int FROM w4h.users
            WHERE api_key_hash IS NOT NULL
              AND api_key_last_used_at IS NOT NULL
              AND COALESCE(status, 'active') = 'active') AS users_key_used,
          (SELECT COALESCE(
              PERCENTILE_CONT(0.5) WITHIN GROUP (
                ORDER BY EXTRACT(EPOCH FROM (api_key_last_used_at - api_key_created_at)) / 3600.0
              ), 0
            )::numeric(10,1)
            FROM w4h.users
            WHERE api_key_hash IS NOT NULL
              AND api_key_created_at IS NOT NULL
              AND api_key_last_used_at IS NOT NULL
              AND api_key_last_used_at >= api_key_created_at) AS median_hours_key_to_first_use
        `
      ),
      query(
        `
        SELECT
          COUNT(*)::int AS responses,
          COUNT(*) FILTER (WHERE survey_type ILIKE '%sus%' OR survey_type ILIKE '%usability%')::int AS usability_responses,
          MAX(timestamp) AS last_response_at
        FROM w4h.survey_responses
        `
      ).catch(() => ({ rows: [{ responses: 0, usability_responses: 0, last_response_at: null }] })),
    ]);

    const summary = summaryRes.rows[0] || {};
    const status = statusRes.rows[0] || {};
    const knowledge = knowledgeRes.rows[0] || {};
    const survey = surveyRes.rows[0] || {};
    const daily = dailyRes.rows.map((r) => ({
      day: r.day instanceof Date ? r.day.toISOString().slice(0, 10) : String(r.day).slice(0, 10),
      count: r.count,
    }));
    const maxDaily = daily.reduce((m, d) => Math.max(m, d.count), 0);
    const totalStatus = status.total ?? 0;
    const successCount = status.success_2xx_3xx ?? 0;
    const error4xx = status.error_4xx ?? 0;
    const error5xx = status.error_5xx ?? 0;

    res.json({
      period_days: days,
      summary: {
        events_24h: summary.events_24h ?? 0,
        events_period: summary.events_period ?? 0,
        unique_users_period: summary.unique_users_period ?? 0,
        unique_users_api: summary.unique_users_api ?? 0,
        unique_users_ui: summary.unique_users_ui ?? 0,
        logins_period: summary.logins_period ?? 0,
        page_views_period: summary.page_views_period ?? 0,
        sessions_period: summary.sessions_period ?? 0,
        active_keys: summary.active_keys ?? 0,
        keys_used_7d: summary.keys_used_7d ?? 0,
        keys_never_used: neverUsedRes.rows[0]?.n ?? 0,
        avg_duration_ms: summary.avg_duration_ms ?? 0,
        duration_p50_ms: summary.duration_p50_ms ?? 0,
        duration_p95_ms: summary.duration_p95_ms ?? 0,
        success_count: successCount,
        error_4xx: error4xx,
        error_5xx: error5xx,
        status_unknown: status.status_unknown ?? 0,
        error_rate_pct:
          totalStatus > 0
            ? Math.round(((error4xx + error5xx) / totalStatus) * 1000) / 10
            : 0,
      },
      daily,
      daily_max: maxDaily,
      by_event_type: byTypeRes.rows,
      by_role: byRoleRes.rows,
      top_users: topUsersRes.rows,
      api_keys: keysRes.rows.map((k) => ({
        user_id: k.user_id,
        name: k.name,
        email: k.email,
        role: k.role,
        status: k.status,
        api_key_prefix: k.api_key_prefix,
        api_key_created_at: k.api_key_created_at,
        api_key_last_used_at: k.api_key_last_used_at,
        active: !!k.active,
        teams: k.teams || '',
        events_period: k.events_period ?? 0,
      })),
      recent_events: recentRes.rows.map((e) => ({
        id: e.id,
        timestamp: e.timestamp,
        event_type: e.event_type,
        duration_ms: e.duration_ms,
        metadata: e.metadata,
        user_id: e.user_id,
        user_name: e.user_name,
        user_email: e.user_email,
        api_key_prefix: e.api_key_prefix,
      })),
      knowledge_transfer: {
        invites_pending: knowledge.invites_pending ?? 0,
        users_with_key: knowledge.users_with_key ?? 0,
        users_key_used: knowledge.users_key_used ?? 0,
        median_hours_key_to_first_use: Number(knowledge.median_hours_key_to_first_use) || 0,
        note: 'Proxy for knowledge transfer until invite→task milestones (D-19c) are fully instrumented.',
      },
      usability: {
        responses: survey.responses ?? 0,
        usability_responses: survey.usability_responses ?? 0,
        last_response_at: survey.last_response_at,
        note: 'SUS / satisfaction surveys (D-19d) are not collected in-app yet.',
      },
      workflow: {
        note: 'Ingest/query completion rates appear when richer usage_events types ship (D-19b / D-15).',
        event_types_present: byTypeRes.rows.map((r) => r.event_type),
      },
      note: 'Plaintext API keys are never shown — prefix only. SC-CTSI study metrics surface (D-19*) — not the HIPAA audit ledger.',
    });
  } catch (error) {
    console.error('getUsage error:', error);
    res.status(500).json({ error: error.message || 'Failed to load usage' });
  }
}

/**
 * GET /config/audit-log — HIPAA audit ledger (H-11 / ADR-011).
 * Super admin: unscoped. Admin: pass project_id (own study only).
 * Query: days|from&to, action, resource_type, user_id, q, project_id, page, limit, format=json|csv
 */
async function canReadProjectAudit(user, projectRef) {
  if (!user || !projectRef) return null;
  const found = await query(
    `SELECT id::text AS id, slug, team_id::text AS team_id
     FROM w4h.projects
     WHERE id::text = $1 OR slug = $1`,
    [String(projectRef)]
  );
  if (!found.rowCount) return null;
  const row = found.rows[0];
  if (isSuperAdmin(user.role)) return row;
  if (row.team_id) {
    const tm = await query(
      `SELECT role FROM w4h.team_members WHERE team_id = $1::uuid AND user_id = $2`,
      [row.team_id, user.id]
    );
    if (tm.rows[0]?.role === 'team_admin') return row;
  }
  const mem = await query(
    `SELECT role FROM w4h.project_members WHERE project_id = $1::uuid AND user_id = $2`,
    [row.id, user.id]
  );
  if (mem.rows[0]?.role === 'owner') return row;
  return null;
}

async function getAuditLog(req, res) {
  const projectRef = String(req.query.project_id || '').trim();
  let scopedProject = null;

  if (isAdmin(req.user)) {
    if (projectRef) scopedProject = await canReadProjectAudit(req.user, projectRef);
  } else {
    if (!projectRef) {
      return res.status(403).json({ error: 'project_id is required' });
    }
    scopedProject = await canReadProjectAudit(req.user, projectRef);
    if (!scopedProject) {
      return res.status(403).json({ error: 'Unauthorized' });
    }
  }

  try {
    const format = String(req.query.format || 'json').toLowerCase() === 'csv' ? 'csv' : 'json';
    const page = Math.max(1, Number.parseInt(String(req.query.page || '1'), 10) || 1);
    const limit = Math.min(500, Math.max(10, Number.parseInt(String(req.query.limit || '50'), 10) || 50));
    const offset = (page - 1) * limit;

    const action = String(req.query.action || '').trim();
    const resourceType = String(req.query.resource_type || '').trim();
    const userIdRaw = String(req.query.user_id || '').trim();
    const q = String(req.query.q || '').trim();
    const fromRaw = String(req.query.from || '').trim();
    const toRaw = String(req.query.to || '').trim();
    const days = Math.min(365, Math.max(1, Number.parseInt(String(req.query.days || '30'), 10) || 30));

    const where = [];
    const params = [];

    if (fromRaw && toRaw) {
      params.push(fromRaw);
      where.push(`a.timestamp >= $${params.length}::timestamptz`);
      params.push(toRaw);
      where.push(`a.timestamp <= $${params.length}::timestamptz`);
    } else {
      params.push(String(days));
      where.push(`a.timestamp > now() - ($${params.length} || ' days')::interval`);
    }

    if (action) {
      params.push(action);
      where.push(`a.action = $${params.length}`);
    }
    if (resourceType) {
      params.push(resourceType);
      where.push(`a.resource_type = $${params.length}`);
    }
    if (userIdRaw) {
      const uid = Number.parseInt(userIdRaw, 10);
      if (Number.isFinite(uid)) {
        params.push(uid);
        where.push(`a.user_id = $${params.length}`);
      }
    }
    if (q) {
      params.push(`%${q}%`);
      const p = `$${params.length}`;
      where.push(`(
        a.action ILIKE ${p}
        OR COALESCE(a.resource_type, '') ILIKE ${p}
        OR COALESCE(a.resource_id, '') ILIKE ${p}
        OR COALESCE(u.email, '') ILIKE ${p}
        OR COALESCE(u.name, '') ILIKE ${p}
        OR COALESCE(a.details::text, '') ILIKE ${p}
      )`);
    }

    let scopedPidSql = null;
    if (scopedProject) {
      params.push(scopedProject.id);
      const pid = `$${params.length}`;
      scopedPidSql = pid;
      params.push(scopedProject.slug || scopedProject.id);
      const pslug = `$${params.length}`;
      let extra = '';
      if (scopedProject.team_id) {
        params.push(scopedProject.team_id);
        const tid = `$${params.length}`;
        extra = `
          OR COALESCE(a.details->>'team_id', '') = ${tid}::text
        `;
      }
      params.push(req.user.id);
      const actor = `$${params.length}`;
      where.push(`(
        a.resource_id IN (${pid}::text, ${pslug}::text)
        OR COALESCE(a.details->>'project_id', '') IN (${pid}::text, ${pslug}::text)
        OR COALESCE(a.details->>'project_slug', '') = ${pslug}::text
        OR (
          a.resource_type IN ('dataset', 'project', 'project_member', 'audit_log')
          AND (
            a.resource_id IN (${pid}::text, ${pslug}::text)
            OR a.resource_id IN (
              SELECT pd.dataset_id::text
              FROM w4h.project_datasets pd
              WHERE pd.project_id = ${pid}::uuid
            )
          )
        )
        OR (
          a.resource_type = 'audit_log'
          AND (
            a.user_id = ${actor}
            OR a.user_id IN (
              SELECT pm.user_id FROM w4h.project_members pm
              WHERE pm.project_id = ${pid}::uuid
            )
          )
        )
        ${extra}
      )`);
    }

    const whereSql = where.length ? `WHERE ${where.join(' AND ')}` : '';
    const projectJoin = scopedPidSql
      ? `LEFT JOIN w4h.project_members pm
           ON pm.user_id = a.user_id AND pm.project_id = ${scopedPidSql}::uuid`
      : '';
    const actorRoleSelect = scopedProject
      ? `CASE
           WHEN pm.role = 'owner' THEN 'Admin'
           WHEN pm.role IN ('editor', 'viewer', 'admin') THEN 'User'
           WHEN u.role IN ('super_admin', 'admin') THEN 'Super admin'
           ELSE NULL
         END AS actor_role_label,
         u.role AS user_role`
      : `u.role AS user_role, NULL::text AS actor_role_label`;

    const listSql = `
      SELECT
        a.id,
        a.timestamp,
        a.user_id,
        u.name AS user_name,
        u.email AS user_email,
        ${actorRoleSelect},
        a.action,
        a.resource_type,
        a.resource_id,
        a.ip_address::text AS ip_address,
        a.user_agent,
        a.details
      FROM w4h.audit_log a
      LEFT JOIN w4h.users u ON u.id = a.user_id
      ${projectJoin}
      ${whereSql}
      ORDER BY a.timestamp DESC, a.id DESC
      LIMIT $${params.length + 1} OFFSET $${params.length + 2}
    `;

    const countSql = `
      SELECT COUNT(*)::int AS n
      FROM w4h.audit_log a
      LEFT JOIN w4h.users u ON u.id = a.user_id
      ${whereSql}
    `;

    const summarySql = `
      SELECT
        COUNT(*)::int AS total_period,
        COUNT(*) FILTER (
          WHERE a.timestamp > now() - interval '24 hours'
        )::int AS total_24h,
        COUNT(DISTINCT a.user_id)::int AS distinct_users,
        COUNT(*) FILTER (
          WHERE a.action ILIKE '%fail%' OR a.action ILIKE '%break_glass%'
        )::int AS attention_events
      FROM w4h.audit_log a
      LEFT JOIN w4h.users u ON u.id = a.user_id
      ${whereSql}
    `;

    const actionsSql = `
      SELECT a.action, COUNT(*)::int AS count
      FROM w4h.audit_log a
      LEFT JOIN w4h.users u ON u.id = a.user_id
      ${whereSql}
      GROUP BY a.action
      ORDER BY count DESC
      LIMIT 30
    `;

    const resourceTypesSql = `
      SELECT COALESCE(a.resource_type, '(none)') AS resource_type, COUNT(*)::int AS count
      FROM w4h.audit_log a
      LEFT JOIN w4h.users u ON u.id = a.user_id
      ${whereSql}
      GROUP BY COALESCE(a.resource_type, '(none)')
      ORDER BY count DESC
      LIMIT 20
    `;

    if (format === 'csv') {
      const exportLimit = Math.min(10000, Math.max(limit, 1000));
      const exportSql = `
        SELECT
          a.id,
          a.timestamp,
          a.user_id,
          u.name AS user_name,
          u.email AS user_email,
          a.action,
          a.resource_type,
          a.resource_id,
          a.ip_address::text AS ip_address,
          a.user_agent,
          a.details
        FROM w4h.audit_log a
        LEFT JOIN w4h.users u ON u.id = a.user_id
        ${whereSql}
        ORDER BY a.timestamp DESC, a.id DESC
        LIMIT $${params.length + 1}
      `;
      const exportRes = await query(exportSql, [...params, exportLimit]);

      await writeAuditLog(req, {
        userId: req.user.id,
        action: 'audit.export',
        resourceType: 'audit_log',
        resourceId: scopedProject?.id || null,
        details: {
          format: 'csv',
          row_count: exportRes.rowCount,
          project_id: scopedProject?.id || null,
          project_slug: scopedProject?.slug || null,
          team_id: scopedProject?.team_id || null,
          filters: { action, resource_type: resourceType, user_id: userIdRaw || null, q: q || null, days, from: fromRaw || null, to: toRaw || null },
        },
      }).catch((err) => console.warn('Audit log write failed:', err.message));

      const header = [
        'id', 'timestamp', 'user_id', 'user_name', 'user_email', 'action',
        'resource_type', 'resource_id', 'ip_address', 'user_agent', 'details',
      ];
      const lines = [header.join(',')];
      for (const row of exportRes.rows) {
        lines.push([
          row.id,
          row.timestamp ? new Date(row.timestamp).toISOString() : '',
          row.user_id ?? '',
          csvEscape(row.user_name),
          csvEscape(row.user_email),
          csvEscape(row.action),
          csvEscape(row.resource_type),
          csvEscape(row.resource_id),
          csvEscape(row.ip_address),
          csvEscape(row.user_agent),
          csvEscape(typeof row.details === 'object' ? JSON.stringify(row.details) : row.details),
        ].join(','));
      }

      res.setHeader('Content-Type', 'text/csv; charset=utf-8');
      res.setHeader('Content-Disposition', `attachment; filename="w4h-audit-log.csv"`);
      return res.send(lines.join('\n'));
    }

    await writeAuditLog(req, {
      userId: req.user.id,
      action: 'audit.view',
      resourceType: 'audit_log',
      resourceId: scopedProject?.id || null,
      details: {
        page,
        limit,
        project_id: scopedProject?.id || null,
        project_slug: scopedProject?.slug || null,
        team_id: scopedProject?.team_id || null,
        filters: { action: action || null, resource_type: resourceType || null, user_id: userIdRaw || null, q: q || null, days, from: fromRaw || null, to: toRaw || null },
      },
    }).catch((err) => console.warn('Audit log write failed:', err.message));

    const listParams = [...params, limit, offset];
    const [listRes, countRes, summaryRes, actionsRes, resourceTypesRes] = await Promise.all([
      query(listSql, listParams),
      query(countSql, params),
      query(summarySql, params),
      query(actionsSql, params),
      query(resourceTypesSql, params),
    ]);

    const total = countRes.rows[0]?.n ?? 0;

    res.json({
      period_days: fromRaw && toRaw ? null : days,
      from: fromRaw || null,
      to: toRaw || null,
      page,
      limit,
      total,
      total_pages: Math.max(1, Math.ceil(total / limit)),
      summary: summaryRes.rows[0] || {
        total_period: 0,
        total_24h: 0,
        distinct_users: 0,
        attention_events: 0,
      },
      actions: actionsRes.rows,
      resource_types: resourceTypesRes.rows,
      rows: listRes.rows.map((r) => ({
        id: r.id,
        timestamp: r.timestamp,
        user_id: r.user_id,
        user_name: r.user_name,
        user_email: r.user_email,
        user_role: r.user_role,
        actor_role_label: r.actor_role_label || null,
        action: r.action,
        resource_type: r.resource_type,
        resource_id: r.resource_id,
        ip_address: r.ip_address,
        user_agent: r.user_agent,
        details: r.details,
      })),
      note: 'Append-only HIPAA ledger. Viewing and exporting this page are themselves audited (audit.view / audit.export).',
    });
  } catch (error) {
    console.error('getAuditLog error:', error);
    res.status(500).json({ error: error.message || 'Failed to load audit log' });
  }
}

function csvEscape(value) {
  if (value == null) return '';
  const s = String(value);
  if (/[",\n\r]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
  return s;
}

/** GET /config/database-settings — Get current database settings (even if not fully configured). */
function getDatabaseSettings(req, res) {
  if (!isAdmin(req.user)) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  res.json({
    host: process.env.DB_HOST || '',
    port: process.env.DB_PORT || '5432',
    database: process.env.DB_DATABASE || '',
    user: process.env.DB_USER || ''
  });
}

/** GET /config/mailer-settings — Get current SMTP settings (without password). */
function getMailerSettings(req, res) {
  if (!isAdmin(req.user)) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  const cfg = getMailerConfig();
  res.json({
    host: cfg.host,
    port: cfg.port,
    secure: cfg.secure,
    user: cfg.user,
    from: cfg.from,
    replyTo: cfg.replyTo,
    configured: isMailerConfigured(),
    hasPassword: !!cfg.pass
  });
}

/** POST /config/mailer-settings — Save SMTP settings to .env. Admin only. */
async function postMailerSettings(req, res) {
  if (!isAdmin(req.user)) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  const host = String(req.body?.host || '').trim();
  const port = String(req.body?.port || '').trim();
  const secure = !!req.body?.secure;
  const user = String(req.body?.user || '').trim();
  const from = String(req.body?.from || '').trim();
  const replyTo = String(req.body?.replyTo || '').trim();
  const incomingPassword = req.body?.password;

  if (!host || !port || !from) {
    return res.status(400).json({
      success: false,
      error: 'Required SMTP fields: host, port, from'
    });
  }

  try {
    const effectivePassword = incomingPassword === undefined || incomingPassword === null || incomingPassword === ''
      ? (process.env.SMTP_PASS || '')
      : String(incomingPassword);

    await writeEnvValues({
      SMTP_HOST: host,
      SMTP_PORT: port,
      SMTP_SECURE: secure ? 'true' : 'false',
      SMTP_USER: user,
      SMTP_PASS: effectivePassword,
      SMTP_FROM: from,
      SMTP_REPLY_TO: replyTo
    });

    return res.json({
      success: true,
      message: 'SMTP settings saved',
      configured: isMailerConfigured()
    });
  } catch (error) {
    return res.status(500).json({
      success: false,
      error: 'Failed to save SMTP settings',
      details: error.message
    });
  }
}

/** POST /config/mailer-test — Send a test email. Admin only.
 *  Optional SMTP fields in the body are used for this send only (no Save required).
 *  Omitted password falls back to the saved SMTP_PASS.
 */
async function postMailerTest(req, res) {
  if (!isAdmin(req.user)) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  const to = String(req.body?.to || '').trim();
  if (!to) {
    return res.status(400).json({ success: false, error: 'Test recipient email is required' });
  }

  const smtpOverrides = {
    host: req.body?.host,
    port: req.body?.port,
    secure: req.body?.secure,
    user: req.body?.user,
    password: req.body?.password,
    from: req.body?.from,
    replyTo: req.body?.replyTo
  };

  // If the client sent any SMTP field, treat this as an override payload
  const hasOverride = ['host', 'port', 'from', 'user', 'password', 'replyTo'].some(
    (k) => req.body?.[k] !== undefined && req.body?.[k] !== null && String(req.body[k]).trim() !== ''
  ) || req.body?.secure !== undefined;

  try {
    await sendMail({
      to,
      subject: 'W4H SMTP Test',
      text: 'SMTP test email from W4H settings.',
      html: '<p>SMTP test email from <strong>W4H</strong> settings.</p>',
      smtp: hasOverride ? smtpOverrides : null
    });

    return res.json({
      success: true,
      message: hasOverride
        ? `Test email sent to ${to} using the form values (not saved yet).`
        : `Test email sent to ${to}`
    });
  } catch (error) {
    return res.status(500).json({
      success: false,
      error: 'SMTP test failed',
      details: error.message
    });
  }
}

/** GET /config/app-settings — Get general app settings. Admin only. */
function getAppSettings(req, res) {
  if (!isAdmin(req.user)) {
    return res.status(403).json({ error: 'Unauthorized' });
  }
  res.json({
    helpEmail: process.env.HELP_EMAIL || '',
    usageTracking: process.env.USAGE_TRACKING === 'true',
  });
}

/** POST /config/app-settings — Save general app settings. Admin only. */
async function postAppSettings(req, res) {
  if (!isAdmin(req.user)) {
    return res.status(403).json({ error: 'Unauthorized' });
  }
  const helpEmail = String(req.body?.helpEmail ?? '').trim();
  const usageTracking = !!req.body?.usageTracking;

  try {
    await writeEnvValues({ HELP_EMAIL: helpEmail, USAGE_TRACKING: usageTracking ? 'true' : 'false' });
    return res.json({ success: true, message: 'App settings saved.' });
  } catch (error) {
    return res.status(500).json({ success: false, error: 'Failed to save app settings.', details: error.message });
  }
}

/**
 * POST /config/database
 * Configure database credentials
 * Body: { host, port, database, user, password }
 */
async function postInitDatabase(req, res) {
  const { host, port, database, user, password } = req.body;
  const effectivePassword = password || process.env.DB_PASSWORD;

  // Validate required fields
  if (!host || !port || !database || !user) {
    return res.status(400).json({
      success: false,
      error: 'Required database fields: host, port, database, user'
    });
  }

  if (!effectivePassword) {
    return res.status(400).json({
      success: false,
      error: 'Password is required when no existing database password is configured'
    });
  }

  try {
    // Test the connection first
    const testPool = new Pool({
      host,
      port: parseInt(port),
      database,
      user,
      password: effectivePassword,
      connectionTimeoutMillis: 5000
    });

    let databaseCreated = false;
    let testPoolEnded = false;
    try {
      await testPool.query('SELECT 1');
    } catch (connError) {
      if (process.env.W4H_DEBUG) console.log(`[config] Connection error code: ${connError.code}, message: ${connError.message}`);
      
      // If database doesn't exist, try to create it
      if (connError.code === '3D000' || connError.message?.includes('does not exist')) { 
        if (process.env.W4H_DEBUG) console.info(`[config] Database '${database}' does not exist. Attempting to create it...`);
        await testPool.end();
        testPoolEnded = true;
        
        // Connect to 'postgres' database to create the target database
        const adminPool = new Pool({
          host,
          port: parseInt(port),
          database: 'postgres', // Connect to default postgres database
          user,
          password: effectivePassword,
          connectionTimeoutMillis: 5000
        });

        try {
          // Test connection to postgres database first
          await adminPool.query('SELECT 1');
        } catch (postgresError) {
          await adminPool.end();
          throw new Error(`Cannot connect to PostgreSQL server with provided credentials: ${postgresError.message}. Make sure host, port, username, and password are correct, and PostgreSQL is running.`);
        }

        try {
          // First check if user has CREATEDB privilege
          const privCheck = await adminPool.query(
            "SELECT usecreatedb FROM pg_user WHERE usename = $1",
            [user]
          );
          
          if (privCheck.rows.length === 0) {
            throw new Error(`User '${user}' not found on PostgreSQL server. Check that the username exists and is correct.`);
          }
          
          if (!privCheck.rows[0].usecreatedb) {
            throw new Error(`User '${user}' does not have CREATEDB privilege. Your DBA needs to run: ALTER USER "${user}" CREATEDB;`);
          }

          await adminPool.query(`CREATE DATABASE "${database}" TEMPLATE template1 ENCODING 'UTF8'`);
          if (process.env.W4H_DEBUG) console.info(`[config] Database '${database}' created successfully`);
          databaseCreated = true;
        } catch (createError) {
          await adminPool.end();
          throw createError;
        }

        await adminPool.end();

        // Reconnect to the newly created database to verify
        const newTestPool = new Pool({
          host,
          port: parseInt(port),
          database,
          user,
          password: effectivePassword,
          connectionTimeoutMillis: 5000
        });

        try {
          await newTestPool.query('SELECT 1');
          await newTestPool.end();
        } catch (verifyError) {
          throw new Error(`Database created but cannot connect to it: ${verifyError.message}`);
        }
      } else {
        // Not a "database doesn't exist" error, so it's a connection problem
        throw new Error(`Cannot connect to database '${database}': ${connError.message}. Check that PostgreSQL is running, the host/port are correct, and the username/password are valid.`);
      }
    }

    if (!testPoolEnded) {
      await testPool.end();
    }

    // Generate TOKEN_SECRET if it doesn't exist
    if (!process.env.TOKEN_SECRET) {
      process.env.TOKEN_SECRET = crypto.randomBytes(64).toString('hex');
    }

    // Update or add database config
    const dbConfig = {
      DB_HOST: host,
      DB_PORT: port,
      DB_DATABASE: database,
      DB_USER: user,
      DB_PASSWORD: effectivePassword,
      TOKEN_SECRET: process.env.TOKEN_SECRET
    };

    // Connection successful — update `.env` in a stable, template-driven way
    await writeEnvValues(dbConfig);

    // Reset the database pool with new credentials
    await db.resetPool();

    // Initialize database schema (create tables if needed)
    const initResult = await initializeDatabase();
    if (!initResult.success) {
      console.error('Database initialization failed:', initResult.error);
      return res.status(500).json({
        success: false,
        error: 'Database configured but schema initialization failed',
        details: initResult.error
      });
    }

    let needsInitialAdmin = false;
    try {
      needsInitialAdmin = !(await hasAnyUsers());
    } catch (userCheckError) {
      console.warn('Unable to verify initial admin status:', userCheckError.message);
    }

    const message = databaseCreated 
      ? `Database '${database}' created and configured successfully.`
      : 'Database configuration saved and applied successfully.';

    res.json({
      success: true,
      message,
      databaseCreated,
      requiresRestart: false,
      initialized: initResult.success,
      requiresAdminSetup: needsInitialAdmin
    });

  } catch (error) {
    console.error('Database configuration error:', error);
    
    // Provide specific error messages based on error type
    let errorMessage = error.message || 'Unknown error';
    let userMessage = 'Failed to configure database';
    
    if (error.message?.includes('ECONNREFUSED')) {
      userMessage = 'Connection refused - PostgreSQL may not be running or accepting connections';
    } else if (error.message?.includes('ENOTFOUND')) {
      userMessage = 'Database host not found - check the hostname/IP address';
    } else if (error.message?.includes('ETIMEDOUT')) {
      userMessage = 'Connection timed out - network or firewall issue';
    } else if (error.message?.includes('does not have CREATEDB')) {
      userMessage = 'User does not have CREATEDB privilege - ask DBA to grant it';
    } else if (error.message?.includes('not found on PostgreSQL')) {
      userMessage = 'User does not exist on PostgreSQL server - check username';
    } else if (error.message?.includes('password authentication failed')) {
      userMessage = 'Incorrect password - check credentials';
    } else if (error.message?.includes('role')) {
      userMessage = 'Role/user error - check PostgreSQL user exists and has correct privileges';
    } else if (error.message?.includes('database')) {
      userMessage = 'Database error - ' + error.message;
    }
    
    res.status(500).json({
      success: false,
      error: userMessage,
      details: errorMessage,
      code: error.code
    });
  }
}

/** POST /config/admin-setup — Create initial admin account. Public only when no users exist yet. */
async function postAdminSetup(req, res) {
  const { email, password, name } = req.body;

  if (!email || !password) {
    return res.status(400).json({
      success: false,
      error: 'email and password are required'
    });
  }

  try {
    const usersExist = await hasAnyUsers();
    if (usersExist) {
      return res.status(400).json({
        success: false,
        error: 'Initial admin has already been configured'
      });
    }

    const adminResult = await createInitialAdmin({
      email: email.trim(),
      password,
      name: name ? name.trim() : undefined
    });

    if (!adminResult.success) {
      return res.status(400).json({
        success: false,
        error: adminResult.error
      });
    }

    const token = auth.sign({
      id: adminResult.user.id,
      email: adminResult.user.email,
      name: adminResult.user.name,
      role: 'super_admin',
      firstTimeLogin: true
    });

    let mailWarning = null;
    if (isMailerConfigured()) {
      try {
        await sendMail({
          to: adminResult.user.email,
          subject: 'Your W4H admin account is ready',
          text: `Hello ${adminResult.user.name},\n\nYour W4H admin account has been created successfully.\nYou can now sign in to complete setup.\n`,
          html: `<p>Hello ${adminResult.user.name},</p><p>Your <strong>W4H admin account</strong> has been created successfully.</p><p>You can now sign in to complete setup.</p>`
        });
      } catch (mailError) {
        mailWarning = mailError.message;
      }
    }

    res.json({
      success: true,
      message: 'Initial admin account created successfully',
      token,
      ...(mailWarning ? { mailWarning: `Admin created but email failed: ${mailWarning}` } : {}),
      user: {
        id: adminResult.user.id,
        email: adminResult.user.email,
        name: adminResult.user.name,
        role: 'super_admin',
        firstTimeLogin: true
      }
    });
  } catch (error) {
    console.error('Initial admin setup error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to create initial admin account',
      details: error.message
    });
  }
}

/** GET /config/test — Test database connection. Admin only. */
async function getTest(req, res) {
  if (!isAdmin(req.user)) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  if (!process.env.DB_HOST || !process.env.DB_PORT || !process.env.DB_DATABASE || 
      !process.env.DB_USER || !process.env.DB_PASSWORD) {
    return res.status(400).json({
      success: false,
      error: 'Database not configured'
    });
  }

  try {
    const testPool = new Pool({
      host: process.env.DB_HOST,
      port: parseInt(process.env.DB_PORT),
      database: process.env.DB_DATABASE,
      user: process.env.DB_USER,
      password: process.env.DB_PASSWORD,
      connectionTimeoutMillis: 5000
    });

    const result = await testPool.query('SELECT NOW() as current_time, version() as version');
    await testPool.end();

    res.json({
      success: true,
      message: 'Database connection successful',
      serverTime: result.rows[0].current_time,
      version: result.rows[0].version
    });

  } catch (error) {
    console.error('Database test error:', error);
    res.status(500).json({
      success: false,
      error: 'Database connection failed',
      details: error.message
    });
  }
}

/** POST /config/reset — Reset database connection pool. Admin only. */
async function postReset(req, res) {
  if (!isAdmin(req.user)) {
    return res.status(403).json({ error: 'Unauthorized' });
  }
  
  try {
    const success = await db.resetPool();
    
    if (success) {
      res.json({
        success: true,
        message: 'Database pool reset successfully'
      });
    } else {
      res.status(400).json({
        success: false,
        error: 'Database not configured'
      });
    }
  } catch (error) {
    console.error('Pool reset error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to reset pool',
      details: error.message
    });
  }
}
