import { query } from '../../db.js';
import { encryptUserApiKey, decryptUserApiKey } from '../utils/apiKeyCrypto.js';
import { isSuperAdmin, isTeamAdminOnTeam } from '../utils/user.js';

export const DEFAULT_MODEL = 'claude-haiku-4-5-20251001';
export const DEFAULT_TEAM_QUOTA = 250_000;
export const BILLING = Object.freeze({
  BYOK: 'byok',
  QUOTA: 'quota',
});

function maskKey(plaintext, prefix) {
  if (prefix) return prefix;
  const key = String(plaintext || '').trim();
  if (!key) return null;
  const last = key.slice(-4);
  const head = key.slice(0, 7);
  return `${head}…${last}`;
}

function prefixFromKey(plaintext) {
  const key = String(plaintext || '').trim();
  if (!key) return null;
  return maskKey(key);
}

export async function getPlatformAiSettings() {
  const res = await query(
    `SELECT enabled, anthropic_api_key_encrypted, anthropic_key_prefix, model,
            default_monthly_token_quota, ai_phi_filter, updated_at, updated_by
     FROM w4h.platform_ai_settings WHERE id = 1`
  );
  const row = res.rows[0] || {};
  const envKey = String(process.env.ANTHROPIC_API_KEY || '').trim();
  const stored = decryptUserApiKey(row.anthropic_api_key_encrypted);
  const apiKey = stored || envKey || null;
  return {
    enabled: row.enabled !== false,
    model: row.model || DEFAULT_MODEL,
    default_monthly_token_quota: row.default_monthly_token_quota ?? DEFAULT_TEAM_QUOTA,
    ai_phi_filter: row.ai_phi_filter === 'permissive' ? 'permissive' : 'strict',
    has_key: !!apiKey,
    key_prefix: row.anthropic_key_prefix || (apiKey ? prefixFromKey(apiKey) : null),
    apiKey,
    key_source: stored ? 'db' : envKey ? 'env' : null,
    updated_at: row.updated_at || null,
    updated_by: row.updated_by || null,
  };
}

export async function updatePlatformAiSettings(patch, { userId } = {}) {
  const current = await getPlatformAiSettings();
  const enabled = patch.enabled != null ? !!patch.enabled : current.enabled;
  const model = String(patch.model || current.model || DEFAULT_MODEL).trim() || DEFAULT_MODEL;
  const quota = patch.default_monthly_token_quota != null
    ? Number(patch.default_monthly_token_quota)
    : current.default_monthly_token_quota;
  const phi = patch.ai_phi_filter === 'permissive' ? 'permissive' : (patch.ai_phi_filter === 'strict' ? 'strict' : current.ai_phi_filter);

  let enc = undefined;
  let prefix = undefined;
  if (patch.clear_anthropic_api_key) {
    enc = null;
    prefix = null;
  } else if (Object.prototype.hasOwnProperty.call(patch, 'anthropic_api_key')) {
    const key = String(patch.anthropic_api_key || '').trim();
    if (key) {
      enc = encryptUserApiKey(key);
      prefix = prefixFromKey(key);
    }
  }

  await query(
    `INSERT INTO w4h.platform_ai_settings (
       id, enabled, model, default_monthly_token_quota, ai_phi_filter, updated_at, updated_by
     ) VALUES (1, $1, $2, $3, $4, NOW(), $5)
     ON CONFLICT (id) DO UPDATE SET
       enabled = EXCLUDED.enabled,
       model = EXCLUDED.model,
       default_monthly_token_quota = EXCLUDED.default_monthly_token_quota,
       ai_phi_filter = EXCLUDED.ai_phi_filter,
       updated_at = NOW(),
       updated_by = EXCLUDED.updated_by`,
    [enabled, model, Number.isFinite(quota) ? quota : null, phi, userId || null]
  );

  if (enc !== undefined) {
    await query(
      `UPDATE w4h.platform_ai_settings
       SET anthropic_api_key_encrypted = $1, anthropic_key_prefix = $2
       WHERE id = 1`,
      [enc, prefix]
    );
  }
  return publicPlatformSettings(await getPlatformAiSettings());
}

export function publicPlatformSettings(full) {
  return {
    enabled: full.enabled,
    model: full.model,
    default_monthly_token_quota: full.default_monthly_token_quota,
    ai_phi_filter: full.ai_phi_filter,
    has_key: full.has_key,
    key_prefix: full.key_prefix,
    key_source: full.key_source,
    updated_at: full.updated_at,
  };
}

export async function ensureTeamAiSettings(teamId) {
  await query(
    `INSERT INTO w4h.team_ai_settings (team_id)
     VALUES ($1::uuid)
     ON CONFLICT (team_id) DO NOTHING`,
    [teamId]
  );
}

export async function getTeamAiSettings(teamId) {
  if (!teamId) return null;
  await ensureTeamAiSettings(teamId);
  const res = await query(
    `SELECT t.name AS team_name, s.enabled, s.anthropic_api_key_encrypted, s.anthropic_key_prefix,
            s.monthly_token_quota, s.ai_phi_filter, s.updated_at, s.updated_by
     FROM w4h.team_ai_settings s
     JOIN w4h.teams t ON t.id = s.team_id
     WHERE s.team_id = $1::uuid`,
    [teamId]
  );
  const row = res.rows[0];
  if (!row) return null;
  const apiKey = decryptUserApiKey(row.anthropic_api_key_encrypted);
  const platform = await getPlatformAiSettings();
  return {
    team_id: teamId,
    team_name: row.team_name,
    enabled: row.enabled !== false,
    has_key: !!apiKey,
    key_prefix: row.anthropic_key_prefix || (apiKey ? prefixFromKey(apiKey) : null),
    apiKey,
    monthly_token_quota: row.monthly_token_quota ?? platform.default_monthly_token_quota,
    quota_is_override: row.monthly_token_quota != null,
    ai_phi_filter: row.ai_phi_filter === 'permissive' ? 'permissive' : 'strict',
    updated_at: row.updated_at,
  };
}

export function publicTeamSettings(full) {
  if (!full) return null;
  return {
    team_id: full.team_id,
    team_name: full.team_name,
    enabled: full.enabled,
    has_key: full.has_key,
    key_prefix: full.key_prefix,
    monthly_token_quota: full.monthly_token_quota,
    quota_is_override: full.quota_is_override,
    ai_phi_filter: full.ai_phi_filter,
    updated_at: full.updated_at,
    billing_source: full.has_key ? BILLING.BYOK : BILLING.QUOTA,
  };
}

export async function updateTeamAiSettings(teamId, patch, { userId, isSuper } = {}) {
  await ensureTeamAiSettings(teamId);
  const current = await getTeamAiSettings(teamId);
  const enabled = patch.enabled != null ? !!patch.enabled : current.enabled;
  const phi = patch.ai_phi_filter === 'permissive' ? 'permissive'
    : patch.ai_phi_filter === 'strict' ? 'strict'
      : current.ai_phi_filter;

  const sets = ['enabled = $2', 'ai_phi_filter = $3', 'updated_at = NOW()', 'updated_by = $4'];
  const values = [teamId, enabled, phi, userId || null];
  let p = 5;

  if (isSuper && Object.prototype.hasOwnProperty.call(patch, 'monthly_token_quota')) {
    const q = patch.monthly_token_quota == null || patch.monthly_token_quota === ''
      ? null
      : Number(patch.monthly_token_quota);
    sets.push(`monthly_token_quota = $${p++}`);
    values.push(Number.isFinite(q) ? q : null);
  }

  if (patch.clear_anthropic_api_key) {
    sets.push(`anthropic_api_key_encrypted = $${p++}`);
    values.push(null);
    sets.push(`anthropic_key_prefix = $${p++}`);
    values.push(null);
  } else if (Object.prototype.hasOwnProperty.call(patch, 'anthropic_api_key')) {
    const key = String(patch.anthropic_api_key || '').trim();
    if (key) {
      sets.push(`anthropic_api_key_encrypted = $${p++}`);
      values.push(encryptUserApiKey(key));
      sets.push(`anthropic_key_prefix = $${p++}`);
      values.push(prefixFromKey(key));
    }
  }

  await query(
    `UPDATE w4h.team_ai_settings SET ${sets.join(', ')} WHERE team_id = $1::uuid`,
    values
  );
  return publicTeamSettings(await getTeamAiSettings(teamId));
}

export async function userCanUseTeam(user, teamId) {
  if (!teamId) return isSuperAdmin(user?.role);
  if (isSuperAdmin(user?.role)) return true;
  const res = await query(
    `SELECT 1 FROM w4h.team_members WHERE team_id = $1::uuid AND user_id = $2 LIMIT 1`,
    [teamId, user?.id]
  );
  return res.rowCount > 0;
}

export async function userCanManageTeamAi(user, teamId) {
  return isTeamAdminOnTeam(user?.id, teamId, user?.role);
}

async function tokensUsedThisMonth(teamId, billingSource = BILLING.QUOTA) {
  const res = await query(
    `SELECT COALESCE(SUM(
        COALESCE((metadata->>'input_tokens')::int, 0)
        + COALESCE((metadata->>'output_tokens')::int, 0)
      ), 0)::int AS used
     FROM w4h.usage_events
     WHERE event_type = 'ai.chat'
       AND metadata->>'billing_source' = $1
       AND metadata->>'team_id' = $2
       AND timestamp >= date_trunc('month', NOW())`,
    [billingSource, String(teamId)]
  );
  return res.rows[0]?.used || 0;
}

export async function getTeamQuotaSnapshot(teamId) {
  const team = await getTeamAiSettings(teamId);
  const platform = await getPlatformAiSettings();
  const used = teamId ? await tokensUsedThisMonth(teamId) : 0;
  const limit = team?.monthly_token_quota ?? platform.default_monthly_token_quota;
  return {
    used,
    limit,
    remaining: limit == null ? null : Math.max(0, Number(limit) - used),
    billing_source: team?.has_key ? BILLING.BYOK : BILLING.QUOTA,
  };
}

export async function getTeamAiUsage(teamId) {
  const team = await getTeamAiSettings(teamId);
  const snapshot = await getTeamQuotaSnapshot(teamId);
  const byokUsed = teamId ? await tokensUsedThisMonth(teamId, BILLING.BYOK) : 0;
  const members = await query(
    `SELECT u.id, u.name, u.email, tm.role,
            COUNT(e.id)::int AS calls,
            COALESCE(SUM(COALESCE((e.metadata->>'input_tokens')::int, 0)), 0)::int AS input_tokens,
            COALESCE(SUM(COALESCE((e.metadata->>'output_tokens')::int, 0)), 0)::int AS output_tokens
     FROM w4h.team_members tm
     JOIN w4h.users u ON u.id = tm.user_id
     LEFT JOIN w4h.usage_events e
       ON e.user_id = u.id
      AND e.event_type = 'ai.chat'
      AND e.metadata->>'team_id' = $1
      AND e.timestamp >= date_trunc('month', NOW())
     WHERE tm.team_id = $1::uuid
     GROUP BY u.id, u.name, u.email, tm.role
     ORDER BY (COALESCE(SUM(COALESCE((e.metadata->>'input_tokens')::int, 0)), 0)
             + COALESCE(SUM(COALESCE((e.metadata->>'output_tokens')::int, 0)), 0)) DESC,
              u.name ASC`,
    [String(teamId)]
  );

  return {
    team_id: teamId,
    team_name: team?.team_name || null,
    has_key: !!team?.has_key,
    key_prefix: team?.key_prefix || null,
    billing_source: team?.has_key ? BILLING.BYOK : BILLING.QUOTA,
    platform_credits: {
      used: snapshot.used,
      limit: snapshot.limit,
      remaining: snapshot.remaining,
    },
    team_key_used: byokUsed,
    members: members.rows.map((r) => ({
      user_id: r.id,
      name: r.name,
      email: r.email,
      role: r.role,
      calls: r.calls,
      input_tokens: r.input_tokens,
      output_tokens: r.output_tokens,
      tokens: (r.input_tokens || 0) + (r.output_tokens || 0),
    })),
  };
}

export async function listTeamQuotaRows() {
  const platform = await getPlatformAiSettings();
  const teams = await query(
    `SELECT t.id, t.name, t.slug,
            (s.anthropic_api_key_encrypted IS NOT NULL AND s.anthropic_api_key_encrypted <> '') AS has_key,
            s.monthly_token_quota,
            COALESCE(s.ai_phi_filter, 'strict') AS ai_phi_filter,
            COALESCE(s.enabled, true) AS enabled
     FROM w4h.teams t
     LEFT JOIN w4h.team_ai_settings s ON s.team_id = t.id
     WHERE t.status = 'active'
     ORDER BY t.name`
  );

  const usage = await query(
    `SELECT metadata->>'team_id' AS team_id,
            metadata->>'billing_source' AS billing_source,
            COALESCE(SUM(
              COALESCE((metadata->>'input_tokens')::int, 0)
              + COALESCE((metadata->>'output_tokens')::int, 0)
            ), 0)::int AS used
     FROM w4h.usage_events
     WHERE event_type = 'ai.chat'
       AND timestamp >= date_trunc('month', NOW())
     GROUP BY 1, 2`
  );
  const usedQuota = new Map();
  const usedByok = new Map();
  for (const r of usage.rows) {
    const id = String(r.team_id || '');
    if (r.billing_source === BILLING.BYOK) usedByok.set(id, r.used);
    else usedQuota.set(id, r.used);
  }

  return teams.rows.map((t) => {
    const id = String(t.id);
    const limit = t.monthly_token_quota ?? platform.default_monthly_token_quota;
    const used = usedQuota.get(id) || 0;
    return {
      team_id: t.id,
      name: t.name,
      slug: t.slug,
      has_byok: !!t.has_key,
      enabled: t.enabled !== false,
      ai_phi_filter: t.ai_phi_filter || 'strict',
      monthly_token_quota: limit,
      used,
      used_byok: usedByok.get(id) || 0,
      remaining: limit == null ? null : Math.max(0, Number(limit) - used),
    };
  });
}

/**
 * Resolve Anthropic key: team BYOK first, else Super-admin provisioned platform key + team quota.
 */
export async function authorizeAiCall({ user, teamId, shell }) {
  const platform = await getPlatformAiSettings();
  if (!platform.enabled) {
    return { allowed: false, status: 403, error: 'AI chat is disabled on this platform.' };
  }

  const adminShell = shell === 'admin' && isSuperAdmin(user?.role);
  if (adminShell) {
    if (!platform.apiKey) {
      return {
        allowed: false,
        status: 503,
        error: 'No platform Claude key is configured. Add one in Settings.',
        empty: true,
      };
    }
    return {
      allowed: true,
      apiKey: platform.apiKey,
      billingSource: BILLING.QUOTA,
      model: platform.model,
      phiFilter: platform.ai_phi_filter,
      teamId: null,
      keyScope: 'platform',
    };
  }

  if (!teamId) {
    return {
      allowed: false,
      status: 400,
      error: 'Select a team to use AI chat, or ask a Super admin to configure a platform key.',
      empty: true,
    };
  }

  if (!(await userCanUseTeam(user, teamId))) {
    return { allowed: false, status: 403, error: 'You are not a member of this team.' };
  }

  const team = await getTeamAiSettings(teamId);
  if (!team?.enabled) {
    return { allowed: false, status: 403, error: 'AI chat is disabled for this team.' };
  }

  const phiFilter = team.ai_phi_filter || platform.ai_phi_filter;

  if (team.apiKey) {
    return {
      allowed: true,
      apiKey: team.apiKey,
      billingSource: BILLING.BYOK,
      model: platform.model,
      phiFilter,
      teamId,
      keyScope: 'team',
    };
  }

  if (!platform.apiKey) {
    return {
      allowed: false,
      status: 503,
      empty: true,
      error: 'No Claude API key is available. An Admin can add the lab Anthropic key under Project → People, or a Super admin can provision a platform key with a per-team quota.',
    };
  }

  const quota = await getTeamQuotaSnapshot(teamId);
  if (quota.limit != null && quota.used >= Number(quota.limit)) {
    return {
      allowed: false,
      status: 429,
      error: 'This team has used its provisioned AI token quota for the month. Ask a Super admin to raise the quota, or add a lab Anthropic key under Project → People.',
      quota,
    };
  }

  return {
    allowed: true,
    apiKey: platform.apiKey,
    billingSource: BILLING.QUOTA,
    model: platform.model,
    phiFilter,
    teamId,
    keyScope: 'platform',
    quota,
  };
}

export async function getAiStatus({ user, teamId, shell }) {
  const auth = await authorizeAiCall({ user, teamId, shell });
  const platform = await getPlatformAiSettings();
  let team = null;
  let quota = null;
  if (teamId && (isSuperAdmin(user?.role) || await userCanUseTeam(user, teamId))) {
    team = publicTeamSettings(await getTeamAiSettings(teamId));
    quota = await getTeamQuotaSnapshot(teamId);
  }
  return {
    available: !!auth.allowed,
    empty: !!auth.empty,
    error: auth.allowed ? null : auth.error,
    status: auth.status || (auth.allowed ? 200 : 403),
    key_scope: auth.keyScope || null,
    billing_source: auth.billingSource || null,
    model: platform.model,
    phi_filter: auth.phiFilter || team?.ai_phi_filter || platform.ai_phi_filter,
    phi_safe: (auth.phiFilter || team?.ai_phi_filter || platform.ai_phi_filter) !== 'permissive',
    team,
    quota,
    platform_enabled: platform.enabled,
    can_manage_team: teamId ? await userCanManageTeamAi(user, teamId) : false,
    can_manage_platform: isSuperAdmin(user?.role),
  };
}
