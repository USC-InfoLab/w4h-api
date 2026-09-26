import express from 'express';
import { authenticate, requirePlatformAdmin } from '../auth.js';
import { isSuperAdmin } from '../lib/utils/user.js';
import { recordUsageEventSafe } from '../lib/utils/usage.js';
import { writeAuditLogSafe } from '../lib/utils/audit.js';
import { scanPhiMessages } from '../lib/ai/phi.js';
import { createClaudeMessage, streamClaudeMessage } from '../lib/ai/claude.js';
import { buildSystemPrompt, normalizeMode } from '../lib/ai/prompt.js';
import {
  authorizeAiCall,
  getAiStatus,
  getPlatformAiSettings,
  publicPlatformSettings,
  updatePlatformAiSettings,
  getTeamAiSettings,
  publicTeamSettings,
  updateTeamAiSettings,
  userCanManageTeamAi,
  userCanUseTeam,
  listTeamQuotaRows,
  getTeamAiUsage,
} from '../lib/ai/keys.js';
import {
  listConversations,
  getConversation,
  createConversation,
  updateConversation,
  deleteConversation,
  appendTurn,
} from '../lib/ai/conversations.js';

const router = express.Router();

function contextFromBody(body = {}) {
  return {
    shell: body.shell === 'admin' ? 'admin' : 'app',
    route: String(body.route || '').slice(0, 200),
    teamId: body.team_id && body.team_id !== 'null' && body.team_id !== 'undefined' ? body.team_id : null,
    teamName: body.team_name ? String(body.team_name).slice(0, 120) : null,
    projectId: body.project_id || null,
    datasetId: body.dataset_id || null,
    docSlug: body.doc_slug ? String(body.doc_slug).slice(0, 200) : null,
    contributionSlug: body.contribution_slug ? String(body.contribution_slug).slice(0, 200) : null,
    mode: normalizeMode(body.mode),
    confirmNoPhi: !!body.confirm_no_phi,
    stream: body.stream !== false,
  };
}

router.get('/status', authenticate, async (req, res) => {
  try {
    const ctx = contextFromBody(req.query);
    const status = await getAiStatus({
      user: req.user,
      teamId: ctx.teamId,
      shell: ctx.shell,
    });
    res.json(status);
  } catch (err) {
    console.error('ai status:', err);
    res.status(500).json({ error: err.message });
  }
});

router.get('/settings/platform', authenticate, requirePlatformAdmin, async (_req, res) => {
  try {
    const full = await getPlatformAiSettings();
    const teams = await listTeamQuotaRows();
    res.json({ ...publicPlatformSettings(full), teams });
  } catch (err) {
    console.error('ai platform settings:', err);
    res.status(500).json({ error: err.message });
  }
});

router.patch('/settings/platform', authenticate, requirePlatformAdmin, async (req, res) => {
  try {
    const settings = await updatePlatformAiSettings(req.body || {}, { userId: req.user.id });
    writeAuditLogSafe(req, {
      userId: req.user.id,
      action: 'ai.platform_settings',
      resourceType: 'platform_ai_settings',
      resourceId: '1',
      details: {
        enabled: settings.enabled,
        has_key: settings.has_key,
        phi_filter: settings.ai_phi_filter,
      },
    });
    const teams = await listTeamQuotaRows();
    res.json({ ...settings, teams });
  } catch (err) {
    console.error('ai platform settings patch:', err);
    res.status(500).json({ error: err.message });
  }
});

router.get('/settings/team/:teamId', authenticate, async (req, res) => {
  try {
    const { teamId } = req.params;
    if (!(await userCanUseTeam(req.user, teamId))) {
      return res.status(403).json({ error: 'Unauthorized' });
    }
    const full = await getTeamAiSettings(teamId);
    const usage = await getTeamAiUsage(teamId);
    res.json({ ...publicTeamSettings(full), usage });
  } catch (err) {
    console.error('ai team settings:', err);
    res.status(500).json({ error: err.message });
  }
});

router.patch('/settings/team/:teamId', authenticate, async (req, res) => {
  try {
    const { teamId } = req.params;
    if (!(await userCanManageTeamAi(req.user, teamId))) {
      return res.status(403).json({ error: 'Team admin required' });
    }
    const settings = await updateTeamAiSettings(teamId, req.body || {}, {
      userId: req.user.id,
      isSuper: isSuperAdmin(req.user.role),
    });
    writeAuditLogSafe(req, {
      userId: req.user.id,
      action: 'ai.team_settings',
      resourceType: 'team',
      resourceId: teamId,
      details: {
        has_key: settings.has_key,
        phi_filter: settings.ai_phi_filter,
        monthly_token_quota: settings.monthly_token_quota,
      },
    });
    res.json(settings);
  } catch (err) {
    console.error('ai team settings patch:', err);
    res.status(500).json({ error: err.message });
  }
});

router.get('/conversations', authenticate, async (req, res) => {
  try {
    const rows = await listConversations(req.user.id);
    res.json({ conversations: rows });
  } catch (err) {
    console.error('ai conversations:', err);
    res.status(500).json({ error: err.message });
  }
});

router.post('/conversations', authenticate, async (req, res) => {
  try {
    const conv = await createConversation(req.user.id, { title: req.body?.title });
    res.status(201).json(conv);
  } catch (err) {
    console.error('ai conversations create:', err);
    res.status(500).json({ error: err.message });
  }
});

router.get('/conversations/:id', authenticate, async (req, res) => {
  try {
    const conv = await getConversation(req.user.id, req.params.id);
    if (!conv) return res.status(404).json({ error: 'Not found' });
    res.json(conv);
  } catch (err) {
    console.error('ai conversation get:', err);
    res.status(500).json({ error: err.message });
  }
});

router.patch('/conversations/:id', authenticate, async (req, res) => {
  try {
    const conv = await updateConversation(req.user.id, req.params.id, req.body || {});
    if (!conv) return res.status(404).json({ error: 'Not found' });
    res.json(conv);
  } catch (err) {
    console.error('ai conversation patch:', err);
    res.status(500).json({ error: err.message });
  }
});

router.delete('/conversations/:id', authenticate, async (req, res) => {
  try {
    const ok = await deleteConversation(req.user.id, req.params.id);
    if (!ok) return res.status(404).json({ error: 'Not found' });
    res.json({ ok: true });
  } catch (err) {
    console.error('ai conversation delete:', err);
    res.status(500).json({ error: err.message });
  }
});

router.post('/chat', authenticate, async (req, res) => {
  const started = Date.now();
  const ctx = contextFromBody(req.body);
  const messages = req.body?.messages;

  if (!Array.isArray(messages) || messages.length === 0) {
    return res.status(400).json({ error: 'messages array required' });
  }
  for (const msg of messages) {
    if ((msg.role !== 'user' && msg.role !== 'assistant') || typeof msg.content !== 'string') {
      return res.status(400).json({ error: 'Invalid message format' });
    }
  }

  let auth;
  try {
    auth = await authorizeAiCall({
      user: req.user,
      teamId: ctx.teamId,
      shell: ctx.shell,
    });
  } catch (err) {
    console.error('ai authorize:', err);
    return res.status(500).json({ error: err.message });
  }

  if (!auth.allowed) {
    return res.status(auth.status || 403).json({ error: auth.error || 'AI not available', empty: auth.empty });
  }

  const phiHit = scanPhiMessages(messages);
  if (phiHit.blocked && auth.phiFilter === 'strict') {
    recordUsageEventSafe({
      userId: req.user.id,
      eventType: 'ai.chat_blocked',
      durationMs: Date.now() - started,
      metadata: {
        reasons: phiHit.reasons,
        team_id: ctx.teamId,
        route: ctx.route,
        mode: ctx.mode,
      },
    });
    return res.status(400).json({
      error: 'PHI-safe mode blocked this message. Remove identifiers, emails, GPS, or raw samples, then try again.',
      code: 'PHI_BLOCKED',
      reasons: phiHit.reasons,
    });
  }
  if (phiHit.blocked && auth.phiFilter === 'permissive' && !ctx.confirmNoPhi) {
    return res.status(400).json({
      error: 'This message looks like it may include identifiers. Confirm it contains no PHI to send in permissive mode.',
      code: 'PHI_CONFIRM',
      reasons: phiHit.reasons,
    });
  }
  if (phiHit.blocked && auth.phiFilter === 'permissive') {
    writeAuditLogSafe(req, {
      userId: req.user.id,
      action: 'ai.chat_permissive',
      resourceType: 'ai_chat',
      details: { reasons: phiHit.reasons, team_id: ctx.teamId, mode: ctx.mode },
    });
  }

  let system;
  try {
    system = await buildSystemPrompt({
      mode: ctx.mode,
      shell: ctx.shell,
      route: ctx.route,
      role: req.user.role,
      user: req.user,
      teamId: ctx.teamId,
      teamName: ctx.teamName,
      projectId: ctx.projectId,
      datasetId: ctx.datasetId,
      docSlug: ctx.docSlug,
      contributionSlug: ctx.contributionSlug,
      phiFilter: auth.phiFilter,
    });
  } catch (err) {
    console.error('ai chat prompt:', err);
    return res.status(500).json({ error: err.message || 'Chat error' });
  }

  const claudeArgs = {
    apiKey: auth.apiKey,
    model: auth.model,
    system,
    messages: messages.map((m) => ({ role: m.role, content: m.content })),
    maxTokens: ctx.mode === 'irb' || ctx.mode === 'analysis' ? 4096 : 2048,
  };

  const persistAndRecord = async (reply) => {
    const lastUser = [...messages].reverse().find((m) => m.role === 'user')?.content || '';
    const conv = await appendTurn(req.user.id, req.body.conversation_id, {
      userContent: lastUser,
      assistantContent: reply.text,
      messages: [...messages, { role: 'assistant', content: reply.text }],
    });

    recordUsageEventSafe({
      userId: req.user.id,
      eventType: 'ai.chat',
      durationMs: Date.now() - started,
      metadata: {
        team_id: ctx.teamId,
        billing_source: auth.billingSource,
        key_scope: auth.keyScope,
        model: reply.model,
        mode: ctx.mode,
        input_tokens: reply.usage.input_tokens,
        output_tokens: reply.usage.output_tokens,
        conversation_id: conv?.id,
      },
    });

    return conv;
  };

  if (!ctx.stream) {
    try {
      const reply = await createClaudeMessage(claudeArgs);
      const conv = await persistAndRecord(reply);
      res.json({
        reply: reply.text,
        conversation: conv,
        model: reply.model,
        usage: reply.usage,
        phi_safe: auth.phiFilter !== 'permissive',
      });
    } catch (err) {
      console.error('ai chat:', err);
      res.status(err.status || 500).json({ error: err.message || 'Chat error' });
    }
    return;
  }

  // Streaming (SSE) path — same auth/PHI gate above, same persistence/usage
  // accounting below, only the transport to the client differs.
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache, no-transform');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no'); // disable nginx/proxy response buffering
  res.flushHeaders?.();

  const send = (event) => res.write(`data: ${JSON.stringify(event)}\n\n`);

  try {
    const reply = await streamClaudeMessage({
      ...claudeArgs,
      onDelta: (text) => send({ type: 'delta', text }),
    });
    const conv = await persistAndRecord(reply);
    send({
      type: 'done',
      conversation: conv,
      model: reply.model,
      usage: reply.usage,
      phi_safe: auth.phiFilter !== 'permissive',
    });
  } catch (err) {
    console.error('ai chat stream:', err);
    send({ type: 'error', error: err.message || 'Chat error' });
  } finally {
    res.end();
  }
});

export default router;
