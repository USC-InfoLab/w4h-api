const ANTHROPIC_URL = 'https://api.anthropic.com/v1/messages';
const ANTHROPIC_MODELS_URL = 'https://api.anthropic.com/v1/models';
const ANTHROPIC_VERSION = '2023-06-01';

/** Family cost bands, most expensive first. */
const MODEL_COST_TIER = { fable: 0, opus: 1, sonnet: 2, haiku: 3 };

function modelCostTier(id) {
  const s = String(id || '').toLowerCase();
  if (s.includes('haiku')) return MODEL_COST_TIER.haiku;
  if (s.includes('sonnet')) return MODEL_COST_TIER.sonnet;
  if (s.includes('fable')) return MODEL_COST_TIER.fable;
  if (s.includes('opus')) return MODEL_COST_TIER.opus;
  return 4;
}

function modelVersionParts(id) {
  return String(id || '').match(/\d+/g)?.map(Number) || [];
}

/** Most expensive family first; within a family, newer version/date first. */
export function sortClaudeModelsByCost(models) {
  return [...models].sort((a, b) => {
    const ta = modelCostTier(a.id);
    const tb = modelCostTier(b.id);
    if (ta !== tb) return ta - tb;
    const va = modelVersionParts(a.id);
    const vb = modelVersionParts(b.id);
    const n = Math.max(va.length, vb.length);
    for (let i = 0; i < n; i += 1) {
      const da = va[i] ?? 0;
      const db = vb[i] ?? 0;
      if (da !== db) return db - da;
    }
    return String(a.id).localeCompare(String(b.id));
  });
}

/** Used when Anthropic cannot be reached or the key is missing/invalid. */
export const CLAUDE_MODEL_CATALOG = sortClaudeModelsByCost([
  { id: 'claude-haiku-4-5-20251001', name: 'Claude Haiku 4.5' },
  { id: 'claude-sonnet-4-5-20250929', name: 'Claude Sonnet 4.5' },
  { id: 'claude-sonnet-4-20250514', name: 'Claude Sonnet 4' },
  { id: 'claude-opus-4-1-20250805', name: 'Claude Opus 4.1' },
  { id: 'claude-opus-4-20250514', name: 'Claude Opus 4' },
]);

function anthropicHeaders(apiKey) {
  return {
    'content-type': 'application/json',
    'x-api-key': apiKey,
    'anthropic-version': ANTHROPIC_VERSION,
  };
}

function displayNameForModel(id, apiName) {
  if (apiName) return apiName;
  const hit = CLAUDE_MODEL_CATALOG.find((m) => m.id === id);
  if (hit) return hit.name;
  return id;
}

/**
 * Ask Anthropic whether this key works, and list models it may call.
 * Does not log or return the key. Time-bounded so Settings cannot hang.
 *
 * status: none | live | invalid | unreachable
 */
export async function inspectAnthropicKey(apiKey) {
  const key = String(apiKey || '').trim();
  if (!key) {
    return { status: 'none', detail: null, models: CLAUDE_MODEL_CATALOG };
  }
  try {
    const res = await fetch(ANTHROPIC_MODELS_URL, {
      headers: anthropicHeaders(key),
      signal: AbortSignal.timeout(8000),
    });
    if (res.status === 401 || res.status === 403) {
      const body = await res.json().catch(() => ({}));
      const detail = body?.error?.message || 'Anthropic rejected this key.';
      return { status: 'invalid', detail, models: CLAUDE_MODEL_CATALOG };
    }
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      const detail = body?.error?.message || `Anthropic returned ${res.status}`;
      return { status: 'unreachable', detail, models: CLAUDE_MODEL_CATALOG };
    }
    const body = await res.json().catch(() => ({}));
    const rows = [...(Array.isArray(body.data) ? body.data : [])];
    let hasMore = !!body.has_more;
    let afterId = body.last_id;
    for (let page = 0; hasMore && afterId && page < 4; page += 1) {
      const more = await fetch(`${ANTHROPIC_MODELS_URL}?after_id=${encodeURIComponent(afterId)}`, {
        headers: anthropicHeaders(key),
        signal: AbortSignal.timeout(8000),
      });
      if (!more.ok) break;
      const extra = await more.json().catch(() => ({}));
      if (Array.isArray(extra.data)) rows.push(...extra.data);
      hasMore = !!extra.has_more;
      afterId = extra.last_id;
    }
    const models = [];
    const seen = new Set();
    for (const row of rows) {
      const id = String(row.id || row.model || '').trim();
      if (!id || seen.has(id)) continue;
      seen.add(id);
      models.push({ id, name: displayNameForModel(id, row.display_name) });
    }
    return {
      status: 'live',
      detail: null,
      models: models.length ? sortClaudeModelsByCost(models) : CLAUDE_MODEL_CATALOG,
    };
  } catch (err) {
    return {
      status: 'unreachable',
      detail: err?.name === 'TimeoutError' ? 'Anthropic timed out.' : (err?.message || 'Could not reach Anthropic.'),
      models: CLAUDE_MODEL_CATALOG,
    };
  }
}

export async function createClaudeMessage({
  apiKey,
  model,
  system,
  messages,
  maxTokens = 2048,
}) {
  const res = await fetch(ANTHROPIC_URL, {
    method: 'POST',
    headers: anthropicHeaders(apiKey),
    body: JSON.stringify({
      model,
      max_tokens: maxTokens,
      system,
      messages,
    }),
  });

  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = body?.error?.message || body?.error || res.statusText;
    const err = new Error(`Claude request failed (${res.status}): ${detail}`);
    err.status = res.status >= 400 && res.status < 600 ? res.status : 502;
    throw err;
  }

  const text = Array.isArray(body.content)
    ? body.content.filter((p) => p.type === 'text').map((p) => p.text).join('\n')
    : '';

  return {
    text,
    model: body.model || model,
    usage: {
      input_tokens: body.usage?.input_tokens ?? 0,
      output_tokens: body.usage?.output_tokens ?? 0,
    },
  };
}

/**
 * Same call as createClaudeMessage, but reads the response as an SSE stream
 * and invokes onDelta(textChunk) as each content_block_delta arrives, so the
 * caller can forward text to a client incrementally instead of waiting for
 * the full reply. Resolves with the same {text, model, usage} shape once the
 * stream ends, for persistence/usage-accounting call sites that need the total.
 */
export async function streamClaudeMessage({
  apiKey,
  model,
  system,
  messages,
  maxTokens = 2048,
  onDelta,
  signal,
}) {
  const res = await fetch(ANTHROPIC_URL, {
    method: 'POST',
    headers: anthropicHeaders(apiKey),
    body: JSON.stringify({
      model,
      max_tokens: maxTokens,
      system,
      messages,
      stream: true,
    }),
    signal,
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const detail = body?.error?.message || body?.error || res.statusText;
    const err = new Error(`Claude request failed (${res.status}): ${detail}`);
    err.status = res.status >= 400 && res.status < 600 ? res.status : 502;
    throw err;
  }

  let text = '';
  let modelOut = model;
  let inputTokens = 0;
  let outputTokens = 0;
  let buffer = '';
  let sawStop = false;

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split('\n\n');
    buffer = events.pop() || '';
    for (const raw of events) {
      const dataLine = raw.split('\n').find((l) => l.startsWith('data:'));
      if (!dataLine) continue;
      let evt;
      try {
        evt = JSON.parse(dataLine.slice(5).trim());
      } catch {
        continue;
      }
      if (evt.type === 'message_start') {
        modelOut = evt.message?.model || modelOut;
        inputTokens = evt.message?.usage?.input_tokens ?? inputTokens;
      } else if (evt.type === 'content_block_delta' && evt.delta?.type === 'text_delta') {
        text += evt.delta.text;
        onDelta?.(evt.delta.text);
      } else if (evt.type === 'message_delta') {
        outputTokens = evt.usage?.output_tokens ?? outputTokens;
      } else if (evt.type === 'message_stop') {
        sawStop = true;
      } else if (evt.type === 'error') {
        const detail = evt.error?.message || 'Claude stream error';
        const err = new Error(detail);
        err.status = 502;
        throw err;
      }
    }
  }

  if (!sawStop) {
    const err = new Error('Claude stream ended without message_stop (connection dropped mid-reply)');
    err.status = 502;
    err.partialText = text;
    throw err;
  }

  return {
    text,
    model: modelOut,
    usage: { input_tokens: inputTokens, output_tokens: outputTokens },
  };
}
