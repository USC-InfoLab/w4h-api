const ANTHROPIC_URL = 'https://api.anthropic.com/v1/messages';
const ANTHROPIC_VERSION = '2023-06-01';

export async function createClaudeMessage({
  apiKey,
  model,
  system,
  messages,
  maxTokens = 2048,
}) {
  const res = await fetch(ANTHROPIC_URL, {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      'x-api-key': apiKey,
      'anthropic-version': ANTHROPIC_VERSION,
    },
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
}) {
  const res = await fetch(ANTHROPIC_URL, {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      'x-api-key': apiKey,
      'anthropic-version': ANTHROPIC_VERSION,
    },
    body: JSON.stringify({
      model,
      max_tokens: maxTokens,
      system,
      messages,
      stream: true,
    }),
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
      } else if (evt.type === 'error') {
        const detail = evt.error?.message || 'Claude stream error';
        const err = new Error(detail);
        err.status = 502;
        throw err;
      }
    }
  }

  return {
    text,
    model: modelOut,
    usage: { input_tokens: inputTokens, output_tokens: outputTokens },
  };
}
