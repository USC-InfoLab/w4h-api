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
