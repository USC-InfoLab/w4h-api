import { createMermaidRenderer } from 'mermaid-isomorphic';

const render = createMermaidRenderer();

/**
 * Replace `div.mermaid` source blocks with rendered SVG so stored doc HTML
 * matches what the admin editor saves (plain v-html in the app).
 */
export async function transformMermaidHtml(html) {
  if (
    process.env.W4H_MERMAID_DISABLE === '1' ||
    process.env.W4H_MERMAID_DISABLE === 'true'
  ) {
    return html;
  }
  if (!html?.includes('mermaid')) return html;

  const diagrams = [...html.matchAll(/<div[^>]*class="mermaid"[^>]*>([\s\S]*?)<\/div>/g)];

  for (const [block, source] of diagrams) {
    try {
      const [result] = await render([source.trim()]);
      const svg = result?.value?.svg;
      if (result?.status === 'fulfilled' && svg) {
        html = html.replace(block, svg);
      } else if (result?.status === 'rejected') {
        console.warn('[docs-mermaid-render] Diagram render rejected:', result.reason?.message || result.reason);
      }
    } catch (e) {
      console.warn('[docs-mermaid-render] Failed to render diagram:', e.message);
    }
  }

  return html;
}

/**
 * Session-shaped wrapper so callers that open/close a render session still work.
 * mermaid-isomorphic does not need a long-lived browser handle.
 */
export async function createMermaidRenderSession() {
  if (
    process.env.W4H_MERMAID_DISABLE === '1' ||
    process.env.W4H_MERMAID_DISABLE === 'true'
  ) {
    return null;
  }

  return {
    transformHtml: transformMermaidHtml,
    async close() { /* no-op */ },
  };
}
