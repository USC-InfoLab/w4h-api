import MarkdownIt from 'markdown-it';
import markdownItAnchor from 'markdown-it-anchor';
import markdownItTaskLists from 'markdown-it-task-lists';

/**
 * Heading fragment id / anchor slug. Must match w4h-app/src/utils/docsHeadingId.js (docsHeadingHashId).
 * @param {string} str
 * @returns {string}
 */
function headingHashFromTitle(str) {
  return String(str || '')
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[\p{Emoji_Presentation}\p{Extended_Pictographic}]/gu, '')
    .replace(/[^\w\s-]/g, '')
    .trim()
    .toLowerCase()
    .replace(/\s+/g, '-')
    .replace(/^-+|-+$/g, '');
}

function stripHTML(html) {
  if (!html) return '';
  return String(html).replace(/<[^>]*>/g, '');
}

/**
 * TOC derived from rendered HTML so ids match markdown-it-anchor output exactly.
 * @param {string} html
 * @returns {Array<{ id: string, level: number, title: string }>}
 */
function extractTocFromRenderedHtml(html) {
  if (!html) return [];
  const toc = [];
  const re = /<h([1-6])[^>]*\bid="([^"]+)"[^>]*>([\s\S]*?)<\/h\1>/gi;
  let m;
  while ((m = re.exec(html)) !== null) {
    const depth = parseInt(m[1], 10);
    const id = m[2];
    const title = stripHTML(m[3]).replace(/\s+/g, ' ').trim();
    toc.push({ id, level: depth + 1, title });
  }
  return toc;
}

function createMarkdownIt() {
  const md = new MarkdownIt({ html: true, breaks: true })
    .use(markdownItAnchor, { slugify: headingHashFromTitle })
    .use(markdownItTaskLists, {
      enabled: true,
      label: true,
      labelAfter: true,
      disabled: false,
    });

  const originalFenceRule = md.renderer.rules.fence;
  md.renderer.rules.fence = (tokens, idx, options, env, self) => {
    const token = tokens[idx];
    const lang = (token.info || '').trim().split(/\s+/g)[0];

    if (lang === 'mermaid') {
      return `<div id="mermaid-${idx}" class="mermaid">${token.content}</div>`;
    }
    return originalFenceRule(tokens, idx, options, env, self);
  };

  return md;
}

let mdInstance;

function getMd() {
  if (!mdInstance) mdInstance = createMarkdownIt();
  return mdInstance;
}

/**
 * Render stored markdown to HTML + TOC for persistence (do not trust client html/toc).
 * @param {string} mdSource
 * @returns {{ html: string, toc: object[] }}
 */
function renderDoc(mdSource) {
  const md = getMd();
  const html = md.render(mdSource || '');
  const toc = extractTocFromRenderedHtml(html);
  return { html, toc };
}

export { renderDoc, headingHashFromTitle, stripHTML, extractTocFromRenderedHtml };
