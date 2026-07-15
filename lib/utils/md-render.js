import MarkdownIt from 'markdown-it';
import markdownItAnchor from 'markdown-it-anchor';
import markdownItTaskLists from 'markdown-it-task-lists';
import markdownItAttrs from 'markdown-it-attrs';

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

function decodeHtmlEntities(text) {
  return String(text).replace(/&(#(?:x[\da-fA-F]+|\d+)|[a-zA-Z]+);/g, (match, entity) => {
    if (entity[0] === '#') {
      const code = entity[1].toLowerCase() === 'x'
        ? parseInt(entity.slice(2), 16)
        : parseInt(entity.slice(1), 10);
      return Number.isFinite(code) ? String.fromCodePoint(code) : match;
    }
    const named = { amp: '&', lt: '<', gt: '>', quot: '"', apos: "'", nbsp: '\u00a0' };
    return named[entity] ?? match;
  });
}

function stripHTML(html) {
  if (!html) return '';
  return decodeHtmlEntities(String(html).replace(/<[^>]*>/g, ''));
}

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
    })
    .use(markdownItAttrs);

  // Auto-add target="_blank" rel="noopener noreferrer" to external links
  const defaultLinkOpenRule = md.renderer.rules.link_open || ((tokens, idx, options, env, self) => self.renderToken(tokens, idx, options));
  md.renderer.rules.link_open = (tokens, idx, options, env, self) => {
    const token = tokens[idx];
    const hrefIndex = token.attrIndex('href');
    if (hrefIndex >= 0) {
      const href = token.attrs[hrefIndex][1];
      const isExternal = /^https?:\/\//i.test(href);
      if (isExternal) {
        token.attrSet('target', '_blank');
        token.attrSet('rel', 'noopener noreferrer');
      }
    }
    return defaultLinkOpenRule(tokens, idx, options, env, self);
  };

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
 * Drop a leading H1 from the body. Canonical page title lives in frontmatter /
 * the title field; the viewer renders that as the page H1.
 */
function stripLeadingTitleHeading(body, _title) {
  if (body == null || body === '') return body == null ? '' : String(body);

  let out = String(body).replace(/^\uFEFF?\s*/, '');

  const atx = out.match(/^#\s+.+?(?:\r?\n|$)/);
  if (atx) {
    return out.slice(atx[0].length).replace(/^\r?\n*/, '');
  }

  const htmlH1 = out.match(/^<h1\b[^>]*>[\s\S]*?<\/h1>\s*/i);
  if (htmlH1) {
    return out.slice(htmlH1[0].length);
  }

  return String(body);
}

function renderDoc(mdSource) {
  const md = getMd();
  const html = md.render(mdSource || '');
  const toc = extractTocFromRenderedHtml(html);
  return { html, toc };
}

export {
  renderDoc,
  headingHashFromTitle,
  stripHTML,
  extractTocFromRenderedHtml,
  stripLeadingTitleHeading,
};