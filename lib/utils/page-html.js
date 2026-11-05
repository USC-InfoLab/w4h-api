/**
 * Project pages stored as HTML (ADR-030): sanitize, add heading ids for the TOC,
 * and derive a Markdown copy for export.
 */
import sanitizeHtml from 'sanitize-html';
import TurndownService from 'turndown';
import { gfm } from 'turndown-plugin-gfm';
import { headingHashFromTitle, stripHTML, extractTocFromRenderedHtml } from './md-render.js';

const SAFE_COLOR = /^(#[0-9a-f]{3,8}|rgba?\([\d\s.,%]+\)|[a-z]+)$/i;

const SANITIZE_OPTIONS = {
  allowedTags: [
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'br', 'hr', 'blockquote', 'pre', 'code',
    'strong', 'b', 'em', 'i', 's', 'del', 'u', 'sub', 'sup', 'mark', 'span',
    'ul', 'ol', 'li', 'input', 'label',
    'a', 'img', 'figure', 'figcaption',
    'table', 'colgroup', 'col', 'thead', 'tbody', 'tfoot', 'tr', 'th', 'td',
    'div',
  ],
  allowedAttributes: {
    '*': ['id'],
    a: ['href', 'title', 'target', 'rel'],
    img: ['src', 'alt', 'title', 'width', 'height'],
    th: ['colspan', 'rowspan', 'colwidth', 'style'],
    td: ['colspan', 'rowspan', 'colwidth', 'style'],
    col: ['style', 'span'],
    span: ['style'],
    mark: ['style', 'data-color'],
    code: ['class'],
    pre: ['class'],
    div: ['class'],
    ul: ['class', 'data-type'],
    li: ['class', 'data-type', 'data-checked'],
    input: ['type', 'checked', 'disabled'],
    ol: ['start'],
  },
  allowedClasses: {
    div: ['mermaid'],
    code: [/^language-[\w-]+$/],
    pre: ['code-block'],
    ul: ['contains-task-list'],
    li: ['task-list-item'],
  },
  allowedStyles: {
    '*': {
      color: [SAFE_COLOR],
      'background-color': [SAFE_COLOR],
      'text-align': [/^(left|right|center|justify)$/],
      width: [/^\d+(\.\d+)?(px|%|em|rem)$/],
      'min-width': [/^\d+(\.\d+)?(px|%|em|rem)$/],
    },
  },
  allowedSchemes: ['http', 'https', 'mailto'],
  allowedSchemesByTag: { img: ['http', 'https'] },
  allowProtocolRelative: false,
  // Drop images whose source was removed (e.g. data: URLs).
  exclusiveFilter: (frame) => frame.tag === 'img' && !frame.attribs.src,
  transformTags: {
    a: (tagName, attribs) => {
      const href = attribs.href || '';
      if (/^https?:\/\//i.test(href)) {
        return { tagName, attribs: { ...attribs, target: '_blank', rel: 'noopener noreferrer' } };
      }
      return { tagName, attribs };
    },
    input: (tagName, attribs) =>
      attribs.type === 'checkbox'
        ? { tagName, attribs: { type: 'checkbox', disabled: 'disabled', ...(attribs.checked != null ? { checked: 'checked' } : {}) } }
        : { tagName: 'span', attribs: {} },
  },
};

/** Remove anything a page must not carry (scripts, handlers, iframes, javascript: URLs). */
export function sanitizePageHtml(html) {
  return sanitizeHtml(String(html || ''), SANITIZE_OPTIONS);
}

/** Give every heading a stable, unique id so the TOC can link to it. */
export function addHeadingIds(html) {
  const used = new Set();
  return String(html || '').replace(/<h([1-6])([^>]*)>([\s\S]*?)<\/h\1>/gi, (match, level, attrs, inner) => {
    const base = headingHashFromTitle(stripHTML(inner)) || 'section';
    let id = base;
    let n = 1;
    while (used.has(id)) id = `${base}-${n++}`;
    used.add(id);
    const rest = String(attrs).replace(/\s*\bid="[^"]*"/i, '');
    return `<h${level} id="${id}"${rest}>${inner}</h${level}>`;
  });
}

let turndown;
function getTurndown() {
  if (turndown) return turndown;
  turndown = new TurndownService({
    headingStyle: 'atx',
    codeBlockStyle: 'fenced',
    bulletListMarker: '-',
    emDelimiter: '*',
  });
  turndown.use(gfm);
  // A line break inside a table cell must stay on one Markdown row.
  turndown.addRule('cellBreak', {
    filter: (node) => {
      if (node.nodeName !== 'BR') return false;
      for (let p = node.parentNode; p; p = p.parentNode) {
        if (p.nodeName === 'TD' || p.nodeName === 'TH') return true;
      }
      return false;
    },
    replacement: () => '<br>',
  });
  // Keep colour and highlight as inline HTML; Markdown has no equivalent.
  turndown.keep(['mark', 'u', 'sub', 'sup']);
  return turndown;
}

/** Markdown copy of a page for export (lossy, ADR-030 / Doc-20). */
export function pageHtmlToMarkdown(html) {
  const src = String(html || '')
    // Mermaid blocks become fenced code so their line breaks survive.
    .replace(/<div class="mermaid">([\s\S]*?)<\/div>/g, '<pre><code class="language-mermaid">$1</code></pre>')
    // Editor tables carry column sizes and a <p> per cell; GFM tables need neither.
    .replace(/<colgroup>[\s\S]*?<\/colgroup>/g, '')
    .replace(/<(t[hd])([^>]*)>([\s\S]*?)<\/\1>/g, (_m, tag, attrs, inner) => {
      const flat = inner.replace(/<\/p>\s*<p[^>]*>/g, '<br>').replace(/^\s*<p[^>]*>|<\/p>\s*$/g, '');
      return `<${tag}${attrs}>${flat}</${tag}>`;
    });
  return getTurndown().turndown(src).trim() + '\n';
}

/** Sanitize, add heading ids, then derive the TOC and the Markdown copy. */
export function preparePageHtml(rawHtml) {
  const html = addHeadingIds(sanitizePageHtml(rawHtml));
  return { html, toc: extractTocFromRenderedHtml(html), md: pageHtmlToMarkdown(html) };
}
