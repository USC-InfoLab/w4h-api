import archiver from 'archiver';
import { stringify as stringifyYaml } from 'yaml';

import { splitFrontmatter } from './docs-repo-import.js';
import { USER_ROLES } from './utils/user.js';
import { query } from '../db.js';

/**
 * Reverse applyAppUrlTemplate (docs-repo-import): put {{APP_URL}} back for repo-friendly exports.
 * Tries APP_URL and VITE_APP_URL plus slash/no-slash variants.
 */
function restoreAppUrlPlaceholder(text) {
  if (text == null || text === '') return text;
  let s = String(text);
  const bases = [];
  for (const v of [process.env.APP_URL, process.env.VITE_APP_URL]) {
    const t = typeof v === 'string' ? v.trim() : '';
    if (!t) continue;
    bases.push(t);
    bases.push(t.endsWith('/') ? t.slice(0, -1) : `${t}/`);
  }
  const unique = [...new Set(bases)].sort((a, b) => b.length - a.length);
  for (const base of unique) {
    const escaped = base.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const re = new RegExp(
      `${escaped}(?=/|\\?|#|\\)|\\s|"|'|[,;.:!?]|$)`,
      'g'
    );
    s = s.replace(re, '{{APP_URL}}');
  }
  return s;
}

/**
 * Root-relative /docs/... links (e.g. after import with empty APP_URL) → {{APP_URL}}/docs/...
 * Matches HTML href= and Markdown () links used in w4h-documentation/content.
 */
function restoreRootRelativeAppDocLinks(text) {
  let s = String(text);
  s = s.replace(/href\s*=\s*"\/docs\//gi, 'href="{{APP_URL}}/docs/');
  s = s.replace(/href\s*=\s*"\/docs"/gi, 'href="{{APP_URL}}/docs"');
  s = s.replace(/href\s*=\s*'\/docs\//gi, "href='{{APP_URL}}/docs/");
  s = s.replace(/href\s*=\s*'\/docs'/gi, "href='{{APP_URL}}/docs'");
  s = s.replace(/\]\(\s*\/docs\//g, ']({{APP_URL}}/docs/');
  s = s.replace(/\]\(\s*\/docs#/g, ']({{APP_URL}}/docs#');
  s = s.replace(/\]\(\s*\/docs\?/g, ']({{APP_URL}}/docs?');
  s = s.replace(/\]\(\s*\/docs\)/g, ']({{APP_URL}}/docs)');
  return s;
}

/** Zip path for exported markdown: content/<doc_type>/<slug>.md or content/<slug>.md when doc_type is empty. */
function markdownPathFromDocType(slug, docType) {
  const safe =
    String(slug ?? 'doc')
      .replace(/[/\\?%*:|"<>]/g, '-')
      .replace(/^\.+/, '') || 'doc';
  const typeDir =
    String(docType ?? '')
      .trim()
      .toLowerCase()
      .replace(/[/\\?%*:|"<>]/g, '-')
      .replace(/^\.+/, '')
      .replace(/\s+/g, '-') || '';
  if (!typeDir) {
    return `content/${safe}.md`;
  }
  return `content/${typeDir}/${safe}.md`;
}

/** Markdown file body matching w4h-documentation/content (YAML frontmatter + body). */
function markdownFileForExport(row) {
  const raw = row.md == null ? '' : String(row.md);
  const { body } = splitFrontmatter(raw);
  const titleFromSlug = String(row.slug ?? '')
    .split(/[-_]/g)
    .filter(Boolean)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
  const title =
    (row.title != null && String(row.title).trim()) || titleFromSlug;
  const description = row.description != null ? String(row.description).trim() : '';
  const fmYaml = stringifyYaml({ title, description }).trimEnd();
  let out = `---\n${fmYaml}\n---\n${body}`;
  out = restoreRootRelativeAppDocLinks(out);
  out = restoreAppUrlPlaceholder(out);
  return out;
}

export {
  restoreAppUrlPlaceholder,
  restoreRootRelativeAppDocLinks,
  markdownPathFromDocType,
  markdownFileForExport
}