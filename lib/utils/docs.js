import fs from 'fs-extra';
import MarkdownIt from 'markdown-it';
import path from 'path';
import { parse as parseYaml, stringify as stringifyYaml } from 'yaml';

import { query, DOCS_TABLE } from '../../db.js';
import { createMermaidRenderSession } from '../docs-mermaid-render.js';

////////////////////////////////////////////////////////////////////
// Import functionality.
////////////////////////////////////////////////////////////////////

function applyAppUrlTemplate(text) {
  if (text == null || text === '') return text;
  const base = process.env.APP_URL || process.env.VITE_APP_URL || '';
  return String(text).replace(/\{\{APP_URL\}\}/g, base);
}

function stripInlineTags(s) {
  return String(s || '').replace(/<[^>]*>/g, '');
}

function buildTocFromMarkdown(markdown) {
  const md = new MarkdownIt({ html: true, breaks: true });
  const tokens = md.parse(markdown || '', {});
  const toc = [];
  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i];
    if (token.type === 'heading_open') {
      const level = Number(token.tag.slice(1));
      if (level === 1) continue; // h1 is the doc title, skip it in TOC
      const inline = tokens[i + 1];
      if (inline && inline.type === 'inline') {
        const headingTitle = stripInlineTags(inline.content);
        const id = `${headingTitle.toLowerCase().replace(/[^\w]+/g, '-')}`;
        toc.push({ id, level: level + 1, title: headingTitle });
      }
    }
  }
  return toc;
}

function titleFromSlug(slug) {
  return String(slug)
    .split(/[-_]/g)
    .filter(Boolean)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
}

/** ```mermaid → div.mermaid (same as MdDocEditable); Playwright then turns these into SVG at import.
 *  Also rewrites relative *.md links to app doc routes: ../guide/quick-start.md → /docs/quick-start */
function createDocsMarkdownIt() {
  const md = new MarkdownIt({ html: true, breaks: true });

  const originalFence = md.renderer.rules.fence;
  md.renderer.rules.fence = (tokens, idx, options, env, self) => {
    const token = tokens[idx];
    const lang = (token.info || '').trim().split(/\s+/)[0];
    if (lang === 'mermaid') {
      env.mermaidFenceIndex = (env.mermaidFenceIndex ?? 0) + 1;
      const slugPart = String(env.docSlug || 'doc').replace(/[^a-zA-Z0-9_-]/g, '-');
      const id = `mermaid-${slugPart}-${env.mermaidFenceIndex}`;
      return `<div id="${id}" class="mermaid">${token.content}</div>\n`;
    }
    return originalFence(tokens, idx, options, env, self);
  };

  // Rewrite relative .md links to app doc routes so they work after import.
  // e.g. ../guide/quick-start.md → /docs/quick-start
  // Absolute links (http/https/{{APP_URL}}) are left untouched.
  const originalLinkOpen = md.renderer.rules.link_open || ((tokens, idx, options, env, self) => self.renderToken(tokens, idx, options));
  md.renderer.rules.link_open = (tokens, idx, options, env, self) => {
    const token = tokens[idx];
    const hrefIdx = token.attrIndex('href');
    if (hrefIdx >= 0) {
      const href = token.attrs[hrefIdx][1];
      if (href && !href.startsWith('http') && !href.startsWith('{{') && !href.startsWith('/') && !href.startsWith('#') && href.endsWith('.md')) {
        const slug = path.basename(href, '.md');
        token.attrs[hrefIdx][1] = `{{APP_URL}}/docs/${slug}`;
      }
    }
    return originalLinkOpen(tokens, idx, options, env, self);
  };

  return md;
}

function splitFrontmatter(raw) {
  if (!raw.startsWith('---\n')) {
    return { meta: {}, body: raw };
  }
  const end = raw.indexOf('\n---\n', 4);
  if (end === -1) {
    return { meta: {}, body: raw };
  }
  const yamlBlock = raw.slice(4, end);
  const body = raw.slice(end + 5);
  try {
    const meta = parseYaml(yamlBlock);
    return { meta: meta && typeof meta === 'object' ? meta : {}, body };
  } catch {
    return { meta: {}, body: raw };
  }
}

function extractTitleFromMarkdown(body) {
  const match = body.match(/^#\s+(.+)$/m);
  return match ? match[1].trim() : null;
}

async function collectMarkdownFiles(dir) {
  const files = [];
  if (!(await fs.pathExists(dir))) return files;
  const entries = await fs.readdir(dir, { withFileTypes: true });
  for (const e of entries) {
    const full = path.join(dir, e.name);
    if (e.isDirectory()) {
      files.push(...(await collectMarkdownFiles(full)));
    } else if (e.isFile() && e.name.endsWith('.md')) {
      const base = path.basename(e.name, '.md');
      // Skip all-caps filenames like README, LICENSE, CONTRIBUTING, CODE_OF_CONDUCT
      if (base === base.toUpperCase()) continue;
      files.push(full);
    }
  }
  return files;
}

async function importDocs() {
  const docsRepoPath = process.env.DOCS_REPO_PATH;

  const mdFiles = await collectMarkdownFiles(docsRepoPath);
  if (mdFiles.length === 0) {
    return { imported: false, reason: `No markdown files found under ${docsRepoPath}` };
  }

  const mdRenderer = createDocsMarkdownIt();
  let inserted = 0;
  let mermaidSession = null;
  try {
    await query('BEGIN');
    mermaidSession = await createMermaidRenderSession();

    for (const mdPath of mdFiles) {
      const base = path.basename(mdPath, '.md');
      const rawFile = await fs.readFile(mdPath, 'utf8');
      const { meta: front, body } = splitFrontmatter(rawFile);

      const slug = (typeof front.slug === 'string' && front.slug.trim()) || base;
      const title =
        (typeof front.title === 'string' && front.title.trim()) ||
        extractTitleFromMarkdown(body) ||
        titleFromSlug(slug);
      const description =
        (typeof front.description === 'string' && front.description.trim()) || '';

      const md = body;
      const renderEnv = { docSlug: slug };
      let html = applyAppUrlTemplate(mdRenderer.render(md, renderEnv));
      if (mermaidSession) {
        try {
          html = await mermaidSession.transformHtml(html);
        } catch (e) {
          console.warn(`[docs-repo-import] Mermaid SSR failed for "${slug}": ${e.message}`);
        }
      }
      const toc = buildTocFromMarkdown(md);

      const card_title = String(front.card_title ?? '');
      const action_label = String(front.action_label ?? 'Open');
      const doc_type = String(front.doc_type ?? '');
      const sort_order = +front.sort_order || 0;
      const hidden = !!front.hidden;
      const is_feature_card = !!front.is_feature_card;
      const show_in_docs = front.show_in_docs !== false;
      const card_order = +front.card_order || 0;

      await query(
        `INSERT INTO ${DOCS_TABLE} (
          slug, title, card_title, description, action_label, doc_type,
          sort_order, hidden, is_feature_card, show_in_docs, card_order,
          md, html, toc, updated_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14::jsonb, NOW())
        ON CONFLICT (slug) DO UPDATE SET
          title = EXCLUDED.title,
          card_title = EXCLUDED.card_title,
          description = EXCLUDED.description,
          action_label = EXCLUDED.action_label,
          doc_type = EXCLUDED.doc_type,
          sort_order = EXCLUDED.sort_order,
          hidden = EXCLUDED.hidden,
          is_feature_card = EXCLUDED.is_feature_card,
          show_in_docs = EXCLUDED.show_in_docs,
          card_order = EXCLUDED.card_order,
          md = EXCLUDED.md,
          html = EXCLUDED.html,
          toc = EXCLUDED.toc,
          updated_at = NOW()`,
        [
          slug,
          title,
          card_title,
          description,
          action_label,
          doc_type,
          sort_order,
          hidden,
          is_feature_card,
          show_in_docs,
          card_order,
          md,
          html,
          JSON.stringify(toc)
        ]
      );
      inserted++;
    }

    if (inserted === 0) {
      await query('ROLLBACK');
      return { imported: false, reason: 'No documentation pages imported (missing .md files?)' };
    }

    await query('COMMIT');
    return { imported: true, count: inserted };
  } catch (e) {
    try {
      await query('ROLLBACK');
    } catch {
      /* ignore */
    }
    throw e;
  } finally {
    if (mermaidSession) {
      await mermaidSession.close().catch(() => { });
    }
  }
}

////////////////////////////////////////////////////////////////////
// Export functionality.
////////////////////////////////////////////////////////////////////

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

/** Zip path for exported markdown: <doc_type>/<slug>.md or <slug>.md when doc_type is empty.
 *  Produces paths that match the w4h-docs repo layout so the zip can be extracted on top of it. */
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
    return `${safe}.md`;
  }
  return `${typeDir}/${safe}.md`;
}

/** Markdown file body matching w4h-documentation/content (YAML frontmatter + body).
 *  The `name` frontmatter field stores the slug so the correct filename can be recovered on re-import. */
function markdownFileForExport(row) {
  const raw = row.md == null ? '' : String(row.md);
  const { body } = splitFrontmatter(raw);
  const slug = String(row.slug ?? '');
  const derivedTitle = slug
    .split(/[-_]/g)
    .filter(Boolean)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
  const title = (row.title != null && String(row.title).trim()) || derivedTitle;
  const description = row.description != null ? String(row.description).trim() : '';
  const fm = {
    name: slug,
    title,
    description,
    ...(row.card_title != null && { card_title: String(row.card_title) }),
    ...(row.action_label != null && { action_label: String(row.action_label) }),
    ...(row.doc_type != null && row.doc_type !== '' && { doc_type: String(row.doc_type) }),
    sort_order: +row.sort_order || 0,
    hidden: !!row.hidden,
    is_feature_card: !!row.is_feature_card,
    show_in_docs: row.show_in_docs !== false,
    card_order: +row.card_order || 0,
  };
  const fmYaml = stringifyYaml(fm).trimEnd();
  let out = `---\n${fmYaml}\n---\n${body}`;
  out = restoreAppUrlPlaceholder(out);
  return out;
}

export {
  importDocs,
  restoreAppUrlPlaceholder,
  markdownPathFromDocType,
  markdownFileForExport,
  splitFrontmatter
}