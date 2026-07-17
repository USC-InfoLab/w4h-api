import fs from 'fs-extra';
import MarkdownIt from 'markdown-it';
import markdownItAnchor from 'markdown-it-anchor';
import path from 'path';
import { execFile } from 'child_process';
import { promisify } from 'util';
import { parse as parseYaml, stringify as stringifyYaml } from 'yaml';

const execFileAsync = promisify(execFile);

import { query, DOCS_TABLE, DOC_TYPES_TABLE } from '../../db.js';
import { createMermaidRenderSession } from '../docs-mermaid-render.js';
import {
  headingHashFromTitle,
  extractTocFromRenderedHtml,
  stripLeadingTitleHeading,
} from './md-render.js';

////////////////////////////////////////////////////////////////////
// Import functionality.
////////////////////////////////////////////////////////////////////

function applyAppUrlTemplate(text) {
  if (text == null || text === '') return text;
  const base = process.env.APP_URL || '';
  return String(text).replace(/\{\{APP_URL\}\}/g, base);
}

function titleFromSlug(slug) {
  return String(slug)
    .split(/[-_]/g)
    .filter(Boolean)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
}

function createDocsMarkdownIt() {
  // Use the SAME heading slugify as the editor pipeline (md-render.js) so that
  // heading id="" anchors match the ids stored in the TOC — otherwise sidebar /
  // TOC deep-links point at fragments that don't exist on the page.
  const md = new MarkdownIt({ html: true, breaks: true })
    .use(markdownItAnchor, { slugify: headingHashFromTitle });

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

/** Manual frontmatter uses 1–49; ADR-001 … ADR-099 auto-fill 50+N when sort_order is omitted. */
function resolveSortOrder(slug, front) {
  if (Object.prototype.hasOwnProperty.call(front, 'sort_order')
      && front.sort_order !== '' && front.sort_order != null) {
    const n = +front.sort_order;
    return Number.isFinite(n) ? n : 0;
  }
  const m = String(slug).match(/^ADR-(\d+)/i);
  if (m) {
    return 50 + parseInt(m[1], 10);
  }
  return 0;
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

/** Repo index files — not imported as docs pages. */
const SKIP_IMPORT_BASENAMES = new Set([
  'readme',
  'license',
  'contributing',
  'code_of_conduct',
  'code-of-conduct',
  'changelog',
]);

function shouldSkipImportBasename(base) {
  const normalized = String(base).trim().toLowerCase().replace(/[\s-]+/g, '_');
  return SKIP_IMPORT_BASENAMES.has(normalized);
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
      if (shouldSkipImportBasename(base)) continue;
      files.push(full);
    }
  }
  return files;
}

/** Top-level folders in the docs repo define available doc types. */
async function collectRepoDocTypeSlugs(repoRoot) {
  const slugs = new Set();
  const entries = await fs.readdir(repoRoot, { withFileTypes: true });
  for (const entry of entries) {
    if (entry.isDirectory() && !entry.name.startsWith('.')) {
      slugs.add(entry.name);
    }
  }
  return slugs;
}

async function syncDocTypes(slugs) {
  const TYPE_RANK = {
    guide: 10,
    tutorial: 20,
    reference: 30,
    faq: 40,
    internal: 50,
    other: 60,
    dev: 70,
  };
  const ordered = [...slugs].sort(
    (a, b) => (TYPE_RANK[a] ?? 999) - (TYPE_RANK[b] ?? 999) || a.localeCompare(b)
  );
  for (const slug of ordered) {
    await query(
      `INSERT INTO ${DOC_TYPES_TABLE} (slug, label, sort_order)
       VALUES ($1, $2, $3)
       ON CONFLICT (slug) DO UPDATE SET
         label = EXCLUDED.label,
         sort_order = EXCLUDED.sort_order`,
      [slug, titleFromSlug(slug), TYPE_RANK[slug] ?? 999]
    );
  }
  return ordered.length;
}

/**
 * Resolve the "last updated" provenance for a markdown file on import.
 * Precedence: explicit frontmatter (updated_by / updated_at) > git history
 * (author email + commit date) > null (DB falls back to NOW()).
 * Returns { updatedBy, updatedAt } where either may be null.
 */
async function resolveDocProvenance(front, mdPath, repoPath) {
  let updatedBy =
    (typeof front.updated_by === 'string' && front.updated_by.trim()) || null;
  let updatedAt =
    (typeof front.updated_at === 'string' && front.updated_at.trim()) ||
    (front.updated_at instanceof Date ? front.updated_at.toISOString() : null);

  if (updatedBy && updatedAt) {
    return { updatedBy, updatedAt };
  }

  // Fall back to git history for whichever field is still missing.
  try {
    const { stdout } = await execFileAsync(
      'git',
      ['-C', repoPath, 'log', '-1', '--format=%ae%n%aI', '--', mdPath],
      { timeout: 5000 }
    );
    const [gitEmail, gitDate] = stdout.split('\n').map((s) => s.trim());
    if (!updatedBy && gitEmail) updatedBy = gitEmail;
    if (!updatedAt && gitDate) updatedAt = gitDate;
  } catch (e) {
    // Not a git checkout, git unavailable, or file untracked — leave as-is.
    console.warn(`[docs-repo-import] git provenance unavailable for "${path.basename(mdPath)}": ${e.message}`);
  }

  return { updatedBy, updatedAt };
}

async function importDocs() {
  const docsRepoPath = process.env.DOCS_REPO_PATH;

  if (!docsRepoPath) {
    return { imported: false, reason: 'DOCS_REPO_PATH environment variable is not set' };
  }

  const resolved = path.resolve(docsRepoPath);
  if (!(await fs.pathExists(resolved))) {
    return { imported: false, reason: `Docs repo path does not exist: ${resolved}` };
  }

  const mdFiles = await collectMarkdownFiles(resolved);
  if (mdFiles.length === 0) {
    return { imported: false, reason: `No markdown files found under ${resolved}` };
  }

  const mdRenderer = createDocsMarkdownIt();
  let inserted = 0;
  let mermaidSession = null;
  try {
    await query('BEGIN');
    mermaidSession = await createMermaidRenderSession();

    const repoDocTypes = await collectRepoDocTypeSlugs(resolved);
    try {
      const typeCount = await syncDocTypes(repoDocTypes);
      if (typeCount > 0) {
        console.info(`[docs-repo-import] Synced ${typeCount} doc type(s) from repo folders`);
      }
    } catch (e) {
      console.warn(`[docs-repo-import] doc_types sync skipped: ${e.message}`);
    }

    for (const mdPath of mdFiles) {
      const base = path.basename(mdPath, '.md');
      const mdFolderPath = path.dirname(path.resolve(mdPath));
      const docsRepoAbsolutePath = resolved;
      // w4h-docs/home.md -> doc_type = ''
      // w4h-docs/guide/quick-start.md -> doc_type = 'guide'
      const doc_type = mdFolderPath === docsRepoAbsolutePath ? '' : path.basename(mdFolderPath);
      const rawFile = await fs.readFile(mdPath, 'utf8');
      const { meta: front, body } = splitFrontmatter(rawFile);

      // slug from filename (spaces → hyphens); frontmatter slug is ignored
      const slug = base.replace(/\s+/g, '-');
      const title =
        (typeof front.title === 'string' && front.title.trim()) ||
        extractTitleFromMarkdown(body) ||
        titleFromSlug(slug);
      const description =
        (typeof front.description === 'string' && front.description.trim()) || '';

      // Title lives in frontmatter / DB; do not keep a duplicate leading H1 in the body
      const md = stripLeadingTitleHeading(body, title);
      const renderEnv = { docSlug: slug };
      let html = applyAppUrlTemplate(mdRenderer.render(md, renderEnv));
      if (mermaidSession) {
        try {
          html = await mermaidSession.transformHtml(html);
        } catch (e) {
          console.warn(`[docs-repo-import] Mermaid SSR failed for "${slug}": ${e.message}`);
        }
      }
      // Extract the TOC from the rendered HTML so its ids match the heading
      // anchors emitted by markdown-it-anchor above.
      const toc = extractTocFromRenderedHtml(html);

      const card_title = String(front.card_title ?? '');
      const action_label = String(front.action_label ?? 'Open');
      const sort_order = resolveSortOrder(slug, front);
      const hidden = !!front.hidden;
      const is_feature_card = !!front.is_feature_card;
      const show_in_docs = front.show_in_docs !== false;
      const card_order = +front.card_order || 0;

      const { updatedBy, updatedAt } = await resolveDocProvenance(front, mdPath, resolved);

      await query(
        `INSERT INTO ${DOCS_TABLE} (
          slug, title, card_title, description, action_label, doc_type,
          sort_order, hidden, is_feature_card, show_in_docs, card_order,
          md, html, toc, updated_by, updated_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14::jsonb, $15, COALESCE($16::timestamptz, NOW()))
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
          updated_by = EXCLUDED.updated_by,
          updated_at = EXCLUDED.updated_at`,
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
          JSON.stringify(toc),
          updatedBy,
          updatedAt
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
    title,
    description,
    ...(row.card_title && String(row.card_title).trim() ? { card_title: String(row.card_title).trim() } : {}),
    ...(row.action_label && String(row.action_label).trim() && row.action_label !== 'Open' ? { action_label: String(row.action_label).trim() } : {}),
    ...(row.sort_order != null && row.sort_order !== '' ? { sort_order: +row.sort_order } : {}),
    ...(row.hidden ? { hidden: true } : {}),
    ...(row.is_feature_card ? { is_feature_card: true } : {}),
    ...(row.show_in_docs === false ? { show_in_docs: false } : {}),
    ...(+row.card_order ? { card_order: +row.card_order } : {}),
    ...(row.updated_by && String(row.updated_by).trim() ? { updated_by: String(row.updated_by).trim() } : {}),
    ...(row.updated_at ? { updated_at: (row.updated_at instanceof Date ? row.updated_at.toISOString() : String(row.updated_at)) } : {}),
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
};