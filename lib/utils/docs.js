import fs from 'fs-extra';
import MarkdownIt from 'markdown-it';
import markdownItAnchor from 'markdown-it-anchor';
import markdownItAttrs from 'markdown-it-attrs';
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

function publicApiBaseUrl() {
  const raw = process.env.API_PUBLIC_URL || process.env.API_URL || '';
  if (raw) return String(raw).replace(/\/$/, '');
  const port = process.env.PORT || '2026';
  return `http://localhost:${port}`;
}

function applyAppUrlTemplate(text) {
  if (text == null || text === '') return text;
  const appBase = process.env.APP_URL || '';
  const apiBase = publicApiBaseUrl();
  // markdown-it percent-encodes {{…}} inside src/href attrs
  return String(text)
    .replace(/%7B%7BAPP_URL%7D%7D/gi, appBase)
    .replace(/\{\{APP_URL\}\}/g, appBase)
    .replace(/%7B%7BAPI_URL%7D%7D/gi, apiBase)
    .replace(/\{\{API_URL\}\}/g, apiBase);
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
    .use(markdownItAnchor, { slugify: headingHashFromTitle })
    // Same as md-render.js / MdDocEditable — supports [text](url){target="_blank"} etc.
    .use(markdownItAttrs);

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
      const isExternal = /^https?:\/\//i.test(href);
      if (isExternal && token.attrIndex('target') < 0) {
        token.attrSet('target', '_blank');
      }
      if (token.attrIndex('target') >= 0 && token.attrIndex('rel') < 0) {
        token.attrSet('rel', 'noopener noreferrer');
      }
    }
    return originalLinkOpen(tokens, idx, options, env, self);
  };

  // Repo media paths → /docs-media/… (same-origin SPA)
  // Accept: media/…, ./media/…, /media/… (Cursor multi-root workspace-absolute)
  const originalImage = md.renderer.rules.image || ((tokens, idx, options, env, self) => self.renderToken(tokens, idx, options));
  md.renderer.rules.image = (tokens, idx, options, env, self) => {
    const token = tokens[idx];
    const srcIdx = token.attrIndex('src');
    if (srcIdx >= 0) {
      let src = token.attrs[srcIdx][1] || '';
      src = src.replace(/^\.\//, '');
      if (src.startsWith('/media/')) {
        token.attrs[srcIdx][1] = `/docs-media/${src.slice('/media/'.length)}`;
      } else if (src.startsWith('media/')) {
        token.attrs[srcIdx][1] = `/docs-media/${src.slice('media/'.length)}`;
      }
    }
    return originalImage(tokens, idx, options, env, self);
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
    releases: 5,
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

function resolveDocsRepoOrError() {
  const docsRepoPath = process.env.DOCS_REPO_PATH;
  if (!docsRepoPath) {
    return { ok: false, reason: 'DOCS_REPO_PATH environment variable is not set' };
  }
  const resolved = path.resolve(docsRepoPath);
  return { ok: true, resolved };
}

/** Normalize markdown for DB↔repo equality checks (ignore EOL / trailing ws). */
function normalizeMdForCompare(md) {
  return String(md ?? '')
    .replace(/\r\n/g, '\n')
    .replace(/[ \t]+$/gm, '')
    .trim();
}

function compareFingerprint({
  title,
  description,
  card_title,
  action_label,
  doc_type,
  sort_order,
  hidden,
  is_feature_card,
  show_in_docs,
  card_order,
  md,
}) {
  return JSON.stringify({
    title: String(title ?? '').trim(),
    description: String(description ?? '').trim(),
    card_title: String(card_title ?? '').trim(),
    action_label: String(action_label ?? 'Open').trim() || 'Open',
    doc_type: String(doc_type ?? '').trim(),
    sort_order: Number.isFinite(+sort_order) ? +sort_order : 0,
    hidden: !!hidden,
    is_feature_card: !!is_feature_card,
    show_in_docs: show_in_docs !== false,
    card_order: +card_order || 0,
    md: normalizeMdForCompare(md),
  });
}

/**
 * Parse a repo markdown file the same way import does.
 * @returns {{ slug, relPath, mdPath, title, description, card_title, action_label,
 *   doc_type, sort_order, hidden, is_feature_card, show_in_docs, card_order, md, front }}
 */
async function parseRepoDocFile(mdPath, repoRoot) {
  const base = path.basename(mdPath, '.md');
  const mdFolderPath = path.dirname(path.resolve(mdPath));
  // w4h-docs/home.md -> doc_type = ''
  // w4h-docs/guide/quick-start.md -> doc_type = 'guide'
  const doc_type = mdFolderPath === path.resolve(repoRoot) ? '' : path.basename(mdFolderPath);
  const rawFile = await fs.readFile(mdPath, 'utf8');
  const { meta: front, body } = splitFrontmatter(rawFile);
  const slug = base.replace(/\s+/g, '-');
  const title =
    (typeof front.title === 'string' && front.title.trim()) ||
    extractTitleFromMarkdown(body) ||
    titleFromSlug(slug);
  const description =
    (typeof front.description === 'string' && front.description.trim()) || '';
  const md = stripLeadingTitleHeading(body, title);
  return {
    slug,
    relPath: path.relative(repoRoot, mdPath).split(path.sep).join('/'),
    mdPath,
    title,
    description,
    card_title: String(front.card_title ?? ''),
    action_label: String(front.action_label ?? 'Open'),
    doc_type,
    sort_order: resolveSortOrder(slug, front),
    hidden: !!front.hidden,
    is_feature_card: !!front.is_feature_card,
    show_in_docs: front.show_in_docs !== false,
    card_order: +front.card_order || 0,
    md,
    front,
  };
}

async function loadRepoDocsBySlug(repoRoot) {
  const mdFiles = await collectMarkdownFiles(repoRoot);
  const bySlug = new Map();
  for (const mdPath of mdFiles) {
    const parsed = await parseRepoDocFile(mdPath, repoRoot);
    if (bySlug.has(parsed.slug)) {
      console.warn(
        `[docs-repo-import] Duplicate slug "${parsed.slug}" — using ${parsed.relPath}`
      );
    }
    bySlug.set(parsed.slug, parsed);
  }
  return bySlug;
}

/** List uncommitted / untracked markdown files under the docs repo (warn-only). */
async function listUncommittedMarkdown(repoPath) {
  try {
    const { stdout } = await execFileAsync(
      'git',
      ['-C', repoPath, 'status', '--porcelain', '-uall', '--', '.'],
      { timeout: 8000 }
    );
    const files = [];
    for (const line of stdout.split('\n')) {
      if (!line.trim()) continue;
      // porcelain: XY PATH or XY ORIG -> PATH
      const pathPart = line.slice(3).includes(' -> ')
        ? line.slice(3).split(' -> ').pop()
        : line.slice(3);
      const rel = pathPart.replace(/^"|"$/g, '').trim();
      if (!rel.toLowerCase().endsWith('.md')) continue;
      const base = path.basename(rel, '.md');
      if (shouldSkipImportBasename(base)) continue;
      const code = line.slice(0, 2);
      files.push({
        path: rel.split(path.sep).join('/'),
        status: code.trim() || code,
        slug: base.replace(/\s+/g, '-'),
      });
    }
    return files;
  } catch (e) {
    console.warn(`[docs-repo-import] git status unavailable: ${e.message}`);
    return [];
  }
}

async function upsertDocFromParsed(parsed, repoRoot, mdRenderer, mermaidSession) {
  const {
    slug, title, description, card_title, action_label, doc_type,
    sort_order, hidden, is_feature_card, show_in_docs, card_order, md, front, mdPath,
  } = parsed;

  const renderEnv = { docSlug: slug };
  let html = mdRenderer.render(md, renderEnv);
  // Raw HTML <img>/<video> using repo media paths
  html = html.replace(
    /\b(src|poster)=(["'])\.?\/?media\//gi,
    `$1=$2/docs-media/`
  );
  html = html.replace(
    /\b(src|poster)=(["'])\/media\//gi,
    `$1=$2/docs-media/`
  );
  // Legacy imports that baked {{APP_URL}} or its percent-encoded form into media srcs
  html = html.replace(
    /(?:\{\{APP_URL\}\}|%7B%7BAPP_URL%7D%7D)\/docs-media\//gi,
    '/docs-media/'
  );
  html = applyAppUrlTemplate(html);
  if (mermaidSession) {
    try {
      html = await mermaidSession.transformHtml(html);
    } catch (e) {
      console.warn(`[docs-repo-import] Mermaid SSR failed for "${slug}": ${e.message}`);
    }
  }
  const toc = extractTocFromRenderedHtml(html);
  const { updatedBy, updatedAt } = await resolveDocProvenance(front, mdPath, repoRoot);

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
      slug, title, card_title, description, action_label, doc_type,
      sort_order, hidden, is_feature_card, show_in_docs, card_order,
      md, html, JSON.stringify(toc), updatedBy, updatedAt,
    ]
  );
}

/**
 * Compare DB docs to the source repo. Does not mutate anything.
 * @returns {{ ok, reason?, matching, conflicts, dbOnly, repoOnly, uncommitted }}
 */
async function previewDocsSync() {
  const repo = resolveDocsRepoOrError();
  if (!repo.ok) return { ok: false, reason: repo.reason };

  if (!(await fs.pathExists(repo.resolved))) {
    return { ok: false, reason: `Docs repo path does not exist: ${repo.resolved}` };
  }

  const repoBySlug = await loadRepoDocsBySlug(repo.resolved);
  if (repoBySlug.size === 0) {
    return { ok: false, reason: `No markdown files found under ${repo.resolved}` };
  }

  const dbResult = await query(
    `SELECT slug, title, description, card_title, action_label, doc_type,
            sort_order, hidden, is_feature_card, show_in_docs, card_order, md, updated_at
     FROM ${DOCS_TABLE}`
  );
  const dbBySlug = new Map(dbResult.rows.map((r) => [r.slug, r]));

  const matching = [];
  const conflicts = [];
  const dbOnly = [];
  const repoOnly = [];

  for (const [slug, repoDoc] of repoBySlug) {
    const dbDoc = dbBySlug.get(slug);
    if (!dbDoc) {
      repoOnly.push({
        slug,
        path: repoDoc.relPath,
        title: repoDoc.title,
        doc_type: repoDoc.doc_type,
      });
      continue;
    }
    const same =
      compareFingerprint(repoDoc) ===
      compareFingerprint({
        title: dbDoc.title,
        description: dbDoc.description,
        card_title: dbDoc.card_title,
        action_label: dbDoc.action_label,
        doc_type: dbDoc.doc_type,
        sort_order: dbDoc.sort_order,
        hidden: dbDoc.hidden,
        is_feature_card: dbDoc.is_feature_card,
        show_in_docs: dbDoc.show_in_docs,
        card_order: dbDoc.card_order,
        md: dbDoc.md,
      });
    const entry = {
      slug,
      path: repoDoc.relPath,
      title: dbDoc.title || repoDoc.title,
      doc_type: repoDoc.doc_type || dbDoc.doc_type || '',
      repoTitle: repoDoc.title,
      platformTitle: dbDoc.title,
    };
    if (same) matching.push(entry);
    else conflicts.push(entry);
  }

  for (const [slug, dbDoc] of dbBySlug) {
    if (repoBySlug.has(slug)) continue;
    dbOnly.push({
      slug,
      title: dbDoc.title,
      doc_type: dbDoc.doc_type || '',
      updated_at: dbDoc.updated_at,
    });
  }

  const sortBySlug = (a, b) => String(a.slug).localeCompare(String(b.slug));
  matching.sort(sortBySlug);
  conflicts.sort(sortBySlug);
  dbOnly.sort(sortBySlug);
  repoOnly.sort(sortBySlug);

  const uncommitted = await listUncommittedMarkdown(repo.resolved);

  return {
    ok: true,
    repoPath: repo.resolved,
    matching,
    conflicts,
    dbOnly,
    repoOnly,
    uncommitted,
    counts: {
      matching: matching.length,
      conflicts: conflicts.length,
      dbOnly: dbOnly.length,
      repoOnly: repoOnly.length,
      uncommitted: uncommitted.length,
      repo: repoBySlug.size,
      platform: dbBySlug.size,
    },
  };
}

/**
 * Apply per-file sync decisions.
 * decisions.conflicts[slug]: 'repo' | 'platform'  (platform = skip overwrite)
 * decisions.dbOnly[slug]: 'keep' | 'delete'
 * decisions.repoOnly[slug]: 'import' | 'skip'
 * Omitted keys use defaults: conflicts→repo, dbOnly→keep, repoOnly→import.
 * Matching pages are always re-imported from the repo.
 */
async function applyDocsSync(decisions = {}) {
  const repo = resolveDocsRepoOrError();
  if (!repo.ok) return { imported: false, reason: repo.reason };

  if (!(await fs.pathExists(repo.resolved))) {
    return { imported: false, reason: `Docs repo path does not exist: ${repo.resolved}` };
  }

  const repoBySlug = await loadRepoDocsBySlug(repo.resolved);
  if (repoBySlug.size === 0) {
    return { imported: false, reason: `No markdown files found under ${repo.resolved}` };
  }

  const conflictActions = decisions.conflicts && typeof decisions.conflicts === 'object'
    ? decisions.conflicts : {};
  const dbOnlyActions = decisions.dbOnly && typeof decisions.dbOnly === 'object'
    ? decisions.dbOnly : {};
  const repoOnlyActions = decisions.repoOnly && typeof decisions.repoOnly === 'object'
    ? decisions.repoOnly : {};

  const skipSlugs = new Set();
  for (const [slug, action] of Object.entries(conflictActions)) {
    if (action === 'platform') skipSlugs.add(slug);
  }
  for (const [slug, action] of Object.entries(repoOnlyActions)) {
    if (action === 'skip') skipSlugs.add(slug);
  }

  const deleteSlugs = Object.entries(dbOnlyActions)
    .filter(([, action]) => action === 'delete')
    .map(([slug]) => slug);

  const mdRenderer = createDocsMarkdownIt();
  let imported = 0;
  let deleted = 0;
  let skipped = 0;
  let mermaidSession = null;

  try {
    await query('BEGIN');
    mermaidSession = await createMermaidRenderSession();

    const repoDocTypes = await collectRepoDocTypeSlugs(repo.resolved);
    try {
      const typeCount = await syncDocTypes(repoDocTypes);
      if (typeCount > 0) {
        console.info(`[docs-repo-import] Synced ${typeCount} doc type(s) from repo folders`);
      }
    } catch (e) {
      console.warn(`[docs-repo-import] doc_types sync skipped: ${e.message}`);
    }

    for (const [slug, parsed] of repoBySlug) {
      if (skipSlugs.has(slug)) {
        skipped++;
        continue;
      }
      await upsertDocFromParsed(parsed, repo.resolved, mdRenderer, mermaidSession);
      imported++;
    }

    for (const slug of deleteSlugs) {
      // Only delete platform-only docs (never delete a slug that still exists in the repo)
      if (repoBySlug.has(slug)) continue;
      const result = await query(
        `DELETE FROM ${DOCS_TABLE} WHERE slug = $1 RETURNING slug`,
        [slug]
      );
      if (result.rowCount > 0) deleted++;
    }

    if (imported === 0 && deleted === 0) {
      await query('ROLLBACK');
      return {
        imported: false,
        reason: skipped > 0
          ? 'No pages imported (all selected actions skipped overwrite/import)'
          : 'No documentation pages imported (missing .md files?)',
        skipped,
      };
    }

    await query('COMMIT');
    return { imported: true, count: imported, deleted, skipped };
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

/** Full import from repo (db-init / legacy). Overwrites all matching slugs. */
async function importDocs() {
  return applyDocsSync({});
}

////////////////////////////////////////////////////////////////////
// Export functionality.
////////////////////////////////////////////////////////////////////

function restoreAppUrlPlaceholder(text) {
  if (text == null || text === '') return text;
  let s = String(text);
  const restore = (envKeys, placeholder, fallbackBase) => {
    const bases = [];
    for (const v of envKeys) {
      const t = typeof v === 'string' ? v.trim() : '';
      if (!t) continue;
      bases.push(t.replace(/\/$/, ''));
      bases.push(t.endsWith('/') ? t : `${t}/`);
    }
    if (fallbackBase) {
      bases.push(fallbackBase);
      bases.push(`${fallbackBase}/`);
    }
    const unique = [...new Set(bases.filter(Boolean))].sort((a, b) => b.length - a.length);
    for (const base of unique) {
      const escaped = base.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      const re = new RegExp(
        `${escaped}(?=/|\\?|#|\\)|\\s|"|'|[,;.:!?]|$)`,
        'g'
      );
      s = s.replace(re, placeholder);
    }
  };
  restore(
    [process.env.API_PUBLIC_URL, process.env.API_URL],
    '{{API_URL}}',
    publicApiBaseUrl()
  );
  restore(
    [process.env.APP_URL, process.env.VITE_APP_URL],
    '{{APP_URL}}',
    null
  );
  // Prefer repo-relative media paths when exporting
  s = s.replace(/\{\{APP_URL\}\}\/docs-media\//g, 'media/');
  s = s.replace(/\{\{API_URL\}\}\/docs-media\//g, 'media/');
  s = s.replace(/(src|poster)=(["'])\/docs-media\//gi, `$1=$2/media/`);
  s = s.replace(/\]\(\/docs-media\//g, '](/media/');
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
  previewDocsSync,
  applyDocsSync,
  restoreAppUrlPlaceholder,
  markdownPathFromDocType,
  markdownFileForExport,
  splitFrontmatter
};