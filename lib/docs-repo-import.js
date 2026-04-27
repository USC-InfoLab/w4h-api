import axios from 'axios';
import db from '../db.js';
import { createMermaidRenderSession } from './docs-mermaid-render.js';
import fs from 'fs-extra';
import MarkdownIt from 'markdown-it';
import os from 'os';
import path from 'path';
import { fileURLToPath } from 'url';
import { extract } from 'tar';
import { parse as parseYaml } from 'yaml';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
/** Slug → card metadata; markdown still comes from the docs repo under content/ */
const DOCS_CONFIG_PATH = path.join(__dirname, '../data/documentation-configuration.yml');

const DOCS_TABLE = 'w4h.docs';

// Public docs: https://github.com/USC-InfoLab/w4h-documentation (main branch tarball)
const DEFAULT_W4H_DOCS_ARCHIVE_URL =
  'https://github.com/USC-InfoLab/w4h-documentation/archive/refs/heads/main.tar.gz';

async function downloadDocsArchiveToTempRoot(archiveUrl) {
  const tmp = await fs.mkdtemp(path.join(os.tmpdir(), 'w4h-doc-import-'));
  const tgzPath = path.join(tmp, 'repo.tar.gz');
  const extractDir = path.join(tmp, 'extract');
  await fs.mkdirp(extractDir);

  const res = await axios.get(archiveUrl, {
    responseType: 'arraybuffer',
    maxRedirects: 5,
    maxContentLength: 50 * 1024 * 1024
  });
  await fs.writeFile(tgzPath, Buffer.from(res.data));
  await extract({ file: tgzPath, cwd: extractDir });

  const entries = await fs.readdir(extractDir);
  if (entries.length !== 1) {
    throw new Error(`Expected one top-level directory in archive, got: ${entries.join(', ')}`);
  }
  const root = path.join(extractDir, entries[0]);
  if (!(await fs.pathExists(path.join(root, 'content')))) {
    throw new Error(
      `Archive missing content/ under ${root}. ` +
        'Set W4H_DOCS_ARCHIVE_URL / DOCS_REPO_DIR to an archive or tree that includes content/*.md.'
    );
  }
  return root;
}

/**
 * Resolves a filesystem path to the documentation repo root (contains content/ with .md).
 * Doc manifest (slug → card metadata) is read from data/documentation-configuration.yml in w4h-api.
 * Order: W4H_DOCS_ARCHIVE_URL → DOCS_REPO_DIR (if valid) → default USC-InfoLab main tarball.
 */
export async function resolveDocumentationRepoRoot() {
  const archiveUrl = process.env.W4H_DOCS_ARCHIVE_URL?.trim();
  if (archiveUrl) {
    return downloadDocsArchiveToTempRoot(archiveUrl);
  }

  const dir = process.env.DOCS_REPO_DIR?.trim();
  if (dir) {
    const resolved = path.isAbsolute(dir) ? dir : path.resolve(process.cwd(), dir);
    if (await fs.pathExists(path.join(resolved, 'content'))) {
      return resolved;
    }
  }

  return downloadDocsArchiveToTempRoot(DEFAULT_W4H_DOCS_ARCHIVE_URL);
}

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

export function splitFrontmatter(raw) {
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

async function findMarkdownFile(contentRoot, slug) {
  async function walk(dir) {
    if (!(await fs.pathExists(dir))) return null;
    const entries = await fs.readdir(dir, { withFileTypes: true });
    for (const e of entries) {
      const full = path.join(dir, e.name);
      if (e.isDirectory()) {
        const found = await walk(full);
        if (found) return found;
      } else if (e.isFile() && e.name === `${slug}.md`) {
        return full;
      }
    }
    return null;
  }
  return walk(contentRoot);
}

function titleFromSlug(slug) {
  return String(slug)
    .split(/[-_]/g)
    .filter(Boolean)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
}

/** ```mermaid → div.mermaid (same as MdDocEditable); Playwright then turns these into SVG at import. */
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
  return md;
}

export async function tryImportDocsFromRepo(docsRepoRoot) {
  const contentDir = path.join(docsRepoRoot, 'content');

  if (!(await fs.pathExists(docsRepoRoot))) {
    return { imported: false, reason: `Path does not exist: ${docsRepoRoot}` };
  }
  if (!(await fs.pathExists(contentDir))) {
    return { imported: false, reason: `No content/ directory under ${docsRepoRoot}` };
  }
  if (!(await fs.pathExists(DOCS_CONFIG_PATH))) {
    return { imported: false, reason: `Documentation manifest missing: ${DOCS_CONFIG_PATH}` };
  }

  let config;
  try {
    config = parseYaml(await fs.readFile(DOCS_CONFIG_PATH, 'utf8'));
  } catch (e) {
    return { imported: false, reason: `Failed to parse documentation manifest: ${e.message}` };
  }

  const docEntries = config?.docs;
  if (!docEntries || typeof docEntries !== 'object') {
    return { imported: false, reason: 'Documentation manifest has no docs: mapping' };
  }

  const mdRenderer = createDocsMarkdownIt();
  const pool = db.getPool();
  if (!pool) {
    return { imported: false, reason: 'Database pool not available' };
  }

  const client = await pool.connect();
  let inserted = 0;
  let mermaidSession = null;
  try {
    await client.query('BEGIN');
    mermaidSession = await createMermaidRenderSession();

    for (const [slug, meta] of Object.entries(docEntries)) {
      if (!slug || !meta || typeof meta !== 'object') continue;

      const mdPath = await findMarkdownFile(contentDir, slug);
      if (!mdPath) {
        console.warn(`[docs-repo-import] No markdown file for slug "${slug}" under content/`);
        continue;
      }

      const rawFile = await fs.readFile(mdPath, 'utf8');
      const { meta: front, body } = splitFrontmatter(rawFile);

      const title =
        (typeof front.title === 'string' && front.title.trim()) ||
        titleFromSlug(slug);
      const description =
        (typeof front.description === 'string' && front.description.trim()) || '';

      const md = applyAppUrlTemplate(body);
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

      const card_title = meta.card_title != null ? String(meta.card_title) : '';
      const action_label = meta.action_label != null ? String(meta.action_label) : 'Open';
      const doc_type = meta.doc_type != null ? String(meta.doc_type) : '';
      const sort_order = Number.isFinite(Number(meta.sort_order)) ? Number(meta.sort_order) : 0;
      const hidden = Boolean(meta.hidden);
      const is_feature_card = Boolean(meta.is_feature_card);
      const show_in_docs = meta.show_in_docs !== false;
      const card_order = Number.isFinite(Number(meta.card_order)) ? Number(meta.card_order) : 0;

      await client.query(
        `INSERT INTO ${DOCS_TABLE} (
        slug, title, card_title, description, action_label, doc_type,
        sort_order, hidden, is_feature_card, show_in_docs, card_order,
        md, html, toc, updated_at
      ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14::jsonb, NOW())`,
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
      await client.query('ROLLBACK');
      return { imported: false, reason: 'No documentation pages imported (missing .md files?)' };
    }

    await client.query('COMMIT');
    return { imported: true, count: inserted };
  } catch (e) {
    try {
      await client.query('ROLLBACK');
    } catch {
      /* ignore */
    }
    throw e;
  } finally {
    if (mermaidSession) {
      await mermaidSession.close().catch(() => {});
    }
    client.release();
  }
}
