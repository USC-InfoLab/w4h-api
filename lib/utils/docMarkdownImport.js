import fs from 'fs-extra';
import path from 'path';
import yaml from 'js-yaml';
import { fileURLToPath } from 'url';
import { renderDoc } from './docRender.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const FRONT_MATTER = /^---\r?\n([\s\S]*?)\r?\n---\r?\n([\s\S]*)$/;

function parseMarkdownFile(raw) {
  const m = raw.match(FRONT_MATTER);
  if (!m) {
    return { meta: {}, body: raw.trimStart() };
  }
  let meta = {};
  try {
    meta = yaml.load(m[1]) || {};
  } catch {
    meta = {};
  }
  return { meta, body: m[2] ?? '' };
}

function resolveDocsRepoDir(repoDir) {
  if (!repoDir) return null;
  if (path.isAbsolute(repoDir)) return repoDir;
  return path.resolve(path.join(__dirname, '..', '..'), repoDir);
}

function resolveConfigPath(apiRoot, configRelative) {
  const rel = configRelative || 'documentation-configuration.yml';
  return path.isAbsolute(rel) ? rel : path.join(apiRoot, rel);
}

/**
 * Load documentation-configuration.yml from the API repo root.
 * @param {string} apiRoot - absolute path to w4h-api
 */
async function loadDocumentationConfig(apiRoot, configFilename) {
  const rel =
    configFilename || process.env.DOCS_CONFIG_FILE || 'documentation-configuration.yml';
  const configPath = resolveConfigPath(apiRoot, rel);
  if (!(await fs.pathExists(configPath))) {
    return null;
  }
  const raw = await fs.readFile(configPath, 'utf8');
  try {
    return yaml.load(raw) || {};
  } catch {
    return null;
  }
}

/**
 * From nested layout path like "guide/quick-start.md" → { docType, slug }.
 */
function pathToDocTypeAndSlug(relFromContent, layout) {
  const norm = relFromContent.replace(/\\/g, '/');
  if (layout !== 'nested') {
    const base = path.basename(norm, '.md');
    return { docType: '', slug: base };
  }
  const parts = norm.split('/').filter(Boolean);
  if (parts.length < 2) {
    const slug = path.basename(norm, '.md');
    return { docType: '', slug };
  }
  const folder = parts[0];
  const slug = parts[parts.length - 1].replace(/\.md$/i, '');
  const docType = folder === 'root' ? '' : folder;
  return { docType, slug };
}

async function collectMarkdownFiles(contentDir) {
  const out = [];
  async function walk(dir, baseRel) {
    const entries = await fs.readdir(dir, { withFileTypes: true });
    for (const e of entries) {
      const full = path.join(dir, e.name);
      const rel = baseRel ? path.join(baseRel, e.name) : e.name;
      if (e.isDirectory()) {
        await walk(full, rel);
      } else if (e.isFile() && e.name.toLowerCase().endsWith('.md')) {
        out.push({ full, rel: rel.replace(/\\/g, '/') });
      }
    }
  }
  if (await fs.pathExists(contentDir)) {
    await walk(contentDir, '');
  }
  return out;
}

/**
 * Build rows for w4h.docs from markdown files under DOCS_REPO_DIR.
 * @returns {Promise<Array<object>>} rows ready for INSERT (includes html, toc as object)
 */
async function buildDocsRowsFromMarkdownRepo(options = {}) {
  const apiRoot = options.apiRoot || path.join(__dirname, '..', '..');
  const envDir = process.env.DOCS_REPO_DIR;
  const repoDir = resolveDocsRepoDir(options.docsRepoDir ?? envDir);
  if (!repoDir || !(await fs.pathExists(repoDir))) {
    return [];
  }

  const config = (await loadDocumentationConfig(apiRoot, options.configFilename)) || {};
  const contentRoot = config.contentRoot || 'content';
  const layout = config.layout === 'flat' ? 'flat' : 'nested';
  const defaults = config.defaults || {};

  const contentDir = path.join(repoDir, contentRoot);
  const files = await collectMarkdownFiles(contentDir);
  const rows = [];

  for (const { full, rel } of files) {
    const raw = await fs.readFile(full, 'utf8');
    const { meta, body } = parseMarkdownFile(raw);
    const { docType: pathDocType, slug } = pathToDocTypeAndSlug(rel, layout);
    if (!slug) continue;

    const docType = meta.doc_type != null ? String(meta.doc_type) : pathDocType;
    const { html, toc } = renderDoc(body);

    rows.push({
      slug,
      title: meta.title ?? slug,
      card_title: meta.card_title ?? defaults.card_title ?? '',
      description: meta.description ?? '',
      action_label: meta.action_label ?? defaults.action_label ?? 'Open',
      doc_type: docType,
      sort_order: Number(meta.sort_order ?? defaults.sort_order ?? 0),
      hidden: Boolean(meta.hidden ?? defaults.hidden ?? false),
      is_feature_card: Boolean(meta.is_feature_card ?? defaults.is_feature_card ?? true),
      show_in_docs: Boolean(meta.show_in_docs ?? defaults.show_in_docs ?? true),
      card_order: Number(meta.card_order ?? defaults.card_order ?? 0),
      md: body,
      html,
      toc,
    });
  }

  const bySlug = new Map();
  for (const row of rows) {
    if (bySlug.has(row.slug)) {
      console.warn(
        `[w4h-api] Documentation import: duplicate slug "${row.slug}" — using the last file encountered`
      );
    }
    bySlug.set(row.slug, row);
  }
  return [...bySlug.values()];
}

export {
  buildDocsRowsFromMarkdownRepo,
  loadDocumentationConfig,
  resolveDocsRepoDir,
  parseMarkdownFile,
  pathToDocTypeAndSlug,
};
