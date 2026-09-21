import fs from 'fs';
import path from 'path';
import { parse as parseYaml } from 'yaml';
import { githubRawUrl } from './contributions.js';

// Primary kind when several apply — SQL job / extension / dataset first,
// then the interactive and automation surfaces in the same repo.
const KIND_DETECT_ORDER = ['sql', 'postgres_extension', 'dataset', 'notebook', 'cli', 'script'];
const SKIP_DIR_NAMES = new Set(['.git', 'node_modules', '.venv', '__pycache__', '.pytest_cache']);

const KNOWN_TYPES = new Set([
  'text', 'integer', 'bigint', 'double precision', 'numeric', 'boolean',
  'date', 'timestamp', 'timestamptz',
]);

// manifest.yaml's `format` names the sample data's file format (ADR-016
// naming: w4h-sample-<vendor>-<format>) — maps directly onto the
// contributions.language enum's non-code values.
const KNOWN_DATA_FORMATS = new Set(['csv', 'json', 'sql']);

export function validateManifest(manifest) {
  const errors = [];
  const warnings = [];
  if (!manifest || typeof manifest !== 'object') {
    return { errors: ['manifest.yaml is empty or not a mapping'], warnings };
  }
  if (!manifest.vendor || typeof manifest.vendor !== 'string') {
    errors.push('vendor is required (string)');
  }
  if (!manifest.format || typeof manifest.format !== 'string') {
    warnings.push('format is recommended (e.g. "csv", "json") — used to set the catalog language');
  } else if (!KNOWN_DATA_FORMATS.has(manifest.format.toLowerCase())) {
    warnings.push(`format "${manifest.format}" is unusual — expected one of: ${[...KNOWN_DATA_FORMATS].join(', ')}`);
  }
  const signals = Array.isArray(manifest.signals) ? manifest.signals : null;
  if (!signals || !signals.length) {
    errors.push('signals must be a non-empty array');
    return { errors, warnings };
  }

  const seenSlugs = new Set();
  signals.forEach((sig, i) => {
    const label = `signal[${i}]${sig?.slug ? ` '${sig.slug}'` : ''}`;
    if (!sig?.slug || typeof sig.slug !== 'string') {
      errors.push(`${label}: slug is required`);
    } else if (seenSlugs.has(sig.slug)) {
      errors.push(`${label}: duplicate slug`);
    } else {
      seenSlugs.add(sig.slug);
    }
    if (!sig?.file || typeof sig.file !== 'string') {
      errors.push(`${label}: file is required`);
    }

    const columns = Array.isArray(sig?.columns) ? sig.columns : null;
    if (!columns || !columns.length) {
      errors.push(`${label}: columns must be a non-empty array`);
    }
    const columnNames = new Set();
    (columns || []).forEach((col, ci) => {
      if (!col?.name || typeof col.name !== 'string') {
        errors.push(`${label}: columns[${ci}] is missing name`);
      } else {
        columnNames.add(col.name);
      }
      if (!col?.type || typeof col.type !== 'string') {
        errors.push(`${label}: columns[${ci}] is missing type`);
      } else if (!KNOWN_TYPES.has(col.type)) {
        warnings.push(`${label}: columns[${ci}] type "${col.type}" is unusual — check it matches a Postgres type`);
      }
    });

    const mapping = sig?.geomts_mapping;
    if (!mapping || typeof mapping !== 'object') {
      errors.push(`${label}: geomts_mapping is required`);
    } else {
      if (!mapping.subject || typeof mapping.subject !== 'string') {
        errors.push(`${label}: geomts_mapping.subject is required`);
      }
      const values = Array.isArray(mapping.values) ? mapping.values : [];
      values.forEach((v, vi) => {
        if (v?.source && columnNames.size && !columnNames.has(v.source)) {
          warnings.push(
            `${label}: geomts_mapping.values[${vi}].source "${v.source}" has no matching column — check spelling or add the column`
          );
        }
      });
    }
  });

  return { errors, warnings };
}

function humanizeTitle(slug) {
  return String(slug || '')
    .replace(/^w4h-(sample-)?/, '')
    .replace(/-importer$/, '')
    .split(/[-_]/)
    .filter(Boolean)
    .map((w) => w[0].toUpperCase() + w.slice(1))
    .join(' ');
}

export function parseRepoUrl(repoUrl) {
  const url = String(repoUrl || '').trim().replace(/\.git$/i, '');
  const m = /^https?:\/\/github\.com\/([^/]+)\/([^/#?]+)/i.exec(url);
  if (!m) return null;
  return { owner: m[1], repo: m[2], url };
}

async function fetchRawText(repoUrl, repoRef, path) {
  const raw = githubRawUrl(repoUrl, repoRef, path);
  if (!raw) return null;
  try {
    const res = await fetch(raw, { headers: { Accept: 'text/plain' } });
    if (!res.ok) return null;
    return await res.text();
  } catch {
    return null;
  }
}

// Strips inline Markdown (links, images, bold/italic, code spans) down to
// plain text — the admin form shows this in plain <input> fields, not a
// Markdown renderer, so raw "**"/"[]()" syntax would just show up as noise.
function stripMarkdownInline(text) {
  return String(text || '')
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/`([^`]*)`/g, '$1')
    .replace(/\*\*([^*]*)\*\*/g, '$1')
    .replace(/__([^_]*)__/g, '$1')
    .replace(/\*([^*]*)\*/g, '$1')
    .replace(/\s+/g, ' ')
    .trim();
}

function summarizeReadme(md) {
  const lines = String(md || '').split('\n');
  let title = '';
  const bodyLines = [];
  for (const line of lines) {
    const h = /^#\s+(.+)/.exec(line);
    if (!title && h) {
      title = stripMarkdownInline(h[1]).slice(0, 80);
      continue;
    }
    if (title) bodyLines.push(line);
  }
  const paragraph = bodyLines
    .join('\n')
    .split(/\n\s*\n/)
    .map((p) => p.replace(/\s+/g, ' ').trim())
    .find((p) => p && !p.startsWith('#') && !p.startsWith('!') && !p.startsWith('['));
  return { title, summary: paragraph ? stripMarkdownInline(paragraph).slice(0, 200) : '' };
}

export function isSqlJobManifest(manifest) {
  if (!manifest || typeof manifest !== 'object') return false;
  return manifest.kind === 'sql'
    || (Array.isArray(manifest.requires) && Array.isArray(manifest.produces) && !manifest.signals);
}

export function isDatasetManifest(manifest) {
  if (!manifest || typeof manifest !== 'object') return false;
  if (isSqlJobManifest(manifest)) return false;
  return Array.isArray(manifest.signals);
}

export function pyprojectHasScripts(text) {
  const src = String(text || '');
  return /^\s*\[project\.scripts\]/m.test(src) || /^\s*\[tool\.poetry\.scripts\]/m.test(src);
}

function isSkippedControlPath(filePath) {
  return /(^|\/)tests?\//.test(filePath) || /(^|\/)examples?\//.test(filePath);
}

export function detectContributionKinds({ paths = [], manifest = null, pyprojectText = '' } = {}) {
  const files = (paths || []).map((p) => String(p || '').replace(/\\/g, '/'));
  const found = new Set();

  if (isSqlJobManifest(manifest)) found.add('sql');
  else if (isDatasetManifest(manifest)) found.add('dataset');

  if (files.some((p) => p.endsWith('.ipynb'))) found.add('notebook');
  if (pyprojectHasScripts(pyprojectText)) found.add('cli');
  if (files.some((p) => p.endsWith('.control') && !isSkippedControlPath(p))) {
    found.add('postgres_extension');
  }

  return KIND_DETECT_ORDER.filter((k) => found.has(k));
}

export function detectContributionLanguage({ kinds = [], manifest = null, paths = [], pyprojectText = '' } = {}) {
  if (kinds.includes('dataset')) {
    const format = String(manifest?.format || '').toLowerCase();
    return KNOWN_DATA_FORMATS.has(format) ? format : 'other';
  }
  if (kinds.includes('sql') || kinds.includes('postgres_extension')) return 'sql';
  const files = (paths || []).map((p) => String(p || '').replace(/\\/g, '/'));
  if (files.some((p) => p.endsWith('.r') || p.endsWith('.R') || p === 'DESCRIPTION' || p.endsWith('/DESCRIPTION'))) {
    return 'r';
  }
  if (pyprojectText || files.some((p) => p.endsWith('.py') || p.endsWith('.ipynb'))) return 'python';
  return kinds.length ? 'other' : 'python';
}

async function githubJson(url) {
  try {
    const res = await fetch(url, {
      headers: {
        Accept: 'application/vnd.github+json',
        'User-Agent': 'w4h-api-contributions-inspect',
      },
    });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

async function fetchGithubBlobPaths(owner, repo, ref) {
  const url = `https://api.github.com/repos/${encodeURIComponent(owner)}/${encodeURIComponent(repo)}/git/trees/${encodeURIComponent(ref)}?recursive=1`;
  const data = await githubJson(url);
  const tree = Array.isArray(data?.tree) ? data.tree : [];
  if (!tree.length) return null;
  return tree.filter((n) => n?.type === 'blob' && n.path).map((n) => n.path);
}

async function fetchGithubDirFilePaths(owner, repo, ref, dir = '') {
  const suffix = dir ? `/${dir.split('/').map(encodeURIComponent).join('/')}` : '';
  const url = `https://api.github.com/repos/${encodeURIComponent(owner)}/${encodeURIComponent(repo)}/contents${suffix}?ref=${encodeURIComponent(ref)}`;
  const data = await githubJson(url);
  if (!Array.isArray(data)) return [];
  return data.filter((n) => n?.type === 'file' && n.path).map((n) => n.path);
}

async function fetchGithubPathsFallback(owner, repo, ref) {
  const [root, notebook, notebooks] = await Promise.all([
    fetchGithubDirFilePaths(owner, repo, ref, ''),
    fetchGithubDirFilePaths(owner, repo, ref, 'notebook'),
    fetchGithubDirFilePaths(owner, repo, ref, 'notebooks'),
  ]);
  const paths = [...root, ...notebook, ...notebooks];
  return paths.length ? paths : null;
}

function walkLocalFiles(root, { maxFiles = 400 } = {}) {
  const out = [];
  const walk = (dir) => {
    if (out.length >= maxFiles) return;
    let entries;
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      if (SKIP_DIR_NAMES.has(entry.name) || entry.name.startsWith('.')) continue;
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) walk(full);
      else if (entry.isFile()) out.push(path.relative(root, full).split(path.sep).join('/'));
      if (out.length >= maxFiles) return;
    }
  };
  walk(root);
  return out;
}

function listLocalSiblingPaths(repo) {
  const root = path.resolve(process.cwd(), '..', repo);
  try {
    if (!fs.existsSync(root) || !fs.statSync(root).isDirectory()) return null;
  } catch {
    return null;
  }
  const paths = walkLocalFiles(root);
  return paths.length ? { root, paths } : null;
}

function readLocalText(root, relPath) {
  try {
    return fs.readFileSync(path.join(root, relPath), 'utf8');
  } catch {
    return null;
  }
}

async function resolveRepoPaths(parsed, repoRef) {
  const remote = await fetchGithubBlobPaths(parsed.owner, parsed.repo, repoRef);
  if (remote) return { paths: remote, localRoot: null };
  const local = listLocalSiblingPaths(parsed.repo);
  if (local) return { paths: local.paths, localRoot: local.root };
  const fallback = await fetchGithubPathsFallback(parsed.owner, parsed.repo, repoRef);
  return { paths: fallback || [], localRoot: null };
}

async function detectPairedDataset(parsed, repoRef, slug) {
  if (!slug.endsWith('-importer')) return null;
  const pairedSlug = slug.slice(0, -'-importer'.length);
  const pairedUrl = `https://github.com/${parsed.owner}/${pairedSlug}`;
  const pairedManifestText = await fetchRawText(pairedUrl, repoRef, 'manifest.yaml');
  if (!pairedManifestText) return null;
  try {
    const pairedManifest = parseYaml(pairedManifestText);
    return {
      slug: pairedSlug,
      repo_url: pairedUrl,
      title: pairedManifest?.title || null,
      vendor: pairedManifest?.vendor || null,
    };
  } catch {
    return null;
  }
}

function applyManifestCopy(manifest, { title, summary }) {
  let nextTitle = title;
  let nextSummary = summary;
  if (isSqlJobManifest(manifest)) {
    if (manifest.title) nextTitle = manifest.title;
    if (manifest.summary) nextSummary = manifest.summary;
  } else if (isDatasetManifest(manifest)) {
    if (manifest.title) nextTitle = manifest.title;
    if (manifest.description) nextSummary = manifest.description;
  }
  return { title: nextTitle, summary: nextSummary };
}

function manifestValidation(manifest) {
  if (isSqlJobManifest(manifest)) {
    if (!manifest.sql) {
      return { errors: ['sql path is required (e.g. sql/compute.sql)'], warnings: [] };
    }
    return { errors: [], warnings: [] };
  }
  if (isDatasetManifest(manifest) || manifest) {
    return validateManifest(manifest);
  }
  return { errors: [], warnings: [] };
}

// Best-effort auto-fill for the Contributions admin form. GitHub is queried
// unauthenticated (no token config) — one recursive tree listing plus a few
// raw-file fetches, fine for occasional Super-admin curation. A sibling
// checkout next to w4h-api is used only when the GitHub tree is unavailable
// (private or not-yet-pushed first-party repos).
export async function inspectRepository({ repo_url, repo_ref } = {}) {
  const parsed = parseRepoUrl(repo_url);
  if (!parsed) {
    return { error: 'repo_url must be a github.com repository URL' };
  }
  const repoRef = String(repo_ref || 'main').trim() || 'main';
  const slug = parsed.repo;
  const publisher = parsed.owner;

  const [{ paths, localRoot }, readmeText, manifestText, pyprojectText] = await Promise.all([
    resolveRepoPaths(parsed, repoRef),
    fetchRawText(parsed.url, repoRef, 'README.md'),
    fetchRawText(parsed.url, repoRef, 'manifest.yaml'),
    fetchRawText(parsed.url, repoRef, 'pyproject.toml'),
  ]);

  const readme = readmeText || (localRoot ? readLocalText(localRoot, 'README.md') : null);
  const manifestSrc = manifestText || (localRoot ? readLocalText(localRoot, 'manifest.yaml') : null);
  const pyprojectSrc = pyprojectText || (localRoot ? readLocalText(localRoot, 'pyproject.toml') : null);

  const { title: readmeTitle, summary: readmeSummary } = summarizeReadme(readme);

  let manifest = null;
  let validation = { errors: [], warnings: [] };
  if (manifestSrc) {
    try {
      manifest = parseYaml(manifestSrc);
    } catch (e) {
      validation = { errors: [`manifest.yaml failed to parse: ${e.message}`], warnings: [] };
    }
  }

  let kinds = detectContributionKinds({
    paths,
    manifest,
    pyprojectText: pyprojectSrc,
  });
  // ADR-016 importers always ship both surfaces; keep this if the tree listing
  // missed the notebook (private repo, truncated tree) but the slug is clear.
  if (slug.endsWith('-importer') && !kinds.includes('notebook') && !kinds.includes('cli')) {
    kinds = ['notebook', 'cli'];
  }

  const language = detectContributionLanguage({
    kinds,
    manifest,
    paths,
    pyprojectText: pyprojectSrc,
  });
  const kind = kinds[0] || '';

  let title = readmeTitle || humanizeTitle(slug);
  let summary = readmeSummary;
  if (manifest) {
    ({ title, summary } = applyManifestCopy(manifest, { title, summary }));
    if (validation.errors.length === 0) {
      validation = manifestValidation(manifest);
    }
  }

  let pairedDataset = null;
  if (slug.endsWith('-importer')) {
    pairedDataset = await detectPairedDataset(parsed, repoRef, slug);
    if (pairedDataset && !readmeSummary && (pairedDataset.title || pairedDataset.vendor)) {
      summary = `Imports the ${pairedDataset.title || pairedDataset.vendor} sample package (${pairedDataset.slug}) into W4H.`;
    }
  }

  return {
    suggested: {
      slug,
      publisher,
      title,
      summary,
      kind,
      kinds,
      language,
      repo_url: parsed.url,
      repo_ref: repoRef,
      docs_path: 'README.md',
    },
    manifest,
    validation,
    pairedDataset,
  };
}
