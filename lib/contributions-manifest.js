import { parse as parseYaml } from 'yaml';
import { githubRawUrl } from './contributions.js';

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

function parseRepoUrl(repoUrl) {
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

// Best-effort auto-fill for the Contributions admin form. GitHub is queried
// unauthenticated (no token config), so this stays to a couple of raw-file
// fetches rather than the Contents API — fine for occasional admin curation,
// not meant for bulk/automated use.
export async function inspectRepository({ repo_url, repo_ref } = {}) {
  const parsed = parseRepoUrl(repo_url);
  if (!parsed) {
    return { error: 'repo_url must be a github.com repository URL' };
  }
  const repoRef = String(repo_ref || 'main').trim() || 'main';
  const slug = parsed.repo;
  const publisher = parsed.owner;

  const [readmeText, manifestText] = await Promise.all([
    fetchRawText(parsed.url, repoRef, 'README.md'),
    fetchRawText(parsed.url, repoRef, 'manifest.yaml'),
  ]);

  const { title: readmeTitle, summary: readmeSummary } = summarizeReadme(readmeText);

  let manifest = null;
  let validation = { errors: [], warnings: [] };
  let kind = 'notebook';
  let kinds = ['notebook'];
  let language = 'python';
  let title = readmeTitle || humanizeTitle(slug);
  let summary = readmeSummary;
  let pairedDataset = null;

  if (manifestText) {
    try {
      manifest = parseYaml(manifestText);
    } catch (e) {
      validation = { errors: [`manifest.yaml failed to parse: ${e.message}`], warnings: [] };
    }
    if (manifest) {
      validation = validateManifest(manifest);
      kind = 'dataset';
      kinds = ['dataset'];
      const format = String(manifest.format || '').toLowerCase();
      language = KNOWN_DATA_FORMATS.has(format) ? format : 'other';
      if (manifest.title) title = manifest.title;
      if (manifest.description) summary = manifest.description;
    }
  } else if (slug.endsWith('-importer')) {
    kinds = ['notebook', 'cli'];
    // ADR-016 pairing: <sample-repo>-importer imports <sample-repo>. Same
    // convention contribution-runner.js's pairedDataSlugFor() uses at run
    // time; probed directly against GitHub here (not the catalog) since the
    // paired row may not be cataloged yet — order isn't enforced (ADR-018).
    const pairedSlug = slug.slice(0, -'-importer'.length);
    const pairedUrl = `https://github.com/${parsed.owner}/${pairedSlug}`;
    const pairedManifestText = await fetchRawText(pairedUrl, repoRef, 'manifest.yaml');
    if (pairedManifestText) {
      try {
        const pairedManifest = parseYaml(pairedManifestText);
        pairedDataset = {
          slug: pairedSlug,
          repo_url: pairedUrl,
          title: pairedManifest?.title || null,
          vendor: pairedManifest?.vendor || null,
        };
        if (!readmeSummary && (pairedManifest?.title || pairedManifest?.vendor)) {
          summary = `Imports the ${pairedManifest.title || pairedManifest.vendor} sample package (${pairedSlug}) into W4H.`;
        }
      } catch {
        // Paired repo's manifest is unreadable — not this row's problem; the
        // paired contribution's own inspect/validate will surface that.
      }
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
