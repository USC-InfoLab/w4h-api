import { query } from '../db.js';
import { renderDoc } from './utils/md-render.js';

const KINDS = new Set(['notebook', 'cli', 'script', 'postgres_extension', 'dataset', 'sql']);
const LANGUAGES = new Set(['python', 'r', 'sql', 'csv', 'json', 'other']);
const STATUSES = new Set(['draft', 'published', 'deprecated']);
const KIND_ORDER = ['notebook', 'cli', 'script', 'sql', 'postgres_extension', 'dataset'];

const SELECT_COLS = `id, slug, publisher, title, summary, kind, kinds, language,
  repo_url, repo_ref, docs_path, runnable, status, version, latest_version,
  latest_repo_ref, version_checked_at, created_at, updated_at`;

function uniqKinds(list) {
  const seen = new Set();
  const out = [];
  for (const raw of list || []) {
    const k = String(raw || '').trim();
    if (!KINDS.has(k) || seen.has(k)) continue;
    seen.add(k);
    out.push(k);
  }
  out.sort((a, b) => KIND_ORDER.indexOf(a) - KIND_ORDER.indexOf(b));
  return out;
}

function parseKinds(body, fallbackKind = '') {
  let list = [];
  if (Array.isArray(body?.kinds)) list = body.kinds;
  else if (typeof body?.kinds === 'string' && body.kinds.trim()) {
    list = body.kinds.split(/[,\s]+/);
  }
  if (body?.kind != null && String(body.kind).trim()) {
    list = [body.kind, ...list];
  }
  if (fallbackKind) list = [...list, fallbackKind];
  const unique = uniqKinds(list);
  const primary = String(body?.kind || fallbackKind || '').trim();
  if (KINDS.has(primary) && unique.includes(primary)) {
    return [primary, ...unique.filter((k) => k !== primary)];
  }
  return unique;
}

export function githubRawUrl(repoUrl, repoRef, docsPath) {
  const url = String(repoUrl || '').trim().replace(/\.git$/i, '');
  const ref = String(repoRef || 'main').trim() || 'main';
  const pathPart = String(docsPath || 'README.md').replace(/^\/+/, '');
  const m = /^https?:\/\/github\.com\/([^/]+)\/([^/#?]+)/i.exec(url);
  if (!m) return null;
  return `https://raw.githubusercontent.com/${m[1]}/${m[2]}/${encodeURIComponent(ref)}/${pathPart
    .split('/')
    .map(encodeURIComponent)
    .join('/')}`;
}

function githubBlobUrl(repoUrl, repoRef, docsPath) {
  const url = String(repoUrl || '').trim().replace(/\.git$/i, '');
  const ref = String(repoRef || 'main').trim() || 'main';
  const pathPart = String(docsPath || 'README.md').replace(/^\/+/, '');
  const m = /^https?:\/\/github\.com\/([^/]+)\/([^/#?]+)/i.exec(url);
  if (!m) return url || null;
  return `https://github.com/${m[1]}/${m[2]}/blob/${ref}/${pathPart}`;
}

async function fetchReadmeHtml(row) {
  const raw = githubRawUrl(row.repo_url, row.repo_ref, row.docs_path);
  const docs_url = githubBlobUrl(row.repo_url, row.repo_ref, row.docs_path);
  if (!raw) {
    return { docs_html: null, docs_url, docs_error: 'README is only fetched for public GitHub repos.' };
  }
  try {
    const res = await fetch(raw, { headers: { Accept: 'text/plain' } });
    if (!res.ok) {
      return { docs_html: null, docs_url, docs_error: `Could not load README (${res.status}).` };
    }
    const md = await res.text();
    const { html } = renderDoc(md);
    return { docs_html: html, docs_url, docs_error: null };
  } catch (e) {
    return { docs_html: null, docs_url, docs_error: e.message || 'Failed to fetch README.' };
  }
}

function haystack(row) {
  const kinds = uniqKinds([...(row.kinds || []), row.kind]).join(' ');
  return [row.slug, row.publisher, row.title, row.summary, kinds, row.language]
    .filter(Boolean)
    .join(' ')
    .toLowerCase();
}

function versionFromRepoRef(ref) {
  const m = /^v?(\d+)\.(\d+)\.(\d+)/i.exec(String(ref || '').trim());
  return m ? `${m[1]}.${m[2]}.${m[3]}` : null;
}

function normalizeVersion(value, repoRef) {
  if (value === undefined) return undefined;
  if (value == null || String(value).trim() === '') {
    return versionFromRepoRef(repoRef) || null;
  }
  return versionFromRepoRef(value) || String(value).trim() || null;
}

export function isUpdateAvailable(row) {
  const latest = String(row?.latest_repo_ref || '').trim();
  const pin = String(row?.repo_ref || '').trim();
  if (!latest) return false;
  return latest !== pin;
}

export function presentContribution(row, { includeAdminFields = false } = {}) {
  if (!row) return null;
  if (includeAdminFields) {
    return { ...row, update_available: isUpdateAvailable(row) };
  }
  const {
    latest_version: _lv,
    latest_repo_ref: _lr,
    version_checked_at: _vc,
    ...rest
  } = row;
  return rest;
}

export async function listContributions({ q = '', kind = '', language = '', includeUnpublished = false } = {}) {
  const statusClause = includeUnpublished ? '' : `WHERE status = 'published'`;
  const res = await query(
    `SELECT ${SELECT_COLS} FROM w4h.contributions ${statusClause} ORDER BY title ASC`
  );
  const queryText = String(q || '').trim().toLowerCase();
  const kindFilter = String(kind || '').trim().toLowerCase();
  const langFilter = String(language || '').trim().toLowerCase();
  return res.rows.filter((row) => {
    if (kindFilter) {
      const kinds = uniqKinds([...(row.kinds || []), row.kind]);
      if (!kinds.includes(kindFilter)) return false;
    }
    if (langFilter && String(row.language).toLowerCase() !== langFilter) return false;
    if (queryText && !haystack(row).includes(queryText)) return false;
    return true;
  });
}

export async function getContribution(slug, { withDocs = false, includeUnpublished = false } = {}) {
  const key = String(slug || '').trim();
  if (!key) return null;
  const res = await query(
    `SELECT ${SELECT_COLS} FROM w4h.contributions WHERE lower(slug) = lower($1)`,
    [key]
  );
  const row = res.rows[0] || null;
  if (!row) return null;
  if (!includeUnpublished && row.status !== 'published') return null;
  if (!withDocs) return row;
  const docs = await fetchReadmeHtml(row);
  return { ...row, ...docs };
}

function normalizeBody(body, { partial = false } = {}) {
  const slug = body?.slug != null ? String(body.slug).trim().toLowerCase() : '';
  const publisher = body?.publisher != null ? String(body.publisher).trim() : '';
  const title = body?.title != null ? String(body.title).trim() : '';
  const summary = body?.summary == null ? null : String(body.summary);
  const kinds = parseKinds(body);
  const kind = kinds[0] || '';
  const language = body?.language != null ? String(body.language).trim() : 'other';
  const repo_url = body?.repo_url != null ? String(body.repo_url).trim() : '';
  const repo_ref = body?.repo_ref != null ? String(body.repo_ref).trim() || 'main' : 'main';
  const docs_path = body?.docs_path != null ? String(body.docs_path).trim() || 'README.md' : 'README.md';
  const status = body?.status != null ? String(body.status).trim() : 'published';
  const version = normalizeVersion(body?.version, repo_ref);

  if (!partial) {
    if (!slug || !publisher || !title || !repo_url) {
      return { error: 'slug, publisher, title, and repo_url are required' };
    }
    if (!kinds.length) {
      return { error: 'kind must include notebook, cli, script, postgres_extension, or dataset' };
    }
    if (!LANGUAGES.has(language)) return { error: 'language must be python, r, sql, csv, json, or other' };
    if (!STATUSES.has(status)) return { error: 'status must be draft, published, or deprecated' };
  } else {
    if (body?.kind != null || body?.kinds != null) {
      if (!kinds.length) return { error: 'invalid kind' };
    }
    if (body?.language != null && !LANGUAGES.has(language)) return { error: 'invalid language' };
    if (body?.status != null && !STATUSES.has(status)) return { error: 'invalid status' };
  }

  return {
    slug,
    publisher,
    title,
    summary,
    kind,
    kinds,
    language,
    repo_url,
    repo_ref,
    docs_path,
    status,
    version: version === undefined ? versionFromRepoRef(repo_ref) : version,
  };
}

export async function createContribution(body, userId) {
  const n = normalizeBody(body);
  if (n.error) return { ok: false, status: 400, error: n.error };
  try {
    const res = await query(
      `INSERT INTO w4h.contributions
         (slug, publisher, title, summary, kind, kinds, language, repo_url, repo_ref, docs_path, status, version, created_by, updated_at)
       VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13, NOW())
       RETURNING ${SELECT_COLS}`,
      [
        n.slug,
        n.publisher,
        n.title,
        n.summary,
        n.kind,
        n.kinds,
        n.language,
        n.repo_url,
        n.repo_ref,
        n.docs_path,
        n.status,
        n.version,
        userId || null,
      ]
    );
    return { ok: true, row: res.rows[0] };
  } catch (e) {
    if (e.code === '23505') return { ok: false, status: 409, error: 'slug already exists' };
    throw e;
  }
}

export async function updateContribution(slug, body) {
  const existing = await getContribution(slug, { includeUnpublished: true });
  if (!existing) return { ok: false, status: 404, error: 'Contribution not found' };

  const next = { ...existing };
  if (body?.slug != null) next.slug = String(body.slug).trim().toLowerCase();
  if (body?.publisher != null) next.publisher = String(body.publisher).trim();
  if (body?.title != null) next.title = String(body.title).trim();
  if (body?.summary !== undefined) next.summary = body.summary == null ? null : String(body.summary);
  if (body?.kind != null || body?.kinds != null) {
    next.kinds = parseKinds(body, next.kind);
    next.kind = next.kinds[0] || next.kind;
  }
  if (body?.language != null) next.language = String(body.language).trim();
  if (body?.repo_url != null) next.repo_url = String(body.repo_url).trim();
  if (body?.repo_ref != null) next.repo_ref = String(body.repo_ref).trim() || 'main';
  if (body?.docs_path != null) next.docs_path = String(body.docs_path).trim() || 'README.md';
  if (body?.status != null) next.status = String(body.status).trim();
  if (body?.version !== undefined || body?.repo_ref != null) {
    const derived = normalizeVersion(
      body?.version !== undefined ? body.version : next.version,
      next.repo_ref
    );
    next.version = derived === undefined ? next.version : derived;
  }

  if (!next.slug || !next.publisher || !next.title || !next.repo_url) {
    return { ok: false, status: 400, error: 'slug, publisher, title, and repo_url are required' };
  }
  next.kinds = uniqKinds([...(next.kinds || []), next.kind]);
  next.kind = next.kinds[0] || next.kind;
  if (!next.kinds.length || !LANGUAGES.has(next.language) || !STATUSES.has(next.status)) {
    return { ok: false, status: 400, error: 'invalid kind, language, or status' };
  }

  try {
    const res = await query(
      `UPDATE w4h.contributions SET
         slug = $2, publisher = $3, title = $4, summary = $5, kind = $6, kinds = $7, language = $8,
         repo_url = $9, repo_ref = $10, docs_path = $11, status = $12, version = $13, updated_at = NOW()
       WHERE lower(slug) = lower($1)
       RETURNING ${SELECT_COLS}`,
      [
        slug,
        next.slug,
        next.publisher,
        next.title,
        next.summary,
        next.kind,
        next.kinds,
        next.language,
        next.repo_url,
        next.repo_ref,
        next.docs_path,
        next.status,
        next.version ?? null,
      ]
    );
    return { ok: true, row: res.rows[0] };
  } catch (e) {
    if (e.code === '23505') return { ok: false, status: 409, error: 'slug already exists' };
    throw e;
  }
}

export async function deleteContribution(slug) {
  const res = await query(
    `DELETE FROM w4h.contributions WHERE lower(slug) = lower($1) RETURNING slug`,
    [String(slug || '').trim()]
  );
  return res.rowCount > 0;
}
