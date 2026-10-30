/**
 * Catalog pin vs GitHub release/tag (ADR-018). Super admin checks for updates
 * and promotes a tag onto repo_ref — never auto-follow main.
 */
import fs from 'fs';
import path from 'path';
import { execFileSync } from 'child_process';
import { query } from '../db.js';
import { getContribution, listContributions, isUpdateAvailable } from './contributions.js';
import { inspectRepository, parseRepoUrl } from './contributions-manifest.js';
import { githubJson } from './github-client.js';

const SEMVER_TAG = /^v?(\d+)\.(\d+)\.(\d+)(?:[-+].*)?$/i;
const BRANCH_REFS = new Set(['main', 'master', 'HEAD', 'develop', 'dev']);

export function parseSemverTag(ref) {
  const raw = String(ref || '').trim();
  const m = SEMVER_TAG.exec(raw);
  if (!m) return null;
  return {
    major: Number(m[1]),
    minor: Number(m[2]),
    patch: Number(m[3]),
    version: `${m[1]}.${m[2]}.${m[3]}`,
    tag: raw,
  };
}

export function versionFromRepoRef(ref) {
  return parseSemverTag(ref)?.version || null;
}

export function compareSemver(a, b) {
  if (!a && !b) return 0;
  if (!a) return -1;
  if (!b) return 1;
  if (a.major !== b.major) return a.major - b.major;
  if (a.minor !== b.minor) return a.minor - b.minor;
  return a.patch - b.patch;
}

function sortSemverTags(tagNames) {
  return tagNames
    .map((name) => parseSemverTag(name))
    .filter(Boolean)
    .sort(compareSemver)
    .reverse();
}

function localSiblingRoot(repoName) {
  return path.resolve(process.cwd(), '..', repoName);
}

/** Same sibling layout as contribution inspect (../{repo} next to w4h-api). */
function hasLocalSiblingCheckout(repoName) {
  try {
    return fs.existsSync(path.join(localSiblingRoot(repoName), '.git'));
  } catch {
    return false;
  }
}

function listLocalSemverTags(repoName) {
  if (!hasLocalSiblingCheckout(repoName)) return [];
  try {
    const out = execFileSync('git', ['tag', '-l'], {
      cwd: localSiblingRoot(repoName),
      encoding: 'utf8',
      timeout: 15_000,
    });
    return sortSemverTags(out.split('\n').map((line) => line.trim()).filter(Boolean));
  } catch {
    return [];
  }
}

/** Same sibling + gh auth as manual `gh release view` (private repos). */
function fetchLocalReleaseViaGh(owner, repoName, tag) {
  if (!hasLocalSiblingCheckout(repoName)) return null;
  const ref = String(tag || '').trim();
  if (!ref) return null;
  try {
    const out = execFileSync(
      'gh',
      ['release', 'view', ref, '--repo', `${owner}/${repoName}`, '--json', 'body,name,url,publishedAt'],
      { cwd: localSiblingRoot(repoName), encoding: 'utf8', timeout: 20_000 }
    );
    const data = JSON.parse(out);
    return {
      tag: ref,
      name: data.name || ref,
      body: typeof data.body === 'string' ? data.body : '',
      html_url: data.url || githubReleasePageUrl(`https://github.com/${owner}/${repoName}`, ref),
      published_at: data.publishedAt || null,
    };
  } catch {
    return null;
  }
}

function githubTagsFailureHint(status, repoName) {
  if (status === 404) return 'GitHub repository not found or not public.';
  if (status === 403) {
    return `GitHub API rate limit. Clone ${repoName} next to w4h-api (w4h-repos workspace) or try again later.`;
  }
  return `Could not list tags (${status || 'network error'}).`;
}

export async function listGithubSemverTags(repoUrl) {
  const parsed = parseRepoUrl(repoUrl);
  if (!parsed) {
    return { error: 'repo_url must be a github.com repository URL', tags: [] };
  }
  if (hasLocalSiblingCheckout(parsed.repo)) {
    return {
      tags: listLocalSemverTags(parsed.repo),
      owner: parsed.owner,
      repo: parsed.repo,
      source: 'local',
    };
  }
  const url = `https://api.github.com/repos/${encodeURIComponent(parsed.owner)}/${encodeURIComponent(parsed.repo)}/tags?per_page=100`;
  const res = await githubJson(url);
  if (res.ok) {
    const tags = sortSemverTags(
      (Array.isArray(res.data) ? res.data : []).map((t) => t?.name).filter(Boolean)
    );
    return { tags, owner: parsed.owner, repo: parsed.repo };
  }
  return { error: githubTagsFailureHint(res.status, parsed.repo), tags: [] };
}

export function githubReleasePageUrl(repoUrl, tag) {
  const parsed = parseRepoUrl(repoUrl);
  const ref = String(tag || '').trim();
  if (!parsed || !ref) return null;
  return `https://github.com/${parsed.owner}/${parsed.repo}/releases/tag/${encodeURIComponent(ref)}`;
}

export async function fetchGithubRelease(repoUrl, tag) {
  const parsed = parseRepoUrl(repoUrl);
  const ref = String(tag || '').trim();
  if (!parsed || !ref) return null;
  const pageUrl = githubReleasePageUrl(repoUrl, ref);
  const url = `https://api.github.com/repos/${encodeURIComponent(parsed.owner)}/${encodeURIComponent(parsed.repo)}/releases/tags/${encodeURIComponent(ref)}`;
  const res = await githubJson(url);
  let fromApi = null;
  if (res.ok && res.data) {
    fromApi = {
      tag: ref,
      name: res.data.name || ref,
      body: typeof res.data.body === 'string' ? res.data.body : '',
      html_url: res.data.html_url || pageUrl,
      published_at: res.data.published_at || null,
    };
    if (fromApi.body.trim()) return fromApi;
  }

  const local = fetchLocalReleaseViaGh(parsed.owner, parsed.repo, ref);
  if (local) return local;

  if (fromApi) return fromApi;
  if (!pageUrl) return null;
  return {
    tag: ref,
    name: ref,
    body: '',
    html_url: pageUrl,
    published_at: null,
  };
}

function pinMessage(row, latest) {
  const pin = String(row.repo_ref || 'main').trim() || 'main';
  if (!latest) {
    if (BRANCH_REFS.has(pin) || !parseSemverTag(pin)) {
      return `No releases; still on ${pin}.`;
    }
    return `No GitHub tags newer than ${pin}.`;
  }
  if (latest.tag === pin || latest.version === row.version) {
    return `Pinned ${latest.tag} is current.`;
  }
  return `${latest.tag} available (pinned ${row.version ? `v${row.version}` : pin}).`;
}

function cacheColumns(row, latest) {
  if (!latest) {
    return { latest_version: null, latest_repo_ref: null };
  }
  const pin = parseSemverTag(row.repo_ref) || parseSemverTag(row.version ? `v${row.version}` : '');
  const newer = !pin || compareSemver(latest, pin) > 0 || latest.tag !== String(row.repo_ref || '');
  if (!newer && latest.tag === String(row.repo_ref || '')) {
    return { latest_version: latest.version, latest_repo_ref: latest.tag };
  }
  if (!newer) {
    return { latest_version: pin?.version || row.version, latest_repo_ref: row.repo_ref };
  }
  return { latest_version: latest.version, latest_repo_ref: latest.tag };
}

async function persistCheck(slug, cache) {
  await query(
    `UPDATE w4h.contributions
        SET latest_version = $2, latest_repo_ref = $3, version_checked_at = NOW()
      WHERE lower(slug) = lower($1)`,
    [slug, cache.latest_version, cache.latest_repo_ref]
  );
}

function pairedSlugFor(slug) {
  return String(slug || '').endsWith('-importer')
    ? String(slug).slice(0, -'-importer'.length)
    : null;
}

export async function checkContributionUpdates({ slug } = {}) {
  const includeUnpublished = true;
  const rows = slug
    ? [await getContribution(slug, { includeUnpublished })].filter(Boolean)
    : await listContributions({ includeUnpublished });
  if (slug && !rows.length) {
    return { ok: false, status: 404, error: 'Contribution not found' };
  }

  const results = [];
  const tagsByRepo = new Map();
  for (const row of rows) {
    const parsed = parseRepoUrl(row.repo_url);
    const cacheKey = parsed ? `${parsed.owner}/${parsed.repo}` : String(row.repo_url || row.slug);
    let listed = tagsByRepo.get(cacheKey);
    if (!listed) {
      listed = await listGithubSemverTags(row.repo_url);
      tagsByRepo.set(cacheKey, listed);
    }
    const latest = listed.tags[0] || null;
    const cache = listed.error
      ? { latest_version: row.latest_version, latest_repo_ref: row.latest_repo_ref }
      : cacheColumns(row, latest);
    const note = listed.error || pinMessage(row, latest);
    if (!listed.error) {
      await persistCheck(row.slug, cache);
    }
    const next = {
      ...row,
      ...cache,
      version_checked_at: listed.error ? row.version_checked_at : new Date().toISOString(),
    };
    const release = latest ? await fetchGithubRelease(row.repo_url, latest.tag) : null;
    results.push({
      slug: row.slug,
      version: row.version,
      repo_ref: row.repo_ref,
      latest_version: cache.latest_version,
      latest_repo_ref: cache.latest_repo_ref,
      update_available: isUpdateAvailable(next),
      note,
      error: listed.error || null,
      release,
    });
  }

  const refreshed = slug
    ? [await getContribution(slug, { includeUnpublished })].filter(Boolean)
    : await listContributions({ includeUnpublished });

  return {
    ok: true,
    results,
    rows: refreshed,
    updates: results.filter((r) => r.update_available).length,
    without_releases: results.filter((r) => !r.latest_repo_ref && !r.error).length,
  };
}

export async function previewOrPinContributionVersion(slug, repoRef, { confirm = false } = {}) {
  const existing = await getContribution(slug, { includeUnpublished: true });
  if (!existing) return { ok: false, status: 404, error: 'Contribution not found' };

  const repo_ref = String(repoRef || existing.latest_repo_ref || '').trim();
  if (!repo_ref) {
    return { ok: false, status: 400, error: 'repo_ref is required (a GitHub tag such as v0.2.0)' };
  }

  const inspect = await inspectRepository({
    repo_url: existing.repo_url,
    repo_ref,
  });
  if (inspect.error) {
    return { ok: false, status: 400, error: inspect.error };
  }

  const parsed = parseSemverTag(repo_ref);
  const version = parsed?.version || versionFromRepoRef(repo_ref);
  const release = await fetchGithubRelease(existing.repo_url, repo_ref);

  let pairedUpdate = null;
  const pairedSlug = pairedSlugFor(existing.slug);
  if (pairedSlug) {
    const paired = await getContribution(pairedSlug, { includeUnpublished: true });
    if (paired && isUpdateAvailable(paired)) {
      pairedUpdate = {
        slug: paired.slug,
        title: paired.title,
        version: paired.version,
        repo_ref: paired.repo_ref,
        latest_version: paired.latest_version,
        latest_repo_ref: paired.latest_repo_ref,
      };
    }
  }

  const preview = {
    slug: existing.slug,
    from: { version: existing.version, repo_ref: existing.repo_ref },
    to: { version, repo_ref },
    inspect,
    release,
    pairedUpdate,
  };

  if (!confirm) {
    return { ok: true, preview, row: existing };
  }

  const listed = await listGithubSemverTags(existing.repo_url);
  const newest = listed.tags?.[0] ?? null;
  const pinnedSem = parseSemverTag(repo_ref);
  let latest_version = version;
  let latest_repo_ref = repo_ref;
  if (newest && pinnedSem && compareSemver(newest, pinnedSem) > 0) {
    latest_version = newest.version;
    latest_repo_ref = newest.tag;
  }

  const suggested = inspect.suggested || {};
  await query(
    `UPDATE w4h.contributions SET
       repo_ref = $2,
       version = $3,
       latest_version = $4,
       latest_repo_ref = $5,
       version_checked_at = NOW(),
       title = COALESCE(NULLIF($6, ''), title),
       summary = COALESCE(NULLIF($7, ''), summary),
       kind = COALESCE(NULLIF($8, ''), kind),
       kinds = CASE WHEN cardinality($9::text[]) > 0 THEN $9::text[] ELSE kinds END,
       language = COALESCE(NULLIF($10, ''), language),
       docs_path = COALESCE(NULLIF($11, ''), docs_path),
       updated_at = NOW()
     WHERE lower(slug) = lower($1)`,
    [
      existing.slug,
      repo_ref,
      version,
      latest_version,
      latest_repo_ref,
      suggested.title || '',
      suggested.summary ?? null,
      suggested.kind || '',
      suggested.kinds || [],
      suggested.language || '',
      suggested.docs_path || '',
    ]
  );

  const row = await getContribution(existing.slug, { includeUnpublished: true });
  return { ok: true, preview, row };
}
