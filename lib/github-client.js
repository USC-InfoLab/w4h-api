/**
 * Shared GitHub REST helpers for contribution inspect / version check.
 * Unauthenticated only (no token config) — see lib/contributions-manifest.js.
 */

export async function githubJson(url) {
  try {
    const res = await fetch(url, {
      headers: {
        Accept: 'application/vnd.github+json',
        'User-Agent': 'w4h-api-contributions-inspect',
      },
    });
    if (!res.ok) return { ok: false, status: res.status, data: null };
    return { ok: true, status: res.status, data: await res.json() };
  } catch (e) {
    return { ok: false, status: 0, data: null, error: e.message };
  }
}
