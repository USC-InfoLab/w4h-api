/**
 * Triggers a runnable catalog contribution's standardized `run.sh` entrypoint
 * on the calling user's behalf (ADR-022). Authorization for the actual
 * dataset write happens where it already does — the ingest endpoint's
 * `assertCanIngestDataset` check, hit by the script's own API call — not here.
 *
 * Modality B1 only (no Docker): `run.sh` lives in a sibling checkout of the
 * contribution's repo, next to this one, named by its slug. That checkout
 * (and its ADR-016 paired data repo, if any) is cloned on first run and
 * best-effort fast-forwarded on later runs — see `ensureContributionRepos`.
 */
import { execFile, spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import { promisify } from 'util';
import { query } from '../db.js';
import { getContribution } from './contributions.js';
import { decryptUserApiKey } from './utils/apiKeyCrypto.js';
import { issueApiKeyForUser } from './utils/apiKeys.js';

const execFileAsync = promisify(execFile);

const SIBLING_REPOS_ROOT = process.env.CONTRIBUTION_RUNNER_REPO_ROOT
  || path.resolve(process.cwd(), '..');

const RUN_SELECT_COLS = `id, contribution_slug, dataset_id, started_by, status,
  started_at, finished_at, exit_code, log, progress`;

// Matches PROGRESS_MARKER in w4h_sample_fitbit_csv_importer/csv_import.py — any
// runnable contribution's script can emit this line-prefixed JSON convention to
// drive the in-app progress bar instead of leaving users staring at a spinner.
const PROGRESS_MARKER = '@@PROGRESS@@ ';

function scriptPathForSlug(slug) {
  return path.join(SIBLING_REPOS_ROOT, slug, 'run.sh');
}

// ADR-016 paired sample+importer naming (e.g. w4h-sample-fitbit-csv-importer /
// w4h-sample-fitbit-csv) — there's no explicit FK for this in the catalog, so
// it's inferred from the slug. run.sh's default paths assume its sibling data
// repo is checked out next to it (see e.g. W4H_SAMPLE_FITBIT_CSV_PATH).
function pairedDataSlugFor(slug) {
  return slug.endsWith('-importer') ? slug.slice(0, -'-importer'.length) : null;
}

// Clone a contribution's repo next to this one if it's missing; if it's
// already there, best-effort fast-forward it. Never force/reset — a sibling
// checkout may be someone's own working copy with local changes, and losing
// that silently would be far worse than running against a stale checkout.
async function ensureRepoCloned(slug, repoUrl, repoRef) {
  const dest = path.join(SIBLING_REPOS_ROOT, slug);
  const ref = repoRef || 'main';
  if (fs.existsSync(dest)) {
    try {
      await execFileAsync('git', ['-C', dest, 'pull', '--ff-only', 'origin', ref], { timeout: 30_000 });
    } catch (e) {
      console.warn(`[contribution-runner] could not fast-forward ${slug} (leaving as-is): ${e.message}`);
    }
    return;
  }
  await execFileAsync('git', ['clone', '--branch', ref, '--single-branch', repoUrl, dest], { timeout: 60_000 });
}

async function ensureContributionRepos(contribution) {
  await ensureRepoCloned(contribution.slug, contribution.repo_url, contribution.repo_ref);
  const pairedSlug = pairedDataSlugFor(contribution.slug);
  if (!pairedSlug) return;
  const paired = await getContribution(pairedSlug, { includeUnpublished: true });
  if (!paired) return;
  try {
    await ensureRepoCloned(paired.slug, paired.repo_url, paired.repo_ref);
  } catch (e) {
    // Best-effort: the paired data repo isn't required for every mode
    // (e.g. `sync --file ...` supplies its own data); let run.sh surface it.
    console.warn(`[contribution-runner] could not prepare paired repo ${pairedSlug}: ${e.message}`);
  }
}

function defaultApiBase() {
  return process.env.CONTRIBUTION_RUNNER_API_BASE
    || `http://localhost:${process.env.PORT || 2026}`;
}

async function resolveApiKeyForUser(userId) {
  const res = await query(`SELECT api_key_encrypted FROM w4h.users WHERE id = $1`, [userId]);
  const existing = decryptUserApiKey(res.rows[0]?.api_key_encrypted || null);
  if (existing) return existing;
  const issued = await issueApiKeyForUser(userId);
  return issued.api_key;
}

async function appendLog(runId, text) {
  await query(
    `UPDATE w4h.contribution_runs SET log = log || $2 WHERE id = $1`,
    [runId, text]
  ).catch(() => {});
}

async function updateProgress(runId, progress) {
  await query(
    `UPDATE w4h.contribution_runs SET progress = $2::jsonb WHERE id = $1`,
    [runId, JSON.stringify(progress)]
  ).catch(() => {});
}

// Buffers a stream that may split lines across chunk boundaries and hands
// complete lines to `onLine`. Call `flush()` once the stream ends to deliver
// any trailing partial line (e.g. a script that doesn't end its last write
// with a newline).
function createLineRouter(onLine) {
  let buffer = '';
  return {
    push(chunk) {
      buffer += chunk;
      const lines = buffer.split('\n');
      buffer = lines.pop();
      for (const line of lines) onLine(line);
    },
    flush() {
      if (buffer) {
        onLine(buffer);
        buffer = '';
      }
    },
  };
}

// Routes a runnable script's stdout: `@@PROGRESS@@ {...json...}` lines update
// the run's structured `progress` column (for the UI's progress bar) instead
// of the raw log; every other line is appended to `log` as before.
function handleStdoutLine(runId, line) {
  if (line.startsWith(PROGRESS_MARKER)) {
    try {
      const progress = JSON.parse(line.slice(PROGRESS_MARKER.length));
      updateProgress(runId, progress);
      return;
    } catch {
      // Not valid JSON — fall through and show it as a plain log line.
    }
  }
  appendLog(runId, `${line}\n`);
}

/**
 * Queue and start a run. Returns immediately with the run id; the script
 * continues in the background and updates the row as it progresses.
 *
 * `mode`: 'update' (default) appends/fills missing rows without touching
 * existing tables (run.sh import --mode append); 'full' truncates and
 * reloads every signal table from source (run.sh import --mode replace).
 * Both forward to run.sh's existing `import [extra CLI args]` contract —
 * no new entrypoint, just the `--mode` the importer CLI already accepts.
 */
export async function startContributionRun({ slug, datasetId, user, mode = 'update' }) {
  const importMode = mode === 'full' ? 'replace' : 'append';
  const contribution = await getContribution(slug, { includeUnpublished: true });
  if (!contribution) {
    return { ok: false, status: 404, error: 'Contribution not found' };
  }
  if (!contribution.runnable) {
    return { ok: false, status: 400, error: 'This contribution is not runnable from the app' };
  }

  try {
    await ensureContributionRepos(contribution);
  } catch (e) {
    return { ok: false, status: 500, error: `Could not prepare ${slug}'s repository: ${e.message}` };
  }

  const scriptPath = scriptPathForSlug(slug);
  if (!fs.existsSync(scriptPath)) {
    return { ok: false, status: 500, error: `run.sh not found for ${slug} (expected ${scriptPath})` };
  }

  const insertRes = await query(
    `INSERT INTO w4h.contribution_runs (contribution_slug, dataset_id, started_by, status)
     VALUES ($1, $2, $3, 'queued') RETURNING ${RUN_SELECT_COLS}`,
    [slug, datasetId, user.id]
  );
  const run = insertRes.rows[0];

  let apiKey;
  try {
    apiKey = await resolveApiKeyForUser(user.id);
  } catch (e) {
    await query(
      `UPDATE w4h.contribution_runs SET status = 'failed', finished_at = NOW(), log = $2 WHERE id = $1`,
      [run.id, `Could not resolve an API key for this user: ${e.message}`]
    );
    return { ok: true, run: { ...run, status: 'failed' } };
  }

  const env = {
    ...process.env,
    W4H_API_KEY: apiKey,
    W4H_API_BASE: defaultApiBase(),
    W4H_DATASET_ID: datasetId,
    // run.sh execs a Python console-script; with stdout piped (not a TTY, as it is
    // here), Python fully buffers stdout by default, so progress only appears in one
    // burst at exit instead of streaming into contribution_runs.log as it happens.
    PYTHONUNBUFFERED: '1',
  };

  await query(`UPDATE w4h.contribution_runs SET status = 'running' WHERE id = $1`, [run.id]);

  const child = spawn(scriptPath, ['import', '--mode', importMode], { env, cwd: path.dirname(scriptPath) });
  const stdoutRouter = createLineRouter((line) => handleStdoutLine(run.id, line));
  child.stdout.on('data', (chunk) => stdoutRouter.push(chunk.toString()));
  child.stderr.on('data', (chunk) => appendLog(run.id, chunk.toString()));

  child.on('error', (err) => {
    stdoutRouter.flush();
    query(
      `UPDATE w4h.contribution_runs
       SET status = 'failed', finished_at = NOW(), log = log || $2
       WHERE id = $1`,
      [run.id, `\n[run.sh failed to start: ${err.message}]`]
    ).catch(() => {});
  });

  child.on('close', (code) => {
    stdoutRouter.flush();
    query(
      `UPDATE w4h.contribution_runs
       SET status = $2, exit_code = $3, finished_at = NOW()
       WHERE id = $1`,
      [run.id, code === 0 ? 'succeeded' : 'failed', code]
    ).catch(() => {});
  });

  return { ok: true, run: { ...run, status: 'running' } };
}

export async function getContributionRun(runId) {
  const res = await query(
    `SELECT ${RUN_SELECT_COLS} FROM w4h.contribution_runs WHERE id = $1`,
    [runId]
  );
  return res.rows[0] || null;
}
