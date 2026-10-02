/**
 * Triggers a runnable catalog contribution on the calling user's behalf (ADR-022,
 * ADR-023). Python importers use standardized `run.sh`; SQL jobs call
 * runSqlContribution() in-process — never HTTP-loop to the dataset SQL route.
 *
 * Modality B1 only for run.sh (no Docker): script lives in a sibling checkout.
 */
import { execFile, spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import { promisify } from 'util';
import { parse as parseYaml } from 'yaml';
import { query } from '../db.js';
import { getContribution } from './contributions.js';
import { loadUpload, listUploadFiles, removeUpload } from './contribution-uploads.js';
import { runSqlContribution } from './sql-contribution-engine.js';
import { refreshDatasetSummaryData } from './dataset-summary-refresh.js';
import { decryptUserApiKey } from './utils/apiKeyCrypto.js';
import { issueApiKeyForUser } from './utils/apiKeys.js';

const execFileAsync = promisify(execFile);

const SIBLING_REPOS_ROOT = process.env.CONTRIBUTION_RUNNER_REPO_ROOT
  || path.resolve(process.cwd(), '..');

const RUN_SELECT_COLS = `id, contribution_slug, dataset_id, started_by, status,
  started_at, finished_at, exit_code, log, progress, version, repo_ref`;

const PROGRESS_MARKER = '@@PROGRESS@@ ';

function contributionKinds(contribution) {
  return [...new Set([...(contribution?.kinds || []), contribution?.kind].filter(Boolean))];
}

function isSqlContribution(contribution) {
  return contributionKinds(contribution).includes('sql');
}

function scriptPathForSlug(slug) {
  return path.join(SIBLING_REPOS_ROOT, slug, 'run.sh');
}

function pairedDataSlugFor(slug) {
  return slug.endsWith('-importer') ? slug.slice(0, -'-importer'.length) : null;
}

function liveSampleSiblingSlug(repoRoot) {
  const yamlPath = path.join(repoRoot, 'contribution.yaml');
  if (!fs.existsSync(yamlPath)) return null;
  try {
    const doc = parseYaml(fs.readFileSync(yamlPath, 'utf8'));
    const sample = doc?.tests?.live?.sample;
    if (sample?.kind === 'sibling' && sample?.repo) return String(sample.repo);
  } catch (e) {
    console.warn(`[contribution-runner] could not read ${yamlPath}: ${e.message}`);
  }
  return null;
}

async function isDirtyWorktree(dest) {
  try {
    const { stdout } = await execFileAsync('git', ['-C', dest, 'status', '--porcelain'], { timeout: 10_000 });
    return String(stdout || '').trim().length > 0;
  } catch {
    return true;
  }
}

async function isDetachedHead(dest) {
  try {
    await execFileAsync('git', ['-C', dest, 'symbolic-ref', '-q', 'HEAD'], { timeout: 5_000 });
    return false;
  } catch {
    return true;
  }
}

async function checkoutPinnedRef(dest, ref) {
  await execFileAsync('git', ['-C', dest, 'fetch', '--tags', '--force', 'origin'], { timeout: 30_000 });
  try {
    await execFileAsync('git', ['-C', dest, 'checkout', '--detach', ref], { timeout: 15_000 });
  } catch {
    await execFileAsync('git', ['-C', dest, 'checkout', '--detach', `origin/${ref}`], { timeout: 15_000 });
  }
}

async function cloneAtRef(repoUrl, dest, ref) {
  try {
    await execFileAsync('git', ['clone', '--branch', ref, repoUrl, dest], { timeout: 60_000 });
  } catch {
    await execFileAsync('git', ['clone', repoUrl, dest], { timeout: 60_000 });
    await execFileAsync('git', ['-C', dest, 'checkout', '--detach', ref], { timeout: 15_000 });
  }
}

async function ensureRepoCloned(slug, repoUrl, repoRef) {
  const dest = path.join(SIBLING_REPOS_ROOT, slug);
  const ref = repoRef || 'main';
  if (!fs.existsSync(dest)) {
    await cloneAtRef(repoUrl, dest, ref);
    return;
  }
  if (await isDirtyWorktree(dest)) {
    console.warn(`[contribution-runner] ${slug} has a dirty working tree (leaving as-is)`);
    return;
  }
  // Local sibling checkouts stay on their branch (developer working copy).
  // Detached HEAD is a runner-managed clone: fetch tags and check out the pin.
  if (!(await isDetachedHead(dest))) {
    try {
      await execFileAsync('git', ['-C', dest, 'pull', '--ff-only', 'origin', ref], { timeout: 30_000 });
    } catch (e) {
      console.warn(`[contribution-runner] could not fast-forward ${slug} (leaving as-is): ${e.message}`);
    }
    return;
  }
  try {
    await checkoutPinnedRef(dest, ref);
  } catch (e) {
    console.warn(`[contribution-runner] could not checkout ${ref} for ${slug} (leaving as-is): ${e.message}`);
  }
}

async function ensureContributionRepos(contribution) {
  await ensureRepoCloned(contribution.slug, contribution.repo_url, contribution.repo_ref);
  const dest = path.join(SIBLING_REPOS_ROOT, contribution.slug);
  const sampleSlug = liveSampleSiblingSlug(dest) || pairedDataSlugFor(contribution.slug);
  if (!sampleSlug) return;
  const paired = await getContribution(sampleSlug, { includeUnpublished: true });
  if (!paired) return;
  try {
    await ensureRepoCloned(paired.slug, paired.repo_url, paired.repo_ref);
  } catch (e) {
    console.warn(`[contribution-runner] could not prepare sample repo ${sampleSlug}: ${e.message}`);
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

function handleStdoutLine(runId, line) {
  if (line.startsWith(PROGRESS_MARKER)) {
    try {
      const progress = JSON.parse(line.slice(PROGRESS_MARKER.length));
      updateProgress(runId, progress);
      return;
    } catch {
      // fall through
    }
  }
  appendLog(runId, `${line}\n`);
}

function formatSqlRunLog(result) {
  if (result.write_back) {
    const lines = [
      `Materialized derived signal: ${result.signal || '?'}`,
      `Table: ${result.physical_schema}.${result.physical_table}`,
      `Rows inserted: ${result.inserted ?? 0}`,
    ];
    if (result.table_prefix) lines.push(`Table prefix: ${result.table_prefix}`);
    return `${lines.join('\n')}\n`;
  }
  const lines = [
    `Preview (${result.row_count ?? 0} row(s), write_back=false)`,
  ];
  if (result.columns?.length) lines.push(`Columns: ${result.columns.join(', ')}`);
  return `${lines.join('\n')}\n`;
}

async function finalizeSqlRun(runId, result) {
  if (!result.ok) {
    await query(
      `UPDATE w4h.contribution_runs
       SET status = 'failed', finished_at = NOW(), exit_code = 1, log = $2
       WHERE id = $1`,
      [runId, `${result.error || 'SQL contribution failed'}\n`]
    );
    return;
  }
  const progress = {
    write_back: result.write_back,
    row_count: result.row_count ?? null,
    inserted: result.inserted ?? null,
    physical_table: result.physical_table ?? null,
    signal: result.signal ?? null,
  };
  await query(
    `UPDATE w4h.contribution_runs
     SET status = 'succeeded', finished_at = NOW(), exit_code = 0,
         log = $2, progress = $3::jsonb
     WHERE id = $1`,
    [runId, formatSqlRunLog(result), JSON.stringify(progress)]
  );
  refreshSummaryAfterRun(runId);
}

/**
 * A successful run may have loaded or derived data: bring the Analytics cohort cards up to date
 * (ADR-019). Best-effort — a failure here never changes the run's outcome.
 */
function refreshSummaryAfterRun(runId) {
  query(`SELECT dataset_id FROM w4h.contribution_runs WHERE id = $1`, [runId])
    .then((r) => (r.rowCount ? refreshDatasetSummaryData(r.rows[0].dataset_id) : null))
    .catch((e) => console.warn('summary refresh after run failed:', e.message));
}

async function startSqlContributionRun({ slug, datasetId, user, contribution }) {
  const insertRes = await query(
    `INSERT INTO w4h.contribution_runs (contribution_slug, dataset_id, started_by, status, version, repo_ref)
     VALUES ($1, $2, $3, 'running', $4, $5) RETURNING ${RUN_SELECT_COLS}`,
    [slug, datasetId, user.id, contribution?.version || null, contribution?.repo_ref || null]
  );
  const run = insertRes.rows[0];
  try {
    const result = await runSqlContribution(user, datasetId, slug, {});
    await finalizeSqlRun(run.id, result);
    const statusRes = await query(
      `SELECT ${RUN_SELECT_COLS} FROM w4h.contribution_runs WHERE id = $1`,
      [run.id]
    );
    return { ok: true, run: statusRes.rows[0], sqlResult: result };
  } catch (e) {
    await query(
      `UPDATE w4h.contribution_runs
       SET status = 'failed', finished_at = NOW(), exit_code = 1, log = $2
       WHERE id = $1`,
      [run.id, `${e.message || 'SQL contribution failed'}\n`]
    );
    const statusRes = await query(
      `SELECT ${RUN_SELECT_COLS} FROM w4h.contribution_runs WHERE id = $1`,
      [run.id]
    );
    return { ok: true, run: statusRes.rows[0] };
  }
}

/**
 * Queue and start a run. Importers return immediately while run.sh continues;
 * SQL jobs finish in-process but still write contribution_runs for polling parity.
 */
export async function startContributionRun({ slug, datasetId, user, mode = 'update', uploadId = null }) {
  const importMode = mode === 'full' ? 'replace' : 'append';
  const contribution = await getContribution(slug, { includeUnpublished: true });
  if (!contribution) {
    return { ok: false, status: 404, error: 'Contribution not found' };
  }
  if (!contribution.runnable) {
    return { ok: false, status: 400, error: 'This contribution is not runnable from the app' };
  }

  if (isSqlContribution(contribution)) {
    const result = await startSqlContributionRun({ slug, datasetId, user, contribution });
    if (!result.ok) return result;
    return {
      ok: true,
      run: result.run,
      sqlResult: result.sqlResult,
      runKind: 'sql',
    };
  }

  let dataDir = null;
  if (uploadId) {
    const upload = loadUpload(uploadId, user);
    if (!upload.ok) return { ok: false, status: upload.status, error: upload.error };
    if (upload.owner.slug !== slug || upload.owner.dataset_id !== datasetId) {
      return { ok: false, status: 400, error: 'Upload does not belong to this contribution/dataset' };
    }
    if (!listUploadFiles(upload.dir).length) {
      return { ok: false, status: 400, error: 'Upload has no files (it may have expired); choose your files again' };
    }
    dataDir = upload.dir;
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
    `INSERT INTO w4h.contribution_runs (contribution_slug, dataset_id, started_by, status, version, repo_ref)
     VALUES ($1, $2, $3, 'queued', $4, $5) RETURNING ${RUN_SELECT_COLS}`,
    [slug, datasetId, user.id, contribution.version || null, contribution.repo_ref || null]
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
    return { ok: true, run: { ...run, status: 'failed' }, runKind: 'import' };
  }

  const env = {
    ...process.env,
    W4H_API_KEY: apiKey,
    W4H_API_BASE: defaultApiBase(),
    W4H_DATASET_ID: datasetId,
    ...(dataDir ? { W4H_DATA_DIR: dataDir } : {}),
    PYTHONUNBUFFERED: '1',
  };

  await query(`UPDATE w4h.contribution_runs SET status = 'running' WHERE id = $1`, [run.id]);

  const child = spawn(scriptPath, ['import', '--mode', importMode], { env, cwd: path.dirname(scriptPath) });
  const stdoutRouter = createLineRouter((line) => handleStdoutLine(run.id, line));
  child.stdout.on('data', (chunk) => stdoutRouter.push(chunk.toString()));
  child.stderr.on('data', (chunk) => appendLog(run.id, chunk.toString()));

  const cleanupUpload = () => {
    try {
      if (uploadId) removeUpload(uploadId);
    } catch (e) {
      console.error('contribution run: upload cleanup failed:', e.message);
    }
  };

  child.on('error', (err) => {
    stdoutRouter.flush();
    query(
      `UPDATE w4h.contribution_runs
       SET status = 'failed', finished_at = NOW(), log = log || $2
       WHERE id = $1`,
      [run.id, `\n[run.sh failed to start: ${err.message}]`]
    ).catch(() => {});
    cleanupUpload();
  });

  child.on('close', (code) => {
    stdoutRouter.flush();
    query(
      `UPDATE w4h.contribution_runs
       SET status = $2, exit_code = $3, finished_at = NOW()
       WHERE id = $1`,
      [run.id, code === 0 ? 'succeeded' : 'failed', code]
    )
      .then(() => { if (code === 0) refreshSummaryAfterRun(run.id); })
      .catch(() => {});
    cleanupUpload();
  });

  return { ok: true, run: { ...run, status: 'running' }, runKind: 'import' };
}

export async function getContributionRun(runId) {
  const res = await query(
    `SELECT ${RUN_SELECT_COLS} FROM w4h.contribution_runs WHERE id = $1`,
    [runId]
  );
  return res.rows[0] || null;
}

export { isSqlContribution, contributionKinds };
