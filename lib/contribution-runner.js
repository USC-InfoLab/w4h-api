/**
 * Triggers a runnable catalog contribution on the calling user's behalf (ADR-022,
 * ADR-023). Python importers use standardized `run.sh`; SQL jobs call
 * runSqlContribution() in-process — never HTTP-loop to the dataset SQL route.
 *
 * Modality B1 for run.sh: the script lives in a sibling checkout. With
 * CONTRIBUTION_RUNNER_URL set (Docker), run.sh runs in the runner service instead
 * (contribution-delegate.js); this process keeps the run row, log and cleanup.
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
import { assertCanIngestDataset } from './ingest-csv.js';
import {
  delegateUrl, loadDelegateModules, delegateModule, delegateRunnerReachable, startDelegatedChild,
} from './contribution-delegate.js';
import { refreshDatasetSummaryData } from './dataset-summary-refresh.js';
import { decryptUserApiKey } from './utils/apiKeyCrypto.js';
import { issueApiKeyForUser } from './utils/apiKeys.js';
import { recordUsageEventSafe } from './utils/usage.js';

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

// Keep database writes in the order this process observes output and completion.
const runQueues = new Map();
function serial(runId, task) {
  const result = (runQueues.get(runId) || Promise.resolve()).then(task);
  // A failed write must not block later writes; the caller still sees its rejection.
  const tail = result.catch(() => {});
  runQueues.set(runId, tail);
  tail.then(() => { if (runQueues.get(runId) === tail) runQueues.delete(runId); });
  return result;
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

const OUTPUT_INCOMPLETE_NOTICE = '\n[run output incomplete: one or more log writes failed]\n';

const snapshotKey = (s) => s?.signal_index ?? s?.signal;

/**
 * Run totals from `@@PROGRESS@@` snapshots. Each is cumulative for its signal, so keep the
 * last one per signal and sum. Null when none carries a numeric `inserted`.
 */
export function runTotals(snapshots) {
  const last = new Map();
  for (const s of snapshots) if (Number.isFinite(s?.inserted)) last.set(snapshotKey(s), s);
  if (!last.size) return null;
  const rows = [...last.values()];
  const sum = (k) => rows.reduce((n, s) => n + (Number.isFinite(s[k]) ? s[k] : 0), 0);
  const planned = rows.map((s) => s.signal_count).filter(Number.isFinite);
  return {
    row_count: sum('inserted'),
    rows_skipped: sum('skipped'),
    signals: last.size,
    signals_planned: planned.length ? Math.max(...planned) : null,
  };
}

/**
 * The one usage event per finished run (ADR-031 §7, D-19b). `meta` carries counts and enums
 * only: never a slug, dataset id, file name, key or log text. Never throws.
 */
function recordRunEvent({ userId, ok, startedAt, finishedAt, meta = {}, recordUsage = recordUsageEventSafe }) {
  try {
    const durationMs = new Date(finishedAt) - new Date(startedAt);
    recordUsage({
      userId,
      eventType: ok ? 'contribution.run.succeeded' : 'contribution.run.failed',
      durationMs: Number.isFinite(durationMs) ? durationMs : null,
      metadata: Object.fromEntries(
        Object.entries({ route: '/contributions/:slug/run', ...meta }).filter(([, v]) => v != null)
      ),
    });
  } catch (e) {
    console.warn('contribution run: usage event failed:', e.message);
  }
}

/** Observe one importer process. `close` follows stdio closure and any spawn `error`. */
export function monitorImporterRun({
  runId, child, cleanupUpload, persist = query, refreshSummary = refreshSummaryAfterRun, recordUsage = recordUsageEventSafe,
  runMeta = {},
}) {
  const snapshots = new Map(); // last progress snapshot per signal, for runTotals
  let logWriteFailed = false;
  let progressWriteFailed = false;
  let spawnError = null;

  const appendLog = (text) => serial(runId, async () => {
    try {
      const result = await persist(
        `UPDATE w4h.contribution_runs SET log = log || $2 WHERE id = $1`,
        [runId, text]
      );
      if (result?.rowCount === 0) throw new Error('run row is missing');
    } catch (e) {
      if (!logWriteFailed) console.error(`contribution run ${runId}: log write failed:`, e.message);
      logWriteFailed = true;
    }
  });

  const updateProgress = (progress) => serial(runId, async () => {
    try {
      const result = await persist(
        `UPDATE w4h.contribution_runs SET progress = $2::jsonb WHERE id = $1`,
        [runId, JSON.stringify(progress)]
      );
      if (result?.rowCount === 0) throw new Error('run row is missing');
    } catch (e) {
      // Progress is a cosmetic snapshot, separate from the run log.
      if (!progressWriteFailed) console.warn(`contribution run ${runId}: progress write failed:`, e.message);
      progressWriteFailed = true;
    }
  });

  const stdoutRouter = createLineRouter((line) => {
    if (line.startsWith(PROGRESS_MARKER)) {
      try {
        const snapshot = JSON.parse(line.slice(PROGRESS_MARKER.length));
        // A snapshot without a count (e.g. rows_done only) must not replace the signal's last count.
        if (Number.isFinite(snapshot?.inserted)) snapshots.set(snapshotKey(snapshot), snapshot);
        updateProgress(snapshot);
        return;
      } catch {
        // Malformed markers remain visible in the log.
      }
    }
    appendLog(`${line}\n`);
  });

  // Stream decoding preserves UTF-8 characters split across chunks.
  child.stdout?.setEncoding('utf8');
  child.stderr?.setEncoding('utf8');
  child.stdout?.on('data', (chunk) => stdoutRouter.push(chunk));
  child.stderr?.on('data', (chunk) => appendLog(chunk));
  child.on('error', (err) => {
    spawnError = err;
    console.error(`contribution run ${runId}: run.sh failed to start:`, err.message);
  });

  return new Promise((resolve) => {
    child.on('close', async (code, signal) => {
      stdoutRouter.flush();
      // The process no longer needs its files, even if database writes stall.
      try { cleanupUpload(); } catch (e) {
        console.error(`contribution run ${runId}: upload cleanup failed:`, e.message);
      }
      const succeeded = code === 0 && !spawnError;
      let finalized = false;
      try {
        await serial(runId, async () => {
          const notices = [];
          if (spawnError) notices.push(`\n[run.sh failed to start (${spawnError.code || 'unknown error'})]\n`);
          else if (signal) notices.push(`\n[run.sh terminated by signal ${signal}]\n`);
          if (logWriteFailed) notices.push(OUTPUT_INCOMPLETE_NOTICE);
          const result = await persist(
            `UPDATE w4h.contribution_runs
             SET status = $2, exit_code = $3, finished_at = NOW(), log = log || $4
             WHERE id = $1
             RETURNING started_by, started_at, finished_at`,
            [runId, succeeded ? 'succeeded' : 'failed', code, notices.join('')]
          );
          if (result?.rowCount === 0) throw new Error('run row is missing');
          const run = result?.rows?.[0];
          if (run) {
            recordRunEvent({
              userId: run.started_by,
              ok: succeeded,
              startedAt: run.started_at,
              finishedAt: run.finished_at,
              recordUsage,
              meta: {
                ...runMeta,
                ...runTotals(snapshots.values()),
                exit_code: Number.isInteger(code) ? code : null,
                // The delegate client names runner-side failures; anything else is the importer's exit.
                failure: succeeded ? null : spawnError ? 'spawn_error' : child.failure || 'importer_exit',
              },
            });
          }
        });
        finalized = true;
        if (succeeded) {
          try { refreshSummary(runId); } catch (e) {
            console.warn(`contribution run ${runId}: summary refresh failed:`, e.message);
          }
        }
      } catch (e) {
        console.error(`contribution run ${runId}: final update failed:`, e.message);
      } finally {
        resolve(finalized);
      }
    });
  });
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

/** Usage-event meta for a SQL run: rows written (write-back) or returned (preview). */
export function sqlRunMeta(result) {
  if (!result.ok) return { runner: 'sql', exit_code: 1, failure: 'sql_error' };
  return result.write_back
    ? { runner: 'sql', mode: 'write_back', exit_code: 0, row_count: result.inserted ?? null }
    : { runner: 'sql', mode: 'preview', exit_code: 0, row_count: result.row_count ?? null };
}

async function finalizeSqlRun(runId, result, {
  persist = query, recordUsage = recordUsageEventSafe, refreshSummary = refreshSummaryAfterRun,
} = {}) {
  const record = (res) => {
    const run = res?.rows?.[0];
    if (run) {
      recordRunEvent({
        userId: run.started_by, ok: result.ok, startedAt: run.started_at, finishedAt: run.finished_at,
        meta: sqlRunMeta(result), recordUsage,
      });
    }
  };
  if (!result.ok) {
    record(await persist(
      `UPDATE w4h.contribution_runs
       SET status = 'failed', finished_at = NOW(), exit_code = 1, log = $2
       WHERE id = $1
       RETURNING started_by, started_at, finished_at`,
      [runId, `${result.error || 'SQL contribution failed'}\n`]
    ));
    return;
  }
  const progress = {
    write_back: result.write_back,
    row_count: result.row_count ?? null,
    inserted: result.inserted ?? null,
    physical_table: result.physical_table ?? null,
    signal: result.signal ?? null,
  };
  record(await persist(
    `UPDATE w4h.contribution_runs
     SET status = 'succeeded', finished_at = NOW(), exit_code = 0,
         log = $2, progress = $3::jsonb
     WHERE id = $1
     RETURNING started_by, started_at, finished_at`,
    [runId, formatSqlRunLog(result), JSON.stringify(progress)]
  ));
  refreshSummary(runId);
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

/** Deps are injectable for tests; the route passes none. */
export async function startSqlContributionRun({
  slug, datasetId, user, contribution,
  persist = query, runSql = runSqlContribution, recordUsage = recordUsageEventSafe, refreshSummary = refreshSummaryAfterRun,
}) {
  const deps = { persist, recordUsage, refreshSummary };
  const insertRes = await persist(
    `INSERT INTO w4h.contribution_runs (contribution_slug, dataset_id, started_by, status, version, repo_ref)
     VALUES ($1, $2, $3, 'running', $4, $5) RETURNING ${RUN_SELECT_COLS}`,
    [slug, datasetId, user.id, contribution?.version || null, contribution?.repo_ref || null]
  );
  const run = insertRes.rows[0];
  let finalized = false;
  try {
    const result = await runSql(user, datasetId, slug, {});
    await finalizeSqlRun(run.id, result, deps);
    finalized = true;
    const statusRes = await persist(
      `SELECT ${RUN_SELECT_COLS} FROM w4h.contribution_runs WHERE id = $1`,
      [run.id]
    );
    return { ok: true, run: statusRes.rows[0], sqlResult: result };
  } catch (e) {
    // One outcome per run: a status read failing after finalize must not re-finalize it as failed.
    if (!finalized) await finalizeSqlRun(run.id, { ok: false, error: e.message }, deps);
    const statusRes = await persist(
      `SELECT ${RUN_SELECT_COLS} FROM w4h.contribution_runs WHERE id = $1`,
      [run.id]
    );
    return { ok: true, run: statusRes.rows[0] };
  }
}

/**
 * The Docker api image has no Python (w4h-docker sets CONTRIBUTION_RUNNER_NO_PYTHON): say
 * where importers run instead of failing later on a clone or a missing run.sh. Null when
 * Python contributions can run here. SQL contributions are not affected.
 */
export function pythonRunUnavailable() {
  if (!process.env.CONTRIBUTION_RUNNER_NO_PYTHON) return null;
  return {
    status: 503,
    error: 'In-app Run of importers is not available in this Docker install (the API container has no Python). '
      + 'Open Jupyter (the link Start W4H prints) and run the importer notebook in the work folder, '
      + 'e.g. import_embraceplus_csv_file.ipynb (EmbracePlus) or import_fitbit_csv.ipynb (Fitbit sample).',
  };
}

/**
 * Docker: the runner service does not have this importer baked in. Null when not
 * delegating, when the runner has it, or when the runner was unreachable (the run then
 * fails with the runner client's readable line instead). Call loadDelegateModules() first.
 */
export function delegatedRunUnavailable(slug) {
  if (!delegateUrl() || !delegateRunnerReachable() || delegateModule(slug)) return null;
  return {
    ok: false,
    status: 503,
    error: `${slug} is not included in this installation's runner. Open Jupyter to run it from a notebook.`,
  };
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

  const unavailable = pythonRunUnavailable();
  if (unavailable) return { ok: false, ...unavailable };

  let dataDir = null;
  let uploadMeta = {};
  if (uploadId) {
    const upload = loadUpload(uploadId, user);
    if (!upload.ok) return { ok: false, status: upload.status, error: upload.error };
    if (upload.owner.slug !== slug || upload.owner.dataset_id !== datasetId) {
      return { ok: false, status: 400, error: 'Upload does not belong to this contribution/dataset' };
    }
    const files = listUploadFiles(upload.dir);
    if (!files.length) {
      return { ok: false, status: 400, error: 'Upload has no files (it may have expired); choose your files again' };
    }
    dataDir = upload.dir;
    uploadMeta = { files: files.length, bytes: files.reduce((n, f) => n + f.size, 0) };
  }

  // The SQL branch above runs its own ACL; an importer writes into the dataset with the caller's key.
  const access = await assertCanIngestDataset(user, datasetId);
  if (!access.ok) return { ok: false, status: access.status, error: access.error };

  const delegated = Boolean(delegateUrl());
  let scriptPath = null;
  let version = contribution.version || null;
  let repoRef = contribution.repo_ref || null;
  if (delegated) {
    await loadDelegateModules();
    const refused = delegatedRunUnavailable(slug);
    if (refused) return refused;
    // Record what the runner has baked; unknown only when the runner was unreachable.
    version = delegateModule(slug)?.version ?? null;
    repoRef = null;
  } else {
    try {
      await ensureContributionRepos(contribution);
    } catch (e) {
      return { ok: false, status: 500, error: `Could not prepare ${slug}'s repository: ${e.message}` };
    }

    scriptPath = scriptPathForSlug(slug);
    if (!fs.existsSync(scriptPath)) {
      return { ok: false, status: 500, error: `run.sh not found for ${slug} (expected ${scriptPath})` };
    }
  }

  const insertRes = await query(
    `INSERT INTO w4h.contribution_runs (contribution_slug, dataset_id, started_by, status, version, repo_ref)
     VALUES ($1, $2, $3, 'queued', $4, $5) RETURNING ${RUN_SELECT_COLS}`,
    [slug, datasetId, user.id, version, repoRef]
  );
  const run = insertRes.rows[0];
  const runMeta = { runner: delegated ? 'delegate' : 'local', mode: importMode, ...uploadMeta };

  let apiKey;
  try {
    apiKey = await resolveApiKeyForUser(user.id);
  } catch (e) {
    const failed = await query(
      `UPDATE w4h.contribution_runs SET status = 'failed', finished_at = NOW(), log = $2 WHERE id = $1
       RETURNING finished_at`,
      [run.id, `Could not resolve an API key for this user: ${e.message}`]
    );
    recordRunEvent({
      userId: user.id, ok: false, startedAt: run.started_at, finishedAt: failed.rows?.[0]?.finished_at,
      meta: { ...runMeta, failure: 'key_unavailable' },
    });
    return { ok: true, run: { ...run, status: 'failed' }, runKind: 'import' };
  }

  await query(`UPDATE w4h.contribution_runs SET status = 'running' WHERE id = $1`, [run.id]);

  let child;
  if (delegated) {
    // The runner pulls the upload's files over HTTP with this key; never log the body.
    child = startDelegatedChild({
      slug, api_key: apiKey, dataset_id: datasetId, mode: importMode, upload_id: uploadId || undefined,
    });
  } else {
    const env = {
      ...process.env,
      W4H_API_KEY: apiKey,
      W4H_API_BASE: defaultApiBase(),
      W4H_DATASET_ID: datasetId,
      ...(dataDir ? { W4H_DATA_DIR: dataDir, W4H_PLAN_UPLOAD: '1' } : {}),
      PYTHONUNBUFFERED: '1',
    };
    const runArgs = ['import', '--mode', importMode];
    if (dataDir) runArgs.push('--data-dir', dataDir);
    child = spawn(scriptPath, runArgs, { env, cwd: path.dirname(scriptPath) });
  }

  const cleanupUpload = () => {
    if (uploadId) removeUpload(uploadId);
  };

  monitorImporterRun({ runId: run.id, child, cleanupUpload, runMeta });

  return { ok: true, run: { ...run, status: 'running' }, runKind: 'import' };
}

export async function getContributionRun(runId) {
  const res = await query(
    `SELECT ${RUN_SELECT_COLS} FROM w4h.contribution_runs WHERE id = $1`,
    [runId]
  );
  return res.rows[0] || null;
}

/**
 * Boot-time cleanup: a `queued`/`running` row is only ever advanced by this process
 * (monitorImporterRun, or the SQL path), and that dies with it. Any such row at startup is
 * orphaned from a prior process (crash, kill, restart) and would stay `running` forever.
 * `before` is the boot time: the reap runs after the routes are live, so a run started in
 * that window belongs to this process and must be left alone.
 */
export async function reapOrphanedRuns({ persist = query, before = new Date(), recordUsage = recordUsageEventSafe } = {}) {
  const res = await persist(
    `UPDATE w4h.contribution_runs
     SET status = 'failed', finished_at = NOW(), exit_code = NULL,
         log = log || $1
     WHERE status IN ('queued', 'running') AND started_at < $2
     RETURNING id, contribution_slug, started_by, started_at, finished_at`,
    ['\n[orphaned: API process restarted while this run was in progress]\n', before]
  );
  if (res.rowCount) {
    console.warn(`[contribution-runner] reaped ${res.rowCount} orphaned run(s) on boot: ${res.rows.map(r => `${r.contribution_slug}(${r.id})`).join(', ')}`);
    for (const r of res.rows) {
      recordRunEvent({
        userId: r.started_by, ok: false, startedAt: r.started_at, finishedAt: r.finished_at, recordUsage,
        meta: { failure: 'api_restart' },
      });
    }
  }
  return res.rowCount;
}

export { isSqlContribution, contributionKinds };
