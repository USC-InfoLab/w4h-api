import express from 'express';
import { authenticate, authenticateJwtOrApiKey } from '../auth.js';
import { isSuperAdmin } from '../lib/utils/user.js';
import {
  listContributions,
  getContribution,
  createContribution,
  updateContribution,
  deleteContribution,
  presentContribution,
} from '../lib/contributions.js';
import { startContributionRun, getContributionRun } from '../lib/contribution-runner.js';
import { assertCanIngestDataset } from '../lib/ingest-csv.js';
import {
  createUpload, loadUpload, saveUploadFile, deleteUploadFile, listUploadFiles,
  MAX_FILES, MAX_FILE_SIZE_BYTES, MAX_TOTAL_SIZE_BYTES,
} from '../lib/contribution-uploads.js';
import { inspectRepository } from '../lib/contributions-manifest.js';
import {
  checkContributionUpdates,
  previewOrPinContributionVersion,
} from '../lib/contributions-versions.js';
import { writeAuditLogSafe } from '../lib/utils/audit.js';

const router = express.Router();

function requireSuperAdmin(req, res, next) {
  if (!isSuperAdmin(req.user?.role)) {
    return res.status(403).json({ error: 'Unauthorized' });
  }
  next();
}

function presentRows(rows, req) {
  const includeAdminFields = isSuperAdmin(req.user?.role);
  return rows.map((row) => presentContribution(row, { includeAdminFields }));
}

function presentRow(row, req) {
  return presentContribution(row, { includeAdminFields: isSuperAdmin(req.user?.role) });
}

router.get('/', authenticateJwtOrApiKey, list);
router.get('/runs/:id', authenticateJwtOrApiKey, getRun);
router.post('/inspect', authenticate, requireSuperAdmin, inspect);
router.post('/check-updates', authenticate, requireSuperAdmin, checkUpdates);
router.post('/:slug/update-version', authenticate, requireSuperAdmin, updateVersion);
// Admin-chosen files for importers that take them: stage, then run with upload_id.
router.post('/:slug/uploads', authenticateJwtOrApiKey, (req, res) => createUploadRoute(req, res));
router.get('/uploads/:uploadId', authenticateJwtOrApiKey, (req, res) => getUploadRoute(req, res));
router.put('/uploads/:uploadId/files/:name', authenticateJwtOrApiKey, (req, res) => putUploadFile(req, res));
router.delete('/uploads/:uploadId/files/:name', authenticateJwtOrApiKey, (req, res) => deleteUploadFileRoute(req, res));
router.get('/:slug', authenticateJwtOrApiKey, get);
router.post('/', authenticate, requireSuperAdmin, create);
router.post('/:slug/run', authenticateJwtOrApiKey, run);
router.patch('/:slug', authenticate, requireSuperAdmin, patch);
router.delete('/:slug', authenticate, requireSuperAdmin, remove);

async function list(req, res) {
  try {
    const q = typeof req.query.q === 'string' ? req.query.q : '';
    const kind = typeof req.query.kind === 'string' ? req.query.kind : '';
    const language = typeof req.query.language === 'string' ? req.query.language : '';
    const includeUnpublished = isSuperAdmin(req.user?.role) && req.query.all === '1';
    const rows = await listContributions({ q, kind, language, includeUnpublished });
    res.json(presentRows(rows, req));
  } catch (e) {
    res.status(500).json({ error: e.message || 'Failed to load contributions' });
  }
}

async function get(req, res) {
  try {
    const includeUnpublished = isSuperAdmin(req.user?.role);
    const row = await getContribution(req.params.slug, { withDocs: true, includeUnpublished });
    if (!row) return res.status(404).json({ error: 'Contribution not found' });
    res.json(presentRow(row, req));
  } catch (e) {
    res.status(500).json({ error: e.message || 'Failed to load contribution' });
  }
}

async function create(req, res) {
  try {
    const result = await createContribution(req.body || {}, req.user?.id);
    if (!result.ok) return res.status(result.status || 400).json({ error: result.error });
    res.status(201).json(presentRow(result.row, req));
  } catch (e) {
    res.status(500).json({ error: e.message || 'Failed to create contribution' });
  }
}

async function patch(req, res) {
  try {
    const result = await updateContribution(req.params.slug, req.body || {});
    if (!result.ok) return res.status(result.status || 400).json({ error: result.error });
    res.json(presentRow(result.row, req));
  } catch (e) {
    res.status(500).json({ error: e.message || 'Failed to update contribution' });
  }
}

async function inspect(req, res) {
  try {
    const result = await inspectRepository(req.body || {});
    if (result.error) return res.status(400).json({ error: result.error });
    res.json(result);
  } catch (e) {
    res.status(500).json({ error: e.message || 'Failed to inspect repository' });
  }
}

async function checkUpdates(req, res) {
  try {
    const slug = typeof req.body?.slug === 'string' ? req.body.slug.trim() : '';
    const result = await checkContributionUpdates({ slug: slug || undefined });
    if (!result.ok) return res.status(result.status || 400).json({ error: result.error });
    res.json({
      results: result.results,
      rows: presentRows(result.rows, req),
      updates: result.updates,
      without_releases: result.without_releases,
    });
  } catch (e) {
    res.status(500).json({ error: e.message || 'Failed to check for updates' });
  }
}

async function updateVersion(req, res) {
  try {
    const repoRef = req.body?.repo_ref != null ? String(req.body.repo_ref).trim() : '';
    const confirm = req.body?.confirm === true;
    const result = await previewOrPinContributionVersion(req.params.slug, repoRef, { confirm });
    if (!result.ok) return res.status(result.status || 400).json({ error: result.error });
    res.json({
      preview: result.preview,
      row: presentRow(result.row, req),
    });
  } catch (e) {
    res.status(500).json({ error: e.message || 'Failed to update contribution version' });
  }
}

async function remove(req, res) {
  try {
    const ok = await deleteContribution(req.params.slug);
    if (!ok) return res.status(404).json({ error: 'Contribution not found' });
    res.json({ success: true });
  } catch (e) {
    res.status(500).json({ error: e.message || 'Failed to delete contribution' });
  }
}

async function run(req, res) {
  try {
    const datasetId = String(req.body?.dataset_id || '').trim();
    if (!datasetId) return res.status(400).json({ error: 'dataset_id is required' });
    const mode = req.body?.mode === 'full' ? 'full' : 'update';
    const slug = req.params.slug;
    const uploadId = req.body?.upload_id ? String(req.body.upload_id) : null;
    const result = await startContributionRun({ slug, datasetId, user: req.user, mode, uploadId });
    if (!result.ok) return res.status(result.status || 400).json({ error: result.error });
    if (result.runKind === 'sql') {
      const sql = result.sqlResult;
      writeAuditLogSafe(req, {
        action: sql?.ok && sql.write_back ? 'dataset.derive' : 'dataset.query',
        resourceType: 'dataset',
        resourceId: datasetId,
        details: {
          contribution_slug: slug,
          run_id: result.run.id,
          write_back: sql?.write_back ?? null,
          physical_table: sql?.physical_table ?? null,
          inserted: sql?.inserted ?? null,
          row_count: sql?.row_count ?? null,
        },
      });
    } else {
      writeAuditLogSafe(req, {
        action: mode === 'full' ? 'dataset.import.full_reimport' : 'dataset.import.update',
        resourceType: 'dataset',
        resourceId: datasetId,
        details: { contribution_slug: slug, run_id: result.run.id, mode },
      });
    }
    res.status(202).json({ ...result.run, mode: result.runKind === 'import' ? mode : undefined });
  } catch (e) {
    res.status(500).json({ error: e.message || 'Failed to start run' });
  }
}

// ponytail: hardcoded slug; move to a contribution.yaml `inputs: files` flag when a 2nd module needs uploads
const UPLOAD_SLUGS = new Set(['w4h-fitbit-csv-file-importer']);

async function createUploadRoute(req, res) {
  try {
    const datasetId = String(req.body?.dataset_id || '').trim();
    if (!datasetId) return res.status(400).json({ error: 'dataset_id is required' });
    const contribution = await getContribution(req.params.slug, { includeUnpublished: isSuperAdmin(req.user?.role) });
    if (!contribution?.runnable || !UPLOAD_SLUGS.has(contribution.slug)) {
      return res.status(404).json({ error: 'This contribution does not accept uploaded files' });
    }
    const access = await assertCanIngestDataset(req.user, datasetId);
    if (!access.ok) return res.status(access.status).json({ error: access.error });
    const { id } = createUpload({ user: req.user, slug: contribution.slug, datasetId });
    res.status(201).json({
      upload_id: id,
      limits: { max_files: MAX_FILES, max_file_bytes: MAX_FILE_SIZE_BYTES, max_total_bytes: MAX_TOTAL_SIZE_BYTES },
    });
  } catch (e) {
    res.status(500).json({ error: e.message || 'Failed to create upload' });
  }
}

function withUpload(handler) {
  return async (req, res) => {
    try {
      const upload = loadUpload(req.params.uploadId, req.user);
      if (!upload.ok) return res.status(upload.status).json({ error: upload.error });
      await handler(req, res, upload);
    } catch (e) {
      res.status(500).json({ error: e.message || 'Upload request failed' });
    }
  };
}

const getUploadRoute = withUpload(async (_req, res, upload) => {
  res.json({ files: listUploadFiles(upload.dir) });
});

const putUploadFile = withUpload(async (req, res, upload) => {
  const result = await saveUploadFile(upload.dir, req.params.name, req);
  if (!result.ok) return res.status(result.status).json({ error: result.error });
  res.json({ name: result.name, size: result.size });
});

const deleteUploadFileRoute = withUpload(async (req, res, upload) => {
  deleteUploadFile(upload.dir, req.params.name);
  res.json({ success: true });
});

async function getRun(req, res) {
  try {
    const row = await getContributionRun(req.params.id);
    if (!row) return res.status(404).json({ error: 'Run not found' });
    res.json(row);
  } catch (e) {
    res.status(500).json({ error: e.message || 'Failed to load run' });
  }
}

export default router;
