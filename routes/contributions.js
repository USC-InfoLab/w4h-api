import express from 'express';
import { authenticate, authenticateJwtOrApiKey } from '../auth.js';
import { isSuperAdmin } from '../lib/utils/user.js';
import {
  listContributions,
  getContribution,
  createContribution,
  updateContribution,
  deleteContribution,
} from '../lib/contributions.js';

const router = express.Router();

function requireSuperAdmin(req, res, next) {
  if (!isSuperAdmin(req.user?.role)) {
    return res.status(403).json({ error: 'Unauthorized' });
  }
  next();
}

router.get('/', authenticateJwtOrApiKey, list);
router.get('/:slug', authenticateJwtOrApiKey, get);
router.post('/', authenticate, requireSuperAdmin, create);
router.patch('/:slug', authenticate, requireSuperAdmin, patch);
router.delete('/:slug', authenticate, requireSuperAdmin, remove);

async function list(req, res) {
  try {
    const q = typeof req.query.q === 'string' ? req.query.q : '';
    const kind = typeof req.query.kind === 'string' ? req.query.kind : '';
    const language = typeof req.query.language === 'string' ? req.query.language : '';
    const includeUnpublished = isSuperAdmin(req.user?.role) && req.query.all === '1';
    const rows = await listContributions({ q, kind, language, includeUnpublished });
    res.json(rows);
  } catch (e) {
    res.status(500).json({ error: e.message || 'Failed to load contributions' });
  }
}

async function get(req, res) {
  try {
    const includeUnpublished = isSuperAdmin(req.user?.role);
    const row = await getContribution(req.params.slug, { withDocs: true, includeUnpublished });
    if (!row) return res.status(404).json({ error: 'Contribution not found' });
    res.json(row);
  } catch (e) {
    res.status(500).json({ error: e.message || 'Failed to load contribution' });
  }
}

async function create(req, res) {
  try {
    const result = await createContribution(req.body || {}, req.user?.id);
    if (!result.ok) return res.status(result.status || 400).json({ error: result.error });
    res.status(201).json(result.row);
  } catch (e) {
    res.status(500).json({ error: e.message || 'Failed to create contribution' });
  }
}

async function patch(req, res) {
  try {
    const result = await updateContribution(req.params.slug, req.body || {});
    if (!result.ok) return res.status(result.status || 400).json({ error: result.error });
    res.json(result.row);
  } catch (e) {
    res.status(500).json({ error: e.message || 'Failed to update contribution' });
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

export default router;
