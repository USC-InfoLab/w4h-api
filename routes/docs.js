import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
const router = express.Router();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

import { authenticate } from '../auth.js';
import { USER_ROLES } from '../lib/utils/user.js';
import { query } from '../lib/utils/db.js';
import { renderDoc } from '../lib/utils/docRender.js';
import { pipeMarkdownZipToResponse } from '../lib/utils/markdownZip.js';
import { loadDocumentationConfig } from '../lib/utils/docMarkdownImport.js';

import crypto from 'crypto';

router.get('/', authenticate, list);
router.get('/search', authenticate, search);
router.get('/cards/feature', authenticate, featureCards);
router.get('/export/markdown-zip', authenticate, exportMarkdownZip);
router.get('/:slug', authenticate, get);
router.put('/:slug', authenticate, put);
router.delete('/:slug', authenticate, del);

const DOCS_TABLE = 'w4h.docs';

// Review buckets: export + CRUD :slug here; markdown seeding + YAML live under lib/db-init.js and lib/utils/docMarkdownImport.js.

async function featureCards(req, res) {
  if (![USER_ROLES.ADMIN, USER_ROLES.RESEARCHER].includes(req.user.role)) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  try {
    const result = await query(
      `SELECT slug, title, card_title, doc_type, sort_order, description, action_label, 
              is_feature_card, card_order, hidden, md, html, toc, updated_at
       FROM ${DOCS_TABLE}
       WHERE is_feature_card = true
       ORDER BY card_order ASC, sort_order ASC`
    );

    res.json(result.rows);
  } catch (err) {
    console.error('Error fetching feature cards:', err.message || err);
    res.status(500).json({ error: 'Failed to fetch feature cards', details: err.message });
  }
}

async function list(req, res) {
  if (![USER_ROLES.ADMIN, USER_ROLES.RESEARCHER].includes(req.user.role)) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  try {
    const result = await query(
      `SELECT slug, title, card_title, doc_type, sort_order, toc, description, action_label, hidden,
       is_feature_card, card_order, show_in_docs, updated_at
      FROM ${DOCS_TABLE}
      ORDER BY 
        CASE doc_type
          WHEN '' THEN 1
          WHEN 'guide' THEN 2
          WHEN 'tutorial' THEN 3
          WHEN 'reference' THEN 4
          WHEN 'faq' THEN 5
          ELSE 5
        END,
        sort_order ASC`
    );

    if (req.user.role === USER_ROLES.RESEARCHER) {
      //filter out docs with slugs starting with a different city and -datasets
      const cityPrefix = req.user.city_id ? `${req.user.city_id}-` : '';

      // Exclude only if it ends with '-datasets' AND does NOT start with current city prefix
      result.rows = result.rows.filter(doc => !(doc.slug.endsWith('-datasets') && !doc.slug.startsWith(cityPrefix)));
    }

    res.json(result.rows);
  } catch (err) {
    res.status(500).json({ error: 'Failed to fetch document' });
  }
}

async function get(req, res) {
  if (![USER_ROLES.ADMIN, USER_ROLES.RESEARCHER].includes(req.user.role)) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  try {
    const slug = req.params.slug;

    const result = await query(
      `SELECT *
       FROM ${DOCS_TABLE}
       WHERE slug = $1`,
      [slug]
    );

    res.json(result.rows[0]);
  } catch (err) {
    res.status(500).json({ error: 'Failed to fetch document' });
  }
}

// PUT /docs/:slug  → update a doc by slug (admin/researcher only)
async function put(req, res) {
  if (![USER_ROLES.ADMIN, USER_ROLES.RESEARCHER].includes(req.user.role)) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  // baseSlug is what comes from the URL params (the current slug)
  const urlSlug = req.params.slug;

  const {
    id, // 🎯 You MUST pass the internal 'id' from the frontend
    title = null,
    slug: storySlug = null, // The new slug from the form
    card_title = null,
    doc_type = '',
    description = null,
    action_label = 'Open',
    card_order = 0,
    sort_order = 0,
    hidden = false,
    is_feature_card = true,
    show_in_docs = true,
    md = '',
  } = req.body;

  const { html, toc } = renderDoc(md);
  const tocJson = JSON.stringify(toc);

  try {
    // 1. Determine the final slug and handle uniqueness
    let finalSlug = (storySlug || urlSlug || crypto.randomUUID()).trim();

    let isUnique = false;
    let counter = 0;
    while (!isUnique) {
      let currentTry = counter === 0 ? finalSlug : `${finalSlug}-${counter}`;

      // Ensure this slug isn't taken by a DIFFERENT record
      const checkResult = await query(
        `SELECT id FROM ${DOCS_TABLE} WHERE slug = $1 AND ($2::uuid IS NULL OR id != $2)`,
        [currentTry, id || null]
      );

      if (checkResult.rows.length === 0) {
        finalSlug = currentTry;
        isUnique = true;
      } else {
        counter++;
      }
    }

    // 2. Perform Upsert by ID (not Slug)
    // 🎯 Use 'ON CONFLICT (id)' to allow the slug itself to change
    const queryText = `
      INSERT INTO ${DOCS_TABLE} (
        id, slug, title, card_title, card_order, doc_type, description, 
        action_label, sort_order, hidden, is_feature_card, show_in_docs, 
        md, html, toc, updated_at
      )
      VALUES (COALESCE($1, gen_random_uuid()), $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, NOW())
      ON CONFLICT (id) 
      DO UPDATE SET
        slug = EXCLUDED.slug,
        title = EXCLUDED.title,
        card_title = EXCLUDED.card_title,
        card_order = EXCLUDED.card_order,
        doc_type = EXCLUDED.doc_type,
        description = EXCLUDED.description,
        action_label = EXCLUDED.action_label,
        sort_order = EXCLUDED.sort_order,
        hidden = EXCLUDED.hidden,
        is_feature_card = EXCLUDED.is_feature_card,
        show_in_docs = EXCLUDED.show_in_docs,
        md = EXCLUDED.md,
        html = EXCLUDED.html,
        toc = EXCLUDED.toc,
        updated_at = NOW()
      RETURNING *;
    `;

    const result = await query(queryText, [
      id || null, finalSlug, title, card_title, card_order, doc_type, description,
      action_label, sort_order, hidden, is_feature_card, show_in_docs,
      md, html, tocJson
    ]);

    res.json(result.rows[0]);
  } catch (err) {
    console.error('Error saving doc:', err);
    res.status(500).json({ error: 'Failed to save document' });
  }
}

// DELETE /docs/:slug  → delete a doc by slug (admin ONLY)
async function del(req, res) {
  if (req.user.role !== USER_ROLES.ADMIN) { // Restricted to Admin
    return res.status(403).json({ error: 'Unauthorized' });
  }

  const slug = req.params.slug;

  try {
    const result = await query(
      `DELETE FROM ${DOCS_TABLE}
       WHERE slug = $1
       RETURNING slug`,
      [slug]
    );

    if (result.rowCount === 0) {
      return res.status(404).json({ error: 'Document not found' });
    }

    res.json({ message: `Document with slug '${slug}' deleted successfully` });
  } catch (err) {
    console.error('Error deleting doc:', err);
    res.status(500).json({ error: 'Failed to delete document' });
  }
}
/** Admin-only: zipped Markdown export (paths follow documentation-configuration layout). */
async function exportMarkdownZip(req, res) {
  if (req.user.role !== USER_ROLES.ADMIN) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  try {
    const apiRoot = path.join(__dirname, '..');
    const cfg = (await loadDocumentationConfig(apiRoot)) || {};
    const layout = cfg.layout === 'flat' ? 'flat' : 'nested';

    const includeHidden =
      req.query.includeHidden === 'true' ||
      req.query.includeHidden === '1' ||
      req.query.include_hidden === 'true';

    let sql = `SELECT slug, title, description, doc_type, md FROM ${DOCS_TABLE}`;
    if (!includeHidden) {
      sql += ' WHERE COALESCE(hidden, false) = false';
    }
    sql += ' ORDER BY slug';

    const result = await query(sql);

    await pipeMarkdownZipToResponse(res, result.rows, { layout });
  } catch (err) {
    console.error('exportMarkdownZip', err);
    if (!res.headersSent) {
      res.status(500).json({ error: 'Failed to export documentation' });
    }
  }
}

async function search(req, res) {
  if (![USER_ROLES.ADMIN, USER_ROLES.RESEARCHER].includes(req.user.role)) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  const term = req.query.term?.trim() || '';
  if (!term) return res.json([]);

  // Transform 'Vega' into 'Vega:*' to allow prefix matching (Veg -> Vega)
  const tsQueryTerm = term.split(/\s+/).filter(t => t).map(t => `${t}:*`).join(' & ');

  try {
    const result = await query(
      `SELECT 
        slug, title, doc_type, updated_at,
        -- Use COALESCE to prevent NULL values from breaking the concatenation
        ts_rank_cd(
          setweight(to_tsvector('english', COALESCE(title, '')), 'A') || 
          setweight(to_tsvector('english', COALESCE(md, '')), 'D'), 
          to_tsquery('english', $1)
        ) AS rank,
        ts_headline('english', COALESCE(md, ''), to_tsquery('english', $1), 
          'StartSel=<b>, StopSel=</b>, MaxWords=40, MinWords=20, MaxFragments=1'
        ) AS snippet
       FROM ${DOCS_TABLE}
       WHERE 
         (to_tsvector('english', COALESCE(title, '')) || to_tsvector('english', COALESCE(md, ''))) @@ to_tsquery('english', $1)
         AND hidden = false
       ORDER BY rank DESC, updated_at DESC
       LIMIT 50`,
      [tsQueryTerm] // Using the transformed term for better matching
    );

    res.json(result.rows);
  } catch (err) {
    console.error('Search error:', err);
    res.status(500).json({ error: 'Failed to search documents' });
  }
}

export default router;