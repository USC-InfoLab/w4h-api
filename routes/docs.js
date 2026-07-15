import express from 'express';
import archiver from 'archiver';

const router = express.Router();

import { DOCS_TABLE, DOC_TYPES_TABLE, query } from '../db.js';
import { authenticate } from '../auth.js';
import { USER_ROLES } from '../lib/utils/user.js';
import { markdownPathFromDocType, markdownFileForExport, splitFrontmatter } from '../lib/utils/docs.js';
import { renderDoc, stripLeadingTitleHeading } from '../lib/utils/md-render.js';

import crypto from 'crypto';


// Docs routes
router.get('/', authenticate, list);
router.get('/search', authenticate, search);
router.get('/cards/feature', authenticate, featureCards);
router.get('/export', authenticate, exportDocs);
router.post('/refresh', authenticate, refreshDocs);
router.get('/types', authenticate, listDocTypes);
router.post('/types', authenticate, createDocType);
router.get('/:slug', authenticate, get);
router.put('/:slug', authenticate, put);
router.delete('/:slug', authenticate, del);

// Admin-only: refresh docs from source (overwrites existing docs with same slug)
async function refreshDocs(req, res) {
  if (req.user.role !== USER_ROLES.ADMIN) {
    return res.status(403).json({ error: 'Unauthorized' });
  }
  try {
    const { importDocs } = await import('../lib/utils/docs.js');
    const result = await importDocs();
    if (result.imported) {
      res.json({ success: true, count: result.count });
    } else {
      res.status(400).json({ success: false, reason: result.reason });
    }
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
}

async function listDocTypes(req, res) {
  if (![USER_ROLES.ADMIN, USER_ROLES.RESEARCHER].includes(req.user.role)) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  const DOC_TYPE_LABELS = {
    guide: 'Guides',
    tutorial: 'Tutorials',
    reference: 'Reference',
    faq: 'FAQs',
    internal: 'Internal',
    other: 'Other',
    dev: 'Developer',
  };

  function labelFor(slug) {
    return DOC_TYPE_LABELS[slug] || slug.charAt(0).toUpperCase() + slug.slice(1);
  }

  const bySlug = new Map();

  try {
    const result = await query(
      `SELECT slug, label, sort_order
       FROM ${DOC_TYPES_TABLE}
       ORDER BY sort_order ASC, slug ASC`
    );
    for (const row of result.rows) {
      bySlug.set(row.slug, row);
    }
  } catch (err) {
    console.warn('doc_types table unavailable:', err.message);
  }

  // Always include types actually in use (folder names from imported docs)
  const distinct = await query(
    `SELECT DISTINCT doc_type AS slug
     FROM ${DOCS_TABLE}
     WHERE doc_type IS NOT NULL AND doc_type <> ''
     ORDER BY slug ASC`
  );
  for (const row of distinct.rows) {
    if (!bySlug.has(row.slug)) {
      bySlug.set(row.slug, { slug: row.slug, label: labelFor(row.slug), sort_order: 999 });
    }
  }

  const types = [...bySlug.values()].sort((a, b) => {
    const order = (slug) => {
      const ranks = { guide: 2, tutorial: 3, reference: 4, faq: 5, internal: 6, other: 7, dev: 8 };
      return ranks[slug] ?? 99;
    };
    return order(a.slug) - order(b.slug) || a.slug.localeCompare(b.slug);
  });

  res.json(types);
}

async function createDocType(req, res) {
  if (req.user.role !== USER_ROLES.ADMIN) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  const rawSlug = String(req.body?.slug ?? '').trim().toLowerCase();
  const label = String(req.body?.label ?? '').trim();
  const sortOrder = Number.isFinite(+req.body?.sort_order) ? +req.body.sort_order : 0;

  if (!rawSlug) {
    return res.status(400).json({ error: 'Slug is required' });
  }
  if (!/^[a-z][a-z0-9-]*$/.test(rawSlug)) {
    return res.status(400).json({ error: 'Slug must start with a letter and contain only lowercase letters, numbers, and hyphens' });
  }
  if (!label) {
    return res.status(400).json({ error: 'Label is required' });
  }

  try {
    const result = await query(
      `INSERT INTO ${DOC_TYPES_TABLE} (slug, label, sort_order)
       VALUES ($1, $2, $3)
       ON CONFLICT (slug) DO NOTHING
       RETURNING slug, label, sort_order`,
      [rawSlug, label, sortOrder]
    );

    if (result.rows.length === 0) {
      return res.status(409).json({ error: `Doc type '${rawSlug}' already exists` });
    }

    res.status(201).json(result.rows[0]);
  } catch (err) {
    console.error('Error creating doc type:', err);
    res.status(500).json({ error: 'Failed to create doc type' });
  }
}

async function exportDocs(req, res) {
  if (req.user.role !== USER_ROLES.ADMIN) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  try {
    const result = await query(
      `SELECT
        slug, title, description, md, doc_type, card_title, action_label, sort_order,
        hidden, is_feature_card, show_in_docs, card_order, updated_by, updated_at
       FROM ${DOCS_TABLE}
       ORDER BY sort_order ASC NULLS LAST, slug ASC`
    );

    res.setHeader('Content-Type', 'application/zip');
    res.setHeader('Content-Disposition', 'attachment; filename="w4h-docs-bundle.zip"');

    const archive = archiver('zip', { zlib: { level: 9 } });
    archive.on('error', (err) => {
      console.error('Zip archive error:', err);
      if (!res.headersSent) {
        res.status(500).json({ error: 'Failed to build export archive', details: err.message });
      } else {
        res.end();
      }
    });

    archive.pipe(res);

    for (const row of result.rows) {
      const content = markdownFileForExport(row);
      // Use the name field from the exported YAML frontmatter as the zip entry path,
      // falling back to slug + doc_type if not present.
      const { meta } = splitFrontmatter(content);
      const name = typeof meta.name === 'string' && meta.name.trim() ? meta.name.trim() : row.slug;
      const outputPath = markdownPathFromDocType(name, row.doc_type);
      archive.append(content, { name: outputPath });
    }

    await archive.finalize();
  } catch (err) {
    console.error('Export markdown zip error:', err);
    if (!res.headersSent) {
      res.status(500).json({ error: 'Failed to export documentation markdown', details: err.message });
    }
  }
}

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
          ELSE 6
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
    is_feature_card = false,
    show_in_docs = true,
    md = '',
  } = req.body;

  const cleanedMd = stripLeadingTitleHeading(md, title);
  const { html, toc } = renderDoc(cleanedMd);
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
        md, html, toc, updated_by, updated_at
      )
      VALUES (COALESCE($1, gen_random_uuid()), $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, NOW())
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
        updated_by = EXCLUDED.updated_by,
        updated_at = NOW()
      RETURNING *;
    `;

    const result = await query(queryText, [
      id || null, finalSlug, title, card_title, card_order, doc_type, description,
      action_label, sort_order, hidden, is_feature_card, show_in_docs,
      cleanedMd, html, tocJson, req.user.email || null
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