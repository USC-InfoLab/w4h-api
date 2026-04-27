import express from 'express';
import archiver from 'archiver';
import { stringify as stringifyYaml } from 'yaml';

const router = express.Router();

import { authenticate } from '../auth.js';
import { splitFrontmatter } from '../lib/docs-repo-import.js';
import { USER_ROLES } from '../lib/utils/user.js';
import { query } from '../lib/utils/db.js';

import crypto from 'crypto';

router.get('/', authenticate, list);
router.get('/search', authenticate, search);
router.get('/cards/feature', authenticate, featureCards);
router.get('/export/markdown.zip', authenticate, exportMarkdownZip);
router.get('/:slug', authenticate, get);
router.put('/:slug', authenticate, put);
router.delete('/:slug', authenticate, del);

const DOCS_TABLE = 'w4h.docs';

/** Zip path for exported markdown: content/<doc_type>/<slug>.md or content/<slug>.md when doc_type is empty. */
function markdownPathFromDocType(slug, docType) {
  const safe =
    String(slug ?? 'doc')
      .replace(/[/\\?%*:|"<>]/g, '-')
      .replace(/^\.+/, '') || 'doc';
  const typeDir =
    String(docType ?? '')
      .trim()
      .toLowerCase()
      .replace(/[/\\?%*:|"<>]/g, '-')
      .replace(/^\.+/, '')
      .replace(/\s+/g, '-') || '';
  if (!typeDir) {
    return `content/${safe}.md`;
  }
  return `content/${typeDir}/${safe}.md`;
}

/** Markdown file body matching w4h-documentation/content (YAML frontmatter + body). */
function markdownFileForExport(row) {
  const raw = row.md == null ? '' : String(row.md);
  const { body } = splitFrontmatter(raw);
  const titleFromSlug = String(row.slug ?? '')
    .split(/[-_]/g)
    .filter(Boolean)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
  const title =
    (row.title != null && String(row.title).trim()) || titleFromSlug;
  const description = row.description != null ? String(row.description).trim() : '';
  const fmYaml = stringifyYaml({ title, description }).trimEnd();
  return `---\n${fmYaml}\n---\n${body}`;
}

/** Admin only: stream docs bundle zip (markdown + root documentation-configuration.yml). */
async function exportMarkdownZip(req, res) {
  if (req.user.role !== USER_ROLES.ADMIN) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  try {
    const result = await query(
      `SELECT
        slug, title, description, md, doc_type, card_title, action_label, sort_order,
        hidden, is_feature_card, show_in_docs, card_order
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

    const config = { docs: {} };
    for (const row of result.rows) {
      config.docs[row.slug] = {
        doc_type: row.doc_type != null ? String(row.doc_type) : '',
        card_title: row.card_title != null ? String(row.card_title) : '',
        action_label: row.action_label != null ? String(row.action_label) : 'Open',
        sort_order: Number.isFinite(Number(row.sort_order)) ? Number(row.sort_order) : 0,
        hidden: Boolean(row.hidden),
        is_feature_card: Boolean(row.is_feature_card),
        show_in_docs: row.show_in_docs !== false,
        card_order: Number.isFinite(Number(row.card_order)) ? Number(row.card_order) : 0
      };

      const outputPath = markdownPathFromDocType(row.slug, row.doc_type);
      archive.append(markdownFileForExport(row), { name: outputPath });
    }

    archive.append(stringifyYaml(config), { name: 'documentation-configuration.yml' });

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
    html = '',
    toc = []
  } = req.body;

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