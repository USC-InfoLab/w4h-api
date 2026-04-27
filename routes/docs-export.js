import archiver from 'archiver';
import { stringify as stringifyYaml } from 'yaml';

import { splitFrontmatter } from '../lib/docs-repo-import.js';
import { USER_ROLES } from '../lib/utils/user.js';
import { query } from '../lib/utils/db.js';

const DOCS_TABLE = 'w4h.docs';

/**
 * Reverse applyAppUrlTemplate (docs-repo-import): put {{APP_URL}} back for repo-friendly exports.
 * Tries APP_URL and VITE_APP_URL plus slash/no-slash variants.
 */
function restoreAppUrlPlaceholder(text) {
  if (text == null || text === '') return text;
  let s = String(text);
  const bases = [];
  for (const v of [process.env.APP_URL, process.env.VITE_APP_URL]) {
    const t = typeof v === 'string' ? v.trim() : '';
    if (!t) continue;
    bases.push(t);
    bases.push(t.endsWith('/') ? t.slice(0, -1) : `${t}/`);
  }
  const unique = [...new Set(bases)].sort((a, b) => b.length - a.length);
  for (const base of unique) {
    const escaped = base.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const re = new RegExp(
      `${escaped}(?=/|\\?|#|\\)|\\s|"|'|[,;.:!?]|$)`,
      'g'
    );
    s = s.replace(re, '{{APP_URL}}');
  }
  return s;
}

/**
 * Root-relative /docs/... links (e.g. after import with empty APP_URL) → {{APP_URL}}/docs/...
 * Matches HTML href= and Markdown () links used in w4h-documentation/content.
 */
function restoreRootRelativeAppDocLinks(text) {
  let s = String(text);
  s = s.replace(/href\s*=\s*"\/docs\//gi, 'href="{{APP_URL}}/docs/');
  s = s.replace(/href\s*=\s*"\/docs"/gi, 'href="{{APP_URL}}/docs"');
  s = s.replace(/href\s*=\s*'\/docs\//gi, "href='{{APP_URL}}/docs/");
  s = s.replace(/href\s*=\s*'\/docs'/gi, "href='{{APP_URL}}/docs'");
  s = s.replace(/\]\(\s*\/docs\//g, ']({{APP_URL}}/docs/');
  s = s.replace(/\]\(\s*\/docs#/g, ']({{APP_URL}}/docs#');
  s = s.replace(/\]\(\s*\/docs\?/g, ']({{APP_URL}}/docs?');
  s = s.replace(/\]\(\s*\/docs\)/g, ']({{APP_URL}}/docs)');
  return s;
}

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
  let out = `---\n${fmYaml}\n---\n${body}`;
  out = restoreRootRelativeAppDocLinks(out);
  out = restoreAppUrlPlaceholder(out);
  return out;
}

/** Admin only: stream docs bundle zip (markdown + root documentation-configuration.yml). */
export async function handleMarkdownZipExport(req, res) {
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
