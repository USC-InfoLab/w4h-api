import archiver from 'archiver';
import yaml from 'js-yaml';
import { markdownRelpath } from './markdownPaths.js';

function dumpFrontmatter(data) {
  return yaml
    .dump(data, {
      defaultFlowStyle: false,
      lineWidth: 120,
      sortKeys: false,
      noRefs: true,
    })
    .trimEnd();
}

/**
 * Stream a zip of Markdown files (nested layout) to an Express response.
 * @param {import('express').Response} res
 * @param {Array<{ slug: string, title: string, description: string, doc_type: string, md: string }>} rows
 * @param {{ layout?: string, filename?: string }} options
 */
function pipeMarkdownZipToResponse(res, rows, options = {}) {
  const layout = options.layout === 'flat' ? 'flat' : 'nested';
  const filename = options.filename || 'w4h-docs-markdown.zip';

  res.setHeader('Content-Type', 'application/zip');
  res.setHeader('Content-Disposition', `attachment; filename="${filename}"`);

  const archive = archiver('zip', { zlib: { level: 9 } });

  return new Promise((resolve, reject) => {
    archive.on('error', reject);
    archive.on('warning', (err) => {
      if (err.code !== 'ENOENT') console.warn('archiver warning', err);
    });

    archive.pipe(res);

    for (const row of rows) {
      const rel = markdownRelpath(layout, row.doc_type, row.slug);
      if (!rel) continue;
      const frontmatter = {
        title: row.title ?? '',
        description: row.description ?? '',
      };
      const body = String(row.md || '').replace(/\s*$/, '\n');
      const text = `---\n${dumpFrontmatter(frontmatter)}\n---\n\n${body}`;
      archive.append(Buffer.from(text, 'utf8'), { name: rel });
    }

    archive
      .finalize()
      .then(() => resolve())
      .catch((err) => {
        if (typeof res.destroy === 'function') {
          try {
            res.destroy(err);
          } catch {
            /* ignore */
          }
        }
        reject(err);
      });
  });
}

export { pipeMarkdownZipToResponse, dumpFrontmatter };
