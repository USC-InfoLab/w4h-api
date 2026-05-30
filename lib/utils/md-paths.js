import path from 'path';

function folderForDocType(docType) {
  const raw = String(docType || '').trim().toLowerCase();
  if (!raw) return 'root';
  const safe = raw.replace(/[^a-z0-9_-]/g, '-').replace(/-+/g, '-');
  return safe.replace(/^-+|-+$/g, '') || 'root';
}

function markdownRelpath(layout, docType, slug) {
  const safeSlug = String(slug || '').replaceAll('/', '-').trim();
  if (!safeSlug) return null;
  if (layout === 'flat') return `${safeSlug}.md`;
  return path.join(folderForDocType(docType), `${safeSlug}.md`).replace(/\\/g, '/');
}

export { folderForDocType, markdownRelpath };
