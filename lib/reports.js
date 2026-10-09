import crypto from 'crypto';

/**
 * Published notebook reports (ADR-029 §5): a notebook's interactive HTML kept as a snapshot and
 * shown in the app inside a sandboxed iframe. The HTML is unvetted, so it is shown with
 * `sandbox="allow-scripts"` (no same-origin) AND this CSP, which blocks every network request.
 */
export const REPORT_MAX_BYTES = Number(process.env.REPORT_MAX_BYTES) || 8 * 1024 * 1024;

export const REPORT_CSP = [
  "default-src 'none'",
  "script-src 'unsafe-inline' 'unsafe-eval'",
  "style-src 'unsafe-inline'",
  'img-src data: blob:',
  'font-src data:',
  'media-src data: blob:',
  'worker-src blob:',
  "connect-src 'none'",
  "form-action 'none'",
  "base-uri 'none'",
].join('; ');

const META = `<meta http-equiv="Content-Security-Policy" content="${REPORT_CSP}">`;

/** Put the CSP first (after any doctype) so no script can run before it applies. */
export function injectReportCsp(html) {
  const text = String(html || '');
  const doctype = text.match(/^\s*<!doctype[^>]*>/i);
  if (doctype) return doctype[0] + META + text.slice(doctype[0].length);
  return META + text;
}

export function sha256(text) {
  return crypto.createHash('sha256').update(text).digest('hex');
}

export function looksLikeHtml(text) {
  return /<\s*(html|body|div|script|svg|head|!doctype)/i.test(text);
}
