/**
 * SQL identifier quoting (ADR-003). Double-quoting preserves exact case/spaces/punctuation
 * in table and column names, matching Postgres's own identifier rules — the canonical source
 * for quoting any dataset/signal/column identifier spliced into a SQL string.
 */

const MAX_BYTES = 63; // Postgres NAMEDATALEN - 1; silently truncates past this otherwise

function validateIdent(name) {
  const s = String(name ?? '').trim();
  if (!s) {
    throw new Error('SQL identifier cannot be empty');
  }
  if (s.includes('\0')) {
    throw new Error(`Invalid SQL identifier: ${name}`);
  }
  if (Buffer.byteLength(s, 'utf8') > MAX_BYTES) {
    throw new Error(`SQL identifier exceeds ${MAX_BYTES} bytes: ${s}`);
  }
  return s;
}

function quoteIdent(name) {
  return `"${validateIdent(name).replace(/"/g, '""')}"`;
}

export { validateIdent, quoteIdent };
