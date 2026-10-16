/**
 * File uploads for importer contributions that take user-selected files (e.g.
 * w4h-fitbit-csv-file-importer). The Admin's files are staged in a folder on a volume
 * shared with the `notebooks` runner; the module reads them via W4H_DATA_DIR.
 *
 * Layout: <UPLOADS_ROOT>/<upload id>/  .owner.json  <files>.csv
 * No database rows — ownership lives in .owner.json.
 *
 * Streams each file straight to disk (no multipart parser, no buffering), so the 1M+ row
 * intraday exports are fine.
 */
import crypto from 'crypto';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { Transform } from 'stream';
import { pipeline } from 'stream/promises';
export const UPLOADS_ROOT = process.env.CONTRIBUTION_UPLOADS_ROOT
  || path.join(os.tmpdir(), 'w4h-contribution-uploads');

// Same limits the old in-browser importer enforced.
export const MAX_FILES = 200;
export const MAX_FILE_SIZE_BYTES = 200 * 1024 * 1024;
export const MAX_TOTAL_SIZE_BYTES = 1024 ** 3;
const STALE_AFTER_MS = 24 * 60 * 60 * 1000;

const UPLOAD_ID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;
const SAFE_NAME_RE = /^[A-Za-z0-9][A-Za-z0-9 ._()-]*\.csv$/i;
const OWNER_FILE = '.owner.json';

export function isValidUploadId(id) {
  return typeof id === 'string' && UPLOAD_ID_RE.test(id);
}

export function uploadDir(id) {
  if (!isValidUploadId(id)) throw new Error('Invalid upload id');
  return path.join(UPLOADS_ROOT, id);
}

/** The runner and API run as different users; both need to write into the folder. */
function shareable(target, mode) {
  try {
    fs.chmodSync(target, mode);
  } catch {
    // Not the owner (e.g. the volume root itself) — the image sets the right mode up front.
  }
}

function sweepStale() {
  try {
    for (const name of fs.readdirSync(UPLOADS_ROOT)) {
      if (!isValidUploadId(name)) continue;
      const dir = path.join(UPLOADS_ROOT, name);
      if (Date.now() - fs.statSync(dir).mtimeMs > STALE_AFTER_MS) fs.rmSync(dir, { recursive: true, force: true });
    }
  } catch {
    // best effort
  }
}

export function createUpload({ user, slug, datasetId }) {
  fs.mkdirSync(UPLOADS_ROOT, { recursive: true });
  shareable(UPLOADS_ROOT, 0o770);
  sweepStale();
  const id = crypto.randomUUID();
  const dir = uploadDir(id);
  fs.mkdirSync(dir);
  shareable(dir, 0o770);
  fs.writeFileSync(
    path.join(dir, OWNER_FILE),
    JSON.stringify({ user_id: user.id, slug, dataset_id: datasetId, created_at: new Date().toISOString() })
  );
  return { id, dir };
}

/**
 * Returns { dir, owner } if the upload exists and belongs to this user. No
 * super-admin bypass: these are staged, not-yet-ingested participant CSVs
 * (PHI), and master's ingest path (lib/ingest-csv.js) deliberately dropped
 * its own super-admin bypass for the same reason — stay consistent with it.
 */
export function loadUpload(id, user) {
  if (!isValidUploadId(id)) return { ok: false, status: 400, error: 'Invalid upload id' };
  const dir = uploadDir(id);
  let owner;
  try {
    owner = JSON.parse(fs.readFileSync(path.join(dir, OWNER_FILE), 'utf8'));
  } catch {
    return { ok: false, status: 404, error: 'Upload not found or expired' };
  }
  if (owner.user_id !== user.id) {
    return { ok: false, status: 403, error: 'This upload belongs to another user' };
  }
  return { ok: true, dir, owner };
}

export function listUploadFiles(dir) {
  return fs.readdirSync(dir)
    .filter((f) => SAFE_NAME_RE.test(f))
    .map((name) => ({ name, size: fs.statSync(path.join(dir, name)).size }));
}

export function sanitizeFileName(raw) {
  const name = path.basename(String(raw || '')).trim();
  return SAFE_NAME_RE.test(name) ? name : null;
}

/** Streams `req` into <dir>/<name>, enforcing per-file, total and count limits. */
export async function saveUploadFile(dir, rawName, req) {
  const name = sanitizeFileName(rawName);
  if (!name) return { ok: false, status: 400, error: 'File name must be a plain .csv name (letters, digits, space . _ ( ) -)' };
  const existing = listUploadFiles(dir).filter((f) => f.name !== name);
  if (existing.length >= MAX_FILES) {
    return { ok: false, status: 413, error: `Too many files (limit ${MAX_FILES})` };
  }
  const usedBytes = existing.reduce((sum, f) => sum + f.size, 0);

  let bytes = 0;
  let limitError = null;
  const counter = new Transform({
    transform(chunk, _enc, cb) {
      bytes += chunk.length;
      if (bytes > MAX_FILE_SIZE_BYTES) {
        limitError = `"${name}" exceeds the ${MAX_FILE_SIZE_BYTES / 1024 / 1024}MB per-file limit`;
        return cb(new Error(limitError));
      }
      if (usedBytes + bytes > MAX_TOTAL_SIZE_BYTES) {
        limitError = `Files exceed the ${MAX_TOTAL_SIZE_BYTES / 1024 / 1024}MB combined limit`;
        return cb(new Error(limitError));
      }
      cb(null, chunk);
    },
  });

  const tmp = path.join(dir, `.${name}.part`);
  try {
    await pipeline(req, counter, fs.createWriteStream(tmp, { mode: 0o660 }));
  } catch (e) {
    fs.rmSync(tmp, { force: true });
    return { ok: false, status: limitError ? 413 : 400, error: limitError || `Upload failed: ${e.message}` };
  }
  if (bytes === 0) {
    fs.rmSync(tmp, { force: true });
    return { ok: false, status: 400, error: `"${name}" is empty` };
  }
  fs.renameSync(tmp, path.join(dir, name));
  shareable(path.join(dir, name), 0o660);
  return { ok: true, name, size: bytes };
}

export function deleteUploadFile(dir, rawName) {
  const name = sanitizeFileName(rawName);
  if (!name) return false;
  fs.rmSync(path.join(dir, name), { force: true });
  return true;
}

export function removeUpload(id) {
  if (isValidUploadId(id)) fs.rmSync(uploadDir(id), { recursive: true, force: true });
}
