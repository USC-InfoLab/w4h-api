import fs from 'fs';

/**
 * Built-in installs (ADR-031): share the person's API key with the notebooks container,
 * so the importers find W4H_API_KEY without a copy/paste. The kit's builtin overlay sets
 * W4H_NOTEBOOK_KEY_FILE and mounts its folder read-only into notebooks.
 *
 * The first non-Super-admin key written claims the file; keys of other users never
 * overwrite it, and that user's rotations replace it. Best effort: a failure only means
 * the person pastes the key from Profile instead.
 */
export function shareKeyWithNotebooks({ userId, role, apiKey }, file = process.env.W4H_NOTEBOOK_KEY_FILE) {
  if (!file || role === 'super_admin') return false;
  try {
    let owner = null;
    try {
      owner = JSON.parse(fs.readFileSync(file, 'utf8')).user_id;
    } catch (e) {
      if (e.code !== 'ENOENT') throw e;
    }
    if (owner != null && String(owner) !== String(userId)) return false;
    // 644: the notebooks container reads it as another uid; ./data (700) keeps other host users out.
    // Write + rename so a starting kernel never reads half a file.
    const tmp = `${file}.tmp`;
    fs.writeFileSync(tmp, `${JSON.stringify({ user_id: userId, api_key: apiKey })}\n`, { mode: 0o644 });
    fs.chmodSync(tmp, 0o644); // `mode` is reduced by the umask
    fs.renameSync(tmp, file);
    return true;
  } catch (e) {
    console.warn(`[api-keys] could not share the API key with notebooks (${file}): ${e.code || e.message}`);
    return false;
  }
}

/** Revoked key: remove the shared copy if it is this user's, so the next key can claim the file. */
export function forgetNotebookKey(userId, file = process.env.W4H_NOTEBOOK_KEY_FILE) {
  if (!file) return false;
  try {
    if (String(JSON.parse(fs.readFileSync(file, 'utf8')).user_id) !== String(userId)) return false;
    fs.rmSync(file);
    return true;
  } catch (e) {
    if (e.code !== 'ENOENT') {
      console.warn(`[api-keys] could not remove the shared notebooks API key (${file}): ${e.code || e.message}`);
    }
    return false;
  }
}
