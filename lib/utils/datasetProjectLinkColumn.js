import { query } from '../../db.js';

/**
 * Which column on w4h.datasets points at a catalog project (new name: project_id,
 * legacy name: study_id). Resolved once per process.
 */
let cache = null;

export async function resolveDatasetProjectLinkColumn() {
  if (cache) return cache;
  const r = await query(
    `SELECT column_name FROM information_schema.columns
     WHERE table_schema = 'w4h' AND table_name = 'datasets'
       AND column_name IN ('project_id', 'study_id')`
  );
  const names = new Set(r.rows.map((row) => row.column_name));
  if (names.has('project_id')) cache = 'project_id';
  else if (names.has('study_id')) cache = 'study_id';
  else {
    throw new Error('w4h.datasets has neither project_id nor study_id');
  }
  return cache;
}
