import { query } from '../db.js';
import { computeMetsSummary } from './dataset-mets-summary.js';
import { computeWearSummary, mergeSummaryKeys } from './dataset-wear-summary.js';

/**
 * Recompute the cached cohort summaries (`mets_summary`, `wear_compliance`) of a dataset and
 * store them in `datasets.summary` (ADR-019). Shared by `POST /datasets/:id/summary/refresh` and
 * by the in-app run trigger, so a successful import or SQL job leaves the Analytics cards current.
 *
 * `wearProtocol` overrides the protocol; otherwise the one already stored is kept, so a refresh
 * never silently resets a team's chosen thresholds.
 */
export async function refreshDatasetSummaryData(datasetId, { wearProtocol = null } = {}) {
  const current = await query(`SELECT summary FROM w4h.datasets WHERE id = $1`, [datasetId]);
  if (!current.rowCount) return { ok: false, status: 404, error: 'Dataset not found' };

  const existing = current.rows[0].summary;
  const protocol = wearProtocol || existing?.wear_compliance?.protocol || null;

  const [mets, wear] = await Promise.all([
    computeMetsSummary(datasetId),
    computeWearSummary(datasetId, protocol),
  ]);
  if (wear?.invalid) return { ok: false, status: 400, error: wear.reason };

  const next = mergeSummaryKeys(existing, { mets_summary: mets, wear_compliance: wear });
  const updated = await query(
    `UPDATE w4h.datasets
     SET summary = $2::jsonb, summary_updated_at = NOW()
     WHERE id = $1
     RETURNING summary, summary_updated_at`,
    [datasetId, JSON.stringify(next)]
  );
  return { ok: true, mets, wear, summary: updated.rows[0].summary, summary_updated_at: updated.rows[0].summary_updated_at };
}
