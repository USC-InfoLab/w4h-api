/**
 * Chart kinds the app renders, and the contribution display contract (ADR-029 §4).
 * Kept free of imports so SQL contributions and the query layer can both use it.
 */

export const CHART_KINDS = ['line', 'bar', 'box', 'heatmap', 'scatter', 'events', 'table'];

/**
 * Contribution display contract (ADR-029 §4): a manifest may say how its output is shown, from
 * the fixed chart set only. Anything else is dropped — contributions never ship UI code.
 */
export function sanitizeDisplay(raw) {
  const list = Array.isArray(raw) ? raw : (raw ? [raw] : []);
  const ident = (v) => (typeof v === 'string' && /^[A-Za-z_][A-Za-z0-9_ .-]{0,62}$/.test(v) ? v : null);
  const out = [];
  for (const item of list.slice(0, 6)) {
    if (!item || typeof item !== 'object') continue;
    const chart = String(item.chart || item.type || '').toLowerCase();
    if (!CHART_KINDS.includes(chart)) continue;
    out.push({
      chart,
      title: typeof item.title === 'string' ? item.title.slice(0, 120) : null,
      x: ident(item.x),
      y: ident(item.y),
      group: ident(item.group),
    });
  }
  return out;
}
