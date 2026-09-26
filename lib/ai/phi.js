const EMAIL_RE = /\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/i;
const SSN_RE = /\b\d{3}-\d{2}-\d{4}\b/;
const MRN_RE = /\b(?:mrn|medical[\s_-]?record)[:\s#]*[A-Z0-9-]{4,}\b/i;
const SUBJECT_ID_RE = /\b(?:subject|participant|patient)[_\s-]?(?:id|identifier)\s*[:=]\s*[A-Z0-9._-]{2,}\b/i;
const UUID_RE = /\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b/i;
const GPS_RE = /\b(-?\d{1,2}\.\d{4,}),\s*(-?\d{1,3}\.\d{4,})\b/;
// Excludes a fractional tail (e.g. "96.80872602824272" from a high-precision
// measurement) — real long-digit identifiers (MRNs, phone numbers) don't
// carry a decimal point, so requiring the run not be preceded by "." avoids
// flagging legitimate numeric data as PHI.
const LONG_DIGIT_RE = /(?<!\.)\b\d{9,}\b/;

const BLOCK_HINTS = [
  /raw (?:row|timeseries|time-series|signal)/i,
  /gps track/i,
  /lat(?:itude)?\s*[:=]/i,
  /lon(?:gitude)?\s*[:=]/i,
];

export function scanPhiText(text) {
  const src = String(text || '');
  if (!src.trim()) return { blocked: false, reasons: [] };
  const reasons = [];
  if (EMAIL_RE.test(src)) reasons.push('email');
  if (SSN_RE.test(src)) reasons.push('ssn');
  if (MRN_RE.test(src)) reasons.push('mrn');
  if (SUBJECT_ID_RE.test(src)) reasons.push('subject_id');
  if (GPS_RE.test(src)) reasons.push('gps');
  if (LONG_DIGIT_RE.test(src)) reasons.push('long_identifier');
  for (const re of BLOCK_HINTS) {
    if (re.test(src)) {
      reasons.push('raw_signal_or_location');
      break;
    }
  }
  return { blocked: reasons.length > 0, reasons };
}

export function scanPhiMessages(messages, { allowUuids = true } = {}) {
  const reasons = new Set();
  for (const msg of messages || []) {
    const hit = scanPhiText(msg?.content);
    for (const r of hit.reasons) {
      if (r === 'long_identifier' && allowUuids && UUID_RE.test(String(msg?.content || ''))) {
        continue;
      }
      reasons.add(r);
    }
  }
  return { blocked: reasons.size > 0, reasons: [...reasons] };
}
