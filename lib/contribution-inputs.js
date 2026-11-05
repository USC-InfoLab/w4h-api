/**
 * File-importer dialog copy lives on the `files` input in the sibling
 * repo's contribution.yaml. The Select importer dialog renders that;
 * it does not hard-code device instructions.
 */
import fs from 'fs';
import path from 'path';
import { parse as parseYaml } from 'yaml';

const SIBLING_REPOS_ROOT = process.env.CONTRIBUTION_RUNNER_REPO_ROOT
  || path.resolve(process.cwd(), '..');

const SLUG_RE = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

function text(value, max) {
  const s = String(value ?? '').replace(/\s+/g, ' ').trim();
  return s ? s.slice(0, max) : '';
}

const ACCEPT_RE = /^(csv|zip)$/;

function acceptList(value) {
  const list = Array.isArray(value) ? value : (value == null || value === '' ? [] : [value]);
  const out = [];
  for (const item of list) {
    const s = String(item || '').trim().toLowerCase().replace(/^\./, '');
    if (ACCEPT_RE.test(s) && !out.includes(s)) out.push(s);
  }
  return out.length ? out : ['csv'];
}
function hints(value) {
  const list = Array.isArray(value) ? value : (value == null || value === '' ? [] : [value]);
  const out = [];
  for (const item of list) {
    const s = String(item || '').trim().replace(/\\/g, '/');
    if (!s || s.length > 80 || s.includes('..') || s.startsWith('/') || /[\0\n]/.test(s)) continue;
    out.push(s);
  }
  return out;
}

export function fileInputForSlug(slug) {
  const key = String(slug || '').trim();
  if (!SLUG_RE.test(key)) return null;
  const root = path.resolve(SIBLING_REPOS_ROOT);
  const repo = path.resolve(root, key);
  if (repo !== root && !repo.startsWith(root + path.sep)) return null;
  const yamlPath = path.join(repo, 'contribution.yaml');
  if (!fs.existsSync(yamlPath)) return null;
  let doc;
  try {
    doc = parseYaml(fs.readFileSync(yamlPath, 'utf8'));
  } catch (e) {
    console.warn(`[contribution-inputs] could not read ${yamlPath}: ${e.message}`);
    return null;
  }
  const inputs = Array.isArray(doc?.inputs) ? doc.inputs : [];
  const input = inputs.find((item) => item && item.type === 'files');
  if (!input) return null;
  return {
    summary: text(input.summary, 400),
    note: text(input.note, 500),
    drop_label: text(input.drop_label, 120),
    files_button: text(input.files_button, 80),
    folder_button: text(input.folder_button, 80),
    ignored: text(input.ignored, 200),
    empty: text(input.empty, 300),
    include: hints(input.include),
    detect: hints(input.detect),
    accept: acceptList(input.accept),
  };
}

export function acceptsFileUpload(slug) {
  return Boolean(fileInputForSlug(slug));
}
