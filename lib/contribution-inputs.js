/**
 * File-importer dialog copy lives on the `files` input in the sibling
 * repo's contribution.yaml. The Select importer dialog renders that;
 * it does not hard-code device instructions.
 */
import fs from 'fs';
import path from 'path';
import { parse as parseYaml } from 'yaml';
import { delegateUrl, delegateModule } from './contribution-delegate.js';

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

/** The `files` input of a contribution.yaml text, shaped for the app; null when there is none. */
export function fileInputFromYaml(yamlText) {
  if (!yamlText) return null;
  let doc;
  try {
    doc = parseYaml(yamlText);
  } catch (e) {
    console.warn(`[contribution-inputs] could not parse contribution.yaml: ${e.message}`);
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

/**
 * Delegate mode (CONTRIBUTION_RUNNER_URL, Docker): the yaml comes from the
 * runner's /modules, cached by loadDelegateModules (the router loads it before
 * each request). Otherwise it is read from the sibling checkout.
 */
export function fileInputForSlug(slug) {
  const key = String(slug || '').trim();
  if (!SLUG_RE.test(key)) return null;
  if (delegateUrl()) return fileInputFromYaml(delegateModule(key)?.contribution_yaml ?? null);
  const root = path.resolve(SIBLING_REPOS_ROOT);
  const repo = path.resolve(root, key);
  if (repo !== root && !repo.startsWith(root + path.sep)) return null;
  const yamlPath = path.join(repo, 'contribution.yaml');
  if (!fs.existsSync(yamlPath)) return null;
  try {
    return fileInputFromYaml(fs.readFileSync(yamlPath, 'utf8'));
  } catch (e) {
    console.warn(`[contribution-inputs] could not read ${yamlPath}: ${e.message}`);
    return null;
  }
}

export function acceptsFileUpload(slug) {
  return Boolean(fileInputForSlug(slug));
}
