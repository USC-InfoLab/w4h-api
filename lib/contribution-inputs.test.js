import test, { beforeEach, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import http from 'node:http';
import { fileInputFromYaml, fileInputForSlug } from './contribution-inputs.js';
import { loadDelegateModules, _resetDelegateCache } from './contribution-delegate.js';

const EMBRACE_SLUG = 'w4h-embraceplus-csv-file-importer';
const EMBRACE_SIBLING = new URL(`../../${EMBRACE_SLUG}/contribution.yaml`, import.meta.url);
// Copy of the sibling's `inputs` block, for checkouts without the sibling.
const EMBRACE_YAML = `slug: ${EMBRACE_SLUG}
inputs:
  - key: dataset_id
    type: dataset
  - key: files
    type: files
    required: false
    summary: Empatica EmbracePlus Cloud dump.
    drop_label: Drop zip files or a folder
    accept:
      - csv
      - zip
    include: aggregated_per_minute/
    detect:
      - participant_data/
      - aggregated_per_minute/
`;
const embraceYaml = () => (fs.existsSync(EMBRACE_SIBLING) ? fs.readFileSync(EMBRACE_SIBLING, 'utf8') : EMBRACE_YAML);

const GLUCOSE_YAML = `slug: w4h-glucose-monitor-csv-file-importer
inputs:
  - name: files
    type: files
    required: false
`;

const savedUrl = process.env.CONTRIBUTION_RUNNER_URL;
let server;

beforeEach(() => {
  delete process.env.CONTRIBUTION_RUNNER_URL;
  _resetDelegateCache();
});

afterEach(async () => {
  if (savedUrl === undefined) delete process.env.CONTRIBUTION_RUNNER_URL;
  else process.env.CONTRIBUTION_RUNNER_URL = savedUrl;
  _resetDelegateCache();
  if (server) {
    const s = server;
    server = null;
    await new Promise((r) => { s.close(() => r()); s.closeAllConnections(); });
  }
});

test('fileInputFromYaml reads accept, include and detect from the embraceplus yaml', () => {
  const input = fileInputFromYaml(embraceYaml());
  assert.deepEqual(input.accept, ['csv', 'zip']);
  assert.deepEqual(input.include, ['aggregated_per_minute/']);
  assert.deepEqual(input.detect, ['participant_data/', 'aggregated_per_minute/']);
});

test('fileInputFromYaml: a files input with no accept still gives an input', () => {
  const input = fileInputFromYaml(GLUCOSE_YAML);
  assert.ok(input);
  // acceptList's existing default; the routes fall back to csv as well.
  assert.deepEqual(input.accept, ['csv']);
  assert.deepEqual(input.include, []);
});

test('fileInputFromYaml: no files input, bad yaml or no text give null', () => {
  assert.equal(fileInputFromYaml('inputs: []'), null);
  assert.equal(fileInputFromYaml(':::'), null);
  assert.equal(fileInputFromYaml(null), null);
});

test('delegate mode: file inputs come from the runner /modules, not the disk', async () => {
  server = http.createServer((req, res) => {
    res.setHeader('content-type', 'application/json');
    res.end(JSON.stringify([
      { slug: EMBRACE_SLUG, version: '1.0.0', contribution_yaml: embraceYaml() },
      { slug: 'w4h-sample-fitbit-csv-importer', version: '1.0.0', contribution_yaml: null },
    ]));
  });
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  process.env.CONTRIBUTION_RUNNER_URL = `http://127.0.0.1:${server.address().port}`;

  await loadDelegateModules();

  assert.deepEqual(fileInputForSlug(EMBRACE_SLUG).accept, ['csv', 'zip']);
  assert.equal(fileInputForSlug('w4h-sample-fitbit-csv-importer'), null);
  assert.equal(fileInputForSlug('not-on-the-runner'), null);
  assert.equal(fileInputForSlug('../etc'), null);
});
