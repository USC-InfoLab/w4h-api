import { test } from 'node:test';
import assert from 'node:assert/strict';
import express from 'express';
import { routeTemplate, shouldRecordRequest } from './usage.js';

test('routeTemplate joins baseUrl + route path, never the raw path', () => {
  const req = { baseUrl: '/datasets', route: { path: '/:id/import/csv-batch' }, originalUrl: '/datasets/42/import/csv-batch' };
  assert.equal(routeTemplate(req), '/datasets/:id/import/csv-batch');
  assert.equal(routeTemplate({ baseUrl: '/teams', route: { path: '/' } }), '/teams');
  assert.equal(routeTemplate({ originalUrl: '/nope/42' }), null);
});

test('shouldRecordRequest skips polling and telemetry endpoints', () => {
  const r = (method, baseUrl, path) => ({ method, baseUrl, route: { path } });
  assert.equal(shouldRecordRequest(r('GET', '/config', '/status')), false);
  assert.equal(shouldRecordRequest(r('POST', '/config', '/client-usage')), false);
  assert.equal(shouldRecordRequest(r('POST', '/users', '/token-valid')), false);
  assert.equal(shouldRecordRequest(r('POST', '/teams', '/')), true);
  assert.equal(shouldRecordRequest(r('GET', '/config', '/usage')), true);
  assert.equal(shouldRecordRequest({ method: 'GET' }), false, 'unmatched routes are not recorded');
});

test('route template is still available when the response finishes (logger mounted before routers)', async () => {
  const app = express();
  let seen;
  app.use((req, res, next) => {
    res.on('finish', () => { seen = routeTemplate(req); });
    next();
  });
  const datasets = express.Router();
  datasets.post('/:id/import/csv-batch', (req, res) => res.json({ ok: true }));
  app.use('/datasets', datasets);

  const server = app.listen(0);
  try {
    const { port } = server.address();
    await fetch(`http://127.0.0.1:${port}/datasets/42/import/csv-batch`, { method: 'POST' });
    await new Promise((r) => setImmediate(r));
    assert.equal(seen, '/datasets/:id/import/csv-batch');
  } finally {
    server.close();
  }
});
