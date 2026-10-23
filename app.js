import path from 'path';
import * as url from 'url';
import { join } from 'path';
import { readFileSync } from 'fs';
const __dirname = url.fileURLToPath(new URL('.', import.meta.url));
const pkg = JSON.parse(readFileSync(join(__dirname, 'package.json'), 'utf8'));

import dotenv from 'dotenv'
import { envFilePath } from './lib/env-path.js';
dotenv.config({ path: envFilePath() });

if (process.env.W4H_DEBUG) {
  console.log('DB_HOST:', process.env.DB_HOST);
}

import { execa } from 'execa';
import express from 'express';
import cors from 'cors';

import users from './routes/users.js';
import datasets from './routes/datasets.js';
import docs from './routes/docs.js';
import config from './routes/config.js';
import platform from './routes/platform.js';
import projects from './routes/projects.js';
import teams from './routes/teams.js';
import contributions from './routes/contributions.js';
import ai from './routes/ai.js';
import accessRequests from './routes/access-requests.js';
import { isConfigured } from './db.js';
import { ensureSchema } from './lib/db-init.js';
import { attachApiKeyUsageLogger } from './lib/utils/usage.js';
import { urlencodedBodyParser } from './lib/urlencoded-body-parser.js';

const app = express();
app.use(cors());

app.use(express.json({ limit: '10mb' }));
app.use(urlencodedBodyParser);
app.use(attachApiKeyUsageLogger);

app.use(express.static(path.join(__dirname, 'public')));

// Docs repo media (screenshots, GIFs for What's New, etc.): DOCS_REPO_PATH/media → /docs-media
{
  const docsRepo = process.env.DOCS_REPO_PATH
    ? path.resolve(__dirname, process.env.DOCS_REPO_PATH)
    : null;
  if (docsRepo) {
    app.use(
      '/docs-media',
      express.static(path.join(docsRepo, 'media'), {
        fallthrough: true,
        maxAge: process.env.NODE_ENV === 'production' ? '1d' : 0,
      })
    );
  }
}

app.on('uncaughtException', err => {
  console.log('uncaught exception: ', err);
  process.send({ finish: false });
});

app.listen(process.env.PORT, async () => {
  const releaseLabel = [pkg.releaseName, pkg.version].filter(Boolean).join(' ');
  console.log(`W4H API ${releaseLabel} running at: http://localhost:${process.env.PORT}`);
  if (isConfigured()) {
    try {
      await ensureSchema();
      console.log('✓ Database schema ensured');
    } catch (err) {
      console.warn('⚠ Schema ensure skipped/failed:', err.message);
    }
  }
});

app.use('/users', users);
app.use('/datasets', datasets);
app.use('/docs', docs);
app.use('/config', config);
app.use('/platform', platform);
app.use('/projects', projects);
app.use('/teams', teams);
app.use('/contributions', contributions);
app.use('/packages', contributions);
app.use('/ai', ai);
app.use('/access-requests', accessRequests);

app.get('/', async function (req, res) {
  const baked = String(process.env.GIT_COMMIT || '').trim();
  if (baked) {
    return res.send(`W4H API ${baked}`);
  }
  try {
    const { stdout } = await execa('git', ['rev-parse', '--short', 'HEAD']);
    res.send(`W4H API ${stdout.trim()}`);
  } catch {
    res.send('W4H API unknown');
  }
});
