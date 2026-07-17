import path from 'path';
import * as url from 'url';
import { join } from 'path';
const __dirname = url.fileURLToPath(new URL('.', import.meta.url));

import dotenv from 'dotenv'
dotenv.config({ path: join(__dirname, '.env') });

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

const app = express();
app.use(cors());

app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ limit: '10mb', extended: true }));

app.use(express.static(path.join(__dirname, 'public')));

app.on('uncaughtException', err => {
  console.log('uncaught exception: ', err);
  process.send({ finish: false });
});

app.listen(process.env.PORT, () => {
  console.log('App running at: http://localhost:' + process.env.PORT);
});

app.use('/users', users);
app.use('/datasets', datasets);
app.use('/docs', docs);
app.use('/config', config);
app.use('/platform', platform);
app.use('/projects', projects);
app.use('/teams', teams);

app.get('/', async function (req, res) {
  const { stdout } = await execa`git rev-parse --short HEAD`;
  res.send(`W4H API ${stdout.trim()}`);
});
