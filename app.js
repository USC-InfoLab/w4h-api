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
import bodyParser from 'body-parser';

import users from './routes/users.js';
import mailer from './routes/mailer.js';
import datasets from './routes/datasets.js';
import docs from './routes/docs.js';
import config from './routes/config.js';
import platform from './routes/platform.js';
import projects from './routes/projects.js';

const app = express();
app.use(cors());

// Increase JSON body limit to e.g. 10mb
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ limit: '10mb', extended: true }));

// app.use(express.urlencoded({extended: true})); 
// app.use(bodyParser.json()); // support json encoded bodies
// app.use(bodyParser.urlencoded({ extended: true })); // support encoded bodies
app.use(express.static(path.join(__dirname, 'public')));

app.on('uncaughtException', err => {
  console.log('uncaught exception: ', err);
  process.send({ finish: false });
});

app.listen(process.env.PORT, () => {
  console.log('App running at: http://localhost:' + process.env.PORT);
});

///////////////////////////////////////////////////////////////////////

app.use('/users', users);
app.use('/mailer', mailer);
app.use('/datasets', datasets);
app.use('/docs', docs);
app.use('/config', config);
app.use('/platform', platform);
app.use('/projects', projects);

app.get('/', async function (req, res) {
  const { stdout } = await execa`git rev-parse --short HEAD`;
  res.send(`W4H API ${stdout.trim()}`);
});
