import dotenv from 'dotenv';
import fs from 'fs-extra';
import path from 'path';
import { fileURLToPath } from 'url';
import { getPool, withTransaction } from '../db.js';

dotenv.config();

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const sqlPath = path.join(__dirname, '../data/platform_roles.sql');

try {
  const sql = await fs.readFile(sqlPath, 'utf8');
  const appRole = process.env.W4H_APP_DB_ROLE || process.env.DB_USER;
  await withTransaction(async (client) => {
    await client.query(`SELECT set_config('w4h.app_role', $1, true)`, [appRole]);
    await client.query(sql);
  });
  console.log(`w4h_platform role bootstrapped for application role ${appRole}`);
} catch (error) {
  console.error(error.message || error);
  process.exitCode = 1;
} finally {
  await getPool()?.end();
}
