import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const API_ROOT = path.join(__dirname, '..');

/** Runtime `.env` path — override with `W4H_ENV_FILE` (Docker persists `/app/data/.env`). */
export function envFilePath() {
  const override = String(process.env.W4H_ENV_FILE || '').trim();
  if (override) return path.resolve(override);
  return path.join(API_ROOT, '.env');
}

export function envExamplePath() {
  return path.join(API_ROOT, '.env.example');
}
