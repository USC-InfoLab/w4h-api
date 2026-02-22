import * as url from 'url';
import { join } from 'path';
const __dirname = url.fileURLToPath(new URL('.', import.meta.url));

import dotenv from 'dotenv'
dotenv.config({ path: join(__dirname, '../.env') });

export {
};
