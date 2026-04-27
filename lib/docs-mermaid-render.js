import fs from 'fs-extra';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const MERMAID_MIN = path.join(__dirname, '../node_modules/mermaid/dist/mermaid.min.js');

/**
 * Mermaid diagram rendering for doc import: headless Chromium + mermaid.min.js replaces
 * `div.mermaid` source with SVG so stored `html` matches the admin editor save path (plain v-html in the app).
 */
export async function createMermaidRenderSession() {
  if (process.env.W4H_MERMAID_DISABLE === '1' || process.env.W4H_MERMAID_DISABLE === 'true') {
    return null;
  }
  if (!(await fs.pathExists(MERMAID_MIN))) {
    console.warn('[docs-mermaid-render] mermaid.min.js missing; run: npm install mermaid');
    return null;
  }

  let chromium;
  try {
    ({ chromium } = await import('playwright'));
  } catch (e) {
    console.warn(
      '[docs-mermaid-render] Playwright not available (%s). ' +
        'Diagrams stay as source in HTML. Install: npm i playwright && npx playwright install chromium',
      e.message
    );
    return null;
  }

  let browser;
  let page;
  try {
    browser = await chromium.launch({ headless: true });
    page = await browser.newPage();
    page.setDefaultTimeout(120_000);
    await page.setContent(
      '<!DOCTYPE html><html><head><meta charset="utf-8"></head><body><div id="w4h-md-root"></div></body></html>'
    );
    await page.addScriptTag({ path: MERMAID_MIN });
    await page.evaluate(() => {
      window.mermaid.initialize({ startOnLoad: false, theme: 'default' });
    });
  } catch (e) {
    console.warn('[docs-mermaid-render] Browser or Mermaid init failed:', e.message);
    if (browser) await browser.close().catch(() => {});
    return null;
  }

  return {
    /**
     * @param {string} html markdown-rendered HTML for one doc
     * @returns {Promise<string>}
     */
    async transformHtml(html) {
      if (!html || !html.includes('mermaid')) return html;
      await page.evaluate((fragment) => {
        document.getElementById('w4h-md-root').innerHTML = fragment;
      }, html);
      await page.evaluate(async () => {
        await window.mermaid.run({ querySelector: '#w4h-md-root .mermaid' });
      });
      return page.evaluate(() => document.getElementById('w4h-md-root').innerHTML);
    },
    async close() {
      await browser.close().catch(() => {});
    }
  };
}
