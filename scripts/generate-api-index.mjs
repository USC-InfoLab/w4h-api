#!/usr/bin/env node
/**
 * Generate API_INDEX.md — structural map of mounts, routes, lib, schema.
 * Run: npm run index:api
 * Not hooked into start/build.
 */
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const ROOT = path.resolve(__dirname, '..')
const OUT = path.join(ROOT, 'API_INDEX.md')

function walkFiles(dir, exts, base = ROOT) {
  if (!fs.existsSync(dir)) return []
  const out = []
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (entry.name === 'node_modules' || entry.name.startsWith('.')) continue
    const full = path.join(dir, entry.name)
    if (entry.isDirectory()) {
      out.push(...walkFiles(full, exts, base))
    } else if (exts.some((e) => entry.name.endsWith(e))) {
      out.push(path.relative(base, full).split(path.sep).join('/'))
    }
  }
  return out.sort((a, b) => a.localeCompare(b))
}

function parseMounts(appSrc) {
  const mounts = []
  for (const m of appSrc.matchAll(/app\.use\(\s*['"]([^'"]+)['"]\s*,\s*(\w+)\s*\)/g)) {
    mounts.push({ mount: m[1], exportName: m[2] })
  }
  // Resolve import name → routes/*.js
  const imports = new Map()
  for (const m of appSrc.matchAll(/import\s+(\w+)\s+from\s+['"]\.\/(routes\/[^'"]+)['"]/g)) {
    imports.set(m[1], m[2])
  }
  return mounts.map((row) => ({
    mount: row.mount,
    file: imports.get(row.exportName) || row.exportName,
  }))
}

function parseRouteHandlers(filePath, mount) {
  const src = fs.readFileSync(filePath, 'utf8')
  const rows = []
  for (const m of src.matchAll(
    /router\.(get|post|put|patch|delete)\(\s*['`]([^'`]*)['`]/g
  )) {
    const method = m[1].toUpperCase()
    const sub = m[2]
    const full = (mount.replace(/\/$/, '') + (sub.startsWith('/') ? sub : `/${sub}`)).replace(
      /\/+/g,
      '/'
    )
    // Auth hints on same line (best-effort)
    const lineStart = src.lastIndexOf('\n', m.index) + 1
    const lineEnd = src.indexOf('\n', m.index)
    const line = src.slice(lineStart, lineEnd === -1 ? undefined : lineEnd)
    const auth = []
    if (/authenticate|authenticateJwt/i.test(line)) auth.push('auth')
    if (/requirePlatformAdmin|requirePlatformAdmin/.test(line)) auth.push('platformAdmin')
    if (/requireTeamAdmin/.test(line)) auth.push('teamAdmin')
    if (/ApiKey|apiKey|ProjectDbToken/i.test(line)) auth.push('apiKey?')
    rows.push({ method, path: full, auth: auth.length ? auth.join(', ') : '—' })
  }
  return rows
}

function parseCreateTables(sqlPath) {
  if (!fs.existsSync(sqlPath)) return []
  const src = fs.readFileSync(sqlPath, 'utf8')
  // Strip SQL line/block comments before matching
  const cleaned = src
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/--[^\n]*/g, '')
  const tables = new Set()
  for (const m of cleaned.matchAll(
    /CREATE TABLE(?:\s+IF NOT EXISTS)?\s+((?:[a-zA-Z_][\w]*\.)?[a-zA-Z_][\w]*)/gi
  )) {
    tables.add(m[1])
  }
  return [...tables].sort((a, b) => a.localeCompare(b))
}

function groupByDir(relPaths, rootLabel) {
  const groups = new Map()
  for (const rel of relPaths) {
    const dir = path.posix.dirname(rel)
    const key = dir === '.' ? rootLabel : dir
    if (!groups.has(key)) groups.set(key, [])
    groups.get(key).push(path.posix.basename(rel))
  }
  return groups
}

function main() {
  const appSrc = fs.readFileSync(path.join(ROOT, 'app.js'), 'utf8')
  const mounts = parseMounts(appSrc)
  const routeFiles = walkFiles(path.join(ROOT, 'routes'), ['.js'])
  const mountedFiles = new Set(mounts.map((m) => m.file))
  const unmounted = routeFiles.filter((f) => !mountedFiles.has(f))

  /** @type {Map<string, {method:string,path:string}[]>} */
  const byMount = new Map()
  byMount.set('(root)', [{ method: 'GET', path: '/' }])
  for (const m of mounts) {
    const full = path.join(ROOT, m.file)
    if (!fs.existsSync(full)) continue
    byMount.set(m.mount, parseRouteHandlers(full, m.mount))
  }
  const endpointCount = [...byMount.values()].reduce((n, rows) => n + rows.length, 0)

  const libFiles = walkFiles(path.join(ROOT, 'lib'), ['.js'])
  const schemaTables = parseCreateTables(path.join(ROOT, 'data', 'schema.sql'))
  const aclTables = parseCreateTables(path.join(ROOT, 'data', 'projects_acl.sql'))
  const allTables = [...new Set([...schemaTables, ...aclTables])].sort((a, b) =>
    a.localeCompare(b)
  )

  const generatedAt = new Date().toISOString().slice(0, 10)
  const mountRows = mounts
    .map((m) => `| \`${m.mount}\` | \`${m.file}\` |`)
    .join('\n')

  const epSections = []
  for (const [mount, rows] of byMount) {
    const label = mount === '(root)' ? 'Root' : `\`${mount}\``
    const items = rows.map((e) => `${e.method} \`${e.path}\``).join(' · ')
    epSections.push(`- **${label}:** ${items}`)
  }

  const libLines = []
  for (const [dir, files] of groupByDir(libFiles, 'lib')) {
    libLines.push(`- \`${dir}/\`: ${files.map((f) => `\`${f}\``).join(', ')}`)
  }

  const md = `# API Index

Structural map of \`w4h-api\` for agents and humans. **Index only** — not API docs.

- Regenerate: \`npm run index:api\`
- Generated: ${generatedAt}
- Do not hand-edit the lists below; change code, then re-run the script.

## Orientation

| Area | Path |
|------|------|
| Entry | \`app.js\` |
| Auth | \`auth.js\` |
| DB helpers | \`db.js\`, \`lib/db-init.js\` |
| Routes | \`routes/\` |
| Lib | \`lib/\` |
| Schema | \`data/schema.sql\`, \`data/projects_acl.sql\` |
| CLI | \`bin/w4h\` |

## Mounts

| Mount | File |
|-------|------|
${mountRows}

${
  unmounted.length
    ? `Unmounted route modules: ${unmounted.map((f) => `\`${f}\``).join(', ')}\n`
    : ''
}
## Endpoints (${endpointCount})

${epSections.join('\n')}

## Lib (${libFiles.length})

${libLines.join('\n')}

## Schema tables (${allTables.length})

From \`CREATE TABLE\` in \`data/*.sql\` (may lag live DB):

${allTables.map((t) => `\`${t}\``).join(' · ')}
`

  fs.writeFileSync(OUT, md.trimEnd() + '\n', 'utf8')
  const lineCount = md.trimEnd().split('\n').length
  console.log(
    `Wrote ${path.relative(ROOT, OUT)} (${lineCount} lines, ${mounts.length} mounts, ${endpointCount} endpoints, ${libFiles.length} lib files, ${allTables.length} tables)`
  )
  if (lineCount > 180) {
    console.warn(`Warning: API_INDEX.md is ${lineCount} lines (prefer ~50–150).`)
  }
}

main()
