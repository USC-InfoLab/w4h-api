import { query } from '../../db.js';
import { listContributions } from '../contributions.js';
import { isSuperAdmin, isTeamAdminOnTeam } from '../utils/user.js';

const MODES = new Set(['help', 'import', 'irb', 'analysis']);

export function normalizeMode(mode) {
  const m = String(mode || 'help').toLowerCase();
  return MODES.has(m) ? m : 'help';
}

function clip(text, max = 240) {
  const t = String(text || '').replace(/\s+/g, ' ').trim();
  if (t.length <= max) return t;
  return `${t.slice(0, max - 1)}…`;
}

async function loadHelpIndex() {
  try {
    const res = await query(
      `SELECT slug, title, description
       FROM w4h.docs
       WHERE COALESCE(hidden, false) = false
       ORDER BY sort_order NULLS LAST, title
       LIMIT 40`
    );
    return res.rows.map((r) => {
      const slug = r.slug || '';
      return `- [${r.title || slug}](/docs/${slug})${r.description ? ` — ${clip(r.description, 120)}` : ''}`;
    });
  } catch {
    return [];
  }
}

async function loadDatasetMeta(datasetId) {
  if (!datasetId) return null;
  const res = await query(
    `SELECT id, name, description, device_type, sampling_rate, subject_count,
            date_start, date_end, irb_reference
     FROM w4h.datasets
     WHERE id = $1`,
    [String(datasetId)]
  );
  return res.rows[0] || null;
}

async function loadProjectMeta(projectId) {
  if (!projectId) return null;
  const res = await query(
    `SELECT id, slug, name, description, irb_protocol_id, irb_expiry_date
     FROM w4h.projects
     WHERE id::text = $1 OR slug = $1
     LIMIT 1`,
    [String(projectId)]
  );
  return res.rows[0] || null;
}

function projectEditPath(project) {
  const pid = project?.slug || project?.id;
  return pid ? `/admin/projects/${pid}/edit` : '/admin/projects';
}

async function loadProjectPageIndex(project) {
  if (!project?.id) return [];
  try {
    const res = await query(
      `SELECT slug, title
       FROM w4h.project_pages
       WHERE project_id = $1::uuid
       ORDER BY sort_order ASC, title ASC
       LIMIT 30`,
      [project.id]
    );
    const pid = project.slug || project.id;
    const href = `/admin/projects/${pid}/edit`;
    return res.rows.map((r) => {
      const title = r.title || r.slug;
      return `- [${title}](${href}) (project page \`${r.slug}\`)`;
    });
  } catch {
    return [];
  }
}

function contributionLines(rows) {
  return (rows || []).slice(0, 24).map((p) => {
    const slug = p.slug || '';
    const kinds = Array.isArray(p.kinds) && p.kinds.length ? p.kinds.join('+') : p.kind;
    return `- [${p.title || slug}](/contributions/${slug}) (${p.publisher}; ${kinds}/${p.language})${p.summary ? ` — ${clip(p.summary, 140)}` : ''}`;
  });
}

async function resolveUiRole({ user, teamId, shell }) {
  const platformRole = user?.role;
  if (isSuperAdmin(platformRole)) {
    return {
      uiRole: 'Super admin',
      shellLabel: shell === 'admin' ? 'Platform Admin' : 'app',
      note: 'This person is Super admin. In Platform Admin they bootstrap SMTP, first team, invite Team admins, audit, usage, settings, and Contributions catalog CRUD. They do not run day-to-day trials. Do not tell them to create a PI project unless they asked about a test tenant.',
    };
  }
  if (teamId && (await isTeamAdminOnTeam(user?.id, teamId, platformRole))) {
    return {
      uiRole: 'Admin',
      shellLabel: 'app',
      note: 'Team admin (UI: Admin) on the active team. They create projects, link datasets, invite members, share datasets, and set the lab AI key on Project → People. They do not use Platform Admin. Super admin is not a project membership.',
    };
  }
  return {
    uiRole: 'User',
    shellLabel: 'app',
    note: 'Project User. They work in assigned projects: linked datasets, project pages, personal API key / project token, browse Contributions. They do not create teams, ingest via privileged DDL, or manage catalog contributions.',
  };
}

function roleInstructions(ui) {
  return `ROLE-BASED ANSWERS:
You are speaking to a **${ui.uiRole}** in the **${ui.shellLabel}** shell.
${ui.note}
- Super admin: bootstrap (SMTP, first team, invite Team admin), platform settings, audit/usage, curate Contributions. Not day-to-day trial work.
- Admin: own-team projects, datasets, members, shares, lab AI key at Project → People (/admin/projects/{slug}/edit?tab=people). Point to app-shell routes, not /admin/home unless they are Super admin.
- User: analysis in a project they belong to; issue/use tokens; open Help and Contributions. If they need a dataset created or a member invited, tell them to ask an Admin.
Do not describe capabilities they do not have. If a task needs a higher role, say who can do it.`;
}

function linkInstructions() {
  return `LINKS (always when a matching page exists in the indexes below):
- Use Markdown links with in-app paths. Do not invent slugs or URLs.
- Platform Help: [Title](/docs/{slug}) from HELP INDEX.
- Contributions: [Title](/contributions/{slug}) from CONTRIBUTIONS.
- Project (settings, members, lab AI key): [Title](/admin/projects/{slug}/edit?tab=people). Never use /projects/{slug} or /team/settings — those are not the project UI.
- Platform Admin pages (Super admin, Platform Admin shell only — never offer these to Admin/User): [Contributions catalog](/admin/contributions), [Platform settings](/admin/settings), [Audit log](/admin/audit-log), [Usage](/admin/usage), [Teams](/admin/teams).
- If nothing in the indexes matches, do not fabricate a link; say the page is not in Help/project docs.
- Prefer a short "See also" line with 1–3 links over dumping the whole index.`;
}

function contributingInstructions() {
  return `PUBLISHING A CONTRIBUTION (when asked how to add/publish a sample dataset, importer, notebook, CLI, or extension):
- One git repo per tool. Sample-data repos ship a manifest.yaml (not manifest.json) with column types and GeoMTS mapping (ADR-016).
- Dataset vs. importer order: cataloging the two w4h.contributions rows has no enforced order — they are independent rows. Publish the sample-data repo first (or at least make it public before anyone runs the importer): the in-app "Run import…" flow resolves the paired data repo automatically by ADR-016 naming (\`<importer-repo>\` minus \`-importer\`) and fails to clone it if it isn't there yet.
- Today a Super admin adds a contribution from [Contributions catalog](/admin/contributions), either by pasting a repo URL and clicking "Fetch repo info" (auto-fills slug/publisher/title/summary/kind/language from the README and, if present, manifest.yaml, and shows manifest sanity-check errors/warnings) or by filling the form manually.
- This only checks first-party/admin-curated contributions. Community self-publish, with its own validation, is future work — do not imply it exists yet.`;
}

function modeInstructions(mode) {
  switch (mode) {
    case 'import':
      return `You are helping import wearable data into W4H.
Explain Contributions (notebooks, CLI, scripts, postgres extensions), personal API keys, CSV/JSON ingest, and GeoMTS role mapping (do not tell them to rename columns to subject_id/ts/lat/lon).
Link Help and contribution pages from the indexes. Never ask them to paste raw rows or subject identifiers.`;
    case 'irb':
      return `You are helping draft IRB / protocol language for wearable research on W4H.
Cover data privacy, HIPAA-aligned access (team tenancy, audit log, no shared sample keys), device descriptions, and a draft Data Management Plan from project metadata (PI/team/dataset types/IRB protocol id when provided).
This is drafting assistance, not legal or IRB approval. Flag that institutional review is required.
Link HIPAA / Help pages when listed. Do not include real subject identifiers.`;
    case 'analysis':
      return `You are the W4H Analytics assistant (not platform Help).
The data is already in W4H. Help with research methods and notebook/SQL snippets: derived metrics (e.g. energy expenditure from heart rate / accelerometer), filtering, plotting, aggregation, wear-time.
Use schema/column names and GeoMTS role mapping only — never raw PHI rows or invented subject IDs.
Cite published formulas when you use them; say they need protocol/IRB review for a specific study.
Prefer the project's API + personal API key. Link Contributions and Help when relevant. W4H does not host project notebooks; point at project pages + the researcher's git repo.
If they ask how to use the UI (invite, keys, which screen), answer briefly and say Help mode is for platform how-to.`;
    default:
      return `You are the W4H platform Help assistant (not the Analytics/methods assistant).
Answer how to use teams, projects, datasets, Contributions, API keys, and in-app screens for this user's role.
Do not write analysis pipelines or metric formulas here. If they ask how to compute or plot wearable signals, give a one-line pointer and tell them to use Analytics mode.`;
  }
}

export async function buildSystemPrompt({
  mode,
  shell,
  route,
  role,
  user,
  teamId,
  teamName,
  projectId,
  datasetId,
  docSlug,
  contributionSlug,
  phiFilter,
}) {
  const [helpLines, project, dataset, contribs] = await Promise.all([
    loadHelpIndex(),
    loadProjectMeta(projectId),
    loadDatasetMeta(datasetId),
    listContributions().catch(() => []),
  ]);
  const pageLines = await loadProjectPageIndex(project);
  const contribLines = contributionLines(contribs);
  const m = normalizeMode(mode);
  const ui = await resolveUiRole({ user: user || { role }, teamId, shell });

  const context = [
    `Shell: ${shell === 'admin' ? 'Platform Admin' : 'app'}`,
    `Route: ${route || '(unknown)'}`,
    `UI role: ${ui.uiRole} (stored platform role: ${role || '(unknown)'})`,
    teamName ? `Active team: ${teamName}` : null,
    project
      ? `Active project: ${project.name} (open [${project.name}](${projectEditPath(project)}); IRB ${project.irb_protocol_id || 'n/a'}, expiry ${project.irb_expiry_date || 'n/a'})`
      : null,
    dataset
      ? `Active dataset: ${dataset.name || dataset.id} (device ${dataset.device_type || 'n/a'}, subjects ${dataset.subject_count ?? 'n/a'}, range ${dataset.date_start || '?'}–${dataset.date_end || '?'}, IRB ref ${dataset.irb_reference || 'n/a'})`
      : null,
    docSlug ? `Current Help page: [/docs/${docSlug}](/docs/${docSlug})` : null,
    contributionSlug ? `Current contribution: [/contributions/${contributionSlug}](/contributions/${contributionSlug})` : null,
    `PHI filter: ${phiFilter}`,
  ].filter(Boolean);

  return `You are PHI-safe AI chat for Wearables for Health (W4H), a multi-tenant clinical research platform.
Team is the tenancy boundary. Project is the work unit. UI roles are Super admin · Admin · User.
GeoMTS is a role mapping over physical tables (w4h.tables), not forced column names.

${modeInstructions(m)}

${roleInstructions(ui)}

${contributingInstructions()}

${linkInstructions()}

PHI-SAFE RULES (always):
- Never request, repeat, or invent subject IDs, emails, MRNs, GPS tracks, or raw timeseries samples.
- Use metadata only: names, schema/column names and types, subject counts, date ranges.
- If the user pastes what looks like PHI, refuse and tell them to remove identifiers.
- You are not a substitute for IRB, HIPAA, or institutional counsel.

CURRENT CONTEXT:
${context.map((l) => `- ${l}`).join('\n')}

HELP INDEX (link these when relevant):
${helpLines.length ? helpLines.join('\n') : '(none loaded)'}

PROJECT PAGES (this project only; omit if none):
${pageLines.length ? pageLines.join('\n') : '(none — do not invent project-doc links)'}

CONTRIBUTIONS:
${contribLines.length ? contribLines.join('\n') : '(none loaded)'}

Keep answers concise and actionable. End with a See also line of Markdown links when the indexes have a match.`;
}
