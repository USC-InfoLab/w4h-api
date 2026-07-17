CREATE SCHEMA IF NOT EXISTS w4h;

-- Enable pgcrypto extension for gen_random_uuid()
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ---------------------------------------------------------------------------
-- Users — platform roles: super_admin · team_admin · project_user
-- (legacy values admin / researcher / clinician still accepted until migrated)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS w4h.users (
  id SERIAL PRIMARY KEY,
  name VARCHAR(255) NOT NULL,
  email VARCHAR(255) UNIQUE NOT NULL,
  password VARCHAR(255) NOT NULL,
  role VARCHAR(50) NOT NULL CHECK (role IN (
    'super_admin', 'team_admin', 'project_user',
    'admin', 'researcher', 'clinician'
  )),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_users_email ON w4h.users(email);

ALTER TABLE w4h.users ADD COLUMN IF NOT EXISTS api_key_hash text;
ALTER TABLE w4h.users ADD COLUMN IF NOT EXISTS api_key_prefix text;
ALTER TABLE w4h.users ADD COLUMN IF NOT EXISTS api_key_created_at timestamp with time zone;
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_api_key_hash ON w4h.users(api_key_hash) WHERE api_key_hash IS NOT NULL;

-- User lifecycle: invited → active → discontinued (ADR-010)
ALTER TABLE w4h.users ADD COLUMN IF NOT EXISTS status text NOT NULL DEFAULT 'active';
ALTER TABLE w4h.users DROP CONSTRAINT IF EXISTS users_status_check;
ALTER TABLE w4h.users ADD CONSTRAINT users_status_check
  CHECK (status IN ('invited', 'active', 'discontinued'));
ALTER TABLE w4h.users ALTER COLUMN password DROP NOT NULL;

-- ---------------------------------------------------------------------------
-- Invitations — time-limited tokens for password set / account activation (U-14)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS w4h.invitations (
  id          SERIAL PRIMARY KEY,
  user_id     INTEGER NOT NULL REFERENCES w4h.users(id) ON DELETE CASCADE,
  token_hash  TEXT NOT NULL UNIQUE,
  invited_by  INTEGER REFERENCES w4h.users(id) ON DELETE SET NULL,
  expires_at  TIMESTAMPTZ NOT NULL,
  used_at     TIMESTAMPTZ,
  revoked_at  TIMESTAMPTZ,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_invitations_user_id ON w4h.invitations(user_id);
CREATE INDEX IF NOT EXISTS idx_invitations_expires_at ON w4h.invitations(expires_at);

-- ---------------------------------------------------------------------------
-- Projects — research workspaces/projects containing members, datasets, and notebooks
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS w4h.projects (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  slug text UNIQUE NOT NULL, -- URL-friendly short name (e.g. 'cardio-study')
  name text NOT NULL, -- Descriptive human-readable project name
  description text, -- Summary of the project goals and scope
  status text NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'archived')), -- Status flag (active/archived)
  visibility text NOT NULL DEFAULT 'org-private' CHECK (visibility IN ('org-private', 'project-private', 'internal')), -- Accessibility scope
  irb_protocol_id text, -- Institutional Review Board protocol tracking number
  irb_expiry_date date, -- IRB approval expiration date
  created_by integer NOT NULL REFERENCES w4h.users(id) ON DELETE RESTRICT, -- User who created the project
  created_at timestamp without time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at timestamp without time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE w4h.projects ADD COLUMN IF NOT EXISTS launchers jsonb NOT NULL DEFAULT '{}'::jsonb;

CREATE INDEX IF NOT EXISTS idx_projects_slug ON w4h.projects(slug);
CREATE INDEX IF NOT EXISTS idx_projects_created_by ON w4h.projects(created_by);

-- ---------------------------------------------------------------------------
-- Datasets — references to structured/unstructured dataset resources
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS w4h.datasets (
  id text COLLATE pg_catalog."default" NOT NULL,
  project_id text COLLATE pg_catalog."default", -- Legacy optional reference to project text ID
  name text COLLATE pg_catalog."default",
  description text COLLATE pg_catalog."default",
  CONSTRAINT datasets_pkey1 PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_datasets_project_id ON w4h.datasets(project_id);

ALTER TABLE IF EXISTS w4h.datasets ADD COLUMN IF NOT EXISTS project_id text COLLATE pg_catalog."default";
ALTER TABLE IF EXISTS w4h.datasets DROP COLUMN IF EXISTS hidden;

-- ---------------------------------------------------------------------------
-- User Projects (Legacy compatibility mapping)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS w4h.user_projects (
  user_id integer NOT NULL,
  project_id uuid NOT NULL, -- References UUID-based projects.id
  CONSTRAINT user_projects_pkey PRIMARY KEY (user_id, project_id),
  CONSTRAINT user_projects_user_fk FOREIGN KEY (user_id) REFERENCES w4h.users (id) ON DELETE CASCADE,
  CONSTRAINT user_projects_fk FOREIGN KEY (project_id) REFERENCES w4h.projects (id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_user_projects_user_id ON w4h.user_projects(user_id);
CREATE INDEX IF NOT EXISTS idx_user_projects_project_id ON w4h.user_projects(project_id);

-- ---------------------------------------------------------------------------
-- Login codes — short-lived email OTP for passwordless sign-in (SMTP required)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS w4h.login_codes (
  id          SERIAL PRIMARY KEY,
  user_id     INTEGER NOT NULL REFERENCES w4h.users(id) ON DELETE CASCADE,
  code_hash   TEXT NOT NULL,
  expires_at  TIMESTAMPTZ NOT NULL,
  attempts    INTEGER NOT NULL DEFAULT 0,
  used_at     TIMESTAMPTZ,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_login_codes_user_id ON w4h.login_codes(user_id);
CREATE INDEX IF NOT EXISTS idx_login_codes_expires_at ON w4h.login_codes(expires_at);

-- ---------------------------------------------------------------------------
-- Docs — documentation pages and knowledgebase articles
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS w4h.docs (
  id uuid NOT NULL DEFAULT gen_random_uuid (),
  slug text COLLATE pg_catalog."default" NOT NULL,
  title text COLLATE pg_catalog."default",
  card_title text COLLATE pg_catalog."default",
  description text COLLATE pg_catalog."default",
  action_label text COLLATE pg_catalog."default",
  doc_type text COLLATE pg_catalog."default",
  sort_order integer DEFAULT 0,
  hidden boolean DEFAULT false,
  updated_at timestamp without time zone,
  is_feature_card boolean DEFAULT false,
  show_in_docs boolean DEFAULT true,
  md text COLLATE pg_catalog."default",
  html text COLLATE pg_catalog."default",
  toc jsonb,
  card_order integer DEFAULT 0,
  CONSTRAINT docs_pkey PRIMARY KEY (id),
  CONSTRAINT docs_slug_key UNIQUE (slug)
);

-- Track who last updated each doc (email of in-app editor, or git author on import)
ALTER TABLE w4h.docs ADD COLUMN IF NOT EXISTS updated_by text COLLATE pg_catalog."default";

CREATE INDEX IF NOT EXISTS idx_docs_slug ON w4h.docs(slug);

-- ---------------------------------------------------------------------------
-- Doc Types — sidebar groupings and doc-type picker options
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS w4h.doc_types (
  slug text PRIMARY KEY,
  label text NOT NULL,
  sort_order integer NOT NULL DEFAULT 0
);

-- ---------------------------------------------------------------------------
-- Project Members — user roles and permissions within specific projects
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS w4h.project_members (
  project_id uuid NOT NULL REFERENCES w4h.projects(id) ON DELETE CASCADE,
  user_id integer NOT NULL REFERENCES w4h.users(id) ON DELETE CASCADE,
  role text NOT NULL CHECK (role IN ('owner', 'editor', 'viewer')), -- Member access role (owner, editor, viewer)
  invited_by integer REFERENCES w4h.users(id) ON DELETE SET NULL, -- User who invited this member
  joined_at timestamp without time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (project_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_project_members_user_id ON w4h.project_members(user_id);
CREATE INDEX IF NOT EXISTS idx_project_members_project_id ON w4h.project_members(project_id);

-- ---------------------------------------------------------------------------
-- Project Datasets — many-to-many link between projects and associated datasets
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS w4h.project_datasets (
  project_id uuid NOT NULL REFERENCES w4h.projects(id) ON DELETE CASCADE,
  dataset_id text NOT NULL REFERENCES w4h.datasets(id) ON DELETE RESTRICT,
  access_type text NOT NULL DEFAULT 'read' CHECK (access_type IN ('read', 'read_write')), -- Access level on the dataset
  approved_by integer REFERENCES w4h.users(id) ON DELETE SET NULL, -- User who approved linking the dataset
  approved_at timestamp without time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (project_id, dataset_id)
);

CREATE INDEX IF NOT EXISTS idx_project_datasets_dataset_id ON w4h.project_datasets(dataset_id);

-- ---------------------------------------------------------------------------
-- Project Notebooks — analytics notebooks (Jupyter, RStudio, VS Code) registered to a project
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS w4h.project_notebooks (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id uuid NOT NULL REFERENCES w4h.projects(id) ON DELETE CASCADE,
  title text NOT NULL, -- Title of the notebook
  description text, -- Purpose details
  purpose text NOT NULL DEFAULT 'analysis' CHECK (purpose IN ('exploration', 'preprocessing', 'analysis', 'visualization', 'report')),
  repo_url text, -- URL to git repo hosting the notebook
  repo_ref text, -- Branch/commit/tag reference
  notebook_path text NOT NULL, -- Path to the notebook file inside the repo
  runtime_type text NOT NULL DEFAULT 'jupyter' CHECK (runtime_type IN ('jupyter', 'rstudio', 'vscode', 'quarto', 'other')), -- Target IDE environment
  launchers jsonb NOT NULL DEFAULT '["jupyter"]'::jsonb, -- Supported launching options
  created_by integer REFERENCES w4h.users(id) ON DELETE SET NULL,
  created_at timestamp without time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at timestamp without time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_project_notebooks_project_id ON w4h.project_notebooks(project_id);

-- ---------------------------------------------------------------------------
-- Project Docs — link documentation pages to specific projects
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS w4h.project_docs (
  project_id uuid NOT NULL REFERENCES w4h.projects(id) ON DELETE CASCADE,
  doc_slug text NOT NULL REFERENCES w4h.docs(slug) ON DELETE CASCADE, -- References w4h.docs slug
  linked_by integer REFERENCES w4h.users(id) ON DELETE SET NULL,
  linked_at timestamp without time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (project_id, doc_slug)
);

-- ---------------------------------------------------------------------------
-- Project ACL Rules — granular resource access control list configurations
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS w4h.project_acl_rules (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id uuid NOT NULL REFERENCES w4h.projects(id) ON DELETE CASCADE,
  subject_type text NOT NULL CHECK (subject_type IN ('role', 'user')), -- Target of rule (role or specific user)
  subject_key text NOT NULL, -- Role name (e.g. 'researcher') or user ID
  resource_type text NOT NULL CHECK (resource_type IN ('project', 'dataset', 'doc', 'derived_table', 'notebook')), -- Target resource category
  resource_key text, -- Specific resource ID/slug, or NULL to apply to all of resource_type
  permission text NOT NULL CHECK (permission IN ('view', 'edit', 'manage_members', 'link_dataset', 'create_derived', 'launch_notebook')), -- Action permitted/denied
  effect text NOT NULL DEFAULT 'allow' CHECK (effect IN ('allow', 'deny')), -- Rule effect (allow or deny)
  created_by integer REFERENCES w4h.users(id) ON DELETE SET NULL,
  created_at timestamp without time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_project_acl_project_id ON w4h.project_acl_rules(project_id);
CREATE INDEX IF NOT EXISTS idx_project_acl_subject ON w4h.project_acl_rules(subject_type, subject_key);

-- ---------------------------------------------------------------------------
-- Project DB Tokens — tokens for direct programmatic database queries
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS w4h.project_db_tokens (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id uuid NOT NULL REFERENCES w4h.projects(id) ON DELETE CASCADE,
  user_id integer NOT NULL REFERENCES w4h.users(id) ON DELETE CASCADE,
  token_hash text NOT NULL UNIQUE, -- Cryptographic hash of the database access token
  expires_at timestamp without time zone NOT NULL, -- Expiration date and time
  issued_at timestamp without time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  revoked_at timestamp without time zone -- Cancellation date and time if manually revoked
);

CREATE INDEX IF NOT EXISTS idx_project_db_tokens_lookup ON w4h.project_db_tokens(project_id, user_id, expires_at);

-- ---------------------------------------------------------------------------
-- Project Organizations — external institutions or partners associated with a project
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS w4h.project_organizations (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id uuid NOT NULL REFERENCES w4h.projects(id) ON DELETE CASCADE,
  short text, -- Short name/abbreviation (e.g. 'NIH')
  name text NOT NULL, -- Full name of organization (e.g. 'National Institutes of Health')
  website_url text, -- Web URL for the organization
  sort_order integer NOT NULL DEFAULT 0,
  created_at timestamp without time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_project_organizations_project_id ON w4h.project_organizations(project_id, sort_order, name);

-- ---------------------------------------------------------------------------
-- HIPAA Audit Log — append-only record of all access and mutations to PHI.
-- Required by HIPAA Security Rule §164.312(b). Never delete rows; retain 6 years.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS w4h.audit_log (
  id          BIGSERIAL PRIMARY KEY,
  timestamp   TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  user_id     INTEGER REFERENCES w4h.users(id) ON DELETE SET NULL,
  action      VARCHAR(100) NOT NULL,  -- e.g. 'subject.view', 'data.export', 'user.login'
  resource_type VARCHAR(100),         -- e.g. 'subject', 'dataset', 'user'
  resource_id   TEXT,                 -- id of the affected record
  ip_address    INET,
  user_agent    TEXT,
  details       JSONB                 -- any additional context
);

CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp   ON w4h.audit_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_log_user_id     ON w4h.audit_log(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_action      ON w4h.audit_log(action);
CREATE INDEX IF NOT EXISTS idx_audit_log_resource    ON w4h.audit_log(resource_type, resource_id);

-- ---------------------------------------------------------------------------
-- Usage Events — feature usage and efficiency metrics.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS w4h.usage_events (
  id          BIGSERIAL PRIMARY KEY,
  timestamp   TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  user_id     INTEGER REFERENCES w4h.users(id) ON DELETE SET NULL,
  session_id  TEXT,                   -- client-generated session UUID
  event_type  VARCHAR(100) NOT NULL,  -- e.g. 'dataset.import', 'query.run', 'page.view'
  duration_ms INTEGER,                -- time taken for the action, if measurable
  metadata    JSONB                   -- e.g. { dataset_id, row_count, page, error }
);

CREATE INDEX IF NOT EXISTS idx_usage_events_timestamp  ON w4h.usage_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_usage_events_user_id    ON w4h.usage_events(user_id);
CREATE INDEX IF NOT EXISTS idx_usage_events_event_type ON w4h.usage_events(event_type);
CREATE INDEX IF NOT EXISTS idx_usage_events_session_id ON w4h.usage_events(session_id);

-- ---------------------------------------------------------------------------
-- Survey Responses — periodic questionnaire data for efficiency/ROI reporting.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS w4h.survey_responses (
  id            BIGSERIAL PRIMARY KEY,
  timestamp     TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  user_id       INTEGER REFERENCES w4h.users(id) ON DELETE SET NULL,
  survey_type   VARCHAR(100) NOT NULL,  -- e.g. 'efficiency', 'satisfaction'
  responses     JSONB NOT NULL          -- { question_id: answer, ... }
);

CREATE INDEX IF NOT EXISTS idx_survey_responses_user_id     ON w4h.survey_responses(user_id);
CREATE INDEX IF NOT EXISTS idx_survey_responses_survey_type ON w4h.survey_responses(survey_type);

-- ---------------------------------------------------------------------------
-- Teams — research teams / trials (tenancy boundary)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS w4h.teams (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  slug text UNIQUE NOT NULL,
  name text NOT NULL,
  status text NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'deactivated')),
  created_at timestamp without time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at timestamp without time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_teams_slug ON w4h.teams(slug);
CREATE INDEX IF NOT EXISTS idx_teams_status ON w4h.teams(status);

CREATE TABLE IF NOT EXISTS w4h.team_members (
  team_id uuid NOT NULL REFERENCES w4h.teams(id) ON DELETE CASCADE,
  user_id integer NOT NULL REFERENCES w4h.users(id) ON DELETE CASCADE,
  PRIMARY KEY (team_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_team_members_user_id ON w4h.team_members(user_id);

-- Projects belong to exactly one owning team (ADR-012)
ALTER TABLE w4h.projects ADD COLUMN IF NOT EXISTS team_id uuid REFERENCES w4h.teams(id) ON DELETE RESTRICT;
CREATE INDEX IF NOT EXISTS idx_projects_team_id ON w4h.projects(team_id);

-- Datasets belong to an owning team (nullable for legacy catalog rows)
ALTER TABLE w4h.datasets ADD COLUMN IF NOT EXISTS team_id uuid REFERENCES w4h.teams(id) ON DELETE SET NULL;
CREATE INDEX IF NOT EXISTS idx_datasets_team_id ON w4h.datasets(team_id);

-- Dataset catalog metadata (D-02)
ALTER TABLE w4h.datasets ADD COLUMN IF NOT EXISTS device_type text;
ALTER TABLE w4h.datasets ADD COLUMN IF NOT EXISTS sampling_rate text;
ALTER TABLE w4h.datasets ADD COLUMN IF NOT EXISTS subject_count integer;
ALTER TABLE w4h.datasets ADD COLUMN IF NOT EXISTS date_start date;
ALTER TABLE w4h.datasets ADD COLUMN IF NOT EXISTS date_end date;
ALTER TABLE w4h.datasets ADD COLUMN IF NOT EXISTS irb_reference text;
ALTER TABLE w4h.datasets ADD COLUMN IF NOT EXISTS created_by integer REFERENCES w4h.users(id) ON DELETE SET NULL;
ALTER TABLE w4h.datasets ADD COLUMN IF NOT EXISTS created_at timestamp without time zone NOT NULL DEFAULT CURRENT_TIMESTAMP;
ALTER TABLE w4h.datasets ADD COLUMN IF NOT EXISTS updated_at timestamp without time zone NOT NULL DEFAULT CURRENT_TIMESTAMP;

-- Cross-team dataset shares (ADR-002 / ADR-012) — push-share registry
CREATE TABLE IF NOT EXISTS w4h.dataset_shares (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  dataset_id text NOT NULL REFERENCES w4h.datasets(id) ON DELETE CASCADE,
  target_team_id uuid NOT NULL REFERENCES w4h.teams(id) ON DELETE CASCADE,
  target_project_id uuid REFERENCES w4h.projects(id) ON DELETE CASCADE,
  access_type text NOT NULL DEFAULT 'read' CHECK (access_type IN ('read', 'read_write')),
  shared_by integer REFERENCES w4h.users(id) ON DELETE SET NULL,
  shared_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT unique_dataset_team_project_share
    UNIQUE (dataset_id, target_team_id, target_project_id)
);

CREATE INDEX IF NOT EXISTS idx_dataset_shares_lookup
  ON w4h.dataset_shares (dataset_id, target_team_id);
CREATE INDEX IF NOT EXISTS idx_dataset_shares_target_team
  ON w4h.dataset_shares (target_team_id);

-- Project membership roles: owner / editor / viewer only (migrate legacy admin → editor)
UPDATE w4h.project_members SET role = 'editor' WHERE role = 'admin';
ALTER TABLE w4h.project_members DROP CONSTRAINT IF EXISTS project_members_role_check;
ALTER TABLE w4h.project_members ADD CONSTRAINT project_members_role_check
  CHECK (role IN ('owner', 'editor', 'viewer'));

-- Widen role check on existing databases (CREATE TABLE IF NOT EXISTS does not alter checks)
ALTER TABLE w4h.users DROP CONSTRAINT IF EXISTS users_role_check;
ALTER TABLE w4h.users ADD CONSTRAINT users_role_check
  CHECK (role IN (
    'super_admin', 'team_admin', 'project_user',
    'admin', 'researcher', 'clinician'
  ));