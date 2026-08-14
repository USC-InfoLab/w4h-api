BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS w4h.projects (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  slug text UNIQUE NOT NULL,
  name text NOT NULL,
  description text,
  status text NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'archived')),
  visibility text NOT NULL DEFAULT 'org-private' CHECK (visibility IN ('org-private', 'project-private', 'internal')),
  irb_protocol_id text,
  irb_expiry_date date,
  created_by integer NOT NULL REFERENCES w4h.users(id) ON DELETE RESTRICT,
  created_at timestamp without time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at timestamp without time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE w4h.projects
  ADD COLUMN IF NOT EXISTS launchers jsonb NOT NULL DEFAULT '{}'::jsonb;

CREATE INDEX IF NOT EXISTS idx_projects_slug ON w4h.projects(slug);
CREATE INDEX IF NOT EXISTS idx_projects_created_by ON w4h.projects(created_by);

CREATE TABLE IF NOT EXISTS w4h.project_members (
  project_id uuid NOT NULL REFERENCES w4h.projects(id) ON DELETE CASCADE,
  user_id integer NOT NULL REFERENCES w4h.users(id) ON DELETE CASCADE,
  role text NOT NULL CHECK (role IN ('owner', 'editor', 'viewer', 'admin')),
  invited_by integer REFERENCES w4h.users(id) ON DELETE SET NULL,
  joined_at timestamp without time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (project_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_project_members_user_id ON w4h.project_members(user_id);
CREATE INDEX IF NOT EXISTS idx_project_members_project_id ON w4h.project_members(project_id);

CREATE TABLE IF NOT EXISTS w4h.project_datasets (
  project_id uuid NOT NULL REFERENCES w4h.projects(id) ON DELETE CASCADE,
  dataset_id text NOT NULL REFERENCES w4h.datasets(id) ON DELETE RESTRICT,
  access_type text NOT NULL DEFAULT 'read' CHECK (access_type IN ('read', 'read_write')),
  approved_by integer REFERENCES w4h.users(id) ON DELETE SET NULL,
  approved_at timestamp without time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (project_id, dataset_id)
);

CREATE INDEX IF NOT EXISTS idx_project_datasets_dataset_id ON w4h.project_datasets(dataset_id);

CREATE TABLE IF NOT EXISTS w4h.project_notebooks (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id uuid NOT NULL REFERENCES w4h.projects(id) ON DELETE CASCADE,
  title text NOT NULL,
  description text,
  purpose text NOT NULL DEFAULT 'analysis'
    CHECK (purpose IN ('exploration', 'preprocessing', 'analysis', 'visualization', 'report')),
  repo_url text,
  repo_ref text,
  notebook_path text NOT NULL,
  runtime_type text NOT NULL DEFAULT 'jupyter'
    CHECK (runtime_type IN ('jupyter', 'rstudio', 'vscode', 'quarto', 'other')),
  launchers jsonb NOT NULL DEFAULT '["jupyter"]'::jsonb,
  created_by integer REFERENCES w4h.users(id) ON DELETE SET NULL,
  created_at timestamp without time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at timestamp without time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_project_notebooks_project_id ON w4h.project_notebooks(project_id);

CREATE TABLE IF NOT EXISTS w4h.project_pages (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id uuid NOT NULL REFERENCES w4h.projects(id) ON DELETE CASCADE,
  slug text NOT NULL,
  title text NOT NULL,
  doc_type text NOT NULL DEFAULT 'guide',
  sort_order integer NOT NULL DEFAULT 0,
  md text NOT NULL DEFAULT '',
  html text NOT NULL DEFAULT '',
  toc jsonb NOT NULL DEFAULT '[]'::jsonb,
  created_by integer REFERENCES w4h.users(id) ON DELETE SET NULL,
  updated_by text,
  created_at timestamp without time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at timestamp without time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE (project_id, slug)
);

CREATE INDEX IF NOT EXISTS idx_project_pages_project_id
  ON w4h.project_pages(project_id, sort_order, title);

CREATE TABLE IF NOT EXISTS w4h.project_docs (
  project_id uuid NOT NULL REFERENCES w4h.projects(id) ON DELETE CASCADE,
  doc_slug text NOT NULL REFERENCES w4h.docs(slug) ON DELETE CASCADE,
  linked_by integer REFERENCES w4h.users(id) ON DELETE SET NULL,
  linked_at timestamp without time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (project_id, doc_slug)
);

CREATE TABLE IF NOT EXISTS w4h.project_acl_rules (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id uuid NOT NULL REFERENCES w4h.projects(id) ON DELETE CASCADE,
  subject_type text NOT NULL CHECK (subject_type IN ('role', 'user')),
  subject_key text NOT NULL,
  resource_type text NOT NULL CHECK (resource_type IN ('project', 'dataset', 'doc', 'derived_table', 'notebook')),
  resource_key text,
  permission text NOT NULL CHECK (
    permission IN ('view', 'edit', 'manage_members', 'link_dataset', 'create_derived', 'launch_notebook')
  ),
  effect text NOT NULL DEFAULT 'allow' CHECK (effect IN ('allow', 'deny')),
  created_by integer REFERENCES w4h.users(id) ON DELETE SET NULL,
  created_at timestamp without time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_project_acl_project_id ON w4h.project_acl_rules(project_id);
CREATE INDEX IF NOT EXISTS idx_project_acl_subject ON w4h.project_acl_rules(subject_type, subject_key);

CREATE TABLE IF NOT EXISTS w4h.project_db_tokens (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id uuid NOT NULL REFERENCES w4h.projects(id) ON DELETE CASCADE,
  user_id integer NOT NULL REFERENCES w4h.users(id) ON DELETE CASCADE,
  token_hash text NOT NULL UNIQUE,
  expires_at timestamp without time zone NOT NULL,
  issued_at timestamp without time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  revoked_at timestamp without time zone
);

CREATE INDEX IF NOT EXISTS idx_project_db_tokens_lookup ON w4h.project_db_tokens(project_id, user_id, expires_at);

CREATE TABLE IF NOT EXISTS w4h.project_organizations (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id uuid NOT NULL REFERENCES w4h.projects(id) ON DELETE CASCADE,
  short text,
  name text NOT NULL,
  website_url text,
  sort_order integer NOT NULL DEFAULT 0,
  created_at timestamp without time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_project_organizations_project_id
  ON w4h.project_organizations(project_id, sort_order, name);

ALTER TABLE w4h.project_organizations DROP COLUMN IF EXISTS role;

ALTER TABLE w4h.project_organizations ADD COLUMN IF NOT EXISTS website_url text;

DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = 'w4h'
      AND table_name = 'project_organizations'
      AND column_name = 'url'
  ) THEN
    UPDATE w4h.project_organizations
    SET website_url = COALESCE(NULLIF(TRIM(website_url), ''), url);
    ALTER TABLE w4h.project_organizations DROP COLUMN url;
  END IF;
END $$;

ALTER TABLE w4h.users ADD COLUMN IF NOT EXISTS api_key_hash text;
ALTER TABLE w4h.users ADD COLUMN IF NOT EXISTS api_key_prefix text;
ALTER TABLE w4h.users ADD COLUMN IF NOT EXISTS api_key_created_at timestamp with time zone;
ALTER TABLE w4h.users ADD COLUMN IF NOT EXISTS api_key_last_used_at timestamp with time zone;
ALTER TABLE w4h.users ADD COLUMN IF NOT EXISTS api_key_encrypted text;
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_api_key_hash ON w4h.users(api_key_hash) WHERE api_key_hash IS NOT NULL;

ALTER TABLE w4h.datasets ADD COLUMN IF NOT EXISTS summary jsonb;
ALTER TABLE w4h.datasets ADD COLUMN IF NOT EXISTS summary_updated_at timestamp with time zone;

CREATE TABLE IF NOT EXISTS w4h.tables (
  dataset_id text NOT NULL REFERENCES w4h.datasets(id) ON DELETE CASCADE,
  name text NOT NULL,
  metabase_url text,
  source_name text,
  source_url text,
  PRIMARY KEY (dataset_id, name)
);

CREATE INDEX IF NOT EXISTS idx_tables_dataset_id ON w4h.tables(dataset_id);

COMMIT;
