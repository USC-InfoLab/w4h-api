CREATE SCHEMA IF NOT EXISTS w4h;

CREATE TABLE IF NOT EXISTS w4h.users (
  id SERIAL PRIMARY KEY,
  name VARCHAR(255) NOT NULL,
  email VARCHAR(255) UNIQUE NOT NULL,
  password VARCHAR(255) NOT NULL,
  role VARCHAR(50) NOT NULL CHECK (role IN ('admin', 'researcher', 'clinician')),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_users_email ON w4h.users(email);

CREATE TABLE IF NOT EXISTS w4h.projects (
  id text COLLATE pg_catalog."default" NOT NULL,
  name text COLLATE pg_catalog."default" NOT NULL,
  description text COLLATE pg_catalog."default",
  CONSTRAINT projects_pkey PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS w4h.datasets (
  id text COLLATE pg_catalog."default" NOT NULL,
  project_id text COLLATE pg_catalog."default",
  name text COLLATE pg_catalog."default",
  description text COLLATE pg_catalog."default",
  CONSTRAINT datasets_pkey1 PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_datasets_project_id ON w4h.datasets(project_id);

CREATE TABLE IF NOT EXISTS w4h.user_projects (
  user_id integer NOT NULL,
  project_id text COLLATE pg_catalog."default" NOT NULL,
  CONSTRAINT user_projects_pkey PRIMARY KEY (user_id, project_id),
  CONSTRAINT user_projects_user_fk FOREIGN KEY (user_id)
    REFERENCES w4h.users (id)
    ON DELETE CASCADE,
  CONSTRAINT user_projects_fk FOREIGN KEY (project_id)
    REFERENCES w4h.projects (id)
    ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_user_projects_user_id ON w4h.user_projects(user_id);
CREATE INDEX IF NOT EXISTS idx_user_projects_project_id ON w4h.user_projects(project_id);

ALTER TABLE IF EXISTS w4h.datasets
  ADD COLUMN IF NOT EXISTS project_id text COLLATE pg_catalog."default";

ALTER TABLE IF EXISTS w4h.datasets
  DROP COLUMN IF EXISTS hidden;
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
CREATE INDEX IF NOT EXISTS idx_docs_slug ON w4h.docs(slug);

CREATE TABLE IF NOT EXISTS w4h.user_api_keys (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  user_id integer NOT NULL,
  token_hash text NOT NULL,
  prefix text NOT NULL,
  created_at timestamp without time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  revoked_at timestamp without time zone,
  CONSTRAINT user_api_keys_pkey PRIMARY KEY (id),
  CONSTRAINT user_api_keys_user_fk FOREIGN KEY (user_id)
    REFERENCES w4h.users (id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_user_api_keys_token_active
  ON w4h.user_api_keys (token_hash)
  WHERE revoked_at IS NULL;

CREATE UNIQUE INDEX IF NOT EXISTS idx_user_api_keys_one_active_per_user
  ON w4h.user_api_keys (user_id)
  WHERE revoked_at IS NULL;