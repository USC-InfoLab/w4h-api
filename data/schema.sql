-- Core schema objects for W4H
-- Idempotent: safe to run multiple times

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

CREATE TABLE IF NOT EXISTS w4h.studies (
  id text COLLATE pg_catalog."default" NOT NULL,
  name text COLLATE pg_catalog."default" NOT NULL,
  description text COLLATE pg_catalog."default",
  CONSTRAINT studies_pkey PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS w4h.datasets (
  id text COLLATE pg_catalog."default" NOT NULL,
  study_id text COLLATE pg_catalog."default",
  name text COLLATE pg_catalog."default",
  description text COLLATE pg_catalog."default",
  CONSTRAINT datasets_pkey1 PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_datasets_study_id ON w4h.datasets(study_id);

CREATE TABLE IF NOT EXISTS w4h.user_studies (
  user_id integer NOT NULL,
  study_id text COLLATE pg_catalog."default" NOT NULL,
  CONSTRAINT user_studies_pkey PRIMARY KEY (user_id, study_id),
  CONSTRAINT user_studies_user_fk FOREIGN KEY (user_id)
    REFERENCES w4h.users (id)
    ON DELETE CASCADE,
  CONSTRAINT user_studies_study_fk FOREIGN KEY (study_id)
    REFERENCES w4h.studies (id)
    ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_user_studies_user_id ON w4h.user_studies(user_id);
CREATE INDEX IF NOT EXISTS idx_user_studies_study_id ON w4h.user_studies(study_id);

ALTER TABLE IF EXISTS w4h.datasets
  ADD COLUMN IF NOT EXISTS study_id text COLLATE pg_catalog."default";

ALTER TABLE IF EXISTS w4h.datasets
  DROP COLUMN IF EXISTS hidden;

CREATE TABLE IF NOT EXISTS w4h.docs (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  slug text COLLATE pg_catalog."default" NOT NULL,
  title text COLLATE pg_catalog."default",
  card_title text COLLATE pg_catalog."default",
  description text COLLATE pg_catalog."default",
  action_label text COLLATE pg_catalog."default",
  doc_type text COLLATE pg_catalog."default",
  sort_order integer,
  hidden boolean,
  updated_at timestamp without time zone,
  is_feature_card boolean,
  show_in_docs boolean,
  md text COLLATE pg_catalog."default",
  html text COLLATE pg_catalog."default",
  toc jsonb,
  card_order integer,
  CONSTRAINT docs_pkey PRIMARY KEY (id),
  CONSTRAINT docs_slug_key UNIQUE (slug)
);

CREATE INDEX IF NOT EXISTS idx_docs_slug ON w4h.docs(slug);