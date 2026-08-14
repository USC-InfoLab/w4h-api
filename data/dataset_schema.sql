CREATE TABLE IF NOT EXISTS {{schema}}.subjects (
  id text PRIMARY KEY,
  external_id text UNIQUE,
  enrollment_start date,
  enrollment_end date,
  created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS {{schema}}.dataset_tables (
  id bigserial PRIMARY KEY,
  table_name text NOT NULL UNIQUE,
  signal_type text NOT NULL UNIQUE,
  columns_def jsonb NOT NULL DEFAULT '[]'::jsonb,
  created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP
);

DROP TABLE IF EXISTS {{schema}}.dataset_files;
