-- Add per-project launcher defaults (IDE / Jupyter URLs). Run after projects_acl.sql.
-- Safe to re-run.

BEGIN;

ALTER TABLE w4h.projects
  ADD COLUMN IF NOT EXISTS launchers jsonb NOT NULL DEFAULT '{}'::jsonb;

COMMENT ON COLUMN w4h.projects.launchers IS
  'JSON: jupyterBaseUrl, rstudioBaseUrl, vscodeTemplate — used by project hub launch buttons.';

COMMIT;
