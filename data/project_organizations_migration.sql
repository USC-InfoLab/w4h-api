-- Add explicit collaborating organizations per project.
-- Safe to re-run.

BEGIN;

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

COMMENT ON TABLE w4h.project_organizations IS
  'Admin-managed collaborating organizations displayed on the project hub.';

COMMIT;
