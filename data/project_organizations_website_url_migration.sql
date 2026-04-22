-- Store org homepage as website_url (not "url") — some clients/JSON paths mishandle a nested "url" key.
-- Safe to re-run.

BEGIN;

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

COMMENT ON COLUMN w4h.project_organizations.website_url IS
  'Optional organization homepage; shown as a link on the project hub.';

COMMIT;
