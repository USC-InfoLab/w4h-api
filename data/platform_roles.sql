-- Run once as a PostgreSQL role with CREATEROLE and database-owner privileges.
-- The configured W4H application role is granted membership so the provisioning
-- service can use SET LOCAL ROLE for its narrowly scoped DDL transaction.
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'w4h_platform') THEN
    CREATE ROLE w4h_platform NOLOGIN CREATEROLE;
  END IF;
END
$$;

GRANT USAGE ON SCHEMA w4h TO w4h_platform;
GRANT REFERENCES ON w4h.datasets TO w4h_platform;

DO $$
DECLARE
  app_role text := COALESCE(
    NULLIF(current_setting('w4h.app_role', true), ''),
    current_user
  );
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = app_role) THEN
    RAISE EXCEPTION 'W4H application role % does not exist', app_role;
  END IF;
  EXECUTE format(
    'GRANT CREATE ON DATABASE %I TO w4h_platform',
    current_database()
  );
  EXECUTE format('GRANT w4h_platform TO %I', app_role);
END
$$;
