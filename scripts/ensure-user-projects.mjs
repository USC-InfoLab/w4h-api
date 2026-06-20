import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
dotenv.config({ path: path.join(__dirname, '..', '.env') });

const { query } = await import('../db.js');

async function projectsIdType() {
  const r = await query(
    `SELECT c.data_type, c.udt_name
     FROM information_schema.columns c
     WHERE c.table_schema = 'w4h'
       AND c.table_name = 'projects'
       AND c.column_name = 'id'`
  );
  if (r.rowCount === 0) {
    throw new Error('Table w4h.projects not found. Apply schema (data/schema.sql and/or data/projects_acl.sql) first.');
  }
  const { data_type: dataType, udt_name: udtName } = r.rows[0];
  if (udtName === 'uuid' || dataType === 'uuid') return 'uuid';
  return 'text';
}

async function main() {
  const idSqlType = await projectsIdType();
  const projectIdCol =
    idSqlType === 'uuid'
      ? 'project_id uuid NOT NULL'
      : 'project_id text COLLATE pg_catalog."default" NOT NULL';

  await query(`
    CREATE TABLE IF NOT EXISTS w4h.user_projects (
      user_id integer NOT NULL,
      ${projectIdCol},
      CONSTRAINT user_projects_pkey PRIMARY KEY (user_id, project_id),
      CONSTRAINT user_projects_user_fk FOREIGN KEY (user_id)
        REFERENCES w4h.users (id)
        ON DELETE CASCADE,
      CONSTRAINT user_projects_fk FOREIGN KEY (project_id)
        REFERENCES w4h.projects (id)
        ON DELETE CASCADE
    )
  `);

  await query(`
    CREATE INDEX IF NOT EXISTS idx_user_projects_user_id ON w4h.user_projects(user_id)
  `);
  await query(`
    CREATE INDEX IF NOT EXISTS idx_user_projects_project_id ON w4h.user_projects(project_id)
  `);

  await query(`
    DO $$
    BEGIN
      IF EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'w4h' AND table_name = 'user_projects'
      ) THEN
        INSERT INTO w4h.user_projects (user_id, project_id)
        SELECT us.user_id, p.id
        FROM w4h.user_projects us
        INNER JOIN w4h.projects p ON p.id::text = us.project_id::text
        ON CONFLICT (user_id, project_id) DO NOTHING;
      END IF;
    END
    $$
  `);

  console.log(`OK: w4h.user_projects exists (project_id ${idSqlType}); backfilled from user_projects if applicable.`);
}

main()
  .then(() => process.exit(0))
  .catch((err) => {
    console.error(err);
    process.exit(1);
  });
