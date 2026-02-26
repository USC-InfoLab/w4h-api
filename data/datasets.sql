-- w4h.datasets tbale
-- DROP TABLE IF EXISTS w4h.datasets;

CREATE TABLE IF NOT EXISTS w4h.datasets
(
    id text COLLATE pg_catalog."default" NOT NULL,
    study_id text COLLATE pg_catalog."default",
    name text COLLATE pg_catalog."default",
    description text COLLATE pg_catalog."default",
    CONSTRAINT datasets_pkey1 PRIMARY KEY (id)
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS w4h.datasets
    OWNER to xtownla;

REVOKE ALL ON TABLE w4h.datasets FROM xtown_consumers;

GRANT ALL ON TABLE w4h.datasets TO xtown_admins WITH GRANT OPTION;

GRANT SELECT ON TABLE w4h.datasets TO xtown_consumers;

GRANT ALL ON TABLE w4h.datasets TO xtown_developers;