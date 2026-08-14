# W4H API

The [Wearables for Health (W4H) Toolkit](https://infolab.usc.edu/projects/W4H/) is a suite of open-source tools for managing, analyzing, and visualizing wearable data used in health applications. This repository provides the [W4H Web API](https://github.com/USC-InfoLab/w4h-api), the web services for the [W4H Web App](https://github.com/USC-InfoLab/w4h-app).

## Setup

```bash
cp .env.example .env  #no need to edit for initial set up
npm install
npm start
```

To reset the installation and go through the initial set up, unset `DB_DATABASE` in .env.

## Dataset schema provisioning

The shared application tables currently remain in `w4h`. Creating a dataset
provisions an immutable PostgreSQL schema named `ds_<uuid>`. The
`w4h.dataset_registry` table is the only source used to resolve that name.

After `schema.sql` has initialized the database, bootstrap the privileged
NOLOGIN role once using a PostgreSQL account with `CREATEROLE` and
database-owner privileges:

```bash
# Defaults to granting w4h_platform membership to DB_USER.
# Set W4H_APP_DB_ROLE when the privileged bootstrap account is different.
W4H_APP_DB_ROLE=w4h_api npm run bootstrap:platform-role
```

Creating a dataset provisions a `ds_<uuid>` schema with `subjects` and
`dataset_tables`. CSV ingestion creates typed signal tables (`heart_rate`,
`weight`, …) in that schema. The owning team's `team_<uuid>_admin` role
receives `ALL` on that schema. Notebook migration, `w4h_global`, and the future
`w4h` to `w4h_app` rename are separate follow-up work.

Database integration tests require an explicitly enabled disposable database.
The database name must contain `test`; otherwise the test process refuses to
run:

```bash
W4H_INTEGRATION_TEST=1 DB_DATABASE=w4h_test npm test
```

## Dataset-scoped CSV ingestion

Team admins and Super admins can perform the initial immutable ingest for an
existing dataset:

```text
POST /datasets/:datasetId/ingest
Content-Type: multipart/form-data
```

The request requires `subjects_csv`, `signal_csv`, and a JSON `mapping` field.
Each CSV is limited to 10 MB. This Fitbit example maps subjects plus heart-rate
data into `subjects` and a typed `heart_rate` table in the dataset schema:

```bash
curl -X POST "http://localhost:2026/datasets/DATASET_ID/ingest" \
  -H "Authorization: Bearer $W4H_TOKEN" \
  -F "subjects_csv=@../w4h-datasets/fitbit_subjects.csv;type=text/csv" \
  -F "signal_csv=@../w4h-datasets/fitbit_heart_rate.csv;type=text/csv" \
  -F 'mapping={"device_type":"fitbit","subjects":{"id":"id","enrollment_start":"start_date","enrollment_end":"end_date"},"signal":{"signal_type":"heart_rate","subject_id":"id","timestamp":{"combine":["date","timestamp"]},"columns":[{"source":"value","target":"bpm","type":"integer"}],"sampling_hz":0.0166666667}}'
```

Supported mapped value types are `text`, `integer`, `double precision`,
`boolean`, `date`, and `timestamptz`. Source column names are validated against
the uploaded headers. Target names must be lowercase SQL-safe identifiers.

The API validates both files before loading, creates typed tables in the
dataset schema (subjects plus the mapped signal table such as `heart_rate` or
`weight`), registers them in `dataset_tables`, updates `schema_def`, and sets
the dataset to `ready`. Initial ingestion is
immutable: a `ready` dataset rejects replacement or append requests. Direct
notebook DDL into `public` remains a legacy development path and should not be
used for this flow.

## CLI

The CLI will be used to access admin and recurring tasks. Run `bin/w4h` or see launch.json for examples.

In development it is convenient to add `w4h` to the path:

```sh
mkdir ~/bin
ln -s $PWD/bin/w4h ~/bin/w4h
```

and add an export in a your `.bashrc` or `.zshrc`:

```sh
export PATH=$HOME/bin:$PATH
```

When deploying you can 

---
