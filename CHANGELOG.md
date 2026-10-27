# Changelog

## 4.1.2 (unreleased)

### Upgrading from 4.1.1

- PostgreSQL 18 is required: the December 2025 reference dumps were written by
  `pg_dump` 18. With Docker Compose, recreate the database volume
  (`docker compose down -v`), then run `fantasia initialize` again: a
  PostgreSQL 16 volume is not readable by 18, and the new image would start an
  empty cluster next to the old files. Outside Docker, install
  `postgresql-client-18`.
- The HPC scripts reference the `cbbio/fantasia:v4.1.2` image and versioned
  `.sif` names (`pgvector_pg18.sif`, `fantasia_v4.1.2.sif`), so images cached
  by 4.1.1 are rebuilt.
- `fantasia initialize` and `fantasia run` now exit with status 1 on failures
  that previously ended with 0. Job scripts that ignored the exit status now
  stop.

### Fixes

- PostgreSQL 18 across the stack: `postgresql-client-18` in the Docker image,
  `pgvector/pgvector:0.8.1-pg18` in Compose and the HPC scripts, and PG18 data
  paths (`PGDATA`) (#57, #61, #50).
- `fantasia initialize` exits with status 1 when the restore fails; `pg_restore`
  errors reach the log files, with a hint when the client is too old. Errors
  limited to `SET transaction_timeout` (client 18 against an older server) are
  reported as warnings, since the data is restored (#57).
- `fantasia run` checks the reference database before the embedding stage and
  stops with an explicit error when it has no embeddings, instead of skipping
  every batch and finishing without results (#61).
- Errors from `fantasia run` (including "Pipeline execution failed") now reach
  the log files; they were sent to an unconfigured logger.
- Database connections name the `psycopg2` driver explicitly. SQLAlchemy 2.1
  selects `psycopg` (v3) for bare `postgresql://` URLs, which made
  `fantasia initialize` fail on fresh installations before reaching
  `pg_restore`. Credentials with special characters are now escaped.
- Downloads of the reference dump are written to a temporary file and renamed
  when complete, so an interrupted download no longer leaves a truncated dump
  that later runs would try to restore.
- Alignment metrics skip pairs with an empty sequence instead of crashing in
  parasail, which discarded the results of the whole batch. The metrics are
  left empty, post-processing treats them as missing, and the number of
  skipped pairs is logged once (#60).
- PostgreSQL clients use UTF-8 (`PGCLIENTENCODING=UTF8`) and the deployment
  scripts create UTF-8 clusters, fixing decoding errors under `LC_ALL=C`; the
  non-ASCII character in `constants.yaml` was removed (#55).
- `embeddings_url` accepts `file://` URLs and local paths, used in place
  without copying (#54).
- The default `--config`, `constants.yaml` and `go-basic.obo` are found from the
  installation directory instead of the working directory, so the image no
  longer depends on `WORKDIR /app` (#46).
- The `embeddings_url` key may be absent (#53). It stays unset in the packaged
  configuration because `initialize` resets the database before restoring.
- The `proteins` column of `summary.csv` lists proteins in a stable order;
  previously the order could change between identical runs.
- HPC scripts:
  - stop when `fantasia initialize` fails;
  - wait until PostgreSQL and RabbitMQ accept connections (with a timeout)
    instead of sleeping a fixed time;
  - register their cleanup before starting the services and stop only their
    own services (by PID, or `pg_ctl stop`). Previously cleanup ran only at the
    end, matched an unused path or every `rabbitmq-server` on the node, and
    removed a shared `/tmp/fantasia` directory;
  - array jobs use per-task service directories, so tasks no longer delete
    each other's database.

### Added

- `fail_on_task_errors: true` in the packaged configurations. With
  protein-information-system 3.2 or later, `fantasia run` exits with status 1
  when tasks failed and were discarded or a worker process crashed, after
  writing the partial results. Older versions ignore the key.

### Documentation

- Reorganized user documentation around Getting started, User guide,
  Deployment, Reference, Explanation, Benchmarks, Performance,
  Troubleshooting, and Development.
- Added an executable first-run path, configuration/output references,
  documentation smoke checks, and a maintainer audit.
- Corrected Python/Sphinx requirements, reference URLs, distance-setting paths,
  `k` syntax, taxonomy behavior, and gzip FASTA guidance.
- Removed obsolete CAFA analysis pages and branding logo assets.
- Documented PostgreSQL 18 requirements (including the host client for the
  quick start), exit status, `delete_queues` behaviour by
  protein-information-system version, sharing one reference database (#56),
  HPC script behaviour, and troubleshooting for an empty reference database,
  `TaskFailuresError` and empty FASTA records.

## 4.1.1

- See the Git tag and repository history for the release changes.
