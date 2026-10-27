PostgreSQL failures
===================

Connection refused
   Diagnose with ``docker compose ps postgres`` and ``nc -z localhost 5432``.
   Start the service or correct ``DB_HOST``/``DB_PORT``.

Authentication failed
   Ensure YAML credentials match the deployed database. Test with ``psql``.

Vector extension missing
   Query ``pg_extension`` as shown in `PostgreSQL and pgvector <../deployment/postgresql.rst>`_. Use a
   pgvector-enabled server and a user allowed to create the extension.

Schema permission denied
   The initializer recreates ``public``. Use a dedicated database whose owner
   is the configured user, or have an administrator grant schema ownership and
   CREATE/USAGE rights.

Missing reference tables
   Confirm initialization completed and inspect its log. Do not point lookup
   at an empty database or a benchmark-output archive. ``fantasia run`` stops
   with "The reference database contains no embeddings" in that case.

``pg_restore: error: unsupported version (1.16) in file header``
   The dump was written by PostgreSQL 18 and the ``pg_restore`` in use is
   older. Use the FANTASIA 4.1.2 image or install ``postgresql-client-18``.

``UnicodeDecodeError: 'ascii' codec can't decode byte``
   The cluster was created with ``SQL_ASCII`` (for example under
   ``LC_ALL=C``). FANTASIA 4.1.2 sets ``PGCLIENTENCODING=UTF8``; recreate the
   cluster with ``--encoding=UTF8`` for a permanent fix.
