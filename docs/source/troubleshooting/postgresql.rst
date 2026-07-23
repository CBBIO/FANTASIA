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
   at an empty database or a benchmark-output archive.
