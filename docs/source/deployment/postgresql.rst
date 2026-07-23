PostgreSQL and pgvector
=======================

The packaged local configuration uses host ``localhost``, port ``5432``,
database ``BioData``, user ``usuario``, and password ``clave``. These are local
development values. PostgreSQL 16 and the ``vector`` extension are required.

Connection checks:

.. code-block:: bash

   PGPASSWORD=clave psql -h localhost -p 5432 -U usuario -d BioData \
     -c 'SELECT 1;'
   PGPASSWORD=clave psql -h localhost -p 5432 -U usuario -d BioData \
     -c "SELECT extversion FROM pg_extension WHERE extname='vector';"

Initialization uses ``pg_restore`` and resets ``public``. The configured user
must be able to recreate the schema and create/use the vector extension. See
:doc:`/troubleshooting/postgresql` for authentication and permission failures.
