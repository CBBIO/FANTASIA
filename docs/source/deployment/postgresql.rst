PostgreSQL and pgvector
=======================

The packaged local configuration uses host ``localhost``, port ``5432``,
database ``BioData``, user ``usuario``, and password ``clave``. These are local
development values. PostgreSQL 18 and the ``vector`` extension are required.

Create the cluster with UTF-8 encoding. A cluster initialised under
``LC_ALL=C`` uses ``SQL_ASCII``; FANTASIA forces a UTF-8 client encoding, but
``initdb --encoding=UTF8 --locale=C.UTF-8`` (or ``POSTGRES_INITDB_ARGS`` for the
official image) avoids the problem at the source. Check with
``SHOW server_encoding;``.

The ``postgres`` 18 images store data in ``/var/lib/postgresql/18/docker`` and
declare ``/var/lib/postgresql`` as the volume. Mount ``/var/lib/postgresql``, or
set ``PGDATA`` explicitly when binding ``/var/lib/postgresql/data`` (as the
Apptainer/Singularity scripts in ``deployment/`` do). Data directories created
by PostgreSQL 16 cannot be opened by 18: recreate the volume and run
``fantasia initialize`` again.

Connection checks:

.. code-block:: bash

   PGPASSWORD=clave psql -h localhost -p 5432 -U usuario -d BioData \
     -c 'SELECT 1;'
   PGPASSWORD=clave psql -h localhost -p 5432 -U usuario -d BioData \
     -c "SELECT extversion FROM pg_extension WHERE extname='vector';"

Initialization uses ``pg_restore`` and resets ``public``. The configured user
must be able to recreate the schema and create/use the vector extension. See
`PostgreSQL failures <../troubleshooting/postgresql.rst>`_ for authentication and permission failures.
