Docker and Docker Compose
=========================

The Python process normally runs on the host while Compose supplies PostgreSQL
and RabbitMQ:

.. code-block:: bash

   docker compose up -d
   docker compose ps
   docker compose logs postgres
   docker compose logs rabbitmq

The service names are ``fantasia-postgres`` and ``fantasia-rabbitmq``. Local
ports are 5432, 5672, and 15672. Data persist in named Docker volumes.

.. warning::

   ``docker compose down -v`` deletes those volumes, including the restored
   reference database. Use ``docker compose down`` when data must persist.

The repository Dockerfile builds a Python 3.12 CLI image containing PostgreSQL
client 16 and MMseqs2; it does not replace the database or broker services.
