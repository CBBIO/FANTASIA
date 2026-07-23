RabbitMQ failures
=================

Connection refused or tasks never start
   Run ``docker compose ps rabbitmq``, ``docker compose logs rabbitmq``, and
   ``nc -z localhost 5672``. Correct host, port, username, and password.

Stale tasks or queues
   Stop competing FANTASIA runs, restart the dedicated broker, and keep
   ``delete_queues: true`` for normal isolated runs. Do not clear queues used by
   other projects.

Interrupted run
   Check whether a valid ``embeddings.h5`` was completed. If so, launch a new
   lookup-only run; otherwise rerun embedding with a new prefix.
