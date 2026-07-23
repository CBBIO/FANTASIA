RabbitMQ
========

RabbitMQ carries embedding tasks. The local service listens on port 5672; its
management interface is at ``http://localhost:15672``. The packaged local user
is ``guest`` with password ``guest``.

.. code-block:: bash

   docker compose ps rabbitmq
   docker compose logs rabbitmq
   nc -z localhost 5672 && echo 'RabbitMQ port reachable'

``delete_queues: true`` removes stale queues during normal startup/teardown.
Concurrent runs sharing one broker can still interfere through resources and
queue naming; prefer sequential launches unless the deployment has been
designed for isolation.
