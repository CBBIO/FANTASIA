RabbitMQ
========

RabbitMQ carries embedding tasks. The local service listens on port 5672; its
management interface is at ``http://localhost:15672``. The packaged local user
is ``guest`` with password ``guest``.

.. code-block:: bash

   docker compose ps rabbitmq
   docker compose logs rabbitmq
   nc -z localhost 5672 && echo 'RabbitMQ port reachable'

``delete_queues: true`` removes stale queues at startup, through the
management API on ``rabbitmq_port_http``. What it deletes depends on the
installed protein-information-system:

- 3.2 or later: only FANTASIA's own queues (``sequenceembedder_*`` and
  ``embeddinglookup_*``).
- 3.1.x: *every* queue on the broker, including other applications' queues and
  their pending messages.

Queue names are fixed per model, not per run, so two FANTASIA runs on the same
broker always share queues. Launch runs sequentially, or give each run its own
broker.

Several users or runs can share one PostgreSQL reference database: lookup only
reads from it. Run ``fantasia initialize`` once, since it resets the
``public`` schema, and skip it in later runs (leave ``embeddings_url`` empty or
do not call ``initialize``).
