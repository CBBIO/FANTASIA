Troubleshooting
===============

Start with the timestamped log, ``experiment_config.yaml``, and
``model_provenance.yaml`` in the experiment directory. The log contains the
traceback for fatal errors; the two YAML files establish the effective settings
and model/software versions used by that run.

Fast diagnosis
--------------

.. list-table::
   :header-rows: 1

   * - Symptom
     - First page to check
   * - Installation or import failure
     - `installation troubleshooting <installation.rst>`_
   * - Database connection, pgvector, or initialization failure
     - `PostgreSQL troubleshooting <postgresql.rst>`_
   * - Queue, publisher, consumer, or connection refusal
     - `RabbitMQ troubleshooting <rabbitmq.rst>`_
   * - CUDA out of memory or device mismatch
     - `GPU troubleshooting <gpu.rst>`_
   * - FASTA parsing, compressed input, or identifier problem
     - `input-file troubleshooting <input_files.rst>`_
   * - Missing, empty, or confusing result files
     - `results troubleshooting <results.rst>`_

Useful first checks
-------------------

Run these from the repository root and compare the paths with the saved
configuration:

.. code-block:: bash

   docker compose ps
   nvidia-smi
   df -h .
   tail -n 100 ./lookup/logs/*.log

Do not delete a failed experiment before preserving its log and both YAML
metadata files. For lookup-only runs, also confirm that the supplied
``embeddings.h5`` was produced with the same model and layer as the reference
database.

Troubleshooting topics
----------------------

.. toctree::
   :maxdepth: 1

   installation
   postgresql
   rabbitmq
   gpu
   input_files
   results
