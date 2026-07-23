Deployment
==========

FANTASIA Full combines a Python application with PostgreSQL/pgvector,
RabbitMQ, protein-language-model files, and optionally CUDA. For a local GPU
workstation, Docker Compose is the shortest supported deployment; CPU and HPC
setups need additional planning.

Select a deployment
-------------------

- **Local workstation with NVIDIA GPU:** :doc:`docker` and :doc:`gpu`.
- **Existing services:** configure :doc:`postgresql` and :doc:`rabbitmq`.
- **CPU-only host:** :doc:`cpu`; expect substantially longer embedding times.
- **Generic Slurm cluster:** :doc:`hpc_slurm`.
- **Test individual services:** use the PostgreSQL and RabbitMQ pages before
  diagnosing pipeline code.

Before a production run, verify:

.. code-block:: bash

   docker compose ps
   nvidia-smi
   df -h .

The application host must be able to reach both services and must have enough
disk for the selected reference database, Hugging Face model cache, query
embeddings, and result files. Do not initialize FANTASIA against a PostgreSQL
database containing unrelated data: initialization replaces its ``public``
schema.

Deployment guides
-----------------

.. toctree::
   :maxdepth: 1

   docker
   postgresql
   rabbitmq
   gpu
   cpu
   hpc_slurm
