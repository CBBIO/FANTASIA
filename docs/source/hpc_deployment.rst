.. _fantasia_hpc_deployment:

=======================================
HPC Deployment Guide
=======================================

.. contents:: Table of Contents
   :depth: 2

Step 1: Connect to the HPC via VPN
==================================

To access the HPC system, you must first connect to the private network using a VPN.

**Instructions:**

1. Open the VPN client configured on your system.
2. Enter the credentials provided by the system administrator.
3. Connect to the VPN and verify that the connection is successful.
4. Once connected, open a terminal and test the connection to the HPC:

   .. code-block:: console

      ssh user@hpc.domain.com

   Replace ``user`` with your username and ``hpc.domain.com`` with the HPC server address.

5. If this is your first time connecting, accept the host key by typing ``yes`` when prompted.

Step 2: Reserve Resources on the Cluster
========================================

Once inside the cluster, you need to reserve resources to execute the pipeline.

**Command Used:**

.. code-block:: console

   salloc --partition=vision --gres=gpu:1 --cpus-per-task=64 --mem=128G --time=03-00:00:00

**Command Breakdown:**

- ``--partition=vision``: Specifies the partition to use.
- ``--gres=gpu:1``: Reserves 1 GPU for the job.
- ``--cpus-per-task=64``: Allocates 64 CPUs for the job.
- ``--mem=128G``: Allocates 128 GB of RAM.
- ``--time=03-00:00:00``: Sets a maximum runtime of 3 days for the job.

**Expected Output:**

.. code-block:: console

   salloc: Pending job allocation 5134882
   salloc: job 5134882 queued and waiting for resources
   salloc: job 5134882 has been allocated resources
   salloc: Granted job allocation 5134882
   salloc: Waiting for resource configuration
   salloc: Nodes vision-04 are ready for job

Step 3: Connect to the Assigned Node
====================================

After resources are allocated, connect to the assigned node for pipeline execution.

**Command:**

.. code-block:: console

   ssh vision-04

**Notes:**

- Replace ``vision-04`` with the node name shown in the ``salloc`` output.
- Once connected to the node, you are ready to configure and execute the pipeline.

Step 3.1: Create Separate Screen Sessions
=========================================

To keep each service (RabbitMQ, PostgreSQL, and FANTASIA) isolated, it is recommended to run each one in its own `screen` session:

1. **Create/attach a session for RabbitMQ**:

   .. code-block:: console

      screen -S rabbitmq

   - This opens (or attaches to) a screen session named ``rabbitmq``.
   - To detach from it (but leave it running), press ``Ctrl + A`` followed by ``D``.

2. **Create/attach a session for PostgreSQL**:

   .. code-block:: console

      screen -S postgres

3. **Create/attach a session for FANTASIA**:

   .. code-block:: console

      screen -S fantasia

**Managing Screen Sessions:**

- To detach from a session while it keeps running, press ``Ctrl + A`` then ``D``.
- To reattach to a session by name:

  .. code-block:: console

     screen -r rabbitmq
     screen -r postgres
     screen -r fantasia

This allows you to run each service separately, check logs independently, and ensure that if any service crashes or needs debugging, it will not interrupt the others.

Step 4: Load Required Modules
=============================

Before running the pipeline (and/or building containers), load the necessary modules into the node environment.
These commands can be run in any session, but you can typically run them once in your main terminal or in each session if needed:

.. code-block:: console

   module load gcc/13.2.0
   module load hdf5/1.14.0
   module load singularity/3.11.3
   module load cuda/12.0.0
   module load openmpi/4.1.1

**Notes:**

- Ensure the loaded module versions are compatible with the pipeline.
- If a module is unavailable, contact the HPC system administrator for assistance.


Step 5: Build and Configure the Singularity Container for PostgreSQL in RAM
===========================================================================

.. note::
   It is recommended that you run **all PostgreSQL-related commands** inside
   the ``postgres`` screen session created in Step 3.1. This ensures PostgreSQL
   remains isolated from other services.

5.1. Build the Container
------------------------

Use the following command to build a Singularity container from the official pgvector image:

.. code-block:: console

   singularity build pgvector.sif docker://pgvector/pgvector:pg16

5.2. Create Directories in /dev/shm
-----------------------------------

Since we are running PostgreSQL entirely in RAM, create separate directories in ``/dev/shm`` (a tmpfs filesystem):

.. code-block:: console

   mkdir -p /dev/shm/pgvector_data
   mkdir -p /dev/shm/pgvector_temp

**Why /dev/shm?**
- ``/dev/shm`` is a volatile filesystem stored in memory. Data here offers very fast I/O, but **all data will be lost** when the job ends or the node reboots.
- Plan a backup/restore strategy if you need to preserve important results.

5.3. Initialize the Database in RAM
-----------------------------------

Next, initialize a new PostgreSQL cluster within the RAM-based directory:

.. code-block:: console

   singularity exec pgvector.sif initdb -D /dev/shm/pgvector_data

5.4. Start the PostgreSQL Server in RAM
---------------------------------------

Launch the PostgreSQL server, pointing to the RAM directories:

.. code-block:: console

   singularity exec pgvector.sif postgres \
       -D /dev/shm/pgvector_data \
       -k /dev/shm/pgvector_temp

**Tips**:
- Run this inside your ``postgres`` screen session so that PostgreSQL continues running even if you detach (Ctrl +A, D).
- The ``-k /dev/shm/pgvector_temp`` argument configures PostgreSQL to listen on a Unix domain socket located in ``/dev/shm``, which is handy for local connections within the same HPC node.

5.5. Verify and Configure Permissions
-------------------------------------

In another terminal (or by reattaching the same screen session), test connectivity:

.. code-block:: console

   singularity exec pgvector.sif psql -h /dev/shm/pgvector_temp -d postgres

If the connection succeeds, your PostgreSQL instance is live in RAM.


Step 6: Configure PostgreSQL User, Database, and Restart the Server
===================================================================

Once you have verified the service by running:

.. code-block:: console

   singularity exec pgvector.sif psql -h /dev/shm/pgvector_temp -d postgres

you will be inside the PostgreSQL interactive shell (``psql``). From there, you can create users, databases, and adjust settings as needed.

6.1. Create a User and Database
-------------------------------

Run these commands directly in the PostgreSQL shell:

.. code-block:: sql

   CREATE USER usuario WITH PASSWORD 'clave' SUPERUSER;
   CREATE DATABASE "BioData" OWNER usuario;
   GRANT ALL PRIVILEGES ON DATABASE "BioData" TO usuario;

   ALTER SYSTEM SET shared_buffers = '16GB';
   ALTER SYSTEM SET effective_cache_size = '64GB';
   ALTER SYSTEM SET work_mem = '256MB';

- Replace ``usuario`` and ``clave`` with your desired username and password.
- The above `ALTER SYSTEM` commands modify server parameters (for example, memory settings).

When finished, exit the PostgreSQL client:

.. code-block:: console

   \q

6.2. Restarting PostgreSQL
--------------------------

Some configuration changes require a server restart to take effect. In your ``postgres`` screen session (where the server is running), you can stop and start PostgreSQL as follows:

1. **Stop the PostgreSQL Server**:

   .. code-block:: console

      singularity exec pgvector.sif pg_ctl -D /dev/shm/pgvector_data restart

2. **Start the PostgreSQL Server**:

   .. code-block:: console

      singularity exec pgvector.sif pg_ctl -D /dev/shm/pgvector_data start -l /dev/shm/pgvector_data/pg_log.txt


With the server restarted, your new settings and user/database configuration are now active.

Step 7: Build and Run RabbitMQ
==============================

Switch to (or create) the ``rabbitmq`` screen session for these commands:

1. **Build the Singularity container for RabbitMQ**:

   .. code-block:: console

      singularity build rabbitmq.sif docker://rabbitmq:management

2. **Create the data directory** in your home (or local storage):

   .. code-block:: console

      mkdir -p ~/rabbitmq_data

3. **Start the RabbitMQ server** within the container:

   .. code-block:: console

      singularity exec --bind ~/rabbitmq_data:/var/lib/rabbitmq rabbitmq.sif rabbitmq-server

You can leave RabbitMQ running in this screen session. Detach with ``Ctrl + A, D`` if desired.

Step 8: Build and Configure the Singularity Container for FANTASIA
===================================================================

This step can be done in your main terminal or in the ``fantasia`` session:

**Build the Container:**

.. code-block:: console

   singularity build fantasia.sif docker://frapercan/fantasia

**Notes:**

- Ensure you have permissions to build containers in the HPC environment.

Step 9: Run FANTASIA
====================

Finally, in the ``fantasia`` screen session, run the FANTASIA pipeline:

.. code-block:: console

   singularity exec --bind ~/fantasia:/fantasia fantasia.sif python3 -m fantasia.main run \
      --fasta /fantasia/fantasia/input/PMET_1_tardigrade_subsample.fasta \
      --prefix PMET_1_tardigrade_subsample.fasta \
      --length_filter 50000000 \
      --redundancy_filter 0. \
      --sequence_queue_package 1000 \
      --esm \
      --prost \
      --prot \
      --distance_threshold 1:1.2,2:0.7,3:0.7 \
      --batch_size 1:32,2:32,3:32

**Explanation of the Command**

- ``--bind ~/fantasia:/fantasia``: Mounts your local ``~/fantasia`` directory inside the container at ``/fantasia``.
- ``python3 -m fantasia.main run``: Executes the main ``run`` function of FANTASIA.
- **Arguments**:
  - ``--fasta``: Specifies the input FASTA file.
  - ``--prefix``: Sets a prefix for output files.
  - ``--length_filter``: Filters out sequences longer than 50,000,000.
  - ``--redundancy_filter``: Specifies the redundancy threshold (0.0).
  - ``--sequence_queue_package``: Determines the size of sequence batches (1000 sequences per package).
  - ``--esm``, ``--prost``, ``--prot``: Enables different processing modes or models in the pipeline.
  - ``--distance_threshold``: Sets thresholds for distances across different embedding types.
  - ``--batch_size``: Specifies batch sizes for different embedding types.

**Output**

- Results will be stored in the directory mounted to ``/fantasia``.
- You should see log messages in the terminal indicating the pipeline’s progress.

----

By using three separate ``screen`` sessions—one for RabbitMQ, one for PostgreSQL (in RAM), and one for FANTASIA—you keep each service isolated, simplifying monitoring and troubleshooting. Running PostgreSQL in `/dev/shm` can provide a major performance boost, but **note** that all data will be lost when the HPC job ends or the node reboots. Make sure to export or back up any results before terminating your job.
