Process multiple proteomes
==========================

Run one proteome at a time on a shared GPU unless resources have been measured
for concurrency. The supplied sequential launcher accepts a config, input
directory, output base, and prefix:

.. code-block:: bash

   ./scripts/run_sequential_proteomes.sh \
     ./config/prott5_full.yaml ./data/proteomes ./lookup batch

Inspect the script and configuration before production use. For scheduler
arrays and per-job service deployment, see :doc:`/deployment/hpc_slurm`.
