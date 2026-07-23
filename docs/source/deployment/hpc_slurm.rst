HPC and Slurm
=============

The ``deployment/`` directory contains site-specific CESGA, Greisenwald, IFB,
single-job, and array templates. They encode local module, path, container, and
scheduler assumptions; review every variable before submission.

General requirements:

- mount persistent PostgreSQL data, model cache, inputs, configs, and outputs;
- give every concurrent task isolated database/broker resources or serialize
  jobs;
- use absolute paths for ``constants``, input, output, and logs;
- stage the reference dump once where possible;
- request GPU resources for configs using CUDA;
- use ``--limit_per_entry`` to set the neighbourhood size.

The legacy detailed site pages remain available at :doc:`cesga_job`,
:doc:`cesga_array_job`, :doc:`greisenwald_job`, and
:doc:`greisenwald_array_job`.
