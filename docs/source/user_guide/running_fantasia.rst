Run FANTASIA
============

Every run combines a YAML configuration with optional CLI overrides:

.. code-block:: bash

   poetry run fantasia run --config CONFIG.yaml --input INPUT \
     --prefix NAME --base_directory OUTPUT_BASE --log_path LOG_DIRECTORY

Full mode embeds a FASTA and then performs lookup. Embedding-only mode stops
after HDF5 creation; lookup-only mode accepts an existing HDF5. Each invocation
creates ``OUTPUT_BASE/experiments/NAME_<timestamp>/`` and stores the resolved
``experiment_config.yaml`` there.

Relative paths in the configuration (input FASTA, output directories) are
resolved from the working directory; use absolute paths on HPC systems. The
default ``--config`` and the ``constants.yaml`` it names are found from the
installation, so they work from any directory. At least one
model must be enabled in ``embedding.models``.
