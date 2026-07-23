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

Run from the repository root when using relative paths such as
``./fantasia/constants.yaml``. Use absolute paths on HPC systems. At least one
model must be enabled in ``embedding.models``.
