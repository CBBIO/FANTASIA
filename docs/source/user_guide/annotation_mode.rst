Annotation mode
===============

Annotation mode is intended for an unknown proteome not deliberately present
in the reference set. Use cosine distance, ``k=1``, and no self-exclusion-style
filter:

.. code-block:: bash

   poetry run fantasia run --config ./config/prott5_full.yaml \
     --input ./data/my_proteome.faa.gz --prefix my_annotation \
     --base_directory ./lookup --log_path ./lookup/logs \
     --limit_per_entry 1

Do not interpret embedding proximity or ``final_score`` as a calibrated
probability. Inspect the donor, evidence, and alignment fields when evaluating
individual assignments.
