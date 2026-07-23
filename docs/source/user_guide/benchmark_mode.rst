Benchmark and leakage-control mode
==================================

A controlled benchmark differs from routine annotation. Retrieve several
candidates, exclude explicit query taxa before distance computation, and apply
sequence-identity filtering to the retrieved neighbourhood:

.. code-block:: bash

   poetry run fantasia run --config ./config/prott5_full.yaml \
     --input ./data/mouse.fasta --prefix mouse_k5 \
     --base_directory ./lookup --log_path ./lookup/logs \
     --limit_per_entry 5 \
     --taxonomy_ids_to_exclude 10090,10091,57486

Taxonomy matching is exact; ``get_descendants`` is deprecated and disabled, and any true value is rejected. Post-hoc filtering:

.. code-block:: bash

   python scripts/filter_raw_results_by_identity.py \
     lookup/experiments/mouse_k5_<timestamp>/raw_results/prot-t5/layer_0 \
     --threshold 0.90 --threshold 0.95 \
     --output-dir lookup/filtered_mouse

The filter selects the best surviving donor by ``reliability_index``. With a
small ``k``, some queries can lose all donors; increasing ``k`` may provide
additional candidates. Filtering changes the evaluation set and is not an
accuracy guarantee.
