Validate and inspect a first result
===================================

Find the newest tutorial run and verify the main table:

.. code-block:: bash

   experiment=$(find ./lookup/experiments -maxdepth 1 -type d \
     -name 'first_search_*' | sort | tail -n 1)
   test -f "$experiment/experiment_config.yaml"
   test -s "$experiment/embeddings.h5"
   test -s "$experiment/summary.csv"
   head -n 5 "$experiment/summary.csv"

``summary.csv`` is the principal post-processed accession-by-GO table. A query
can occupy several rows because it can receive several GO terms. The raw files
under ``raw_results/<model>/layer_<n>/`` retain individual donor-supported GO
rows and should be used when donor identity, embedding distance, or alignment
metrics matter.

The raw CSVs are sharded **per query protein**. They are not a consolidated
proteome table. To concatenate every original row for one model and layer:

.. code-block:: bash

   python scripts/merge_raw_results.py \
     "$experiment/raw_results/prot-t5/layer_0" \
     --output "$experiment/raw_results/prot-t5/layer_0_merged.csv" \
     --add-source-file

Run the command separately for each model/layer directory. It performs no
min/max/mean aggregation; ``--add-source-file`` records the source shard.

Keep ``experiment_config.yaml`` and the log with any shared result. See
`Output files <../reference/output_files.rst>`_ for the complete output tree and column
interpretation.
