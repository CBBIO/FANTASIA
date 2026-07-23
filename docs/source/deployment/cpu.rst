CPU execution
=============

CPU operation requires two YAML changes:

.. code-block:: yaml

   embedding:
     device: cpu

   lookup:
     use_gpu: false

``--device cpu`` changes the embedding device only; it does not override
``lookup.use_gpu``. Copy a packaged config, apply both settings, and run the
normal command. CPU embeddings and large reference comparisons can be much
slower than GPU execution; validate on ``config/prott5_test.yaml`` before a
complete proteome.
