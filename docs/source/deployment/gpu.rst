GPU execution
=============

Packaged configs use ``embedding.device: cuda`` and
``lookup.use_gpu: true``. Verify CUDA before a run:

.. code-block:: bash

   nvidia-smi
   poetry run python -c 'import torch; print(torch.cuda.is_available())'

Embedding and lookup are sequential within one run. Peak memory depends on the
active model/layer, sequence length, embedding batch size, reference matrix,
and lookup batch size. On out-of-memory errors, lower the relevant model
``batch_size`` for embedding or ``lookup.batch_size`` for lookup. Run models or
proteomes sequentially to avoid GPU contention.
