GPU failures
============

CUDA unavailable
   Check ``nvidia-smi`` and
   ``python -c 'import torch; print(torch.cuda.is_available())'``. Fix the
   driver/runtime or switch both embedding and lookup to CPU.

Embedding out of memory
   Lower ``embedding.models.<name>.batch_size`` and avoid concurrent models or
   jobs. Long proteins can dominate memory.

Lookup out of memory
   Lower ``lookup.batch_size``. The reference matrix, embedding dimension, and
   distance buffers determine lookup memory.

Unexpected CPU lookup
   Inspect saved config: ``embedding.device`` and ``lookup.use_gpu`` are
   independent.
