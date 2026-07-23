Embeddings
==========

FANTASIA delegates tokenization, model loading, and representation extraction
to model-specific PIS backends. The YAML selects models, output-relative
layers, model batch size, device, and an optional pre-embedding truncation
length. ``max_sequence_length: 0`` means FANTASIA does not truncate; it does not
override model-specific limits.

Layer index 0 is the final/output representation. Intermediate layers require
a multilayer reference database. Query vectors are reusable only with matching
model identity, dimension, and layer conventions.
