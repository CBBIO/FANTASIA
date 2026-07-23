Reproduce published workflows
=============================

Use the configs and scripts under ``tests/benchmark/`` as release-specific
templates. Reproduction requires the same input checksum, software commit,
reference release, model revisions, layers, distance, k, taxonomy exclusions,
sequence-length policy, hardware/software environment, and post-processing.

Do not substitute current upstream model weights or a newer reference release
and call the run exact reproduction. Store new outputs under a new prefix and
compare manifests before interpreting numerical differences.
