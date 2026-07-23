# FANTASIA documentation audit

Audit date: 2026-07-23. Audited release checkout: `fantasia-4.1.1` at
`c3ddd2a`. The source code, packaged YAML, tests, scripts, container files, and
CI workflows were treated as authoritative. This is a maintainer document, not
a user tutorial.

| Documentation claim | Previous source | Implementation source | Status | Required correction |
|---|---|---|---|---|
| A first run is `fantasia initialize` followed by `fantasia run` with no arguments. | Old RTD experiment setup | `fantasia/src/helpers/parser.py`, packaged configs | Incomplete | Show an explicit config, input, output base, log path, and reference URL. |
| Python 3.11 builds are supported. | `.readthedocs.yml` | `pyproject.toml`, CI | Contradictory | Use Python 3.12; package constraint is `>=3.12,<3.13`. |
| Sphinx 7.2.6 builds with Shibuya 2025.9.24. | `pyproject.toml` | Strict local build | Contradictory | Use Sphinx 8.x; declare only documentation build dependencies in `docs/requirements.txt`. |
| Users must decompress gzip FASTA inputs. | Deployment notes | `fantasia/src/helpers/parser.py`, `tests/test_fasta_input.py` | Obsolete | Plain and gzip-compressed FASTA are supported directly. |
| Lookup distance is configured at `embedding.distance_metric`. | Old method pages | `fantasia/config.yaml`, `fantasia/src/lookup.py` | Obsolete | Document `lookup.distance_metric`; note legacy path support only in reference material. |
| `--k 5` is a valid CLI option. | Old Slurm page | CLI parser | Incorrect | Use `--limit_per_entry 5`. |
| Annotation defaults to `k=5`. | Old database/search page | `fantasia/config.yaml` | Incorrect | Repository default is `lookup.limit_per_entry: 1`. |
| Reference records 17167843 and 17151847 are current. | Old database/search page and CLI help | `fantasia/config.yaml`, Zenodo HTTP validation | Obsolete | Use records 17795871 and 17793273 and their current filenames. |
| `data/` and `lookup/` already exist after cloning. | Implied by examples | Repository file inventory | Incomplete | Explicitly create them with `mkdir -p`; both are local working folders. |
| Initialization merely “indexes embeddings.” | Old quickstart | `fantasia/main.py`, `helpers.load_dump_to_db` | Misleading | Explain that a PostgreSQL dump is downloaded and restored and `public` is reset. |
| PostgreSQL and RabbitMQ are optional for a full run. | Scattered installation prose | `check_services`, embedder and lookup code | Incorrect | Document both as required services for the complete Full workflow. |
| MMseqs2 is always required. | Old installation prerequisites | Config default and lookup code | Incomplete | It is required only when redundancy masking is enabled. |
| `parasail` is an external executable. | Reviewer concern / older prose | `pyproject.toml`, lookup imports | Incorrect | It is a Python dependency used for alignment metrics. |
| Taxonomy filtering happens after nearest-neighbour retrieval. | `fantasia/config.yaml` comment | Reference loading in `lookup.py` | Contradictory | State that exact-ID filters restrict eligible references before distance computation. |
| Taxonomy descendant expansion is supported. | Historical CLI/docs | `main.load_and_merge_config` | Obsolete | `get_descendants` is disabled; list exact IDs explicitly. |
| `summary.csv` is one row per protein. | Ambiguous README wording | `lookup.post_processing` | Incorrect | It is an accession-by-GO summary, further separated by model/layer evidence. |
| Raw results are one consolidated file. | Ambiguous user documentation | `lookup.store_entry` | Incorrect | They are per-query CSV shards under model/layer directories; use the merge utility when required. |
| `final_score` is a probability. | Possible interpretation of old output docs | Post-processing code | Misleading | It is a configuration-dependent heuristic ranking score. |
| A model name fixes exact weights forever. | Old model list | Hugging Face loading/caches | Incomplete | Record revision, serialization, checksums, software environment, and run config. |
| Query sequences are capped at 2,000 aa by default. | Historical benchmark context | Packaged config | Obsolete | `embedding.max_sequence_length: 0` means no query truncation. Model limits may still cause failures. |
| CPU mode requires only `--device cpu`. | CLI examples | Packaged config and lookup initialization | Incomplete | Also set `lookup.use_gpu: false` in YAML. |
| All five models should be enabled in one routine run. | Feature-oriented README | Runtime guidance/tests | Incomplete | A single model is the simplest workflow; sequential model runs make resource use easier to diagnose. |
| Read the Docs should follow the manuscript structure. | Previous `index.rst` | User task analysis | Obsolete | Replace primary navigation with Getting started, User guide, Deployment, Reference, Explanation, Benchmarks, Performance, Troubleshooting, Development, and Citation. |
| The repository contains `poetry.lock`, `examples/`, `CHANGELOG.md`, `CONTRIBUTING.md`, and `CITATION.cff`. | Requested audit inventory | Repository file inventory | Missing | Do not claim they exist; add maintainer files where useful and record the absence of a lock file. |
| The minimal Full workflow is lightweight. | “5 min” homepage badge | Reference dump sizes and model/runtime needs | Misleading | State that Full requires services and a ~3.1 GB reference download; direct lightweight users to FANTASIA-Lite. |

## Validation evidence

- `python -m py_compile fantasia/src/helpers/parser.py fantasia/src/lookup.py`
- CLI help inspected from `/home/arojas/anaconda3/envs/fantasia-py312`.
- Focused execution-mode and gzip FASTA tests: 4 passed.
- Strict Sphinx 8.2.3 build with `-W --keep-going`: passed.
- Current final-layer and multilayer Zenodo file URLs: HTTP 200; reported
  download sizes approximately 3.1 GB and 17.1 GB.

The full database restore, model download, GPU embedding, and end-to-end lookup
were not repeated for this documentation audit because they require large
external data and services. Existing recorded integration runs were used only
to corroborate paths and outputs; static validation is not presented as a new
scientific run.
