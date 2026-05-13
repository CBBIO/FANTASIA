# FANTASIA Batch Utilities

Utilities to prepare and execute **FANTASIA** experiments in batches, with traceability at the level of:

- **proteins filtered by length**
- **batch**
- **model**
- **experiment / session**
- **internal FANTASIA logs**

This README mainly documents two scripts:

1. `sort_and_batch_fasta_by_length_v2.py`
2. `run_fantasia_batches_with_checkpoint_REVISED_v6.py`

---

## Workflow Goal

The recommended workflow is:

1. Take a large FASTA file (for example, UniRef50 representatives).
2. Filter sequences by a minimum length threshold.
3. Save traceability for removed sequences.
4. Implicitly sort the retained sequences from shortest to longest.
5. Build homogeneous batches of fixed size (for example, 500,000 proteins per batch).
6. Run **FANTASIA** on those batches, **model by model**.
7. Store results and logs in a clear hierarchical structure for downstream analysis.

---

# 1. `sort_and_batch_fasta_by_length_v2.py`

## What It Does

This script:

- reads a FASTA / FA / FASTA.GZ file in *streaming* mode;
- computes the length of each sequence;
- removes sequences below a configurable minimum length;
- stores traceability for removed sequences;
- keeps valid sequences ordered from shortest to longest;
- generates batches of configurable size;
- produces summary files in JSON and CSV format.

---

## Main Outputs

Inside `--output-dir`, the script typically creates:

- `length_sorted_batch_summary.json`
- `length_sorted_batch_manifest.csv`
- `filtered_out_below_min_length.fasta` *(or the configured name)*
- `filtered_out_below_min_length_manifest.csv` *(or the configured name)*
- batch files such as:
  - `uniref50_sorted_batch_00001.fasta`
  - `uniref50_sorted_batch_00002.fasta`
  - etc.

---

## Traceability of Filtered Sequences

Unless explicitly disabled with `--no-write-filtered-trace`, the script saves:

### Filtered FASTA
Contains the full sequences that were discarded because of their length.

### Filtered CSV Manifest
Contains one row per removed sequence with:

- `identifier`
- `full_header`
- `sequence_length`
- `filter_reason`

This makes it possible to know exactly which sequences were discarded and why.

---

## Important Parameters

### Input / Output

- `--input`  
  Path to the input FASTA file.

- `--output-dir`  
  Directory where batches and summaries will be written.

### Filtering and Batching

- `--batch-size`  
  Number of proteins per batch.  
  **Default:** `500000`

- `--min-length-kept`  
  Minimum length to keep.  
  **Default:** `51`

  > If you use `51`, all sequences of **50 aa or less** are removed.

- `--report-threshold`  
  Lets you report how many sequences are below or exactly equal to specific thresholds.  
  Typical values are `30` and `50`.

### Filtering Traceability

- `--filtered-fasta-name`
- `--filtered-manifest-name`
- `--no-write-filtered-trace`

### Other

- `--output-prefix`
- `--report-only`
- `--keep-temp`
- `--overwrite`
- `--args-file`

---

## Example: Direct Usage

```bash
python sort_and_batch_fasta_by_length_v2.py \
  --input /home/avelasco/fantasia/data/uniref50_representatives.fasta \
  --output-dir /home/avelasco/fantasia/uniref50_batches_sorted_500k \
  --batch-size 500000 \
  --min-length-kept 51 \
  --output-prefix uniref50_sorted \
  --filtered-fasta-name filtered_out_below_51aa.fasta \
  --filtered-manifest-name filtered_out_below_51aa_manifest.csv \
  --overwrite
```

---

## Example: `params_sort_batches.txt`

```txt
# =========================
# Input and output
# =========================
--input /home/avelasco/fantasia/data/uniref50_representatives.fasta
--output-dir /home/avelasco/fantasia/uniref50_batches_sorted_500k

# =========================
# Batching
# =========================
--batch-size 500000
--min-length-kept 51

# =========================
# Output prefix
# =========================
--output-prefix uniref50_sorted

# =========================
# Thresholds to report
# =========================
--report-threshold 30
--report-threshold 50

# =========================
# Traceability files for
# filtered sequences
# =========================
--filtered-fasta-name filtered_out_below_51aa.fasta
--filtered-manifest-name filtered_out_below_51aa_manifest.csv

# =========================
# General behavior
# =========================
--overwrite
```

### Run Using `--args-file`

```bash
python sort_and_batch_fasta_by_length_v2.py \
  --args-file /home/avelasco/scripts/params_sort_batches.txt
```

---

## Report-Only Mode

If you only want to know how many sequences are 30 aa / 50 aa without building batches:

```bash
python sort_and_batch_fasta_by_length_v2.py \
  --input /home/avelasco/fantasia/data/uniref50_representatives.fasta \
  --output-dir /home/avelasco/fantasia/uniref50_length_report \
  --report-only
```

---

# 2. `run_fantasia_batches_with_checkpoint_REVISED_v6.py`

## What It Does

This script runs **FANTASIA** on a directory of batches with an extended execution logic:

- processes **all or selected batches** from a directory;
- lets you choose **all or selected models** from the `models:` block in `config.yaml`;
- enables **only one model per execution** (`enabled: True`) and disables the rest;
- processes **all batches of one model** before switching to the next model;
- updates `config.yaml` in a controlled way to modify:
  - `input`
  - `prefix`
  - `log_path`
  - model `enabled` flags
- keeps a **checkpoint JSON** with the state of each run (*model + batch*);
- reorganizes the outputs under `experiments/` and `logs/` using a session / model / batch hierarchy.

---

## Execution Logic

If, for example, you select:

- models: `ESM,Prot-T5`
- batches: `all`

the execution order will be:

1. `ESM` → batch 1
2. `ESM` → batch 2
3. `ESM` → batch 3
4. ...
5. `Prot-T5` → batch 1
6. `Prot-T5` → batch 2
7. ...

That means:

> **One model is completed across all its batches before moving to the next model.**

This is designed to avoid loading multiple models into GPU memory at the same time.

---

## Final `experiments/` Structure

The intended hierarchy is:

```text
experiments/
└── sample_uniref50_sorted_batches_20260512160252/
    └── ESM/
        └── uniref50_sorted_batch_00001_20260512160252/
            └── embedding.h5
```

And similarly for each batch and each model.

---

## Final `logs/` Structure

Logs are organized by session, model, and batch, for example:

```text
logs/
└── sample_uniref50_sorted_batches_20260512160252/
    └── ESM/
        └── uniref50_sorted_batch_00001_20260512160252/
            └── Logs_20260429172457/
                ├── debug.log
                └── info.log
```

---

## Important Parameters

### Config and batches

- `--config`
- `--batches-dir`
- `--batch-pattern`
- `--batch-select`

### Models

- `--model-select`
- `--session-name-base`

### YAML keys

- `--input-key`
- `--prefix-key`
- `--log-path-key`

### Runner state and logs

- `--checkpoint`
- `--logs-dir`
- `--runner-cmd`

### Other

- `--max-batches`
- `--dry-run`
- `--no-restore-config`
- `--args-file`

---

## Batch Selection

The `--batch-select` parameter lets you specify which batches to process, using the order discovered in the batch directory.

Valid examples:

- `all`
- `1`
- `last`
- `1,last`
- `1,2`
- `1-5`
- `3-last`

---

## Model Selection

The `--model-select` parameter lets you choose which models to run.

Supported values:

- `all`
- `enabled`
- an explicit list, for example:
  - `ESM,Prot-T5`
  - `ESM3c,Ankh3-Large`

---

## Example: Direct Usage

```bash
python run_fantasia_batches_with_checkpoint_REVISED_v6.py \
  --config /home/avelasco/fantasia/config.yaml \
  --batches-dir /home/avelasco/fantasia/uniref50_batches_sorted_500k \
  --batch-pattern "*_batch_*.fasta" \
  --batch-select all \
  --model-select all \
  --session-name-base sample_uniref50_sorted_batches \
  --runner-cmd "poetry run fantasia run" \
  --input-key input \
  --prefix-key prefix \
  --log-path-key log_path \
  --checkpoint /home/avelasco/fantasia/fantasia_batches_checkpoint_v6.json \
  --logs-dir /home/avelasco/fantasia/fantasia_batch_runner_logs_v6
```

---

## Example: `params_fantasia_models.txt`

```txt
# =========================
# Main config
# =========================
--config /home/avelasco/fantasia/config.yaml

# =========================
# Batches
# =========================
--batches-dir /home/avelasco/fantasia/uniref50_batches_sorted_500k
--batch-pattern "*_batch_*.fasta"
--batch-select all

# =========================
# Models
# =========================
--model-select all

# Base name for the session
--session-name-base sample_uniref50_sorted_batches

# =========================
# Execution
# =========================
--runner-cmd "poetry run fantasia run"

# =========================
# Config keys
# =========================
--input-key input
--prefix-key prefix
--log-path-key log_path

# =========================
# Runner state and logs
# =========================
--checkpoint /home/avelasco/fantasia/fantasia_batches_checkpoint_v6.json
--logs-dir /home/avelasco/fantasia/fantasia_batch_runner_logs_v6
```

### Run Using `--args-file`

```bash
python run_fantasia_batches_with_checkpoint_REVISED_v6.py \
  --args-file /home/avelasco/scripts/params_fantasia_models.txt
```

---

## Examples of Partial Selection

### Only Prot-T5

```txt
--model-select Prot-T5
```

### Only models that were `enabled: True` in the original config

```txt
--model-select enabled
```

### Only ESM and Prot-T5

```txt
--model-select ESM,Prot-T5
```

### Only first and last batch

```txt
--batch-select 1,last
```

### Only first and second batch

```txt
--batch-select 1,2
```

---

## Example: Dry Run

Strongly recommended before launching many models or many batches:

```bash
python run_fantasia_batches_with_checkpoint_REVISED_v6.py \
  --args-file /home/avelasco/scripts/params_fantasia_models.txt \
  --dry-run
```

This lets you verify:

- model selection
- batch selection
- checkpoint creation
- session structure
- expected config modifications

without actually running FANTASIA.

---

# Recommended Full Workflow

## Step 1 — Create length-sorted batches

```bash
python sort_and_batch_fasta_by_length_v2.py \
  --args-file /home/avelasco/scripts/params_sort_batches.txt
```

## Step 2 — Run FANTASIA on all batches and all models

```bash
python run_fantasia_batches_with_checkpoint_REVISED_v6.py \
  --args-file /home/avelasco/scripts/params_fantasia_models.txt
```

## Step 3 — Analyze timing afterwards from `info.log` / `debug.log`

If you also have the log analysis script available, the next step is to process the logs generated by FANTASIA in order to estimate generation / writing times without slowing down the main execution.

---

# Practical Recommendations

- Run these scripts from the **root of the FANTASIA project** to avoid issues with relative paths inside `config.yaml`.
- Test first with `--dry-run`.
- Start with a reduced test, for example:
  - `--batch-select 1,2`
  - `--model-select ESM,Prot-T5`
- Inspect the checkpoint after the first runs.
- Keep the original FASTA untouched and generate all outputs into separate directories.

---

# Suggested `scripts/` Layout

Example:

```text
scripts/
├── run_fantasia_batches_with_checkpoint_REVISED_v6.py
├── sort_and_batch_fasta_by_length_v2.py
├── params_fantasia_models.txt
├── params_sort_batches.txt
└── README.md
```

---

# Final Notes

- `sort_and_batch_fasta_by_length_v2.py` **does not modify** the original FASTA file; it only reads it and writes new outputs.
- `run_fantasia_batches_with_checkpoint_REVISED_v6.py` **does temporarily modify** `config.yaml`, but restores it at the end unless `--no-restore-config` is used.
- v6 is specifically designed so that **only one model is active per run**, precisely to avoid loading multiple models into GPU memory at the same time.
