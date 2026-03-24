#!/usr/bin/env bash

# Run fantasia embedding sequentially for a fixed list of prefixes.
# Each run writes to its own nohup log file.

set -euo pipefail

prefixes=(
  GEVG1
  OALV1
  PCAU1
  PMUR1
  SPUN1
  NVEC1
  OANA1
  PMAX1
  SMAR2
)

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
input_dir="/home/alexdoro/DATASETS/prott5_all_layers_20260323"

if ! command -v poetry >/dev/null 2>&1; then
  echo "ERROR: poetry is not installed or not in PATH." >&2
  exit 1
fi

cd "$repo_dir"

echo "Running only_embed sequentially for ${#prefixes[@]} prefixes"

total="${#prefixes[@]}"
for i in "${!prefixes[@]}"; do
  prefix="${prefixes[$i]}"
  current="$((i + 1))"
  input_file="$input_dir/${prefix}_genomes.pep"
  log_file="fantasia_only_embed_${prefix}.log"

  if [[ ! -f "$input_file" ]]; then
    echo "[$current/$total] SKIP $prefix: input file not found: $input_file" >&2
    continue
  fi

  echo "[$current/$total] START $prefix"
  nohup poetry run fantasia run --input "$input_file" --only_embed True --prefix "$prefix" > "$log_file" 2>&1 &
  pid="$!"

  # Wait for current run so execution remains sequential.
  wait "$pid"
  exit_code="$?"

  if [[ "$exit_code" -eq 0 ]]; then
    echo "[$current/$total] DONE  $prefix (log: $log_file)"
  else
    echo "[$current/$total] FAIL  $prefix (exit=$exit_code, log: $log_file)" >&2
  fi

done

echo "All prefixes processed."
