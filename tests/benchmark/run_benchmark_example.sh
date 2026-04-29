#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

CONFIG_PATH="${FANTASIA_CONFIG:-${REPO_ROOT}/config/musm_benchmark_prott5_k5.yaml}"
INPUT_FASTA="${FANTASIA_INPUT:-${REPO_ROOT}/data_sample/MUSM_10090.fasta}"
BASE_DIR="${FANTASIA_BASE_DIR:-${REPO_ROOT}/fantasia_benchmark_runs}"
PREFIX="${FANTASIA_PREFIX:-musm_benchmark_prott5_k5}"
DEVICE="${FANTASIA_DEVICE:-cuda}"
DB_HOST="${FANTASIA_DB_HOST:-localhost}"
DB_PORT="${FANTASIA_DB_PORT:-5432}"
TAXONOMY_EXCLUDE="${FANTASIA_TAXONOMY_EXCLUDE:-10090}"
K="${FANTASIA_K:-5}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config file not found: ${CONFIG_PATH}" >&2
  exit 1
fi

if [[ ! -f "${INPUT_FASTA}" ]]; then
  echo "Input FASTA not found: ${INPUT_FASTA}" >&2
  exit 1
fi

mkdir -p "${BASE_DIR}/logs"

cd "${REPO_ROOT}"

cmd=(
  python -m fantasia.main run
  --config "${CONFIG_PATH}"
  --input "${INPUT_FASTA}"
  --prefix "${PREFIX}"
  --base_directory "${BASE_DIR}"
  --log_path "${BASE_DIR}/logs"
  --device "${DEVICE}"
  --DB_HOST "${DB_HOST}"
  --DB_PORT "${DB_PORT}"
  --limit_per_entry "${K}"
  --taxonomy_ids_to_exclude "${TAXONOMY_EXCLUDE}"
  --only_lookup false
  --only_embedding false
)

printf 'Running benchmark example:\n'
printf '  %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
