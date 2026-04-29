#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

CONFIG_PATH="${FANTASIA_CONFIG:-${REPO_ROOT}/config/prott5_test.yaml}"
INPUT_FASTA="${FANTASIA_INPUT:-${REPO_ROOT}/data_sample/PRUB1_longiso.pep}"
BASE_DIR="${FANTASIA_BASE_DIR:-${REPO_ROOT}/fantasia_test_runs}"
PREFIX="${FANTASIA_PREFIX:-annotation_example}"
DEVICE="${FANTASIA_DEVICE:-cuda}"
DB_HOST="${FANTASIA_DB_HOST:-localhost}"
DB_PORT="${FANTASIA_DB_PORT:-5432}"

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
  --only_lookup false
  --only_embedding false
)

printf 'Running annotation example:\n'
printf '  %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
