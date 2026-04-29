#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

BASE_CONFIG="${FANTASIA_CONFIG:-${REPO_ROOT}/fantasia/config.yaml}"
INPUT_FASTA="${FANTASIA_INPUT:-${REPO_ROOT}/data_sample/PRUB1_longiso.pep}"
BASE_DIR="${FANTASIA_BASE_DIR:-${REPO_ROOT}/fantasia_prub_annotation_runs}"
LOG_DIR="${BASE_DIR}/logs"
PREFIX="${FANTASIA_PREFIX:-prub_all_models_k1}"
DEVICE="${FANTASIA_DEVICE:-cuda}"
DB_HOST="${FANTASIA_DB_HOST:-localhost}"
DB_PORT="${FANTASIA_DB_PORT:-5432}"

if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "Config file not found: ${BASE_CONFIG}" >&2
  exit 1
fi

if [[ ! -f "${INPUT_FASTA}" ]]; then
  echo "Input FASTA not found: ${INPUT_FASTA}" >&2
  exit 1
fi

mkdir -p "${BASE_DIR}" "${LOG_DIR}"
TMP_CONFIG="$(mktemp "/tmp/${PREFIX}.XXXXXX.yaml")"
trap 'rm -f "${TMP_CONFIG}"' EXIT

BASE_CONFIG="${BASE_CONFIG}" INPUT_FASTA="${INPUT_FASTA}" PREFIX="${PREFIX}" python - <<'PY' > "${TMP_CONFIG}"
import os
import yaml

with open(os.environ["BASE_CONFIG"]) as handle:
    conf = yaml.safe_load(handle)

conf["prefix"] = os.environ["PREFIX"]
conf["input"] = os.environ["INPUT_FASTA"]
conf["limit_execution"] = 0
conf["only_lookup"] = False
conf["only_embedding"] = False
conf.setdefault("embedding", {})["device"] = os.environ.get("FANTASIA_DEVICE", "cuda")
conf["embedding"]["max_sequence_length"] = 0

for cfg in conf["embedding"]["models"].values():
    cfg["enabled"] = True
    cfg["layer_index"] = [0]

lookup = conf.setdefault("lookup", {})
lookup["use_gpu"] = True
lookup["distance_metric"] = "cosine"
lookup["limit_per_entry"] = 1
lookup.setdefault("taxonomy", {})["exclude"] = []
lookup["taxonomy"]["include_only"] = []
lookup["taxonomy"]["get_descendants"] = False

conf["taxonomy"] = lookup["taxonomy"]

print(yaml.safe_dump(conf, sort_keys=False))
PY

cd "${REPO_ROOT}"

cmd=(
  python -m fantasia.main run
  --config "${TMP_CONFIG}"
  --base_directory "${BASE_DIR}"
  --log_path "${LOG_DIR}"
  --device "${DEVICE}"
  --DB_HOST "${DB_HOST}"
  --DB_PORT "${DB_PORT}"
  --limit_per_entry 1
  --only_lookup false
  --only_embedding false
)

printf 'Running PRUB all-model annotation:\n'
printf '  %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"

