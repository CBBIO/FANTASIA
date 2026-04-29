#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

BASE_CONFIG="${FANTASIA_CONFIG:-${REPO_ROOT}/config/prott5_full.yaml}"
BASE_DIR="${FANTASIA_BASE_DIR:-${REPO_ROOT}/fantasia_no_truncation_benchmark_runs}"
LOG_DIR="${BASE_DIR}/logs"
REPORT_DIR="${BASE_DIR}/missing_embeddings"
DB_HOST="${FANTASIA_DB_HOST:-localhost}"
DB_PORT="${FANTASIA_DB_PORT:-5432}"
DEVICE="${FANTASIA_DEVICE:-cuda}"
MODE_FILTER="${1:-all}"
MODEL_FILTER="${2:-all}"

mkdir -p "${BASE_DIR}" "${LOG_DIR}" "${REPORT_DIR}"

models=("Prot-T5" "ESM" "ESM3c" "Ankh3-Large" "Prost-T5")
model_slugs=("prot-t5" "esm" "esm3c" "ankh3-large" "prost-t5")
modes=("baseline" "self-exclude" "self-exclude-redundancy")

run_one() {
  local organism="$1" taxid="$2" input_fasta="$3" mode="$4" model="$5" model_slug="$6" k="$7"
  local prefix cfg log_file exp_dir embeddings report exclude redundancy

  case "${mode}" in
    baseline) exclude=""; redundancy="0" ;;
    self-exclude) exclude="${taxid}"; redundancy="0" ;;
    self-exclude-redundancy) exclude="${taxid}"; redundancy="0.95" ;;
    *) echo "Unknown mode: ${mode}" >&2; exit 1 ;;
  esac

  prefix="no_trunc_${organism}_${mode}_${model_slug}_k${k}"
  cfg="$(mktemp "/tmp/${prefix}.XXXXXX.yaml")"

  BASE_CONFIG="${BASE_CONFIG}" INPUT_FASTA="${input_fasta}" PREFIX="${prefix}" MODEL="${model}" TAXID="${taxid}" MODE="${mode}" K="${k}" REDUNDANCY="${redundancy}" python - <<'PY' > "${cfg}"
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

for key, cfg in conf["embedding"]["models"].items():
    cfg["enabled"] = key == os.environ["MODEL"]
    cfg["layer_index"] = [0]

lookup = conf.setdefault("lookup", {})
lookup["use_gpu"] = True
lookup["distance_metric"] = "cosine"
lookup["limit_per_entry"] = int(os.environ["K"])
lookup.setdefault("taxonomy", {})["get_descendants"] = False
lookup["taxonomy"]["exclude"] = [os.environ["TAXID"]] if os.environ["MODE"] != "baseline" else []

redundancy = lookup.setdefault("redundancy", {})
redundancy["identity"] = float(os.environ["REDUNDANCY"])
redundancy["coverage"] = 0.7
redundancy["threads"] = 10

conf["taxonomy"] = lookup["taxonomy"]
conf["redundancy"] = redundancy

print(yaml.safe_dump(conf, sort_keys=False))
PY

  log_file="${LOG_DIR}/${prefix}_$(date '+%Y%m%d%H%M%S').log"
  echo "Running ${prefix}"
  cmd=(
    python -m fantasia.main run
    --config "${cfg}" \
    --base_directory "${BASE_DIR}" \
    --log_path "${LOG_DIR}" \
    --device "${DEVICE}" \
    --DB_HOST "${DB_HOST}" \
    --DB_PORT "${DB_PORT}" \
    --limit_per_entry "${k}" \
    --redundancy_identity "${redundancy}" \
    --redundancy_coverage 0.7 \
    --threads 10
  )
  if [[ -n "${exclude}" ]]; then
    cmd+=(--taxonomy_ids_to_exclude "${exclude}")
  fi

  set +e
  "${cmd[@]}" > "${log_file}" 2>&1
  exit_code=$?
  set -e

  exp_dir="$(find "${BASE_DIR}/experiments" -maxdepth 1 -type d -name "${prefix}_*" 2>/dev/null | sort | tail -n 1 || true)"
  embeddings="${exp_dir}/embeddings.h5"
  report="${REPORT_DIR}/${prefix}_missing_embeddings.tsv"
  python "${SCRIPT_DIR}/report_missing_embeddings.py" \
    --input "${input_fasta}" \
    --embeddings "${embeddings}" \
    --output "${report}" >> "${log_file}" 2>&1 || true

  rm -f "${cfg}"
  echo "${prefix}	exit=${exit_code}	log=${log_file}	report=${report}"
  return 0
}

cd "${REPO_ROOT}"

for organism in mouse worm; do
  case "${organism}" in
    mouse) taxid="10090"; input_fasta="${REPO_ROOT}/data_sample/MUSM_10090.fasta" ;;
    worm) taxid="6239"; input_fasta="${REPO_ROOT}/data_sample/UP000001940_6239.fasta" ;;
  esac

  for mode in "${modes[@]}"; do
    [[ "${MODE_FILTER}" != "all" && "${MODE_FILTER}" != "${mode}" ]] && continue
    for idx in "${!models[@]}"; do
      model="${models[$idx]}"
      model_slug="${model_slugs[$idx]}"
      [[ "${MODEL_FILTER}" != "all" && "${MODEL_FILTER}" != "${model_slug}" ]] && continue
      run_one "${organism}" "${taxid}" "${input_fasta}" "${mode}" "${model}" "${model_slug}" 1
    done

    if [[ "${MODEL_FILTER}" == "all" || "${MODEL_FILTER}" == "prot-t5" ]]; then
      run_one "${organism}" "${taxid}" "${input_fasta}" "${mode}" "Prot-T5" "prot-t5" 3
    fi
  done
done
