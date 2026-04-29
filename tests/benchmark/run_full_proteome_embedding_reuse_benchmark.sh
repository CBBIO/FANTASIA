#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

BASE_CONFIG="${FANTASIA_CONFIG:-${REPO_ROOT}/config/prott5_full.yaml}"
BASE_DIR="${FANTASIA_BASE_DIR:-${REPO_ROOT}/fantasia_full_proteome_benchmark_runs}"
LOG_DIR="${BASE_DIR}/logs"
REPORT_DIR="${BASE_DIR}/missing_embeddings"
MANIFEST="${BASE_DIR}/embeddings_manifest.tsv"
DB_HOST="${FANTASIA_DB_HOST:-localhost}"
DB_PORT="${FANTASIA_DB_PORT:-5433}"
DEVICE="${FANTASIA_DEVICE:-cuda}"
ORGANISM_FILTER="${1:-mouse}"
MODEL_FILTER="${2:-all}"
K_VALUES="${FANTASIA_K_VALUES:-1 3 5}"
MODE_VALUES="${FANTASIA_MODES:-baseline}"
DISTANCE_THRESHOLD="${FANTASIA_DISTANCE_THRESHOLD:-}"
REUSE_EMBEDDINGS="${FANTASIA_REUSE_EMBEDDINGS:-true}"

models=("Prot-T5" "ESM" "ESM3c" "Ankh3-Large" "Prost-T5")
model_slugs=("prot-t5" "esm" "esm3c" "ankh3-large" "prost-t5")
read -r -a modes <<< "${MODE_VALUES}"

mkdir -p "${BASE_DIR}" "${LOG_DIR}" "${REPORT_DIR}"
[[ -f "${MANIFEST}" ]] || printf "organism\ttaxid\tmodel\tembeddings_path\tstatus\tlog_file\n" > "${MANIFEST}"

organism_input() {
  case "$1" in
    mouse) echo "${REPO_ROOT}/data_sample/MUSM_10090.fasta" ;;
    worm) echo "${REPO_ROOT}/data_sample/UP000001940_6239.fasta" ;;
    *) echo "Unknown organism: $1" >&2; exit 1 ;;
  esac
}

organism_taxid() {
  case "$1" in
    mouse) echo "10090" ;;
    worm) echo "6239" ;;
    *) echo "Unknown organism: $1" >&2; exit 1 ;;
  esac
}

make_config() {
  local output="$1" prefix="$2" input="$3" model="$4" only_embedding="$5" only_lookup="$6" k="$7" taxid="$8" mode="$9" redundancy="${10}"
  BASE_CONFIG="${BASE_CONFIG}" PREFIX="${prefix}" INPUT="${input}" MODEL="${model}" ONLY_EMBEDDING="${only_embedding}" ONLY_LOOKUP="${only_lookup}" K="${k}" TAXID="${taxid}" MODE="${mode}" REDUNDANCY="${redundancy}" DISTANCE_THRESHOLD="${DISTANCE_THRESHOLD}" python - <<'PY' > "${output}"
import os
import yaml

with open(os.environ["BASE_CONFIG"]) as handle:
    conf = yaml.safe_load(handle)

conf["prefix"] = os.environ["PREFIX"]
conf["input"] = os.environ["INPUT"]
conf["limit_execution"] = 0
conf["only_embedding"] = os.environ["ONLY_EMBEDDING"].lower() == "true"
conf["only_lookup"] = os.environ["ONLY_LOOKUP"].lower() == "true"
conf.setdefault("embedding", {})["device"] = os.environ.get("FANTASIA_DEVICE", "cuda")
conf["embedding"]["max_sequence_length"] = 0

for key, cfg in conf["embedding"]["models"].items():
    cfg["enabled"] = key == os.environ["MODEL"]
    cfg["layer_index"] = [0]
    if cfg["enabled"]:
        threshold = os.environ.get("DISTANCE_THRESHOLD", "").strip()
        cfg["distance_threshold"] = float(threshold) if threshold else False

lookup = conf.setdefault("lookup", {})
lookup["use_gpu"] = True
lookup["distance_metric"] = "cosine"
lookup["limit_per_entry"] = int(os.environ["K"])
lookup.setdefault("taxonomy", {})["get_descendants"] = False
lookup["taxonomy"]["exclude"] = [os.environ["TAXID"]] if os.environ["MODE"] != "baseline" else []
lookup["taxonomy"]["include_only"] = []

redundancy = lookup.setdefault("redundancy", {})
redundancy["identity"] = float(os.environ["REDUNDANCY"])
redundancy["coverage"] = 0.7
redundancy["threads"] = 10

conf["taxonomy"] = lookup["taxonomy"]
conf["redundancy"] = redundancy

print(yaml.safe_dump(conf, sort_keys=False))
PY
}

find_latest_embeddings() {
  local prefix="$1"
  find "${BASE_DIR}/experiments" -maxdepth 2 -type f -path "*/${prefix}_*/embeddings.h5" 2>/dev/null | sort | tail -n 1 || true
}

run_embedding_once() {
  local organism="$1" taxid="$2" input="$3" model="$4" model_slug="$5"
  local prefix cfg log_file embeddings report exit_code
  prefix="embed_full_${organism}_${model_slug}"
  embeddings="$(find_latest_embeddings "${prefix}")"

  if [[ "${REUSE_EMBEDDINGS}" == "true" && -n "${embeddings}" && -f "${embeddings}" ]]; then
    echo "Reusing embeddings: ${embeddings}" >&2
  else
    cfg="$(mktemp "/tmp/${prefix}.XXXXXX.yaml")"
    make_config "${cfg}" "${prefix}" "${input}" "${model}" true false 1 "${taxid}" baseline 0
    log_file="${LOG_DIR}/${prefix}_$(date '+%Y%m%d%H%M%S').log"
    echo "Embedding full proteome | organism=${organism} model=${model_slug}" >&2
    set +e
    python -m fantasia.main run \
      --config "${cfg}" \
      --base_directory "${BASE_DIR}" \
      --log_path "${LOG_DIR}" \
      --device "${DEVICE}" \
      --DB_HOST "${DB_HOST}" \
      --DB_PORT "${DB_PORT}" \
      --only_embedding true \
      > "${log_file}" 2>&1
    exit_code=$?
    set -e
    rm -f "${cfg}"
    embeddings="$(find_latest_embeddings "${prefix}")"
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${organism}" "${taxid}" "${model_slug}" "${embeddings}" "${exit_code}" "${log_file}" >> "${MANIFEST}"
  fi

  if [[ -n "${embeddings}" && -f "${embeddings}" ]]; then
    report="${REPORT_DIR}/${prefix}_missing_embeddings.tsv"
    python "${SCRIPT_DIR}/report_missing_embeddings.py" \
      --input "${input}" \
      --embeddings "${embeddings}" \
      --output "${report}" >&2 || true
    echo "${embeddings}"
  fi
}

run_lookup_only() {
  local organism="$1" taxid="$2" embeddings="$3" model="$4" model_slug="$5" mode="$6" k="$7"
  local redundancy prefix cfg log_file exclude_arg=()
  case "${mode}" in
    baseline) redundancy="0" ;;
    self-exclude) redundancy="0" ; exclude_arg=(--taxonomy_ids_to_exclude "${taxid}") ;;
    self-exclude-redundancy) redundancy="0.95" ; exclude_arg=(--taxonomy_ids_to_exclude "${taxid}") ;;
    *) echo "Unknown mode: ${mode}" >&2; exit 1 ;;
  esac
  prefix="lookup_full_${organism}_${mode}_${model_slug}_k${k}"
  cfg="$(mktemp "/tmp/${prefix}.XXXXXX.yaml")"
  make_config "${cfg}" "${prefix}" "${embeddings}" "${model}" false true "${k}" "${taxid}" "${mode}" "${redundancy}"
  log_file="${LOG_DIR}/${prefix}_$(date '+%Y%m%d%H%M%S').log"
  echo "Lookup-only | organism=${organism} mode=${mode} model=${model_slug} k=${k}"
  python -m fantasia.main run \
    --config "${cfg}" \
    --input "${embeddings}" \
    --base_directory "${BASE_DIR}" \
    --log_path "${LOG_DIR}" \
    --device "${DEVICE}" \
    --DB_HOST "${DB_HOST}" \
    --DB_PORT "${DB_PORT}" \
    --only_lookup true \
    --limit_per_entry "${k}" \
    --redundancy_identity "${redundancy}" \
    --redundancy_coverage 0.7 \
    --threads 10 \
    "${exclude_arg[@]}" \
    > "${log_file}" 2>&1 || true
  rm -f "${cfg}"
}

case "${ORGANISM_FILTER}" in
  mouse) organisms=("mouse") ;;
  worm) organisms=("worm") ;;
  all) organisms=("mouse" "worm") ;;
  *) echo "Usage: $0 [mouse|worm|all] [prot-t5|esm|esm3c|ankh3-large|prost-t5|all]" >&2; exit 1 ;;
esac

cd "${REPO_ROOT}"

for organism in "${organisms[@]}"; do
  taxid="$(organism_taxid "${organism}")"
  input="$(organism_input "${organism}")"
  for idx in "${!models[@]}"; do
    model="${models[$idx]}"
    model_slug="${model_slugs[$idx]}"
    [[ "${MODEL_FILTER}" != "all" && "${MODEL_FILTER}" != "${model_slug}" ]] && continue
    embeddings="$(run_embedding_once "${organism}" "${taxid}" "${input}" "${model}" "${model_slug}" | tail -n 1)"
    [[ -z "${embeddings}" || ! -f "${embeddings}" ]] && continue
    for mode in "${modes[@]}"; do
      for k in ${K_VALUES}; do
        run_lookup_only "${organism}" "${taxid}" "${embeddings}" "${model}" "${model_slug}" "${mode}" "${k}"
      done
    done
  done
done
