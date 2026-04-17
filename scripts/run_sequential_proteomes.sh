#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [CONFIG_PATH] <PROTEOMES_ROOT> [EXPERIMENTS_DIR] [RUN_TAG]

Description:
  Run FANTASIA jobs sequentially across a predefined set of folders under
  PROTEOMES_ROOT. For each detected FASTA-like file (.pep, .faa, .fa, .fasta,
  optionally .gz), the script launches:

    python -m fantasia.main run --config CONFIG_PATH --input <file> --prefix <prefix>

Arguments:
  CONFIG_PATH
      Optional path to the FANTASIA YAML config.
      Default: ${REPO_ROOT}/config/prott5_full.yaml

  PROTEOMES_ROOT
      Root directory containing the expected taxon subfolders (for example:
      acoela, brachiopoda, bryozoa, ctenophora, etc.).

  EXPERIMENTS_DIR
      Optional experiments directory used to detect existing runs and skip
      prefixes that were already launched.
      Default: \$HOME/fantasia_runs/experiments

  RUN_TAG
      Optional prefix tag prepended to generated run names.
      Default: basename(CONFIG_PATH) without the .yaml suffix

Behavior:
  - Scans only the predefined folder list embedded in this script.
  - Accepts .pep, .faa, .fa, .fasta and their .gz versions.
  - If only a .gz file is present, it is decompressed with gunzip -k.
  - Skips a run when a matching experiment directory already exists.
  - Executes runs sequentially, one proteome at a time.

Examples:
  $(basename "$0") ./config/prott5_full.yaml /data/proteomes
  $(basename "$0") ./config/ankh_test.yaml /data/proteomes /data/fantasia_runs/experiments
  $(basename "$0") ./config/prott5_full.yaml /data/proteomes /data/fantasia_runs/experiments prott5
EOF
}

CONFIG_PATH="${1:-${REPO_ROOT}/config/prott5_full.yaml}"
ROOT="${2:-}"
EXPERIMENTS_DIR="${3:-$HOME/fantasia_runs/experiments}"
RUN_TAG="${4:-$(basename "${CONFIG_PATH}" .yaml)}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ -z "${ROOT}" ]]; then
  usage >&2
  exit 1
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config file not found: ${CONFIG_PATH}" >&2
  exit 1
fi

if [[ ! -d "${ROOT}" ]]; then
  echo "Proteomes root directory not found: ${ROOT}" >&2
  exit 1
fi

folders=(
  proteome1
  proteome2
  proteome3
  proteome4
)

choose_input_file() {
  local folder="$1"
  local base="$2"
  local ext
  for ext in .pep .faa .fa .fasta; do
    if [[ -f "${folder}/${base}${ext}" ]]; then
      printf '%s\n' "${folder}/${base}${ext}"
      return 0
    fi
  done
  for ext in .pep .faa .fa .fasta; do
    if [[ -f "${folder}/${base}${ext}.gz" ]]; then
      echo "Decompressing ${folder}/${base}${ext}.gz" >&2
      gunzip -k "${folder}/${base}${ext}.gz"
      printf '%s\n' "${folder}/${base}${ext}"
      return 0
    fi
  done
  return 1
}

for folder_name in "${folders[@]}"; do
  folder_path="${ROOT}/${folder_name}"
  [[ -d "${folder_path}" ]] || continue

  declare -A bases=()

  while IFS= read -r file_path; do
    file_name="$(basename "$file_path")"
    file_name="${file_name%.gz}"
    case "$file_name" in
      *.pep|*.faa|*.fa|*.fasta)
        base="${file_name%.*}"
        bases["$base"]=1
        ;;
    esac
  done < <(find "${folder_path}" -maxdepth 1 -type f \( -name '*.pep' -o -name '*.faa' -o -name '*.fa' -o -name '*.fasta' -o -name '*.pep.gz' -o -name '*.faa.gz' -o -name '*.fa.gz' -o -name '*.fasta.gz' \) | sort)

  for base in "${!bases[@]}"; do
    input_file="$(choose_input_file "${folder_path}" "${base}")"
    run_prefix="${RUN_TAG}_${folder_name}_${base}"

    if [[ -d "${EXPERIMENTS_DIR}" ]] && find "${EXPERIMENTS_DIR}" -maxdepth 1 -type d -name "${run_prefix}_*" | grep -q .; then
      echo "Skipping ${run_prefix}: existing run detected."
      continue
    fi

    echo "Running ${run_prefix}"
    python -m fantasia.main run \
      --config "${CONFIG_PATH}" \
      --input "${input_file}" \
      --prefix "${run_prefix}"
  done
done
