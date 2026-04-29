#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [OPTIONS] [CONFIG_PATH] <PROTEOMES_ROOT> [EXPERIMENTS_DIR] [RUN_TAG]

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

Options:
  --all-folders
      Ignore the embedded folder list and scan every immediate subdirectory
      under PROTEOMES_ROOT.

  --size-batch lt10|gt10lt20|nonarth-rest|arthropoda
      Select folders dynamically based on the number of unique proteomes
      detected in each folder:
        lt10         fewer than 10 proteomes
        gt10lt20     more than 10 and fewer than 20 proteomes
        nonarth-rest everything else except arthropoda
        arthropoda   only the arthropoda folder

  --folders folder1,folder2
      Restrict the run to a comma-separated list of folder names.

  --exclude-folders folder1,folder2
      Skip the given comma-separated folder names.

Behavior:
  - Scans only the predefined folder list embedded in this script.
  - Accepts .pep, .faa, .fa, .fasta and their .gz versions.
  - If only a .gz file is present, it is decompressed with gunzip -k.
  - Skips a run when a matching experiment directory already exists.
  - Executes runs sequentially, one proteome at a time.

Examples:
  $(basename "$0") config/prott5_full.yaml /data/proteomes
  $(basename "$0") config/ankh_test.yaml /data/proteomes /data/fantasia_runs/experiments
  $(basename "$0") config/prott5_full.yaml /data/proteomes /data/fantasia_runs/experiments prott5
  $(basename "$0") --all-folders --exclude-folders arthropoda config/prott5_full.yaml /data/proteomes /data/fantasia_runs/experiments prott5
  $(basename "$0") --folders arthropoda config/prott5_full.yaml /data/proteomes /data/fantasia_runs/experiments prott5
  $(basename "$0") --size-batch lt10 config/prott5_full.yaml /data/proteomes /data/fantasia_runs/experiments prott5
  $(basename "$0") --size-batch nonarth-rest config/prott5_full.yaml /data/proteomes /data/fantasia_runs/experiments prott5
EOF
}

ALL_FOLDERS=0
FOLDERS_ARG=""
EXCLUDE_FOLDERS_ARG=""
SIZE_BATCH=""

while (($# > 0)); do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --all-folders)
      ALL_FOLDERS=1
      shift
      ;;
    --size-batch)
      SIZE_BATCH="${2:-}"
      shift 2
      ;;
    --folders)
      FOLDERS_ARG="${2:-}"
      shift 2
      ;;
    --exclude-folders)
      EXCLUDE_FOLDERS_ARG="${2:-}"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      break
      ;;
  esac
done

CONFIG_PATH="${1:-${REPO_ROOT}/config/prott5_full.yaml}"
ROOT="${2:-}"
EXPERIMENTS_DIR="${3:-$HOME/fantasia_runs/experiments}"
RUN_TAG="${4:-$(basename "${CONFIG_PATH}" .yaml)}"

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

if [[ -n "${SIZE_BATCH}" && ( -n "${FOLDERS_ARG}" || "${ALL_FOLDERS}" -eq 1 ) ]]; then
  echo "--size-batch cannot be combined with --folders or --all-folders." >&2
  exit 1
fi

case "${SIZE_BATCH}" in
  ""|lt10|gt10lt20|nonarth-rest|arthropoda)
    ;;
  *)
    echo "Invalid --size-batch value: ${SIZE_BATCH}" >&2
    echo "Expected one of: lt10, gt10lt20, nonarth-rest, arthropoda" >&2
    exit 1
    ;;
esac

default_folders=(
  acoela
  brachiopoda
  bryozoa
  ctenophora
  cycliophora
  dicyemida
  entoprocta
  gastrotricha
  hemichordata
  micrognathozoa
  nematomorpha
  nemertodermatida
  outgroup
  phoronida
  placozoa
  porifera
  priapulida
  tardigrada
  xenoturbellida
)

declare -a folders=()
declare -A excluded_folders=()

if [[ -n "${EXCLUDE_FOLDERS_ARG}" ]]; then
  IFS=',' read -r -a exclude_items <<< "${EXCLUDE_FOLDERS_ARG}"
  for folder in "${exclude_items[@]}"; do
    [[ -n "${folder}" ]] && excluded_folders["${folder}"]=1
  done
fi

if [[ -n "${FOLDERS_ARG}" ]]; then
  IFS=',' read -r -a folders <<< "${FOLDERS_ARG}"
elif ((ALL_FOLDERS)); then
  while IFS= read -r folder; do
    folders+=("${folder}")
  done < <(find "${ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
else
  folders=("${default_folders[@]}")
fi

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

folder_base_count() {
  local folder_path="$1"
  declare -A seen_bases=()
  local file_name
  local base

  while IFS= read -r file_path; do
    file_name="$(basename "$file_path")"
    file_name="${file_name%.gz}"
    case "$file_name" in
      *.pep|*.faa|*.fa|*.fasta)
        base="${file_name%.*}"
        seen_bases["$base"]=1
        ;;
    esac
  done < <(find "${folder_path}" -maxdepth 1 -type f \( -name '*.pep' -o -name '*.faa' -o -name '*.fa' -o -name '*.fasta' -o -name '*.pep.gz' -o -name '*.faa.gz' -o -name '*.fa.gz' -o -name '*.fasta.gz' \) | sort)

  printf '%s\n' "${#seen_bases[@]}"
}

matches_size_batch() {
  local folder_name="$1"
  local proteome_count="$2"

  case "${SIZE_BATCH}" in
    "")
      return 0
      ;;
    lt10)
      (( proteome_count < 10 ))
      return
      ;;
    gt10lt20)
      (( proteome_count > 10 && proteome_count < 20 ))
      return
      ;;
    nonarth-rest)
      [[ "${folder_name}" == "arthropoda" ]] && return 1
      (( proteome_count < 10 )) && return 1
      (( proteome_count > 10 && proteome_count < 20 )) && return 1
      return 0
      ;;
    arthropoda)
      [[ "${folder_name}" == "arthropoda" ]]
      return
      ;;
  esac

  return 1
}

for folder_name in "${folders[@]}"; do
  [[ -n "${folder_name}" ]] || continue
  [[ -z "${excluded_folders[${folder_name}]+x}" ]] || continue

  folder_path="${ROOT}/${folder_name}"
  [[ -d "${folder_path}" ]] || continue

  proteome_count="$(folder_base_count "${folder_path}")"
  matches_size_batch "${folder_name}" "${proteome_count}" || continue

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
