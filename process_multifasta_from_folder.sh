#!/usr/bin/env bash
set -euo pipefail

BASE_CONFIG="/home/alexdoro/repositories/FANTASIA/fantasia/config.yaml"
# Directory containing the grouped OX_*.fasta files
FASTAS_DIR="/home/alexdoro/fantasia/datasets/ox_grouped_fastas"
OUT_DIR="/home/alexdoro/fantasia/experiments/ox_runs"

mkdir -p "$OUT_DIR"

# Llista de FASTA
shopt -s nullglob
FASTAS=("${FASTAS_DIR}"/OX_*.fasta)
TOTAL=${#FASTAS[@]}

if (( TOTAL == 0 )); then
    echo "❌ No FASTA files found matching: ${FASTAS_DIR}/OX_*.fasta" >&2
    exit 1
fi

COUNT=0

echo "🧬 Starting FANTASIA batch: ${TOTAL} organisms"
echo "--------------------------------------------------"

for fasta in "${FASTAS[@]}"; do
    COUNT=$((COUNT + 1))

    fname=$(basename "$fasta")
    ox_id=${fname#OX_}
    ox_id=${ox_id%.fasta}

    echo ""
    echo "▶ [${COUNT}/${TOTAL}] Processing OX_${ox_id}"
    echo "    FASTA: ${fasta}"
    echo "--------------------------------------------------"

    poetry run fantasia run \
    --config "$BASE_CONFIG" \
    --input "$fasta" \
    --taxonomy_ids_to_exclude "$ox_id" \
    --prefix "OX_${ox_id}" \
    --base_directory "${OUT_DIR}"

    echo "✔ Finished OX_${ox_id} (${COUNT}/${TOTAL})"
done

echo ""
echo "🎉 All ${TOTAL} organisms processed successfully"
