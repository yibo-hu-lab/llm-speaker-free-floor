#!/usr/bin/env bash
# Reproduce every open-weight experiment in the paper, then build the tables.
#   6 models x 7 datasets:  mechanism, role, controls, paraphrase, preamble, userattr,
#                           and the mechanism grid with a random wrong target
#   anchor models x anchor datasets (Qwen2.5-7B, Llama-3.1-8B on ARC-Challenge, MMLU-Pro):
#                           dose, source-prefix probe, message role
# Frontier-model, open-ended and justification runs call the OpenAI API and are
# run separately (see README).
set -euo pipefail
cd "$(dirname "$0")"

MODELS=(
  Qwen/Qwen2.5-1.5B-Instruct
  Qwen/Qwen2.5-3B-Instruct
  Qwen/Qwen2.5-7B-Instruct
  meta-llama/Llama-3.1-8B-Instruct
  mistralai/Mistral-7B-Instruct-v0.3
  google/gemma-2-9b-it
)
ANCHOR_MODELS=(Qwen/Qwen2.5-7B-Instruct meta-llama/Llama-3.1-8B-Instruct)
ANCHOR_DATASETS=(arc_challenge mmlu_pro)

run() {  # model dataset grid [extra args...]
  local model=$1 ds=$2 grid=$3; shift 3
  local tag; tag=$(basename "$model")
  local name=$grid
  [[ " $* " == *" --target random "* ]] && name="${grid}_random"
  local prefix="out/${name}/${tag}_${ds}"
  [ -f "${prefix}_long.csv" ] && { echo "skip ${prefix}"; return; }
  python run_arbitration.py --model "$model" --items "data/${ds}.jsonl" --grid "$grid" \
    --out-prefix "$prefix" --allow-download "$@"
}

for m in "${MODELS[@]}"; do
  for f in data/*.jsonl; do
    ds=$(basename "$f" .jsonl)
    for grid in mechanism role controls paraphrase preamble userattr; do
      run "$m" "$ds" "$grid"
    done
    run "$m" "$ds" mechanism --target random
  done
done

for m in "${ANCHOR_MODELS[@]}"; do
  for ds in "${ANCHOR_DATASETS[@]}"; do
    run "$m" "$ds" dose
    run "$m" "$ds" prefix
  done
done

python make_tables.py --runs out --out tables
