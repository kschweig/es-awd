#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

RUN_DIR="${RUN_DIR:-grpo_experiments/runs/Qwen2.5-3B-Instruct_grpo_countdown_seed44}"
DEVICE_ID="${DEVICE_ID:-0}"
SUBSET_SIZE="${SUBSET_SIZE:-2048}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

DATASETS=(
  countdown
  gsm8k
  proofwriter
  hellaswag
  piqa
  arc-challenge
  mmlu-pro
)

SUBSET_DATASETS=(
  hellaswag
  piqa
  arc-challenge
  mmlu-pro
)

SCRIPT_ARGS=()
PRINT_OUTPUT_FILES=0

for arg in "$@"; do
  case "$arg" in
    --print-output-files)
      PRINT_OUTPUT_FILES=1
      ;;
    *)
      SCRIPT_ARGS+=("$arg")
      ;;
  esac
done

output_path_for_dataset() {
  local dataset=$1
  printf '%s/%s.jsonl\n' "${RUN_DIR%/}" "$dataset"
}

dataset_uses_subset_size() {
  local dataset=$1

  for subset_dataset in "${SUBSET_DATASETS[@]}"; do
    if [[ "$subset_dataset" == "$dataset" ]]; then
      return 0
    fi
  done

  return 1
}

if [[ "$PRINT_OUTPUT_FILES" == "1" ]]; then
  for dataset in "${DATASETS[@]}"; do
    output_path_for_dataset "$dataset"
  done
  exit 0
fi

for dataset in "${DATASETS[@]}"; do
  output_path=$(output_path_for_dataset "$dataset")

  if [[ "$SKIP_EXISTING" == "1" && -f "$output_path" ]]; then
    echo "Skipping $dataset for $RUN_DIR because $output_path already exists."
    continue
  fi

  cmd=(
    python "$REPO_ROOT/evaluation/evaluate_forgetting.py"
    --run_dir "$RUN_DIR"
    --cuda_devices "$DEVICE_ID"
    --dataset "$dataset"
  )

  if dataset_uses_subset_size "$dataset"; then
    cmd+=(--subset_size "$SUBSET_SIZE")
  fi

  cmd+=("${SCRIPT_ARGS[@]}")

  echo "Running ${dataset} evaluation for $RUN_DIR"
  "${cmd[@]}"
done
