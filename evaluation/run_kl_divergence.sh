#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
DRY_RUN="${DRY_RUN:-0}"
SUBSET_SIZE="${SUBSET_SIZE:-50}"
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

usage() {
  cat <<'EOF'
Usage: bash evaluation/run_kl_divergence.sh <folder> <gpu> [evaluate_kl_divergence args...]

If <folder> is already an ES or GRPO run directory, the script runs
`evaluate_kl_divergence.py` for the standard evaluation datasets in that folder.

Otherwise it searches recursively for eligible ES / GRPO subfolders and runs
the KL divergence evaluation for each one.

Extra arguments are forwarded to `evaluation/evaluate_kl_divergence.py`.

Environment:
  DRY_RUN=1        Print commands without executing them.
  SKIP_EXISTING=1  Skip datasets whose `<dataset>_kl.jsonl` already exists.
  SUBSET_SIZE=50   Subset size for evaluation datasets.
EOF
}

if [[ $# -gt 0 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then
  usage
  exit 0
fi

if [[ $# -lt 2 ]]; then
  usage >&2
  exit 1
fi

TARGET_DIR_INPUT=$1
CUDA_DEVICES_VALUE=$2
shift 2

FORWARD_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      ;;
    --no-dry-run)
      DRY_RUN=0
      ;;
    --skip-existing)
      SKIP_EXISTING=1
      ;;
    --no-skip-existing)
      SKIP_EXISTING=0
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      FORWARD_ARGS+=("$1")
      ;;
  esac
  shift
done

TARGET_DIR=$(python3 - "$TARGET_DIR_INPUT" <<'PY'
import sys
from pathlib import Path

print(Path(sys.argv[1]).expanduser().resolve())
PY
)

if [[ ! -d "$TARGET_DIR" ]]; then
  echo "Target folder does not exist: $TARGET_DIR" >&2
  exit 1
fi

output_path_for_dataset() {
  local run_dir=$1
  local dataset=$2

  printf '%s/%s_kl.jsonl\n' "${run_dir%/}" "$dataset"
}

list_eligible_run_dirs() {
  python3 - "$TARGET_DIR" <<'PY'
import os
import re
import sys
from pathlib import Path

target = Path(sys.argv[1])
step_pattern = re.compile(r"_global_step_(\d+)$")


def has_grpo_checkpoints(path: Path) -> bool:
    try:
        children = sorted(path.iterdir())
    except OSError:
        return False

    for child in children:
        if not child.is_dir():
            continue
        if step_pattern.search(child.name) and (child / "config.json").is_file():
            return True
    return False


def is_eligible_run_dir(path: Path) -> bool:
    is_es = (path / "args.json").is_file() and (path / "iteration_updates").is_dir()
    is_grpo = (path / "model.txt").is_file() and has_grpo_checkpoints(path)
    return is_es or is_grpo


if is_eligible_run_dir(target):
    print(target)
    raise SystemExit(0)

run_dirs: list[Path] = []
for current_dir, dirnames, _filenames in os.walk(target):
    path = Path(current_dir)
    if is_eligible_run_dir(path):
        run_dirs.append(path)
        dirnames[:] = []
        continue
    dirnames[:] = sorted(dirnames)

for run_dir in sorted(run_dirs):
    print(run_dir)
PY
}

run_kl_for_dir() {
  local run_dir=$1
  local dataset
  local output_path
  local cmd

  for dataset in "${DATASETS[@]}"; do
    output_path=$(output_path_for_dataset "$run_dir" "$dataset")

    if [[ "$SKIP_EXISTING" == "1" && -f "$output_path" ]]; then
      echo "Skipping $dataset for $run_dir because $output_path already exists."
      continue
    fi

    cmd=(
      python "$REPO_ROOT/evaluation/evaluate_kl_divergence.py"
      --run_dir "$run_dir"
      --cuda_devices "$CUDA_DEVICES_VALUE"
      --dataset "$dataset"
    )

    cmd+=(--subset_size "$SUBSET_SIZE")

    cmd+=("${FORWARD_ARGS[@]}")

    echo "Running ${dataset} KL divergence evaluation for $run_dir"
    "${cmd[@]}"
  done
}

print_run_kl_command() {
  local run_dir=$1
  local dataset
  local output_path

  for dataset in "${DATASETS[@]}"; do
    output_path=$(output_path_for_dataset "$run_dir" "$dataset")

    if [[ "$SKIP_EXISTING" == "1" && -f "$output_path" ]]; then
      echo "Skipping $dataset for $run_dir because $output_path already exists."
      continue
    fi

    printf '%q ' \
      python "$REPO_ROOT/evaluation/evaluate_kl_divergence.py" \
      --run_dir "$run_dir" \
      --cuda_devices "$CUDA_DEVICES_VALUE" \
      --dataset "$dataset"
    printf '%q ' --subset_size "$SUBSET_SIZE"
    if [[ ${#FORWARD_ARGS[@]} -gt 0 ]]; then
      printf '%q ' "${FORWARD_ARGS[@]}"
    fi
    printf '\n'
  done
}

mapfile -t RUN_DIRS < <(list_eligible_run_dirs)

if [[ ${#RUN_DIRS[@]} -eq 0 ]]; then
  echo "No ES/GRPO run directories found under $TARGET_DIR."
  exit 0
fi

if [[ ${#RUN_DIRS[@]} -eq 1 && "${RUN_DIRS[0]}" == "$TARGET_DIR" ]]; then
  echo "Target folder is an eligible run directory."
else
  echo "Found ${#RUN_DIRS[@]} eligible run directories under $TARGET_DIR."
fi

for run_dir in "${RUN_DIRS[@]}"; do
  echo
  echo "==> $(basename "$run_dir")"
  if [[ "$DRY_RUN" == "1" ]]; then
    print_run_kl_command "$run_dir"
  else
    run_kl_for_dir "$run_dir"
  fi
done
