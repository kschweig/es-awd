#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/venv/bin/python"

usage() {
    cat <<'EOF'
Usage: ./grpo_experiments/merge_fsdp_to_hf.sh <experiment-dir> [output-root]

Merges all verl FSDP actor checkpoints found under:
  grpo_experiments/checkpoints/.../<experiment>/global_step*/actor

into standard Hugging Face model directories using:
  python -m verl.model_merger merge --backend fsdp

Arguments:
  experiment-dir  Path to the experiment directory that contains global_step* subdirs.
  output-root     Optional output root. Defaults to:
                  grpo_experiments/merged
EOF
}

resolve_existing_dir() {
    local path="$1"
    if [[ ! -d "${path}" ]]; then
        echo "Directory does not exist: ${path}" >&2
        exit 1
    fi
    (
        cd "${path}"
        pwd
    )
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage >&2
    exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi

EXPERIMENT_DIR="$(resolve_existing_dir "$1")"
EXPERIMENT_NAME="$(basename "${EXPERIMENT_DIR}")"

if [[ $# -eq 2 ]]; then
    OUTPUT_ROOT="$2"
else
    OUTPUT_ROOT="${REPO_ROOT}/grpo_experiments/merged"
fi

mkdir -p "${OUTPUT_ROOT}"
OUTPUT_ROOT="$(
    cd "${OUTPUT_ROOT}"
    pwd
)"

mapfile -t STEP_DIRS < <(find "${EXPERIMENT_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'global_step*' | sort -V)

if [[ ${#STEP_DIRS[@]} -eq 0 ]]; then
    echo "No global_step* directories found under ${EXPERIMENT_DIR}" >&2
    exit 1
fi

echo "Merging FSDP checkpoints:"
echo "  experiment dir: ${EXPERIMENT_DIR}"
echo "  output root:    ${OUTPUT_ROOT}"
echo

for STEP_DIR in "${STEP_DIRS[@]}"; do
    STEP_NAME="$(basename "${STEP_DIR}")"
    ACTOR_DIR="${STEP_DIR}/actor"
    TARGET_DIR="${OUTPUT_ROOT}/${EXPERIMENT_NAME}_${STEP_NAME}"

    if [[ ! -d "${ACTOR_DIR}" ]]; then
        echo "Skipping ${STEP_DIR}: missing actor directory" >&2
        continue
    fi

    if [[ ! -f "${ACTOR_DIR}/fsdp_config.json" ]]; then
        echo "Skipping ${ACTOR_DIR}: missing fsdp_config.json" >&2
        continue
    fi

    if [[ ! -d "${ACTOR_DIR}/huggingface" ]]; then
        echo "Skipping ${ACTOR_DIR}: missing huggingface directory" >&2
        continue
    fi

    shopt -s nullglob
    MODEL_SHARDS=("${ACTOR_DIR}"/model_world_size_*_rank_*.pt)
    shopt -u nullglob

    if [[ ${#MODEL_SHARDS[@]} -eq 0 ]]; then
        echo "Skipping ${ACTOR_DIR}: no model shards found" >&2
        continue
    fi

    mkdir -p "${TARGET_DIR}"

    echo "Processing ${STEP_NAME}"
    echo "  actor dir:  ${ACTOR_DIR}"
    echo "  target dir: ${TARGET_DIR}"

    "${PYTHON_BIN}" -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "${ACTOR_DIR}" \
        --target_dir "${TARGET_DIR}"

    echo
done
