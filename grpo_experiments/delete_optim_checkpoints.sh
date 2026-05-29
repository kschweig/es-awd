#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <subfolder-under-grpo_experiments>" >&2
    exit 1
fi

TARGET_DIR="${SCRIPT_DIR}/$1"

if [[ ! -d "${TARGET_DIR}" ]]; then
    echo "Subfolder does not exist: ${TARGET_DIR}" >&2
    exit 1
fi

MATCH_COUNT="$(find "${TARGET_DIR}" -type f -name 'optim_*.pt' | wc -l)"

if [[ "${MATCH_COUNT}" -eq 0 ]]; then
    echo "No optim_*.pt files found under ${TARGET_DIR}"
    exit 0
fi

echo "Deleting ${MATCH_COUNT} optim_*.pt file(s) under ${TARGET_DIR}:"
find "${TARGET_DIR}" -type f -name 'optim_*.pt' -print -delete
