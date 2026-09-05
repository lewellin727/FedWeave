#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${REPO_ROOT}/config.yaml"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${REPO_ROOT}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python executable not found: ${PYTHON_BIN}. Activate the FedWeave environment or set PYTHON_BIN." >&2
    exit 1
fi

SAVE_DIR="$("${PYTHON_BIN}" -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["train"]["save_dir"])' "${CONFIG_PATH}")"
if [[ "${SAVE_DIR}" == /path/to/* ]]; then
    echo "Please set train.save_dir in config.yaml before running the experiments." >&2
    exit 1
fi

CAA_ROOT="${SAVE_DIR}/caa"
MODEL_NAME="llama3.2-1b-instruct"
AUGMENT_MODEL="llama3.2-1b-instruct"
NUM_SILOS=6
K=5

run_fedweave_eval() {
    local dataset="$1"
    local dataset_type="$2"
    local checkpoint_profile="$3"
    local caa_alpha="$4"
    local coverage_threshold="$5"
    local checkpoint_path="${CAA_ROOT}/${checkpoint_profile}/caa.pt"
    local best_checkpoint_path="${checkpoint_path%.pt}_best.pt"

    if [[ ! -f "${checkpoint_path}" && ! -f "${best_checkpoint_path}" ]]; then
        echo "CAA checkpoint not found for ${checkpoint_profile}. Run train.sh first." >&2
        exit 1
    fi

    local command=(
        "${PYTHON_BIN}" main.py
        --mode test
        --stage all
        --datasets "${dataset}:${dataset_type}"
        --model_name "${MODEL_NAME}"
        --augment_model "${AUGMENT_MODEL}"
        --num_silos "${NUM_SILOS}"
        --k "${K}"
        --caa_alpha "${caa_alpha}"
        --caa_save_path "${checkpoint_path}"
    )

    if [[ "${coverage_threshold}" != "none" ]]; then
        command+=(--coverage_min_gain "${coverage_threshold}")
    fi

    echo "Running ${dataset}:${dataset_type}"
    "${command[@]}"
}
