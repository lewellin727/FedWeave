#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

TRAIN_DATASETS="hotpotqa:bridge,hotpotqa:comparison,2wikimultihopqa:bridge_comparison,2wikimultihopqa:comparison,2wikimultihopqa:inference,2wikimultihopqa:compositional"

echo "Training document LoRAs"
"${PYTHON_BIN}" main.py \
    --mode train \
    --stage offline \
    --datasets "${TRAIN_DATASETS}" \
    --model_name "${MODEL_NAME}" \
    --augment_model "${AUGMENT_MODEL}" \
    --max_entries 500

train_caa() {
    local checkpoint_profile="$1"
    local caa_alpha="$2"
    local checkpoint_path="${CAA_ROOT}/${checkpoint_profile}/caa.pt"
    local best_checkpoint_path="${checkpoint_path%.pt}_best.pt"

    if [[ -f "${checkpoint_path}" || -f "${best_checkpoint_path}" ]]; then
        echo "Skipping ${checkpoint_profile}: checkpoint already exists."
        return
    fi

    echo "Training CAA ${checkpoint_profile}"
    "${PYTHON_BIN}" main.py \
        --mode train \
        --stage caa \
        --datasets "${TRAIN_DATASETS}" \
        --model_name "${MODEL_NAME}" \
        --augment_model "${AUGMENT_MODEL}" \
        --caa_k_per_combo 500 \
        --caa_num_epochs 9 \
        --caa_alpha "${caa_alpha}" \
        --caa_save_path "${checkpoint_path}"
}

train_caa alpha_0.1 0.1
train_caa alpha_0.5 0.5
train_caa alpha_1.0 1.0
