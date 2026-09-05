#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/run_hotpotqa.sh"
bash "${SCRIPT_DIR}/run_2wikimultihopqa.sh"
bash "${SCRIPT_DIR}/run_complexwebquestions.sh"
bash "${SCRIPT_DIR}/run_popqa.sh"
