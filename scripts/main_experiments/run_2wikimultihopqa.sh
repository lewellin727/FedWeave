#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

run_fedweave_eval 2wikimultihopqa bridge_comparison alpha_0.1 0.05 0.5
run_fedweave_eval 2wikimultihopqa comparison alpha_0.1 0.05 2.0
run_fedweave_eval 2wikimultihopqa inference alpha_1.0 0.2 none
run_fedweave_eval 2wikimultihopqa compositional alpha_1.0 0.5 0.1
