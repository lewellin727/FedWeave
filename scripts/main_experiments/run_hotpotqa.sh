#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

run_fedweave_eval hotpotqa bridge alpha_0.5 0.05 5.0
run_fedweave_eval hotpotqa comparison alpha_0.1 0.05 none
