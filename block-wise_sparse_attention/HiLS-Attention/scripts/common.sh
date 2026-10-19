#!/usr/bin/env bash
# Resolve from this file so invocation is independent of the caller's cwd.
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$PROJECT_ROOT"
CODE_DIR=block-wise_sparse_attention/HiLS-Attention
HARNESS_DIR=block-wise_sparse_attention/lm-evaluation-harness
export PYTHONPATH="$PROJECT_ROOT/$CODE_DIR:$PROJECT_ROOT/$HARNESS_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-.cache/huggingface}"
export RULER_DATA_DIR="${RULER_DATA_DIR:-${HF_HOME}/ruler}"
export NLTK_DATA="${NLTK_DATA:-${RULER_DATA_DIR}/nltk_data}"
export TOKENIZERS_PARALLELISM=false
PYTHON_BIN=${PYTHON_BIN:-python3}
validate_variant() {
  case "$1" in
    hils_345m|ttr_345m|ttr_conv_345m|hils_1b|ttr_1b) ;;
    *) echo "Unknown variant: $1. Choose hils_345m, ttr_345m, ttr_conv_345m, hils_1b or ttr_1b." >&2; exit 2 ;;
  esac
}
run_command() {
  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' "$@"; printf '\n'
  else
    "$@"
  fi
}
