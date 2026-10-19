#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
# Install CUDA-enabled PyTorch for your system before running this script.
"$PYTHON_BIN" -m pip install -r "$CODE_DIR/requirements.txt"
"$PYTHON_BIN" -m pip install flash-attn==2.8.3 --no-build-isolation
"$PYTHON_BIN" -m pip install -e "$HARNESS_DIR[hf,ruler]"
