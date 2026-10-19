#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"
python -m pip install -e ./visual_token_merger/vl_swift_merger/transformers -e ./visual_token_merger/vl_swift_merger/ms-swift
python -m pip install 'qwen_vl_utils>=0.0.14' deepspeed
# The training launchers use flash_attn. Install after PyTorch and the CUDA toolkit.
python -m pip install flash-attn --no-build-isolation
