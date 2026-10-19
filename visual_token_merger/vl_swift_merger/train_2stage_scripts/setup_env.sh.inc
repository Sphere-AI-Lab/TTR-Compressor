# Shared launcher configuration. All user paths are relative to the repository root.
TRAINING_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${TRAINING_ROOT}/../.." && pwd)"
cd "${REPO_ROOT}"
export QWEN3_MERGER_VARIANT="${QWEN3_MERGER_VARIANT:-ttr}"
export QWEN3_MERGER_SIZE="${QWEN3_MERGER_SIZE:-3}"
case "${QWEN3_MERGER_VARIANT}" in
    mlp|ttr|ttr_conv) ;;
    *) echo "Unknown QWEN3_MERGER_VARIANT: ${QWEN3_MERGER_VARIANT}" >&2; exit 1 ;;
esac
case "${QWEN3_MERGER_SIZE}" in
    2|3) ;;
    *) echo "QWEN3_MERGER_SIZE must be 2 or 3" >&2; exit 1 ;;
esac
export PYTHONPATH="${TRAINING_ROOT}/transformers/src:${TRAINING_ROOT}/ms-swift:${PYTHONPATH:-}"
case "${QWEN3_MERGER_VARIANT}" in
    mlp) merger_init_tag=lecun_normal_init ;;
    ttr|ttr_conv) merger_init_tag=pytorch_linear_default_init ;;
esac
export MERGER_RUN_DIR="${MERGER_RUN_DIR:-outputs/visual_token_merger/${QWEN3_MERGER_VARIANT}${QWEN3_MERGER_SIZE}_${QWEN3_MERGER_SIZE}/stage0_to_stage1_${merger_init_tag}}"
export MODEL_PATH="${MODEL_PATH:-models/Qwen3-VL-4B-Instruct-Init}"
export STAGE0_DATASET="${STAGE0_DATASET:-datasets/stage0_vision_projector_llava558k.jsonl}"
export STAGE1_DATASET="${STAGE1_DATASET:-datasets/stage1_video400k_image780k_m4multi200k_merged.jsonl}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${GPUS_PER_NODE}}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-${RANK:-0}}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

# Numeric sorting works on Linux and macOS, and ignores incomplete/non-checkpoint folders.
latest_checkpoint() {
    python - "$1" <<'PY'
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
checkpoints = [p for p in root.glob('checkpoint-*') if p.is_dir() and p.name[11:].isdigit()]
if not checkpoints:
    sys.exit(f'No checkpoint found under {root}; complete the previous stage first.')
print(max(checkpoints, key=lambda p: int(p.name[11:])))
PY
}
python -c 'import pathlib, os, transformers, swift; roots = os.environ["PYTHONPATH"].split(":"); modules = (transformers, swift)
for root, module in zip(roots, modules):
    expected, actual = pathlib.Path(root).resolve(), pathlib.Path(module.__file__).resolve()
    assert expected in actual.parents, (actual, expected)
    print("Using", module.__name__ + ":", actual)'
