#!/bin/bash
# Stage 1: continue from the latest stage0 checkpoint without reinitializing mergers
# Set NNODES, NODE_RANK and MASTER_ADDR for multi-node training.
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TRAINING_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export QWEN3_MERGER_SIZE="${QWEN3_MERGER_SIZE:-3}"
source "${TRAINING_ROOT}/train_2stage_scripts/setup_env.sh.inc"
unset QWEN3_MERGER_INIT

export FPS_MAX_FRAMES=64

LAST_STAGE_DIR="${LAST_STAGE_DIR:-${MERGER_RUN_DIR}/stage0}"
LAST_CKPT=$(latest_checkpoint "${LAST_STAGE_DIR}")
echo "Loading from: ${LAST_CKPT}"

swift sft \
    --model "${LAST_CKPT}" \
    --model_type qwen3_vl \
    --dataset "${STAGE1_DATASET}" \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --attn_impl flash_attn \
    --padding_free true \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr": 0}' \
    --weight_decay 0.02 \
    --freeze_vit true \
    --freeze_aligner false \
    --freeze_llm false \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing true \
    --gradient_accumulation_steps 8 \
    --save_steps 1000 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --max_length 16384 \
    --output_dir "${MERGER_RUN_DIR}/stage1" \
    --warmup_ratio 0.03 \
    --deepspeed zero1 \
    --dataset_num_proc 8 \
    --dataloader_num_workers 4 \
    --add_version false
