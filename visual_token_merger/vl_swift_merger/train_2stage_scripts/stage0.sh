#!/bin/bash
# Stage 0: vision projector (train aligner only) | LLaVA-558K-Webdataset
# component: vision projector  (freeze_vit=true, freeze_llm=true, freeze_aligner=false)
# Set NNODES, NODE_RANK and MASTER_ADDR for multi-node training.
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TRAINING_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export QWEN3_MERGER_SIZE="${QWEN3_MERGER_SIZE:-3}"
source "${TRAINING_ROOT}/train_2stage_scripts/setup_env.sh.inc"
export QWEN3_MERGER_INIT=random

export FPS_MAX_FRAMES=64

swift sft \
    --model "${MODEL_PATH}" \
    --model_type qwen3_vl \
    --dataset "${STAGE0_DATASET}" \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --attn_impl flash_attn \
    --padding_free true \
    --learning_rate 1e-3 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr": 1e-5}' \
    --weight_decay 0.01 \
    --freeze_vit true \
    --freeze_aligner false \
    --freeze_llm true \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing true \
    --gradient_accumulation_steps 4 \
    --save_steps 1000 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --max_length 16384 \
    --output_dir "${MERGER_RUN_DIR}/stage0" \
    --warmup_ratio 0.03 \
    --deepspeed zero1 \
    --dataset_num_proc 8 \
    --dataloader_num_workers 4 \
    --add_version false
