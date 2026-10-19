#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
if [[ $# -lt 1 || "$1" == --help ]]; then
  echo "Usage: bash $CODE_DIR/scripts/train.sh VARIANT [training overrides...]"
  echo "Paths (CORPUS_PATH, OUTPUT_DIR, TOKENIZER) are relative to the clone root."
  exit 0
fi
VARIANT=$1; shift
validate_variant "$VARIANT"
case "$VARIANT" in
  *_345m)
    RECIPE=pretrain_345M_8K_30B.yaml
    DEFAULT_STEPS=30000; DEFAULT_BATCH=128; DEFAULT_MICRO=4
    DEFAULT_TRAIN_SIZE=10000000000; DEFAULT_CKPT=false ;;
  *_1b)
    RECIPE=pretrain_1b_8K_100B.yaml
    DEFAULT_STEPS=43000; DEFAULT_BATCH=256; DEFAULT_MICRO=2
    DEFAULT_TRAIN_SIZE=300000000000; DEFAULT_CKPT=true ;;
esac
CORPUS_PATH=${CORPUS_PATH:-data/tokenized}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/$VARIANT}
TOKENIZER=${TOKENIZER:-$CODE_DIR/configs/olmo3_vocab}
if [[ "${DRY_RUN:-0}" != 1 && ! -d "$CORPUS_PATH" ]]; then
  echo "Tokenized corpus missing: $CORPUS_PATH (set CORPUS_PATH)." >&2; exit 2
fi
NNODES=${NNODES:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
NODE_RANK=${NODE_RANK:-0}
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
DEFAULT_SHARD_SIZE=$WORLD_SIZE
if [[ "$VARIANT" == *_1b && "$WORLD_SIZE" -ge 8 ]]; then DEFAULT_SHARD_SIZE=8; fi
SHARD_SIZE=${DATA_PARALLEL_SHARD_SIZE:-$DEFAULT_SHARD_SIZE}
if (( SHARD_SIZE < 1 || WORLD_SIZE % SHARD_SIZE != 0 )); then
  echo "DATA_PARALLEL_SHARD_SIZE must divide the total GPU count." >&2; exit 2
fi
launch=(--nnodes "$NNODES" --nproc_per_node "$NPROC_PER_NODE" --node_rank "$NODE_RANK")
if [[ "$NNODES" == 1 ]]; then
  launch+=(--standalone)
else
  : "${MASTER_ADDR:?Set MASTER_ADDR for multi-node training}"
  launch+=(--master_addr "$MASTER_ADDR" --master_port "${MASTER_PORT:-29500}")
fi
run_command "$PYTHON_BIN" -m torch.distributed.run "${launch[@]}" \
  "$CODE_DIR/tasks/pretrain_with_ruler.py" "$CODE_DIR/configs/training_recipes/$RECIPE" \
  --model.config_path "$CODE_DIR/configs/$VARIANT.json" \
  --model.tokenizer_path "$TOKENIZER" \
  --data.train_path "$CORPUS_PATH" --data.max_seq_len "${MAX_SEQ_LEN:-8192}" \
  --data.train_size "${TRAIN_SIZE:-$DEFAULT_TRAIN_SIZE}" \
  --data.data_type "${DATA_TYPE:-ruler_0.00}" --data.datasets_type olmo3 \
  --data.sort_files true --data.num_workers "${NUM_WORKERS:-16}" \
  --train.init_device "${INIT_DEVICE:-meta}" --train.data_parallel_mode fsdp2 \
  --train.data_parallel_shard_size "$SHARD_SIZE" --train.enable_full_shard true --train.use_wandb "${USE_WANDB:-false}" \
  --train.enable_gradient_checkpointing "${GRADIENT_CKPT:-$DEFAULT_CKPT}" \
  --train.rmpad false --train.rmpad_with_pos_ids false --train.enable_mixed_precision \
  --train.wandb_project TTR --train.wandb_name "$VARIANT" \
  --train.micro_batch_size "${MICRO_BATCH_SIZE:-$DEFAULT_MICRO}" \
  --train.global_batch_size "${GLOBAL_BATCH_SIZE:-$DEFAULT_BATCH}" \
  --train.lr 3e-4 --train.lr_min 3e-5 --train.no_decay_params norm bias embed \
  --train.save_epochs 1 --train.ulysses_parallel_size 1 --train.save_steps "${SAVE_STEPS:-5000}" \
  --train.max_steps "${MAX_STEPS:-$DEFAULT_STEPS}" \
  --train.load_checkpoint_path auto --train.output_dir "$OUTPUT_DIR" "$@"
