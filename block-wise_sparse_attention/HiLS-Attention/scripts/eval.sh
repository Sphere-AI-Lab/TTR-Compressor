#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
if [[ $# -lt 2 || "$1" == --help ]]; then
  echo "Usage: bash $CODE_DIR/scripts/eval.sh VARIANT general|ruler [lm-eval overrides...]"
  echo "MODEL overrides the HF checkpoint path; all paths are relative to the clone root."
  exit 0
fi
VARIANT=$1; SUITE=$2; shift 2
validate_variant "$VARIANT"
case "$SUITE" in general|ruler) ;; *) echo "Only general and ruler are supported." >&2; exit 2 ;; esac
case "$VARIANT" in *_1b) DEFAULT_STEP=43000 ;; *) DEFAULT_STEP=30000 ;; esac
MODEL=${MODEL:-outputs/$VARIANT/checkpoints/global_step_${GLOBAL_STEP:-$DEFAULT_STEP}/hf_ckpt}
TOKENIZER=${TOKENIZER:-$CODE_DIR/configs/olmo3_vocab}
OUTPUT=${OUTPUT:-outputs/$VARIANT/eval/$SUITE}
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  "$PYTHON_BIN" "$CODE_DIR/scripts/validate_checkpoint.py" "$CODE_DIR/configs/$VARIANT.json" "$MODEL"
fi
launch=()
if [[ "${NUM_PROCESSES:-1}" -gt 1 ]]; then
  launch=(-m accelerate.commands.launch --multi_gpu --num_processes "$NUM_PROCESSES")
fi
if [[ "$SUITE" == general ]]; then
  TASKS=wikitext,lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge,boolq,sciq,copa,openbookqa
  LENGTH=8192; BATCH=${BATCH_SIZE:-auto}; extra=()
else
  TASKS=${RULER_TASKS:-niah_single_1,niah_single_2,niah_single_3,niah_multikey_1}
  LENGTH=32768; BATCH=${BATCH_SIZE:-1}
  extra=(--metadata '{"max_seq_lengths":[4096,8192,16384,32768]}')
fi
run_command "$PYTHON_BIN" ${launch[@]+"${launch[@]}"} "$CODE_DIR/eval/eval_lm_harness.py" \
  --model hils_attn_hf --model_args "pretrained=$MODEL,tokenizer=$TOKENIZER,dtype=bfloat16,max_length=$LENGTH,attn_implementation=${ATTN_IMPLEMENTATION:-flash_attention_2}" \
  --tasks "$TASKS" --batch_size "$BATCH" --num_fewshot 0 --device cuda \
  --output_path "$OUTPUT" --show_config ${extra[@]+"${extra[@]}"} "$@"
