# Rethinking Sequence Compression with Test-time Regression

<p align="center">
  <a href="https://spherelab.ai/ttr-compressor">🌐 Homepage</a> |
  <a href="https://arxiv.org/pdf/https://arxiv.org/abs/XXXX.XXXXX">📑 Paper</a>  |
  <a href="https://arxiv.org/abs/https://arxiv.org/abs/XXXX.XXXXX">📖 arXiv</a> |
  <a href="https://huggingface.co/collections/SphereLab/ttr-compressor">🤗 Models</a> 
</p>


## 🔔News
🔥[2026-10-20]: We released our paper, codes, and models.

## Quick Start

### Block-wise Sparse Attention

Training and evaluation code for TTR block-wise sparse attention is in `block-wise_sparse_attention/HiLS-Attention`.

**Environment:** Follow the environment setup instructions in the [HiLS-Attention](https://github.com/Tencent-Hunyuan/HiLS-Attention) project. Run the commands below from this repository's root directory.

#### Training

Prepare tokenized `.data` shards in `data/tokenized` using `block-wise_sparse_attention/HiLS-Attention/preprocess/build_olmo3_datasets_from_parquet.py`, then launch training:

```bash
CORPUS_PATH=data/tokenized NPROC_PER_NODE=8 \
  bash block-wise_sparse_attention/HiLS-Attention/scripts/train.sh ttr_345m
```

Choose one of the following variants: `hils_345m`, `ttr_345m`, `ttr_conv_345m`, `hils_1b`, or `ttr_1b`. Set `NPROC_PER_NODE` to the number of GPUs.

#### Evaluation

```bash
MODEL=outputs/ttr_345m/checkpoints/global_step_30000/hf_ckpt \
  bash block-wise_sparse_attention/HiLS-Attention/scripts/eval.sh ttr_345m ruler
```

### Visual Token Merger

Training code for TTR visual token merging with Qwen3-VL is in `visual_token_merger/vl_swift_merger`.

**Environment:** Follow the environment setup instructions in the [ms-swift](https://github.com/modelscope/ms-swift) project, then install the bundled Transformers, ms-swift, and training dependencies. Run the commands below from this repository's root directory.

```bash
bash visual_token_merger/vl_swift_merger/train_2stage_scripts/install.sh
```

#### Training

Prepare the initial model weights and conversation [datasets](https://huggingface.co/datasets/sihany/ttr_vl_data_jsonl) in ms-swift JSONL format, then run stage0 followed by stage1:

```bash
# Replace the LLM weights in Qwen3-VL-4B-Instruct with those from Qwen3-4B-Instruct.
export MODEL_PATH=models/Qwen3-VL-4B-Instruct-Init
export STAGE0_DATASET=datasets/stage0_vision_projector_llava558k.jsonl
export STAGE1_DATASET=datasets/stage1_video400k_image780k_m4multi200k_merged.jsonl
export QWEN3_MERGER_VARIANT=ttr
export QWEN3_MERGER_SIZE=3
export NPROC_PER_NODE=8

bash visual_token_merger/vl_swift_merger/train_2stage_scripts/stage0.sh
# After stage0 finishes on all nodes:
bash visual_token_merger/vl_swift_merger/train_2stage_scripts/stage1.sh
```

Choose `ttr`, `ttr_conv` (with a depthwise 3×3 convolution on the value branch), or the `mlp` baseline. Set `QWEN3_MERGER_SIZE` to `2` or `3`, and `NPROC_PER_NODE` to the number of GPUs.

#### Evaluation

Use [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) for evaluation.



##  Citation

If you find our work and this codebase helpful, please consider starring this repo and cite:

```bibtex
@article{yang2026ttrcompressor,
  title={Rethinking Sequence Compression with Test-time Regression},
  author={Yang, Sihan and Huang, Jie and Li, Hanxuan and Guo, Yifan
          and Sun, Weigao and Zhong, Yiran and Liu, Weiyang},
  journal={arXiv preprint arXiv:2610.31574},
  year={2026}
}
```



## Contact
- Sihan Yang: sihany077@gmail.com