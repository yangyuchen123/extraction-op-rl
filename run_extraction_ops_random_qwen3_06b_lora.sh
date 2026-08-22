#!/usr/bin/env bash
# One-GPU SFT warm-start over random maps (Stage B1 prerequisite).
set -euo pipefail
set -x
source /root/miniconda3/etc/profile.d/conda.sh
conda activate verl-agent
cd /root/verl-agent

MODEL_PATH=${MODEL_PATH:-/root/autodl-tmp/models/Qwen3-0.6B}
EXPERT_DIR=${EXPERT_DIR:-/root/data/verl-agent/extraction_ops_random_expert}
SFT_DATA_DIR=${SFT_DATA_DIR:-/root/data/verl-agent/extraction_ops_random_sft}
SAVE_DIR=${SAVE_DIR:-/root/autodl-tmp/checkpoints/extraction_ops_random_sft_qwen3_06b_lora}
EPOCHS=${EPOCHS:-1}

mkdir -p "$SAVE_DIR"
torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  -m verl.trainer.fsdp_sft_trainer \
  data.train_files="$SFT_DATA_DIR/train.parquet" \
  data.val_files="$SFT_DATA_DIR/validation.parquet" \
  data.prompt_key=prompt \
  data.response_key=response \
  "data.prompt_dict_keys=[]" \
  "data.response_dict_keys=[]" \
  data.train_batch_size=32 \
  data.micro_batch_size_per_gpu=4 \
  data.max_length=2048 \
  data.truncation=error \
  model.partial_pretrain="$MODEL_PATH" \
  model.strategy=fsdp \
  model.lora_rank=16 \
  model.lora_alpha=32 \
  model.target_modules=all-linear \
  model.enable_gradient_checkpointing=True \
  model.fsdp_config.cpu_offload=True \
  optim.lr=1e-4 \
  trainer.default_local_dir="$SAVE_DIR" \
  trainer.default_hdfs_dir=null \
  trainer.project_name=extraction_ops_random_sft \
  trainer.experiment_name=qwen3_06b_lora_random_v1 \
  "trainer.logger=['console']" \
  trainer.total_epochs="$EPOCHS" \
  "$@"
