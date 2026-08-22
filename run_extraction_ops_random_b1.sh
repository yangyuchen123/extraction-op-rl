#!/usr/bin/env bash
# Stage B1 pilot: random maps + outcome reward + conservative GiGPO.
# Verifies that RL does not collapse validation when the reward is
# result-oriented and the policy is warm-started from random-map SFT.
set -euo pipefail
set -x
source /root/miniconda3/etc/profile.d/conda.sh
conda activate verl-agent
cd /root/verl-agent

# Warm-started model: merged random-map SFT (created by run_extraction_ops_random_qwen3_06b_lora.sh + merge).
MODEL_PATH=${MODEL_PATH:-/root/autodl-tmp/models/Qwen3-0.6B-extraction-ops-random-sft}

# Reward / world mode.
REWARD_SCHEME=${REWARD_SCHEME:-outcome}
RANDOM_MAPS=${RANDOM_MAPS:-true}
ENV_SEED=${ENV_SEED:-20260714}
MAX_ENV_STEPS=${MAX_ENV_STEPS:-60}

# Pilot size.
EPOCHS=${EPOCHS:-5}
TRAIN_SIZE=${TRAIN_SIZE:-8}
VAL_SIZE=${VAL_SIZE:-8}
GROUP_SIZE=${GROUP_SIZE:-4}

# Conservative RL (Stage B1 recipe: low LR + high KL).
LR=${LR:-2e-7}
KL_COEF=${KL_COEF:-0.05}

# Throughput profile (single 3090, no offload).
ENVS_PER_WORKER=${ENVS_PER_WORKER:-4}
ENV_CPUS_PER_WORKER=${ENV_CPUS_PER_WORKER:-0.25}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-8}
PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE:-8}
LOG_PROB_MICRO_BATCH_SIZE=${LOG_PROB_MICRO_BATCH_SIZE:-8}
PARAM_OFFLOAD=${PARAM_OFFLOAD:-false}
OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-false}
REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-false}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-false}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.45}
ENFORCE_EAGER=${ENFORCE_EAGER:-false}
VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
if [[ -n "$VLLM_ATTENTION_BACKEND" ]]; then
  export VLLM_ATTENTION_BACKEND
fi

# Validate every step during this short pilot.
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-true}
TEST_FREQ=${TEST_FREQ:-1}
SAVE_FREQ=${SAVE_FREQ:-5}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/root/autodl-tmp/checkpoints/verl_agent_extraction_ops_b1}

MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-64}
DATA_DIR=${DATA_DIR:-/root/data/verl-agent/extraction_ops_b1}

python examples/data_preprocess/prepare_extraction_ops.py \
  --output "$DATA_DIR" --train-size "$TRAIN_SIZE" --val-size "$VAL_SIZE"

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=gigpo \
  data.train_files="$DATA_DIR/train.parquet" \
  data.val_files="$DATA_DIR/test.parquet" \
  data.train_batch_size="$TRAIN_SIZE" \
  data.val_batch_size="$VAL_SIZE" \
  data.max_prompt_length=4096 \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.actor.optim.lr="$LR" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.lora_rank=16 \
  actor_rollout_ref.model.lora_alpha=32 \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.actor.ppo_mini_batch_size=$((TRAIN_SIZE * GROUP_SIZE)) \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef="$KL_COEF" \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.model.enable_gradient_checkpointing="$GRADIENT_CHECKPOINTING" \
  actor_rollout_ref.actor.fsdp_config.param_offload="$PARAM_OFFLOAD" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="$OPTIMIZER_OFFLOAD" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO_BATCH_SIZE" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager="$ENFORCE_EAGER" \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO_BATCH_SIZE" \
  actor_rollout_ref.ref.fsdp_config.param_offload="$REF_PARAM_OFFLOAD" \
  actor_rollout_ref.actor.use_invalid_action_penalty=False \
  algorithm.use_kl_in_reward=False \
  algorithm.gamma=0.95 \
  algorithm.gigpo.step_advantage_w=1.0 \
  algorithm.gigpo.mode=mean_std_norm \
  env.env_name=extraction_ops \
  env.seed="$ENV_SEED" \
  env.max_steps="$MAX_ENV_STEPS" \
  +env.random_maps="$RANDOM_MAPS" \
  +env.reward_scheme="$REWARD_SCHEME" \
  +env.envs_per_worker="$ENVS_PER_WORKER" \
  env.history_length=0 \
  env.rollout.n="$GROUP_SIZE" \
  env.resources_per_worker.num_cpus="$ENV_CPUS_PER_WORKER" \
  "trainer.logger=['console']" \
  trainer.project_name=verl_agent_extraction_ops_b1 \
  trainer.experiment_name=gigpo_random_outcome_pilot \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.default_local_dir="$CHECKPOINT_DIR" \
  trainer.default_hdfs_dir=null \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.total_epochs="$EPOCHS" \
  trainer.val_before_train="$VAL_BEFORE_TRAIN" \
  ray_init.num_cpus="$RAY_NUM_CPUS" \
  "$@"
