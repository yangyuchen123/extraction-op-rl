#!/bin/bash
# Launch B1 training (save every step) + checkpoint cleaner.
set -e
export SAVE_FREQ=1
cd /root/verl-agent

setsid bash examples/gigpo_trainer/run_extraction_ops_random_b1.sh \
  > /root/autodl-tmp/extraction_ops_b1_pilot3.log 2>&1 < /dev/null &
echo "training launched pid=$!"

setsid /root/miniconda3/envs/verl-agent/bin/python \
  /root/verl-agent/examples/gigpo_trainer/clean_ckpts.py \
  /root/autodl-tmp/checkpoints/verl_agent_extraction_ops_b1 \
  /root/autodl-tmp/adapters/b1 \
  > /root/autodl-tmp/clean_ckpts.log 2>&1 < /dev/null &
echo "cleaner launched pid=$!"
