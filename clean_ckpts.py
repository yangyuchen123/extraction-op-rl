"""Watch a verl checkpoint dir and keep only the per-step LoRA adapters.

The full FSDP checkpoint is ~3 GB per step (model + optimizer state), while the
LoRA adapter is ~40 MB. During a save_freq=1 run this would exhaust the data
disk, so this background process copies each step's lora_adapter into a compact
archive dir and deletes the full checkpoint immediately.

Usage:
  python clean_ckpts.py <checkpoint_experiment_dir> <adapter_archive_dir>
"""
from __future__ import annotations

import glob
import os
import shutil
import sys
import time


def main() -> None:
    watch_dir = sys.argv[1]
    archive_dir = sys.argv[2]
    os.makedirs(archive_dir, exist_ok=True)
    seen: set[str] = set()
    while True:
        for step_dir in sorted(glob.glob(os.path.join(watch_dir, "global_step_*"))):
            name = os.path.basename(step_dir)
            lora = os.path.join(step_dir, "actor", "lora_adapter")
            if name in seen or not os.path.isdir(lora):
                continue
            dest = os.path.join(archive_dir, name)
            if not os.path.isdir(dest):
                shutil.copytree(lora, dest)
            seen.add(name)
            shutil.rmtree(step_dir, ignore_errors=True)
            print(f"[clean_ckpts] archived {name} adapter, removed full checkpoint", flush=True)
        time.sleep(15)


if __name__ == "__main__":
    main()
