"""B1.5 prep: back up GiGPO eval artifacts to local, upload fixed eval script."""
import os
import paramiko

from server_env import (
    SERVER_HOST as host,
    SERVER_PORT as port,
    SERVER_USER as user,
    SERVER_PWD as pwd,
)

local_root = "/home/administrator/extraction_ops_archive/b1"
os.makedirs(local_root, exist_ok=True)

# Remote GiGPO eval results -> local backup (small, before deleting checkpoints).
backup = {
    "/root/autodl-tmp/extraction_ops_gigpo1_eval.json": "gigpo1_eval.json",
    "/root/autodl-tmp/extraction_ops_gigpo_procedural_20step.log": "gigpo_procedural_20step.log",
    "/root/autodl-tmp/extraction_ops_gigpo_sft_1step.log": "gigpo_sft_1step.log",
    "/root/autodl-tmp/extraction_ops_gigpo_step5_test16_eval.json": "gigpo_step5_eval.json",
    "/root/autodl-tmp/extraction_ops_gigpo_step10_test16_eval.json": "gigpo_step10_eval.json",
    "/root/autodl-tmp/extraction_ops_gigpo_step15_test16_eval.json": "gigpo_step15_eval.json",
    "/root/autodl-tmp/extraction_ops_gigpo_step20_test16_eval.json": "gigpo_step20_eval.json",
}

# Local (fixed) eval script -> remote.
upload = {
    "/home/administrator/extraction_ops_stage/evaluate_extraction_ops_three_layer.py":
        "/root/verl-agent/examples/evaluation/evaluate_extraction_ops_three_layer.py",
}

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(host, port=port, username=user, password=pwd, timeout=30)
sftp = client.open_sftp()

for remote, name in backup.items():
    local = os.path.join(local_root, name)
    try:
        sftp.get(remote, local)
        print(f"backed up {name} ({os.path.getsize(local)//1024} KB)")
    except Exception as e:
        print(f"BACKUP FAILED {name}: {e}")

for local, remote in upload.items():
    try:
        sftp.put(local, remote)
        print(f"uploaded {os.path.basename(local)} -> {remote}")
    except Exception as e:
        print(f"UPLOAD FAILED {local}: {e}")

sftp.close()
client.close()
print("done")
