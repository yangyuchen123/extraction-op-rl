import paramiko, os

from server_env import (
    SERVER_HOST as host,
    SERVER_PORT as port,
    SERVER_USER as user,
    SERVER_PWD as pwd,
)

local_root = "/home/administrator/extraction_ops_archive/b1"
os.makedirs(local_root, exist_ok=True)

remote_files = {
    "/root/autodl-tmp/eval_archive/b1/sweep_summary.json": "sweep_summary.json",
    "/root/autodl-tmp/eval_archive/b1/training_metrics.json": "training_metrics.json",
    "/root/autodl-tmp/eval_archive/b1/eval_step1.json": "eval_step1.json",
    "/root/autodl-tmp/eval_archive/b1/eval_step2.json": "eval_step2.json",
    "/root/autodl-tmp/eval_archive/b1/eval_step3.json": "eval_step3.json",
    "/root/autodl-tmp/eval_archive/b1/eval_step4.json": "eval_step4.json",
    "/root/autodl-tmp/eval_archive/b1/eval_step5.json": "eval_step5.json",
    "/root/autodl-tmp/extraction_ops_b1_pilot3.log": "training_pilot3.log",
    "/root/autodl-tmp/clean_ckpts.log": "clean_ckpts.log",
}

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(host, port=port, username=user, password=pwd, timeout=30)
sftp = client.open_sftp()
for remote, name in remote_files.items():
    local = os.path.join(local_root, name)
    try:
        sftp.get(remote, local)
        size = os.path.getsize(local)
        print(f"downloaded {name} ({size//1024} KB)")
    except Exception as e:
        print(f"FAILED {name}: {e}")
sftp.close()
client.close()
print(f"\nall files -> {local_root}")
