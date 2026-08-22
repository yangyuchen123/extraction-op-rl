"""Wait for B1.5 multi-epoch SFT to finish, then print tail + checkpoints.

Usage: python3 b15_wait.py [timeout_seconds]
"""
import paramiko
import os
import sys

host = os.environ.get("GPU_SERVER_HOST", "connect.nmb2.seetacloud.com")
port = int(os.environ.get("GPU_SERVER_PORT", "14970"))
user = os.environ.get("GPU_SERVER_USER", "root")
pwd = os.environ["GPU_SERVER_PWD"]  # 必填，通过环境变量提供，勿硬编码
timeout = int(sys.argv[1]) if len(sys.argv) > 1 else 3600

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(host, port=port, username=user, password=pwd, timeout=30)
cmd = (
    "while pgrep -f fsdp_sft_trainer > /dev/null; do sleep 30; done; "
    "echo '=== SFT DONE ==='; "
    "tr '\\r' '\\n' < /root/autodl-tmp/b15_epochs8_train.log | grep -E 'Epoch|val/loss|train/loss' | tail -6; "
    "echo '=== checkpoints ==='; "
    "ls /root/autodl-tmp/checkpoints/extraction_ops_random_sft_qwen3_06b_lora_epochs8/ 2>/dev/null; "
    "echo '=== disk ==='; df -h /root/autodl-tmp | tail -1"
)
stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
out = stdout.read().decode("utf-8", "replace")
err = stderr.read().decode("utf-8", "replace")
print(out)
if err:
    print("STDERR:", err[-1500:])
client.close()
