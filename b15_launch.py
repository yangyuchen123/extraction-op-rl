"""B1.5 launch: upload SFT script + clean_ckpts, then start 0.6B multi-epoch SFT."""
import paramiko
import os

host = os.environ.get("GPU_SERVER_HOST", "connect.nmb2.seetacloud.com")
port = int(os.environ.get("GPU_SERVER_PORT", "14970"))
user = os.environ.get("GPU_SERVER_USER", "root")
pwd = os.environ["GPU_SERVER_PWD"]  # 必填，通过环境变量提供，勿硬编码

upload = {
    "/home/administrator/extraction_ops_stage/run_extraction_ops_random_qwen3_06b_lora.sh":
        "/root/run_extraction_ops_random_qwen3_06b_lora.sh",
    "/home/administrator/extraction_ops_stage/clean_ckpts.py":
        "/root/clean_ckpts.py",
}

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(host, port=port, username=user, password=pwd, timeout=30)
sftp = client.open_sftp()
for local, remote in upload.items():
    sftp.put(local, remote)
    print(f"uploaded {local} -> {remote}")
sftp.close()

# Launch: clean_ckpts watchdog (LoRA checkpoints are tiny; this is a safety net)
# + 0.6B SFT with EPOCHS=8, new SAVE_DIR, log to data disk.
cmd = (
    "cd /root && "
    "nohup python3 /root/clean_ckpts.py "
    "/root/autodl-tmp/checkpoints/extraction_ops_random_sft_qwen3_06b_lora_epochs8 "
    "/root/autodl-tmp/adapters/b15_epochs8 "
    ">> /root/autodl-tmp/clean_ckpts_b15.log 2>&1 & "
    "echo 'watchdog started'; "
    "cd /root/verl-agent && "
    "EPOCHS=8 "
    "SAVE_DIR=/root/autodl-tmp/checkpoints/extraction_ops_random_sft_qwen3_06b_lora_epochs8 "
    "nohup bash /root/run_extraction_ops_random_qwen3_06b_lora.sh "
    "> /root/autodl-tmp/b15_epochs8_train.log 2>&1 & "
    "echo 'training launched'; sleep 5; "
    "ps aux | grep -E 'torchrun|fsdp_sft' | grep -v grep | head -3"
)
stdin, stdout, stderr = client.exec_command(cmd, timeout=120)
out = stdout.read().decode("utf-8", "replace")
err = stderr.read().decode("utf-8", "replace")
print(out)
if err:
    print("STDERR:", err[-2000:])
client.close()
