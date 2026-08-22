import paramiko, os, sys

host = os.environ.get("GPU_SERVER_HOST", "connect.nmb2.seetacloud.com")
port = int(os.environ.get("GPU_SERVER_PORT", "14970"))
user = os.environ.get("GPU_SERVER_USER", "root")
pwd = os.environ["GPU_SERVER_PWD"]  # 必填，通过环境变量提供，勿硬编码
timeout = int(sys.argv[1]) if len(sys.argv) > 1 else 900

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(host, port=port, username=user, password=pwd, timeout=30)
cmd = (
    "while pgrep -f fsdp_sft_trainer > /dev/null; do sleep 20; done; "
    "echo '=== SFT 完成，日志尾部 ==='; tail -8 /root/autodl-tmp/extraction_ops_random_sft.log; "
    "echo '=== checkpoint ==='; ls /root/autodl-tmp/checkpoints/extraction_ops_random_sft_qwen3_06b_lora/ 2>/dev/null"
)
stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
out = stdout.read().decode("utf-8", "replace")
err = stderr.read().decode("utf-8", "replace")
print(out)
if err:
    print("STDERR:", err)
client.close()
