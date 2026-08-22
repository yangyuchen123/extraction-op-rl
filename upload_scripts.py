import paramiko
import os

host = os.environ.get("GPU_SERVER_HOST", "connect.nmb2.seetacloud.com")
port = int(os.environ.get("GPU_SERVER_PORT", "14970"))
user = os.environ.get("GPU_SERVER_USER", "root")
pwd = os.environ["GPU_SERVER_PWD"]  # 必填，通过环境变量提供，勿硬编码

files = {
    "/home/administrator/extraction_ops_stage/evaluate_extraction_ops_three_layer.py":
        "/root/verl-agent/examples/evaluation/evaluate_extraction_ops_three_layer.py",
    "/home/administrator/extraction_ops_stage/run_three_layer_sweep.py":
        "/root/verl-agent/examples/evaluation/run_three_layer_sweep.py",
    "/home/administrator/extraction_ops_stage/parse_training_log.py":
        "/root/verl-agent/examples/evaluation/parse_training_log.py",
    "/home/administrator/extraction_ops_stage/clean_ckpts.py":
        "/root/verl-agent/examples/gigpo_trainer/clean_ckpts.py",
}

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(host, port=port, username=user, password=pwd, timeout=30)
sftp = client.open_sftp()
for local, remote in files.items():
    sftp.put(local, remote)
    print(f"uploaded {local} -> {remote}")
sftp.close()
client.close()
print("done")
