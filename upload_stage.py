import paramiko
import os

host = os.environ.get("GPU_SERVER_HOST", "connect.nmb2.seetacloud.com")
port = int(os.environ.get("GPU_SERVER_PORT", "14970"))
user = os.environ.get("GPU_SERVER_USER", "root")
pwd = os.environ["GPU_SERVER_PWD"]  # 必填，通过环境变量提供，勿硬编码

files = {
    "/home/administrator/extraction_ops_stage/generator.py":
        "/root/verl-agent/agent_system/environments/env_package/extraction_ops/generator.py",
    "/home/administrator/extraction_ops_stage/expert.py":
        "/root/verl-agent/agent_system/environments/env_package/extraction_ops/expert.py",
    "/home/administrator/extraction_ops_stage/apply_world_edits.py":
        "/tmp/apply_world_edits.py",
    "/home/administrator/extraction_ops_stage/check_extraction_ops_random_maps.py":
        "/root/verl-agent/examples/evaluation/check_extraction_ops_random_maps.py",
    "/home/administrator/extraction_ops_stage/test_extraction_ops_random_maps.py":
        "/root/verl-agent/tests/environments/test_extraction_ops_random_maps.py",
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
print("all files uploaded")
