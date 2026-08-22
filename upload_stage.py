import paramiko

from server_env import (
    SERVER_HOST as host,
    SERVER_PORT as port,
    SERVER_USER as user,
    SERVER_PWD as pwd,
)

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
    "/home/administrator/extraction_ops_stage/server_env.py":
        "/root/verl-agent/examples/evaluation/server_env.py",
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
