import paramiko

from server_env import (
    SERVER_HOST as host,
    SERVER_PORT as port,
    SERVER_USER as user,
    SERVER_PWD as pwd,
)

files = {
    "/home/administrator/extraction_ops_stage/evaluate_extraction_ops_three_layer.py":
        "/root/verl-agent/examples/evaluation/evaluate_extraction_ops_three_layer.py",
    "/home/administrator/extraction_ops_stage/run_three_layer_sweep.py":
        "/root/verl-agent/examples/evaluation/run_three_layer_sweep.py",
    "/home/administrator/extraction_ops_stage/parse_training_log.py":
        "/root/verl-agent/examples/evaluation/parse_training_log.py",
    "/home/administrator/extraction_ops_stage/clean_ckpts.py":
        "/root/verl-agent/examples/gigpo_trainer/clean_ckpts.py",
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
print("done")
