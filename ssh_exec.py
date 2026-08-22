import paramiko, sys
from server_env import (
    SERVER_HOST as host,
    SERVER_PORT as port,
    SERVER_USER as user,
    SERVER_PWD as pwd,
)
cmd = sys.argv[1]
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(host, port=port, username=user, password=pwd, timeout=30)
stdin, stdout, stderr = client.exec_command(cmd, timeout=120)
out = stdout.read().decode('utf-8', 'replace')
err = stderr.read().decode('utf-8', 'replace')
print(out)
if err:
    print("STDERR:", err)
client.close()
