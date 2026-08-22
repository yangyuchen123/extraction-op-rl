import paramiko, sys, re, time

from server_env import (
    SERVER_HOST as host,
    SERVER_PORT as port,
    SERVER_USER as user,
    SERVER_PWD as pwd,
)
timeout = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
log = "/root/autodl-tmp/extraction_ops_b1_pilot.log"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(host, port=port, username=user, password=pwd, timeout=30)

# Wait until training finishes (global_step 5) or processes exit.
cmd = (
    f"for i in $(seq 1 200); do "
    f"  grep -aq 'training/global_step:5.000' {log} && break; "
    f"  pgrep -f 'main_ppo' >/dev/null || break; "
    f"  sleep 15; "
    f"done; "
    f"echo '=== 训练指标轨迹 ==='; "
    f"grep -aE 'step:[0-9] - ' {log} | grep -aoE 'step:[0-9] - .*success_rate:[0-9.]+' | head -20; "
    f"echo; echo '=== advantage/grad/score 摘要 ==='; "
    f"grep -aoE 'critic/advantages/(mean|max|min):-?[0-9.]+|actor/grad_norm:[0-9.]+|episode/success_rate:[0-9.]+|episode/reward/mean:[0-9.]+' {log} | tail -40; "
    f"echo; echo '=== 是否仍在运行 ==='; pgrep -f main_ppo >/dev/null && echo RUNNING || echo FINISHED; "
    f"echo; echo '=== checkpoint ==='; ls /root/verl-agent/checkpoints/verl_agent_extraction_ops_b1/gigpo_random_outcome_pilot/ 2>/dev/null"
)
stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
out = stdout.read().decode("utf-8", "replace")
print(out)
client.close()
