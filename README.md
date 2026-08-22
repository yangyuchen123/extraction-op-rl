# Extraction Operator RL

基于强化学习训练「提取算子」（Extraction Operator）的项目：让 LLM 在图结构/随机地图任务上学习信息提取与动作决策。

## 目录结构

| 文件 | 说明 |
| --- | --- |
| `generator.py` | 随机地图 / 任务生成器 |
| `expert.py` | 专家策略（BFS 参考解） |
| `evaluate_extraction_ops_three_layer.py` | 三层结构评测脚本 |
| `run_extraction_ops_random_b1.sh` | B1 阶段训练入口（SFT + RL） |
| `run_extraction_ops_random_qwen3_06b_lora.sh` | Qwen3 0.6B LoRA 训练 |
| `run_extraction_ops_random_qwen3_17b_lora.sh` | Qwen3 1.7B LoRA 训练 |
| `run_extraction_ops_random_recovery_sft.sh` | 恢复 SFT |
| `run_three_layer_sweep.py` | 三层超参扫描 |
| `wait_sft.py` / `wait_rl.py` / `b15_wait.py` | 远程训练轮询等待 |
| `sync_to_local.py` | 远程产物同步回本地 |
| `upload_scripts.py` / `upload_stage.py` | 脚本上传到训练机 |
| `ssh_exec.py` | 远程命令执行 |
| `parse_training_log.py` / `diag_trace.py` / `tf_replay_diagnostic.py` | 日志解析与诊断 |
| `apply_*.py` | 历史修复补丁（dedup/reward/wiring 等） |
| `*.md` | 实验日志、bug 修复报告、研究路线图 |

## 安全说明

远程 GPU 服务器凭据**不硬编码在仓库中**，通过环境变量注入：

```bash
export GPU_SERVER_HOST=connect.nmb2.seetacloud.com
export GPU_SERVER_PORT=14970
export GPU_SERVER_USER=root
export GPU_SERVER_PWD=你的密码
```

## 快速开始

```bash
# 1. 生成任务 + 专家参考
python3 generator.py

# 2. 上传脚本到训练机
python3 upload_stage.py

# 3. 启动训练（SFT → RL）
bash run_extraction_ops_random_b1.sh

# 4. 等待训练完成后同步结果
python3 sync_to_local.py
```
