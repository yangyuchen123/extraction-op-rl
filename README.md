# Extraction Ops RL

基于强化学习训练「撤离行动」（Extraction Ops）策略的项目：让 LLM 在随机地图上学习完整撤离流程——出生点取钥匙 → 开锁进入目标房间 → 取得密件 → 抵达撤离点撤离。

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

## 实验成果（results/）

B1 阶段完整训练/评测数据见 `results/b1/`，与三篇实验文档（`extraction_ops_full_experiment_journal.md`、`extraction_ops_random_maps_bugfix_report.md`、`extraction_ops_research_roadmap.md`）配套参考。

| 文件 | 内容 | 关键发现 |
| --- | --- | --- |
| `sweep_summary.json` / `b1_rich_sweep_summary.json` | 分步 sweep 评测（64 局/步） | 随机地图 success_rate 仅 4~9%；`loop_episode_ratio` 全为 1.0 → 模型普遍死循环 |
| `training_metrics.json` | PPO 训练指标（每步） | 训练中 success_rate 波动大，val/success_rate 与训练不一致 |
| `dagger_round1_manifest.json` | DAgger 第 1 轮恢复数据 | policy↔expert 分歧率 48%（2649/5506）→ 学到的只是高频捷径 |
| `eval_step1~5.json` | 每步全量评测原始记录（per-episode） | 提前撤离（未取密件）是主要失败模式 |
| `06b_fixed_three_layer.json` / `recovery_sft_three_layer.json` | 固定世界 / 恢复 SFT 三层评测 | 恢复 SFT 3-seed 100%；随机图仍差 |
| `gigpo*` | GIGPO 训练日志与分步评测 | 20 步 GIGPO 退化到 0% → RL 方向需重设计 |
| `training_pilot3.log` / `clean_ckpts.log` | 原始训练日志 | — |

**核心结论**：BFS 专家保证可行性；SFT 在固定世界 100%、随机图仅 18~25%；模型学到的是“冲撤离点”全局捷径而非逐图路径规划；直接 RL（GIGPO）在随机图上退化，需要先解决探索/死循环与 reward 对齐问题（详见路线图）。

## 安全说明

远程 GPU 服务器凭据**不硬编码在仓库中**，存放在本地 `.env`（已被 `.gitignore` 忽略）：

```bash
# 1. 复制模板并填入真实密码
cp .env.example .env
# 2. 编辑 .env（本机即可，无需每次 export）
vim .env
```

所有远程脚本通过 `server_env.py` 统一加载凭据，优先级：**环境变量 `GPU_SERVER_*` > `.env` 文件**（远程机器无 `.env` 时用环境变量）。上传脚本（`upload_stage.py` / `upload_scripts.py`）已包含 `server_env.py`。

> ⚠️ 防泄露检查：提交前确认 `.env` 未被跟踪 —— `git check-ignore .env` 应输出规则；`.env.example` 只有占位符，可安全入库。

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
