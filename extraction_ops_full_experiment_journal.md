# Extraction Ops × verl-agent 完整实验日志（接管后全程记录）

> 记录人：assistant（2026-08-21 起接管）
> 范围：从接管远程实验环境开始的全部过程、决策、结果与数据产物
> 前史（接管前）：见 `docs/archive/` 中的归档状态文档与实验日志

---

## 0. 接管时的初始状态（2026-08-21）

通过 SSH 连接远程主机（AutoDL RTX 3090 24GB），发现 `/root/verl-agent` 是 verl-agent 仓库 + Extraction Ops（逃离塔科夫风格提取任务）实验。

接管时已完成的（前史）：
- 固定世界：SFT 100%、GiGPO 空转（advantage=0）
- 8 模板 procedural：Expert 100% 可行、SFT held-out 18~25%、20 步 GiGPO 退化到 0%、recovery SFT/DAgger 3-seed 100%
- 文档停在"recovery SFT 超时未完成"的旧状态（实际后来已补完 3-seed）

当时遗留问题：
- 文档过时未归档
- 所有实验代码未纳入 git 版本控制
- 尚无随机地图（只有 8 个固定模板）

---

## 1. 归档 + Git 版本控制（2026-08-21）

### 归档（只移动，不删除）
- `docs/extraction_ops_current_experiment_status.md` → `docs/archive/...archived_2026-08-21.md`
- `experiment_artifacts/remote_procedural_v1/EXPERIMENT_JOURNAL.md` → `.../archive/EXPERIMENT_JOURNAL_archived_2026-08-21.md`

### Git 版本控制
- 配置仓库级身份：`extraction-ops-dev`
- 创建分支 `extraction-ops`（master 保持对齐上游）
- 6 个逻辑分块提交：核心环境 / 测试 / 数据预处理脚本 / 训练评测脚本 / 归档文档 / 本地微调
- 大文件（checkpoints 3GB、outputs、wandb、logs）由 .gitignore 覆盖，未误提交

---

## 2. 阶段 A：有界随机图生成器（2026-08-21）

### 实现
- `generator.py`：确定性有界随机图生成器
  - 核心连通图（spawn/key/extraction 都在核心，无门可通）
  - 1 个锁定目标房间（叶子节点，唯一锁门）
  - 0~2 个无锁死胡同房间
  - `layout_id`（拓扑+角色哈希）、`structural_invariants`（可行性不变量）、`validate_definition`（Expert 权威验证）
- `world.py`：新增 `random_maps` 模式、`definition` 注入、`spawn`/`layout_id` 进 snapshot
- `expert.py`：动态解析 KNOWN MAP 图 + 通用门处理（去硬编码 `dorm_front_door`）

### 过程中修的两个 bug
1. `_parse_exits` 用 `split(", ")` 被 `door=xxx` 里的 `, ` 切碎 → 改正则整体匹配
2. 生成器额外边与生成树撞出重复边 → 加 `frozenset` 去重

### 结果
- **冒烟：300/300 种子全部被 BFS 专家通关**，300 个互不相同布局，步数 13~24（上限 60）
- 回归金丝雀：原有 15 测试全过（固定世界 + 8 模板未破坏）
- 提交 `2addf2e`

---

## 3. 研究路线图文档（2026-08-21）

创建 `docs/extraction_ops_research_roadmap.md`，确立三个创新方向 + 一个方法论：

| 方向 | 科学问题 |
|---|---|
| A 多解探索 | RL 能否发现老师没演示过的解（RL > 模仿的唯一证据） |
| B 复杂度/长度泛化 | 模型能否泛化到训练分布外的路径长度/钥匙数 |
| C 无解检测 | 模型能否学会"判断无解并主动放弃" |
| 课程学习（跨方向） | 按难度递进训练能否改善收敛与泛化 |

关键判断：**单解任务下 SFT 抄老师就够、RL 没价值；多解探索才是 RL 价值所在**。

---

## 4. 阶段 B1：结果导向奖励（2026-08-21）

### 稀疏奖励问题的确认
用户提出"DAgger 不是已经解决稀疏了吗？"，澄清了监督信号 vs 奖励信号的区别：
- DAgger 解决**探索冷启动**（给好起点），不解决**RL 运行时信号**（信用分配 + 局部最优）
- 证据：20-step GiGPO 有 warm-start 依然崩到 0%

### 实现
- `world.py` 新增 `REWARD_SCHEMES`（三种方案）：

| 事件 | sparse | milestone（旧） | outcome（B1） |
|---|---:|---:|---:|
| 捡钥匙 | 0 | +0.05 | +0.02 |
| 捡账本 | 0 | +0.15 | +0.03 |
| 成功撤离 | +1.0 | +1.0 | +1.0 |
| 提前撤离 | 0 | +0.05 | **0（不再奖励失败）** |
| 非法动作 | -0.02 | -0.02 | -0.02 |

### 演示脚本结果（seed=42，真实地图）
- 成功轨迹：sparse +1.00 / milestone +1.20 / outcome +1.05
- 循环失败轨迹：sparse 0.00 / milestone **+0.20（陷阱）** / outcome +0.05
- 结论：outcome 把"半途而废"的诱惑从 16.7% 压到 4.8%

---

## 5. B1 接线 + 首次随机地图 SFT（2026-08-21 ~ 22）

### 接线
- `envs.py` / `env_manager.py`：`reward_scheme`、`random_maps`、`random_maps_config` 一路传到底
- `generate_extraction_ops_expert.py` / `evaluate_extraction_ops_policy.py`：加 `--random-maps`

### 随机地图 SFT（0.6B）【注意：此阶段观察有 bug，见 §8】
- 数据：192 局 expert（128 train / 32 val / 32 test），train 2076 步
- 配置：Qwen3-0.6B + LoRA rank16 + lr 1e-4 + batch 32 + 1 epoch
- 训练：loss 0.01~0.02，val/loss 0.028
- **结果（当时）：训练数据 seeds 0-31 仅 9.4%，held-out 6.25%（32 局）**
- 当时误判为"0.6B 容量不够"，实际是观察 bug（见 §8）

---

## 6. B1 RL Pilot（5 步，2026-08-22）

### 配置
- 起点：随机地图 SFT 合并模型（warm-start）
- outcome 奖励 + lr 2e-7 + kl 0.05 + random_maps
- 5 步，save_freq=1 + checkpoint cleaner（只保留 40MB LoRA adapter，删 3GB 完整 checkpoint）

### 结果
- **训练器内置 val（8 局）全程"稳定 25%"**（后来证明是假象，见 §7）
- advantage 非零（-6.7~+4.5）、grad_norm 非零（0.31~0.41）、reward max 1.05（outcome 接线正确）
- 训练成功率 28% → 34%
- **结论（当时）**：outcome 奖励 + 保守 RL 没让它崩（对比上次 20-step GiGPO 崩到 0%）

### 事故
`save_freq=1` 把 3GB/步完整 checkpoint 存根目录 `/`（30G），2 步撑爆磁盘 → 改存数据盘 `/root/autodl-tmp`

---

## 7. 三层验证系统（2026-08-22）

### 用户设计的三层验证
| 层 | 数量 | seed | 用途 |
|---|---|---|---|
| Fixed-Val | 64 | 固定 | 看是否真的进步 |
| Rolling-Val | 64/step | 每步新 seed | 看泛化（防背固定图） |
| Train-like | 64 | 训练分布 | 看训练分布学习 |

### 实现
- `evaluate_extraction_ops_three_layer.py`（三层验证 + 支持加载 LoRA adapter）
- `run_three_layer_sweep.py`（遍历每步 adapter）
- `clean_ckpts.py`（后台保留 LoRA adapter、删完整 checkpoint）
- `parse_training_log.py`（训练日志 → 结构化指标 JSON）

### 关键发现：25% 是假象
三层验证（64 局/层）揭示真实成功率 **5~9%，不是 25%**：

```
 step │ fixed │ rolling │ train_like
   1  │ 7.8%  │  4.7%   │  7.8%
   5  │ 7.8%  │  3.1%   │  7.8%
```

原因：训练器内置 val 只有 8 局（粒度 12.5%）+ 固定种子，25%=2/8 是巧合。

### Trace 增强 + 循环检测器
- trace 增加 inventory / milestones / state_hash / reward_components / world_time_seconds
- 循环检测器（语义状态重复检测，排除 clock）
- 发现：**100% 局有循环**，能定位"卡在哪个房间、拿了什么"

---

## 8. DAgger 第一轮（2026-08-22）

### 语义去重修复
- 初次采集 `duplicate_observations_dropped: 0`（去重失效）
- 原因：去重基于原始文本（含 CLOCK，每步都变），随机地图每局地图又不同
- 修复：改语义去重（seed + location + inventory + milestones）
- 结果：5506 决策 → **607 个唯一语义状态**（89% 循环被折叠）

### DAgger 结果（当时，观察仍有 bug）
- 采集：128 局，policy 与 expert 48% 分歧，成功 9.4%
- 混合：expert 2076 + recovery 607×2 = 3290 步
- Recovery SFT：训练数据仅 **12.5%**（4/32），held-out 5~9%
- 当时误判"0.6B 容量上限"，实际观察 bug 未修

---

## 9. ★ 关键转折：观察 Bug 的发现与修复（2026-08-22）

### 用户质疑
"0.6B 硬背 trace 应该背得下来，应该是训练方法出问题了，SFT 失败有没有可能？"

### 根因（已确认）
`world.py::_observation()` 中 mission intel 和 KNOWN MAP 只在 `procedural or include_brief` 时显示。随机地图 `procedural=False`，因此 **step 0 后 mission + 地图全部消失**：

| 观察 | step 0 | step 1 |
|---|---|---|
| KEY/TARGET INTEL | ✅ | ❌ |
| KNOWN MAP | ✅ | ❌ |

模型从第二步起"失明"，不知道目标在哪、地图长什么样。

### 修复（commit `b017494`）
随机地图每步显示 mission + 地图。22 测试全过。

---

## 10. 修复后 0.6B 重新训练与分析（2026-08-22）

### 重训（同数据同配置，只换修复后 observation）
- 训练数据 seeds 0-31：9.4%（但平均步数 45→**20**，不再死循环）
- 三层验证（64 局/层）：

| 层 | 修复前 | 修复后 |
|---|---:|---:|
| fixed_val | 7.8% | **10.9%** |
| rolling_val | 3~9% | **9.4%** |
| train_like | 7.8% | 4.7% |

### 失败模式剧变（修复起效的铁证）
| 失败模式 | 修复前 | 修复后 |
|---|---:|---:|
| max_steps（瞎循环） | ~52% | ~23% |
| extracted_without_objective（提前撤离） | ~15% | **~48%** |

### Trace 分析 → 精确能力边界
- Seed 0 型（48%）：直接冲撤离点（走捷径，跳过拿钥匙/账本）
- Seed 1/2 型（15%）：**完整做完 get key→get objective，卡在回撤离点的多跳路由**（sector_5↔sector_3 循环）

| 能力 | 0.6B |
|---|---|
| 格式/合法性 | ✅ 100% |
| 交互原语（open/search/pickup/unlock） | ✅ |
| 单跳/两跳移动 | ✅ |
| 多子目标排序（三跳链） | ❌ |
| 去撤离点多跳 BFS 路由 | ❌ |

---

## 11. 系统盘清理（2026-08-22）

- 删除 pip cache 4.1G、flash-attention 源码 3G、alfworld cache 2.5G、conda pkgs 230M
- 系统盘 `/`：67% → 35%（释放约 9G）
- 验证：flash_attn / torch / vllm / world 均正常

---

## 12. 1.7B 对照准备（2026-08-22）

- 下载 Qwen3-1.7B（`/root/autodl-tmp/models/Qwen3-1.7B`）
- 写好 1.7B SFT 脚本（micro_batch=2 防 OOM，有效 batch 仍 32 不变）
- **受控对比设计**：训练方法、数据、环境、LoRA 配置全部不变，只换模型规模

---

## 13. 核心方法论沉淀

1. **低 SFT loss ≠ 学会任务**：loss 0.017 但 online 5~9%，因为训练 prompt 缺失关键信息。
2. **失败模式分布比成功率更诊断**：成功率 7.8%→10.9% 几乎不变，但"瞎循环→提前撤离"才揭示真实进展。
3. **怀疑训练 bug 优先于怀疑容量**：用户坚持质疑"SFT 失败"是对的，先排除数据 bug 再谈 scaling。
4. **固定小验证集不可信**：8 局 val 的 25% 是假象，64 局三层验证才揭示真实 5~9%。
5. **语义去重**：循环会让 DAgger 数据被重复状态灌满，必须按语义状态（非原始文本）去重。
6. **checkpoint 只存 adapter**：完整 FSDP checkpoint 3GB/步会撑爆盘，LoRA adapter 40MB 足够诊断。

---

## 14. 完整产物清单

### 模型（数据盘 `/root/autodl-tmp/models/`）
| 模型 | 说明 |
|---|---|
| Qwen3-0.6B | 基座 |
| Qwen3-1.7B | 基座（已下载，未训） |
| Qwen3-0.6B-extraction-ops-random-sft | 随机地图 SFT（**观察 bug 版**，已弃用） |
| Qwen3-0.6B-extraction-ops-random-recovery-sft | DAgger 第一轮（观察 bug 版，已弃用） |
| Qwen3-0.6B-extraction-ops-random-sft-fixed | **修复后 0.6B（当前基准）** |

### 数据
- expert 数据：`/root/data/verl-agent/extraction_ops_random_expert`（192 局）
- SFT 数据：`/root/data/verl-agent/extraction_ops_random_sft`
- recovery 数据：`/root/data/verl-agent/extraction_ops_random_recovery`

### 评测（`/root/autodl-tmp/eval_archive/` + 本地 `/home/administrator/extraction_ops_archive/`）
- `06b_fixed_three_layer.json`：修复后 0.6B 三层验证
- `b1_rich/sweep_summary.json`：B1 pilot 5 步三层学习曲线（观察 bug 版）
- `recovery_sft_three_layer.json`：DAgger 第一轮三层验证
- 各步完整 trace（可回放）

### Git 提交（分支 extraction-ops）
```
9b585d8 docs: 记录观察 bug 修复与 0.6B 能力边界
b017494 fix: 随机地图每步显示 mission+地图
f2c3f92 feat: 随机地图 DAgger recovery + 语义去重
a5b5987 feat: 增强 eval trace + 循环检测器
6ae2331 feat: 三层验证 + checkpoint cleaner + 日志解析
dbcc81e feat: B1 pipeline launcher
20d6673 fix: checkpoint 存数据盘
83cfd35 feat: 奖励方案 sparse/milestone/outcome
b9a2873 docs: 研究路线图
2addf2e feat: 有界随机图生成器
```

---

## 15. 待办（下一步）★ 已按 2026-08-22 复盘修改

**规划变更：1.7B 缓行，先做 B1.5 SFT 管线验证**

触发原因：修复观察 bug 后 0.6B 重训，train loss 0.017 但 train_like 三层验证仅 9.4%（连训练 seeds 都学不会）。0.6B 硬背 2076 步理应背得下来——**这是训练侧问题，不是容量上限；1.7B 用同配置同样学不会**。

- [ ] **B1.5-1：0.6B 多 epoch 重训**（同数据同配置，仅 EPOCHS=5~10），跑三层验证——train_like 应大幅上升（记忆生效）
- [ ] **B1.5-2：teacher-forcing 回放诊断**——expert 每步 obs 喂合并模型测 next-action 准确率，区分 exposure bias vs prompt 不一致 bug
- [ ] **B1.5-3：评估脚本显式传 reward_scheme="outcome"**（当前默认 milestone 给提前撤离 +0.05，48% 提前撤离局全在赚正奖励，指标被污染）
- [ ] 判据：train_like ≥ 50% → 确认记忆不足，再测 held-out 泛化；泛化才允许训 1.7B；仍 < 30% → 深挖数据生成，1.7B 无限期顺延
- [ ] 若 1.7B 最终仍卡撤离路由 → 课程学习 / 更多 DAgger / 检查路由表示
- [ ] 更新 roadmap 中 B1/B1.5 结论
- [ ] 若 1.7B 学会规划 → 回到方向 A（多解探索）和方向 B（复杂度）

---

## 16. 当前实验叙事（一句话总结）

**从"跑通玩具游戏"到"定位模型能力边界"**：发现并修复了观察层 bug（mission/地图在 step1 后消失），把"0.6B 学不会"的错误结论修正为精确的能力画像——**0.6B 学得会动作原语、学不会多步规划（三跳子目标链 + 多跳路由）**。但 train_like 仅 9.4% + 48% 提前撤离局全拿正奖励（milestone 评估 trap）这两个新证据表明：**scaling 假说尚未到验证时机——0.6B 连训练数据都没背下来（1 epoch 记忆不足 / 频率偏置 / exposure bias），必须先修 SFT 管线（B1.5），再谈 1.7B**。
