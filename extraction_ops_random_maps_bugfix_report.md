# Extraction Ops 随机地图：观察 Bug 修复与 0.6B 能力边界报告

> 记录日期：2026-08-22
> 关联：`docs/archive/` 中的历史状态文档；本文档记录随机地图阶段的关键 bug、修复与修复后 0.6B 的能力分析。
> 核心结论一句话：**此前 0.6B "SFT 失败" 的根因是观察层 bug（mission/地图在 step1 后消失），不是模型容量；修复后模型仍卡在"多步规划"，这为 1.7B 对照提供了精确假设。**

---

## 1. 关键 Bug 与修复

### 现象

随机地图 SFT 训练 loss 极低（0.017），但 online 成功率只有 5~9%，连训练数据（seeds 0-127）也只 9.4%，且 100% 局循环。用户提出质疑："0.6B 硬背 trace 应该背得下来，是不是 SFT 失败了？"

### 根因（已确认）

`world.py::_observation()` 中：

```python
lines = self._mission_lines() if self.procedural or include_brief else []
if include_brief:
    lines.append("KNOWN MAP: ...")
```

随机地图模式 `procedural=False`，因此 **mission intel（KEY INTEL / TARGET INTEL / VALID EXTRACTION）和 KNOWN MAP 只在 reset 时（step 0）出现，step 1 起全部消失**。

验证（seed=0）：

| 观察 | step 0 | step 1 |
|---|---|---|
| KEY INTEL | ✅ | ❌ |
| TARGET INTEL | ✅ | ❌ |
| KNOWN MAP | ✅ | ❌ |

后果：模型从第二步起就是"瞎子"，不知道目标在哪、地图长什么样，只能瞎循环。

对比解释：
- **8 模板 procedural 能成功**：`procedural=True` 每步都显示 mission；
- **固定世界能成功**：mission 恒定，模型可凭局部观察死记轨迹；
- **随机地图失败**：mission/地图每步缺失 + 布局变化 → 无解。

### 修复（commit `b017494`）

```python
lines = self._mission_lines() if self.procedural or self.random_maps or include_brief else []
if include_brief or self.random_maps:
    lines.append("KNOWN MAP: ...")
```

随机地图现在每步显示 mission + 地图。22 个回归测试全过。

---

## 2. 修复后 0.6B 重新训练与三层验证

### 受控设置

- 数据：重新生成 192 局 expert（128 train / 32 val / 32 test），train 2076 步
- 配置：Qwen3-0.6B + LoRA rank 16 + lr 1e-4 + batch 32 + 1 epoch（与之前完全一致）
- 模型：`/root/autodl-tmp/models/Qwen3-0.6B-extraction-ops-random-sft-fixed`

### 三层验证结果（64 局/层）

| 层 | 修复前 | 修复后 | 平均步数（修复后） |
|---|---:|---:|---:|
| fixed_val | 7.8% | **10.9%** | 26.9 |
| rolling_val | 3~9% | **9.4%** | 28.8 |
| train_like | 7.8% | 4.7% | 33.3 |

训练数据（seeds 0-31）：修复前 9.4%（平均 45 步）→ 修复后 9.4%（平均 **20 步**）。

### 失败模式变化（这是修复起效的直接证据）

| 失败模式 | 修复前 | 修复后 |
|---|---:|---:|
| max_steps（瞎循环） | ~52% | ~23% |
| extracted_without_objective（提前撤离） | ~15% | **~48%** |
| 平均步数 | 45 | 27 |

模型从"盲循环"变成"主动决策"（提前撤离），证明它现在**看得见**了，但规划仍弱。

---

## 3. Trace 分析：0.6B 的精确能力边界

对 seeds 0/1/2 的逐步 trace 分析：

**Seed 0（48% 典型）——走捷径：**
```
move sector_7 (撤离点) → extract  → 提前撤离失败
```

**Seed 1/2（15% 典型）——前半段全对，卡在最后：**
```
move key → open → search → pickup key → unlock → open → move target
→ open → search → pickup objective   ✅ 全对
→ route 回撤离点：sector_5 ↔ sector_3 无限循环   ❌
```

### 能力画像

| 能力 | 状态 |
|---|---|
| 动作格式 / 合法性 | ✅ 100% |
| 交互原语（open/search/pickup/unlock） | ✅ 学会 |
| 单跳/两跳移动（去钥匙、去账本） | ✅ 学会 |
| 多子目标排序（钥匙→账本→撤离 三跳链） | ❌ 不会 |
| 去撤离点的多跳 BFS 路由 | ❌ 循环 |

### 结论

1. 之前的"SFT 失败"是**观察 bug**，不是容量问题（用户判断正确）；
2. 修复后暴露**第二个真问题**：0.6B 能学"动作原语"，但学不会"多步规划"（子目标排序 + 多跳路由）；
3. 62%（seed 0 型捷径）说明模型对任务顺序理解不稳，倾向于"看到撤离点就冲"。

---

## 4. 对 1.7B 对照的精确假设

**可证伪假设**：

> 0.6B 学得会动作原语、学不会多步规划。若 1.7B 在**同一份修复后数据 + 同一配置**下，能把"钥匙→账本→撤离"这条三跳子目标链补上（尤其是最后的撤离路由），则证明**规划能力随模型规模 scaling**；若 1.7B 同样卡在撤离路由，则说明该任务的规划瓶颈不在规模而在别处（数据/课程/表示）。

**受控变量**：训练方法、数据、环境、LoRA 配置全部不变，只换 `model.path`（0.6B → 1.7B）。

---

## 5. 产物清单

| 产物 | 路径 |
|---|---|
| 修复后 0.6B 模型 | `/root/autodl-tmp/models/Qwen3-0.6B-extraction-ops-random-sft-fixed` |
| 重新生成的 expert 数据 | `/root/data/verl-agent/extraction_ops_random_expert`（192 局） |
| SFT 数据 | `/root/data/verl-agent/extraction_ops_random_sft` |
| 三层验证 | `/root/autodl-tmp/eval_archive/06b_fixed_three_layer.json` |
| 1.7B 基座模型 | `/root/autodl-tmp/models/Qwen3-1.7B`（已下载） |
| 本地归档 | `/home/administrator/extraction_ops_archive/` |

---

## 6. 方法论沉淀（重要教训）

1. **低 SFT loss ≠ 学会任务**：loss 0.017 但 online 5~9%，因为训练 prompt 本身缺失关键信息（观察 bug）。
2. **失败模式分布比成功率更能诊断**：成功率几乎不变（7.8%→10.9%），但"瞎循环→提前撤离"的转变才揭示了真实进展。
3. **三层验证 + 循环检测器**的价值：fixed/rolling/train-like 区分了"进步/泛化/训练分布学习"，循环检测器直接定位"卡在哪个房间、拿了什么"。
4. **怀疑"训练方法"优先于"模型容量"**：本次用户坚持质疑 SFT 失败是对的，先排除了训练 bug 再谈 scaling。

---

## 7. 待办（2026-08-22 复盘后修改：1.7B 缓行，先做 B1.5 SFT 管线验证）

**修订原因**：修复后 0.6B train loss 0.017 但 train_like 仅 9.4%（连训练 seeds 都学不会）+ 48% 提前撤离局全部拿到 +0.05/+0.10（评估默认 milestone 的 reward trap）。0.6B 硬背 2076 步理应背得下来——**这是训练侧问题（1 epoch 记忆不足 / 频率偏置 / exposure bias），不是容量上限，1.7B 同配置同样学不会**。

- [ ] 0.6B 多 epoch 重训（5~10 epoch，同数据同配置）验证 train_like 是否上升（记忆生效）
- [ ] teacher-forcing 回放诊断：expert 每步 obs 喂合并模型测 next-action 准确率，区分 exposure bias vs prompt 不一致
- [ ] 评估脚本显式传 `reward_scheme="outcome"`，堵住评估侧 reward trap
- [ ] 判据：train_like ≥ 50% 且 held-out 有泛化 → 才训 1.7B 对照；否则深挖 SFT 管线
- [ ] 若 1.7B 最终仍卡撤离路由 → 尝试课程学习 / 更多 DAgger / 检查路由表示
- [ ] 更新 `docs/extraction_ops_research_roadmap.md` 中 B1/B1.5 的结论
