# TextShield-R2 论文项目计划（完善版）

> 基于 TextShield-R1 (AAAI 2026) 的改进工作
> 目标会议/期刊：ECCV 2026 / ACM MM 2026 / TPAMI（根据完成度选择）
> 今日：2026-03-15
> 参考：VMUnet论文项目计划格式

---

## 一、已完成的工作（继承自 TextShield-R1）

### 1. 代码实现

| 模块 | 文件 | 状态 |
|------|------|------|
| GRPO 奖励函数 (5个) | orm.py | 完成 |
| 推理管线 (准备/推理/后处理) | pipeline.py | 完成 |
| SFT数据生成 | prepare_sft_data.py | 完成 |
| 图像预处理 (28x resize) | resize_image_dir.py | 完成 |
| 评估脚本 (分类/OCR/IoU/推理) | eval_*.py | 完成 |
| OCR修正评估 | eval_iou_with_ocr_rectification.py | 完成 |
| BBox可视化 | render_bbox_mask.py | 完成 |

### 2. 已有实验结果（TextShield-R1 基线）

| 设置 | Test Cls. | Test OCR | Test Loc. | Test Res. | CIS Cls. | CTM Cls. | CL Cls. |
|------|-----------|----------|-----------|-----------|----------|----------|---------|
| Baseline (Qwen2.5-VL-7B) | 79.1 | 24.3 | 18.2 | 42.9 | 71.1 | 73.6 | 85.1 |
| **TextShield-R1** | **88.1** | **47.6** | **57.8** | **58.8** | **72.9** | **88.8** | **85.5** |

### 3. 已有资源

| 资源 | 描述 |
|------|------|
| TFR Benchmark | 45k+图像，16种语言，10种篡改方法 |
| 预训练权重 | Forensic Continual Pre-training 模型 |
| ms-swift框架 | GRPO训练框架已调通 |
| 训练数据 | SFT 12,983样本 + GRPO 51,690样本 |

---

## 二、现有局限性分析

| 编号 | 局限性 | 具体表现 | 严重程度 |
|------|--------|---------|---------|
| L1 | **单轮推理，无自校验** | 模型一次性输出结果，无法检查和修正自身错误 | 高 |
| L2 | **GRPO的长度偏差与熵崩塌** | 长回答获得更多梯度，训练后期探索不足 | 高 |
| L3 | **奖励函数粗粒度** | 仅有结果级(outcome-level)奖励，推理过程无监督 | 高 |
| L4 | **单尺度视觉感知** | 固定分辨率，对微小篡改痕迹感知不足 | 中 |
| L5 | **预测逻辑不一致** | 判断为real却输出篡改位置等自相矛盾 | 中 |
| L6 | **跨域泛化不足** | CIS/CTM/CL集上性能下降 | 中 |

---

## 三、创新点设计（核心亮点）

### 核心叙事

TextShield-R1 首次将强化学习引入篡改文本检测，但其采用的 **GRPO 算法存在长度偏差和熵崩塌问题**（DAPO, 2025），且 **仅有结果级奖励无法保证推理链质量**（PRIME, 2025）。同时，人类取证专家的工作方式是 **"观察-假设-验证-修正"的多轮推理**，而非一次性判断。我们从最新强化学习研究中引入三项关键技术升级，并设计多轮取证推理框架，全面提升检测精度和推理质量。

---

### 创新点 1：DAPO — 解耦自适应策略优化（替换GRPO）⭐ 核心贡献

**动机**：GRPO存在三个已被证实的问题（ByteDance Seed & Tsinghua AIR, arXiv:2503.14476）：
1. **熵崩塌**：训练后期策略坍缩到固定模式，丧失探索能力
2. **长度偏差**：长回答的梯度累积大于短回答，不论质量高低
3. **无效样本浪费**：全对/全错的prompt组无法产生学习信号

**方法**：用 DAPO (Decoupled Clip and Dynamic sAmpling Policy Optimization) 替换 GRPO，引入四项技术改进：

| 技术 | 作用 | 对取证任务的意义 |
|------|------|----------------|
| **Clip-Higher** (非对称裁剪) | 放宽策略更新上界（ε_high > ε_low），鼓励探索 | 避免模型总是输出相似的取证分析 |
| **Dynamic Sampling** (动态采样) | 过滤全对/全错的prompt，集中训练边界样本 | 聚焦于困难篡改案例的学习 |
| **Token-Level Loss** (令牌级损失) | 按batch内总token数归一化，而非先序列后batch | 解决分类答案短(几个token)/定位解释长(数百token)的不平衡 |
| **Overlong Reward Shaping** (超长惩罚) | 对过长输出施加软惩罚而非硬截断 | 防止模型产生冗长无关的取证"推理" |

**理论支撑**：DAPO在AIME 2024上达到50%准确率，比DeepSeek-R1-Zero训练步数减少50%。在我们的任务中，Token-Level Loss 尤为关键——取证任务输出包含短分类标签+长推理链+中等长度坐标，三者token数差异极大。

**技术实现**：

```python
# DAPO 核心配置（基于ms-swift扩展）
dapo_config = {
    "clip_eps_high": 0.28,      # 非对称裁剪上界（鼓励探索）
    "clip_eps_low": 0.2,        # 对称裁剪下界（稳定学习）
    "dynamic_sampling": True,    # 过滤全对/全错prompt
    "token_level_loss": True,    # token级归一化
    "overlong_penalty": 0.1,     # 超长输出的软惩罚系数
    "max_response_length": 2048, # 超过此长度触发惩罚
}
```

---

### 创新点 2：PRIME — 隐式过程奖励模型（无需步骤标注）⭐ 核心贡献

**动机**：TextShield-R1的5个奖励函数均为outcome-level——只看最终答案是否正确，不关心推理过程。坏的推理可能碰巧得到好结果（如猜对分类但推理全错），好的推理可能因小错误被完全惩罚。

**为什么选择 PRIME 而非传统 PRM**：
- 传统PRM（如Math-Shepherd）需要 **步骤级人工标注**，成本极高
- GPT-4o标注（原计划方案）**费用高昂且质量不稳定**
- **PRIME（arXiv:2502.01456）**：仅需outcome标签即可训练隐式PRM，自动推断步骤级奖励

**方法**：Process Reinforcement through Implicit Rewards

| 步骤 | 描述 |
|------|------|
| 1. 训练隐式PRM | 用outcome标签（分类对错、IoU阈值）训练一个小型VLM作为隐式奖励模型 |
| 2. 推断步骤奖励 | 隐式PRM在推理链的每个步骤处计算"进步分数"，而非仅看最终结果 |
| 3. 在线更新 | PRM随策略模型在线更新，缓解分布偏移和奖励黑客问题 |
| 4. 集成到DAPO | 将PRIME的步骤奖励作为额外奖励信号加入DAPO训练 |

**技术优势**：
- **无需步骤标注**：仅需现有的分类/IoU/OCR ground truth
- **2.5x样本效率**：相比纯outcome-level RL
- **自动密集奖励**：将稀疏的最终奖励分解为每步的过程奖励
- **在线更新**：PRM与策略同步优化，避免reward hacking

**技术实现**：

```python
# PRIME 隐式过程奖励模型
class ImplicitPRM:
    def __init__(self, base_model="Qwen2.5-VL-3B"):
        """基于小型VLM的隐式PRM"""
        self.model = load_model(base_model)

    def compute_step_rewards(self, image, reasoning_chain):
        """
        对推理链的每一步计算隐式奖励
        输入：图像 + 推理链文本
        输出：每步的进步分数 (list of floats)
        """
        steps = split_reasoning_steps(reasoning_chain)
        step_rewards = []
        for i, step in enumerate(steps):
            context = steps[:i+1]  # 截至当前步骤的上下文
            # 隐式PRM通过预测"继续推理能否到达正确结果"来估计步骤奖励
            progress_score = self.model.estimate_progress(image, context)
            step_rewards.append(progress_score)
        return step_rewards

# 与DAPO集成
def combined_reward(prediction, ground_truth, image, prm):
    outcome_reward = compute_outcome_rewards(prediction, ground_truth)  # R1的5个奖励
    process_reward = prm.compute_step_rewards(image, prediction)       # PRIME步骤奖励
    return outcome_reward + alpha * mean(process_reward)               # 加权组合
```

---

### 创新点 3：Multi-Turn Grounding Policy Optimization — 多轮视觉定位优化 ⭐ 核心贡献

**动机**：TextShield-R1是单轮推理——模型看一次图像就输出所有结果。但人类取证专家的工作流程是：
1. 先全局浏览，识别可疑区域
2. 放大可疑区域仔细检查
3. 基于局部证据修正判断

**灵感来源**：MGPO (Multi-turn Grounding-based Policy Optimization, arXiv:2507.05920) 证明了多轮视觉定位可通过纯RL学习——仅用二值答案正确性作为奖励，VLM就能自主学会"在哪里看"和"看多细"。

**方法**：设计两轮取证推理框架

```
Round 1 (全局分析):
  输入: 原始图像 + 标准prompt
  输出: <think> 全局推理 </think>
        <answer> 初步判断 + 可疑区域坐标 [x1,y1,x2,y2] </answer>

Round 2 (局部验证):
  输入: 原始图像 + 裁剪放大的可疑区域 + Round 1的初步结论
  输出: <verify> 局部验证分析 </verify>
        <final_answer> 最终判断(修正或确认) </final_answer>
```

**关键创新点**：
- **RL学习定位策略**：模型通过RL自主学习"去看哪里"，而非人工规则裁剪
- **自动Zoom-in**：Round 1预测的bbox区域被自动裁剪放大送入Round 2
- **多图推理**：利用Qwen2.5-VL的多图输入能力，无需架构修改
- **自我修正**：Round 2可以修正Round 1的判断（受DAPO+PRIME联合训练）

**与VLM-R1的区别**：VLM-R1（arXiv:2504.07615）是单轮定位，我们是多轮"定位-放大-验证"闭环。

**奖励设计**：

```python
def multi_turn_reward(round1_output, round2_output, ground_truth):
    """多轮取证推理的复合奖励"""

    # Round 1 奖励：鼓励正确的初步定位
    r1_cls = classification_reward(round1_output, ground_truth)
    r1_loc = iou_reward(round1_output, ground_truth)

    # Round 2 奖励：鼓励有效验证和修正
    r2_cls = classification_reward(round2_output, ground_truth)
    r2_loc = iou_reward(round2_output, ground_truth)
    r2_ocr = ocr_reward(round2_output, ground_truth)

    # 修正奖励：如果Round 2修正了Round 1的错误，额外奖励
    r1_correct = is_correct(round1_output, ground_truth)
    r2_correct = is_correct(round2_output, ground_truth)

    correction_bonus = 0.0
    if not r1_correct and r2_correct:
        correction_bonus = 0.5  # 成功修正
    elif r1_correct and not r2_correct:
        correction_bonus = -0.3  # 过度修正惩罚

    return (r1_cls + r1_loc) * 0.3 + (r2_cls + r2_loc + r2_ocr) * 0.7 + correction_bonus
```

---

### 创新点 4：Consistency-Aware Step Reward — 一致性感知步进奖励 (辅助贡献)

**动机**：受 StepGRPO (R1-VL, arXiv:2503.12937) 启发，我们设计两类步进奖励：

| 奖励类型 | 灵感来源 | 取证任务中的实现 |
|---------|---------|----------------|
| **Step Reasoning Accuracy Reward (StepRAR)** | R1-VL StepRAR | 检查推理链是否包含关键取证步骤（边缘分析、纹理检查、语义分析） |
| **Step Reasoning Validity Reward (StepRVR)** | R1-VL StepRVR | 检查推理链的逻辑一致性（分类/方法/定位/OCR四者不自相矛盾） |

**StepRAR 关键步骤检查**：

```python
KEY_FORENSIC_STEPS = {
    "edge_analysis": ["edge", "boundary", "border", "contour", "artifact"],
    "texture_check": ["texture", "noise", "pattern", "consistency", "smooth"],
    "semantic_check": ["content", "meaning", "context", "font", "style"],
    "method_identify": ["copy", "paste", "generat", "splice", "inpaint"],
}

def step_rar_reward(reasoning_text):
    """奖励推理链包含必要的取证分析步骤"""
    steps_found = 0
    for step_name, keywords in KEY_FORENSIC_STEPS.items():
        if any(kw in reasoning_text.lower() for kw in keywords):
            steps_found += 1
    return steps_found / len(KEY_FORENSIC_STEPS)  # 0.0 ~ 1.0
```

**StepRVR 逻辑一致性检查**：

```python
def step_rvr_reward(prediction):
    """检查预测结果的内部逻辑一致性"""
    score = 1.0
    cls = extract_classification(prediction)
    has_bbox = has_bounding_box(prediction)
    has_method = has_tampering_method(prediction)

    # 判断为real但输出了篡改信息 → 矛盾
    if cls == "real" and (has_bbox or has_method):
        score -= 0.5
    # 判断为tampered但缺少定位 → 不完整
    if cls == "tampered" and not has_bbox:
        score -= 0.3
    # think和answer中的分类不一致 → 矛盾
    think_cls = extract_think_classification(prediction)
    if think_cls and think_cls != cls:
        score -= 0.3

    return max(0, score)
```

---

### 创新点总结与定位

| 排序 | 创新点 | 角色 | 技术来源 | 解决的问题 |
|------|--------|------|---------|-----------|
| 1 | **DAPO替换GRPO** | 核心贡献 | ByteDance 2025 | 长度偏差、熵崩塌、无效样本 (L2) |
| 2 | **PRIME隐式过程奖励** | 核心贡献 | 2025 | 推理过程无监督 (L3) |
| 3 | **多轮视觉定位推理** | 核心贡献 | MGPO/VLM-R1 2025 | 单轮推理、单尺度感知 (L1, L4) |
| 4 | **一致性步进奖励** | 辅助贡献 | StepGRPO 2025 | 预测逻辑不一致 (L5) |

**与TextShield-R1的本质区别**：R1 = "GRPO + outcome rewards + 单轮推理"；R2 = "DAPO + process rewards + 多轮定位推理"，是**强化学习方法论的全面升级**。

---

## 四、完整奖励函数体系（R1 → R2 对比）

### TextShield-R1 奖励函数（5个，全部outcome-level）

| 奖励 | 权重 | 类型 | 文件 |
|------|------|------|------|
| Real/Fake Classification | 1.0 | Outcome | orm.py |
| Forgery Method Detection | 0.5 | Outcome | orm.py |
| Tampering Localization (IoU) | 1.0 | Outcome | orm.py |
| Tampered Text OCR | 1.0 | Outcome | orm.py |
| Format | 0.1 | Outcome | orm.py |

### TextShield-R2 奖励函数（9个 = 5原有 + 4新增）

| 奖励 | 权重 | 类型 | 新增 | 来源 |
|------|------|------|------|------|
| Real/Fake Classification | 1.0 | Outcome | 否 | R1 |
| Forgery Method Detection | 0.5 | Outcome | 否 | R1 |
| Tampering Localization (IoU) | 1.0 | Outcome | 否 | R1 |
| Tampered Text OCR | 1.0 | Outcome | 否 | R1 |
| Format (v2, 支持多轮格式) | 0.1 | Outcome | 更新 | R1+R2 |
| **PRIME Process Reward** | 0.3 | **Process** | **是** | PRIME |
| **Correction Bonus** | 0.5 | **Multi-turn** | **是** | MGPO |
| **StepRAR (关键步骤)** | 0.3 | **Step** | **是** | StepGRPO |
| **StepRVR (逻辑一致性)** | 0.5 | **Step** | **是** | StepGRPO |

### 训练算法对比

| 维度 | R1 (GRPO) | R2 (DAPO) |
|------|-----------|-----------|
| 裁剪策略 | 对称PPO clip | 非对称 Clip-Higher |
| 损失归一化 | 先序列后batch | Token-level全局归一化 |
| 样本利用 | 全部使用 | Dynamic Sampling过滤 |
| 长度控制 | 硬截断 | Overlong Reward Shaping |
| 奖励层级 | Outcome-only | Outcome + Process + Step |
| 推理轮数 | 单轮 | 多轮（全局→局部） |

---

## 五、待完成的工作

### Phase 1：DAPO算法实现与集成（优先级：最高，Week 1-2）

| 任务 | 详细描述 | 预计耗时 |
|------|----------|----------|
| 研究ms-swift DAPO支持 | 检查ms-swift是否已支持DAPO，若不支持则需扩展 | 1天 |
| 实现DAPO四项改进 | Clip-Higher + Dynamic Sampling + Token-Level Loss + Overlong Shaping | 3天 |
| DAPO单元测试 | 在小规模数据上验证DAPO训练稳定性 | 1天 |
| DAPO vs GRPO对比实验 | 用原有5个奖励函数，仅替换算法，验证DAPO本身的提升 | 2天训练 |

**关键决策点**：如果ms-swift不直接支持DAPO，需要评估：
- 方案A：在ms-swift框架内扩展GRPO为DAPO（修改loss计算和采样逻辑）
- 方案B：切换到其他支持DAPO的框架（如verl, OpenRLHF）
- 方案C：仅实现最关键的Token-Level Loss和Dynamic Sampling（最小可行改进）

### Phase 2：PRIME隐式过程奖励模型（优先级：最高，Week 2-3）

| 任务 | 详细描述 | 预计耗时 |
|------|----------|----------|
| 构建PRIME训练数据 | 用TextShield-R1模型生成推理轨迹，标注outcome标签 | 2天 |
| 训练隐式PRM | 基于Qwen2.5-VL-3B微调隐式过程奖励模型 | 1-2天训练 |
| PRM质量验证 | 检验PRM的步骤奖励是否与人类直觉一致 | 1天 |
| 集成到DAPO | 将PRIME奖励加入DAPO的奖励信号 | 1天 |

**PRIME训练数据构建方法**：

```python
# 用R1模型生成多组推理轨迹
for image, ground_truth in train_data:
    trajectories = r1_model.generate(image, num_samples=8, temperature=1.0)
    for traj in trajectories:
        outcome = compute_outcome_reward(traj, ground_truth)
        prm_train_data.append({
            "image": image,
            "trajectory": traj,
            "outcome_label": outcome  # 仅需最终对错标签
        })
```

### Phase 3：多轮定位推理框架（优先级：高，Week 3-4）

| 任务 | 详细描述 | 预计耗时 |
|------|----------|----------|
| 设计多轮prompt模板 | Round 1全局分析 + Round 2局部验证的prompt格式 | 1天 |
| 构建多轮SFT数据 | 将现有单轮数据转换为多轮格式（自动裁剪+GPT-4o生成验证文本） | 3天 |
| 自动Zoom-in模块 | 根据Round 1预测的bbox裁剪放大送入Round 2 | 1天 |
| 多轮Cold Start SFT | 在反思格式数据上做冷启动 | 1-2天训练 |
| 多轮DAPO训练 | 完整多轮推理 + PRIME + 一致性奖励的DAPO训练 | 2-3天训练 |

**多轮Prompt设计**：

```
# Round 1
<image>
Is this image real, entirely generated, or tampered?
If it has been tampered, what method was used, and what are the content
and bounding box coordinates of the tampered text?
Output the thinking process in <think></think> and initial answer in <answer></answer>.

# Round 2 (自动注入Round 1结果 + 裁剪图)
<image_full> <image_crop>
Your initial analysis concluded: {round1_answer}.
The second image is a zoomed-in view of the suspected region [{x1},{y1},{x2},{y2}].
Verify your initial analysis by examining this region closely.
Output verification in <verify></verify> and final answer in <final_answer></final_answer>.
```

### Phase 4：一致性步进奖励实现（优先级：中，Week 4）

| 任务 | 详细描述 | 预计耗时 |
|------|----------|----------|
| 实现StepRAR | 关键取证步骤检查奖励函数 | 0.5天 |
| 实现StepRVR | 逻辑一致性检查奖励函数 | 0.5天 |
| 更新Format Reward v2 | 支持多轮格式的格式奖励 | 0.5天 |
| 奖励权重调优 | Grid search最佳奖励权重组合 | 1天 |

### Phase 5：实验设计与执行（优先级：高，Week 5-8）

#### 5.1 主实验：与SOTA对比

| 方法 | 来源 | 对比意义 |
|------|------|---------|
| GPT-4o | 论文引用 | 通用MLLM上界 |
| Qwen2.5-VL-7B | 自己跑 | 基础MLLM基线 |
| FakeShield | 论文引用 | MLLM取证方法 |
| SIDA | 论文引用 | MLLM取证方法 |
| ForgeryGPT | 论文引用 | MLLM取证方法 |
| TextShield-R1 (GRPO) | 自己跑 | 直接前作基线 |
| **TextShield-R2 (Ours)** | 自己跑 | 完整方法 |

#### 5.2 消融实验设计（关键）

| 编号 | 配置 | 验证什么 | 优先级 |
|------|------|----------|--------|
| A | TextShield-R1 (GRPO + 5 outcome rewards) | 前作基线 | 必做 |
| B | DAPO + 5 outcome rewards (仅替换算法) | **DAPO vs GRPO的直接对比** | 必做 |
| C | DAPO + PRIME (无多轮) | **PRIME过程奖励的贡献** | 必做 |
| D | DAPO + 多轮推理 (无PRIME) | **多轮定位推理的贡献** | 必做 |
| E | DAPO + StepRAR + StepRVR (无PRIME/多轮) | **一致性步进奖励的贡献** | 建议做 |
| F | DAPO + PRIME + 多轮 (无Step Reward) | **核心方法组合** | 必做 |
| **G** | **DAPO + PRIME + 多轮 + Step Rewards (Full)** | **完整方法** | 必做 |

#### 5.3 深度分析实验

| 分析 | 内容 | 论文作用 |
|------|------|---------|
| DAPO训练曲线对比 | GRPO vs DAPO 的entropy/reward/loss曲线 | 证明DAPO训练更稳定 |
| PRIME步骤奖励可视化 | 对取证推理链的每步标注PRIME分数 | 证明PRIME能识别好/坏推理步骤 |
| 修正成功率分析 | 统计Round 2成功修正Round 1错误的比例 | 证明多轮推理有效 |
| 一致性错误率统计 | R1 vs R2 的逻辑矛盾输出比例 | 证明一致性奖励有效 |
| 不同尺度篡改检测率 | 按篡改区域面积分组统计性能 | 证明Zoom-in对小区域有效 |
| Token-Level Loss效果 | 分类/定位/OCR三子任务各自的性能变化 | 证明Token-Level Loss解决了任务不平衡 |

### Phase 6：可视化（优先级：高，与实验并行）

| 内容 | 论文位置 | 实现方式 |
|------|----------|----------|
| **整体框架图** | Fig.1 | draw.io：展示DAPO+PRIME+多轮推理的完整管线 |
| **GRPO vs DAPO训练动态对比** | Fig.2 | matplotlib：entropy曲线、reward曲线 |
| **多轮推理示例图** | Fig.3 | 选典型case：全局→Zoom-in→修正的完整流程 |
| **PRIME步骤奖励热力图** | Fig.4 | 对推理链逐步标注PRIME分数的可视化 |
| **修正成功/失败案例对比** | Fig.5 | 选代表性case展示 |
| **消融结果柱状图** | Fig.6 | matplotlib bar chart |
| **跨域性能雷达图** | Fig.7 | matplotlib radar chart (CIS/CTM/CL) |

### Phase 7：论文撰写（优先级：最高，与实验并行）

详见第六节。

---

## 六、论文写作详细规划

### 标题候选

1. **TextShield-R2: Advancing Tampered Text Detection with Adaptive Policy Optimization and Process-Level Forensic Reasoning** （推荐）
2. **Beyond Outcome Rewards: Process-Supervised Multi-Turn Reasoning for Tampered Text Forensics**
3. **From GRPO to DAPO: Rethinking Reinforcement Learning for Multimodal Forensic Reasoning**

### 故事线

**核心叙事**：TextShield-R1首次将RL引入篡改文本检测，但其GRPO算法和outcome-only奖励存在固有缺陷。我们从2025年RL最新进展中引入三项关键升级——DAPO解决训练不稳定性，PRIME提供过程级监督，多轮定位推理模拟人类取证专家的工作流程——实现了从"单轮猜测"到"多轮验证"的范式转变。

### 贡献列表

1. We replace GRPO with DAPO for forensic reasoning training, introducing asymmetric clipping, dynamic sampling, and token-level loss normalization to address entropy collapse, sample inefficiency, and the output-length imbalance inherent in multi-task forensic prediction.
2. We integrate PRIME (Process Reinforcement through Implicit Rewards) to provide dense, step-level supervision for forensic reasoning chains using only outcome labels, eliminating the need for expensive step-level annotations while achieving 2.5x sample efficiency.
3. We propose a multi-turn grounding-based forensic reasoning framework where the model first performs global analysis, then automatically zooms into suspected regions for local verification, learning the "where to look" policy purely through reinforcement learning.
4. We design consistency-aware step rewards (StepRAR and StepRVR) that enforce both the presence of key forensic evidence steps and logical coherence across classification, localization, and explanation outputs.

### 各Section写作要点

#### Abstract (~250词)

- 1句背景：篡改文本检测 + MLLM + RL的趋势
- 1句前作：TextShield-R1的贡献与局限（GRPO + outcome rewards + 单轮）
- 1句问题：GRPO的熵崩塌/长度偏差 + 缺少过程监督 + 单轮推理的不可逆错误
- 2句方法：DAPO + PRIME + 多轮定位推理 + 一致性步进奖励
- 1句结果：在TFR benchmark上全面超越R1和SOTA
- 1句意义：从"单轮猜测"到"多轮验证推理"的范式升级

#### 1. Introduction (1.5-2页)

- Para 1：篡改文本检测的重要性与挑战
- Para 2：MLLM + RL在该任务中的进展（TextShield-R1、FakeShield等）
- Para 3：**核心问题**——三个层面：
  - 算法层面：GRPO的熵崩塌和长度偏差（引用DAPO论文）
  - 奖励层面：outcome-only奖励无法监督推理过程（引用PRIME论文）
  - 推理层面：单轮推理无法自我验证（引用MGPO/VLM-R1）
- Para 4：人类取证专家的工作流程——"全局扫描→锁定可疑区域→放大检查→修正判断" → 启发多轮推理设计
- Para 5：方法概述 + 贡献列表 (4条)

#### 2. Related Work (1-1.5页)

- **2.1 MLLM-based Tampered Text Detection**: FakeShield, ForgeryGPT, SIDA, TextShield-R1, ETTD
- **2.2 Reinforcement Learning for Multimodal Reasoning**: GRPO → DAPO → RLVR; R1-VL/StepGRPO; VLM-R1; MGPO
- **2.3 Process Reward Models**: PRM800K, Math-Shepherd, PRIME, ThinkPRM; outcome vs process rewards

#### 3. Methodology (3-4页)

- **3.1 Preliminaries**: TextShield-R1回顾 + GRPO公式 + 问题分析
- **3.2 From GRPO to DAPO: Addressing Training Instability**
  - GRPO的三个已证实问题
  - DAPO的四项改进及其在取证任务中的意义
  - Token-Level Loss对多任务输出不平衡的解决
- **3.3 PRIME: Process Supervision Without Step Annotations**
  - 隐式PRM的训练方法
  - 步骤奖励的推断机制
  - 在线更新策略
- **3.4 Multi-Turn Grounding-Based Forensic Reasoning**
  - 两轮推理框架设计
  - 自动Zoom-in机制
  - 修正奖励函数
- **3.5 Consistency-Aware Step Rewards**
  - StepRAR：关键取证步骤检查
  - StepRVR：逻辑一致性验证
- **3.6 Overall Training Pipeline**
  - Pre-training (复用R1) → Multi-turn SFT Cold Start → DAPO with PRIME + Step Rewards

#### 4. Experiments (3-4页)

- **4.1 Experimental Setup**: 数据集(TFR)、指标、实现细节
- **4.2 Comparison with State-of-the-Art**:
  - Table 1: TFR Benchmark 4个测试集全面对比 (Test / CIS / CTM / CL)
  - 重点对比 R1 → R2 的提升
- **4.3 Ablation Study**:
  - Table 2: 七组消融实验 (A→G)
  - 每个创新点的独立贡献量化
- **4.4 Training Dynamics Analysis** (DAPO专属):
  - Fig: GRPO vs DAPO 的entropy/reward训练曲线
  - 分析DAPO如何维持探索和稳定训练
- **4.5 Process Reward Analysis** (PRIME专属):
  - PRIME步骤奖励的可视化案例
  - 有PRIME vs 无PRIME的推理链质量对比
- **4.6 Multi-Turn Reasoning Analysis**:
  - 修正成功率统计
  - 不同尺度篡改的Zoom-in效果
  - 成功修正 vs 过度修正的案例分析
- **4.7 Qualitative Analysis**:
  - 完整多轮推理流程展示
  - R1 vs R2 的检测结果对比

#### 5. Conclusion (0.5页)

- 总结方法和发现
- 局限性：(1) 多轮推理增加延迟 (2) PRIME依赖足够多的outcome标签 (3) 仅在TFR上验证
- 未来方向：(1) 高效多轮推理（Early Exit） (2) 像素级定位 (3) 视频篡改检测

### 关键参考文献清单

| 引用内容 | 文献 | 用途 |
|----------|------|------|
| DAPO | ByteDance Seed & Tsinghua AIR, arXiv:2503.14476 | **核心方法** |
| PRIME | arXiv:2502.01456 | **核心方法** |
| MGPO | arXiv:2507.05920 | **核心方法启发** |
| VLM-R1 | arXiv:2504.07615 | **视觉定位RL** |
| StepGRPO / R1-VL | arXiv:2503.12937 | **步进奖励设计** |
| GRPO | Shao et al., 2024 | 基础算法 |
| DeepSeek-R1 | arXiv:2501.12948 | RL推理的里程碑 |
| Dr. GRPO | 2025 | GRPO的长度偏差分析 |
| lambda-GRPO | arXiv:2510.06870 | GRPO变体统一框架 |
| PRM800K | Lightman et al., 2023 | PRM的理论基础 |
| Math-Shepherd | Wang et al., 2024 | PRM训练方法 |
| ThinkPRM | arXiv:2504.16828 | 生成式PRM |
| Self-Refine | Madaan et al., NeurIPS 2023 | 自反思推理 |
| Reflexion | Shinn et al., NeurIPS 2023 | 反思机制 |
| TextShield-R1 | Qu et al., AAAI 2026 | 直接前作 |
| FakeShield | Xu et al., 2024 | 对比方法 |
| ForgeryGPT | Li et al., arXiv:2410.10238 | 对比方法 |
| SIDA | 2025 | 对比方法 |
| ETTD | arXiv:2412.14816 | 相关工作 |

---

## 七、预期实验结果（目标）

### 主实验目标（vs TextShield-R1）

| 指标 | R1 | R2 目标 | 预期提升 | 主要贡献来源 |
|------|-----|---------|---------|-------------|
| Test Cls. | 88.1 | 91+ | +3% | DAPO稳定训练 + 多轮修正 |
| Test OCR | 47.6 | 53+ | +5% | Zoom-in局部放大 |
| Test Loc. | 57.8 | 63+ | +5% | 多轮定位 + PRIME过程奖励 |
| Test Res. | 58.8 | 65+ | +6% | PRIME + 一致性奖励 |
| CIS Cls. | 72.9 | 77+ | +4% | DAPO探索性 + 泛化 |
| CTM Cls. | 88.8 | 91+ | +2% | 多轮验证修正 |
| CL Cls. | 85.5 | 88+ | +2.5% | 一致性奖励减少矛盾输出 |

### 消融实验预期

| 配置 | Test Cls. | Test Loc. | Test Res. |
|------|-----------|-----------|-----------|
| A: R1 (GRPO) | 88.1 | 57.8 | 58.8 |
| B: DAPO only | 89.5 (+1.4) | 59.0 (+1.2) | 60.5 (+1.7) |
| C: DAPO + PRIME | 90.0 (+0.5) | 60.5 (+1.5) | 62.5 (+2.0) |
| D: DAPO + 多轮 | 90.0 (+0.5) | 61.5 (+2.5) | 62.0 (+1.5) |
| E: DAPO + Step Rewards | 89.8 (+0.3) | 59.5 (+0.5) | 61.5 (+1.0) |
| F: DAPO + PRIME + 多轮 | 90.5 (+0.5) | 62.5 (+1.0) | 64.0 (+2.0) |
| **G: Full** | **91+** | **63+** | **65+** |

---

## 八、代码实现计划

### 需要新建的文件

| 文件 | 功能 | 优先级 |
|------|------|--------|
| `dapo_trainer.py` | DAPO训练器（扩展ms-swift的GRPO） | 最高 |
| `prime_prm.py` | PRIME隐式过程奖励模型 | 最高 |
| `multi_turn_data.py` | 多轮推理训练数据构建 | 高 |
| `zoom_in.py` | 自动裁剪放大模块 | 高 |
| `orm_v2.py` | 扩展奖励函数（一致性 + 步进 + 修正） | 高 |
| `eval_multi_turn.py` | 多轮推理评估（修正成功率等） | 中 |
| `eval_consistency.py` | 一致性评估脚本 | 中 |
| `visualize_prime.py` | PRIME步骤奖励可视化 | 中 |
| `visualize_training.py` | DAPO vs GRPO训练曲线可视化 | 中 |

### 需要修改的文件

| 文件 | 修改内容 |
|------|----------|
| `pipeline.py` | 支持多轮推理的输出解析（`<verify>`, `<final_answer>`） |
| `orm.py` | 新增奖励函数注册 |
| `prepare_sft_data.py` | 扩展为多轮格式数据生成 |

---

## 九、时间规划

### 如果目标 ECCV 2026（截稿约7月初）

| 阶段 | 时间 | 任务 | 产出 |
|------|------|------|------|
| **Week 1-2** | 3月中-3月底 | DAPO实现 + 集成 + 验证 | DAPO训练器，A/B消融对比 |
| **Week 3-4** | 4月初-4月中 | PRIME训练 + 多轮数据构建 | PRM模型 + 多轮SFT数据 |
| **Week 5-6** | 4月中-4月底 | 多轮Cold Start SFT + 一致性奖励实现 | 冷启动模型 + 新奖励函数 |
| **Week 7-9** | 5月初-5月中 | 完整DAPO训练（迭代调优奖励权重） | 最优R2模型 |
| **Week 10-11** | 5月中-6月初 | 主实验 + 消融 + 深度分析 | 所有实验数据 |
| **Week 12-13** | 6月初-6月中 | 可视化 + 论文初稿 | 完整论文v1 |
| **Week 14-15** | 6月中-7月初 | 论文修改 + 润色 + 提交 | 最终提交版 |

### 如果目标 ACM MM 2026（截稿约4月中）

| 阶段 | 时间 | 任务 |
|------|------|------|
| **Week 1** | 3月15-21 | DAPO实现 + PRIME数据构建（集中冲刺） |
| **Week 2** | 3月22-28 | PRIME训练 + 多轮数据 + Cold Start SFT |
| **Week 3** | 3月29-4月4 | 完整DAPO训练 + 实验 |
| **Week 4** | 4月5-11 | 消融 + 深度分析 + 论文撰写 |
| **Week 5** | 4月12-18 | 可视化 + 论文提交 |

---

## 十、与竞争工作的差异化表述

### vs TextShield-R1 (自己的前作)

| 维度 | R1 | R2 |
|------|-----|-----|
| RL算法 | GRPO（有长度偏差/熵崩塌） | DAPO（解决已知缺陷） |
| 奖励层级 | Outcome-only (5个) | Outcome + Process + Step (9个) |
| 过程监督 | 无 | PRIME隐式PRM（无需步骤标注） |
| 推理范式 | 单轮推理 | 多轮定位推理（全局→Zoom-in→修正） |
| 训练信号密度 | 稀疏（最终结果） | 密集（每步过程奖励） |

### vs FakeShield / ForgeryGPT

- 他们是SFT-only方法，无RL训练
- 我们有完整的RL训练管线 + 过程级奖励
- 多轮推理是独有的设计

### vs DeepSeek-R1 / VLM-R1

- DeepSeek-R1/VLM-R1是通用推理/检测，我们专注于取证领域
- 我们的PRIME + 一致性奖励是为取证任务量身定制的
- 多轮Zoom-in是取证领域特有的需求

---

## 十一、Oral/Spotlight录取策略

### 审稿人可能的问题与回应

| 可能的质疑 | 回应策略 |
|-----------|---------|
| "DAPO只是换了个优化器，新颖性有限" | 强调Token-Level Loss对多任务取证输出的特殊意义 + 消融实验量化DAPO的独立贡献 |
| "PRIME不是你们提出的" | 强调首次将PRIME应用于多模态取证推理 + 无需步骤标注的实用性 |
| "多轮推理增加延迟" | 报告延迟数据 + 提出Early Exit策略（高置信度时跳过Round 2）+ 强调精度优先 |
| "为什么不用更强的基座模型" | 7B模型部署成本低 + 方法论与模型规模正交 + 7B上的提升说明方法有效 |

### 提升录取概率的关键

1. **方法论深度**：不是简单"换DAPO涨点"，而是系统性地分析GRPO在取证任务中的缺陷并针对性解决
2. **消融完整**：7组消融明确量化每个组件的贡献
3. **分析透彻**：训练动态对比、PRIME奖励可视化、修正成功率等深度分析
4. **可视化直观**：多轮推理过程展示、PRIME热力图、R1 vs R2对比

---

## 十二、风险评估与预案

| 风险 | 影响 | 预案 |
|------|------|------|
| ms-swift不支持DAPO | 需要大量开发 | 方案C：仅实现Token-Level Loss + Dynamic Sampling（核心改进） |
| PRIME训练数据不够 | PRM质量差 | 增加R1模型的trajectory采样数；或退回使用规则化的StepRAR/StepRVR替代PRIME |
| 多轮推理的DAPO训练不收敛 | 核心创新点失效 | 先用SFT训练多轮推理能力，DAPO仅优化单轮部分 |
| DAPO提升不显著 | DAPO贡献难以支撑 | 聚焦PRIME+多轮推理两个核心创新点，DAPO降为"training improvement" |
| 改进幅度不够 | 论文说服力不足 | 聚焦推理质量(Res.)和定位(Loc.)指标的大幅提升 + 定性分析 |
| 计算资源不足 | 训练时间过长 | 减小num_generations(8→4)、缩短训练epoch、使用更小的PRM |

---

## 十三、提交 Checklist

- [ ] DAPO训练器实现并验证
- [ ] PRIME隐式PRM训练完成并集成
- [ ] 多轮推理数据构建完成
- [ ] 完整DAPO训练完成
- [ ] 主实验在TFR Benchmark 4个测试集上完成
- [ ] 7组消融实验完成
- [ ] 深度分析（训练曲线、PRIME可视化、修正率）完成
- [ ] 7张图表准备完成
- [ ] 论文格式符合目标会议要求
- [ ] 参考文献格式统一
- [ ] 拼写/语法检查通过
- [ ] 匿名化检查（如双盲评审）
- [ ] 代码清理并准备开源
