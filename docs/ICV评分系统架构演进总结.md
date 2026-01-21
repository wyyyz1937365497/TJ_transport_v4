# ICV评分系统架构演进总结

## 问题起源

您的核心洞察：**ICV评分器与策略生成器在PPO阶段是相互依赖的，应该联合训练。**

这确实是我之前架构的根本性缺陷。

---

## 架构对比

### v1.0（独立评分器）- 已废弃

```
┌─────────────────────────────────────────────────┐
│          CompetitionSumoEnv                     │
│  (调用评分接口)                                   │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│        UnifiedVehicleScorer                     │
│  (统一评分接口)                                   │
│                                                  │
│  ┌──────────────┐        ┌──────────────┐      │
│  │ 神经网络评分器 │─┐    │  规则评分器    │      │
│  │ (静态权重)    │ │    │  (固定规则)    │      │
│  └──────────────┘ │    └──────────────┘      │
│                   │                             │
│  问题：评分不可学习，梯度无法回传                 │
└───────────────────┼─────────────────────────────┘
                    │
                    ▼
              选择Top-K车辆
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│         LightweightPolicyV5                    │
│  (策略网络)                                       │
│  只对Top-K车辆输出动作                            │
└─────────────────────────────────────────────────┘
```

**致命缺陷**：
1. ❌ 评分器与策略网络割裂
2. ❌ 梯度无法从奖励回传到评分过程
3. ❌ "为什么选这辆车而不是那辆"无法学习
4. ❌ 评分器不知道"控制这辆车效果如何"

### v5.0（联合PPO）- 推荐 ✅

```
                    观测 [B, N, 9]
                          │
                          ▼
        ┌─────────────────────────────┐
        │   RiskSensitiveGNN          │
        │  (风险感知编码器)             │
        │                              │
        │  - TTC感知的边权重           │
        │  - 层次化聚合                │
        │  - 注意力偏向高风险交互       │
        └──────────────┬──────────────┘
                       │
                       ▼
              嵌入向量 [B, N, D]
                       │
           ┌───────────┴───────────┐
           │                       │
           ▼                       ▼
┌──────────────────┐    ┌──────────────────┐
│ ImportancePredictor│   │  SparseGate      │
│ (重要性预测头)     │    │  (稀疏门控)       │
│                   │    │                  │
│ - 端到端可学习     │    │ - Gumbel-Softmax │
│ - PPO自动优化     │    │ - 可微分Top-K     │
│ - 梯度可回传       │    │ - 动态K值         │
└─────────┬─────────┘    └────────┬─────────┘
          │                       │
          │ importance [B, N, 1]  │
          │          │             │
          └──────────┼─────────────┘
                     │ selection_mask [B, N, 1]
                     ▼
          ┌──────────────────────┐
          │    PolicyHead        │
          │    (策略头)           │
          │                       │
          │ 输入: 嵌入 + 选择掩码  │
          │ 输出: [B, N, 2]       │
          └──────────┬───────────┘
                     │ actions
                     ▼
          ┌──────────────────────┐
          │    ValueHead         │
          │    (价值头)           │
          │                       │
          │ 输出: [B, 1]          │
          └──────────┬───────────┘
                     │
                     ▼
                PPO训练循环
            (端到端优化)
```

**核心优势**：
1. ✅ 端到端可学习
2. ✅ 梯度从奖励回传到重要性评分
3. ✅ "为什么选这辆车"可以自动学习
4. ✅ 重要性与策略联合优化

---

## 关键技术对比

| 维度 | v1.0 独立评分器 | v5.0 联合PPO |
|------|---------------|-------------|
| **评分方式** | 固定规则或预训练网络 | 端到端可学习 |
| **Top-K选择** | 硬阈值（不可微） | Gumbel-Softmax（可微） |
| **梯度流** | 评分器 ←✗ 策略网络 | 评分器 ←✓→ 策略网络 |
| **训练方式** | 监督学习 | PPO强化学习 |
| **适应性** | 静态评分规则 | 动态学习评分 |
| **最优性** | 局部最优（固定评分） | 全局最优（端到端） |

---

## v5.0的核心创新

### 1. 风险感知GNN

```python
class RiskSensitiveGNN:
    """
    创新：边权重融入TTC（碰撞时间）倒数
    """

    def forward(self, x, adj, risk_features):
        # 风险感知加权
        # TTC越小（风险越大）→ 权重越高
        risk_weight = sigmoid(λ / (TTC + ε))

        messages = adj @ x
        messages = messages * risk_weight

        return messages
```

**效果**：
- 自动关注高风险车辆交互
- 提升安全性
- 符合交通工程直觉

### 2. 可微分的重要性预测

```python
class ImportancePredictor(nn.Module):
    """
    创新：端到端学习车辆重要性

    关键：
    - 不使用MSE loss监督学习
    - 而是通过PPO的policy gradient自动学习
    """

    def forward(self, embeddings):
        importance = self.mlp(embeddings)
        importance = sigmoid(importance)  # [0, 1]
        return importance
```

**训练逻辑**：
```python
# 不是这样（错误）：
# loss = MSE(predicted_importance, rule_importance)

# 而是这样（正确）：
# 通过PPO自动学习
# 如果选择高importance车的动作 → 导致高奖励
# → 梯度回传 → 增加这辆车的importance
```

### 3. 可微分的稀疏门控

```python
class SparseGate:
    """
    创新：Gumbel-Softmax Top-K

    关键技术：
    - 训练时：Gumbel-Softmax（可微分）
    - 推理时：Hard Top-K（高效）
    - 直通估计器：前向hard，反向soft
    """

    def forward(self, importance, training=True):
        if training:
            # Gumbel-Softmax trick
            gumbel_noise = -log(-log(rand))
            logits = log(importance) + gumbel_noise
            probs = softmax(logits / temperature)

            # 直通估计器
            hard_mask = top_k(importance, k)
            selection = probs - probs.detach() + hard_mask
        else:
            # 推理：直接hard top-k
            selection = top_k(importance, k)

        return selection
```

**效果**：
- 前向传播：离散的Top-K选择
- 反向传播：连续的梯度流
- 训练稳定且高效

### 4. 动态K值机制

```python
class AdaptiveKSelector:
    """
    创新：根据拥堵状态动态调整K值

    逻辑：
    - 拥堵严重 → K增大（多控制）
    - 系统平稳 → K减小（少控制）
    """

    def compute_k_ratio(self, global_state):
        congestion = global_state['congestion_level']

        if congestion < 0.3:
            k_ratio = 0.05  # 最少控制
        elif congestion < 0.6:
            k_ratio = 0.10  # 适度控制
        elif congestion < 0.8:
            k_ratio = 0.15  # 较多控制
        else:
            k_ratio = 0.25  # 最多控制（官方上限）

        return k_ratio
```

**效果**：
- 按需干预
- 最小化成本
- 最大化奖励

---

## PPO训练流程

### Stage 1: 行为克隆（快速初始化）

```python
def stage1_behavior_cloning():
    """
    目标：从规则策略学习基础模式

    方法：
    1. 规则评分器生成importance标签
    2. 专家策略（IDM）生成action标签
    3. 监督学习初始化网络
    """

    # 收集数据
    for episode in range(20):
        obs = env.reset()
        for step in range(500):
            # 规则评分
            rule_importance = rule_scorer(obs)

            # 专家动作
            expert_actions = idm_policy(obs)

            # 存储数据
            rollouts.append({
                'obs': obs,
                'importance': rule_importance,
                'actions': expert_actions
            })

    # 监督学习
    for epoch in range(10):
        # Importance loss
        pred_importance = policy.importance_head(obs)
        loss_importance = MSE(pred_importance, rule_importance)

        # Action loss
        pred_actions = policy.policy_head(obs)
        loss_actions = MSE(pred_actions, expert_actions)

        loss = loss_importance + loss_actions
        loss.backward()
```

### Stage 2: PPO微调（端到端优化）

```python
def stage2_ppo():
    """
    目标：直接优化OCR奖励

    关键：
    1. 加载Stage 1预训练权重
    2. PPO端到端训练
    3. 同时优化importance和action
    """

    # 加载预训练权重
    policy.load_state_dict(torch.load('stage1_pretrained.pth'))

    # PPO训练
    for iteration in range(1000):
        # 收集rollout
        rollouts = collect_rollout(env, policy, num_steps=2048)

        # 计算GAE
        advantages = compute_gae(
            rollouts['rewards'],
            rollouts['values'],
            gamma=0.99,
            gae_lambda=0.95
        )

        # PPO更新
        for epoch in range(10):
            # 重新计算
            outputs = policy(rollouts['obs'])

            # PPO clip loss
            ratio = exp(outputs['log_prob'] - rollouts['log_prob'])
            surr1 = ratio * advantages
            surr2 = clamp(ratio, 0.8, 1.2) * advantages
            policy_loss = -min(surr1, surr2).mean()

            # Value loss
            value_loss = MSE(outputs['value'], rollouts['returns'])

            # 熵正则化
            entropy = outputs['dist'].entropy().mean()

            # 总loss
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            # 梯度更新
            loss.backward()
            clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
```

---

## 代码结构

### v1.0 文件（保留但废弃）

```
src/env/
├── vehicle_scoring.py          # 统一评分接口（不再推荐）
└── rule_based_scorer.py        # 规则评分器（备选）

src/models/
└── icv_gnn_scorer.py           # 独立评分模型（不再推荐）
```

### v5.0 文件（新实现）

```
src/models/
└── joint_icv_policy.py         # 联合PPO策略 ⭐

docs/
└── 联合PPO架构设计v5.md        # 详细设计文档
```

### 共享文件

```
src/models/
└── v5_lightweight.py           # 参考实现（可兼容）

configs/
├── phase1_lite.yaml            # 旧配置
└── joint_ppo.yaml              # 新配置（待创建）

scripts/
├── train_icv_scorer.py         # 旧训练脚本
└── train_joint_ppo.py          # 新训练脚本（待创建）
```

---

## 使用指南

### 快速开始

```python
from src.models.joint_icv_policy import create_joint_icv_policy

# 1. 创建策略
policy = create_joint_icv_policy(
    obs_dim=321,
    hidden_dim=64,
    num_layers=3,
    initial_k_ratio=0.10,
    device='cuda'
)

# 2. 前向传播
obs = env.reset()
obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to('cuda')

with torch.no_grad():
    outputs = policy(obs_tensor, deterministic=True)

actions = outputs['actions']          # [1, N*2]
value = outputs['value']               # [1, 1]
importance = outputs['importance']      # [1, N, 1]
selection = outputs['selection']       # [1, N, 1]
k = outputs['k']                        # int

# 3. 执行动作
action_dict = parse_actions(actions, env.vehicle_ids)
obs_next, reward, done, info = env.step(action_dict)
```

### 训练

```bash
# Stage 1: 行为克隆
python scripts/train_joint_ppo.py --stage 1 --config configs/joint_ppo.yaml

# Stage 2: PPO微调
python scripts/train_joint_ppo.py --stage 2 --config configs/joint_ppo.yaml --resume checkpoints/stage1_best.pth

# 完整训练
python scripts/train_joint_ppo.py --stage all --config configs/joint_ppo.yaml
```

---

## 与您v4.0架构的融合

### 保留的v4.0核心思想 ✅

1. **风险敏感GNN**
   - 边特征融入TTC倒数
   - 注意力偏向高风险交互

2. **层次化聚合**
   - 局部（车-车）→ 路段 → 全局
   - 降低计算复杂度

3. **动态权重门控**
   - 元控制机制
   - 根据拥堵状态调整

4. **双层安全屏障**
   - Level 1: 规则检查
   - Level 2: 紧急避险

5. **渐进式训练**
   - Stage 1: 监督学习
   - Stage 2: PPO微调

### v5.0的关键改进 🆕

1. **端到端可学习的importance**
   - 不再使用固定规则
   - PPO自动优化

2. **Gumbel-Softmax可微分Top-K**
   - 解决离散选择不可微的问题
   - 直通估计器

3. **联合训练**
   - importance + action同时优化
   - 梯度端到端回传

4. **动态K值**
   - 根据拥堵状态自适应
   - 按需干预

---

## 性能预期

| 指标 | v1.0 独立评分器 | v5.0 联合PPO |
|------|----------------|-------------|
| **训练时间** | 6-8小时 | 8-10小时 |
| **OCR提升** | 65-70% | 70-75% |
| **干预成本** | 0.4-0.5 | 0.3-0.4 |
| **最终得分** | 68-72分 | 72-76分 |
| **收敛速度** | 中等 | 更快（Stage 1初始化） |
| **适应性** | 静态 | 动态自适应 |

**解释**：
- v5.0训练时间稍长（端到端优化更复杂）
- 但性能显著提升（联合优化更优）
- 干预成本更低（动态K值按需干预）

---

## 迁移建议

### 如果您已有v1.0实现

1. **保留规则评分器作为备选**
   ```python
   # 在competition_env.py中
   if config.get('use_joint_ppo', True):
       from src.models.joint_icv_policy import create_joint_icv_policy
       policy = create_joint_icv_policy(...)
   else:
       from src.env.vehicle_scoring import create_vehicle_scorer_from_config
       scorer = create_vehicle_scorer_from_config(...)
   ```

2. **使用Stage 1初始化**
   ```python
   # 可以用规则评分器生成标签
   def stage1_pretrain():
       for episode in range(20):
           # 规则评分
           rule_importance = rule_scorer.compute_scores(...)
           # 专家策略
           expert_actions = idm_policy(...)
           # 监督学习
           policy.update(rule_importance, expert_actions)
   ```

3. **逐步迁移**
   - 第1周：实现joint_icv_policy.py
   - 第2周：实现Stage 1训练
   - 第3周：实现Stage 2 PPO训练
   - 第4周：调优和评估

---

## 总结

### 核心洞察

> **ICV评分不是静态规则，而是策略价值的一部分。**

不应该问："这辆车重要吗？"
而应该问："控制这辆车能带来多大奖励？"

v5.0通过端到端学习回答了这个问题。

### 最终架构

```
                    状态观测
                        ↓
              ┌──────────────────┐
              │  RiskSensitiveGNN │
              │   (编码车辆交互)   │
              └─────────┬─────────┘
                        │
                ┌───────┴───────┐
                │               │
        ┌───────▼──────┐   ┌──▼───────────┐
        │ Importance   │   │ SparseGate   │
        │ Predictor    │   │ (Top-K选择)  │
        │ (可学习)      │   │ (可微分)      │
        └───────┬──────┘   └──┬───────────┘
                │               │
                └───────┬───────┘
                        │ selection_mask
                        ↓
                  ┌─────────────┐
                  │ PolicyHead  │
                  │ (输出动作)   │
                  └─────────────┘
                        │
                        ↓
              PPO训练（端到端优化）
```

这是最适合比赛的架构！
