# ICV评分与策略生成的联合PPO架构设计

## 核心洞察

### 问题分析

我之前实现的ICV评分器存在根本性缺陷：

**错误的架构**（v1.0）:
```
独立评分器 → 选择Top-K车辆 → 策略网络 → 输出动作
     ↓
  静态评分（不可学习）
```

**问题**：
1. 评分器与策略网络割裂，无法端到端优化
2. 评分器不知道"控制这辆车效果如何"
3. 策略网络不知道"为什么选这辆车而不是那辆"
4. 梯度无法从奖励回传到选择过程

**正确的架构**（v5.0）:
```
         状态观测
             ↓
    ┌────────────────┐
    │   GNN编码器    │
    │  (学习表征)     │
    └────────────────┘
             ↓
    ┌────────────────┐
    │  影响力预测头   │  ← 端到端可学习
    │  (价值评分)     │
    └────────────────┘
             ↓
    ┌────────────────┐
    │  稀疏门控机制   │  ← Top-K选择（可微分）
    └────────────────┘
             ↓
    ┌────────────────┐
    │  策略头        │  ← 输出动作
    │  (只对Top-K)    │
    └────────────────┘
             ↓
         动作输出
```

**关键改变**：
- 影响力评分 → **策略价值的一部分**
- Top-K选择 → **Gumbel-Softmax可微分**
- 联合优化 → **PPO端到端训练**

---

## 比赛环境分析

### 路网拓扑（从net.xml提取）

```
主路结构（单向）：
E1 → E2 → E3 → E5 → E6 → E7 → E8 → E9 → E10 → E11 → E12 → E13
│    │    │    │    │    │    │    │    │    │     │     │
│    │    │    │    │    │    │    │    │    │     │     │
J4   J5   J6   J10  J11  J12  J13  J14  J15  J16  J17  J18
│    │         │              │         │    │
│    │         │              │         │    │
└─E23─┘       E16            E18       E24  E20
(汇入)       (驶出)         (驶出)    (驶出)(驶出)

关键汇入点：
- J5: E23汇入（1500 veh/h）
- J15: E17汇入（1500 veh/h）← 最严重瓶颈
- J17: E19汇入（1500 veh/h）← 最严重瓶颈

瓶颈区域位置（归一化s坐标）：
- J15: s ≈ 1300-1400
- J17: s ≈ 1750-1800
```

### 车辆配置（从routes.xml提取）

```
车辆类型：
- HV（人工驾驶）: 75%, accel=0.8, decel=4.5, maxSpeed=30 m/s
- CV（可控车辆）: 25%, 相同参数

流量分布：
- 主路正向: 800+500+1000+400+200 = 2900 veh/h
- 主路反向: 1000 veh/h
- E23汇入: 1500 veh/h
- E15汇入: 1500 veh/h
- E17汇入: 1500 veh/h
- E19汇入: 1500 veh/h
────────────────────────────
总计: 9900 veh/h

关键参数：
- 时间步长: 0.1秒
- 最大时间: 3600秒（36000步）
- 最大在线车辆: 600辆
```

### 评分公式（从docs提取）

```
S_total = (W_efficiency × S_efficiency + W_stability × S_stability) × P_intervention

其中：
S_efficiency = 100 × max(0, (OCR_AI - OCR_Base) / OCR_Base)
S_stability = 100 × [0.4 × I_σv + 0.6 × I_|a|]
P_intervention = exp(-k × C_int)

C_int = (α × Σacmd + β × Σδlc) / (T_total × N_ICV)

参数：
- W_efficiency = 0.7（初赛）
- W_stability = 0.3（初赛）
- k = 0.1
- α = 1.0（加速度权重）
- β = 5.0（换道权重，5倍成本）
- N_ICV = 150（600 × 0.25，官方固定值）
```

**关键洞察**：
1. **干预成本至关重要**：β=5.0意味着换道成本极高
2. **效率是主要目标**：70%权重在OCR
3. **相对性评分**：与baseline比较，不是绝对值
4. **稀疏控制优势**：控制车辆越少，P_intervention越高

---

## v5.0联合PPO架构设计

### 整体架构

```python
class JointICVPolicy(nn.Module):
    """
    ICV评分与策略生成的联合网络

    核心思想：
    1. GNN编码车辆交互
    2. 影响力预测头估计每辆车的价值
    3. 稀疏门控选择Top-K车辆
    4. 策略头输出控制动作
    5. 端到端PPO训练
    """

    def __init__(self, config):
        # 1. GNN编码器
        self.gnn_encoder = HeteroGNN(...)
        # 输入: [B, N, 9] 车辆状态
        # 输出: [B, N, D] 嵌入向量

        # 2. 影响力预测头（Learned Importance Scorer）
        self.importance_head = ImportancePredictor(...)
        # 输入: [B, N, D] GNN嵌入
        # 输出: [B, N, 1] 重要性评分

        # 3. 稀疏门控机制（Differentiable Top-K）
        self.sparse_gate = SparseGate(...)
        # 输入: [B, N, 1] 重要性评分
        # 输出: [B, N, 1] 选择概率（Gumbel-Softmax）

        # 4. 策略头（只对选中的车辆输出动作）
        self.policy_head = PolicyNetwork(...)
        # 输入: [B, N, D+1] GNN嵌入 + 选择标记
        # 输出: [B, N, 2] (加速度, 换道概率)

        # 5. 价值头（全局状态评估）
        self.value_head = ValueNetwork(...)
        # 输入: [B, N, D] 全局池化嵌入
        # 输出: [B, 1] 状态价值
```

### 前向传播流程

```python
def forward(self, obs, deterministic=False):
    # 1. 解析观测
    vehicle_states = parse_observation(obs)  # [B, N, 9]

    # 2. GNN编码
    embeddings = self.gnn_encoder(vehicle_states)  # [B, N, D]

    # 3. 预测重要性（Learned）
    importance_scores = self.importance_head(embeddings)  # [B, N, 1]
    importance_scores = torch.sigmoid(importance_scores)  # [0, 1]

    # 4. 稀疏门控（Differentiable Top-K）
    if self.training:
        # 训练时：Gumbel-Softmax（可微分）
        selection_probs = self.sparse_gate(importance_scores)  # [B, N, 1]
        selection_mask = selection_probs
    else:
        # 推理时：Hard Top-K（不可微分但高效）
        k = self.compute_dynamic_k(importance_scores)
        selection_mask = hard_top_k(importance_scores, k)

    # 5. 策略输出（只对选中的车辆）
    raw_actions = self.policy_head(
        torch.cat([embeddings, selection_mask], dim=-1)
    )  # [B, N, 2]

    # 应用选择掩码
    actions = raw_actions * selection_mask

    # 6. 价值估计
    value = self.value_head(embeddings)  # [B, 1]

    return {
        'actions': actions,
        'value': value,
        'importance': importance_scores,
        'selection': selection_mask,
    }
```

### 关键模块详解

#### 1. 风险敏感GNN编码器

```python
class RiskSensitiveGNN(nn.Module):
    """
    风险感知的异构图神经网络

    创新点：
    - 边特征融入TTC（碰撞时间）倒数
    - 注意力偏向高风险交互
    - 层次化聚合（局部→路段→全局）
    """

    def __init__(self, node_dim=9, hidden_dim=64, num_layers=3):
        # 输入特征：9维
        # [s, d, vs, vd, speed, accel, lane, angle, in_bottleneck]

        # 特征增强（添加风险特征）
        self.risk_encoder = RiskFeatureEncoder(...)
        # 输入: TTC, THW, speed_diff
        # 输出: risk_embedding

        # GNN层（带偏置注意力）
        self.gnn_layers = nn.ModuleList([
            BiasedAttentionGNNLayer(...)
            for _ in range(num_layers)
        ])
        # 偏置 = 1 / (TTC + ε)  # TTC越小，权重越高

        # 层次化池化
        self.lane_pooling = LaneLevelPooling(...)  # 车道级
        self.section_pooling = SectionLevelPooling(...)  # 路段级
```

**关键公式**：
```
注意力权重:
α_ij = softmax(Q_i · K_j + β_risk)

其中 β_risk = λ / (TTC_ij + ε)

如果 TTC_ij < 3.0s:
    α_ij ← α_ij · 5.0  # 强制关注高风险
```

#### 2. 影响力预测头（Learned）

```python
class ImportancePredictor(nn.Module):
    """
    学习车辆的影响力评分

    不同于规则评分器，这是**端到端学习**的：
    - 输入：GNN嵌入（包含全局信息）
    - 输出：影响力评分
    - 目标：最大化累积奖励

    训练信号：
    - 不是MSE loss（监督学习）
    - 而是PPO的policy gradient（强化学习）
    """

    def __init__(self, hidden_dim=64):
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(32, 1)
        )

    def forward(self, embeddings):
        # embeddings: [B, N, D]

        # 多头设计（解耦不同维度的影响力）
        importance = self.mlp(embeddings)  # [B, N, 1]
        importance = torch.sigmoid(importance)  # [0, 1]

        return importance
```

**训练逻辑**：
```python
# 不是这样：
# loss = MSE(predicted_importance, rule_importance)  # 错误！

# 而是这样：
# 通过PPO的policy gradient自动学习
# 如果选择高importance车的动作导致高奖励
# → 梯度回传会增加这辆车的importance
```

#### 3. 稀疏门控机制（Differentiable Top-K）

```python
class SparseGate(nn.Module):
    """
    可微分的稀疏门控

    目标：
    - 训练时：Gumbel-Softmax（可微分）
    - 推理时：Hard Top-K（高效）

    动态K值：
    - 根据全局拥堵状态调整
    - 拥堵严重 → K增大
    - 系统平稳 → K减小
    """

    def __init__(self, initial_k_ratio=0.1):
        self.k_ratio = initial_k_ratio
        self.temperature = 1.0  # Gumbel温度

    def forward(self, importance_scores, training=True):
        # importance_scores: [B, N, 1]

        if training:
            # Gumbel-Softmax trick（可微分的Top-K）
            # 1. 添加Gumbel噪声
            gumbel_noise = -torch.log(-torch.log(
                torch.rand_like(importance_scores) + 1e-10) + 1e-10
            ))
            logits = torch.log(importance_scores + 1e-10) + gumbel_noise

            # 2. 温度缩放（训练时逐渐降低）
            scaled_logits = logits / self.temperature

            # 3. Softmax归一化
            probs = torch.softmax(scaled_logits, dim=1)  # [B, N, 1]

            # 4. 直通估计器（Straight-Through Estimator）
            # 前向：hard top-k
            # 反向：soft梯度
            k = int(self.k_ratio * importance_scores.size(1))
            k = max(k, 5)  # 至少5辆
            k = min(k, importance_scores.size(1))  # 最多N辆

            _, top_k_indices = torch.topk(importance_scores.squeeze(-1), k, dim=1)
            hard_mask = torch.zeros_like(probs)
            hard_mask.scatter_(1, top_k_indices.unsqueeze(-1), 1.0)

            # 直通估计器
            selection_mask = probs - probs.detach() + hard_mask

        else:
            # 推理时：直接hard top-k
            k = int(self.k_ratio * importance_scores.size(1))
            k = max(k, 5)
            k = min(k, importance_scores.size(1))

            _, top_k_indices = torch.topk(importance_scores.squeeze(-1), k, dim=1)
            selection_mask = torch.zeros_like(importance_scores)
            selection_mask.scatter_(1, top_k_indices.unsqueeze(-1), 1.0)

        return selection_mask
```

**关键技术**：
1. **Gumbel-Softmax**：使离散选择可微分
2. **直通估计器**：前向hard，反向soft
3. **动态K值**：根据拥堵状态调整

#### 4. 策略头

```python
class PolicyNetwork(nn.Module):
    """
    策略网络

    输出：
    - 加速度：连续动作 [-3, 2] m/s²
    - 换道概率：连续动作 [0, 1]
    """

    def __init__(self, hidden_dim=64):
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim + 1, 128),  # +1 for selection mask
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 2)  # 2 actions
        )

        # 可学习的log_std（高斯策略）
        self.log_std = nn.Parameter(torch.zeros(2))

    def forward(self, features, deterministic=False):
        # features: [B, N, D+1]

        action_mean = self.actor(features)  # [B, N, 2]

        # 加速度：[-1, 1] → [-3, 2]
        accel = action_mean[:, :, 0:1]
        accel = (accel + 1.0) / 2.0 * (2.0 - (-3.0)) + (-3.0)

        # 换道：[-1, 1] → [0, 1]
        lane_change = action_mean[:, :, 1:2]
        lane_change = (lane_change + 1.0) / 2.0
        lane_change = torch.clamp(lane_change, 0.0, 1.0)

        actions = torch.cat([accel, lane_change], dim=-1)

        if not deterministic:
            # 添加探索噪声
            std = torch.exp(self.log_std).clamp(min=-5, max=2)
            noise = torch.randn_like(actions) * std
            actions = actions + noise

        return actions
```

---

## PPO训练流程

### 1. 数据收集（Rollout）

```python
def collect_rollout(env, policy, num_steps=2048):
    """
    收集PPO训练数据

    关键点：
    - 联合收集（importance + action）
    - 端到端记录梯度
    """

    rollouts = {
        'obs': [],
        'actions': [],
        'rewards': [],
        'values': [],
        'importance': [],
        'selection': [],
        'log_probs': []
    }

    obs = env.reset()

    for _ in range(num_steps):
        # 1. 前向传播
        with torch.set_grad_enabled(True):  # 记录计算图
            outputs = policy(obs)

        # 2. 执行动作
        actions = outputs['actions']
        obs_next, reward, done, info = env.step(actions)

        # 3. 存储rollout数据
        rollouts['obs'].append(obs)
        rollouts['actions'].append(actions)
        rollouts['rewards'].append(reward)
        rollouts['values'].append(outputs['value'])
        rollouts['importance'].append(outputs['importance'])
        rollouts['selection'].append(outputs['selection'])
        rollouts['log_probs'].append(outputs['log_prob'])

        obs = obs_next

    return rollouts
```

### 2. PPO更新

```python
def update_policy(policy, optimizer, rollouts, epoch=10):
    """
    PPO策略更新

    关键点：
    - 同时优化importance和action
    - 熵正则化鼓励探索
    - 梯度裁剪稳定训练
    """

    # 1. 计算GAE（Generalized Advantage Estimation）
    advantages = compute_gae(
        rewards=rollouts['rewards'],
        values=rollouts['values'],
        gamma=0.99,
        gae_lambda=0.95
    )

    # 2. 归一化advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # 3. 多次更新
    for _ in range(epoch):
        # 计算新的log_prob和value
        outputs = policy(rollouts['obs'])
        new_log_probs = outputs['log_probs']
        new_values = outputs['value']

        # PPO clip loss
        ratio = torch.exp(new_log_probs - rollouts['log_probs'])
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = F.mse_loss(new_values, rollouts['returns'])

        # 熵正则化（鼓励探索）
        entropy = outputs['dist'].entropy().mean()

        # 总loss
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

        # 梯度更新
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()
```

### 3. 奖励函数设计（针对评分公式）

```python
def compute_reward(env, prev_state, curr_state, action_dict):
    """
    计算奖励（对齐评分公式）

    设计原则：
    1. OCR提升奖励（70%权重）
    2. 稳定性奖励（30%权重）
    3. 干预成本惩罚（指数衰减）
    """

    # 1. OCR奖励
    ocr_prev = compute_ocr(prev_state)
    ocr_curr = compute_ocr(curr_state)
    ocr_reward = (ocr_curr - ocr_prev) * 100.0  # 归一化

    # 2. 稳定性奖励
    speed_std_curr = compute_speed_std(curr_state)
    accel_std_curr = compute_accel_std(curr_state)

    # 与baseline比较（假设有）
    speed_std_baseline = 5.0  # 从预先统计获取
    accel_std_baseline = 1.0

    i_speed = -(speed_std_curr - speed_std_baseline) / speed_std_baseline
    i_accel = -(accel_std_curr - accel_std_baseline) / accel_std_baseline

    stability_reward = 100.0 * (0.4 * i_speed + 0.6 * i_accel)

    # 3. 干预成本惩罚
    num_controlled = len(action_dict)  # 被控车辆数
    total_accel_cmd = sum(abs(a[0]) for a in action_dict.values())
    total_lane_change = sum(abs(a[1]) for a in action_dict.values())

    c_intervention = (
        1.0 * total_accel_cmd +
        5.0 * total_lane_change
    ) / (env.total_steps * 150.0)  # N_ICV=150（固定值）

    p_intervention = np.exp(-0.1 * c_intervention)

    # 总奖励
    reward = (
        0.7 * ocr_reward +
        0.3 * stability_reward
    ) * p_intervention

    return reward
```

---

## 两阶段训练策略（借鉴您的思路）

### Stage 1: 行为克隆（快速初始化）

```python
def stage1_behavior_cloning():
    """
    目标：从规则策略学习基础模式

    方法：
    1. 使用规则评分器生成importance标签
    2. 使用专家策略（IDM/MOBIL）生成action标签
    3. 监督学习训练网络
    """

    # 1. 收集数据
    rollouts = []
    for episode in range(20):
        obs = env.reset()
        for step in range(500):
            # 规则评分器计算importance
            rule_importance = rule_scorer.compute_importance(obs)

            # 专家策略（IDM）
            expert_actions = expert_policy(obs)

            # 存储数据
            rollouts.append({
                'obs': obs,
                'importance': rule_importance,
                'actions': expert_actions
            })

            obs, _, _, _ = env.step(expert_actions)

    # 2. 监督学习
    for epoch in range(10):
        for batch in dataloader(rollouts):
            # Importance预测
            pred_importance = policy.importance_head(batch['obs'])
            loss_importance = F.mse_loss(pred_importance, batch['importance'])

            # 动作预测
            pred_actions = policy.policy_head(batch['obs'])
            loss_actions = F.mse_loss(pred_actions, batch['actions'])

            # 总loss
            loss = loss_importance + loss_actions

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 保存预训练权重
    torch.save(policy.state_dict(), 'stage1_pretrained.pth')
```

### Stage 2: PPO微调（端到端优化）

```python
def stage2_ppo_finetune():
    """
    目标：直接优化OCR奖励

    关键点：
    1. 加载Stage 1预训练权重
    2. PPO端到端训练
    3. 冻结GNN编码器（可选）
    4. 只训练importance头和策略头
    """

    # 1. 加载预训练权重
    policy.load_state_dict(torch.load('stage1_pretrained.pth'))

    # 2. 可选：冻结GNN编码器
    for param in policy.gnn_encoder.parameters():
        param.requires_grad = False

    # 3. PPO训练
    for iteration in range(1000):
        # 收集rollout
        rollouts = collect_rollout(env, policy, num_steps=2048)

        # PPO更新
        update_policy(policy, optimizer, rollouts, epoch=10)

        # 日志
        if iteration % 10 == 0:
            eval_policy(policy, eval_env)
```

---

## 动态K值机制（自适应控制）

```python
class AdaptiveKSelector:
    """
    动态K值选择器

    逻辑：
    - 拥堵严重 → 多控制（K增大）
    - 系统平稳 → 少控制（K减小）
    """

    def compute_k_ratio(self, global_state):
        """
        根据全局状态计算K比例

        Args:
            global_state: {
                'avg_speed': float,
                'speed_std': float,
                'queue_length': float,
                'congestion_level': float  # [0, 1]
            }

        Returns:
            k_ratio: float ∈ [0.05, 0.25]
        """

        congestion = global_state['congestion_level']

        if congestion < 0.3:
            # 通畅：最少控制
            k_ratio = 0.05
        elif congestion < 0.6:
            # 轻度拥堵：适度控制
            k_ratio = 0.10
        elif congestion < 0.8:
            # 中度拥堵：较多控制
            k_ratio = 0.15
        else:
            # 严重拥堵：最多控制
            k_ratio = 0.25  # 官方上限

        return k_ratio
```

---

## 双层安全屏障

```python
class SafetyShield:
    """
    安全屏障层

    Level 1: 规则检查（轻量）
    Level 2: 紧急避险（重量级）
    """

    def __init__(self):
        self.ttc_threshold = 2.0  # TTC < 2s 触发
        self.max_decel = -4.5     # 最大减速度
        self.max_accel = 2.0      # 最大加速度

    def filter_action(self, action, vehicle_state):
        """
        过滤不安全的动作

        Args:
            action: [accel, lane_change]
            vehicle_state: {speed, position, ...}

        Returns:
            safe_action: 修正后的动作
        """

        accel, lane_change = action

        # Level 1: 物理限制检查
        accel = np.clip(accel, self.max_decel, self.max_accel)

        # Level 2: TTC检查
        ttc = compute_ttc(vehicle_state)
        if ttc < self.ttc_threshold:
            # 极度危险：强制制动
            accel = self.max_decel
            lane_change = 0.0  # 禁止换道

            # 给予负奖励
            safety_reward = -10.0
        else:
            safety_reward = 0.0

        safe_action = [accel, lane_change]
        return safe_action, safety_reward
```

---

## 训练完整流程

```python
def main():
    # 1. 初始化
    config = load_config('configs/joint_ppo.yaml')
    env = CompetitionSumoEnv(config)
    policy = JointICVPolicy(config)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

    # 2. Stage 1: 行为克隆
    print("Stage 1: Behavior Cloning")
    stage1_behavior_cloning()

    # 3. Stage 2: PPO微调
    print("Stage 2: PPO Fine-tuning")
    stage2_ppo_finetune()

    # 4. 评估
    print("Evaluation")
    eval_results = eval_policy(policy, env, num_episodes=10)
    print(f"OCR: {eval_results['ocr']:.2%}")
    print(f"Intervention Cost: {eval_results['cost']:.3f}")
    print(f"Final Score: {eval_results['score']:.2f}")

    # 5. 保存最终模型
    torch.save(policy.state_dict(), 'joint_icv_policy.pth')
```

---

## 配置文件示例

```yaml
# joint_ppo.yaml

model:
  gnn:
    node_dim: 9
    hidden_dim: 64
    num_layers: 3
    dropout: 0.1
    interaction_radius: 0.15

  importance_head:
    hidden_dim: 64
    output_dim: 1

  sparse_gate:
    initial_k_ratio: 0.10
    min_k: 5
    max_k: 150  # 官方上限
    gumbel_temperature: 1.0

  policy_head:
    action_dim: 2
    hidden_dim: 128

training:
  stage1:
    num_episodes: 20
    epochs: 10
    batch_size: 32
    learning_rate: 1e-3

  stage2:
    total_timesteps: 100000
    num_envs: 8
    learning_rate: 3e-4
    gamma: 0.99
    gae_lambda: 0.95
    clip_range: 0.2
    n_steps: 2048
    batch_size: 64
    n_epochs: 10

  optimizer:
    weight_decay: 1e-5

  safety:
    ttc_threshold: 2.0
    max_accel: 2.0
    max_decel: -4.5

rewards:
  w_efficiency: 0.7
  w_stability: 0.3
  k_penalty: 0.1
  alpha: 1.0  # 加速度权重
  beta: 5.0   # 换道权重
```

---

## 总结：为什么v5.0更优？

### 相比独立评分器的优势

1. **端到端优化**
   - 重要性评分 → 可学习参数
   - 梯度从奖励回传到选择过程
   - 自动发现"什么样的车辆重要"

2. **动态适应**
   - 不同场景 → 不同重要性模式
   - 拥堵时优先控制瓶颈车辆
   - 平稳时减少控制

3. **训练稳定**
   - Stage 1预训练提供良好初始化
   - PPO微调端到端优化
   - 避免冷启动问题

### 与您v4.0架构的融合

**保留v4.0的核心思想**：
- ✅ 风险敏感GNN（采纳）
- ✅ 层次化聚合（采纳）
- ✅ 双层安全屏障（采纳）
- ✅ 动态权重门控（采纳）
- ✅ 渐进式训练（采纳）

**v5.0的关键改进**：
- 🆕 端到端可学习的importance
- 🆕 Gumbel-Softmax可微分Top-K
- 🆕 联合PPO训练（importance + action）
- 🆕 动态K值机制
- 🆕 针对评分公式的奖励设计

这应该是最适合比赛的架构！
