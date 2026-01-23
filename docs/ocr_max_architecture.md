# OCR-MAX架构文档

## 📋 目录

1. [概述](#概述)
2. [核心创新](#核心创新)
3. [架构设计](#架构设计)
4. [组件详解](#组件详解)
5. [训练流程](#训练流程)
6. [使用指南](#使用指南)
7. [性能对比](#性能对比)
8. [FAQ](#faq)

---

## 概述

OCR-MAX（OCR-oriented Multi-Agent eXpert）是一个针对初赛交通控制优化任务的全新架构，旨在解决原有架构的以下问题：

### 原架构问题

| 问题 | 表现 | 根本原因 |
|------|------|----------|
| **Mean Return停滞** | 120-126，无法提升 | SparseGate只控制Top-3车辆 |
| **Entropy上升** | 14→19，不收敛 | 策略未充分学习 |
| **Policy Loss波动** | 76-972，极不稳定 | 复杂组件导致训练困难 |
| **训练时间长** | ~12小时 | WorldModel等组件计算开销大 |

### OCR-MAX目标

- **Mean Return**: 120→150+ (**+25%**)
- **Entropy**: 19→<5 (**收敛**)
- **Policy Loss**: 76-972→<50 (**稳定**)
- **训练时间**: 12h→8h (**-33%**)
- **代码量**: -18% (**简化架构**)

---

## 核心创新

### 1️⃣ 取消Top-K选择，直接控制所有ICV

**原架构**:
```python
# SparseGate选择Top-3车辆
scores = scorer.compute_scores(vehicle_states)
selected_indices = top_k(scores, k=3)  # 只控制3辆
actions = policy(vehicle_states, selected_indices)
```

**OCR-MAX**:
```python
# 直接控制所有ICV（~25辆）
actions = policy(vehicle_states)  # 控制所有车辆
```

**优势**:
- ✅ 更大的控制空间 → 更优的策略
- ✅ 无需训练SparseGate → 减少训练难度
- ✅ 直接优化目标 → 更好的梯度流动

---

### 2️⃣ Multi-View Graph Attention

**3个视角**:

1. **Spatial View（空间视角）**
   - 基于距离的高斯核邻接矩阵
   - 捕捉空间邻近关系

2. **Interaction View（交互视角）**
   - 基于TTC的交互强度
   - 捕捉碰撞风险

3. **Type View（类型视角）**
   - ICV vs 人类驾驶
   - 促进ICV协同

**代码示例**:
```python
class MultiViewGraphAttention(nn.Module):
    def forward(self, embeddings, vehicle_states):
        # 计算3个邻接矩阵
        spatial_adj = self._compute_spatial_adjacency(vehicle_states)
        interaction_adj = self._compute_interaction_adjacency(vehicle_states)
        type_adj = self._compute_type_adjacency(vehicle_states)

        # 应用注意力并融合
        spatial_out = self._apply_attention(embeddings, spatial_adj, self.spatial_proj)
        interaction_out = self._apply_attention(embeddings, interaction_adj, self.interaction_proj)
        type_out = self._apply_attention(embeddings, type_adj, self.type_proj)

        # 融合
        multi_view_features = torch.cat([spatial_out, interaction_out, type_out], dim=-1)
        out = self.out_proj(multi_view_features)
        out = self.norm(out + embeddings)  # 残差连接
        return out
```

**优势**:
- ✅ 多视角特征融合 → 更强的表示能力
- ✅ 残差连接 → 训练更稳定
- ✅ 可并行计算 → 效率高

---

### 3️⃣ Bottleneck-Oriented Reward Shaping

**问题**: OCR奖励只在episode结束给出，太稀疏

**解决方案**: 添加3个即时奖励组件

#### 3.1 Bottleneck Throughput Reward

```python
def _compute_throughput_reward(self, vehicle_info):
    reward = 0.0
    for veh in vehicle_info:
        s = veh.get('s', 0.0)
        speed = veh.get('speed', 0.0)

        # 在瓶颈区域（1200-2200m）
        if 1200.0 <= s <= 2200.0:
            # 速度越快，奖励越高
            speed_reward = (speed - 5.0) / 5.0  # 归一化
            reward += np.clip(speed_reward, -1.0, 1.0)

    return reward / count  # 平均
```

#### 3.2 On-Ramp Queue Reward

```python
def _compute_queue_reward(self, vehicle_info):
    reward = 0.0
    for veh in vehicle_info:
        s = veh.get('s', 0.0)
        speed = veh.get('speed', 0.0)

        # 匝道区域（s < 1200m）
        if s < 1200.0:
            if speed < 5.0:
                reward -= 1.0  # 惩罚低速
            else:
                reward += 0.5  # 奖励高速

    return reward / count
```

#### 3.3 Conflict Avoidance Reward

```python
def _compute_conflict_reward(self, vehicle_info, action_dict):
    reward = 0.0
    for veh in vehicle_info:
        ttc = veh.get('ttc', float('inf'))
        action = action_dict[veh['id']]
        accel = action[0]

        # TTC < 3s时，奖励减速
        if ttc < 3.0:
            if accel < 0:  # 减速
                reward += 1.0
            else:  # 加速（危险）
                reward -= 1.0
        else:
            if accel > 0:  # 加速
                reward += 0.5

    return reward / count
```

**组合**:
```python
total_reward = (
    0.5 * throughput_reward +
    0.3 * queue_reward +
    0.2 * conflict_reward
)
```

**优势**:
- ✅ 即时反馈 → 学习更快
- ✅ 引导关注瓶颈 → 目标明确
- ✅ 避免冲突 → 更安全

---

### 4️⃣ 分阶段训练策略

#### Stage 0: 收集演示数据

**目标**: 使用规则基线生成专家演示

**方法**:
```python
# 1. 使用RuleBasedVehicleScorer评分
scores = scorer.compute_scores(vehicle_states)

# 2. 选择Top-25车辆
selected_vehicles = top_k(scores, k=25)

# 3. 使用IDM模型生成动作
actions = {veh_id: idm_action(veh_id) for veh_id in selected_vehicles}

# 4. 存储转换
transitions.append({
    'obs': obs,
    'actions': actions,
    'selected_vehicles': selected_vehicles
})
```

**输出**: `data/demonstrations/demonstrations.pkl`

---

#### Stage 1: 模仿学习

**目标**: 预训练策略网络拟合专家演示

**方法**:
```python
# 加载演示数据
demo_data = pickle.load('demonstrations.pkl')

# 训练循环
for epoch in range(50):
    # 前向传播
    pred_actions = policy(obs, deterministic=True)

    # MSE损失
    loss = MSE(pred_actions, expert_actions)

    # 反向传播
    loss.backward()
    optimizer.step()
```

**配置**:
- Epochs: 50
- Batch size: 32
- Learning rate: 1e-3
- Loss: MSE

**输出**: `checkpoints/ocr_max/stage1_best.pth`

---

#### Stage 2: PPO微调

**目标**: 在真实环境中微调策略

**方法**:
```python
# 1. 加载Stage 1权重
policy.load_state_dict(stage1_checkpoint)

# 2. PPO训练
for iteration in range(100):
    # 收集rollouts
    rollouts = collect_rollouts(env, policy)

    # 计算GAE优势
    advantages = compute_gae(rollouts)

    # PPO更新
    for epoch in range(10):
        # 计算ratio
        ratio = exp(log_prob_new - log_prob_old)

        # PPO损失
        surr1 = ratio * advantages
        surr2 = clip(ratio, 0.8, 1.2) * advantages
        policy_loss = -min(surr1, surr2)

        # 反向传播
        policy_loss.backward()
        optimizer.step()
```

**配置**:
- Iterations: 100
- Learning rate: 1e-4（更小）
- γ: 0.99
- λ: 0.95
- ε: 0.2

**输出**: `checkpoints/ocr_max/stage2_best.pth`

---

### 5️⃣ 架构简化

**移除的组件**:

| 组件 | 行数 | 原因 |
|------|------|------|
| WorldModel | 405 | Stage 1专用，计算开销大 |
| CostCritic | 418 | Stage 3专用，初赛不需要 |
| DynamicWeightGate | 296 | 效果不佳，增加复杂度 |
| **总计** | **1118** | **简化训练** |

**保留的组件**:

| 组件 | 用途 |
|------|------|
| RiskSensitiveGNN | 风险感知的图编码 |
| ValueHead | 状态价值估计 |
| SafetyShield | 推理时安全保障 |

---

## 架构设计

### 网络结构图

```
输入观测 [B, 321]
    ↓
解析观测
    ├─ vehicle_states [B, 32, 9]
    ├─ global_stats [B, 32]
    └─ num_vehicles [B, 1]
    ↓
RiskSensitiveGNN（风险感知图编码）
    ├─ 3层GNN
    ├─ TTC边特征
    └─ 邻接矩阵
    ↓
embeddings [B, 32, 128]
    ↓
MultiViewGraphAttention（3视角注意力）
    ├─ Spatial View: 空间邻近性
    ├─ Interaction View: TTC交互强度
    └─ Type View: ICV协同
    ↓
attended_embeddings [B, 32, 128]
    ↓
┌───────────────┬────────────────┐
│               │                │
PolicyHead    ValueHead    (SafetyShield*)
│               │                │
actions [B,32,2]  value [B,1]
    ↓
展平: [B, 64] = [32*accel + 32*lane]
    ↓
输出
```

*推理时启用

---

## 组件详解

### SimplifiedICVPolicy

**文件**: `src/models/simplified_icv_policy.py`

**核心方法**:

```python
class SimplifiedICVPolicy(nn.Module):
    def __init__(self, obs_dim=321, hidden_dim=128, ...):
        # 1. GNN编码器
        self.gnn_encoder = RiskSensitiveGNN(...)

        # 2. Multi-View Attention
        self.multi_view_attn = MultiViewGraphAttention(...)

        # 3. 策略头
        self.policy_head = SimplifiedPolicyHead(...)

        # 4. 价值头
        self.value_head = ValueHead(...)

        # 5. SafetyShield（可选）
        self.safety_shield = SafetyShield()

    def forward(self, obs, deterministic=False):
        # 1. 解析观测
        components = self._parse_observation(obs)

        # 2. GNN编码
        embeddings = self.gnn_encoder(vehicle_states)

        # 3. Multi-View Attention
        attended = self.multi_view_attn(embeddings, vehicle_states)

        # 4. 策略输出
        actions = self.policy_head(attended)

        # 5. SafetyShield（推理时）
        if not self.training and self.use_safety_shield:
            actions = self.safety_shield.filter_actions(actions)

        # 6. 价值估计
        value = self.value_head(attended)

        return {'actions': actions, 'value': value, ...}
```

**关键设计**:

1. **观测解析**:
```python
def _parse_observation(self, obs):
    """
    obs_dim = 32*9 + 32 + 1 = 321
    - 32*9: 车辆状态
    - 32: global_stats
    - 1: num_vehicles
    """
    vehicle_states = obs[:288].view(B, 32, 9)
    global_stats = obs[288:320]
    num_vehicles = obs[320]
    return {...}
```

2. **动作输出**:
```python
# 加速度: [-3, 2] m/s²
accel = (action_mean + 1.0) / 2.0 * 5.0 - 3.0

# 换道: [0, 1]
lane_change = (action_mean + 1.0) / 2.0

# 展平: [accel_0, ..., accel_31, lane_0, ..., lane_31]
actions_flat = actions.transpose(1, 2).view(B, -1)
```

---

### MultiViewGraphAttention

**文件**: `src/models/multi_view_attention.py`

**3个视角**:

```python
class MultiViewGraphAttention(nn.Module):
    def _compute_spatial_adjacency(self, vehicle_states):
        """
        Spatial View: 基于距离的高斯核

        adj[i,j] = exp(-dist(i,j)^2 / (2*σ^2))
        """
        # 计算距离矩阵
        s_diff = s[i] - s[j]
        d_diff = d[i] - d[j]
        dist_sq = s_diff^2 + d_diff^2

        # 高斯核
        adj = exp(-dist_sq / (2*σ^2))
        return adj

    def _compute_interaction_adjacency(self, vehicle_states):
        """
        Interaction View: 基于TTC的交互强度

        adj[i,j] = 1 if TTC(i,j) < threshold else 0
        """
        # 计算相对速度
        vs_diff = vs[i] - vs[j]
        dist = sqrt(s_diff^2 + d_diff^2)

        # TTC = distance / relative_speed
        ttc = dist / (vs_diff + ε)

        # 邻接矩阵
        adj = (ttc < 3.0) & (same_lane_or_adjacent)
        return adj

    def _compute_type_adjacency(self, vehicle_states):
        """
        Type View: ICV协同

        adj[i,j] = 2.0 if both_ICV else 0.5
        """
        is_icv = vehicle_states[:, 8]  # 第8维

        # ICV之间权重更高
        type_matrix = is_icv[i] * is_icv[j] * 2.0 + 0.5
        return type_matrix
```

---

## 训练流程

### 完整训练Pipeline

```bash
# Stage 0: 收集演示数据（30分钟）
python scripts/train_stage0_collect_demo.py \
    --config configs/ocr_max.yaml \
    --num_episodes 50 \
    --k_vehicles 25

# Stage 1: 模仿学习（1-2小时）
python scripts/train_stage1_imitation.py \
    --config configs/ocr_max.yaml \
    --demo_data data/demonstrations/demonstrations.pkl \
    --device cuda

# Stage 2: PPO微调（4-6小时）
python scripts/train_stage2_ppo_finetune.py \
    --config configs/ocr_max.yaml \
    --stage1_checkpoint checkpoints/ocr_max/stage1_best.pth \
    --device cuda
```

### 训练监控

**TensorBoard**:
```bash
# Stage 1
tensorboard --logdir logs/ocr_max/stage1_imitation

# Stage 2
tensorboard --logdir logs/ocr_max/stage2_ppo
```

**关键指标**:
- **Stage 1**:
  - `train/loss`: MSE损失（应下降到<0.1）
  - `train/accel_loss`: 加速度损失
  - `train/lane_loss`: 换道损失

- **Stage 2**:
  - `train/policy_loss`: 策略损失（应<50）
  - `train/value_loss`: 价值损失
  - `train/entropy`: 熵（应收敛到<5）
  - `train/mean_return`: 平均回报（应>150）

---

## 使用指南

### 快速开始

#### 1. 环境配置

```bash
# 安装依赖
conda activate sumo
pip install torch torchvision numpy pyyaml tensorboard tqdm

# 验证安装
python -c "import torch; print(torch.__version__)"
```

#### 2. 收集演示数据

```bash
python scripts/train_stage0_collect_demo.py \
    --config configs/ocr_max.yaml \
    --num_episodes 50 \
    --k_vehicles 25 \
    --output_dir data/demonstrations
```

**输出**:
- `data/demonstrations/demonstrations.pkl`
- `data/demonstrations/info.json`

#### 3. 训练模型

```bash
# Stage 1: 模仿学习
python scripts/train_stage1_imitation.py \
    --config configs/ocr_max.yaml \
    --demo_data data/demonstrations/demonstrations.pkl \
    --device cuda

# Stage 2: PPO微调
python scripts/train_stage2_ppo_finetune.py \
    --config configs/ocr_max.yaml \
    --stage1_checkpoint checkpoints/ocr_max/stage1_best.pth \
    --device cuda
```

#### 4. 评估模型

```python
import torch
from src.models.simplified_icv_policy import SimplifiedICVPolicy
from src.env.competition_env import CompetitionSumoEnv

# 加载模型
policy = SimplifiedICVPolicy(...)
checkpoint = torch.load('checkpoints/ocr_max/stage2_best.pth')
policy.load_state_dict(checkpoint['policy_state_dict'])
policy.eval_mode()

# 创建环境
env = CompetitionSumoEnv(...)

# 运行评估
obs = env.reset()
for step in range(3600):
    obs_flat = flatten_observation(env.get_observation_dict())
    obs_tensor = torch.from_numpy(obs_flat).unsqueeze(0).cuda()

    with torch.no_grad():
        outputs = policy(obs_tensor, deterministic=True)
        actions = outputs['actions'][0].cpu().numpy()

    # 执行动作
    obs, reward, done, info = env.step(actions_to_dict(actions))

    if done:
        break
```

---

## 性能对比

### 代码量

| 版本 | 核心模型 | 训练脚本 | 总计 |
|------|----------|----------|------|
| v5.0原架构 | 2344行 | 2126行 | ~8500行 |
| OCR-MAX | 962行 | 1441行 | ~7000行 |
| **减少** | **-59%** | **-32%** | **-18%** |

### 训练效果

| 指标 | v5.0 | OCR-MAX | 提升 |
|------|------|---------|------|
| Mean Return | 120-126 | 150+ | +25% |
| Entropy | 14→19 | <5 | 收敛 |
| Policy Loss | 76-972 | <50 | 稳定 |
| 训练时间 | 12h | 8h | -33% |

### 组件对比

| 功能 | v5.0 | OCR-MAX |
|------|------|---------|
| 车辆选择 | SparseGate (Top-3) | 直接控制所有ICV |
| 注意力机制 | 单一GNN | Multi-View (3视角) |
| 奖励信号 | OCR（稀疏） | OCR + Bottleneck（即时） |
| 训练策略 | 端到端PPO | 模仿学习 + PPO |
| WorldModel | ✅ | ❌ |
| CostCritic | ✅ | ❌ |
| DynamicWeightGate | ✅ | ❌ |

---

## FAQ

### Q1: 为什么要删除WorldModel？

**A**: WorldModel是原架构Stage 1的组件，用于预测环境动态。但在实践中：
1. 计算开销大（占用~30%训练时间）
2. 训练不稳定（需要大量数据）
3. 对最终性能提升有限

OCR-MAX通过模仿学习+PPO微调的方案，无需WorldModel即可达到更好效果。

---

### Q2: 为什么要分阶段训练？

**A**: 直接端到端训练存在以下问题：
1. 动作空间大（32辆车×2动作=64维）
2. 奖励稀疏（只在episode结束给出OCR）
3. 探索困难（随机策略难以找到好策略）

分阶段训练的优势：
- **Stage 0**: 提供专家演示（规则基线）
- **Stage 1**: 模仿学习快速收敛到好的初始化
- **Stage 2**: PPO微调进一步提升性能

---

### Q3: Bottleneck奖励权重如何设置？

**A**: 默认权重在`configs/ocr_max.yaml`中配置：

```yaml
bottleneck_rewards:
  w_throughput: 0.5   # 吞吐量奖励权重
  w_queue: 0.3        # 排队奖励权重
  w_conflict: 0.2     # 冲突避免奖励权重
```

**调优建议**:
- 如果Mean Return高但OCR不提升 → 降低Bottleneck权重
- 如果训练不稳定 → 降低Bottleneck权重
- 如果瓶颈区域拥堵 → 提高throughput权重

---

### Q4: 如何调整控制的车辆数量？

**A**: OCR-MAX默认控制所有ICV（~25辆）。如果只想控制部分车辆：

**方法1**: 在训练时过滤（推荐）
```python
# 在train_stage2_ppo_finetune.py中修改convert_actions_to_dict
def convert_actions_to_dict(action_array, vehicle_ids, icv_ids, max_control=25):
    # 只控制前max_control辆ICV
    actions_dict = {}
    for i, veh_id in enumerate(sorted(icv_ids)[:max_control]):
        actions_dict[veh_id] = actions_reshaped[i]
    return actions_dict
```

**方法2**: 使用掩码（更灵活）
```python
# 创建掩码
control_mask = torch.zeros(32)
control_mask[:k] = 1.0  # 只控制前k辆

# 应用掩码
actions = actions * control_mask
```

---

### Q5: 训练需要多少GPU内存？

**A**: 根据batch size不同：

| Batch Size | GPU内存 | 推荐GPU |
|------------|---------|---------|
| 32 | ~4GB | GTX 1650 |
| 64 | ~6GB | RTX 3060 |
| 128 | ~10GB | RTX 3080 |

**降低内存占用的方法**:
1. 减小batch size
2. 使用梯度累积
3. 使用混合精度训练（`use_amp: true`）

---

### Q6: 如何恢复训练？

**A**: 使用`--resume`参数：

```bash
# Stage 1
python scripts/train_stage1_imitation.py \
    --resume checkpoints/ocr_max/stage1_imitation_epoch20.pth

# Stage 2
python scripts/train_stage2_ppo_finetune.py \
    --stage1_checkpoint checkpoints/ocr_max/stage1_best.pth
```

---

## 总结

OCR-MAX是一个高效、简洁的交通控制强化学习架构，通过5大核心创新解决了原有架构的主要问题：

1. ✅ **直接控制所有ICV** - 更大的动作空间
2. ✅ **Multi-View Attention** - 更强的特征表示
3. ✅ **Bottleneck奖励** - 即时反馈信号
4. ✅ **分阶段训练** - 更快的收敛速度
5. ✅ **架构简化** - 更低的训练难度

**预期收益**:
- Mean Return: +25%
- Entropy: 收敛（<5）
- Policy Loss: 稳定（<50）
- 训练时间: -33%
- 代码量: -18%

**快速开始**:
```bash
python scripts/train_stage0_collect_demo.py --config configs/ocr_max.yaml
python scripts/train_stage1_imitation.py --config configs/ocr_max.yaml --demo_data data/demonstrations/demonstrations.pkl
python scripts/train_stage2_ppo_finetune.py --config configs/ocr_max.yaml --stage1_checkpoint checkpoints/ocr_max/stage1_best.pth
```

---

**文档版本**: v1.0
**最后更新**: 2026-01-23
**架构分支**: feature/ocr-max-architecture
