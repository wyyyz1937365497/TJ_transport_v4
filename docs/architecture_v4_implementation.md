# 架构v4.0完整实现指南

## 概述

本文档说明如何将"基于世界模型的分层时空图智能体架构v4.0"集成到现有的训练系统中。

## 架构组成

```
完整系统 = 感知层 + 预测层 + 决策层 + 安全层
```

### 1. 感知层：风险敏感异构图GNN

**文件位置：** `src/models/ideal_policy_v4.py:RiskSensitiveGNN`

**关键特性：**
- ✅ 异构图节点：主线车、匝道车、设施
- ✅ 多种边类型：物理连接、感知范围、交互关系
- ✅ 层次化聚合：局部 → 区域 → 全局
- ⚡ **新增**：风险感知注意力（TTC偏置）

**集成方式：**
```python
from src.models.v4_components import RiskAwareAttention

# 在RiskSensitiveGNN中使用风险感知注意力
risk_attn = RiskAwareAttention(
    embed_dim=64,
    num_heads=4,
    ttc_threshold=3.0
)
```

### 2. 预测层：多尺度世界模型

**文件位置：** `src/models/ideal_policy_v4.py:MultiScaleRSSM`

**关键特性：**
- ✅ 潜在状态空间模型（RSSM）
- ✅ LSTM隐藏状态记忆
- ⚡ **新增**：双头预测解耦（z_flow + z_risk）

**集成方式：**
```python
from src.models.v4_components import MultiScaleRSSMHead

# 在IdealTrafficPolicyV4中添加多尺度头
rssm_head = MultiScaleRSSMHead(
    latent_dim=64,
    flow_dim=32,
    risk_dim=32
)

# 前向传播
z_flow, z_risk, hidden = rssm_head(
    latent_state=z_t,
    hidden_state=hidden_state
)
```

### 3. 决策层：Top-K影响力驱动 + Cost Critic

**文件位置：** `src/models/v4_components.py`

**关键组件：**

#### 3.1 Top-K影响力选择器

```python
from src.models.v4_components import TopKInfluenceSelector

selector = TopKInfluenceSelector(
    max_vehicles=32,
    top_k_ratio=0.25,
    gnn_dim=256,
    prediction_dim=64
)

# 使用
selection_mask, scores, info = selector(
    gnn_embeddings=gnn_output,
    prediction_features=z_flow,
    num_vehicles=num_vehicles
)
```

**优势：**
- 🎯 "按需干预"：只在关键时刻控制关键车辆
- 📈 直接优化 `S_total = S_perf × P_int`
- ⚡ 减少计算量（只计算Top-K车辆）

#### 3.2 Cost Critic（约束优化）

```python
from src.models.v4_components import CostCritic

cost_critic = CostCritic(
    state_dim=256,
    action_dim=64
)

# 同时预测Q值和Cost
q_value, cost_value = cost_critic(state, action)

# 拉格朗日损失
loss, lagrangian = cost_critic.compute_lagrangian_loss(
    q_value, cost_value, advantage,
    cost_limit=1.0,
    lambda_param=0.1
)
```

**优势：**
- 🎯 显式建模干预成本
- 🛡️ 约束优化而非简单惩罚
- 📊 自动平衡效率与成本

### 4. 安全层：双层屏障

**文件位置：** `src/models/architecture_integration.py:DualSafetyShield`

**Level 1：规则卫士**
```python
# 基础动作边界检查
actions = shield._level1_guard(actions)
```

**Level 2：紧急避险**
```python
# TTC < 2.0s时触发紧急制动
actions, safety_info = shield.check_and_correct(
    actions=actions,
    observations={'ttc': ttc_tensor},
    vehicle_ids=vehicle_ids
)
```

## 训练流程集成

### Phase 1：世界模型预训练

**目标：** 训练感知+预测模块

```python
from src.models.architecture_integration import ArchitectureIntegrator

# 1. 创建集成器
integrator = ArchitectureIntegrator(config)

# 2. 构建Phase 1模型
model = integrator.build_phase1_model(device)

# 3. 训练配置
train_config = integrator.get_training_config('phase1')
print(train_config)
# {
#     'name': 'World Model Pretraining',
#     'objectives': ['predict_traffic_flow', 'predict_collision_risk'],
#     'trainable_modules': ['perception_layer', 'prediction_layer']
# }

# 4. 训练（使用WorldModelTrainer）
from src.training.world_model_train_v4 import WorldModelTrainer

trainer = WorldModelTrainer(config=config, device=device)
trainer.train()

# 5. 保存checkpoint
checkpoint_path = 'checkpoints/preliminary/phase1/world_model_final.pth'
trainer.save_checkpoint(checkpoint_path)
```

### Phase 2：PPO强化学习

**目标：** 训练决策模块（使用Phase 1权重）

```python
# 1. 构建Phase 2模型（自动加载Phase 1权重）
model, integrator = create_integrated_system(
    config=config,
    phase='phase2',
    phase1_checkpoint='checkpoints/preliminary/phase1/world_model_final.pth',
    device=device
)

# 2. 训练配置
train_config = integrator.get_training_config('phase2')
print(train_config)
# {
#     'name': 'PPO Reinforcement Learning',
#     'frozen_modules': ['perception_layer', 'prediction_layer'],
#     'trainable_modules': ['decision_layer', 'critic', 'lagrangian']
# }

# 3. 创建PPO训练器
from src.training.custom_ppo_trainer import CustomPPOTrainer

trainer = CustomPPOTrainer(
    policy=model,
    env=env,
    config=config['training']['phase2'],
    device=device
)

# 4. 训练
trainer.learn(total_timesteps=126000)
```

## 渐进式课程学习

**文件位置：** `src/models/architecture_integration.py:ProgressiveCurriculum`

```python
from src.models.architecture_integration import ProgressiveCurriculum

# 创建课程
curriculum = ProgressiveCurriculum(config)

# 执行课程
while not curriculum.is_complete():
    stage = curriculum.get_current_stage()
    print(f"当前阶段: {stage['name']}")

    # 训练当前阶段
    # ...

    # 进入下一阶段
    curriculum.advance_stage()
```

**课程阶段：**

1. **Stage 0: 数据收集**
   - 使用随机策略收集交通数据

2. **Stage 1: 世界模型预训练**
   - 监督学习交通流演化
   - 损失：MSE(轨迹预测)

3. **Stage 2: 风险预测**
   - 学习预测碰撞风险
   - 损失：Binary Crossentropy

4. **Stage 3: 屏蔽RL**
   - 在安全屏障保护下初步学习
   - 算法：PPO + Shield

5. **Stage 4: 约束优化**
   - 引入成本约束精细调优
   - 算法：PPO + Lagrangian

## 完整训练脚本示例

```python
"""
完整训练流程：Phase 1 → Phase 2
"""
import torch
from pathlib import Path

from src.models.architecture_integration import (
    create_integrated_system,
    ProgressiveCurriculum,
    ArchitectureIntegrator
)
from src.env.vec_env import create_parallel_envs
from src.training.world_model_train_v4 import WorldModelTrainer
from src.training.custom_ppo_trainer import CustomPPOTrainer


def main():
    # 1. 加载配置
    config = load_config('configs/competition_preliminary.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ========================================================================
    # Phase 1: 世界模型预训练
    # ========================================================================
    print("\n" + "="*80)
    print("[PHASE 1] 世界模型预训练")
    print("="*80 + "\n")

    # 创建Phase 1模型
    integrator = ArchitectureIntegrator(config)
    model = integrator.build_phase1_model(device)

    # 训练
    trainer = WorldModelTrainer(config=config, device=device)
    phase1_checkpoint = trainer.train()
    trainer.save_checkpoint(phase1_checkpoint)

    # ========================================================================
    # Phase 2: PPO强化学习（课程学习）
    # ========================================================================
    print("\n" + "="*80)
    print("[PHASE 2] PPO强化学习训练")
    print("="*80 + "\n")

    # 创建课程
    curriculum = ProgressiveCurriculum(config)

    # 执行课程
    while not curriculum.is_complete():
        stage = curriculum.get_current_stage()

        print(f"\n{'='*80}")
        print(f"[CURRICULUM] {stage['name']}")
        print(f"描述: {stage['description']}")
        print(f"{'='*80}\n")

        # 根据阶段训练
        if 'World Model' in stage['name']:
            # Stage 1-2: 继续训练世界模型
            # ...（略）
            pass

        elif 'Shielded RL' in stage['name']:
            # Stage 3: 在安全屏障下训练
            model, integrator = create_integrated_system(
                config=config,
                phase='phase2',
                phase1_checkpoint=phase1_checkpoint,
                device=device
            )

            trainer = CustomPPOTrainer(
                policy=model,
                env=env,
                config=config['training']['phase2'],
                device=device
            )

            # 启用安全屏障
            trainer.enable_shield = True
            trainer.learn(total_timesteps=stage['updates'])

        elif 'Constrained' in stage['name']:
            # Stage 4: 引入成本约束
            # 启用Cost Critic
            model.use_cost_critic = True
            trainer.learn(total_timesteps=stage['updates'])

        # 进入下一阶段
        curriculum.advance_stage()

    # ========================================================================
    # 保存最终模型
    # ========================================================================
    final_checkpoint = 'checkpoints/preliminary/final_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
    }, final_checkpoint)

    print(f"\n[SUCCESS] 训练完成！")
    print(f"最终模型: {final_checkpoint}")


if __name__ == '__main__':
    main()
```

## 评估与推理

### 1. 单独评估各模块

```python
# 评估感知层
perception_metrics = evaluate_perception_layer(model, test_data)

# 评估预测层
prediction_metrics = evaluate_prediction_layer(model, test_data)

# 评估决策层
decision_metrics = evaluate_decision_layer(model, env)
```

### 2. 端到端评估

```python
# 完整系统评估
from src.models.architecture_integration import create_integrated_system

model, integrator = create_integrated_system(
    config=config,
    phase='phase2',
    phase1_checkpoint=phase1_checkpoint,
    device=device
)

# 评估
metrics = evaluate_on_environment(
    model=model,
    env=eval_env,
    num_episodes=20
)

print(f"平均速度: {metrics['avg_speed']:.2f} km/h")
print(f"吞吐量: {metrics['throughput']:.2f} veh/h")
print(f"干预率: {metrics['intervention_rate']:.2%}")
```

### 3. 推理模式

```python
# 单步推理
model.eval()
with torch.no_grad():
    observation = env.reset()
    done = False

    while not done:
        # 提取特征
        features_dict = model.extract_features(observation)

        # 预测未来
        z_flow, z_risk = model.predict_future(features_dict)

        # 选择关键车辆（Top-K）
        selection_mask, scores, info = model.topk_selector(
            features_dict['node_embeddings'],
            z_flow,
            num_vehicles
        )

        # 生成动作
        actions, values, log_probs = model(observation)

        # 安全屏障检查
        actions, safety_info = model.safety_shield.check_and_correct(
            actions=actions,
            observations=observation,
            vehicle_ids=vehicle_ids
        )

        # 执行
        observation, reward, done, info = env.step(actions)
```

## 关键优化点

### 1. 针对 `S_total = S_perf × P_int` 的优化

**策略：** "按需干预"

```python
# 平峰期：几乎不控制（P_int ≈ 1）
if congestion_level < 0.3:
    k = max(1, int(0.05 * num_vehicles))  # 只控制5%

# 拥堵初期：精准干预（P_int 仍然较高）
elif congestion_level < 0.7:
    k = int(0.15 * num_vehicles)  # 控制15%

# 严重拥堵：多干预保效率（S_perf优先）
else:
    k = int(0.25 * num_vehicles)  # 控制25%
```

### 2. 训练稳定性优化

**策略：** 渐进式训练

```python
# Stage 1: 先让模型学会"看懂"交通流
train_perception_prediction_only()

# Stage 2: 在安全屏障保护下学习控制
train_with_safety_shield()

# Stage 3: 引入成本约束
train_with_cost_constraint()
```

### 3. 计算效率优化

**策略：** Top-K稀疏控制

```python
# 只对选中的Top-K车辆计算RL策略
selected_vehicles = torch.where(selection_mask)[0]

for veh_id in selected_vehicles:
    action[veh_id] = policy_network(observation[veh_id])

# 其他车辆使用简单跟驰模型（IDM）
other_vehicles = torch.where(~selection_mask)[0]
for veh_id in other_vehicles:
    action[veh_id] = idm_model(observation[veh_id])
```

## 常见问题

### Q1: 如何调整Top-K的比例？

```python
# 在配置文件中
model:
  controller:
    top_k_ratio: 0.25  # 默认25%
    # 可以改为自适应
    adaptive_top_k:
      enabled: true
      min_ratio: 0.05
      max_ratio: 0.30
```

### Q2: 如何监控成本约束？

```python
# 在训练循环中
for step in range(total_steps):
    q_value, cost_value = cost_critic(state, action)

    # 检查是否违反约束
    if cost_value.mean() > cost_limit:
        # 增加拉格朗日乘子
        lambda_param *= 1.1
    else:
        # 减小拉格朗日乘子
        lambda_param *= 0.99
```

### Q3: 如何评估安全屏障的效果？

```python
# 统计安全屏障激活次数
shield_stats = model.safety_shield.stats

print(f"Level 1激活次数: {shield_stats['level1_activations']}")
print(f"Level 2激活次数: {shield_stats['level2_activations']}")
print(f"总检查次数: {shield_stats['total_checks']}")

# 激活率越低说明策略越安全
activation_rate = shield_stats['level2_activations'] / shield_stats['total_checks']
print(f"紧急制动率: {activation_rate:.2%}")
```

## 总结

架构v4.0的核心优势：

1. ✅ **理论深度**：融合Constrained RL、Meta RL、World Model
2. ✅ **工程可行**：模块解耦，渐进式训练
3. ✅ **赛题契合**：直接优化评分公式的各个维度
4. ✅ **安全可靠**：双层屏障保证底线

通过合理使用这些组件，可以构建一个既有理论深度又能落地的智能交通协同控制系统。
