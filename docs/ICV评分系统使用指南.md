# ICV评分系统使用指南

## 概述

ICV评分系统是一个统一的车辆重要性评分框架，支持神经网络评分和规则评分两种方式。

### 核心特性

1. **双模式评分**：神经网络评分（默认）+ 规则评分（备选）
2. **自动回退机制**：神经网络失效时自动切换到规则评分
3. **GNN架构**：基于图神经网络捕捉车辆交互
4. **两阶段训练**：行为克隆（监督学习） + PPO微调（强化学习）

## 文件结构

```
src/env/
├── vehicle_scoring.py          # 统一评分接口（主入口）
├── rule_based_scorer.py        # 规则评分器（备选）
└── competition_env.py          # 比赛环境（使用评分接口）

src/models/
└── icv_gnn_scorer.py           # ICV评分GNN模型

scripts/
├── train_icv_scorer.py         # 训练脚本
└── test_icv_scorer.py          # 测试脚本

configs/
└── icv_scorer_train.yaml       # 训练配置
```

## 快速开始

### 1. 运行测试

```bash
python scripts/test_icv_scorer.py
```

这会运行一系列测试来验证系统功能：
- 规则评分器测试
- GNN模型测试
- ICV GNN评分器测试
- 统一评分器测试

### 2. 训练ICV评分器

```bash
# 完整训练流程（Stage 1 + Stage 2）
python scripts/train_icv_scorer.py --config configs/icv_scorer_train.yaml --stage all

# 只训练Stage 1（行为克隆）
python scripts/train_icv_scorer.py --config configs/icv_scorer_train.yaml --stage stage1

# 只训练Stage 2（PPO微调）
python scripts/train_icv_scorer.py --config configs/icv_scorer_train.yaml --stage stage2
```

### 3. 在环境中使用

```python
from src.env.vehicle_scoring import create_vehicle_scorer_from_config
from src.env.competition_env import CompetitionSumoEnv

# 加载配置
config = {
    'neural_icv_scoring': {
        'enabled': True,
        'checkpoint_path': 'checkpoints/icv_scorer/stage1_best.pth'
    },
    # ... 其他配置
}

# 创建环境（会自动初始化评分器）
env = CompetitionSumoEnv(config=config, device='cuda')

# 获取车辆评分
vehicle_states = env.get_vehicle_states()
context = {
    'traci_lib': env.traci,
    'all_vehicle_ids': list(vehicle_states.keys())
}

# 计算评分
scores = env.vehicle_scorer.compute_scores(vehicle_states, context)

# 获取Top-K车辆
top_k_vehicles = env.vehicle_scorer.get_top_k_vehicles(
    vehicle_states, context, k=10
)
```

## 评分接口说明

### UnifiedVehicleScorer

主要接口类，提供统一的评分功能。

#### 方法

**compute_scores(vehicle_states, context) -> Dict[str, float]**

计算所有车辆的评分。

参数：
- `vehicle_states`: 车辆状态字典 `{veh_id: {s, d, vs, vd, speed, ...}}`
- `context`: 上下文信息 `{traci_lib, all_vehicle_ids, ...}`

返回：
- `scores`: 评分字典 `{veh_id: score}`，评分范围 [0, 1]

---

**get_top_k_vehicles(vehicle_states, context, k, min_score=0.0) -> List[str]**

获取评分最高的K个车辆。

参数：
- `vehicle_states`: 车辆状态字典
- `context`: 上下文信息
- `k`: 返回的车辆数量
- `min_score`: 最低评分阈值

返回：
- `top_k_vehicles`: 按评分排序的车辆ID列表

---

**switch_to_neural(checkpoint_path=None)**

切换到神经网络评分模式。

---

**switch_to_rule()**

切换到规则评分模式。

---

**get_statistics() -> Dict[str, int]**

获取评分器使用统计。

返回：
```python
{
    'neural_calls': int,  # 神经网络评分次数
    'rule_calls': int,    # 规则评分次数
    'errors': int         # 错误次数
}
```

## 配置说明

### 神经网络配置

```yaml
neural_icv_scoring:
  enabled: true                     # 是否启用神经网络
  node_dim: 9                       # 车辆状态特征维度
  hidden_dim: 64                    # GNN隐藏层维度
  num_layers: 3                     # GNN层数
  dropout: 0.1                      # Dropout概率
  interaction_radius: 0.15          # 车辆交互半径
  checkpoint_path: null             # 预训练权重路径
  fallback_on_error: true           # 失败时回退到规则评分
```

### 训练配置

#### Stage 1: 行为克隆

```yaml
stage1_behavior_cloning:
  num_episodes: 20                  # 数据收集的episode数
  num_envs: 8                       # 并行环境数
  epochs: 10                        # 训练轮数
  batch_size: 32                    # 批大小
  learning_rate: 0.001              # 学习率
  early_stopping_patience: 5        # 早停patience
```

#### Stage 2: PPO微调

```yaml
stage2_ppo:
  total_timesteps: 100000           # 总训练步数
  learning_rate: 0.0003             # 学习率
  gamma: 0.99                       # 折扣因子
  clip_range: 0.2                   # PPO裁剪范围
  n_steps: 2048                     # 每次更新的步数
```

## 输入特征说明

车辆状态特征（9维）：

| 索引 | 特征 | 说明 | 范围 |
|------|------|------|------|
| 0 | s | 纵向位置（归一化） | [0, 1] |
| 1 | d | 横向偏移（归一化） | [0, 1] |
| 2 | vs | 纵向速度 | [-30, 30] m/s |
| 3 | vd | 横向速度 | [-10, 10] m/s |
| 4 | speed | 总速度 | [0, 30] m/s |
| 5 | acceleration | 加速度 | [-5, 3] m/s² |
| 6 | lane_index | 车道索引 | [0, 3] |
| 7 | angle | 角度 | [-π, π] |
| 8 | in_bottleneck | 是否在瓶颈区域 | {0, 1} |

## 规则评分器说明

规则评分器基于交通工程理论，综合考虑：

1. **瓶颈区域评分**（权重 40%）
   - 在瓶颈区域内车辆基础分高
   - 接近关键汇入点（J5, J15, J17）的车辆额外加分

2. **速度评分**（权重 25%）
   - 速度越慢越重要（反映拥堵）
   - 减速中的车辆更重要

3. **车道位置评分**（权重 15%）
   - 内侧车道优先（汇入影响大）

4. **距离评分**（权重 15%）
   - 与前车距离越近越关键
   - 使用安全距离判断

5. **TTC评分**（权重 5%）
   - TTC（Time To Collision）越小越重要
   - 反映碰撞风险

## 模型性能

### 预期训练时间

- **Stage 1**: 1.5-2小时（监督学习）
- **Stage 2**: 3-4小时（PPO微调）
- **总计**: 6-8小时

### 预期性能指标

- **OCR提升**: 65-72%
- **干预成本**: 0.4-0.5
- **最终得分**: 68-72分

### 模型大小

- **参数量**: ~50K
- **模型文件**: ~200KB
- **推理速度**: <10ms (32辆车)

## 常见问题

### Q1: 如何选择评分模式？

**A**: 默认使用神经网络评分，它会自动学习车辆重要性。如果神经网络不可用或失败，系统会自动回退到规则评分。

### Q2: 如何调整评分策略？

**A**: 可以通过配置文件调整：
- 神经网络：调整`neural_icv_scoring`配置
- 规则评分：调整`heuristic_selector`配置

### Q3: 如何训练自己的模型？

**A**:
1. 准备配置文件（参考`configs/icv_scorer_train.yaml`）
2. 运行训练脚本：`python scripts/train_icv_scorer.py`
3. 模型保存在`checkpoints/icv_scorer/`目录
4. 在配置中指定`checkpoint_path`使用训练好的模型

### Q4: 评分范围是什么？

**A**: 所有评分都在 [0, 1] 范围内：
- 0.0: 最不重要
- 1.0: 最重要

### Q5: 如何获取评分详细分解？

**A**: 使用`get_score_breakdown`方法（仅规则评分器支持）：

```python
breakdown = scorer.get_score_breakdown(
    vehicle_states, context, veh_id='veh_0'
)
# 返回: {'bottleneck': 0.8, 'speed': 0.7, ...}
```

## 调试和日志

### TensorBoard

训练过程中会自动生成TensorBoard日志：

```bash
tensorboard --logdir runs/icv_scorer
```

### 统计信息

获取评分器使用统计：

```python
stats = env.vehicle_scorer.get_statistics()
print(f"神经网络调用: {stats['neural_calls']}")
print(f"规则评分调用: {stats['rule_calls']}")
print(f"错误次数: {stats['errors']}")
```

## 更新日志

### v1.0.0 (2026-01-21)

- ✅ 实现统一评分接口
- ✅ 实现规则评分器
- ✅ 实现ICV GNN评分模型
- ✅ 实现两阶段训练脚本
- ✅ 添加测试脚本
- ✅ 通过所有集成测试

## 参考资源

- [比赛评分标准](../完整评测流程指南.md)
- [训练文档](../README.md)
- [GNN架构说明](../v5_lightweight.py)
