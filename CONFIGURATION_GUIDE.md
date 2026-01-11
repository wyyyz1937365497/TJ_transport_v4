# 📖 初赛配置文件使用指南

## 概述

本指南说明如何使用初赛专用配置文件进行训练和测试。

---

## 📁 配置文件清单

| 配置文件 | 用途 | 训练时间 | 适用场景 |
|----------|------|----------|----------|
| **competition_preliminary.yaml** | 初赛专用优化配置 | 4-7小时 | 比赛训练、最佳性能 |
| **competition_quick_test.yaml** | 快速测试配置 | 5-10分钟 | 功能验证、代码调试 |

---

## 🚀 快速开始

### 步骤1: 快速验证（推荐）

首次使用前，建议先运行快速测试验证代码正确性：

```bash
# 运行快速测试（5-10分钟）
python train_unified.py --config configs/competition_quick_test.yaml

# 预期输出：
# - Phase 1 (World Model): 2 episodes × 3 epochs
# - Phase 2 (PPO): 1000 timesteps
# - Phase 3: 跳过
# - Phase 4: 跳过
# - 生成XLSX结果文件
```

**验证检查点**：
- ✅ 训练无错误
- ✅ 损失正常下降
- ✅ 生成XLSX结果文件
- ✅ GPU内存正常（<4GB）

### 步骤2: 完整训练（比赛用）

验证通过后，使用初赛专用配置进行完整训练：

```bash
# 完整4阶段训练（4-7小时）
python train_unified.py --config configs/competition_preliminary.yaml

# 可选：仅运行特定阶段
python train_unified.py --config configs/competition_preliminary.yaml --phase 2
```

**训练流程**：
```
Phase 1: World Model预训练 (50 episodes × 30 epochs)
    ↓
Phase 2: PPO策略训练 (200K timesteps)
    ↓
Phase 3: 端到端微调 (100K timesteps)
    ↓
Phase 4: 约束优化训练 (100K timesteps)
    ↓
最终模型: checkpoints/competition/final_model_competition.pth
```

---

## ⚙️ 配置详解

### 初赛专用配置 (competition_preliminary.yaml)

#### 关键优化

1. **Frenet坐标系统**
```yaml
competition:
  use_accurate_frenet: true  # 精度提升10倍
```

2. **预测-决策闭环**
```yaml
competition:
  enable_prediction_loop: true
  prediction_weights:
    risk_vehicle_boost: 1.3       # 风险车辆影响力+30%
    recommended_vehicle_boost: 1.2  # 推荐车辆+20%
```

3. **瓶颈区域优化**
```yaml
competition:
  bottleneck_optimization:
    enabled: true
    junctions: ["J14", "J15", "J17"]  # 3个关键瓶颈
    priority_multiplier: 1.5
```

4. **成本控制（严格比赛标准）**
```yaml
competition:
  intervention_cost:
    alpha: 1.0   # 加速度成本
    beta: 5.0    # 换道成本
```

#### 训练参数对比

| 参数 | 快速测试 | 初赛配置 | 说明 |
|------|----------|----------|------|
| **max_steps** | 600 | 3600 | 仿真时长 |
| **num_parallel_workers** | 2 | 8 | 并行度 |
| **Phase 1 episodes** | 2 | 50 | World Model训练 |
| **Phase 2 timesteps** | 1K | 200K | PPO训练 |
| **Phase 3 timesteps** | - | 100K | 端到端微调 |
| **Phase 4 timesteps** | - | 100K | 约束优化 |
| **batch_size** | 64 | 256 | 批次大小 |
| **top_k** | 3 | 5 | 控制车辆数 |
| **future_steps** | 3 | 5 | 预测步数 |

---

## 🎯 训练阶段说明

### Phase 1: World Model预训练

**目标**: 学习交通动态模式

**关键参数**:
```yaml
num_episodes: 50
epochs: 30
batch_size: 256
learning_rate: 1.0e-4
```

**输出**: `world_model_phase1.pth`

**验证**:
- 预测损失下降
- 未来速度预测准确

### Phase 2: PPO训练（主要阶段）

**目标**: 学习控制策略

**关键参数**:
```yaml
num_envs: 8
total_timesteps: 200000
learning_rate: 3.0e-4
gamma: 0.99
batch_size: 256
n_steps: 4096
```

**输出**: `ppo_phase2.pth`

**验证**:
- 奖励上升
- 价值函数收敛
- 控制动作合理

### Phase 3: 端到端微调

**目标**: 联合优化所有模块

**关键参数**:
```yaml
total_timesteps: 100000
learning_rate: 1.0e-5  # 小学习率
freeze_bn: true        # 冻结BatchNorm
```

**输出**: `e2e_phase3.pth`

**验证**:
- 性能进一步提升
- 稳定性保持

### Phase 4: 约束优化训练

**目标**: 满足干预成本约束

**关键参数**:
```yaml
total_timesteps: 100000
cost_limit: 0.1      # 成本上限
learning_rate: 1.0e-4
```

**输出**: `final_model_competition.pth`

**验证**:
- 成本在约束内
- 性能保持

---

## 📊 监控训练

### WandB可视化（初赛配置启用）

```bash
# 启用WandB后，训练会自动上传到云端
# 访问链接查看实时训练曲线
```

**关键指标**:
- **train/loss**: 训练损失
- **train/reward**: 平均奖励
- **train/value_loss**: 价值损失
- **train/policy_loss**: 策略损失
- **eval/ocr**: OD完成率
- **eval/avg_speed**: 平均速度
- **eval/cost**: 干预成本

### 本地日志

```bash
# 查看训练日志
tail -f logs/competition/*.log

# 查看checkpoint
ls checkpoints/competition/
```

---

## 🧪 测试与评估

### 运行评估

```bash
# 使用训练好的模型评估
python evaluate.py \
  --config configs/competition_preliminary.yaml \
  --model checkpoints/competition/final_model_competition.pth \
  --episodes 10
```

**评估输出**:
- `results/competition/evaluation_results.xlsx` - 比赛格式结果
- `results/competition/evaluation_metrics.json` - 详细指标

### 关键指标检查

| 指标 | 目标值 | 说明 |
|------|--------|------|
| **OD完成率** | >85% | 主要效率指标 |
| **平均速度** | >8 m/s | 流畅度指标 |
| **速度标准差** | <4 m/s | 稳定性指标 |
| **干预成本** | <0.1 | 成本约束 |

---

## 🔧 常见问题排查

### 问题1: GPU内存不足

**现象**: `CUDA out of memory`

**解决方案**:
```yaml
# 减小batch_size
model.gnn.batch_size: 128  # 从256减到128

# 减少并行环境
training.phase2.num_envs: 4  # 从8减到4
```

### 问题2: 训练不收敛

**现象**: 损失震荡或不下降

**解决方案**:
```yaml
# 降低学习率
training.phase2.learning_rate: 1.0e-4  # 从3e-4降到1e-4

# 增大batch size
training.phase2.batch_size: 512  # 从256增到512

# 增加梯度裁剪
training_tricks.max_grad_norm: 0.5  # 从1.0降到0.5
```

### 问题3: 训练速度慢

**现象**: 每个epoch耗时过长

**解决方案**:
```yaml
# 增加并行度
environment.num_parallel_workers: 16  # 从8增到16

# 启用混合精度
training.phase1.use_mixed_precision: true
training.phase2.use_mixed_precision: true

# 减少日志频率
logging.log_every_n_steps: 100  # 从10增到100
```

### 问题4: 成本超限

**现象**: 干预成本超过约束

**解决方案**:
```yaml
# 降低成本上限
training.phase4.cost_limit: 0.05  # 从0.1降到0.05

# 增加成本惩罚
competition.intervention_cost.cost_penalty_factor: 0.02  # 从0.01增到0.02

# 减少换道频率
model.controller.top_k: 3  # 从5减到3
```

---

## 📝 训练检查清单

### 快速测试前

- [ ] 检查SUMO环境路径正确
- [ ] 检查GPU驱动正常
- [ ] 检查磁盘空间充足（>10GB）
- [ ] 检查依赖包完整安装

### 快速测试后

- [ ] 训练无错误完成
- [ ] 损失正常下降
- [ ] 生成XLSX结果文件
- [ ] 日志无异常警告

### 完整训练前

- [ ] 快速测试通过
- [ ] 硬件资源满足要求
- [ ] 预留足够训练时间（4-7小时）
- [ ] 配置WandB（可选）

### 完整训练后

- [ ] 4个阶段全部完成
- [ ] 最终checkpoint存在
- [ ] 评估指标达标
- [ ] XLSX结果格式正确

---

## 🎓 最佳实践

### 1. 渐进式训练

首次训练建议分阶段进行：
```bash
# 先训练Phase 1+2
python train_unified.py --config configs/competition_preliminary.yaml --phase 2

# 检查效果后继续Phase 3+4
python train_unified.py --config configs/competition_preliminary.yaml --phase 3
```

### 2. 超参数调优

基于快速测试结果调整参数：
```bash
# 覆盖默认参数
python train_unified.py \
  --config configs/competition_preliminary.yaml \
  --timesteps 300000 \
  --lr 2.0e-4 \
  --batch-size 512
```

### 3. 模型集成

训练多个模型并集成：
```bash
# 训练3个模型（不同随机种子）
python train_unified.py --config configs/competition_preliminary.yaml --seed 42
python train_unified.py --config configs/competition_preliminary.yaml --seed 123
python train_unified.py --config configs/competition_preliminary.yaml --seed 456

# 集成推理
python ensemble_inference.py \
  --models checkpoints/seed42/final_model.pth \
          checkpoints/seed123/final_model.pth \
          checkpoints/seed456/final_model.pth
```

### 4. 持续监控

启用WandB远程监控：
```yaml
wandb:
  enabled: true
  project: tj-transport-competition
  run_name: experiment_v1
```

---

## 📞 技术支持

如遇问题，请检查：
1. 本指南的"常见问题排查"部分
2. 项目GitHub Issues
3. 代码注释和文档

---

生成时间：2025-01-11
配置版本：v1.0
适用比赛：智能交通协同控制竞赛（初赛）
