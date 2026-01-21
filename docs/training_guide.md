# v5.0完整架构 - 三阶段训练指南

## 📚 训练流程概述

v5.0完整架构采用三阶段课程学习策略：

```
Stage 1: 世界观察者（World Observer）
    ↓  (2-3小时)
Stage 2: 引导探索（Guided Exploration）
    ↓  (3-4小时)
Stage 3: 约束优化（Constrained Optimization）
    ↓  (4-5小时)
最终策略: 完整v5.0架构
```

**总训练时长**: 9-12小时

---

## 🎯 Stage 1: 世界观察者训练

### 功能
训练WorldModel预测未来交通状态演化

### 使用方法

```bash
python scripts/train_stage1_world_observer.py \
    --config configs/v5_complete.yaml \
    --num_episodes 20 \
    --device cuda
```

### 参数说明
- `--config`: 配置文件路径（默认：configs/v5_complete.yaml）
- `--num_episodes`: 收集的episode数（默认：20）
- `--device`: 设备（默认：cuda）

### 输出
- `checkpoints/v5_complete/stage1_best.pth` - 最佳WorldModel权重
- `checkpoints/v5_complete/stage1_final.pth` - 最终WorldModel权重
- 日志文件

### 训练细节

**数据收集**:
- 使用IDM策略生成人工驾驶轨迹
- 收集20个episode，每个episode最多500步
- 总轨迹数：~10,000步

**WorldModel训练**:
- 监督学习
- 损失函数：Loss = MSE(流预测) + 0.5 * BCE(风险预测)
- 优化器：Adam (lr=3e-4)
- 学习率调度：CosineAnnealingLR
- 早停：patience=5

**预期结果**:
- 训练时长：2-3小时
- 最佳loss：<0.01
- WorldModel参数量：~223K

---

## 🚀 Stage 2: 引导探索训练

### 功能
PPO训练优化OCR奖励，使用Stage 1预训练的WorldModel

### 使用方法

```bash
python scripts/train_stage2_guided_exploration.py \
    --config configs/v5_complete.yaml \
    --resume checkpoints/v5_complete/stage1_best.pth \
    --device cuda
```

### 参数说明
- `--config`: 配置文件路径
- `--resume`: Stage 1检查点路径
- `--device`: 设备

### 输出
- `checkpoints/v5_complete/stage2_best.pth` - 最佳策略权重
- `checkpoints/v5_complete/stage2_iter_*.pth` - 定期检查点
- 日志文件

### 训练细节

**PPO配置**:
- 迭代数：1000
- 每次迭代步数：2048
- Batch size：64
- Minibatches：32
- Gamma：0.99
- GAE lambda：0.95
- Clip param：0.2

**优化器**:
- Adam (lr=3e-4)
- 线性学习率调度（10% → 100%）

**损失函数**:
- Policy loss: PPO clip loss
- Value loss: MSE
- Entropy bonus: 0.01

**预期结果**:
- 训练时长：3-4小时
- OCR提升至70-75%
- 干预成本：0.3-0.4

---

## ⚖️ Stage 3: 约束优化训练

### 功能
引入CostCritic和DynamicWeightGate，使用拉格朗日松弛优化

### 使用方法

```bash
python scripts/train_stage3_constrained.py \
    --config configs/v5_complete.yaml \
    --resume checkpoints/v5_complete/stage2_best.pth \
    --device cuda
```

### 参数说明
- `--config`: 配置文件路径
- `--resume`: Stage 2检查点路径
- `--device`: 设备

### 输出
- `checkpoints/v5_complete/stage3_best.pth` - 最佳约束策略权重
- `checkpoints/v5_complete/stage3_iter_*.pth` - 定期检查点
- 日志文件

### 训练细节

**约束优化**:
- 成本阈值：0.5
- 初始λ：0.1
- λ学习率：0.01
- λ自适应更新

**组件启用**:
- CostCritic：预测干预成本
- DynamicWeightGate：动态调整奖励权重

**损失函数**:
```
Loss = policy_loss + 0.5 * value_loss + 0.5 * cost_loss - 0.01 * entropy
```

**预期结果**:
- 训练时长：4-5小时
- OCR：72-75%
- 干预成本：0.3-0.4
- 最终得分：72-76分

---

## 📂 文件结构

```
scripts/
├── train_stage1_world_observer.py    # Stage 1 训练脚本
├── train_stage2_guided_exploration.py  # Stage 2 训练脚本
└── train_stage3_constrained.py        # Stage 3 训练脚本

checkpoints/v5_complete/
├── stage1_best.pth                   # Stage 1 最佳模型
├── stage1_final.pth                   # Stage 1 最终模型
├── stage2_best.pth                   # Stage 2 最佳模型
├── stage2_iter_*.pth                 # Stage 2 中间检查点
├── stage3_best.pth                   # Stage 3 最佳模型
├── stage3_iter_*.pth                 # Stage 3 中间检查点
└── stage3_final.pth                   # Stage 3 最终模型

logs/v5_complete/
├── stage1/                           # Stage 1 日志
├── stage2/                           # Stage 2 日志
└── stage3/                           # Stage 3 日志
```

---

## 🎓 完整训练流程示例

### Step 1: Stage 1训练

```bash
# 训练WorldModel
python scripts/train_stage1_world_observer.py \
    --config configs/v5_complete.yaml \
    --num_episodes 20 \
    --device cuda

# 预期输出：
# Epoch 1/20
#   Loss: 1.2345 | Flow: 0.9876 | Risk: 0.4938
#   ✅ 最佳模型已保存: checkpoints/v5_complete/stage1_best.pth
# ...
# 训练完成！
#   最佳损失: 0.0087
#   训练时长: 2.34 小时
```

### Step 2: Stage 2训练

```bash
# PPO训练
python scripts/train_stage2_guided_exploration.py \
    --config configs/v5_complete.yaml \
    --resume checkpoints/v5_complete/stage1_best.pth \
    --device cuda

# 预期输出：
# 迭代 1/1000
#   Policy Loss: -0.1234 | Value Loss: 0.5678 | Entropy: 0.2345
#   Mean Reward: 0.6789 | Mean Return: 12.3456
#   ✅ 最佳模型已保存: checkpoints/v5_complete/stage2_best.pth
# ...
# 训练完成！
#   训练时长: 3.45 小时
```

### Step 3: Stage 3训练

```bash
# 约束优化
python scripts/train_stage3_constrained.py \
    --config configs/v5_complete.yaml \
    --resume checkpoints/v5_complete/stage2_best.pth \
    --device cuda

# 预期输出：
# 迭代 1/1000
#   Policy Loss: -0.1456 | Value Loss: 0.5123 | Cost Loss: 0.1234
#   Mean Reward: 0.7012 | Mean Cost: 0.3456
#   Lambda: 0.1023 | Constraint Violation: 0.0000
#   ✅ 最佳模型已保存: checkpoints/v5_complete/stage3_best.pth
# ...
# 训练完成！
#   训练时长: 4.56 小时
```

---

## 📊 训练监控

### 关键指标

**Stage 1**:
- Loss（总体损失）
- Flow Loss（流预测MSE）
- Risk Loss（风险预测BCE）

**Stage 2**:
- Policy Loss（策略损失）
- Value Loss（价值损失）
- Entropy（策略熵）
- Mean Reward（平均奖励）
- Mean Return（平均回报）

**Stage 3**:
- Policy Loss（策略损失）
- Value Loss（价值损失）
- Cost Loss（成本损失）
- Mean Reward（平均奖励）
- Mean Cost（平均成本）
- Lambda（拉格朗日乘数）
- Constraint Violation（约束违反）

### TensorBoard（可选）

如需可视化训练过程，可在训练脚本中添加：

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir)
writer.add_scalar('Loss/train', loss, step)
writer.add_scalar('Reward/train', reward, step)
```

---

## 🔧 高级配置

### 调整训练参数

编辑 `configs/v5_complete.yaml`:

**Stage 1**:
```yaml
stage1_world_observer:
  training:
    num_episodes: 50  # 增加episode数
    batch_size: 64    # 调整batch size
```

**Stage 2**:
```yaml
stage2_guided_exploration:
  ppo:
    num_iterations: 2000  # 增加迭代数
    lr: 1.0e-4            # 降低学习率
```

**Stage 3**:
```yaml
stage3_constrained_optimization:
  lagrangian:
    cost_threshold: 0.3   # 降低成本阈值
    lambda_lr: 0.005       # 降低λ学习率
```

### 恢复训练

所有训练脚本都支持从检查点恢复：

```bash
# Stage 2恢复训练
python scripts/train_stage2_guided_exploration.py \
    --resume checkpoints/v5_complete/stage2_iter_500.pth

# Stage 3恢复训练
python scripts/train_stage3_constrained.py \
    --resume checkpoints/v5_complete/stage3_iter_800.pth
```

---

## ⚠️ 注意事项

### 1. 训练顺序

必须按顺序训练：Stage 1 → Stage 2 → Stage 3

### 2. 检查点依赖

- Stage 2依赖Stage 1的WorldModel权重
- Stage 3依赖Stage 2的策略权重

### 3. GPU内存

完整v5.0架构需要较大GPU内存：
- 最小：8GB VRAM
- 推荐：12GB+ VRAM

如果GPU内存不足，可调整：
- 减小`batch_size`
- 减小`num_minibatches`
- 禁用部分组件

### 4. 训练时间

预期训练时长：
- Stage 1: 2-3小时
- Stage 2: 3-4小时
- Stage 3: 4-5小时
- **总计**: 9-12小时

---

## 📈 预期性能

| 指标 | Stage 1 | Stage 2 | Stage 3 |
|------|---------|---------|---------|
| **OCR** | - | 70-75% | 72-75% |
| **干预成本** | - | 0.3-0.4 | 0.3-0.4 |
| **最终得分** | - | 70-74分 | 72-76分 |
| **训练时长** | 2-3h | 3-4h | 4-5h |

---

## 🎉 总结

三阶段训练脚本已全部实现完成！

**实施要点**：
1. ✅ 按顺序训练（Stage 1 → 2 → 3）
2. ✅ 每个阶段独立可运行
3. ✅ 支持从检查点恢复
4. ✅ 完整的监控和日志
5. ✅ 生产级代码质量

**下一步**：
运行三阶段训练，获得最终的v5.0完整策略模型！

---

**文档版本**: v1.0
**最后更新**: 2026-01-21
