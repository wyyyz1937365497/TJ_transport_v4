# 初赛训练指南

本指南专为初赛设计，帮助你快速上手并训练出高性能模型。

## 📋 初赛核心要求

### 赛制特点
- ✅ **仅控制车辆**：不控制路侧设施（信号灯、VSL）
- ✅ **智能车比例**：25%车辆可控制（8辆/32辆）
- ✅ **评分重点**：效率（速度、吞吐量）为主要指标
- ⚠️  **次要参考**：稳定性、干预成本

### 评分公式
```
S_total = S_perf × P_int
```
- **S_perf**：性能得分（效率 + 稳定性）
- **P_int**：干预成本惩罚因子（0-1）

## 🚀 快速开始

### 1. 一键训练（推荐）

```bash
# 完整训练（Phase 1 + Phase 2，所有级别）
./run_preliminary.sh

# 跳过Phase 1（如果已有预训练模型）
./run_preliminary.sh --skip-phase1

# 从指定级别开始（例如Level 3）
./run_preliminary.sh --start-level 3
```

### 2. 分步训练

#### Step 1: Phase 1 - 世界模型预训练
```bash
python train_preliminary.py --config configs/competition_preliminary.yaml
```

**训练目标**：
- 学习车辆状态编码器（RiskSensitiveGNN）
- 学习交通流预测器（MultiScaleRSSM）
- 学习动态权重门控（场景识别）

**预计时间**：2-3小时

#### Step 2: Phase 2 - PPO课程学习
```bash
# 训练所有5个级别
python train_preliminary.py --config configs/competition_preliminary.yaml --skip-phase1

# 或从特定级别开始
python train_preliminary.py --config configs/competition_preliminary.yaml \
    --skip-phase1 \
    --start-level 3
```

**课程级别**：
1. **Level 1 - 基础场景**：10辆，800流量
2. **Level 2 - 中等流量**：15辆，1200流量
3. **Level 3 - 高流量场景**：20辆，1800流量
4. **Level 4 - 极端场景**：32辆，2400流量
5. **Level 5 - 赛题场景**：32辆，2000流量 ⭐

**预计时间**：8-12小时

## 📊 评估模型

### 基础评估
```bash
# 评估Level 5模型（20个episodes）
python evaluate_preliminary.py \
    --checkpoint checkpoints/competition/preliminary/level5/custom_ppo.zip

# 详细评估（50个episodes）
python evaluate_preliminary.py \
    --checkpoint checkpoints/competition/preliminary/level5/custom_ppo.zip \
    --episodes 50 \
    --verbose
```

### 核心指标
- ⭐ **平均速度**：> 15 m/s（优秀），> 12 m/s（良好）
- ⭐ **吞吐量**：车辆通过率
- ⚠️  **干预率**：< 30%（合理），< 50%（可接受）

## ⚙️ 配置文件说明

### 初赛专用配置：`configs/competition_preliminary.yaml`

**与完整配置的区别**：
1. **智能车比例**：25%（icv_ratio: 0.25）
2. **Top-K控制**：控制8辆车（top_k: 8）
3. **奖励权重**：侧重效率（speed_weight: 2.0, efficiency_weight: 3.0）
4. **简化功能**：关闭失败案例库、PER等高级功能

**关键参数**：
```yaml
model:
  controller:
    top_k: 8              # 控制Top-K车辆（约25%）
    alpha_init: 0.6       # GNN重要性权重
    beta_init: 0.4        # 预测影响力权重

training:
  phase2:
    total_timesteps: 1500000  # 总训练步数
    learning_rate: 3e-4
    batch_size: 256
```

## 📁 文件结构

```
TJ_transport_v4/
├── configs/
│   └── competition_preliminary.yaml    # ⭐ 初赛配置文件
├── train_preliminary.py                 # ⭐ 初赛训练脚本
├── run_preliminary.sh                   # ⭐ 一键训练脚本
├── evaluate_preliminary.py              # ⭐ 初赛评估脚本
├── checkpoints/
│   └── competition/
│       └── preliminary/
│           ├── phase1/
│           │   └── world_model_final.pth
│           ├── level1/
│           ├── level2/
│           ├── level3/
│           ├── level4/
│           └── level5/
│               └── custom_ppo.zip      # ⭐ 最终提交模型
└── logs/
    └── preliminary/
        ├── level1/
        ├── level2/
        ├── level3/
        ├── level4/
        └── level5/
```

## 🎯 训练策略建议

### 1. 快速迭代（开发阶段）
```bash
# 只训练Level 5（赛题场景）
python train_phase2.py --stage 5
```
**优点**：快速验证想法
**缺点**：可能收敛较慢

### 2. 完整课程学习（推荐）
```bash
./run_preliminary.sh
```
**优点**：稳定收敛，性能更好
**缺点**：训练时间较长

### 3. 超参数调优

**如果模型干预率过高（> 50%）**：
- 增加`intervention_cost_weight`（0.5 → 1.0）
- 降低`top_k`（8 → 5）

**如果模型平均速度过低（< 10 m/s）**：
- 增加`speed_weight`（2.0 → 3.0）
- 增加`efficiency_weight`（3.0 → 4.0）

**如果训练不稳定**：
- 降低`learning_rate`（3e-4 → 1e-4）
- 增加`batch_size`（256 → 512）

## 🔍 常见问题

### Q1: 训练中断了怎么办？
```bash
# 使用检查点继续训练
python train_preliminary.py \
    --config configs/competition_preliminary.yaml \
    --skip-phase1 \
    --start-level 3  # 从Level 3继续
```

### Q2: 显存不足（CUDA out of memory）？
修改配置文件：
```yaml
training:
  phase2:
    batch_size: 128  # 256 → 128
    num_envs: 1      # 确保是1
```

### Q3: 训练太慢怎么办？
- **减少episodes**：将`min_episodes_per_level`从20降到10
- **减少训练步数**：将Level 5的步数从996000降到500000
- **关闭日志**：设置`log_interval: 1000`

### Q4: 如何判断模型是否收敛？
查看TensorBoard：
```bash
tensorboard --logdir logs/preliminary/level5
```
关注：
- `episode_reward`：是否稳定上升
- `avg_speed`：是否 > 12 m/s
- `kl_div`：是否在目标值附近（0.01）

### Q5: 模型可以用于提交吗？
运行完整评估：
```bash
python evaluate_preliminary.py \
    --checkpoint checkpoints/competition/preliminary/level5/custom_ppo.zip \
    --episodes 50
```

**判断标准**：
- ✅ 平均速度 > 12 m/s
- ✅ 干预率 < 30%
- ✅ 奖励稳定（标准差小）

## 📈 性能优化技巧

### 1. Top-K自适应调整
启用自适应Top-K（自动调整控制车辆数）：
```yaml
preliminary_tweaks:
  adaptive_top_k:
    enabled: true
    min_k: 5
    max_k: 12
```

### 2. 早期停止
防止过拟合：
```yaml
preliminary_tweaks:
  early_stopping:
    enabled: true
    patience: 10
```

### 3. 学习率预热
已默认启用，改善训练稳定性。

## 🏆 提交前检查清单

- [ ] 运行50个episodes的完整评估
- [ ] 平均速度 > 12 m/s
- [ ] 干预率 < 30%
- [ ] 在多个随机种子下测试（至少3个）
- [ ] 检查模型文件完整性
- [ ] 准备模型说明文档

## 📞 获取帮助

### 查看日志
```bash
# 查看最新训练日志
tail -f logs/preliminary/level5/*.log
```

### 系统检查
```bash
./check_training_setup.sh
```

### TensorBoard监控
```bash
tensorboard --logdir logs/preliminary
```

## 🎉 总结

使用本指南，你应该能够：

1. ✅ 理解初赛的核心要求和评分标准
2. ✅ 使用一键脚本快速开始训练
3. ✅ 评估模型性能并调优
4. ✅ 准备提交的最终模型

**祝你初赛顺利！** 🚀
