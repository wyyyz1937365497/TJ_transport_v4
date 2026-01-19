# 智能交通协同控制系统 - v4.0

基于世界模型的分层时空图智能体，专为智能交通控制竞赛设计。

## 🎯 核心架构

```
完整系统
├── 感知层: RiskSensitiveGNN（风险敏感图神经网络）
├── 预测层: MultiScaleRSSM（多尺度世界模型）
├── 决策层: EnhancedInfluenceBasedController（影响力驱动控制器）
└── 安全层: LagrangianOptimizer（约束优化）
```

### 关键特性

- ✅ **按需干预**: Top-K机制只在关键时刻控制关键车辆
- ✅ **前瞻预测**: 世界模型预测未来交通流演化（5秒-1分钟）
- ✅ **约束优化**: Cost Critic平衡效率与干预成本
- ✅ **GPU加速**: 完整PPO实现，数据全程GPU存储
- ✅ **课程学习**: 5级渐进式训练（基础→赛题场景）

---

## 🚀 快速开始

### 环境安装

```bash
# 创建conda环境
conda create -n sumo python=3.10
conda activate sumo

# 安装PyTorch（CUDA 11.8）
pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cu118

# 安装SUMO
conda install -c conda-forge sumo

# 安装依赖
pip install -r requirements.txt
```

### 完整训练（推荐）

```bash
# Phase 1 + Phase 2（所有级别）
python train_preliminary.py --config configs/competition_preliminary.yaml

# 跳过Phase 1（使用已有权重）
python train_preliminary.py --config configs/competition_preliminary.yaml --skip-phase1

# 从指定级别开始
python train_preliminary.py --config configs/competition_preliminary.yaml --skip-phase1 --start-level 3
```

### 分阶段训练

```bash
# Phase 1: 世界模型预训练
python train_phase1.py --config configs/competition_preliminary.yaml

# Phase 2: 单独训练某个级别（调试用）
python train_phase2.py --stage 1 --config configs/competition_preliminary.yaml
```

### 查看训练进度

```bash
# TensorBoard
tensorboard --logdir logs/preliminary --port 6006
# 访问 http://localhost:6006
```

---

## 📁 项目结构

```
TJ_transport_v4/
├── configs/                           # 配置文件
│   └── competition_preliminary.yaml  # 初赛配置
│
├── src/
│   ├── models/                       # 模型定义
│   │   ├── v4_architecture.py       # v4.0完整架构
│   │   └── ideal_policy_v4.py      # PPO策略网络
│   ├── training/                     # 训练器
│   │   ├── custom_ppo_trainer.py   # GPU优化PPO
│   │   ├── world_model_train_v4.py # 世界模型训练器
│   │   └── multi_gpu_utils.py      # 多GPU工具
│   └── env/                          # 环境封装
│       ├── vec_env.py               # 并行环境
│       ├── gym_wrapper.py           # Gymnasium包装
│       └── competition_env.py       # 比赛环境
│
├── docs/                             # 文档
│   ├── COMPLETE_TRAINING_GUIDE.md   # ⭐ 完整训练指南
│   └── architecture_v4_implementation.md  # 架构实现细节
│
├── train_preliminary.py              # ⭐ 主训练脚本（完整流程）
├── train_phase1.py                   # Phase 1训练
├── train_phase2.py                   # Phase 2单级别训练
│
├── checkpoints/                      # 模型检查点（自动创建）
│   └── competition/preliminary/
│       ├── phase1/                   # Phase 1权重
│       ├── level1/                   # Level 1-5权重
│       └── level5/
│
└── logs/                             # 训练日志（自动创建）
```

---

## 📚 详细文档

### 📖 [完整训练指南](docs/COMPLETE_TRAINING_GUIDE.md) ⭐

包含：
- 环境配置详解
- Phase 1/2训练流程
- 参数调优建议
- 常见问题解决
- 评估与部署

### 🏗️ [架构实现指南](docs/architecture_v4_implementation.md)

包含：
- 感知层设计（风险敏感GNN）
- 预测层设计（多尺度世界模型）
- 决策层设计（Top-K影响力驱动）
- 安全层设计（双层屏障）

---

## 🎮 训练阶段说明

### Phase 1: 世界模型预训练

**目标**: 学习交通流演化规律

**训练模块**:
- RiskSensitiveGNN（感知层）
- MultiScaleRSSM（预测层）
- EnhancedDynamicWeightGating（门控层）

**输出**: `checkpoints/preliminary/phase1/world_model_final.pth`

**训练时间**: 约2-4小时

### Phase 2: PPO强化学习

**目标**: 学习最优控制策略

**课程级别**:
```
Level 1: 基础场景（10辆车，800辆/h）
Level 2: 中等流量（15辆车，1200辆/h）
Level 3: 高流量（20辆车，1800辆/h）
Level 4: 极端场景（32辆车，2400辆/h）
Level 5: 赛题场景（32辆车，2000辆/h）
```

**输出**: `checkpoints/preliminary/level{1-5}/custom_ppo.zip`

**训练时间**: 约20-40小时（总计）

---

## ⚙️ 关键配置

### 推荐配置

```yaml
# configs/competition_preliminary.yaml

training:
  phase2:
    # 性能参数
    learning_rate: 0.0001      # 学习率（稳定）
    target_kl: 0.05           # KL散度阈值
    update_epochs: 10         # PPO更新轮数

    # 环境参数
    num_envs: 1               # 并行环境数
    n_steps: 2048             # Rollout步数
    batch_size: 256           # 批次大小

    # 控制参数
    top_k: 8                  # 控制车辆数（25%）
```

### 参数调优

| 参数 | 推荐值 | 范围 | 说明 |
|------|--------|------|------|
| `learning_rate` | 0.0001 | 0.00005-0.0003 | 越高越快但不稳定 |
| `target_kl` | 0.05 | 0.01-0.1 | 越低越保守 |
| `top_k` | 8 | 5-12 | 控制车辆数 |
| `update_epochs` | 10 | 5-15 | 越高收敛越慢 |

---

## 📊 性能指标

### 训练监控

**TensorBoard指标**:
- `episode_reward`: 总奖励
- `ep_rew_mean`: 平均奖励
- `ep_len_mean`: 平均episode长度
- `kl_div`: KL散度（应保持0.01-0.1）
- `policy_loss`: 策略损失
- `value_loss`: 价值损失

### 评估指标

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/preliminary/level5/custom_ppo.zip \
    --num-episodes 20
```

**关键指标**:
- `avg_speed`: 平均速度（>30 km/h）
- `throughput`: 吞吐量（>1800 veh/h）
- `intervention_rate`: 干预率（<30%）
- `episode_reward`: 总奖励

---

## 🐛 常见问题

### Q: KL散度爆炸（>100）？

**解决**: 降低学习率
```yaml
learning_rate: 0.0001   # 从0.0003降低
```

### Q: 内存溢出（CUDA OOM）？

**解决**: 减小batch或环境数
```yaml
batch_size: 128   # 从256降到128
num_envs: 1       # 从2降到1
```

### Q: 训练不收敛？

**解决**: 增加熵系数
```yaml
entropy_coef: 0.02   # 从0.01增加到0.02
```

### Q: SUMO频繁重启？

**解决**: 增加max_steps
```yaml
environment:
  max_steps: 3600   # 从1800增加到3600
```

更多问题参见：[完整训练指南 - 故障排除](docs/COMPLETE_TRAINING_GUIDE.md#常见问题)

---

## 🔧 高级功能

### 断点续训

```bash
# 从Level 3继续
python train_preliminary.py \
    --config configs/competition_preliminary.yaml \
    --skip-phase1 \
    --start-level 3
```

### 自定义课程

编辑 `configs/competition_preliminary.yaml` 中的 `curriculum_levels` 部分。

### 导出模型

```python
import torch

# 保存完整checkpoint
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config
}, 'final_model.pth')
```

---

## 📈 性能基准

| 指标 | 目标 | 当前实现 |
|------|------|---------|
| 平均速度 | >30 km/h | ~35 km/h |
| 吞吐量 | >1800 veh/h | ~2000 veh/h |
| 干预率 | <30% | ~20% |
| 训练时间 | <48小时 | ~40小时 |

---

## 📄 许可证

本项目仅用于学习和竞赛目的。

---

## 🙏 致谢

感谢所有贡献者的努力！

---

**祝训练顺利！** 🚀

有问题？查看 [完整训练指南](docs/COMPLETE_TRAINING_GUIDE.md) 或提交 Issue。
