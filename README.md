# 智能交通协同控制系统 - v5.0

基于世界模型和联合PPO架构的多智能体交通控制系统，专为初赛和复赛设计。

**最新更新**: v5.1 - 特征表示优化，RMSE改善4.3% 🌟

---

## 📢 最新更新 (v5.1)

### 🌟 特征改进成果

- ✅ **RMSE**: 1.06 → 1.0158 (**-4.3%**)
- ✅ **leader_speed_diff**: 提升 **78%** 🌟🌟🌟
- ✅ **风险预测 F1**: 0.60 → 0.6052
- ✅ **R² Score**: 稳定在 0.2249

**新增特征**:
- `leader_gap`: 前车距离
- `leader_speed_diff`: 前车速度差
- `road_type`: 道路类型编码
- `lane_position`: 车道内位置

详见: [CHANGELOG.md](CHANGELOG.md)

---

## 🚀 快速开始

### 前置要求

- Python 3.10+
- CUDA 11.8+ (推荐)
- SUMO 1.19+
- LibSUMO (可选，用于加速)

### 安装

```bash
git clone <repository-url>
cd TJ_transport_v4
conda create -n sumo python=3.10
conda activate sumo
pip install -r requirements.txt
```

### 三步快速启动

#### 初赛（轻量级架构）

```bash
# 使用v5轻量级架构快速训练
python scripts/train_stage2_guided_exploration.py \
    --config configs/phase1_lite.yaml \
    --device cuda
```

#### 复赛（完整架构）

```bash
# Stage 1: 世界观察者（2-3小时）
python scripts/train_stage1_world_observer.py \
    --config configs/v5_complete.yaml \
    --num_episodes 20 \
    --device cuda

# Stage 2: 引导探索（3-4小时）
python scripts/train_stage2_guided_exploration.py \
    --config configs/v5_complete.yaml \
    --resume checkpoints/v5_complete/stage1_best.pth \
    --device cuda

# Stage 3: 约束优化（4-5小时）
python scripts/train_stage3_constrained.py \
    --config configs/v5_complete.yaml \
    --resume checkpoints/v5_complete/stage2_best.pth \
    --device cuda
```

详细训练指南请参考：[docs/training_guide.md](docs/training_guide.md)

---

## 📐 架构概述

### v5.0架构体系

#### 轻量级架构（初赛专用）

**文件**: `src/models/v5_lightweight.py`

**特点**:
- ✅ 直接优化OCR（OD完成率）
- ✅ 稀疏控制：只控制5-10%的关键车辆
- ✅ 移除世界模型：简化架构，加速训练
- ✅ 快速训练：6-8小时即可完成

**适用场景**: 初赛（只控制车辆，不控制信号灯）

#### 完整架构（复赛准备）

**文件**: `src/models/joint_icv_policy.py`

**核心组件**:
1. **RiskSensitiveGNN** - TTC感知的风险感知图神经网络
2. **ImportancePredictor** - 端到端可学习的车辆重要性评分
3. **SparseGate** - Gumbel-Softmax可微分Top-K选择
4. **HierarchicalPooling** - 4层聚合（Vehicle→Lane→Section→Global）
5. **WorldModel** - RSSM世界模型，预测未来状态
6. **CostCritic** - 拉格朗日松弛约束优化
7. **DynamicWeightGate** - 元学习动态权重调整
8. **SafetyShield** - 双层安全屏障

**特点**:
- ✅ 端到端学习：ICV评分与策略联合训练
- ✅ 三阶段课程学习：World Observer → Guided Exploration → Constrained Optimization
- ✅ 完整约束优化：拉格朗日松弛 + 成本预测
- ✅ 生产级代码质量：全部测试通过

**适用场景**: 复赛（控制车辆+信号灯，需要约束优化）

---

## 🎯 训练指南

### 三阶段训练流程

#### Stage 1: 世界观察者（World Observer）

**目标**: 训练WorldModel预测未来交通状态演化

**方法**:
- 使用IDM策略生成人工驾驶轨迹
- 监督学习：MSE(流预测) + BCE(风险预测)

**时长**: 2-3小时

**输出**: `checkpoints/v5_complete/stage1_best.pth`

#### Stage 2: 引导探索（Guided Exploration）

**目标**: PPO训练优化OCR奖励

**方法**:
- 加载Stage 1预训练的WorldModel权重
- 标准PPO训练，固定奖励权重

**时长**: 3-4小时

**输出**: `checkpoints/v5_complete/stage2_best.pth`

**预期**: OCR 70-75%，干预成本 0.3-0.4

#### Stage 3: 约束优化（Constrained Optimization）

**目标**: 引入CostCritic和DynamicWeightGate，最小化干预成本

**方法**:
- 加载Stage 2训练好的策略权重
- 激活CostCritic和DynamicWeightGate
- 拉格朗日松弛优化

**时长**: 4-5小时

**输出**: `checkpoints/v5_complete/stage3_best.pth`

**预期**: OCR 72-75%，干预成本 0.3-0.4，最终得分 72-76分

**总训练时长**: 9-12小时

### 配置文件选择

| 配置文件 | 用途 | 训练时长 | 适用场景 |
|---------|------|---------|---------|
| `configs/v5_complete.yaml` | 完整三阶段训练 | 9-12小时 | 复赛，完整架构 |
| `configs/phase1_lite.yaml` | 轻量级训练 | 6-8小时 | 初赛，快速原型 |

详细配置说明请参考配置文件中的注释。

---

## 📁 项目结构

```
TJ_transport_v4/
├── README.md                           # 项目说明（本文档）
├── SIMPLIFICATION_FIXES_REPORT.md      # 简化修复报告
│
├── scripts/                             # 训练脚本
│   ├── train_stage1_world_observer.py  # Stage 1: 世界观察者
│   ├── train_stage2_guided_exploration.py  # Stage 2: 引导探索
│   └── train_stage3_constrained.py      # Stage 3: 约束优化
│
├── src/
│   ├── models/                          # 模型定义
│   │   ├── joint_icv_policy.py          # v5完整架构 ⭐
│   │   ├── v5_lightweight.py            # v5轻量级架构 ⭐
│   │   ├── world_model.py               # 世界模型
│   │   ├── cost_critic.py               # 成本评论家
│   │   ├── safety_shield.py             # 安全屏障
│   │   ├── dynamic_weight_gate.py       # 动态权重门控
│   │   └── icv_gnn_scorer.py            # ICV评分GNN
│   │
│   ├── env/                             # 环境定义
│   │   └── competition_env.py           # 比赛环境
│   │
│   └── training/                        # 训练模块
│       └── ocr_rewards.py               # OCR奖励计算
│
├── configs/
│   ├── v5_complete.yaml                 # 完整架构配置 ⭐
│   └── phase1_lite.yaml                 # 轻量级配置
│
├── docs/                                # 文档
│   ├── INDEX.md                         # 文档索引 ⭐
│   ├── training_guide.md                # 训练指南
│   ├── 联合PPO架构设计v5.md              # 架构设计
│   ├── 最终架构蓝图v5.0.md              # 架构蓝图
│   ├── v5_COMPLETE_SUMMARY.md           # 实施总结
│   ├── v5_implementation_complete.md    # 实施报告
│   ├── ICV评分系统使用指南.md            # 组件使用指南
│   ├── ICV评分系统架构演进总结.md        # 架构演进
│   ├── ICV评分系统实现总结.md            # 实现总结
│   ├── 交通工程赛道-评测公式.md          # 评测标准
│   └── 赛题.md                          # 赛题说明
│
├── checkpoints/                         # 模型检查点
│   └── v5_complete/                     # v5完整架构检查点
│       ├── stage1_best.pth
│       ├── stage1_final.pth
│       ├── stage2_best.pth
│       ├── stage2_iter_*.pth
│       └── stage3_best.pth
│
└── 仿真环境_初赛_1.0/                    # SUMO仿真环境
    └── 仿真环境-初赛/
        ├── sumo_train.sumocfg            # 训练配置
        └── net.xml                      # 路网文件
```

---

## 📚 文档索引

### 核心文档 ⭐

- 📖 **[训练与评估指南](docs/TRAINING_GUIDE.md)** - 三阶段训练详细教程
- 📐 **[系统架构分析](docs/ARCHITECTURE.md)** - 完整架构设计说明
- 📝 **[改进日志](CHANGELOG.md)** - 版本更新和优化记录

### 快速参考

| 文档 | 说明 |
|------|------|
| [README.md](README.md) | 项目说明（本文档） |
| [CHANGELOG.md](CHANGELOG.md) | 版本更新记录 |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 架构详细分析 |
| [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) | 训练与评估指南 |

### 其他文档

| 文档 | 说明 |
|------|------|
| [docs/INDEX.md](docs/INDEX.md) | 完整文档索引 |
| [docs/training_guide.md](docs/training_guide.md) | 原训练指南 |
| [v5_COMPLETE_SUMMARY.md](v5_COMPLETE_SUMMARY.md) | v5.0 实施总结 |
| [SIMPLIFICATION_FIXES_REPORT.md](SIMPLIFICATION_FIXES_REPORT.md) | 简化修复报告 |

### 历史文档（已归档）

以下文档已整合到新的核心文档中，保留供参考：

- [联合PPO架构设计v5.md](docs/联合PPO架构设计v5.md)
- [最终架构蓝图v5.0.md](docs/最终架构蓝图v5.0.md)
- [ICV评分系统使用指南.md](docs/ICV评分系统使用指南.md)
- [ICV评分系统架构演进总结.md](docs/ICV评分系统架构演进总结.md)
- [ICV评分系统实现总结.md](docs/ICV评分系统实现总结.md)
- [交通工程赛道-评测公式.md](docs/交通工程赛道-评测公式.md)
- [赛题.md](docs/赛题.md)

---

## 🔧 配置说明

### 环境配置

所有配置文件都使用YAML格式，主要配置项：

**环境参数**:
- `max_vehicles`: 最大在线车辆数（官方固定600）
- `inflow_rate`: 总流量（官方固定9900 veh/h）
- `max_steps`: 最大步数（3600秒 ÷ 0.1秒/步 = 36000步）
- `warmup_steps`: 预热步数（300步 = 30秒）

**训练参数**:
- `device`: 设备选择（cuda/cpu）
- `batch_size`: 批次大小
- `learning_rate`: 学习率
- `num_iterations`: 训练迭代数

详细配置请参考各配置文件中的注释。

---

## 📊 性能预期

### v5轻量级架构（初赛）

| 指标 | 预期值 |
|------|--------|
| **OCR** | 68-72% |
| **训练时长** | 6-8小时 |
| **推理速度** | <20ms |

### v5完整架构（复赛）

| 指标 | Stage 1 | Stage 2 | Stage 3 |
|------|---------|---------|---------|
| **OCR** | - | 70-75% | 72-75% |
| **干预成本** | - | 0.3-0.4 | 0.3-0.4 |
| **最终得分** | - | 70-74分 | 72-76分 |
| **训练时长** | 2-3h | 3-4h | 4-5h |

---

## 🐛 常见问题

### Q: 如何选择使用哪个架构？

**A**:
- **初赛**（只控制车辆）：使用 `v5_lightweight.py` + `configs/phase1_lite.yaml`
- **复赛**（车辆+信号灯）：使用 `joint_icv_policy.py` + `configs/v5_complete.yaml`

### Q: 训练需要多长时间？

**A**:
- 轻量级架构：6-8小时
- 完整架构三阶段：9-12小时（Stage 1: 2-3h, Stage 2: 3-4h, Stage 3: 4-5h）

### Q: 如何从检查点恢复训练？

**A**: 使用 `--resume` 参数：
```bash
python scripts/train_stage2_guided_exploration.py \
    --config configs/v5_complete.yaml \
    --resume checkpoints/v5_complete/stage1_best.pth
```

### Q: 显存不足怎么办？

**A**:
1. 减小 `batch_size`
2. 减小 `num_minibatches`
3. 使用 `phase1_lite.yaml`（轻量级架构）

### Q: 如何查看训练进度？

**A**: 训练脚本会实时打印：
- 迭代进度
- 损失值
- 平均奖励
- 学习率

---

## 📈 技术亮点

### 核心创新

1. **端到端可学习的重要性评分**
   - 联合训练ICV评分与策略生成
   - Gumbel-Softmax可微分Top-K选择
   - 动态K值机制

2. **风险感知图神经网络**
   - TTC（Time-To-Collision）感知的注意力机制
   - 异构图建模车辆交互

3. **世界模型**
   - RSSM（Recurrent State Space Model）
   - 双头预测：流演化 + 风险演化
   - 多步预测能力

4. **层次化聚合**
   - 4层聚合：Vehicle→Lane→Section→Global
   - 注意力机制池化
   - 保留空间结构信息

5. **约束优化**
   - 拉格朗日松弛
   - 成本评论家
   - 自适应λ更新

6. **双层安全屏障**
   - Level 1：物理限制裁剪
   - Level 2：TTC检查强制制动
   - CPU执行避免GPU传输开销

---

## 📄 许可证

本项目仅用于学习和竞赛目的。

---

**祝训练顺利！** 🚀

**文档版本**: v5.0
**最后更新**: 2026-01-21
