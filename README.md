# 智能交通协同控制系统 - v6.0

基于**模仿学习 + MPC混合架构**的多智能体交通控制系统，专为初赛设计。

**最新更新**: v6.0 - 从PPO转向模仿学习 + MPC架构 🌟

---

## 📢 最新架构变更 (v6.0)

### 🔄 重大架构升级

经过两周的PPO训练实验（OCR=54.39%，低于baseline 54.69%），我们发现PPO与任务特点不匹配：
- ❌ 稀疏奖励问题：只有episode结束时才有OCR反馈
- ❌ 样本效率低：需要百万级样本
- ❌ 训练不稳定：奖励波动大
- ❌ 协调困难：多车协调困难

**新架构优势**：
- ✅ **模仿学习**：从专家演示学习，稳定高效
- ✅ **MPC专家**：基于模型预测控制，生成最优演示
- ✅ **数据高效**：只需100个episodes（vs 百万级）
- ✅ **收敛稳定**：MSE损失单调下降
- ✅ **可解释性强**：MPC优化目标明确

详见: [docs/IMITATION_LEARNING_MPC.md](docs/IMITATION_LEARNING_MPC.md)

---

## 🚀 快速开始

### 前置要求

- Python 3.10+
- CUDA 11.8+ (推荐)
- SUMO 1.19+
- LibSUMO (可选，用于加速)
- CVXPY 1.7+, OSQP (MPC依赖)

### 安装

```bash
git clone <repository-url>
cd TJ_transport_v4
conda create -n sumo python=3.10
conda activate sumo
pip install -r requirements.txt
pip install cvxpy osqp  # MPC依赖
```

### 三步快速启动

#### 步骤1：收集专家演示数据（使用MPC或IDM）

**使用MPC生成最优演示**（推荐，OCR更高）：
```bash
python scripts/collect_mpc_demonstrations.py \
    --config configs/mpc.yaml \
    --num_episodes 50 \
    --num_workers 4 \
    --output_dir data/demonstrations/mpc
```

**使用IDM生成演示**（快速baseline）：
```bash
python scripts/collect_expert_demonstrations.py \
    --config configs/imitation_learning.yaml \
    --num_episodes 100 \
    --num_workers 8 \
    --output_dir data/demonstrations/expert
```

#### 步骤2：训练模仿学习模型

```bash
python scripts/train_imitation_learning.py \
    --config configs/imitation_learning.yaml \
    --demo_data data/demonstrations/mpc/demonstrations.pkl \
    --output_dir checkpoints/imitation_learning \
    --log_dir logs/imitation_learning
```

训练时间：约2-4小时（50 epochs）

#### 步骤3：评估模型

```bash
python scripts/evaluate_imitation.py \
    --checkpoint checkpoints/imitation_learning/best.pth \
    --config configs/imitation_learning.yaml \
    --num_episodes 10 \
    --baseline_ocr 0.5469
```

---

## 📐 架构概述

### v6.0架构：模仿学习 + MPC

```
┌─────────────────────────────────────────────────────────────┐
│                     训练阶段（离线）                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   MPC控制器 ──► 最优控制序列 ──► 演示数据 ──► 行为克隆训练  │
│   (专家)         (专家演示)         (100 episodes)      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     推理阶段（在线）                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   交通环境 ──► 观测状态 ──► 策略网络 ──► 控制动作 ──► 环境执行 │
│   (SUMO)        (321维)      (MLP)       (加速度, 换道)       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 核心组件

#### 1. MPC控制器（专家）

**文件**: `src/mpc/core/mpc_controller.py`

**核心特性**：
- 基于CVXPY + OSQP的凸优化求解器
- 有限时域最优控制（预测时域N=10，控制时域M=5）
- 多车辆协同优化
- 瓶颈区域优化（s=1200-2200m）
- 安全约束（最小车距、速度限制、加速度限制）

**优化目标**：
```
min Σ [Q_speed*(v - v_desired)² + Q_accel*a² + Q_gap*gap_error² + R*u²]
```

#### 2. 策略网络（学生）

**文件**: `src/models/simplified_icv_policy.py`

**架构**：
- 输入：321维观测（32车辆 × 9特征 + 32维全局统计）
- 网络：4层MLP（321 → 128 → 128 → 128 → 64）
- 输出：64维动作（32车辆 × 2：加速度 + 换道概率）
- 激活：ReLU（隐藏层），Tanh（输出层）

**训练**：
- 损失：MSE between 预测动作 vs 专家动作
- 优化器：Adam (lr=1e-3)
- 学习率调度：ReduceLROnPlateau
- 早停：验证损失10轮无改善

#### 3. 车辆评分器

**文件**: `src/env/rule_based_scorer.py`

**评分因素**：
- 瓶颈区域位置（权重2.0）
- 上游区域位置（权重1.0）
- 速度（权重0.5）
- 车间距（权重1.0）

**策略**：选择Top-25辆ICV进行控制

---

## 📁 项目结构

```
TJ_transport_v4/
├── README.md                           # 项目说明（本文档）
│
├── scripts/                            # 执行脚本
│   ├── collect_mpc_demonstrations.py   # MPC专家演示收集 ⭐
│   ├── collect_expert_demonstrations.py # IDM专家演示收集
│   ├── train_imitation_learning.py     # 模仿学习训练 ⭐
│   └── evaluate_imitation.py           # 模型评估 ⭐
│
├── src/
│   ├── models/                         # 模型定义
│   │   └── simplified_icv_policy.py    # 简化的策略网络 ⭐
│   │
│   ├── mpc/                            # MPC控制器
│   │   ├── core/
│   │   │   ├── mpc_controller.py       # MPC控制器 ⭐
│   │   │   └── traffic_model.py        # 交通流模型 ⭐
│   │   └── utils/
│   │       └── mpc_config.py           # MPC配置
│   │
│   ├── data/                           # 数据处理
│   │   └── imitation_dataset.py        # 模仿学习数据集 ⭐
│   │
│   ├── env/                            # 环境接口
│   │   ├── competition_env.py          # 比赛环境
│   │   └── rule_based_scorer.py        # 规则评分器
│   │
│   └── training/                       # 训练模块（已废弃）
│       └── __init__.py                 # 保留用于向后兼容
│
├── configs/
│   ├── mpc.yaml                        # MPC配置 ⭐
│   └── imitation_learning.yaml         # 模仿学习配置 ⭐
│
├── data/
│   └── demonstrations/                 # 演示数据
│       ├── mpc/                        # MPC演示
│       └── expert/                     # IDM演示
│
├── checkpoints/                        # 模型检查点
│   └── imitation_learning/             # 模仿学习检查点
│
└── docs/                               # 文档
    ├── IMITATION_LEARNING_MPC.md       # 模仿学习+MPC指南 ⭐
    └── 交通工程赛道-评测公式.md        # 评测标准
```

---

## 📚 文档索引

### 核心文档 ⭐

| 文档 | 说明 |
|------|------|
| [README.md](README.md) | 项目说明（本文档） |
| [docs/IMITATION_LEARNING_MPC.md](docs/IMITATION_LEARNING_MPC.md) | 模仿学习+MPC完整指南 |
| [docs/交通工程赛道-评测公式.md](docs/交通工程赛道-评测公式.md) | 评测标准 |

### 历史文档（已归档）

以下文档描述了v5.0的PPO架构，现已废弃：

- [docs/联合PPO架构设计v5.md](docs/联合PPO架构设计v5.md) - PPO架构设计
- [v5_COMPLETE_SUMMARY.md](v5_COMPLETE_SUMMARY.md) - v5.0实施总结

---

## 📊 性能预期

### 模仿学习 + MPC架构

| 指标 | IDM演示 | MPC演示 |
|------|---------|---------|
| **演示OCR** | 56-57% | 57-58% |
| **训练后OCR** | 55-56% | 56-59% |
| **训练时长** | 2-4小时 | 2-4小时 |
| **演示收集** | 6-8小时 | 8-10小时 |
| **总时长** | 8-12小时 | 10-14小时 |

**对比baseline (54.69%)**：
- IDM演示：+0.3-1.3%
- MPC演示：+1.3-4.3%

---

## 🔧 配置说明

### MPC配置（configs/mpc.yaml）

```yaml
mpc:
  prediction_horizon: 10  # 预测时域（1秒）
  control_horizon: 5      # 控制时域（0.5秒）
  dt: 0.1                 # 时间步长

  # 状态权重
  state_weights:
    speed: 10.0           # 速度跟踪权重
    gap: 50.0             # 车间距保持权重（关键！）
    acceleration: 0.1     # 加速度平滑权重

  # 约束
  constraints:
    min_accel: -4.5       # 最大减速度 (m/s²)
    max_accel: 2.0        # 最大加速度 (m/s²)
    max_speed: 30.0       # 最大速度 (m/s)
    min_gap: 2.0          # 最小车距 (m)
```

### 模仿学习配置（configs/imitation_learning.yaml）

```yaml
training:
  num_epochs: 50
  lr: 1.0e-3
  batch_size: 32
  grad_clip: 1.0

  lr_scheduler:
    type: ReduceLROnPlateau
    patience: 5
    factor: 0.5

  early_stopping:
    patience: 10
    min_delta: 0.001
```

---

## 🐛 常见问题

### Q: 为什么从PPO转向模仿学习？

**A**: 经过2周实验，PPO存在问题：
- OCR=54.39%，低于baseline 54.69%
- 训练不稳定，奖励波动大
- 样本效率低，需要百万级样本

模仿学习+MPC优势：
- 收敛稳定，MSE损失单调下降
- 样本效率高，100个episodes即可
- MPC生成的演示更优（预期OCR 57-58%）

### Q: 演示收集需要多长时间？

**A**:
- MPC演示（50 episodes）：8-10小时（4并行）
- IDM演示（100 episodes）：6-8小时（8并行）

### Q: 训练需要多长时间？

**A**: 约2-4小时（50 epochs，单卡GPU）

### Q: 如何选择使用哪种演示？

**A**:
- **追求最高性能**：使用MPC演示（OCR 57-58%）
- **快速验证**：使用IDM演示（OCR 56-57%）

### Q: 显存不足怎么办？

**A**:
1. 减小 `batch_size`（默认32）
2. 减小 `hidden_dim`（默认128）

---

## 📈 技术亮点

### 核心创新

1. **模仿学习 + MPC混合架构**
   - MPC生成最优专家演示
   - 行为克隆高效学习
   - 结合两者优势

2. **MPC最优控制**
   - 凸优化问题（CVXPY + OSQP）
   - 多车辆协同优化
   - 瓶颈区域优化
   - 安全约束处理

3. **数据高效学习**
   - 只需100个演示episodes
   - 稳定的MSE损失
   - 快速收敛（2-4小时）

4. **规则基车辆评分**
   - 基于交通工程理论
   - 瓶颈区域优先
   - Top-K稀疏控制

---

## 📄 许可证

本项目仅用于学习和竞赛目的。

---

**祝训练顺利！** 🚀

**文档版本**: v6.0
**最后更新**: 2026-01-24
