# 模仿学习 + MPC 完整指南

## 目录

- [1. 架构概述](#1-架构概述)
- [2. 为什么选择模仿学习+MPC](#2-为什么选择模仿学习mpc)
- [3. MPC控制器详解](#3-mpc控制器详解)
- [4. 模仿学习训练](#4-模仿学习训练)
- [5. 完整工作流程](#5-完整工作流程)
- [6. 参数调优指南](#6-参数调优指南)
- [7. 故障排查](#7-故障排查)

---

## 1. 架构概述

### 1.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                      训练阶段（离线）                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌───────┐ │
│  │ MPC控制器 │───►│ 最优控制 │───►│ 演示数据 │───►│ 行为  │ │
│  │  (专家)   │    │  序列    │    │(episodes)│    │ 克隆  │ │
│  └──────────┘    └──────────┘    └──────────┘    └───────┘ │
│                                                     │        │
│                                              ┌──────┴─────┐ │
│                                              │ 策略网络    │ │
│                                              │ (学生)      │ │
│                                              └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      推理阶段（在线）                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   交通环境 ──► 观测状态 ──► 策略网络 ──► 控制动作 ──► 环境执行 │
│   (SUMO)        (321维)      (MLP)       (加速度, 换道)       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 核心组件

| 组件 | 文件 | 作用 |
|------|------|------|
| MPC控制器 | `src/mpc/core/mpc_controller.py` | 生成最优控制演示 |
| 交通流模型 | `src/mpc/core/traffic_model.py` | MPC预测模型 |
| 策略网络 | `src/models/simplified_icv_policy.py` | 模仿学习网络 |
| 车辆评分器 | `src/env/rule_based_scorer.py` | 选择控制的车辆 |
| 数据集 | `src/data/imitation_dataset.py` | 演示数据加载 |

---

## 2. 为什么选择模仿学习+MPC

### 2.1 PPO的失败经验

经过2周的PPO训练实验，我们发现：

| 问题 | PPO的表现 | 影响 |
|------|----------|------|
| 稀疏奖励 | 只有episode结束时才有OCR反馈 | 学习效率低 |
| 样本效率 | 需要百万级样本 | 训练时间长 |
| 训练不稳定 | 奖励波动大，难以收敛 | 性能不可预测 |
| 协调困难 | 多车协调难以学习 | 局部最优 |
| **最终OCR** | **54.39%** | **低于baseline 54.69%** |

### 2.2 模仿学习+MPC的优势

| 优势 | 说明 | 效果 |
|------|------|------|
| 数据高效 | 只需100个episodes | 训练时间短 |
| 收敛稳定 | MSE损失单调下降 | 可预测性强 |
| 优质演示 | MPC生成最优控制 | 性能上限高 |
| 可解释性 | MPC优化目标明确 | 易于调试 |
| **预期OCR** | **56-59%** | **比baseline高1.3-4.3%** |

---

## 3. MPC控制器详解

### 3.1 MPC核心思想

**模型预测控制（Model Predictive Control, MPC）**是一种基于优化的控制方法：

1. **预测**：使用交通流模型预测未来N步状态
2. **优化**：求解有限时域优化问题，得到最优控制序列
3. **执行**：执行第一步控制
4. **滚动**：重复上述过程

### 3.2 MPC优化问题

**优化目标**：
```python
min Σₖ [ Q_speed*(vₖ - v_desired)²      # 速度跟踪
        + Q_accel*aₖ²                    # 加速度平滑
        + Q_gap*max(gap_desired - gapₖ, 0)²  # 车间距保持
        + R*uₖ² ]                        # 控制成本
```

**约束条件**：
```python
# 加速度约束
min_accel ≤ uₖ ≤ max_accel  # -4.5 ≤ a ≤ 2.0 m/s²

# 速度约束
0 ≤ vₖ ≤ max_speed  # 0 ≤ v ≤ 30 m/s

# 车间距约束
gapₖ ≥ min_gap  # gap ≥ 2.0 m
```

### 3.3 MPC配置参数

| 参数 | 默认值 | 说明 | 调优建议 |
|------|--------|------|----------|
| `prediction_horizon` | 10 | 预测步数（1秒） | 增大可提高性能，但降低速度 |
| `control_horizon` | 5 | 控制步数（0.5秒） | 通常为prediction_horizon的一半 |
| `Q_speed` | 10.0 | 速度权重 | 增大可提高速度 |
| `Q_gap` | 50.0 | 车间距权重 | 关键参数，影响安全性 |
| `Q_accel` | 0.1 | 加速度平滑权重 | 增大可平滑控制 |
| `R_accel` | 0.5 | 加速度控制权重 | 增大可减少干预 |

### 3.4 MPC求解时间

- **单次求解**：~200ms（使用OSQP求解器）
- **增量优化**：每3步重新求解一次
- **实时性能**：满足0.1s/步要求

---

## 4. 模仿学习训练

### 4.1 行为克隆（Behavioral Cloning）

**核心思想**：从专家演示中学习，最小化预测动作与专家动作的差异。

**损失函数**：
```python
L = MSE(π(s), a_expert) = ||π(s) - a_expert||²
```

**训练流程**：
```python
for epoch in range(num_epochs):
    for batch in dataloader:
        # 前向传播
        pred_actions = policy(batch['obs'])

        # 计算MSE损失
        loss = mse_loss(pred_actions, batch['actions'])

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
```

### 4.2 策略网络架构

**输入**：321维观测
- 32车辆 × 9特征 = 288维
- 32维全局统计 = 32维
- 1维车辆数量 = 1维

**网络**：
```python
SimplifiedICVPolicy(
    obs_dim=321,
    hidden_dim=128,
    num_layers=3,
    num_vehicles=32
)
```

**架构**：
```
Input(321) → Linear(321→128) → ReLU →
Linear(128→128) → ReLU →
Linear(128→128) → ReLU →
Linear(128→64) → Tanh →
Output(64)
```

**输出**：64维动作
- 32车辆 × 2（加速度 + 换道概率）

### 4.3 训练配置

| 参数 | 默认值 | 说明 | 调优建议 |
|------|--------|------|----------|
| `num_epochs` | 50 | 训练轮数 | 数据多时可增大 |
| `batch_size` | 32 | 批次大小 | 显存不足时减小 |
| `lr` | 1e-3 | 学习率 | 通常无需调整 |
| `weight_decay` | 1e-5 | L2正则化 | 防止过拟合 |
| `grad_clip` | 1.0 | 梯度裁剪 | 防止梯度爆炸 |
| `val_split` | 0.2 | 验证集比例 | 通常20% |

---

## 5. 完整工作流程

### 步骤1：安装依赖

```bash
# 激活环境
conda activate sumo

# 安装MPC依赖
pip install cvxpy osqp

# 验证安装
python -c "import cvxpy; print(f'cvxpy version: {cvxpy.__version__}')"
python -c "import src.mpc.core.mpc_controller"
```

### 步骤2：收集MPC演示数据

```bash
python scripts/collect_mpc_demonstrations.py \
    --config configs/mpc.yaml \
    --num_episodes 50 \
    --num_workers 4 \
    --output_dir data/demonstrations/mpc
```

**预期时间**：8-10小时

**输出**：
- `data/demonstrations/mpc/demonstrations.pkl` - 演示数据
- `data/demonstrations/mpc/metadata.json` - 元数据

**验证数据质量**：
```bash
cat data/demonstrations/mpc/metadata.json
```

检查：
- `ocr_mean` > 0.57（比baseline高）
- `total_transitions` > 180,000

### 步骤3：训练模仿学习模型

```bash
python scripts/train_imitation_learning.py \
    --config configs/imitation_learning.yaml \
    --demo_data data/demonstrations/mpc/demonstrations.pkl \
    --output_dir checkpoints/imitation_learning \
    --log_dir logs/imitation_learning
```

**预期时间**：2-4小时

**训练监控**：
```bash
tensorboard --logdir logs/imitation_learning
```

**关键指标**：
- `Loss/train`：应单调下降
- `Loss/val`：应稳定收敛
- 早停：10轮无改善自动停止

### 步骤4：评估模型

```bash
python scripts/evaluate_imitation.py \
    --checkpoint checkpoints/imitation_learning/best.pth \
    --config configs/imitation_learning.yaml \
    --num_episodes 10 \
    --baseline_ocr 0.5469
```

**成功标准**：
- OCR > 56%（比baseline高1.3%）
- 比赛得分 > 2.0

### 步骤5：提交

修改 `submit_solution.py`，加载训练好的模型：

```python
from src.models.simplified_icv_policy import SimplifiedICVPolicy
import torch

# 加载模型
policy = SimplifiedICVPolicy(...)
checkpoint = torch.load('checkpoints/imitation_learning/best.pth')
policy.load_state_dict(checkpoint['policy_state_dict'])
policy.eval()

# 推理
with torch.no_grad():
    outputs = policy(obs_tensor, deterministic=True)
    actions = outputs['actions'][0].cpu().numpy()
```

---

## 6. 参数调优指南

### 6.1 MPC参数调优

**目标**：提高演示数据质量

| 问题 | 调优方法 | 参数调整 |
|------|----------|----------|
| OCR低 | 增加速度跟踪权重 | `Q_speed: 10.0 → 20.0` |
| 追尾风险高 | 增加车距权重 | `Q_gap: 50.0 → 100.0` |
| 控制不平滑 | 增加加速度权重 | `Q_accel: 0.1 → 0.5` |
| 求解慢 | 减小预测时域 | `prediction_horizon: 10 → 5` |

### 6.2 训练参数调优

**目标**：提高模型性能

| 问题 | 调优方法 | 参数调整 |
|------|----------|----------|
| 过拟合 | 增加正则化 | `weight_decay: 1e-5 → 1e-4` |
| 欠拟合 | 增加网络容量 | `hidden_dim: 128 → 256` |
| 训练慢 | 增大批次 | `batch_size: 32 → 64` |
| 显存不足 | 减小批次 | `batch_size: 32 → 16` |

### 6.3 数据质量调优

**目标**：过滤低质量演示

```python
# 在收集脚本中设置更严格的过滤
python scripts/collect_mpc_demonstrations.py \
    --min_ocr 0.56  # 只保留OCR > 56%的episode
```

---

## 7. 故障排查

### 7.1 MPC求解失败

**症状**：
```
[MPC Warning] Solve failed: Solver status: infeasible
```

**原因**：
- 约束过严导致优化问题不可行
- 车辆状态异常

**解决方法**：
1. 检查初始状态是否合理
2. 放宽约束（如减小`Q_gap`）
3. 使用fallback策略（已内置）

### 7.2 训练损失不下降

**症状**：
```
Epoch 1/50: Loss/train = 0.1234
Epoch 10/50: Loss/train = 0.1220  # 几乎不变
```

**原因**：
- 学习率过小
- 数据质量问题
- 网络容量不足

**解决方法**：
1. 增大学习率：`lr: 1e-3 → 5e-3`
2. 检查数据质量（可视化actions）
3. 增加网络层数或宽度

### 7.3 评估OCR低于预期

**症状**：
```
Model OCR: 0.5430 < Baseline: 0.5469
```

**原因**：
- 演示数据质量低
- 过拟合
- 车辆选择策略不当

**解决方法**：
1. 提高演示数据质量（使用MPC而非IDM）
2. 增加数据量（100 episodes）
3. 调整车辆评分权重

### 7.4 显存不足

**症状**：
```
RuntimeError: CUDA out of memory
```

**解决方法**：
```python
# 减小batch_size
batch_size: 32 → 16

# 减小hidden_dim
hidden_dim: 128 → 64

# 或使用CPU
--device cpu
```

---

## 8. 参考文献

### 模仿学习

1. **DemoLight** (CIKM 2019): "Learning Traffic Signal Control from Demonstrations"
   - 关键发现：模仿学习比DRL收敛快3-5倍

2. **Demonstration-guided RL** (2024): Hu et al.
   - 关键发现：在ramp metering上优于baseline 30%

### MPC + RL混合

3. **Airaldi et al. (2025)**: "Reinforcement Learning with Model Predictive Control for Highway Ramp Metering"
   - 关键发现：MPC+RL混合优于纯RL
   - 代码：https://github.com/FilippoAiraldi/mpcrl-for-ramp-metering

### MPC基础

4. **MPC Survey** (2019): Ye et al. "A survey of model predictive control methods for traffic signal control"
   - MPC在交通控制中的应用综述

---

**文档版本**: v1.0
**最后更新**: 2026-01-24
