# 智能交通协同控制系统 - 完整训练指南

本文档提供完整的4阶段训练流程说明和命令参考。

---

## 训练架构概览

本项目包含三个训练脚本：

| 脚本 | 用途 | 训练阶段 | 加速比 | 推荐场景 |
|------|------|----------|--------|----------|
| `train.py` | 完整4阶段训练 | Phase 1-4 | 基准 | 生产环境、完整训练 |
| `train_sb3.py` | 快速PPO训练 | Phase 2 | +30-50% | 快速实验、原型验证 |
| `train_sb3_phase3.py` | 端到端微调（SB3） | Phase 3 | +40-60% | ⭐ Phase 3加速训练 |

---

## 4阶段训练流程

### Phase 1: 世界模型预训练

**目标**：学习基础交通动力学

- 收集真实交通轨迹数据
- 使用LSTM学习状态转移函数
- 支持混合精度训练（FP16）
- GPU加速图构建

**输出**：`checkpoints/world_model_phase1.pth`

**SB3适用性**：不适用（监督学习任务）

---

### Phase 2: PPO训练

**目标**：学习安全控制策略

- 使用近端策略优化（PPO）
- 集成安全屏障（Safety Shield）
- 并行环境加速训练
- 支持多种并行配置（4/16/32环境）

**输出**：`checkpoints/ppo_phase2.pth`

**训练脚本**：
- `train.py` - 原生PyTorch实现
- `train_sb3.py` - ⭐ Stable-Baselines3加速（推荐）

**加速比**：30-50%

---

### Phase 3: 端到端微调 ⭐ 新增SB3版本

**目标**：联合优化所有组件

- 解冻GNN、世界模型、控制器
- 使用较小学习率进行微调
- CosineAnnealing学习率调度
- 冻结BatchNorm层保持稳定性

**输出**：`checkpoints/e2e_phase3.pth`

**训练脚本**：
- `train.py` - 原生PyTorch实现
- `train_sb3_phase3.py` - ⭐⭐ Stable-Baselines3加速（强烈推荐）

**加速比**：40-60%

**架构集成**：
```
完整策略网络 (GNN + World Model + Controller)
         ↓
    SB3 PPO算法
         ↓
   端到端梯度更新
```

---

### Phase 4: 约束优化训练

**目标**：平衡性能与成本

- 使用拉格朗日乘子法
- 动态调整约束强度
- 自动平衡干预成本与性能

**输出**：`checkpoints/final_model.pth`

**SB3适用性**：部分支持（需要自定义回调，加速有限 20-30%）

**推荐**：使用原生 `train.py`

---

## 配置文件说明

### YAML配置文件（推荐）

| 配置文件 | 适用场景 | 并行环境 | Batch Size |
|----------|----------|----------|------------|
| `configs/windows_base.yaml` | 标准PC（4-8核） | 4 | 128 |
| `configs/windows_high_performance.yaml` | 高性能PC（8+核） | 16 | 256 |
| `configs/windows_extreme_performance.yaml` | 高端PC（16+核，大显存） | 32 | 512 |

### JSON配置文件

- `configs/training_config.json` - 基础JSON配置

---

## 完整训练命令

### 使用 train.py（完整4阶段训练）

```bash
# 完整4阶段训练（标准配置）
python train.py --config configs/windows_base.yaml --phase all

# 完整4阶段训练（高性能配置）
python train.py --config configs/windows_high_performance.yaml --phase all

# 完整4阶段训练（极限性能配置）
python train.py --config configs/windows_extreme_performance.yaml --phase all
```

### 单阶段训练

```bash
# 仅Phase 1: 世界模型预训练
python train.py --config configs/windows_base.yaml --phase 1

# 仅Phase 2: PPO训练
python train.py --config configs/windows_base.yaml --phase 2

# 仅Phase 3: 端到端微调
python train.py --config configs/windows_base.yaml --phase 3

# 仅Phase 4: 约束优化
python train.py --config configs/windows_base.yaml --phase 4
```

### 跳过数据收集

如果已有收集的数据，可以跳过SUMO数据收集：

```bash
python train.py --config configs/windows_base.yaml --phase 1 --skip-data-collection
```

### 仅评估模式

```bash
# 评估训练好的模型
python train.py --config configs/windows_base.yaml --eval-only

# 评估并生成XLSX结果
python train.py --config configs/windows_base.yaml --eval-only --generate-xlsx
```

---

## 使用 train_sb3.py（快速PPO训练 - Phase 2）

此脚本专注于Phase 2的PPO训练，使用Stable-Baselines3库。

### 基础用法

```bash
# 使用默认配置
python train_sb3.py

# 使用指定配置文件
python train_sb3.py --config configs/windows_base.yaml
```

### 自定义参数

```bash
# 指定并行环境数和训练步数
python train_sb3.py --envs 8 --timesteps 50000

# 高性能配置（16个并行环境）
python train_sb3.py --config configs/windows_high_performance.yaml

# 极限性能配置（32个并行环境）
python train_sb3.py --config configs/windows_extreme_performance.yaml

# 自定义PPO参数
python train_sb3.py --envs 4 --batch-size 256 --n-steps 4096 --learning-rate 0.0001
```

### 指定阶段

```bash
# 只运行Phase 2（默认）
python train_sb3.py --phase phase2

# 运行所有阶段（会提示其他阶段需要用train.py）
python train_sb3.py --phase all
```

---

## 使用 train_sb3_phase3.py（端到端微调 - Phase 3）⭐ 新增

此脚本专门用于Phase 3的端到端微调，使用Stable-Baselines3加速。
**相比原生实现，可提供40-60%的性能提升。**

### 基础用法

```bash
# 使用默认配置（会自动加载Phase 2权重）
python train_sb3_phase3.py

# 使用指定配置文件
python train_sb3_phase3.py --config configs/windows_base.yaml
```

### 指定Phase 2检查点

```bash
# 从特定检查点加载Phase 2权重
python train_sb3_phase3.py --phase2-checkpoint checkpoints/ppo_phase2.pth

# 使用train_sb3.py输出的检查点
python train_sb3_phase3.py --phase2-checkpoint checkpoints_sb3/ppo_sumo_final.zip
```

### 高性能训练

```bash
# 标准配置（4个并行环境）
python train_sb3_phase3.py --config configs/windows_base.yaml

# 高性能配置（16个并行环境）
python train_sb3_phase3.py --config configs/windows_high_performance.yaml --envs 16

# 极限性能配置（32个并行环境）
python train_sb3_phase3.py --config configs/windows_extreme_performance.yaml --envs 32
```

### 自定义训练参数

```bash
# 指定训练步数
python train_sb3_phase3.py --timesteps 100000

# 自定义学习率（端到端微调建议使用较小学习率）
python train_sb3_phase3.py --learning-rate 0.00001

# 调整批次大小
python train_sb3_phase3.py --batch-size 256

# 不冻结BatchNorm层
python train_sb3_phase3.py --no-freeze-bn
```

### 完整示例

```bash
# 高性能端到端微调
python train_sb3_phase3.py \
    --config configs/windows_high_performance.yaml \
    --envs 16 \
    --timesteps 100000 \
    --learning-rate 0.00001 \
    --batch-size 256
```

### 查看训练进度

```bash
# 启动TensorBoard
tensorboard --logdir=./logs_sb3_phase3/tensorboard/
```

---

## 训练参数说明

### Phase 1 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `num_episodes` | 训练episodes数量 | 20 |
| `epochs` | 训练轮数 | 20 |
| `batch_size` | 批次大小 | 128 |
| `learning_rate` | 学习率 | 1e-4 |
| `num_parallel_workers` | 并行工作进程数 | 4 |
| `use_mixed_precision` | 是否使用混合精度 | false |
| `gradient_clip` | 梯度裁剪阈值 | 1.0 |
| `warmup_epochs` | 学习率warmup轮数 | 5 |

### Phase 2 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `num_envs` | 并行环境数量 | 4 |
| `total_timesteps` | 总训练步数 | 100000 |
| `learning_rate` | 学习率 | 3e-4 |
| `gamma` | 折扣因子 | 0.99 |
| `gae_lambda` | GAE参数 | 0.95 |
| `clip_epsilon` | PPO裁剪参数 | 0.2 |
| `entropy_coef` | 熵系数 | 0.01 |
| `value_loss_coef` | 价值损失系数 | 0.5 |
| `update_epochs` | PPO更新轮数 | 10 |
| `batch_size` | PPO批次大小 | 128 |
| `n_steps` | Rollout步数 | 2048 |

### Phase 3 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `total_timesteps` | 总训练步数 | 50000 |
| `learning_rate` | 学习率 | 1e-5 |
| `freeze_bn` | 是否冻结BatchNorm | true |

### Phase 4 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `total_timesteps` | 总训练步数 | 50000 |
| `cost_limit` | 成本上限 | 0.1 |
| `learning_rate` | 学习率 | 1e-4 |

---

## 训练输出

### 模型检查点

训练过程中会保存以下检查点：

```
checkpoints/
├── world_model_phase1.pth    # Phase 1输出
├── ppo_phase2.pth            # Phase 2输出
├── e2e_phase3.pth            # Phase 3输出
└── final_model.pth           # Phase 4输出（最终模型）
```

### 日志文件

```
logs/
└── training_history.json     # 训练历史记录
```

### TensorBoard日志（train_sb3.py）

```
logs_sb3/
├── tensorboard/              # TensorBoard日志
└── eval/                     # 评估日志
```

查看TensorBoard：

```bash
tensorboard --logdir=./logs_sb3/tensorboard/
```

---

## 性能优化建议

### 标准配置（4核CPU，8GB GPU）

```bash
python train.py --config configs/windows_base.yaml --phase all
```

- 并行环境：4
- 预计时间：2-4小时

### 高性能配置（8+核CPU，16GB GPU）

```bash
python train.py --config configs/windows_high_performance.yaml --phase all
```

- 并行环境：16
- 预计时间：1-2小时

### 极限性能配置（16+核CPU，24GB+ GPU）

```bash
python train.py --config configs/windows_extreme_performance.yaml --phase all
```

- 并行环境：32
- 预计时间：30-60分钟

---

## 常见问题

### Q1: 如何选择配置文件？

根据硬件配置选择：
- 4核CPU + 8GB显存：`windows_base.yaml`
- 8+核CPU + 16GB显存：`windows_high_performance.yaml`
- 16+核CPU + 24GB+显存：`windows_extreme_performance.yaml`

### Q2: 可以中断训练后继续吗？

目前训练脚本不支持断点续训。建议：
1. 一次性完成完整训练
2. 或者逐阶段训练，每个阶段完成后可以中断

### Q3: 如何使用已有数据？

使用 `--skip-data-collection` 标志：

```bash
python train.py --config configs/windows_base.yaml --phase 1 --skip-data-collection
```

### Q4: train.py、train_sb3.py 和 train_sb3_phase3.py 有什么区别？

| 特性 | train.py | train_sb3.py | train_sb3_phase3.py |
|------|----------|--------------|---------------------|
| 训练阶段 | Phase 1-4 | 仅Phase 2 | 仅Phase 3 |
| 实现方式 | 原生PyTorch | Stable-Baselines3 | Stable-Baselines3 |
| 适用场景 | 生产环境 | 快速实验 | ⭐ Phase 3加速 |
| 完整性 | 完整4阶段 | 简化PPO | 完整端到端 |
| 加速比 | 基准 | +30-50% | +40-60% |

### Q5: 如何评估训练好的模型？

```bash
# 仅评估模式
python train.py --config configs/windows_base.yaml --eval-only

# 评估并生成XLSX结果
python train.py --config configs/windows_base.yaml --eval-only --generate-xlsx
```

### Q6: 何时使用 train_sb3_phase3.py？

**推荐场景**：
- 已完成Phase 2训练，需要快速端到端微调
- 需要更快的训练速度（40-60%加速）
- 硬件资源有限，需要高效的训练

**不推荐场景**：
- 从头开始训练（需要先完成Phase 1和Phase 2）
- 需要完全自定义的训练逻辑

---

## 完整训练流程示例

### 方案1: 标准完整训练流程（原生PyTorch）

```bash
# Step 1: 运行完整4阶段训练
python train.py --config configs/windows_base.yaml --phase all

# Step 2: 评估训练好的模型
python train.py --config configs/windows_base.yaml --eval-only --generate-xlsx

# Step 3: 查看训练历史
cat logs/training_history.json
```

### 分阶段训练流程（原生PyTorch）

```bash
# Phase 1: 世界模型预训练
python train.py --config configs/windows_base.yaml --phase 1

# Phase 2: PPO训练
python train.py --config configs/windows_base.yaml --phase 2

# Phase 3: 端到端微调
python train.py --config configs/windows_base.yaml --phase 3

# Phase 4: 约束优化
python train.py --config configs/windows_base.yaml --phase 4

# 评估
python train.py --config configs/windows_base.yaml --eval-only --generate-xlsx
```

---

### 方案2: 混合加速训练流程（推荐）⭐

结合原生和SB3实现，获得最佳性能：

```bash
# Phase 1: 世界模型预训练（使用原生PyTorch）
python train.py --config configs/windows_base.yaml --phase 1

# Phase 2: PPO训练（使用SB3加速，快30-50%）
python train_sb3.py --config configs/windows_base.yaml --phase phase2

# Phase 3: 端到端微调（使用SB3加速，快40-60%）⭐
python train_sb3_phase3.py --config configs/windows_base.yaml \
    --phase2-checkpoint checkpoints_sb3/ppo_sumo_final.zip

# Phase 4: 约束优化（使用原生PyTorch，支持拉格朗日乘子）
python train.py --config configs/windows_base.yaml --phase 4

# 评估
python train.py --config configs/windows_base.yaml --eval-only --generate-xlsx
```

**性能提升**：相比全原生流程，总体加速约35-50%

---

### 方案3: 全SB3加速流程（极致性能）

最大化使用SB3加速（适用于快速实验）：

```bash
# Phase 1: 世界模型预训练（必须使用原生）
python train.py --config configs/windows_base.yaml --phase 1

# Phase 2: PPO训练（SB3加速）
python train_sb3.py --config configs/windows_high_performance.yaml --envs 16

# Phase 3: 端到端微调（SB3加速）⭐
python train_sb3_phase3.py --config configs/windows_high_performance.yaml \
    --envs 16 \
    --timesteps 100000 \
    --phase2-checkpoint checkpoints_sb3/ppo_sumo_final.zip

# 查看TensorBoard
tensorboard --logdir=./logs_sb3_phase3/tensorboard/
```

**性能提升**：相比全原生流程，总体加速约40-60%

---

### 快速实验流程（仅Phase 2）

```bash
# 快速PPO训练
python train_sb3.py --config configs/windows_base.yaml --phase phase2

# 查看TensorBoard
tensorboard --logdir=./logs_sb3/tensorboard/
```

---

## 技术支持

如有问题，请检查：
1. SUMO环境配置是否正确
2. GPU驱动和CUDA是否安装
3. Python依赖是否完整安装

祝训练顺利！
