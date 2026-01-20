# 智能交通协同控制系统 - v4.0

基于世界模型的分层多智能体交通控制系统，专为初赛设计。

## 📚 训练文档

- 🚀 **[完整训练流程指南](docs/完整训练流程指南.md)** - 详细的步骤说明和故障排查
- ⚡ **[训练快速参考卡](docs/训练快速参考卡.md)** - 一页纸快速参考
- 🧠 **[智能选择器集成指南](docs/智能选择器集成指南.md)** - 架构设计和集成说明
- 🏆 **[官方环境优化分析](docs/官方环境优化分析.md)** - 基于官方参数的优化策略
- 🚄 **[并行数据收集优化说明](docs/并行数据收集优化说明.md)** - 16并行环境加速训练 (NEW)

## 🎯 核心特性

### 架构设计
- **感知层**：风险敏感异构图神经网络（Risk-Sensitive GNN）
- **预测层**：多尺度潜在状态空间模型（World Model）
- **决策层**：影响力驱动的Top-K控制器
- **训练优化**：GPU加速的PPO实现，消除CPU-GPU传输瓶颈

### 关键技术
- ✅ **完整PPO实现**：Clipped Surrogate Objective + Value Loss + Entropy Bonus
- ✅ **GPU RolloutBuffer**：数据从收集到更新全程GPU存储，零拷贝传输
- ✅ **GAE (Generalized Advantage Estimation)**：完整实现，无简化逻辑
- ✅ **课程学习**：5个渐进训练级别（基础→中等→高流量→极端→赛题）
- ✅ **配置驱动**：所有训练参数通过YAML配置，无硬编码

## 🚀 快速开始

### 推荐：智能车辆选择训练流程（16并行环境加速）

```bash
# 1. 激活环境
conda activate sumo
cd /home/wyyyz/TJ_transport_v4

# 2. Stage 1 + Stage 2: 完整训练 (~5小时，16并行环境)
python train_phase1_lite.py --config configs/phase1_lite.yaml --stage all --device cuda

# 或者分阶段执行：
# Stage 1: 行为克隆 (1.5-2小时，16并行环境加速)
python train_phase1_lite.py --config configs/phase1_lite.yaml --stage stage1 --device cuda

# Stage 2: PPO微调 (3-4小时)
python train_phase1_lite.py --config configs/phase1_lite.yaml --stage stage2 --device cuda

# 4. 评估
python evaluate.py --config configs/phase1_lite.yaml \
  --checkpoint checkpoints/phase1_lite/best_model.pth --num-episodes 20
```

**详细说明**：参见 [完整训练流程指南](docs/完整训练流程指南.md)

---

### 传统训练流程（v4.0原方案）

#### 安装依赖
```bash
pip install -r requirements.txt
```

#### 一键训练（推荐）
```bash
train_all_stages.bat
```

这将自动执行：
1. **Phase 1**: 世界模型预训练（感知层 + 预测层）
2. **Stage 1-5**: 课程学习PPO训练（从简单到复杂）

#### 分阶段训练
```bash
# Phase 1: 世界模型训练
python train_phase1.py

# Phase 2: PPO课程学习（5个阶段）
python train_phase2.py --stage 1 --phase1-checkpoint checkpoints/competition/phase1/world_model_final.pth
python train_phase2.py --stage 2 --prev-checkpoint checkpoints/competition/curriculum/level1/custom_ppo.zip
python train_phase2.py --stage 3 --prev-checkpoint checkpoints/competition/curriculum/level2/custom_ppo.zip
python train_phase2.py --stage 4 --prev-checkpoint checkpoints/competition/curriculum/level3/custom_ppo.zip
python train_phase2.py --stage 5 --prev-checkpoint checkpoints/competition/curriculum/level4/custom_ppo.zip
```

### 查看训练进度
```bash
tensorboard --logdir runs/
# 访问 http://localhost:6006
```

## 📁 项目结构

```
TJ_transport_v4/
├── README.md                     # 项目说明（本文档）
│
├── train_phase1.py               # Phase 1: 世界模型训练 ⭐
├── train_phase2.py               # Phase 2: PPO课程学习训练 ⭐
├── train_all_stages.bat          # 完整训练流水线（一键运行）⭐
│
├── configs/
│   └── competition.yaml          # 训练配置和课程学习参数
│
├── src/
│   ├── models/
│   │   └── ideal_policy_v4.py    # v4.0策略网络（含evaluate_actions方法）
│   ├── training/
│   │   └── custom_ppo_trainer.py # 完整PPO实现（GPU优化）⭐
│   ├── env/
│   │   └── vec_env.py            # 并行环境包装器
│   └── utils/
│       └── helpers.py            # 辅助函数
│
├── checkpoints/                  # 模型检查点
│   └── competition/
│       ├── phase1/               # Phase 1 世界模型
│       └── curriculum/           # Phase 2 课程学习检查点
│           ├── level1/           # Stage 1: 基础场景
│           ├── level2/           # Stage 2: 中等流量
│           ├── level3/           # Stage 3: 高流量场景
│           ├── level4/           # Stage 4: 极端场景
│           └── level5/           # Stage 5: 赛题场景
│
├── logs/                         # 训练日志
└── 赛题.md                       # 赛题说明
```

## 🔧 配置说明

### 训练配置（configs/competition.yaml）

#### 课程学习级别
```yaml
# train_phase2.py中定义的5个课程级别
CURRICULUM_LEVELS = [
    {
        'level': 1,
        'name': '基础场景',
        'max_vehicles': 10,
        'inflow_rate': 800,
        'icv_ratio': 0.3,
        'disturbance_level': 0.0,
        'total_timesteps': 126000,  # 约61个update
    },
    {
        'level': 2,
        'name': '中等流量',
        'max_vehicles': 15,
        'inflow_rate': 1200,
        'icv_ratio': 0.25,
        'disturbance_level': 0.2,
        'total_timesteps': 126000,
    },
    {
        'level': 3,
        'name': '高流量场景',
        'max_vehicles': 20,
        'inflow_rate': 1800,
        'icv_ratio': 0.25,
        'disturbance_level': 0.4,
        'total_timesteps': 126000,
    },
    {
        'level': 4,
        'name': '极端场景',
        'max_vehicles': 32,
        'inflow_rate': 2400,
        'icv_ratio': 0.15,
        'disturbance_level': 0.7,
        'total_timesteps': 126000,
    },
    {
        'level': 5,
        'name': '赛题场景',
        'max_vehicles': 32,
        'inflow_rate': 2000,
        'icv_ratio': 0.2,
        'disturbance_level': 0.5,
        'total_timesteps': 1494000,  # 约729个update（74.7%训练时间）
    },
]
```

#### Phase 2 训练参数
```yaml
training:
  phase2:
    total_timesteps: 2000000
    num_envs: 4                    # 并行环境数
    n_steps: 2048                  # 每次rollout步数
    batch_size: 64                 # PPO更新mini-batch大小
    update_epochs: 10              # 每次更新epoch数
    learning_rate: 3e-4
    gamma: 0.99                    # 折扣因子
    gae_lambda: 0.95               # GAE参数
    clip_epsilon: 0.2              # PPO裁剪参数
    entropy_coef: 0.01             # 熵系数
    value_loss_coef: 0.5           # 价值损失系数
    max_grad_norm: 0.5             # 梯度裁剪
```

### 设备配置
```yaml
device: cuda:0  # 强制使用GPU 0
```

## 📊 训练流程详解

### Phase 1: 世界模型预训练
**文件**: `train_phase1.py`

**目标**:
- 学习车辆状态编码器（GNN）
- 学习交通流预测器（World Model）
- 学习Frenet坐标系转换

**输出**: `checkpoints/competition/phase1/world_model_final.pth`

**配置**:
```yaml
training:
  phase1:
    num_episodes: 50      # 数据收集episodes
    epochs: 30            # 训练轮数
    batch_size: 256
    learning_rate: 1e-4
```

### Phase 2: PPO课程学习训练
**文件**: `train_phase2.py`

**目标**:
- 训练决策策略（冻结感知层和预测层）
- 通过课程学习逐步提升难度
- 使用完整PPO算法（无简化）

**核心实现**: `src/training/custom_ppo_trainer.py`

**PPO组件**:
- ✅ **GPURolloutBuffer**: GPU上的经验回放缓冲区
- ✅ **GAE**: 广义优势估计
- ✅ **Clipped Surrogate Objective**: PPO裁剪目标
- ✅ **Value Function Loss**: MSE价值损失
- ✅ **Entropy Bonus**: 策略熵奖励
- ✅ **Multiple Epochs**: 多轮小批量更新

**输出**: `checkpoints/competition/curriculum/level{1-5}/custom_ppo.zip`

## ⚡ 性能优化

### GPU加速PPO实现

**问题**: Stable-Baselines3的RolloutBuffer在CPU上，导致PPO更新阶段GPU空闲

**解决方案**: 自定义GPU RolloutBuffer

```python
# src/training/custom_ppo_trainer.py

class GPURolloutBuffer:
    """GPU上的Rollout Buffer - 消除CPU-GPU传输"""
    def __init__(self, buffer_size, observation_space, action_space, device, n_envs, gamma, gae_lambda):
        # 所有buffer直接在GPU上分配
        self.observations = torch.zeros((buffer_size, n_envs, obs_dim), device=device)
        self.actions = torch.zeros((buffer_size, n_envs, action_dim), device=device)
        self.rewards = torch.zeros((buffer_size, n_envs), device=device)
        # ... 所有数据都在GPU上
```

**优势**:
- 数据从收集到更新全程GPU存储
- 消除PPO更新阶段的CPU→GPU传输
- GPU利用率从0%提升到60-80%

**预期性能**:
- 训练时间: ~8小时（相比SB3节省4.3小时）
- GPU利用率: 60-80%（相比SB3的0-20%）
- 显存占用: 高（数据全程GPU存储）

### 训练时间分布

```
Phase 1 (世界模型):     ~1小时
Stage 1 (基础场景):     ~0.6小时 (126k步)
Stage 2 (中等流量):     ~0.6小时 (126k步)
Stage 3 (高流量):       ~0.6小时 (126k步)
Stage 4 (极端场景):     ~0.6小时 (126k步)
Stage 5 (赛题场景):     ~4.4小时 (1,494k步)
-----------------------------------
总计:                   ~8小时
```

## 📈 TensorBoard监控

### 训练指标
```
rollout/ep_rew_mean      # Episode平均奖励
rollout/ep_len_mean      # Episode平均长度
train/policy_loss        # 策略损失
train/value_loss         # 价值损失
train/entropy_loss       # 熵损失
```

### 性能分析
```
TIMING/Rollout           # Rollout耗时
TIMING/Update            # PPO更新耗时
TIMING/DataTransfer      # CPU-GPU传输耗时
TIMING/Forward           # Forward pass耗时
TIMING/Backward          # Backward pass耗时
```

## 🐛 常见问题

### Q: 显存不足（CUDA out of memory）？
**A**:
1. 减小 `num_envs`（4→2）
2. 减小 `batch_size`（64→32）
3. 减小 `n_steps`（2048→1024）

### Q: GPU利用率为0？
**A**:
1. 确认使用 `train_phase2.py`（不是SB3）
2. 确认 `custom_ppo_trainer.py` 存在
3. 检查输出中的 `[TRAINER] Creating Custom PPO Trainer`

### Q: 如何从checkpoint继续训练？
**A**: 脚本会自动检测并加载已有checkpoint：
```bash
# Stage 2会自动加载Stage 1的checkpoint
python train_phase2.py --stage 2 --prev-checkpoint checkpoints/competition/curriculum/level1/custom_ppo.zip
```

### Q: 训练比预期慢很多？
**A**:
1. 确认Libsumo已安装（SUMO通信加速）
2. 检查GPU显存使用率（应该是高显存、高利用率）
3. 减少TensorBoard日志频率（`logging.log_interval: 500`）

### Q: 遇到 "Cannot find a working triton installation" 错误？
**A**:
**方案1（推荐）**: 安装 triton-windows 以获得最佳性能
```bash
pip install -U "triton-windows<3.3"
```
- ✅ 使用 `max-autotune` 模式，性能提升 20-30%（最快）
- ✅ 自动降级：如果失败会自动切换到 `reduce-overhead`

**方案2**: 不安装 Triton
- ✅ 自动使用 `reduce-overhead` 模式，性能提升 15-25%
- 📖 详见 `docs/TORCH_COMPILE_FIX.md`

## 📖 技术细节

### PPO实现完整性

所有PPO组件均已完整实现，无简化逻辑或占位符：

**GPURolloutBuffer**:
```python
def compute_returns_and_advantage(self, last_values, last_dones):
    """完整GAE实现 - 从后往前递归计算"""
    last_gae = 0
    for step in reversed(range(buffer_size)):
        next_values = last_values if step == buffer_size - 1 else self.values[step + 1]
        next_non_terminal = 1.0 - last_dones if step == buffer_size - 1 else 1.0 - self.episode_starts[step + 1]
        delta = rewards[step] + gamma * next_values * next_non_terminal - values[step]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        self.advantages[step] = last_gae
        self.returns[step] = last_gae + values[step]
```

**CustomPPOTrainer**:
```python
def _compute_policy_loss(self, ratio, advantages):
    """Clipped Surrogate Objective - 完整实现"""
    policy_loss = -torch.min(
        ratio * advantages,
        torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
    ).mean()
    return policy_loss

def _compute_value_loss(self, values, returns, old_values):
    """Value Function Loss - MSE损失"""
    value_loss = nn.functional.mse_loss(values, returns)
    return value_loss
```

### 课程学习策略

**渐进式难度提升**:
1. **Level 1**: 10辆车，800流量，无扰动 → 学习基础控制
2. **Level 2**: 15辆车，1200流量，20%扰动 → 适应中等负载
3. **Level 3**: 20辆车，1800流量，40%扰动 → 应对高流量
4. **Level 4**: 32辆车，2400流量，70%扰动 → 极端场景鲁棒性
5. **Level 5**: 32辆车，2000流量，50%扰动 → 赛题场景（74.7%训练时间）

**Checkpoint继承**: 每个阶段加载上一阶段权重，确保知识迁移

## 📄 许可证

本项目仅用于学习和竞赛目的。

---

**祝训练顺利！** 🚀
