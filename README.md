# 智能交通协同控制 - v4.0架构

基于世界模型的分层多智能体交通控制系统，专为初赛设计。

## 🎯 核心特性

### 架构设计
- **感知层**：风险敏感异构图神经网络（Risk-Sensitive GNN）
- **预测层**：多尺度潜在状态空间模型（Multi-Scale RSSM）
- **决策层**：影响力驱动的Top-K控制器（Influence-Based Controller）
- **约束层**：双模态安全屏障 + 拉格朗日优化

### 增强功能（默认启用）
- ✅ **课程学习**：从简单到复杂场景的渐进训练（5个难度级别）
- ✅ **优先经验回放**：高价值状态优先采样（拥堵临界点、安全事件）
- ✅ **失败案例库**：专门训练安全模块（碰撞、急刹、拥堵）
- ✅ **并行数据收集**：多进程加速数据收集
- ✅ **数据缓存**：避免重复收集数据

## 🚀 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 训练模型
```bash
# 完整训练（推荐）
python train.py

# 指定配置文件
python train.py --config configs/competition.yaml

# 单独训练某个阶段
python train.py --phase 2

# 禁用增强功能
python train.py --no-enhancements
```

### 查看训练进度
```bash
tensorboard --logdir logs/
# 访问 http://localhost:6006
```

## 📁 项目结构

```
TJ_transport_v4/
├── train.py                      # 唯一训练入口
├── configs/
│   └── competition.yaml          # 默认配置（增强功能已启用）
├── src/
│   ├── models/
│   │   ├── v4_architecture.py    # v4.0架构核心模块
│   │   └── ideal_policy_v4.py    # SB3 PPO策略网络
│   ├── training/
│   │   ├── train_enhancements.py # 增强功能模块
│   │   └── __init__.py
│   ├── env/
│   │   ├── competition_env.py    # SUMO竞赛环境
│   │   ├── gym_wrapper.py        # Gym包装器
│   │   └── vec_env.py            # 并行环境
│   └── utils/
│       └── helpers.py            # 辅助函数
├── checkpoints/                  # 模型检查点
├── logs/                         # TensorBoard日志
└── 赛题.md                       # 赛题说明
```

## 🔧 配置说明

### 课程学习配置
```yaml
training:
  curriculum:
    enabled: true              # 启用课程学习
    auto_advance: true         # 自动晋级
    min_episodes_per_level: 50 # 每级别最少训练episodes
```

### 优先经验回放配置
```yaml
training:
  prioritized_replay:
    enabled: true        # 启用优先经验回放
    capacity: 100000     # 缓冲区容量
    alpha: 0.6           # 优先级指数
    beta_start: 0.4      # 重要性采样初始值
```

### 失败案例库配置
```yaml
training:
  failure_bank:
    enabled: true               # 启用失败案例库
    max_size: 1000              # 最大存储案例数
    auto_detect: true           # 自动检测失败
    failure_sampling_ratio: 0.3 # 训练时失败案例占比
```

## 📊 训练流程

### Phase 1: 世界模型预训练
- **目标**：学习交通流演化规律
- **方法**：监督学习（预测速度、位置、冲突）
- **数据**：使用课程学习收集多样化数据
- **输出**：`checkpoints/phase1/final.pth`

### Phase 2: PPO训练（冻结感知层）
- **目标**：训练决策策略
- **方法**：PPO强化学习
- **增强**：课程学习 + 优先经验回放 + 失败案例库
- **输出**：`checkpoints/phase2/final.zip`

### Phase 3: 拉格朗日约束优化
- **目标**：端到端微调，满足成本约束
- **方法**：拉格朗日乘子法
- **输出**：`checkpoints/phase3/final.zip`

## 📈 TensorBoard指标

### 基础指标
- `rollout/ep_rew_mean`: 平均奖励
- `rollout/ep_len_mean`: 平均episode长度
- `train/value_loss`: 价值损失
- `train/policy_gradient_loss`: 策略梯度损失

### 课程学习指标
- `curriculum/level`: 当前难度级别（1-5）
- `curriculum/progress`: 总进度（0-1）
- `curriculum/avg_reward`: 当前级别平均奖励

### 失败案例库指标
- `failure_bank/total`: 失败案例总数
- `failure_bank/collision`: 碰撞案例数
- `failure_bank/braking`: 急刹案例数

### 优先经验回放指标
- `replay_buffer/size`: 缓冲区大小
- `replay_buffer/frame`: 当前帧数

## 🎯 性能优化

### 并行数据收集
- **实现**：使用多进程Pool并行收集数据
- **配置**：`training.phase1.num_parallel_workers`（默认4个worker）
- **加速**：数据收集速度提升3-4倍

### 数据缓存
- **实现**：pickle序列化存储收集的数据
- **位置**：`checkpoints/phase1/cache/data.pkl`
- **优势**：避免重复收集，节省时间

### GPU加速
- **自动检测**：CUDA可用时自动使用GPU
- **手动指定**：`python train.py --device cuda`

## ⚙️ 高级用法

### 自定义课程学习难度级别
编辑 `src/training/train_enhancements.py` 中的 `DifficultyLevel` 列表：
```python
DifficultyLevel(
    level=1,
    name="自定义难度",
    max_vehicles=10,
    inflow_rate=800,
    icv_ratio=0.3,
    episodes=100
)
```

### 调整增强功能强度
```yaml
# 激进训练（更快晋级）
training:
  curriculum:
    min_episodes_per_level: 30

# 保守训练（更稳晋级）
training:
  curriculum:
    min_episodes_per_level: 100
```

### 禁用特定增强功能
```bash
# 禁用所有增强功能
python train.py --no-enhancements

# 或在配置文件中单独禁用
training:
  curriculum:
    enabled: false
  prioritized_replay:
    enabled: true
  failure_bank:
    enabled: false
```

## 🐛 常见问题

### Q: 训练很慢怎么办？
A:
1. 减少 `num_parallel_workers`（如果CPU不够）
2. 禁用部分增强功能（`--no-enhancements`）
3. 减少 `total_timesteps`

### Q: 显存不足（CUDA out of memory）？
A:
1. 减小 `batch_size`
2. 减小 `num_envs`
3. 使用CPU训练（`--device cpu`）

### Q: 课程学习一直不晋级？
A:
1. 降低 `min_episodes_per_level`
2. 检查TensorBoard中的 `curriculum/avg_reward`
3. 调整各级别的 `min_reward` 和 `success_threshold`

### Q: 如何从checkpoint继续训练？
A: 脚本会自动检测已有checkpoint并继续训练：
```bash
python train.py --phase all
```

## 📖 参考资料

### v4.0架构详解
- 感知层：风险敏感异构图GNN（TTC、THW特征）
- 预测层：解耦为z_flow（流演化）和z_risk（风险演化）
- 决策层：可学习权重的Top-K控制器（α、β参数）
- 约束层：动态拉格朗日乘子更新

### 增强功能原理
- **课程学习**：Bengio et al. (2009)
- **优先经验回放**：Schaul et al. (2016)
- **失败案例库**：专门针对交通控制场景设计

## 📄 许可证

本项目仅用于学习和竞赛目的。

## 🙏 致谢

感谢所有贡献者的努力！

---

**祝训练顺利！** 🚀
