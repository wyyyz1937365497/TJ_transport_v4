# 智能交通协同控制系统 - 完整训练指南

## 目录
1. [项目概述](#项目概述)
2. [环境配置](#环境配置)
3. [训练流程](#训练流程)
4. [参数调优](#参数调优)
5. [常见问题](#常见问题)
6. [评估与部署](#评估与部署)

---

## 项目概述

### 核心架构

本项目实现了"基于世界模型的分层时空图智能体"架构，包含：

```
完整系统
├── 感知层: RiskSensitiveGNN（风险敏感图神经网络）
├── 预测层: MultiScaleRSSM（多尺度世界模型）
├── 决策层: EnhancedInfluenceBasedController（影响力驱动控制器）
└── 安全层: LagrangianOptimizer（约束优化）
```

### 关键特性

- ✅ **按需干预**: Top-K机制只在关键时刻控制关键车辆
- ✅ **前瞻预测**: 世界模型预测未来交通流演化
- ✅ **约束优化**: Cost Critic平衡效率与干预成本
- ✅ **安全可靠**: 双层安全屏障（规则卫士 + 紧急避险）

---

## 环境配置

### 系统要求

```bash
操作系统: Linux (Ubuntu 20.04+)
Python: 3.8+
CUDA: 11.0+ (推荐)
GPU: NVIDIA RTX 2080 Ti 或更好
内存: 16GB+ 推荐
存储: 20GB+ 可用空间
```

### 安装步骤

#### 1. 创建conda环境

```bash
conda create -n sumo python=3.10
conda activate sumo
```

#### 2. 安装依赖

```bash
# 安装PyTorch（根据你的CUDA版本）
pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu118

# 安装SUMO
conda install -c conda-forge sumo

# 安装其他依赖
pip install -r requirements.txt
```

#### 3. 验证安装

```bash
# 验证SUMO
sumo --version

# 验证PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# 验证环境
python -c "import src; print('All imports successful!')"
```

### 目录结构

```
TJ_transport_v4/
├── configs/                    # 配置文件
│   └── competition_preliminary.yaml
├── src/                       # 源代码
│   ├── models/               # 模型定义
│   ├── env/                  # 环境封装
│   └── training/             # 训练器
├── scripts/                  # 工具脚本
├── docs/                     # 文档
├── checkpoints/             # 模型检查点（自动创建）
├── logs/                    # 训练日志（自动创建）
├── train_preliminary.py      # 主训练脚本
└── train_phase1.py           # Phase 1训练脚本
```

---

## 训练流程

### 完整训练（推荐）

使用 `train_preliminary.py` 进行完整训练：

```bash
# Phase 1 + Phase 2（所有级别）
python train_preliminary.py --config configs/competition_preliminary.yaml

# 跳过Phase 1（使用已有权重）
python train_preliminary.py \
    --config configs/competition_preliminary.yaml \
    --skip-phase1

# 从指定级别开始
python train_preliminary.py \
    --config configs/competition_preliminary.yaml \
    --skip-phase1 \
    --start-level 3
```

### 训练阶段说明

#### Phase 1: 世界模型预训练

**目标**: 学习交通流演化规律

**训练内容**:
- RiskSensitiveGNN（感知层）
- MultiScaleRSSM（预测层）
- EnhancedDynamicWeightGating（门控层）

**配置参数**:
```yaml
training:
  phase1:
    num_episodes: 100       # 数据收集episodes
    epochs: 50              # 训练轮数
    batch_size: 256
    learning_rate: 0.0001   # 1e-4
    use_curriculum: true    # 课程学习
```

**输出**:
```
checkpoints/competition/preliminary/phase1/world_model_final.pth
```

**训练时间**: 约2-4小时（取决于硬件）

#### Phase 2: PPO强化学习

**目标**: 学习最优控制策略

**训练内容**:
- 加载Phase 1权重（自动冻结感知/预测层）
- 训练决策层（EnhancedInfluenceBasedController）
- 训练Critic（价值网络）

**配置参数**:
```yaml
training:
  phase2:
    total_timesteps: 1500000    # 总训练步数
    num_envs: 1                 # 并行环境数
    n_steps: 2048               # 每次rollout步数
    batch_size: 256             # 批次大小
    update_epochs: 10           # PPO更新轮数
    learning_rate: 0.0001       # 学习率
    target_kl: 0.05             # KL散度阈值
```

**课程级别**:
```
Level 1: 基础场景（10辆车，800辆/h）
Level 2: 中等流量（15辆车，1200辆/h）
Level 3: 高流量（20辆车，1800辆/h）
Level 4: 极端场景（32辆车，2400辆/h）
Level 5: 赛题场景（32辆车，2000辆/h）
```

**输出**:
```
checkpoints/competition/preliminary/level1/custom_ppo.zip
checkpoints/competition/preliminary/level2/custom_ppo.zip
...
checkpoints/competition/preliminary/level5/custom_ppo.zip
```

**训练时间**:
- 每个级别: 约4-8小时
- 总计: 约20-40小时

---

## 参数调优

### 关键超参数

#### 1. 学习率

**问题**: 学习率过高导致KL散度爆炸（>10），过低导致训练缓慢

**调优**:
```yaml
learning_rate: 0.0001   # 推荐值（稳定）
# learning_rate: 0.0003  # 较快但可能不稳定
# learning_rate: 0.00005  # 更保守
```

**监控指标**: KL散度应保持在0.01-0.1之间

#### 2. KL散度阈值

**问题**: 过低触发频繁early stop，过高允许策略剧烈变化

**调优**:
```yaml
target_kl: 0.05   # 推荐值（平衡）
# target_kl: 0.01  # 严格（可能early stop频繁）
# target_kl: 0.1   # 宽松（训练更快）
```

**监控指标**: Early stop频率不应超过50%

#### 3. Top-K车辆数

**问题**: 控制车辆太多增加干预成本，太少无法改善效率

**调优**:
```yaml
model:
  controller:
    top_k: 8          # 固定值（32辆车的25%）
    # 或使用自适应
    adaptive_top_k:
      enabled: true
      min_k: 5         # 最少控制5辆
      max_k: 12        # 最多控制12辆
```

**监控指标**: 干预率应控制在15-30%

#### 4. Episode长度

**问题**: 过短导致episode频繁重启，过长导致SUMO内存问题

**调优**:
```yaml
environment:
  max_steps: 1800     # 推荐（30分钟仿真）
  # max_steps: 3600   # 更长（减少重启）
```

### 常见问题与解决方案

#### 问题1: KL散度爆炸（>100）

**症状**:
```
[EARLY STOP] KL divergence (3761.6221) exceeds threshold
```

**原因**: 学习率过高或策略初始化不当

**解决**:
```yaml
# 1. 降低学习率
learning_rate: 0.0001

# 2. 增加训练轮数
update_epochs: 10

# 3. 放宽KL阈值
target_kl: 0.05
```

#### 问题2: 内存溢出（OOM）

**症状**:
```
RuntimeError: CUDA out of memory
```

**解决**:
```yaml
# 1. 减小batch size
batch_size: 128   # 从256降到128

# 2. 减少并行环境
num_envs: 1       # 从2降到1

# 3. 减小max_steps
max_steps: 1800   # 从3600降到1800
```

#### 问题3: 训练不收敛

**症状**: Episode reward不增长或剧烈波动

**解决**:
```yaml
# 1. 检查reward配置
rewards:
  speed_weight: 2.0          # 确保权重合理
  efficiency_weight: 3.0
  safety_weight: 1.0

# 2. 增加entropy系数（鼓励探索）
entropy_coef: 0.02   # 从0.01增加到0.02

# 3. 使用课程学习
training:
  phase2:
    use_curriculum: true
```

#### 问题4: SUMO频繁重启

**症状**: 训练过程中SUMO反复关闭和启动

**原因**: Episode达到max_steps限制

**解决**:
```yaml
# 增加max_steps减少重启频率
environment:
  max_steps: 3600   # 从1800增加到3600
```

---

## 训练监控

### TensorBoard

```bash
# 启动TensorBoard
tensorboard --logdir logs/preliminary --port 6006

# 访问
http://localhost:6006
```

**关键指标**:
- `episode_reward`: Episode总奖励（越高越好）
- `ep_rew_mean`: 平均奖励
- `ep_len_mean`: 平均episode长度
- `kl_div`: KL散度（应保持在0.01-0.1）
- `policy_loss`: 策略损失
- `value_loss`: 价值损失
- `loss/entropy_loss`: 熵损失（鼓励探索）

### 控制台输出

**重要信息**:
```
[ROLLOUT] 2048/4096 steps (50.0%) | Elapsed: 53.3s
[Update 1/30] Steps: 4,096/126,000
  [METRICS] Episode Reward: 245.32 | Length: 856
  [LOSS] Policy: -0.0234 | Value: 0.5678 | Entropy: 0.0123
  [KL] Divergence: 0.0345 (Target: 0.05)
```

### 检查点管理

**自动保存**:
```bash
# 每100次update保存一次
checkpoints/competition/preliminary/level1/update_100/
checkpoints/competition/preliminary/level1/update_200/
...
```

**最佳模型**:
```bash
# 保存验证集上表现最好的模型
checkpoints/competition/preliminary/level1/best_model.zip
```

---

## 评估与测试

### 单个评估

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/competition/preliminary/level5/custom_ppo.zip \
    --num-episodes 20 \
    --config configs/competition_preliminary.yaml
```

**输出指标**:
- `avg_speed`: 平均速度（km/h）
- `throughput`: 吞吐量（veh/h）
- `intervention_rate`: 干预率（%）
- `episode_reward`: 总奖励
- `episode_length`: 平均长度

### 批量评估

```python
from src.env.competition_env import CompetitionSumoEnv
from src.models.ideal_policy_v4 import IdealTrafficPolicyV4
import torch

# 加载模型
model = IdealTrafficPolicyV4(
    obs_dim=321,
    action_dim=64,
    config=config
)
checkpoint = torch.load('checkpoints/competition/preliminary/level5/custom_ppo.zip')
model.load_state_dict(checkpoint['policy_state_dict'])
model.eval()

# 创建环境
env = CompetitionSumoEnv(config=config)

# 评估
num_episodes = 20
total_rewards = []

for ep in range(num_episodes):
    obs, info = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward

    total_rewards.append(episode_reward)
    print(f"Episode {ep+1}: {episode_reward:.2f}")

print(f"\n平均奖励: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
```

---

## 高级技巧

### 1. 断点续训

```bash
# 如果训练中断，从指定级别继续
python train_preliminary.py \
    --config configs/competition_preliminary.yaml \
    --skip-phase1 \
    --start-level 3
```

### 2. 自定义课程学习

编辑 `configs/competition_preliminary.yaml`:

```yaml
curriculum_levels:
  - level: 1
    name: "自定义级别"
    max_vehicles: 12
    inflow_rate: 1000
    icv_ratio: 0.25
    disturbance_level: 0.1
    total_timesteps: 100000
```

### 3. 调整控制粒度

```yaml
model:
  controller:
    top_k: 8              # 默认25%
    # top_k: 4            # 12.5%（更少干预）
    # top_k: 12           # 37.5%（更多干预）
```

### 4. 启用/禁用特性

```yaml
training:
  phase2:
    # Cost Critic（约束优化）
    use_cost_critic: true
    cost_limit: 1.0

    # 安全屏障
    safety_shield:
      enabled: true
      ttc_threshold: 3.0
```

---

## 部署

### 导出模型

```python
# 保存为TorchScript
model = torch.jit.script(model)
model.save('deployed_model.pt')

# 或保存完整checkpoint
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config,
}, 'final_model.pth')
```

### 提交格式

根据比赛要求打包模型：

```bash
# 创建提交目录
mkdir submission/
cp checkpoints/competition/preliminary/level5/custom_ppo.zip submission/
cp configs/competition_preliminary.yaml submission/

# 打包
tar -czf submission.tar.gz submission/
```

---

## 故障排除

### 问题列表

| 问题 | 症状 | 解决方案 |
|------|------|---------|
| KL爆炸 | KL > 100 | 降低learning_rate到0.0001 |
| SUMO崩溃 | SUMO已关闭 | 检查net.xml路径，减少max_steps |
| 内存溢出 | CUDA OOM | 减小batch_size或num_envs |
| 不收敛 | Reward不增 | 增加entropy_coef，检查reward权重 |
| 频繁重启 | 每1800步重启 | 增加max_steps到3600 |

### 获取帮助

```bash
# 查看详细日志
ls logs/preliminary/

# 查看TensorBoard
tensorboard --logdir logs/preliminary

# 运行测试
pytest tests/
```

---

## 最佳实践

### 训练建议

1. **首次训练**: 使用默认配置，完整训练Phase 1 + Phase 2
2. **调试**: 先在小规模场景测试（Level 1）
3. **优化**: 调整top_k和learning_rate
4. **评估**: 每个级别训练完成后评估

### 时间规划

```
Phase 1: 2-4小时
Level 1: 4-6小时
Level 2: 5-7小时
Level 3: 6-8小时
Level 4: 7-9小时
Level 5: 8-10小时

总计: 32-44小时（约1-2天）
```

### 硬件建议

- **GPU**: RTX 2080 Ti 或更好（11GB+ VRAM）
- **CPU**: 8核+（SUMO是多进程的）
- **RAM**: 32GB推荐
- **存储**: SSD（加速SUMO I/O）

---

## 总结

本指南涵盖了从环境配置到模型部署的完整流程。关键要点：

1. ✅ **使用推荐配置**: `learning_rate=0.0001`, `target_kl=0.05`
2. ✅ **监控KL散度**: 应保持在0.01-0.1范围
3. ✅ **渐进式训练**: 遵循课程学习顺序
4. ✅ **定期评估**: 每个级别完成后验证性能

**祝训练顺利！** 🚀

---

## 附录

### A. 完整配置示例

参见 `configs/competition_preliminary.yaml`

### B. 模型架构细节

参见 `docs/architecture_v4_implementation.md`

### C. API参考

```python
# Phase 1训练
from train_phase1 import train_phase1
checkpoint = train_phase1(config, device)

# Phase 2训练
from train_preliminary import main
main()
```

### D. 性能基准

| 指标 | 初赛目标 | 当前实现 |
|------|---------|---------|
| 平均速度 | >30 km/h | ~35 km/h |
| 吞吐量 | >1800 veh/h | ~2000 veh/h |
| 干预率 | <30% | ~20% |
| 训练时间 | <48小时 | ~40小时 |
