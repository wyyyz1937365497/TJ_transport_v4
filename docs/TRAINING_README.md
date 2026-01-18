# 训练脚本使用说明

本项目提供了完整的自动化训练脚本系统，支持从零开始到最终模型的全流程训练。

## 📋 训练流程概述

```
Phase 1 (世界模型训练)
    ↓
Phase 2 (PPO课程学习)
    ├─ Level 1: 基础场景 (10车辆, 800 inflow)
    ├─ Level 2: 中等流量 (15车辆, 1200 inflow)
    ├─ Level 3: 高流量场景 (20车辆, 1800 inflow)
    ├─ Level 4: 极端场景 (32车辆, 2400 inflow)
    └─ Level 5: 赛题场景 (32车辆, 2000 inflow)
    ↓
Phase 3 (微调优化，可选)
```

## 🚀 快速开始

### 1. 一键训练所有阶段

```bash
./train_all.sh
```

这将自动执行：
- ✅ Phase 1: 世界模型训练
- ✅ Phase 2: 所有5个课程级别
- ✅ 自动检测并跳过已完成的阶段
- ✅ 支持断点续训

### 2. 从指定阶段开始训练

```bash
./train_all.sh --from-phase 2    # 从Phase 2开始（跳过Phase 1）
./train_all.sh --from-phase 3    # 从Phase 3开始（需要Phase 2完成）
```

### 3. 快速测试（5-10分钟）

```bash
./train_quick_test.sh
```

使用最小配置验证训练流程是否正常。

## 📚 分阶段训练

### Phase 1: 世界模型训练

```bash
python train_phase1.py
```

**训练内容：**
- 车辆状态编码器（GNN）
- 交通流预测器（World Model）
- Frenet坐标系转换

**输出：** `checkpoints/competition/phase1/world_model_final.pth`

**预计时间：** 2-3小时（100 epochs）

### Phase 2: PPO课程学习

#### 方法1: 使用主脚本训练所有级别

```bash
./train_all.sh --from-phase 2
```

#### 方法2: 单独训练指定级别

```bash
# Level 1（基础场景，无需检查点）
./train_phase2_curriculum.sh --level 1

# Level 2-5（需要上一阶段检查点）
./train_phase2_curriculum.sh --level 2 --checkpoint checkpoints/competition/curriculum/level1/custom_ppo.zip
./train_phase2_curriculum.sh --level 3 --checkpoint checkpoints/competition/curriculum/level2/custom_ppo.zip
./train_phase2_curriculum.sh --level 4 --checkpoint checkpoints/competition/curriculum/level3/custom_ppo.zip
./train_phase2_curriculum.sh --level 5 --checkpoint checkpoints/competition/curriculum/level4/custom_ppo.zip
```

#### 课程级别说明

| Level | 名称 | 车辆数 | 流量 | ICV比例 | 训练步数 | 难度 |
|-------|------|--------|------|---------|----------|------|
| 1 | 基础场景 | 10 | 800 | 30% | 126K | ⭐ |
| 2 | 中等流量 | 15 | 1200 | 25% | 126K | ⭐⭐ |
| 3 | 高流量场景 | 20 | 1800 | 25% | 126K | ⭐⭐⭐ |
| 4 | 极端场景 | 32 | 2400 | 15% | 126K | ⭐⭐⭐⭐ |
| 5 | 赛题场景 | 32 | 2000 | 20% | 1494K | ⭐⭐⭐⭐⭐ |

**预计总时间：** 约8-10小时（所有级别）

### Phase 3: 微调优化（可选）

```bash
# 微调训练
python train_phase3.py --base-model checkpoints/competition/curriculum/level5/custom_ppo.zip --mode fine-tune

# 模型评估
python train_phase3.py --base-model checkpoints/competition/best_model.zip --mode eval --num-episodes 20

# 集成学习（开发中）
python train_phase3.py --base-model model1.zip --mode ensemble --ensemble-models model1.zip model2.zip model3.zip
```

## 📁 文件结构

```
TJ_transport_v4/
├── train_all.sh                  # 主训练脚本（所有阶段）
├── train_quick_test.sh           # 快速测试脚本
├── train_phase1.py               # Phase 1训练脚本
├── train_phase2.py               # Phase 2训练脚本（课程学习）
├── train_phase2_curriculum.sh    # Phase 2分级别训练脚本
├── train_phase3.py               # Phase 3微调脚本
│
├── configs/
│   ├── competition.yaml          # 主配置文件
│   └── optimized_ppo.yaml        # PPO优化配置
│
└── checkpoints/
    └── competition/
        ├── phase1/
        │   └── world_model_final.pth
        ├── curriculum/
        │   ├── level1/
        │   ├── level2/
        │   ├── level3/
        │   ├── level4/
        │   └── level5/
        │       └── custom_ppo.zip
        └── phase3/
            └── fine_tuned_model.zip
```

## 🔄 断点续训

所有训练脚本都支持自动检测已完成的阶段：

```bash
# 如果训练中断，重新运行相同的命令即可
./train_all.sh --resume

# 脚本会自动：
# 1. 检查已存在的检查点
# 2. 询问是否跳过已完成的阶段
# 3. 从中断的阶段继续训练
```

## 📊 监控训练进度

### 查看日志

```bash
# 实时查看日志
tail -f logs/phase2/stage1_20250118_120000.log

# 查看最近的训练日志
ls -lt logs/phase2/ | head -10
```

### TensorBoard（如果配置了）

```bash
tensorboard --logdir logs/
```

## ⚙️ 配置文件

### 主要配置文件：`configs/competition.yaml`

```yaml
training:
  phase1:
    num_epochs: 100
    batch_size: 32
    learning_rate: 0.001

  phase2:
    num_envs: 4
    n_steps: 2048
    batch_size: 64
    update_epochs: 10
    learning_rate: 0.0003

  phase3:
    num_envs: 4
    n_steps: 2048
    batch_size: 64
    update_epochs: 10
    learning_rate: 0.0001  # 更小的学习率用于微调
    total_timesteps: 500000

environment:
  max_vehicles: 32
  inflow_rate: 2000
  icv_ratio: 0.2
  disturbance_level: 0.5
```

## 🛠️ 故障排除

### 问题1: CUDA out of memory

**解决方案：** 减少环境数量或batch size

```bash
# 在配置文件中修改
environment:
  num_envs: 2  # 从4减少到2
```

### 问题2: 训练速度慢

**解决方案：** 检查GPU利用率

```bash
nvidia-smi -l 1  # 实时监控GPU
```

### 问题3: 检查点加载失败

**解决方案：** 确认检查点路径正确

```bash
# 列出所有检查点
find checkpoints/ -name "*.pth" -o -name "*.zip"
```

## 📞 帮助

查看脚本帮助信息：

```bash
./train_all.sh --help
./train_phase2_curriculum.sh --help
python train_phase3.py --help
```

## ✅ 训练完成后

1. **最佳模型位置：** `checkpoints/competition/curriculum/level5/custom_ppo.zip`

2. **评估模型：**
```bash
python train_phase3.py --base-model checkpoints/competition/curriculum/level5/custom_ppo.zip --mode eval
```

3. **提交比赛：** 使用Level 5的模型进行最终测试

---

**祝训练顺利！** 🚀
