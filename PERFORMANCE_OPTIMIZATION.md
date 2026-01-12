# Phase 2 训练性能优化指南

## 🎯 问题诊断

**症状：**
- CPU占用率低（<20%）
- GPU占用率低（10-20%）
- 训练速度慢

**根本原因：** SUMO环境仿真速度慢，成为训练瓶颈

## ⚡ 解决方案

### 方案1：使用优化配置（推荐）

```bash
# 使用优化后的配置文件
python train_v4_ideal.py --config configs/competition_preliminary_optimized.yaml --phase 2
```

**优化内容：**
- ✅ 并行环境：8 → 16（**2倍提升**）
- ✅ Episode长度：3600步 → 1800步（**2倍提升**）
- ✅ Batch大小：256 → 512（**提升GPU利用率**）
- ✅ Rollout步数：4096 → 2048（**更频繁更新**）

**预期效果：**
- 训练速度提升：**3-4倍**
- GPU占用率：60-80%
- CPU占用率：40-60%

### 方案2：手动优化配置

编辑 `configs/competition_preliminary.yaml`：

```yaml
training:
  phase2:
    num_envs: 16        # 关键！增加到16
    n_steps: 2048       # 减少rollout步数
    batch_size: 512     # 增加batch大小

environment:
  max_steps: 1800       # 减少episode长度
  num_parallel_workers: 16
```

### 方案3：根据硬件调整

#### 低配置（8GB内存，4核CPU）
```yaml
training:
  phase2:
    num_envs: 8         # 保持8
    n_steps: 1024       # 进一步减少
    batch_size: 256

environment:
  max_steps: 1200       # 减少到20分钟
```

#### 中等配置（16GB内存，8核CPU）
```yaml
training:
  phase2:
    num_envs: 16        # 推荐配置
    n_steps: 2048
    batch_size: 512

environment:
  max_steps: 1800
```

#### 高配置（32GB内存，16核CPU+）
```yaml
training:
  phase2:
    num_envs: 32        # 激进配置
    n_steps: 4096
    batch_size: 1024

environment:
  max_steps: 3600       # 保持完整长度
```

## 🔍 性能监控

### 1. 查看实时性能

训练时会显示：
```
rollout/    ep_rew_mean    100.00
rollout/    ep_len_mean    1500     # 平均episode长度
time/       fps            50       # 每秒步数（目标：>100）
time/       iterations     10
time/       time_elapsed   120
```

**性能指标：**
- `fps > 100`: 优秀
- `fps 50-100`: 良好
- `fps < 50`: 需要优化

### 2. 检查并行环境

```python
# 在Python中检查
import torch
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"CUDA核心数: {torch.cuda.device_count()}")
print(f"当前GPU: {torch.cuda.get_device_name(0)}")
```

### 3. 监控资源占用

**Windows:**
```bash
# 任务管理器查看GPU和CPU
# 或使用 nvidia-smi
nvidia-smi -l 1
```

**Linux:**
```bash
# 实时监控
htop  # CPU
nvidia-smi -l 1  # GPU
```

## 📊 性能对比

| 配置 | 并行环境 | Episode长度 | 训练速度 | GPU占用 |
|------|---------|-----------|---------|---------|
| 原始 | 8 | 3600步 | 1x | 10-20% |
| 优化 | 16 | 1800步 | 3-4x | 60-80% |
| 激进 | 32 | 1200步 | 6-8x | 80-95% |

## 🚨 常见问题

### Q1: 增加num_envs后内存不足

**解决方案：**
```bash
# 方案A：减少并行环境
num_envs: 12  # 而非16

# 方案B：减少batch_size
batch_size: 256  # 而非512

# 方案C：使用CPU训练
python train_v4_ideal.py --config configs/competition_preliminary.yaml --phase 2 --device cpu
```

### Q2: 训练仍然很慢

**检查清单：**
```bash
# 1. 确认CUDA可用
python -c "import torch; print(torch.cuda.is_available())"

# 2. 检查SUMO版本
sumo --version  # 需要 >1.8.0

# 3. 查看实际并行环境数
# 在训练日志中查找 "Using X parallel environments"

# 4. 检查episode长度
# 应该看到 "episode length: 1800" 而非 3600
```

### Q3: GPU占用仍然很低

**可能原因：**
1. 环境重置太慢
2. 数据传输开销大
3. 模型计算太快，环境是瓶颈

**解决方案：**
```yaml
# 减少n_steps（更频繁更新）
n_steps: 1024

# 增加batch_size（更多GPU计算）
batch_size: 1024

# 如果内存足够，增加num_envs
num_envs: 24
```

## 💡 最佳实践

### 1. 渐进式优化

```bash
# 步骤1：测试配置
python train_v4_ideal.py --config configs/competition_preliminary_optimized.yaml --phase 2

# 步骤2：监控性能（观察fps和GPU占用）

# 步骤3：根据硬件调整
# 如果GPU占用 <50%：增加num_envs
# 如果内存不足：减少num_envs或batch_size
```

### 2. 使用快速测试验证

```bash
# 先用快速配置测试
python train_v4_ideal.py --config configs/competition_quick_test.yaml --phase 2

# 确认无误后再用完整配置
python train_v4_ideal.py --config configs/competition_preliminary_optimized.yaml --phase 2
```

### 3. 分阶段训练

```bash
# Phase 1（通常较快）
python train_v4_ideal.py --config configs/competition_preliminary.yaml --phase 1

# Phase 2（使用优化配置）
python train_v4_ideal.py --config configs/competition_preliminary_optimized.yaml --phase 2

# Phase 3
python train_v4_ideal.py --config configs/competition_preliminary.yaml --phase 3
```

## 🎯 预期性能

使用优化配置后，您应该看到：

```
[INFO] Using 16 parallel environments
[INFO] Episode length: 1800 steps
[TRAIN] FPS: 150-200
[TRAIN] GPU占用: 60-80%
[TRAIN] CPU占用: 40-60%
```

训练时间从 **数小时** 减少到 **30-60分钟**！
