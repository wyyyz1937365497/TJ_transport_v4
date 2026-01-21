# 数据收集缓存功能使用指南

## 概述

为了加速训练过程，所有三阶段训练脚本都支持数据缓存功能。缓存可以避免重复运行耗时的SUMO仿真，显著提升开发效率。

## 缓存机制

### Stage 1: World Observer Training
**缓存内容**: IDM轨迹数据
**缓存位置**: `cache/stage1_trajectories/`
**时间节省**: ~6分钟/episode

### Stage 2: Guided Exploration Training
**缓存内容**: PPO rollout数据
**缓存位置**: `cache/stage2_rollouts/`
**时间节省**: ~2-5分钟/iteration

### Stage 3: Constrained Optimization Training
**缓存内容**: PPO rollout数据（含成本）
**缓存位置**: `cache/stage3_rollouts/`
**时间节省**: ~2-5分钟/iteration

## 使用方法

### 1. 默认使用缓存（推荐）

所有训练脚本默认启用缓存：

```bash
# Stage 1
python scripts/train_stage1_world_observer.py \
    --config configs/v5_complete.yaml \
    --num_episodes 20 \
    --device cuda

# Stage 2
python scripts/train_stage2_guided_exploration.py \
    --config configs/v5_complete.yaml \
    --device cuda

# Stage 3
python scripts/train_stage3_constrained.py \
    --config configs/v5_complete.yaml \
    --device cuda
```

**首次运行**: 收集数据并缓存
**后续运行**: 自动加载缓存（秒级启动）

### 2. 禁用缓存

如果需要重新收集数据：

```bash
python scripts/train_stage1_world_observer.py \
    --config configs/v5_complete.yaml \
    --use_cache False \
    --device cuda
```

### 3. 强制刷新缓存

删除旧缓存并重新收集：

```bash
python scripts/train_stage1_world_observer.py \
    --config configs/v5_complete.yaml \
    --force_refresh \
    --device cuda
```

### 4. 自定义缓存目录

默认缓存目录：
- Stage 1: `cache/stage1_trajectories/`
- Stage 2: `cache/stage2_rollouts/`
- Stage 3: `cache/stage3_rollouts/`

可以指定自定义缓存目录：

```bash
python scripts/train_stage1_world_observer.py \
    --config configs/v5_complete.yaml \
    --cache_dir /path/to/custom/cache \
    --device cuda
```

## 缓存文件命名

缓存文件使用MD5哈希命名，基于配置参数生成：

```
trajectories_<hash>.pkl  # Stage 1
rollouts_<hash>.pkl      # Stage 2/3
```

**示例**:
- `trajectories_a1b2c3d4e5f6.pkl`
- `rollouts_f1e2d3c4b5a6.pkl`

不同的配置参数会生成不同的缓存文件，避免混淆。

## 清理缓存

### 清理所有缓存

```bash
rm -rf cache/
```

### 清理特定阶段缓存

```bash
# 只清理Stage 1缓存
rm -rf cache/stage1_trajectories/

# 只清理Stage 2缓存
rm -rf cache/stage2_rollouts/

# 只清理Stage 3缓存
rm -rf cache/stage3_rollouts/
```

### 查看缓存大小

```bash
# 查看总缓存大小
du -sh cache/

# 查看各阶段缓存大小
du -sh cache/stage1_trajectories/
du -sh cache/stage2_rollouts/
du -sh cache/stage3_rollouts/
```

## 缓存验证

缓存系统包含自动验证机制：

1. **文件存在性检查**: 确认缓存文件存在
2. **数据完整性验证**: 检查数据长度是否匹配
3. **异常处理**: 加载失败时自动回退到重新收集

## 缓存策略建议

### 开发阶段
- ✅ **启用缓存** - 快速迭代
- 使用相同的配置参数以复用缓存

### 调试阶段
- ✅ **启用缓存** - 确保数据一致性
- 使用 `--force_refresh` 更新缓存

### 生产训练
- ⚠️ **慎用缓存** - PPO需要新策略数据
- Stage 2/3缓存效果有限（策略不断更新）
- Stage 1缓存始终有效

### 超参数调优
- ✅ **禁用缓存** (`--use_cache False`) - 避免混淆
- 或使用不同的 `--num_episodes` 参数

## 注意事项

### Stage 2/3 缓存限制

⚠️ **重要**: Stage 2和Stage 3的PPO训练中，缓存效果有限，因为：

1. 策略参数在每个iteration后都会更新
2. 缓存的rollout来自旧策略，不适合新策略训练
3. 缓存主要用于：
   - 调试代码逻辑
   - 重复实验对比
   - 测试环境配置

**建议**: Stage 2/3正常训练时禁用缓存或使用 `--force_refresh`

### Stage 1 缓存

✅ **始终有效**: Stage 1的IDM轨迹缓存可以安全复用，因为：

1. IDM是固定策略（不学习）
2. 相同配置下轨迹完全一致
3. 可以大幅节省时间（20 episodes ≈ 2小时）

## 磁盘空间估算

### Stage 1 缓存
- 单个episode: ~5 MB
- 20 episodes: ~100 MB
- 50 episodes: ~250 MB

### Stage 2/3 缓存
- 单个rollout: ~1-5 MB
- 1000 steps: ~2 MB
- 完整训练: ~50-200 MB

### 总缓存大小
- 完整三阶段缓存: ~300-500 MB

## 故障排除

### 缓存加载失败

**症状**: 显示 "缓存加载失败"
**原因**: 缓存文件损坏或格式不兼容
**解决**:
```bash
rm -rf cache/
# 重新运行训练脚本
```

### 缓存数据不匹配

**症状**: 显示 "缓存数据不匹配"
**原因**: 配置参数变化
**解决**: 使用 `--force_refresh` 或删除对应缓存文件

### 磁盘空间不足

**症状**: 缓存保存失败
**解决**:
```bash
# 清理所有缓存
rm -rf cache/

# 或只清理特定阶段
rm -rf cache/stage1_trajectories/
```

## 命令行参数速查

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_cache` | True | 是否使用缓存 (True/False) |
| `--force_refresh` | False | 强制刷新缓存 |
| `--cache_dir` | cache/* | 自定义缓存目录 |

## 示例工作流

### 首次运行（收集并缓存）

```bash
# Stage 1: 收集20个IDM轨迹（~2小时）
python scripts/train_stage1_world_observer.py \
    --config configs/v5_complete.yaml \
    --num_episodes 20 \
    --device cuda

# 输出: [缓存] 缓存已保存: cache/stage1_trajectories/trajectories_abc123.pkl
```

### 后续运行（加载缓存）

```bash
# Stage 1: 瞬间加载（<1秒）
python scripts/train_stage1_world_observer.py \
    --config configs/v5_complete.yaml \
    --num_episodes 20 \
    --device cuda

# 输出: [缓存] 成功加载20条轨迹（耗时: <1秒）
#       [缓存] 节省了约120分钟的仿真时间
```

### 更新数据（刷新缓存）

```bash
# 修改配置后需要刷新
python scripts/train_stage1_world_observer.py \
    --config configs/v5_complete.yaml \
    --num_episodes 50 \
    --force_refresh \
    --device cuda

# 输出: [数据收集] 开始IDM轨迹收集...
```

## 总结

缓存功能通过智能复用已收集的数据，显著提升开发效率：

- ⚡ **时间节省**: Stage 1节省~2小时，Stage 2/3节省~10-50分钟
- 🚀 **快速迭代**: 无需等待SUMO仿真，立即开始训练
- 💾 **透明使用**: 默认启用，自动管理
- 🎯 **智能验证**: 自动检查缓存完整性

建议在开发和调试阶段充分利用缓存功能！
