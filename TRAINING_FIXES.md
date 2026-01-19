# Phase 2 训练配置修复 - 2024-01-19

## 问题

训练时参数没有正确更新：
- 配置文件显示 `batch_size: 512`, `update_epochs: 10`
- 但训练输出显示 `Batch size: 256`, `Epochs per update: 5`
- 缺少详细的SUMO时间统计输出

## 根本原因

`train_phase2.py` 默认加载的是 `configs/competition.yaml`，而不是 `configs/competition_preliminary.yaml`。

## 修复内容

### 1. 更新 `configs/competition.yaml`

```yaml
# 修复前
phase2:
  num_envs: 1
  n_steps: 4096
  batch_size: 256
  update_epochs: 5

# 修复后
phase2:
  num_envs: 4        # ⭐ 4倍并行
  n_steps: 1024      # ⭐ 减少4倍
  batch_size: 512    # ⭐ 2倍batch
  update_epochs: 10  # ⭐ 2倍epoch
```

### 2. 增强 `src/training/custom_ppo_trainer.py`

#### 添加SUMO时间统计
```python
# 在collect_rollouts中添加
env_step_start = time.perf_counter()
next_obs, rewards, dones, infos = self.env.step(actions_np)
env_step_time = time.perf_counter() - env_step_start
self.timings['env_step'].append(env_step_time)
```

#### 增强训练输出
- ✅ 训练开始时显示完整配置
- ✅ 每次update显示时间分解
- ✅ 进度条显示SUMO平均step时间
- ✅ 每10次update显示详细统计

### 3. 同步更新 `configs/competition_preliminary.yaml`

已确保两个配置文件的phase2配置一致。

## 新的训练输出示例

### 训练开始
```
================================================================================
[TRAIN] Custom PPO Training (GPU Optimized + KL Penalty)
================================================================================
[INFO] Total timesteps: 126,000
[INFO] Device: cuda:0
[INFO] Parallel envs: 4
[INFO] Steps per rollout: 1024
[INFO] Batch size: 512
[INFO] Epochs per update: 10
[INFO] Learning rate: 0.0003
[INFO] Target KL: 0.01
[INFO] Transitions per update: 4096
[INFO] Total updates: 30
```

### 每次update
```
[Update 1/30] Steps: 4,096/126,000 | Rollout: 55.2s | Update: 4.8s | Total: 60.0s
```

### 进度条
```
[TRAIN] Phase 2: 10%|██| 3/30 [02:30<22:30, 20.0s/update, Steps=12,288, SUMO=108ms, Reward=-320, Time=60s]
```

### 详细统计（每10次update）
```
[Update 10/30] Steps: 40,960/126,000
  [TIMING] Rollout: 55.2s | Update: 4.8s | Total: 60.0s
  [SUMO] Avg step: 107.5ms | Total: 27.5s (49.8% of rollout)
  [METRICS] Episode Reward: -245.3 | Length: 1245.2
  [LOSS] Policy: -0.0234 | Value: 0.4521 | Entropy: 0.0123
  [KL] Divergence: 0.0098 (Target: 0.01)
```

## 验证修复

### 1. 检查配置文件
```bash
grep -A 8 "phase2:" configs/competition.yaml
# 应该显示：
#   num_envs: 4
#   n_steps: 1024
#   batch_size: 512
#   update_epochs: 10
```

### 2. 清理旧checkpoint
```bash
rm -rf checkpoints/competition/curriculum
```

### 3. 重新训练
```bash
# 单独训练Stage 1
python train_phase2.py --stage 1

# 或完整训练
./train_all.sh
```

### 4. 验证输出

训练开始时应该看到：
```
[INFO] Parallel envs: 4          # ✅ 应该是4，不是2或1
[INFO] Steps per rollout: 1024   # ✅ 应该是1024，不是2048或4096
[INFO] Batch size: 512           # ✅ 应该是512，不是256
[INFO] Epochs per update: 10     # ✅ 应该是10，不是5
```

## 预期性能提升

| 指标 | 旧配置 | 新配置 | 改善 |
|------|--------|--------|------|
| num_envs | 1-2 | 4 | 2-4倍并行 |
| n_steps | 2048-4096 | 1024 | 更快反馈 |
| batch_size | 256 | 512 | 更好GPU利用 |
| 每update时间 | ~120秒 | ~60秒 | **2倍快** |
| Stage 1总时间 | ~65分钟 | ~35分钟 | **1.86倍快** |

## 故障排查

### 如果仍然显示旧参数

1. 确认配置文件已更新：
   ```bash
   grep -A 5 "phase2:" configs/competition.yaml
   ```

2. 确认没有缓存：
   ```bash
   # 清理Python缓存
   find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
   find . -type f -name "*.pyc" -delete
   ```

3. 确认加载的是正确配置：
   ```bash
   # train_phase2.py默认加载configs/competition.yaml
   # 如果想用其他配置，显式指定：
   python train_phase2.py --stage 1 --config configs/competition_preliminary.yaml
   ```

### 如果SUMO时间仍然很慢（>200ms per step）

1. 检查SUMO配置：
   ```bash
   grep "time-step" configs/competition.yaml
   # 默认应该是0.1s，可以尝试增大到0.2s加速
   ```

2. 检查系统资源：
   ```bash
   # 查看CPU使用
   htop

   # 查看是否有进程卡住
   ps aux | grep sumo
   ```

3. 进一步增加并行：
   ```yaml
   num_envs: 4 → 8  # 如果有足够CPU核心
   ```

## 文件变更清单

- ✅ `configs/competition.yaml` - 更新phase2配置
- ✅ `configs/competition_preliminary.yaml` - 更新phase2配置
- ✅ `src/training/custom_ppo_trainer.py` - 添加SUMO计时和详细输出
- ✅ `docs/performance_optimization.md` - 性能优化文档
- ✅ `TRAINING_FIXES.md` - 本文档

## 下一步

1. 清理旧checkpoint
2. 重新开始训练
3. 观察SUMO avg step时间
4. 根据性能进一步调整
