# Phase 2 训练状态分析与修复 - 2024-01-19

## 📊 当前训练状态

### ✅ 性能提升显著

| 指标 | 优化前 | 现在 | 提升 |
|------|--------|------|------|
| 每个update | ~120秒 | ~52秒 | **2.3倍** 🚀 |
| Rollout时间 | ~110秒 | ~39秒 | **2.8倍** 🚀 |
| SUMO速度 | ~107ms/step | ~28-38ms/step | **2.8-3.8倍** 🚀 |
| 吞吐量 | ~34 steps/s | ~79 steps/s | **2.3倍** 🚀 |

**预期总训练时间**：
- Stage 1: 从65分钟 → **~28分钟** ✅
- 全部5个Stage: 从5.4小时 → **~2.3小时** ✅

### ⚠️ 发现的问题

#### 问题1: KL散度爆炸（严重）

```
[EARLY STOP] KL divergence (0.5264) exceeds threshold (0.0150)
[EARLY STOP] KL divergence (9.1439) exceeds threshold (0.0150)  # 极端！
```

**分析**：
- ✅ **预期KL**：0.01-0.03（正常范围）
- ❌ **实际KL**：0.5-9.3（**超出50-900倍**）
- ❌ **影响**：每个epoch都early stop，训练不充分

**根本原因**：
1. 从Phase 1加载的权重只包含感知和预测层
2. 决策层（actor, critic）是随机初始化的
3. 第一次训练时，随机网络产生剧烈的策略变化
4. 导致新旧策略分布差异巨大 → KL散度爆炸

**解决方案**：
```python
# 前10个update使用宽松的KL阈值（2.0），之后使用标准阈值（0.015）
kl_threshold = 2.0 if self.n_updates < 10 else self.target_kl * 1.5
```

#### 问题2: 学习率调度器警告

```
UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`
```

**原因**：在第一个epoch结束时就调用了lr_scheduler.step()

**影响**：跳过第一个学习率值

**解决方案**：调整lr_scheduler调用时机

### 📈 训练数据分析

#### Rollout进度（从日志推算）

| 步数 | 时间 | 速度 |
|------|------|------|
| 256步 | 7.2秒 | **28ms/step** |
| 512步 | 15.8秒 | **31ms/step** |
| 768步 | 27.1秒 | **35ms/step** |
| 1024步 | 39秒 | **38ms/step** |

**分析**：
- ✅ SUMO并行化效果显著（4个环境）
- ✅ 平均每个SUMO step约30-38ms
- ✅ 比单环境快约3倍

#### Update时间分布

```
Rollout: 39秒 (75%)
Update: 14秒 (25%)
Total: 53秒
```

**优化空间**：
- Rollout已经很高效（SUMO瓶颈已解决）
- Update时间占比小，GPU利用充分

## 🔧 已实施的修复

### 1. KL Early Stop阈值自适应

**文件**: `src/training/custom_ppo_trainer.py`

```python
# 前10个update使用宽松阈值，之后使用标准阈值
kl_threshold = 2.0 if self.n_updates < 10 else self.target_kl * 1.5

if kl_div.item() > kl_threshold:
    print(f"[EARLY STOP] KL divergence ({kl_div.item():.4f}) exceeds threshold ({kl_threshold:.4f}).")
    if self.n_updates < 10:
        print(f"  [INFO] Using relaxed KL threshold for first 10 updates (current: update {self.n_updates + 1})")
    break
```

**效果**：
- ✅ 前10个update允许更大的策略变化（KL < 2.0）
- ✅ 10个update后恢复标准阈值（KL < 0.015）
- ✅ 避免早期训练被过早中断

### 2. 配置文件完善

**文件**: `configs/competition.yaml`

```yaml
phase2:
  # 添加明确的KL阈值配置
  target_kl: 0.01  # KL散度目标值

  # 统一参数命名
  clip_range: 0.2  # 与代码中的clip_epsilon一致
```

## 📝 预期改进

### 修复前（当前状态）
```
[EARLY STOP] KL divergence (0.5264) exceeds threshold (0.0150)
→ 只训练了1-2个mini-batches就停止
→ 策略更新不充分
→ 训练效果差
```

### 修复后（预期）
```
前10个update:
  [INFO] Using relaxed KL threshold for first 10 updates (current: update 1)
  → KL < 2.0，训练完整的10个epoch
  → 策略充分更新
  → 快速适应从Phase 1加载的特征

10个update后:
  → 恢复标准阈值 KL < 0.015
  → 防止策略变化过大
  → 训练稳定
```

## 🎯 下一步行动

### 立即执行（停止当前训练）

当前训练由于KL early stop，效果很差。建议：

1. **停止当前训练**
   ```bash
   # 按 Ctrl+C 停止训练
   ```

2. **清理checkpoint**（可选）
   ```bash
   rm -rf checkpoints/competition/curriculum/level1
   ```

3. **重新开始训练**（使用修复后的代码）
   ```bash
   python train_phase2.py --stage 1
   ```

### 预期结果

修复后的训练应该显示：

```
[Update 1/30] Steps: 4,096/126,000
  [INFO] Using relaxed KL threshold for first 10 updates
  [TIMING] Rollout: 39s | Update: 25s | Total: 64s  # Update时间增加（完整的10个epoch）
  [LOSS] Policy: -0.0234 | Value: 0.4521 | Entropy: 0.0123
  [KL] Divergence: 0.8234 (Relaxed threshold: 2.000)  # KL在合理范围内

[Update 11/30] Steps: 44,036/126,000
  [TIMING] Rollout: 39s | Update: 18s | Total: 57s  # Update时间减少（KL early stop生效）
  [KL] Divergence: 0.0098 (Standard threshold: 0.015)  # KL降到正常范围
```

## 📊 监控指标

训练时关注以下指标：

### 正常范围

| 指标 | 前10个update | 10个update后 |
|------|-------------|--------------|
| KL散度 | 0.1-2.0 | 0.005-0.02 |
| Policy Loss | -0.01 ~ -0.05 | -0.02 ~ -0.1 |
| Value Loss | 0.5 ~ 2.0 | 0.2 ~ 0.8 |
| Episode Reward | 逐渐上升 | 稳定上升 |

### 异常信号

- ❌ KL > 5.0（前10个update）或KL > 0.05（之后）
- ❌ Policy Loss / Value Loss 突然变成NaN
- ❌ Episode Reward 持续下降
- ❌ 所有epoch都在early stop

## 🔄 如果问题仍然存在

### 方案A: 进一步降低学习率

```yaml
# configs/competition.yaml
phase2:
  learning_rate: 0.0001  # 从0.0003降到0.0001
```

### 方案B: 增加warmup steps

```python
# custom_ppo_trainer.py
warmup_iters = min(1000 // (self.n_steps * self.env.num_envs), n_updates)
# 从min(...)改为1000，强制1000步warmup
```

### 方案C: 从头开始训练（不加载Phase 1权重）

```bash
# 修改train_phase2.py，注释掉Phase 1权重加载部分
```

## 📚 相关文档

- `docs/performance_optimization.md` - 性能优化详情
- `docs/CONFIG_UNIFICATION.md` - 配置文件规范
- `TRAINING_FIXES.md` - 训练修复记录

## 总结

✅ **已完成**：
- 性能提升2.3倍（52秒/update）
- SUMO速度提升3倍（30ms/step）
- KL early stop阈值自适应修复

⚠️ **需要行动**：
- 停止当前训练
- 重新开始训练
- 监控KL散度变化

🎯 **预期效果**：
- 训练速度保持2倍提升
- KL散度控制在正常范围
- 策略充分训练和优化
