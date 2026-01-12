# 🚀 超快速测试配置使用指南

## ⚡ 快速开始

```bash
# 一键运行所有阶段（5-10分钟完成）
python train_v4_ideal.py --config configs/competition_ultra_fast_test.yaml --phase all
```

## 📊 配置策略

该配置采用**"三管齐下"**策略加速训练：

### 1️⃣ 减少训练步数（节省时间）

| 阶段 | 原始步数 | 优化步数 | 说明 |
|------|---------|---------|------|
| Phase 1 | 50 episodes × 30 epochs | **3 episodes × 3 epochs** | 快速数据收集 |
| Phase 2 | 200,000 步 | **496 步** | 最少验证训练（16×31=496，1个rollout） |
| Phase 3 | 100,000 步 | **248 步** | 快速微调（4×62=248，1个rollout） |
| Phase 4 | 100,000 步 | **248 步** | 快速约束优化（由Phase 3执行） |

### 2️⃣ 增大batch_size（提升GPU利用率）

| 参数 | 原始值 | 优化值 | 提升 |
|------|--------|--------|------|
| Phase 1 batch | 256 | **128** | 中等batch |
| Phase 2 batch | 256 | **256** | GPU利用率优化 |

### 3️⃣ 增加并行度（加速环境）

| 参数 | 原始值 | 优化值 | 提升 |
|------|--------|--------|------|
| Phase 1 workers | 8 | **8** | 保持 |
| Phase 2 envs | 8 | **16** | ⚡ 2倍并行 |
| Phase 2 n_steps | 4096 | **31** | ⚡ 小rollout快速更新 |

### 4️⃣ 缩短episode长度

| 参数 | 原始值 | 优化值 | 提升 |
|------|--------|--------|------|
| max_steps | 3600秒 (1小时) | **600秒 (1分钟)** | ⚡ 6倍速度 |

## 📈 性能对比

| 配置文件 | 训练时间 | 用途 |
|---------|---------|------|
| `competition_preliminary.yaml` | 2-4小时 | 正式训练 |
| `competition_quick_test.yaml` | 30-60分钟 | 快速验证 |
| **`competition_ultra_fast_test.yaml`** | **5-10分钟** | ⚡ 超快速测试 |

## 🎯 适用场景

### ✅ 适合使用此配置的情况：

1. **代码修改后验证**
   ```bash
   # 修改代码后快速测试
   python train_v4_ideal.py --config configs/competition_ultra_fast_test.yaml --phase all
   ```

2. **CI/CD自动化测试**
   ```bash
   # GitHub Actions / Jenkins
   python train_v4_ideal.py --config configs/competition_ultra_fast_test.yaml --phase all
   ```

3. **环境部署验证**
   ```bash
   # 在新机器上验证环境
   python train_v4_ideal.py --config configs/competition_ultra_fast_test.yaml --phase 1
   ```

4. **功能开发调试**
   ```bash
   # 快速迭代新功能
   python train_v4_ideal.py --config configs/competition_ultra_fast_test.yaml --phase 2
   ```

### ❌ 不适合使用的情况：

1. **正式训练比赛模型** → 使用 `competition_preliminary.yaml`
2. **性能对比测试** → 使用 `competition_quick_test.yaml`
3. **论文实验** → 需要完整训练

## 🔧 分阶段使用

### 只运行Phase 1（数据收集+预训练）
```bash
# 约2-3分钟
python train_v4_ideal.py --config configs/competition_ultra_fast_test.yaml --phase 1
```

### 只运行Phase 2（PPO训练）
```bash
# 约3-4分钟（需要Phase 1完成）
python train_v4_ideal.py --config configs/competition_ultra_fast_test.yaml --phase 2
```

### 只运行Phase 3（端到端微调）
```bash
# 约1分钟（需要Phase 2完成）
python train_v4_ideal.py --config configs/competition_ultra_fast_test.yaml --phase 3
```

### 只运行Phase 4（约束优化）
```bash
# 注意：Phase 4由Phase 3训练器执行
# 运行Phase 3即可，它会自动执行约束优化
```

## 📂 输出文件位置

训练完成后，模型保存在：

```
checkpoints/ultra_fast_test/
├── v4_phase1/
│   └── world_model_final.pth         # Phase 1输出
├── v4_phase2/
│   ├── shielded_ppo.zip              # Phase 2输出（SB3格式）
│   └── shielded_ppo.pth              # Phase 2输出（PyTorch格式）
└── v4_phase3/
    └── ideal_v4_final.zip            # Phase 3输出（最终模型）
```

## 💡 提示和技巧

### 1. 内存不足？

如果遇到内存不足（OOM），调整配置：

```yaml
training:
  phase2:
    num_envs: 8     # 从16减少到8
    batch_size: 256  # 从512减少到256
```

### 2. 想要更快？

如果5-10分钟还是太慢，可以进一步减少：

```yaml
training:
  phase2:
    total_timesteps: 500   # 从1000减少到500
    num_envs: 24          # 从16增加到24
```

### 3. 想要质量更好？

如果担心质量，可以使用快速测试配置：

```bash
# 使用快速测试配置（30-60分钟，质量更好）
python train_v4_ideal.py --config configs/competition_quick_test.yaml --phase all
```

## 🎓 训练流程验证

运行此配置后，您应该看到：

```bash
[PHASE 1] World Model Pre-training
[DONE] Phase 1 complete! (2-3分钟)

[PHASE 2] Shielded PPO Training
[INFO] Using 16 parallel environments
[INFO] Total timesteps: 496 (16 envs × 31 steps × 1 rollout)
[DONE] Phase 2 complete! (约2分钟)

[PHASE 3] Lagrangian Constrained Optimization
[INFO] Using 4 parallel environments
[INFO] Total timesteps: 248 (4 envs × 62 steps × 1 rollout)
[DONE] Phase 3 complete! (约1分钟)

[SUCCESS] Training pipeline finished! (总计约5分钟)
```

## 🚨 常见问题

### Q: 进度条显示的数字是什么意思？

**A:** PPO进度条显示的是**环境步数（environment steps）**，不是策略更新次数：

- **公式**: `显示步数 = num_envs × n_steps × 更新轮数`
- **配置示例**:
  - `num_envs: 16`
  - `n_steps: 31`
  - `total_timesteps: 496`
- **计算**: `16 × 31 = 496` 步/轮次
- **进度条**: 显示 `496/496` → 训练完成

**正常进度显示**:
```
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 496/496 [ 0:00:20 < 0:00:00 , 10 it/s ]
```

如果看到 `1984/1000` 这样的显示，说明配置没有正确读取。修复后应该看到接近 `496/496` 的显示。

### Q: 为什么要增加num_envs和batch_size？

**A:**
- **减少步数** → 减少训练循环次数
- **增加num_envs** → 更多环境并行，收集数据更快
- **增加batch_size** → GPU一次处理更多数据，利用率更高

这样可以在更少的时间内完成足够多的训练步数。

### Q: 质量会不会很差？

**A:** 不会！这个配置只是用来验证流程：
- 验证代码能正常运行
- 验证各阶段的数据流正确
- 验证模型架构无误

如果质量要求高，应该使用完整配置。

### Q: 可以用这个配置参加比赛吗？

**A:** **不可以！** 这个配置仅用于测试，训练严重不足。

参加比赛请使用：
```bash
python train_v4_ideal.py --config configs/competition_preliminary.yaml --phase all
```

## 📋 总结

| 配置 | 时间 | 用途 |
|------|------|------|
| **ultra_fast_test** | ⚡ 5-10分钟 | 快速验证、CI/CD |
| **quick_test** | 📊 30-60分钟 | 功能测试、调试 |
| **preliminary** | 🏋️ 2-4小时 | 正式训练、比赛 |

**推荐工作流：**
1. 修改代码 → `ultra_fast_test` 验证
2. 功能开发 → `quick_test` 测试
3. 准备比赛 → `preliminary` 训练
