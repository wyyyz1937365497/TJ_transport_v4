# 🚀 快速开始指南

## 📌 两个推荐方案

### 方案A：固定环境训练（最快 ⚡）

**8小时完成 | 最简单**

```bash
python train_phase2_stable.py
```

---

### 方案B：多阶段训练（最稳定 🛡️）

**30小时完成 | 渐进式学习**

```bash
train_all_stages.bat
```

---

## 📊 方案对比

| 方案 | 时间 | 稳定性 | 推荐度 |
|------|------|--------|--------|
| 固定环境 | 8h | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 多阶段 | 30h | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## 🔧 高级选项

### 使用 Phase 1 权重

```bash
python train_phase2_stable.py --phase1-checkpoint checkpoints/competition/phase1/world_model_final.pth
```

### 手动多阶段训练

```bash
python train_phase2_stable.py --stage 1
python train_phase2_stable.py --stage 2 --prev-checkpoint checkpoints/competition/curriculum/level1/ppo.zip
python train_phase2_stable.py --stage 3 --prev-checkpoint checkpoints/competition/curriculum/level2/ppo.zip
python train_phase2_stable.py --stage 4 --prev-checkpoint checkpoints/competition/curriculum/level3/ppo.zip
python train_phase2_stable.py --stage 5 --prev-checkpoint checkpoints/competition/curriculum/level4/ppo.zip
```

---

## ❌ 不推荐

```bash
# ❌ 动态课程切换（会导致 BrokenPipeError）
python train.py --phase 2
python train_phase2_stable.py --enable-curriculum
```

---

## 📖 完整文档

- **TRAINING_GUIDE.md** - 完整训练指南
- **PROJECT_STRUCTURE.md** - 项目结构说明

---

**状态**: ✅ 项目已精简
**日期**: 2026-01-16
