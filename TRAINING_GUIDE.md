# 🚀 Phase 2 训练指南

## 📅 更新日期
2026-01-16

---

## 📂 项目脚本说明

### 当前训练脚本

| 脚本 | 用途 | 推荐度 |
|------|------|--------|
| **train_phase2_stable.py** | Phase 2 主训练脚本（固定环境 + 多阶段） | ⭐⭐⭐⭐⭐ |
| **train.py** | 完整训练流程（Phase 1 + 2 + 3） | ⭐⭐⭐ |
| **train_all_stages.bat** | 自动运行所有5个课程阶段 | ⭐⭐⭐⭐ |

### 已删除的冗余脚本
为保持项目可维护性，以下脚本已被删除：
- ❌ `train_phase2.py` - 功能已被 `train_phase2_stable.py` 替代
- ❌ `train_curriculum_stages.py` - 功能已被 `train_phase2_stable.py` 替代
- ❌ `train_all_curriculum_stages.bat` - 已被 `train_all_stages.bat` 替代

---

## 🎯 推荐训练方案

### 方案A：固定环境训练（最快 ⚡）

**适用场景**：快速完成训练（8小时）

```bash
python train_phase2_stable.py
```

**特点**：
- ✅ 8小时完成训练
- ✅ 直接在比赛环境（Level 5）
- ✅ 无 BrokenPipeError
- ✅ 最简单

---

### 方案B：多阶段渐进训练（最稳定 🛡️）

**适用场景**：课程学习效果（30小时）

#### 选项1：自动运行
```bash
train_all_stages.bat
```

#### 选项2：手动逐阶段
```bash
python train_phase2_stable.py --stage 1
python train_phase2_stable.py --stage 2 --prev-checkpoint checkpoints/competition/curriculum/level1/ppo.zip
python train_phase2_stable.py --stage 3 --prev-checkpoint checkpoints/competition/curriculum/level2/ppo.zip
python train_phase2_stable.py --stage 4 --prev-checkpoint checkpoints/competition/curriculum/level3/ppo.zip
python train_phase2_stable.py --stage 5 --prev-checkpoint checkpoints/competition/curriculum/level4/ppo.zip
```

**特点**：
- ✅ 渐进式学习（Level 1 → Level 5）
- ✅ 每个阶段独立，无管道问题
- ✅ 30小时完成

---

## 📊 方案对比

| 方案 | 模式 | 时间 | 难度 | 稳定性 | 推荐度 |
|------|------|------|------|--------|--------|
| **方案A** | 固定环境（Level 5） | 8h | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **方案B** | 多阶段渐进（1→5） | 30h | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## ❌ 不推荐

```bash
# ❌ 动态课程切换（会导致 BrokenPipeError）
python train.py --phase 2
python train_phase2_stable.py --enable-curriculum
```

**原因**：Windows SubprocVecEnv 管道问题

---

## 🔧 高级选项

### 使用 Phase 1 权重初始化

```bash
# 固定环境 + Phase 1 权重
python train_phase2_stable.py --phase1-checkpoint checkpoints/competition/phase1/world_model_final.pth

# 多阶段 + Phase 1 权重
python train_phase2_stable.py --stage 1 --phase1-checkpoint checkpoints/competition/phase1/world_model_final.pth
python train_phase2_stable.py --stage 2 --prev-checkpoint checkpoints/competition/curriculum/level1/ppo.zip
# ... 继续阶段3-5
```

### 使用自定义配置

```bash
python train_phase2_stable.py --config configs/custom.yaml
```

---

## 📖 相关文档

- **QUICKSTART.md** - 快速开始指南
- **configs/competition.yaml** - 训练配置文件

---

## 🎯 总结

### 推荐选择

- **快速完成** → 方案A：`python train_phase2_stable.py`
- **稳定且全面** → 方案B：`train_all_stages.bat`
- **完整流程** → `train.py`（Phase 1 + 2 + 3）

### 核心原则

**避免在训练过程中重建环境**

---

**状态**: ✅ 项目已精简
**推荐**: train_phase2_stable.py
**日期**: 2026-01-16
