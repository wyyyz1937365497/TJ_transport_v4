# 📁 项目结构说明

## 📂 根目录文件

### 🔧 训练脚本（3个）

| 文件 | 说明 | 推荐度 |
|------|------|--------|
| **train_phase2_stable.py** | Phase 2 主训练脚本<br>• 固定环境训练（8小时）<br>• 多阶段训练（30小时）<br>• 融合了所有 Phase 2 功能 | ⭐⭐⭐⭐⭐ |
| **train.py** | 完整训练流程<br>• Phase 1: 世界模型训练<br>• Phase 2: PPO 训练<br>• Phase 3: 约束优化 | ⭐⭐⭐ |
| **train_all_stages.bat** | 自动运行脚本<br>• 依次执行所有5个阶段<br>• 错误检测和中断处理 | ⭐⭐⭐⭐ |

### 📖 文档（3个）

| 文件 | 说明 |
|------|------|
| **TRAINING_GUIDE.md** | 完整训练指南 |
| **QUICKSTART.md** | 快速开始指南 |
| **PROJECT_STRUCTURE.md** | 本文档 |

### ⚙️ 配置文件

| 文件 | 说明 |
|------|------|
| **configs/competition.yaml** | 训练配置（环境参数、超参数等） |

---

## 🗂️ src/ 目录结构

```
src/
├── env/                    # 环境相关
│   ├── gpu_sumo_env.py              # GPU加速环境
│   ├── gpu_sumo_env_optimized.py    # 优化版GPU环境
│   ├── competition_env.py           # 比赛环境包装
│   └── vec_env.py                   # 向量化环境创建
│
├── models/                 # 模型相关
│   ├── ideal_policy_v4.py           # Phase 2 策略网络
│   ├── world_model.py               # Phase 1 世界模型
│   └── ideal_controller.py          # 控制器
│
├── training/               # 训练相关
│   ├── train_enhancements.py        # 训练增强管理器
│   └── multi_gpu_utils.py           # 多GPU工具
│
└── utils/                  # 工具函数
    ├── helpers.py                   # 辅助函数
    └── metrics.py                   # 评估指标
```

---

## 📦 checkpoints/ 目录结构

```
checkpoints/
└── competition/
    ├── phase1/
    │   └── world_model_final.pth            # Phase 1 世界模型
    │
    ├── phase2/
    │   └── shielded_ppo.zip                 # Phase 2 固定环境模型
    │
    └── curriculum/                          # 多阶段训练检查点
        ├── level1/ppo.zip                   # 基础场景
        ├── level2/ppo.zip                   # 中等流量
        ├── level3/ppo.zip                   # 高流量场景
        ├── level4/ppo.zip                   # 极端场景
        └── level5/ppo.zip                   # 赛题场景（同phase2）
```

---

## 🎯 使用场景指南

### 场景1：只想训练 Phase 2

```bash
# 快速训练（8小时）
python train_phase2_stable.py

# 或使用 Phase 1 权重
python train_phase2_stable.py --phase1-checkpoint checkpoints/competition/phase1/world_model_final.pth
```

### 场景2：想要渐进式课程学习

```bash
# 自动运行所有阶段
train_all_stages.bat

# 或手动逐阶段
python train_phase2_stable.py --stage 1
python train_phase2_stable.py --stage 2 --prev-checkpoint checkpoints/competition/curriculum/level1/ppo.zip
# ... 继续阶段3-5
```

### 场景3：完整训练流程

```bash
# Phase 1: 世界模型训练
python train.py --phase 1

# Phase 2: PPO 训练
python train.py --phase 2

# Phase 3: 约束优化
python train.py --phase 3
```

---

## 🔄 脚本关系图

```
train.py (完整流程)
    ├── Phase 1: World Model
    ├── Phase 2: PPO ──────────────┐
    └── Phase 3: Constrained Opt.   │
                                  │
train_phase2_stable.py ───────────┘
    ├── 方案A: 固定环境 (Level 5)
    └── 方案B: 多阶段 (Level 1→5)
         └── train_all_stages.bat (自动化)
```

---

## 🗑️ 已删除的冗余文件

为保持项目可维护性，以下文件已被删除：

### 训练脚本
- ❌ `train_phase2.py` - 功能已被 `train_phase2_stable.py` 替代
- ❌ `train_curriculum_stages.py` - 功能已被 `train_phase2_stable.py` 替代

### 批处理脚本
- ❌ `train_all_curriculum_stages.bat` - 已被 `train_all_stages.bat` 替代

### 旧文档
- ❌ `PHASE2_STABLE_SOLUTION.md` - 已合并到 `TRAINING_GUIDE.md`
- ❌ `MERGE_SUMMARY.md` - 已合并到 `TRAINING_GUIDE.md`
- ❌ `TRAIN_PHASE2_GUIDE.md` - 已合并到 `TRAINING_GUIDE.md`

---

## ✅ 精简后的项目优势

1. **单一入口**：`train_phase2_stable.py` 覆盖所有 Phase 2 训练需求
2. **清晰文档**：3个主要文档，职责明确
3. **自动化**：批处理脚本支持一键运行
4. **可维护**：删除冗余文件，降低维护成本

---

**状态**: ✅ 项目已精简
**日期**: 2026-01-16
