# 项目清理记录

## 📅 清理时间
2026-01-19

## 🎯 清理目标
移除v4架构的过时文件，保持项目可维护性，专注于v5轻量级架构。

---

## 🗑️ 已删除文件（共13个）

### 1. 过时的训练脚本（4个）
- ❌ `train_preliminary.py` - 旧的多阶段训练脚本
- ❌ `train_phase1.py` - 旧的世界模型预训练（v4架构）
- ❌ `train_phase2.py` - 旧的PPO训练脚本（v4架构）
- ❌ `train_phase3.py` - 旧的微调脚本（v4架构）

**替代方案：** `train_phase1_lite.py` （v5轻量级架构）

### 2. 过时的评估脚本（2个）
- ❌ `evaluate_preliminary.py` - 针对preliminary的评估
- ❌ `evaluate_v4_ideal.py` - 针对v4架构的评估

**替代方案：** 待实现 v5 专用评估脚本

### 3. 临时测试脚本（3个）
- ❌ `test_icv_ratio.py` - ICV比例测试
- ❌ `test_obs_space.py` - 观测空间测试
- ❌ `test_smart_icv.py` - Smart ICV测试

**替代方案：** `test_v5_architecture.py` （v5综合测试）

### 4. 过时的文档（4个）
- ❌ `PRELIMINARY_README.md` - Preliminary阶段说明
- ❌ `OPTIMIZATION_SUMMARY.md` - 旧版本的优化总结
- ❌ `MODEL_ARCHITECTURE_EVALUATION.md` - v4架构评估
- ❌ `SMART_ICV_IMPLEMENTATION.md` - Smart ICV实现文档

**替代方案：** `PHASE1_LITE_SUMMARY.md` （v5架构文档）

---

## ✅ 保留的核心文件

### 训练脚本
- ✅ `train_phase1_lite.py` - **主要训练脚本**（v5轻量级架构）

### 测试脚本
- ✅ `test_v5_architecture.py` - **主要测试脚本**（v5架构验证）

### 文档
- ✅ `README.md` - 项目说明
- ✅ `PHASE1_LITE_SUMMARY.md` - **v5架构详细文档**
- ✅ `docs/赛题.md` - 赛题说明
- ✅ `docs/交通工程赛道-评测公式.md` - 评测公式
- ✅ `docs/理想架构.md` - 架构设计思路

### 核心代码
- ✅ `src/models/v5_lightweight.py` - v5轻量级GNN
- ✅ `src/training/ocr_rewards.py` - OCR奖励计算
- ✅ `src/env/sparse_controller.py` - 稀疏控制器
- ✅ `src/env/competition_env.py` - 仿真环境（保留，向后兼容）

---

## 📊 清理前后对比

| 类型 | 清理前 | 清理后 | 减少 |
|------|--------|--------|------|
| **训练脚本** | 5个 | 1个 | -80% |
| **评估脚本** | 2个 | 0个* | -100% |
| **测试脚本** | 4个 | 1个 | -75% |
| **根目录文档** | 6个 | 2个 | -67% |
| **总文件数** | 17个 | 4个 | **-76%** |

*注：v5评估脚本待实现

---

## 🔧 迁移指南

### 从v4迁移到v5

**1. 训练命令**
```bash
# 旧方式（v4）
python train_phase1.py  # Phase 1: 世界模型预训练
python train_phase2.py  # Phase 2: PPO训练

# 新方式（v5）
python train_phase1_lite.py --stage all  # 一次性完成2阶段训练
```

**2. 测试命令**
```bash
# 旧方式（v4）
python test_icv_ratio.py
python test_obs_space.py

# 新方式（v5）
python test_v5_architecture.py  # 综合测试所有模块
```

**3. 配置文件**
```yaml
# 旧配置（v4）
configs/competition.yaml  # 复杂的4阶段配置

# 新配置（v5）
configs/phase1_lite.yaml  # 简化的2阶段配置
```

---

## 📝 遗留功能

### Smart ICV Manager
- **状态：** 保留但标记为遗留
- **位置：** `src/env/smart_icv_manager.py`
- **原因：** 被`competition_env.py`引用（可选功能）
- **配置：** `configs/competition.yaml` 中默认关闭
- **建议：** v5架构使用`sparse_controller.py`替代

---

## 🎯 下一步

1. ✅ 清理完成
2. ⏳ 实现v5专用评估脚本
3. ⏳ 更新README.md说明v5架构
4. ⏳ 创建快速开始指南

---

**总结：** 通过清理过时文件，项目结构更加清晰，专注于v5轻量级架构，可维护性大幅提升。
