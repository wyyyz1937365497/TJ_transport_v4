# 项目结构说明（清理后 v5.0）

## 📁 根目录

```
TJ_transport_v4/
├── 📘 README.md                       # 项目说明
├── 📘 PHASE1_LITE_SUMMARY.md          # v5架构详细文档
├── 📘 PROJECT_CLEANUP.md              # 清理记录
├── 🐍 train_phase1_lite.py            # 主训练脚本
└── 🧪 test_v5_architecture.py         # 测试脚本
```

**根目录特点：** 极简设计，只保留5个核心文件

---

## 📂 核心模块

### 🧠 模型 (`src/models/`)
```
src/models/
├── v5_lightweight.py          # ⭐ v5轻量级OCR-GNN（主模型）
├── ideal_policy_v4.py         # v4理想策略（遗留，向后兼容）
└── v4_architecture.py         # v4架构模块（遗留）
```

### 🎮 环境 (`src/env/`)
```
src/env/
├── competition_env.py         # ⭐ 比赛环境（主环境）
├── gpu_sumo_env_optimized.py  # GPU优化的SUMO环境
├── sparse_controller.py       # ⭐ v5稀疏控制器
├── gym_wrapper.py             # Gym包装器
├── smart_icv_manager.py       # Smart ICV管理器（遗留，可选）
└── vec_env.py                # 向量化环境
```

### 🏋️ 训练 (`src/training/`)
```
src/training/
├── ocr_rewards.py            # ⭐ OCR奖励计算器
├── custom_ppo_trainer.py     # PPO训练器
├── checkpoint_manager.py     # Checkpoint管理
├── train_enhancements.py     # 训练增强功能
├── world_model_train_v4.py   # v4世界模型训练（遗留）
└── multi_gpu_utils.py        # 多GPU工具
```

### 🛠️ 工具 (`src/utils/`)
```
src/utils/
├── helpers.py                # 辅助函数
└── frenet_utils.py           # Frenet坐标系工具
```

---

## ⚙️ 配置文件

```
configs/
├── phase1_lite.yaml          # ⭐ v5轻量级配置（主配置）
└── competition.yaml          # v4比赛配置（遗留）
```

---

## 📚 文档

```
docs/
├── 赛题.md                     # ⭐ 赛题背景和挑战
├── 交通工程赛道-评测公式.md     # ⭐ 官方评测公式
└── 理想架构.md                 # ⭐ 架构设计思路
```

---

## 🎯 核心工作流

### 1️⃣ 训练
```bash
# 完整训练（Stage 1 + Stage 2）
python train_phase1_lite.py --config configs/phase1_lite.yaml --stage all
```

### 2️⃣ 测试
```bash
# 验证架构
python test_v5_architecture.py
```

### 3️⃣ 推理
```python
from src.models.v5_lightweight import create_lightweight_policy_v5

# 加载模型
policy = create_lightweight_policy_v5(obs_dim=321, action_dim=2, config=config)

# 推理
selected_ids, info = policy.select_vehicles(obs)
```

---

## 📊 代码统计

| 模块 | 文件数 | 核心文件 |
|------|--------|----------|
| **模型** | 3个 | v5_lightweight.py |
| **环境** | 6个 | competition_env.py, sparse_controller.py |
| **训练** | 5个 | ocr_rewards.py, custom_ppo_trainer.py |
| **工具** | 2个 | helpers.py |
| **总计** | 19个 | 5个核心文件 |

---

## 🔄 版本对比

### v4 遗留文件（保留但标记为遗留）
- `src/models/ideal_policy_v4.py` - v4理想策略
- `src/models/v4_architecture.py` - v4架构模块
- `src/env/smart_icv_manager.py` - Smart ICV（可选功能）
- `src/training/world_model_train_v4.py` - 世界模型训练
- `configs/competition.yaml` - v4配置

**原因：** 向后兼容，避免破坏现有代码

### v5 核心文件（推荐使用）
- `src/models/v5_lightweight.py` ⭐
- `src/env/sparse_controller.py` ⭐
- `src/training/ocr_rewards.py` ⭐
- `train_phase1_lite.py` ⭐
- `test_v5_architecture.py` ⭐
- `configs/phase1_lite.yaml` ⭐

**原因：** 专为初赛优化，更快、更轻、更高效

---

## 📝 开发指南

### 添加新功能
1. 在`src/models/`中实现模型
2. 在`src/env/`中实现环境
3. 在`train_phase1_lite.py`中集成
4. 在`test_v5_architecture.py`中添加测试

### 代码风格
- Python 3.10+
- PyTorch 2.0+
- 类型提示（Type Hints）
- 文档字符串（Docstrings）

---

**总结：** 项目结构清晰，核心文件突出，便于维护和扩展。
