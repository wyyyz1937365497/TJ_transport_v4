# 改进日志 (CHANGELOG)

本文档记录项目的所有重要改进和变更。

---

## [Unreleased] - 2025-01-21

### 🌟 重大改进：特征表示优化

#### Stage 1 WorldModel 特征改进

**改进内容**：

- 移除冗余特征：`speed`（可由vs, vd计算）
- 移除无意义特征：`angle`（直路场景）
- 移除低信息特征：`in_bottleneck`, `lane_index`

**新增特征**：

- ✅ `leader_gap`: 前车距离（归一化 [0, 1]）
- ✅ `leader_speed_diff`: 前车速度差（归一化 [-1, 1]）
- ✅ `road_type`: 道路类型编码（E1=0.25, E2=0.50, E3=0.75, E5=1.00）
- ✅ `lane_position`: 车道内位置（归一化 [0, 1]）

**性能提升**：

- RMSE: 1.06 → **1.0158** (-4.3%) ✅
- leader_speed_diff MAE: 提升 **78%** 🌟🌟🌟
- 风险预测 F1: 0.60 → 0.6052 (+1%)
- R² Score: 0.23 → 0.2249 (稳定)

**技术细节**：

- 使用 SUMO TraCI API 提取前车信息
- 在数据收集时实时归一化，减少后处理开销
- 评估脚本已更新适配新特征

**文件变更**：

- `scripts/train_stage1_world_observer.py`: 数据收集逻辑更新
- `scripts/evaluate_stage1.py`: 特征尺度和可视化更新

---

### 🐛 Bug 修复

#### 1. R² Score 计算错误

- **问题**: 原始计算使用 `r2 = max(0, 1 - mse)`，不符合统计学定义
- **修复**: 实现正确的 R² 公式 `R² = 1 - SS_res / SS_tot`
- **影响**: 评估结果更准确

#### 2. 除零错误

- **问题**: `normalize_vehicles()` 函数可能除以零
- **修复**: 添加 epsilon 保护，避免除零
- **文件**: `scripts/train_stage1_world_observer.py`

#### 3. 学习率过高

- **问题**: Stage 1 初始学习率 1e-2 导致训练不稳定
- **修复**: 降低至 1e-3，并添加余弦退火
- **效果**: 训练更稳定，收敛更平滑

---

### 📈 配置优化

#### v5_complete.yaml 更新

- `hidden_dim`: 64 → 128（增加模型容量）
- `num_epochs`: 20 → 50（更充分训练）
- `learning_rate`: 添加自适应调度
- `early_stopping.patience`: 10 epochs

---

## [v5.0] - 2025-01-14

### 🎉 初赛/复赛双架构

#### 轻量级架构（初赛专用）

- **文件**: `src/models/v5_lightweight.py`
- **特点**:
  - 直接优化 OCR 奖励
  - 稀疏控制（5-10% 车辆）
  - 移除世界模型简化架构
  - 训练时间 6-8 小时

#### 完整架构（复赛专用）

- **文件**: `src/models/joint_icv_policy.py`
- **三阶段训练**:
  1. Stage 1: World Observer（世界观察者）
  2. Stage 2: Guided Exploration（引导探索）
  3. Stage 3: Constrained Optimization（约束优化）

---

### 🧠 核心组件

#### 1. RiskSensitiveGNN

- TTC（Time-To-Collision）感知注意力
- 异构图建模车辆交互
- 邻接矩阵动态更新

#### 2. ImportancePredictor

- 端到端可学习的车辆重要性评分
- 4层 MLP + GAT
- 输出 [0, 1] 重要性分数

#### 3. SparseGate

- Gumbel-Softmax 可微分 Top-K 选择
- 动态 K 值机制（10-15% 车辆）
- 温度参数自适应

#### 4. HierarchicalPooling

- 4层聚合：Vehicle→Lane→Section→Global
- 注意力机制池化
- 保留空间结构信息

#### 5. WorldModel

- RSSM（Recurrent State Space Model）
- 双头预测：流演化 + 风险演化
- GRU 隐藏状态传递

#### 6. CostCritic

- 预测干预成本
- MSE 损失训练
- 辅助拉格朗日松弛

#### 7. DynamicWeightGate

- 元学习动态权重调整
- 3层 MLP
- 软约束（β=0.1）

#### 8. SafetyShield

- Level 1: 物理限制裁剪
- Level 2: TTC 检查强制制动
- CPU 执行避免 GPU 传输开销

---

### 📝 文档完善

#### 新增文档

- `FEATURE_IMPROVEMENT_PLAN.md`: 特征改进计划
- `FEATURE_COMPARISON_RESULTS.md`: 特征对比结果
- `STAGE1_IMPROVEMENT_STATUS.md`: Stage 1 改进状态
- `STAGE2_READINESS_CHECKLIST.md`: Stage 2 准备清单

#### 更新文档

- `README.md`: 更新特征改进说明
- `docs/training_guide.md`: 添加新特征说明
- 配置文件注释：添加详细参数说明

---

## [v4.x] - 2024-12 ~ 2025-01

### 迭代历史

#### v4.9

- ICV 评分系统优化
- 联合 PPO 架构设计
- 三阶段课程学习框架

#### v4.5

- 世界模型集成
- RSSM 实现
- 风险预测模块

#### v4.0

- 基础 PPO 框架
- SUMO 环境集成
- OCR 奖励函数

---

## 版本命名规则

- **v5.x**: 稳定版本，用于初赛/复赛
- **v4.x**: 开发版本，实验性功能
- **[Unreleased]**: 当前开发中

---

**最后更新**: 2025-01-21
**维护者**: 项目团队
