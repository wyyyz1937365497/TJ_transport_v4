# 训练评估完整指南

**版本**: v1.0
**日期**: 2025-01-21
**适用架构**: v5.0 三阶段训练

---

## 📋 目录

1. [评估概述](#评估概述)
2. [Stage 1 评估：WorldModel](#stage-1-评估worldmodel)
3. [Stage 2 评估：引导探索](#stage-2-评估引导探索)
4. [Stage 3 评估：约束优化](#stage-3-评估约束优化)
5. [评估流程完整示例](#评估流程完整示例)
6. [常见问题与故障排除](#常见问题与故障排除)

---

## 评估概述

### 为什么需要评估？

在训练的每个阶段结束后，**必须进行评估**以确保：

1. **Stage 1**: WorldModel能准确预测交通流状态和风险
2. **Stage 2**: PPO策略能有效提升OCR并控制交通
3. **Stage 3**: 约束优化在保证安全的前提下降低成本

### 评估脚本总览

| 阶段 | 评估脚本 | 关键指标 | 决策阈值 |
|------|---------|---------|---------|
| **Stage 1** | `evaluate_stage1.py` | Flow RMSE, Risk F1 | RMSE < 30, F1 > 0.7 |
| **Stage 2** | `evaluate_stage2.py` | OCR, 速度, 干预率 | OCR > 0.7, 速度 > 15 m/s |
| **Stage 3** | `evaluate_stage3.py` | OCR改善, 成本降低 | OCR提升 > 2%, 干预率降低 |

### 通用使用模式

```bash
python scripts/evaluate_stage<N>.py \
    --config configs/v5_complete.yaml \
    --checkpoint checkpoints/v5_complete/stage<N>_best.pth \
    --num_eval_episodes 20 \
    --visualize \
    --device cuda
```

---

## Stage 1 评估：WorldModel

### 目标

验证WorldModel是否能够：
- 准确预测下一时刻的车辆状态（Flow预测）
- 准确识别碰撞风险（Risk预测）
- 提取有效的交通流表征（Embedding）

### 使用方法

#### 1. 基础评估

```bash
python scripts/evaluate_stage1.py \
    --config configs/v5_complete.yaml \
    --checkpoint checkpoints/v5_complete/stage1_best.pth \
    --num_eval_episodes 20 \
    --device cuda
```

**参数说明**:
- `--config`: 配置文件路径
- `--checkpoint`: Stage 1训练好的WorldModel检查点
- `--num_eval_episodes`: 评估的episode数（默认20）
- `--device`: 设备选择（cuda/cpu）
- `--visualize`: 是否生成可视化图表
- `--skip_training`: 跳过训练，直接评估已有检查点

#### 2. 生成可视化

添加 `--visualize` 参数：

```bash
python scripts/evaluate_stage1.py \
    --config configs/v5_complete.yaml \
    --checkpoint checkpoints/v5_complete/stage1_best.pth \
    --num_eval_episodes 20 \
    --visualize \
    --device cuda
```

**生成的图表**:
- `flow_prediction_scatter.png`: 预测vs实际散点图（6个关键特征）
- `flow_error_bar.png`: 各特征误差条形图
- `risk_confusion_matrix.png`: 风险预测混淆矩阵
- `time_series_prediction.png`: 时间序列预测示例

### 关键指标解读

#### 1. Flow预测指标

| 指标 | 含义 | 优秀 | 良好 | 需改进 |
|------|------|------|------|--------|
| **MSE** | 均方误差（归一化后） | < 0.01 | < 0.05 | > 0.05 |
| **RMSE** | 均方根误差 | < 10 | < 20 | > 20 |
| **MAE** | 平均绝对误差 | < 5 | < 10 | > 10 |
| **R²** | 决定系数 | > 0.95 | > 0.85 | < 0.85 |

**特征级别误差** (在原始尺度上):
- **s (位置)**: RMSE < 50米（优秀），< 100米（良好）
- **speed (速度)**: RMSE < 2 m/s（优秀），< 5 m/s（良好）
- **acceleration (加速度)**: RMSE < 0.5 m/s²（优秀），< 1.0 m/s²（良好）

#### 2. Risk预测指标

| 指标 | 含义 | 优秀 | 良好 | 需改进 |
|------|------|------|------|--------|
| **BCE Loss** | 二元交叉熵损失 | < 0.3 | < 0.5 | > 0.5 |
| **Accuracy** | 准确率 | > 0.90 | > 0.80 | < 0.80 |
| **Precision** | 精确率 | > 0.80 | > 0.60 | < 0.60 |
| **Recall** | 召回率 | > 0.70 | > 0.50 | < 0.50 |
| **F1 Score** | F1分数 | > 0.75 | > 0.60 | < 0.60 |

**注意**: Risk预测通常类别不平衡（绝大多数是安全样本），因此：
- **Accuracy**可能虚高（模型总预测"安全"也能得高准确率）
- **Precision和Recall**更重要，特别是**Recall**（不应漏掉风险）
- **F1 Score**是平衡指标

### 决策标准

#### ✅ 可以进入Stage 2

**必须同时满足**:
1. Flow RMSE < 20（在原始尺度上）
2. Risk F1 Score > 0.60
3. Flow R² > 0.85

**示例输出**:
```
📊 Flow预测评估:
  MSE: 0.0089
  RMSE: 9.43
  MAE: 5.21
  R²: 0.9615
  ✅ Flow预测精度优秀

🎯 Risk预测评估:
  BCE Loss: 0.3245
  Accuracy: 0.8712
  Precision: 0.7234
  Recall: 0.6891
  F1 Score: 0.7058
  ✅ Risk预测精度良好

✅ 决策: WorldModel性能优秀，可以进入Stage 2训练
```

#### ⚠️ 需要改进后重新训练

**如果出现以下情况**:
1. Flow RMSE > 30
2. Risk F1 Score < 0.50
3. 某些特征误差特别大（如位置RMSE > 200米）

**改进措施**:
1. **增加训练数据**: `--num_episodes 50`（从20增加到50）
2. **增加训练轮数**: 修改配置文件 `num_epochs: 50`（从20增加到50）
3. **调整模型容量**: 增加hidden_dim或num_layers
4. **检查归一化**: 确保特征归一化正确应用
5. **数据增强**: 启用noise增加数据多样性

**重新训练命令**:
```bash
# 清除旧缓存（如果修改了数据收集）
rm -rf cache/stage1_trajectories/

# 重新训练
python scripts/train_stage1_world_observer.py \
    --config configs/v5_complete.yaml \
    --num_episodes 50 \
    --force_refresh \
    --device cuda
```

### 常见问题

#### Q1: 为什么Loss很低但RMSE很高？

**原因**: Loss是基于归一化数据计算的，RMSE是反归一化后的原始尺度。

**解决**: 关注原始尺度上的RMSE值，这才是实际预测误差。

#### Q2: Risk预测的Recall很低怎么办？

**原因**: 模型倾向于预测"安全"（类别不平衡）。

**解决**:
1. 增加`risk`损失权重：`loss_weights.risk: 1.0`（从0.5提高）
2. 使用focal loss处理类别不平衡
3. 增加风险样本的数据增强

#### Q3: R²出现负值？

**原因**: 模型预测比简单预测均值还差，严重欠拟合。

**解决**:
1. 检查归一化是否正确
2. 增加模型容量
3. 检查训练是否收敛（Loss是否下降）
4. 增加训练数据

---

## Stage 2 评估：引导探索

### 目标

验证PPO训练的策略是否能够：
- 提升OCR（Output Competition Ratio）
- 提高平均速度（效率）
- 保持安全性（低碰撞率）
- 控制干预成本（低干预率）

### 使用方法

#### 1. 基础评估

```bash
python scripts/evaluate_stage2.py \
    --config configs/v5_complete.yaml \
    --checkpoint checkpoints/v5_complete/stage2_best.pth \
    --num_eval_episodes 20 \
    --device cuda
```

**参数说明**:
- `--checkpoint`: Stage 2训练好的策略检查点
- `--num_eval_episodes`: 评估episode数（默认20）
- `--visualize`: 生成可视化图表
- `--output_dir`: 自定义输出目录

#### 2. 生成可视化

添加 `--visualize` 参数：

```bash
python scripts/evaluate_stage2.py \
    --config configs/v5_complete.yaml \
    --checkpoint checkpoints/v5_complete/stage2_best.pth \
    --num_eval_episodes 20 \
    --visualize \
    --device cuda
```

**生成的图表**:
- `reward_metrics.png`: Reward、OCR、速度、干预率per episode
- `reward_components.png`: 效率、稳定性、成本奖励分解
- `safety_metrics.png`: 碰撞数、急刹车事件统计

### 关键指标解读

#### 1. OCR指标（比赛核心指标）

OCR = Output Competition Ratio，表示完成OD需求的车辆比例。

| OCR范围 | 评级 | 说明 |
|---------|------|------|
| > 0.90 | 🌟 优秀 | 接近最优，90%+车辆完成行程 |
| 0.80-0.90 | ✅ 良好 | 性能不错，80-90%车辆完成行程 |
| 0.70-0.80 | ⚠️ 及格 | 基本可用，70-80%车辆完成行程 |
| < 0.70 | ❌ 需改进 | 性能不足，< 70%车辆完成行程 |

**注意**: OCR是比赛最终评分的核心指标！

#### 2. 效率指标

| 指标 | 含义 | 优秀 | 良好 | 需改进 |
|------|------|------|------|--------|
| **平均速度** | 所有车辆平均速度 | > 20 m/s | > 15 m/s | < 15 m/s |
| **效率奖励** | 效率分量得分 | > 0.8 | > 0.6 | < 0.6 |

**参考**:
- 高速公路设计速度: 25-30 m/s (90-108 km/h)
- 城市快速路: 15-20 m/s (54-72 km/h)
- < 10 m/s (36 km/h) 表示严重拥堵

#### 3. 安全指标

| 指标 | 含义 | 优秀 | 良好 | 需改进 |
|------|------|------|------|--------|
| **碰撞数** | 每episode碰撞次数 | 0 | 0-1 | > 1 |
| **急刹车** | 急刹车事件数 | < 5 | < 20 | > 20 |
| **稳定性奖励** | 稳定性分量得分 | > 0.8 | > 0.6 | < 0.6 |

**目标**: 零碰撞是最高优先级！

#### 4. 成本指标

| 指标 | 含义 | 优秀 | 良好 | 需改进 |
|------|------|------|------|--------|
| **干预率** | 控制的ICV比例 | < 10% | < 30% | > 50% |
| **成本奖励** | 成本分量得分 | > 0.8 | > 0.6 | < 0.6 |

**计算**: 干预率 = 平均控制车辆数 / 总ICV数

**目标**: 尽量少干预，让车辆自主行驶

### 决策标准

#### ✅ 可以进入Stage 3

**必须同时满足**:
1. OCR > 0.70
2. 平均速度 > 15 m/s
3. 碰撞率 < 1.0（每episode）

**示例输出**:
```
📊 奖励指标:
  总Reward: 845.32 ± 45.21

🎯 OCR指标（比赛关键指标）:
  平均OCR: 0.7823 ± 0.0345
  范围: [0.7212, 0.8456]

⚡ 效率指标:
  平均速度: 18.45 ± 2.13 m/s
  效率奖励: 0.7234 ± 0.0567

🛡️ 安全指标:
  总碰撞数: 3
  总急刹车: 47
  稳定性奖励: 0.6812 ± 0.0789

💰 成本指标:
  干预率: 32.45% ± 5.67%
  成本奖励: 0.5891 ± 0.0623

📈 性能评级:
  OCR性能: 良好 ✅
  速度性能: 良好 ✅
  安全性能: 良好 ✅
  干预效率: 良好 ✅

💡 建议:
  ✅ 模型性能良好，可以进入Stage 3约束优化训练
```

#### ⚠️ 需要改进后继续训练Stage 2

**如果出现以下情况**:
1. OCR < 0.70
2. 平均速度 < 15 m/s
3. 碰撞率 > 2.0

**改进措施**:
1. **增加训练迭代**: 修改配置 `num_iterations: 2000`（从1000增加）
2. **调整奖励权重**:
   ```yaml
   rewards:
     weights:
       efficiency: 0.6  # 从0.5提高，更重视效率
       stability: 0.3   # 保持
       cost: 0.1        # 从0.2降低，允许更多干预
   ```
3. **调整学习率**: 尝试3e-4或5e-4
4. **检查收敛**: 查看训练日志，确认reward是否持续上升

**继续训练命令**:
```bash
# 从现有检查点继续训练
python scripts/train_stage2_guided_exploration.py \
    --config configs/v5_complete.yaml \
    --resume checkpoints/v5_complete/stage2_best.pth \
    --device cuda
```

### 常见问题

#### Q1: OCR一直上不去怎么办？

**可能原因**:
1. 训练不充分（迭代次数太少）
2. 奖励权重不合理
3. WorldModel表征质量差

**解决**:
1. 增加训练迭代到2000-3000
2. 提高efficiency奖励权重到0.6-0.7
3. 回去重新训练Stage 1（如果WorldModel质量差）

#### Q2: 干预率太高（>50%）怎么办？

**原因**: 策略过度依赖ICV控制，没有学会让车辆自主行驶。

**解决**:
1. 提高cost奖励权重到0.3-0.4
2. 增加每控制一辆车的惩罚成本
3. 检查SafetyShield是否过度干预

#### Q3: 碰撞率太高怎么办？

**原因**: 稳定性奖励权重太低或策略过于激进。

**解决**:
1. 提高stability奖励权重到0.4-0.5
2. 启用SafetyShield（`execution_mode: inference_only`）
3. 降低学习率，让策略更保守

---

## Stage 3 评估：约束优化

### 目标

验证Stage 3训练的约束策略是否能够：
- 在保持OCR的前提下降低干预成本
- 提升安全性（降低碰撞率）
- 优于Stage 2基线

### 使用方法

#### 1. 评估Stage 3（无对比）

```bash
python scripts/evaluate_stage3.py \
    --config configs/v5_complete.yaml \
    --checkpoint checkpoints/v5_complete/stage3_best.pth \
    --num_eval_episodes 20 \
    --device cuda
```

#### 2. 评估Stage 3并与Stage 2对比

```bash
python scripts/evaluate_stage3.py \
    --config configs/v5_complete.yaml \
    --checkpoint checkpoints/v5_complete/stage3_best.pth \
    --baseline_checkpoint checkpoints/v5_complete/stage2_best.pth \
    --num_eval_episodes 20 \
    --visualize \
    --device cuda
```

**参数说明**:
- `--checkpoint`: Stage 3模型检查点
- `--baseline_checkpoint`: Stage 2基线检查点（用于对比）
- `--visualize`: 生成对比图表

**生成的图表**:
- Stage 2的所有图表（继承）
- `stage2_vs_stage3_comparison.png`: 4个子图对比各指标
- `performance_radar.png`: 雷达图显示性能改善

### 关键指标解读

#### 1. 改善百分比

| 指标 | 改善方向 | 优秀 | 良好 | 需改进 |
|------|---------|------|------|--------|
| **OCR改善** | 越高越好 | > +5% | > +2% | < 0% |
| **干预率降低** | 越低越好 | > -20% | > -10% | > 0% |
| **碰撞数减少** | 越少越好 | > -50% | > -20% | > 0% |
| **速度提升** | 越高越好 | > +10% | > +5% | < 0% |

**注意**:
- 正值表示改善（提升或增加）
- 负值表示改善（降低或减少）
- 对于干预率和碰撞数，负值是好的！

#### 2. 约束满足

Stage 3引入成本约束，需要验证：

| 指标 | 目标 | 优秀 | 良好 | 需改进 |
|------|------|------|------|--------|
| **约束违反率** | 成本超过阈值的比例 | < 5% | < 15% | > 15% |
| **λ (Lambda)** | 拉格朗日乘数 | 稳定 | 收敛 | 发散 |

**Lambda趋势**:
- **下降**: 约束逐渐满足
- **稳定**: 已找到最优平衡点
- **上升**: 约束持续违反，需要更强的惩罚

### 决策标准

#### ✅ Stage 3训练成功

**必须同时满足**:
1. OCR相对于Stage 2提升 > 2%（或持平±1%）
2. 干预率降低 > 5%
3. 碰撞率不增加（或降低）
4. 约束违反率 < 15%

**示例输出**:
```
Stage 2 vs Stage 3 性能对比

📊 OCR对比:
  Stage 2: 0.7823
  Stage 3: 0.8012
  改善: +0.0189 (+2.42%)
  ✅ OCR提升

⚡ 平均速度对比:
  Stage 2: 18.45 m/s
  Stage 3: 18.67 m/s
  改善: +0.22 m/s (+1.19%)

💰 干预率对比:
  Stage 2: 32.45%
  Stage 3: 28.12%
  降低: -4.33% (-13.34%)
  ✅ 干预率降低（更好的成本控制）

🛡️ 安全性对比:
  Stage 2碰撞数: 3
  Stage 3碰撞数: 1
  减少: +2 (+66.67%)
  ✅ 碰撞数减少

📈 总体评价:
  🌟 优秀：Stage 3在所有关键指标上都有提升
```

#### ⚠️ Stage 3需要改进

**如果出现以下情况**:
1. OCR下降 > 2%
2. 干预率反而上升
3. 碰撞率增加

**改进措施**:
1. **调整成本阈值**: `cost_threshold: 0.3`（从0.5降低）
2. **调整lambda学习率**: `lambda_lr: 0.005`（从0.01降低）
3. **增加训练迭代**: 继续训练让lambda收敛
4. **检查CostCritic**: 确保成本预测准确

**继续训练命令**:
```bash
# 从现有检查点继续训练
python scripts/train_stage3_constrained.py \
    --config configs/v5_complete.yaml \
    --resume checkpoints/v5_complete/stage3_best.pth \
    --device cuda
```

### 常见问题

#### Q1: Stage 3的OCR比Stage 2下降了？

**原因**: 约束过强，限制了策略的探索空间。

**解决**:
1. 放宽成本阈值：`cost_threshold: 0.7`（从0.5提高）
2. 降低lambda学习率：`lambda_lr: 0.005`
3. 减小cost_loss_weight：从0.5降到0.3

#### Q2: Lambda一直上升不收敛？

**原因**: 成本约束持续违反，优化器不断加强惩罚。

**解决**:
1. 检查成本定义是否合理
2. 增加成本阈值
3. 启用adaptive_lambda：`adaptive_lambda: true`

#### Q3: 干预率没有下降？

**原因**: DynamicWeightGate没有学到有效的权重分配。

**解决**:
1. 检查DynamicWeightGate是否启用
2. 增加global_dim或hidden_dim
3. 调整temperature参数：`temperature: 0.5`（从1.0降低，让门控更尖锐）

---

## 评估流程完整示例

### 场景1: 从零开始完整训练

```bash
# Step 1: 训练Stage 1
python scripts/train_stage1_world_observer.py \
    --config configs/v5_complete.yaml \
    --num_episodes 20 \
    --force_refresh \
    --device cuda

# Step 2: 评估Stage 1
python scripts/evaluate_stage1.py \
    --config configs/v5_complete.yaml \
    --checkpoint checkpoints/v5_complete/stage1_best.pth \
    --num_eval_episodes 20 \
    --visualize \
    --device cuda

# 检查评估结果：
# - 如果 RMSE < 20 且 F1 > 0.60，继续
# - 否则，改进Stage 1并重新训练

# Step 3: 训练Stage 2（假设Stage 1通过）
python scripts/train_stage2_guided_exploration.py \
    --config configs/v5_complete.yaml \
    --resume checkpoints/v5_complete/stage1_best.pth \
    --device cuda

# Step 4: 评估Stage 2
python scripts/evaluate_stage2.py \
    --config configs/v5_complete.yaml \
    --checkpoint checkpoints/v5_complete/stage2_best.pth \
    --num_eval_episodes 20 \
    --visualize \
    --device cuda

# 检查评估结果：
# - 如果 OCR > 0.70 且速度 > 15，继续
# - 否则，改进Stage 2并继续训练

# Step 5: 训练Stage 3（假设Stage 2通过）
python scripts/train_stage3_constrained.py \
    --config configs/v5_complete.yaml \
    --resume checkpoints/v5_complete/stage2_best.pth \
    --device cuda

# Step 6: 评估Stage 3（对比Stage 2）
python scripts/evaluate_stage3.py \
    --config configs/v5_complete.yaml \
    --checkpoint checkpoints/v5_complete/stage3_best.pth \
    --baseline_checkpoint checkpoints/v5_complete/stage2_best.pth \
    --num_eval_episodes 20 \
    --visualize \
    --device cuda

# 检查评估结果：
# - 如果 OCR提升 > 2% 且干预率降低，训练成功！
# - 否则，改进Stage 3并继续训练
```

### 场景2: 快速验证已有模型

```bash
# 快速评估Stage 1（5 episodes，无可视化）
python scripts/evaluate_stage1.py \
    --config configs/v5_complete.yaml \
    --checkpoint checkpoints/v5_complete/stage1_best.pth \
    --num_eval_episodes 5 \
    --device cuda

# 快速评估Stage 2（5 episodes，无可视化）
python scripts/evaluate_stage2.py \
    --config configs/v5_complete.yaml \
    --checkpoint checkpoints/v5_complete/stage2_best.pth \
    --num_eval_episodes 5 \
    --device cuda

# 快速评估Stage 3（5 episodes，无可视化）
python scripts/evaluate_stage3.py \
    --config configs/v5_complete.yaml \
    --checkpoint checkpoints/v5_complete/stage3_best.pth \
    --num_eval_episodes 5 \
    --device cuda
```

### 场景3: 生成完整评估报告

```bash
# 完整评估Stage 1（20 episodes + 可视化）
python scripts/evaluate_stage1.py \
    --config configs/v5_complete.yaml \
    --checkpoint checkpoints/v5_complete/stage1_best.pth \
    --num_eval_episodes 20 \
    --visualize \
    --output_dir logs/evaluation_report/stage1 \
    --device cuda

# 完整评估Stage 2（20 episodes + 可视化）
python scripts/evaluate_stage2.py \
    --config configs/v5_complete.yaml \
    --checkpoint checkpoints/v5_complete/stage2_best.pth \
    --num_eval_episodes 20 \
    --visualize \
    --output_dir logs/evaluation_report/stage2 \
    --device cuda

# 完整评估Stage 3（20 episodes + 可视化 + 对比）
python scripts/evaluate_stage3.py \
    --config configs/v5_complete.yaml \
    --checkpoint checkpoints/v5_complete/stage3_best.pth \
    --baseline_checkpoint checkpoints/v5_complete/stage2_best.pth \
    --num_eval_episodes 20 \
    --visualize \
    --output_dir logs/evaluation_report/stage3 \
    --device cuda

# 生成评估报告索引
cat > logs/evaluation_report/README.md << 'EOF'
# 评估报告

## Stage 1: WorldModel
- [评估结果](stage1/evaluation_results_*.json)
- [可视化图表](stage1/plots_*/)

## Stage 2: 引导探索
- [评估结果](stage2/evaluation_results_*.json)
- [可视化图表](stage2/plots_*/)

## Stage 3: 约束优化
- [评估结果](stage3/stage3_results_*.json)
- [对比报告](stage3/comparison_*.json)
- [可视化图表](stage3/plots_*/)
EOF
```

---

## 常见问题与故障排除

### Q1: 评估时出现CUDA内存不足

**错误**:
```
RuntimeError: CUDA out of memory
```

**解决**:
```bash
# 方案1: 使用CPU
--device cpu

# 方案2: 减少batch size或eval episodes
--num_eval_episodes 10

# 方案3: 清理GPU缓存
import torch
torch.cuda.empty_cache()
```

### Q2: 评估脚本找不到模型文件

**错误**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'checkpoints/v5_complete/stage1_best.pth'
```

**解决**:
1. 检查路径是否正确
2. 确认训练已完成并保存了检查点
3. 使用绝对路径而非相对路径

```bash
# 检查检查点是否存在
ls -lh checkpoints/v5_complete/

# 使用绝对路径
--checkpoint /home/wyyyz/TJ_transport_v4/checkpoints/v5_complete/stage1_best.pth
```

### Q3: 可视化图表生成失败

**错误**:
```
ValueError: supply at least one argument
```

**原因**: matplotlib后端问题

**解决**:
```bash
# 设置matplotlib后端
export MPLBACKEND=Agg

# 或在评估脚本开头添加
import matplotlib
matplotlib.use('Agg')
```

### Q4: 评估结果JSON文件无法打开

**错误**:
```
JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

**原因**: JSON序列化失败（numpy类型未转换）

**解决**: 评估脚本已内置`convert_to_serializable`函数，如果仍有问题：

```python
# 手动转换numpy类型
import json
import numpy as np

def convert(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert(item) for item in obj]
    else:
        return obj

with open('results.json', 'w') as f:
    json.dump(convert(results), f, indent=2)
```

### Q5: 评估时间太长

**问题**: 评估一个episode需要几分钟，20个episodes要1小时+

**原因**: 仿真正常运行需要时间（1小时仿真时间 ≈ 实际几十秒到几分钟）

**解决**:
```bash
# 方案1: 减少eval episodes
--num_eval_episodes 10  # 从20减少到10

# 方案2: 使用更短的episode
# 修改环境配置中的max_steps
environment:
  max_steps: 18000  # 从36000减半（30分钟仿真）

# 方案3: 并行评估（多GPU）
# 修改评估脚本支持多GPU并行
```

### Q6: Stage 2/3评估时reward全为0或异常

**错误**:
```
Total Reward: 0.00
Mean OCR: 0.0000
```

**原因**: 奖励计算器未正确工作或环境返回异常

**解决**:
1. 检查环境是否正确返回`reward_components`
2. 检查OCRRewardComputer是否正确初始化
3. 查看环境step返回的info字段

```python
# 调试：在评估循环中打印
print(f"Step {step}: reward={reward}, info={info}")
```

### Q7: Stage 3对比时episode数不一致

**错误**:
```
ValueError: shape mismatch: objects cannot be broadcast to a single shape
```

**原因**: Stage 2和Stage 3评估的episode数不同

**解决**:
```bash
# 确保两个评估使用相同的num_eval_episodes
--num_eval_episodes 20

# 或在代码中处理不一致
min_episodes = min(len(stage2_results), len(stage3_results))
```

---

## 附录：评估检查清单

### Stage 1 评估检查清单

- [ ] Flow RMSE < 20（原始尺度）
- [ ] Flow R² > 0.85
- [ ] Risk F1 Score > 0.60
- [ ] Risk Recall > 0.50（不漏掉风险）
- [ ] 无NaN或Inf值
- [ ] 可视化图表生成成功
- [ ] 所有特征误差在合理范围内

### Stage 2 评估检查清单

- [ ] OCR > 0.70
- [ ] 平均速度 > 15 m/s
- [ ] 碰撞率 < 1.0
- [ ] 干预率 < 50%
- [ ] 总Reward > 0
- [ ] 各奖励分量合理（效率、稳定性、成本）
- [ ] 训练收敛（Reward曲线平稳）

### Stage 3 评估检查清单

- [ ] OCR相对Stage 2提升 > 2%（或持平）
- [ ] 干预率降低 > 5%
- [ ] 碰撞率不增加
- [ ] 约束违反率 < 15%
- [ ] Lambda收敛（不持续上升）
- [ ] 对比图表显示明显改善
- [ ] 所有关键指标优于或持平Stage 2

---

## 总结

评估是训练流程中**不可或缺**的一环。通过三个阶段的评估脚本，你可以：

1. **Stage 1**: 确保WorldModel学习到有效的交通流表征
2. **Stage 2**: 验证PPO策略能有效优化OCR和交通流
3. **Stage 3**: 确认约束优化在安全性和成本之间找到最优平衡

**记住**: 每个阶段评估通过后，才能进入下一阶段训练。如果评估不通过，必须改进当前阶段或重新训练。

---

**文档版本**: v1.0
**最后更新**: 2025-01-21
**作者**: Claude Code
