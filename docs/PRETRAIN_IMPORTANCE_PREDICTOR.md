# ImportancePredictor预训练指南

## 概述

预训练ImportancePredictor可以结合规则选择器的优势（稳定的ICV选择）和神经网络的学习能力（自适应优化）。

## 训练流程

### 第1步：收集数据（使用规则选择器）

```bash
python scripts/collect_selection_data.py \
    --config configs/v5_complete.yaml \
    --num_episodes 100 \
    --output_dir data/icv_selection
```

**参数说明**：
- `--num_episodes`: 收集的episode数（建议100-200）
- `--output_dir`: 数据输出目录

**输出文件**：
- `data/icv_selection/observations.npy`: 车辆观测 [N, 32, 9]
- `data/icv_selection/selections.npy`: ICV选择标签 [N, 32]
- `data/icv_selection/metadata.json`: 元数据

**预计时间**：约30-60分钟（100 episodes）

---

### 第2步：预训练ImportancePredictor

```bash
python scripts/pretrain_importance_predictor.py \
    --config configs/v5_complete.yaml \
    --data_dir data/icv_selection \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-3
```

**参数说明**：
- `--epochs`: 训练轮数（默认50）
- `--batch_size`: Batch size（默认32）
- `--lr`: 学习率（默认1e-3）
- `--val_split`: 验证集比例（默认0.2）

**输出文件**：
- `checkpoints/v5_complete/importance_predictor_pretrained.pth`

**预计时间**：约10-20分钟（50 epochs）

**目标指标**：
- Top-K选择准确率 > 0.70（70%+）

---

### 第3步：Stage 2训练（加载预训练权重）

```bash
python scripts/train_stage2_guided_exploration.py \
    --config configs/v5_complete.yaml \
    --resume checkpoints/v5_complete/stage1_best.pth \
    --pretrain_importance checkpoints/v5_complete/importance_predictor_pretrained.pth
```

**新增参数**：
- `--pretrain_importance`: 预训练权重路径

---

## 预期效果

### 对比三种方案

| 方案 | Mean Return | Mean Reward | 训练稳定性 | 收敛速度 |
|------|-------------|-------------|-----------|---------|
| **随机初始化** | ~142 | ~1.45 | 中等 | 慢 |
| **规则选择器** | ~118 | ~1.20 | 稳定（但不学习） | N/A |
| **预训练ImportancePredictor** | **~160+** | **~1.60+** | **高** | **快** |

### 预训练的优势

1. **快速收敛**
   - ImportancePredictor已经知道如何选择重要的车辆
   - PPO只需微调，不需要从头学习

2. **更好的性能**
   - 结合了规则的先验知识
   - 保留了神经网络的适应能力
   - 预期Mean Return提升10-20%

3. **训练稳定**
   - 初始策略已经较好
   - Loss曲线更平滑
   - 更少exploration noise

---

## 完整训练脚本示例

```bash
#!/bin/bash

# 第1步：收集数据（100 episodes，约30分钟）
echo "Step 1: 收集训练数据..."
python scripts/collect_selection_data.py \
    --config configs/v5_complete.yaml \
    --num_episodes 100 \
    --output_dir data/icv_selection

# 第2步：预训练ImportancePredictor（50 epochs，约15分钟）
echo "Step 2: 预训练ImportancePredictor..."
python scripts/pretrain_importance_predictor.py \
    --config configs/v5_complete.yaml \
    --data_dir data/icv_selection \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-3

# 第3步：Stage 2训练（加载预训练权重，100 iterations，约6-8小时）
echo "Step 3: Stage 2 PPO训练..."
python scripts/train_stage2_guided_exploration.py \
    --config configs/v5_complete.yaml \
    --resume checkpoints/v5_complete/stage1_best.pth \
    --pretrain_importance checkpoints/v5_complete/importance_predictor_pretrained.pth

echo "训练完成！"
```

---

## 监控训练

### 关键指标

**预训练阶段**：
- `Train Acc`: Top-K选择准确率（应该>0.70）
- `Val Acc`: 验证集准确率
- `Train Loss`: BCE Loss（应该下降）

**Stage 2训练**：
- `Mean Return`: 应该从150+开始（比随机初始化的142高）
- `Mean Reward`: 应该>1.50
- `Entropy`: 应该逐渐下降（策略越来越确定）

### 期望的训练曲线

```
预训练：
Train Acc: 0.65 → 0.75 → 0.80 → 0.82 (plateau)

Stage 2：
Mean Return: 150 → 155 → 162 → 168 → 170 (plateau)
Mean Reward: 1.50 → 1.55 → 1.60 → 1.62
Entropy: 14.2 → 13.5 → 12.8 → 12.0
```

---

## 故障排查

### 问题1：预训练准确率<0.60

**原因**：
- 数据质量不好（规则选择器不合理）
- 数据量太少

**解决**：
- 增加num_episodes到200+
- 检查规则选择器的权重配置

### 问题2：Stage 2 Mean Return没有提升

**原因**：
- 预训练权重没有正确加载
- ImportancePredictor被PPO破坏了

**解决**：
- 检查日志确认"ImportancePredictor权重已加载"
- 冻结ImportancePredictor参数（不更新）

### 问题3：训练速度慢

**解决**：
- 减少num_episodes（50个足够）
- 减少预训练epochs（20-30即可）
- 使用更小的batch_size

---

## 下一步

训练完成后，运行评估：

```bash
python scripts/evaluate_stage2.py \
    --config configs/v5_complete.yaml \
    --checkpoint checkpoints/v5_complete/stage2_best.pth \
    --num_eval_episodes 20
```

对比三种方案的性能，选择最佳的checkpoint提交。
