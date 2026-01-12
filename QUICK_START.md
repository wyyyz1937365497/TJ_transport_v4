# 快速开始命令

## 🚀 快速测试（5-10分钟）

```bash
# 一键快速测试（推荐）
python train_v4_ideal.py --config configs/competition_quick_test.yaml --phase all
```

## 🏋️ 完整训练（2-4小时）

```bash
# 一键完整训练
python train_v4_ideal.py --config configs/competition_preliminary.yaml --phase all
```

## 📊 评估模型

```bash
# 快速评估
python evaluate_v4_ideal.py \
    --config configs/competition_preliminary.yaml \
    --model checkpoints/competition/final_model_competition.pth \
    --episodes 10

# 生成比赛提交文件
python evaluate_v4_ideal.py \
    --config configs/competition_preliminary.yaml \
    --model checkpoints/competition/final_model_competition.pth \
    --episodes 10 \
    --save_xlsx
```

---

## 分阶段训练

### Phase 1: 世界模型预训练（30-45分钟）

```bash
python train_v4_ideal.py --config configs/competition_preliminary.yaml --phase 1
```

### Phase 2: PPO训练（60-90分钟）

```bash
python train_v4_ideal.py --config configs/competition_preliminary.yaml --phase 2
```

### Phase 3: 约束优化（30-45分钟）

```bash
python train_v4_ideal.py --config configs/competition_preliminary.yaml --phase 3
```

---

## 自定义训练

### 更快完成（减少数据）

```bash
python train_v4_ideal.py \
    --config configs/competition_preliminary.yaml \
    --phase all \
    --episodes 25 \
    --timesteps 100000
```

### 更高质量（增加数据）

```bash
python train_v4_ideal.py \
    --config configs/competition_preliminary.yaml \
    --phase all \
    --episodes 100 \
    --timesteps 300000
```

### CPU训练

```bash
python train_v4_ideal.py \
    --config configs/competition_preliminary.yaml \
    --phase all \
    --device cpu
```

---

## 输出文件位置

**训练checkpoint**：
```
checkpoints/competition/
├── world_model_phase1.pth
├── ppo_phase2.zip
└── final_model_competition.pth
```

**评估结果**：
```
results/competition/
└── evaluation_results.xlsx
```

**训练日志**：
```
logs/competition/
```

---

## 常见命令

### 查看训练日志

```bash
tail -f logs/competition/*.log
```

### 验证环境

```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import traci; print('SUMO: OK')"
```

### 检查GPU

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

## 完整文档

详细使用说明请查看：[TRAINING_GUIDE.md](TRAINING_GUIDE.md)
