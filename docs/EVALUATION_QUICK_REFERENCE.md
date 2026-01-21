# 评估脚本快速参考卡

**版本**: v1.0
**创建日期**: 2025-01-21

---

## 🚀 快速开始

### Stage 1 评估（WorldModel）

```bash
python scripts/evaluate_stage1.py \
    --config configs/v5_complete.yaml \
    --checkpoint checkpoints/v5_complete/stage1_best.pth \
    --num_eval_episodes 20 \
    --visualize \
    --device cuda
```

**关键指标**:
- ✅ Flow RMSE < 20
- ✅ Risk F1 > 0.60
- ✅ R² > 0.85

**决策**: 如果以上指标都满足 → 进入Stage 2

---

### Stage 2 评估（PPO策略）

```bash
python scripts/evaluate_stage2.py \
    --config configs/v5_complete.yaml \
    --checkpoint checkpoints/v5_complete/stage2_best.pth \
    --num_eval_episodes 20 \
    --visualize \
    --device cuda
```

**关键指标**:
- ✅ OCR > 0.70
- ✅ 平均速度 > 15 m/s
- ✅ 碰撞率 < 1.0
- ✅ 干预率 < 50%

**决策**: 如果以上指标都满足 → 进入Stage 3

---

### Stage 3 评估（约束优化）

```bash
python scripts/evaluate_stage3.py \
    --config configs/v5_complete.yaml \
    --checkpoint checkpoints/v5_complete/stage3_best.pth \
    --baseline_checkpoint checkpoints/v5_complete/stage2_best.pth \
    --num_eval_episodes 20 \
    --visualize \
    --device cuda
```

**关键指标**:
- ✅ OCR相对Stage 2提升 > 2%
- ✅ 干预率降低 > 5%
- ✅ 碰撞率不增加

**决策**: 如果以上指标都满足 → 训练成功！🎉

---

## 📊 评估输出文件

### Stage 1 输出

```
logs/v5_complete/eval_stage1/
├── evaluation_results_YYYYMMDD_HHMMSS.json
└── plots_YYYYMMDD_HHMMSS/
    ├── flow_prediction_scatter.png
    ├── flow_error_bar.png
    ├── risk_confusion_matrix.png
    └── time_series_prediction.png
```

### Stage 2 输出

```
logs/v5_complete/eval_stage2/
├── evaluation_results_YYYYMMDD_HHMMSS.json
└── plots_YYYYMMDD_HHMMSS/
    ├── reward_metrics.png
    ├── reward_components.png
    └── safety_metrics.png
```

### Stage 3 输出

```
logs/v5_complete/eval_stage3/
├── stage3_results_YYYYMMDD_HHMMSS.json
├── comparison_YYYYMMDD_HHMMSS.json
└── plots_YYYYMMDD_HHMMSS/
    ├── reward_metrics.png
    ├── reward_components.png
    ├── safety_metrics.png
    ├── stage2_vs_stage3_comparison.png
    └── performance_radar.png
```

---

## 🎯 常用参数

| 参数 | 说明 | 默认值 | 推荐值 |
|------|------|--------|--------|
| `--num_eval_episodes` | 评估episode数 | 20 | 快速: 5, 完整: 20 |
| `--visualize` | 生成可视化 | False | True（生成报告时） |
| `--device` | 设备选择 | cuda | cuda/cpu |
| `--output_dir` | 输出目录 | logs/v5_complete/eval_stage* | 自定义路径 |

---

## 📖 完整文档

详细的评估指南请参考: [docs/EVALUATION_GUIDE.md](docs/EVALUATION_GUIDE.md)

内容包括:
- 每个阶段的详细评估标准
- 指标解读和阈值说明
- 常见问题与故障排除
- 完整的评估流程示例

---

## 🔍 快速诊断

### 问题: Stage 1评估失败

**症状**: Flow RMSE > 30

**诊断**:
```bash
# 检查归一化是否生效
python -c "
import numpy as np
from scripts.train_stage1_world_observer import normalize_vehicle_states

test = np.array([[1000, 10, 30, 5, 30, 3, 3, 360, 1]])
print('Before:', test)
print('After:', normalize_vehicle_states(test))
"
# 输出应该在[0,1]或[-1,1]范围内
```

**解决**:
1. 清除缓存: `rm -rf cache/stage1_trajectories/`
2. 重新训练: `--force_refresh`
3. 增加episodes: `--num_episodes 50`

---

### 问题: Stage 2评估失败

**症状**: OCR < 0.70

**诊断**:
```bash
# 检查训练收敛情况
tail -100 logs/v5_complete/stage2/*.log | grep "Mean Reward"
# Reward应该持续上升
```

**解决**:
1. 继续训练: `--resume checkpoints/v5_complete/stage2_best.pth`
2. 调整奖励权重
3. 增加训练迭代

---

### 问题: Stage 3评估失败

**症状**: OCR相比Stage 2下降

**诊断**:
```bash
# 查看Lambda变化
tail -100 logs/v5_complete/stage3/*.log | grep "Lambda"
# Lambda应该收敛，不应该持续上升
```

**解决**:
1. 放宽成本阈值
2. 降低lambda学习率
3. 继续训练让lambda收敛

---

## ✅ 评估检查清单

### 训练前
- [ ] 确认上一阶段已完成
- [ ] 确认上一阶段评估通过
- [ ] 检查GPU内存充足
- [ ] 清理旧的评估日志（可选）

### 训练中
- [ ] 监控训练日志
- [ ] 检查Loss/Reward趋势
- [ ] 观察无异常错误

### 训练后
- [ ] 运行评估脚本
- [ ] 检查关键指标
- [ ] 查看可视化图表
- [ ] 阅读评估建议
- [ ] 决定下一步行动

---

## 📞 获取帮助

如果遇到问题:
1. 查看 `docs/EVALUATION_GUIDE.md` 的故障排除部分
2. 检查训练日志: `logs/v5_complete/stage*/`
3. 查看评估结果JSON文件中的详细信息

---

**更新日期**: 2025-01-21
**版本**: v1.0
