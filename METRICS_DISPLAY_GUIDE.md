# 📊 训练指标显示说明

## 📅 更新日期
2026-01-16

---

## ✅ 新增功能

### 训练过程中的详细指标显示

现在训练过程中会显示以下类别的指标：

#### 📊 REWARD METRICS（奖励指标）
- **Mean Episode Reward**: 平均episode奖励
  - 衡量策略性能的核心指标
  - 越高越好

- **Mean Episode Length**: 平均episode长度
  - 平均每个episode的步数
  - 反映决策的稳定性

---

#### 📉 LOSS METRICS（损失指标）
- **Value Loss**: 价值函数损失
  - 价值网络预测误差
  - 越低越好

- **Policy Gradient Loss**: 策略梯度损失
  - 策略网络的梯度损失
  - 越低越好

- **Entropy Loss**: 熵损失
  - 探索程度的度量
  - 适度较高有助于探索

---

#### ⚙️ TRAINING METRICS（训练指标）
- **Learning Rate**: 当前学习率
  - 可能会随训练衰减

- **Clip Fraction**: 裁剪比例
  - PPO策略裁剪的比例
  - 理想范围：0.1-0.3

- **Clip Range**: PPO裁剪范围
  - 通常为0.2

- **Exploration Progress**: 探索进度
  - 如果使用探索衰减

- **Entropy**: 熵值
  - 反映策略的随机性
  - 适度值有助于探索

---

#### 📈 PERFORMANCE METRICS（性能指标）
- **Policy Updates**: 策略更新次数
  - 模型参数更新次数

- **Episodes**: 完成的episode数
  - 已完成的训练轮数

---

#### ⚡ SPEED（速度指标）
- **steps/sec**: 每秒训练步数
  - 实时训练速度

- **steps/hour**: 每小时训练步数
  - 用于估算总时间

---

## 📺 输出示例

```
================================================================================
[PROGRESS] 45,000/126,000 steps (35.7%)
[TIME] Elapsed: 10m 30s | ETA: 18m 45s

📊 REWARD METRICS:
   • Mean Episode Reward: 245.67
   • Mean Episode Length: 543.2 steps

📉 LOSS METRICS:
   • Value Loss: 0.023456
   • Policy Gradient Loss: 0.012345
   • Entropy Loss: 0.456789

⚙️  TRAINING METRICS:
   • Learning Rate: 3.00e-04
   • Clip Fraction: 0.234
   • Clip Range: 0.2000
   • Entropy: 0.0234

📈 PERFORMANCE METRICS:
   • Policy Updates: 8
   • Episodes: 156

⚡ SPEED:
   • 71.4 steps/sec
   • 257,143 steps/hour
================================================================================
```

---

## 🎯 如何解读指标

### 训练正常的标志

✅ **奖励上升**：
- Mean Episode Reward 逐渐增加
- 说明策略在改进

✅ **损失下降**：
- Value Loss 和 Policy Loss 逐渐下降
- 说明学习在收敛

✅ **Clip Fraction 适中**：
- 在 0.1-0.3 范围内
- 说明策略更新幅度合理

✅ **训练速度稳定**：
- steps/sec 保持稳定
- 没有明显的性能退化

---

### 需要关注的问题

⚠️ **奖励下降**：
- 如果 Mean Episode Reward 持续下降
- 可能需要调整学习率或检查环境

⚠️ **损失爆炸**：
- Value Loss 或 Policy Loss 突然增大
- 可能是学习率过高或梯度不稳定

⚠️ **Clip Fraction 过高**：
- 持续 > 0.5
- 说明策略变化太大，可能需要降低学习率

⚠️ **熵过低**：
- Entropy < 0.01
- 说明过早收敛，可能陷入局部最优

---

## 📊 训练完成时的最终统计

```
================================================================================
[DONE] Training completed!
================================================================================

📊 FINAL STATISTICS:
   • Total Steps: 129,576
   • Target Steps: 126,000
   • Total Time: 27m 25s
   • Average Speed: 78.6 steps/sec

📈 FINAL METRICS:
   • Final Mean Reward: 312.45
   • Final Value Loss: 0.018765
   • Final Policy Loss: 0.009876
```

---

## 🔧 自定义显示间隔

如果需要修改指标显示频率，可以编辑回调函数中的参数：

```python
# 在 train_phase2_stable.py 的 EnhancedTrainingCallback 类中
self.print_interval = 60  # 默认每60秒打印一次

# 修改为更频繁的显示（例如每30秒）
self.print_interval = 30

# 或修改为步数间隔（需要自定义实现）
```

---

## 💡 提示

1. **观察趋势**：关注指标的变化趋势，而不是绝对值

2. **对比阶段**：不同训练阶段之间对比指标

3. **TensorBoard**：除了终端输出，还可以使用 TensorBoard 查看详细曲线：
   ```bash
   tensorboard --logdir logs/
   ```

4. **保存日志**：所有指标都会自动保存到 TensorBoard 日志目录

---

**状态**: ✅ 指标显示已增强
**日期**: 2026-01-16
