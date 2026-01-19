# 初赛轻量级架构 v5.0 总结

## 📊 架构概览

基于对官方评测公式的深入分析，我们实现了一个**专门针对初赛的轻量级架构**，核心目标是：

1. ✅ **直接优化OCR**（而非速度/吞吐量）
2. ✅ **最小化干预成本**（避免指数惩罚）
3. ✅ **快速训练**（6-8小时 vs 原架构3天）

---

## 🎯 核心设计理念

### 1. **干预成本的真实含义**

通过分析官方脚本 `main.py`，我们发现：

> **ICV渗透率不是环境的固有属性，而是你的算法决定控制多少辆车！**

```python
# 官方框架允许你控制任何车辆
vehicle_ids = traci.vehicle.getIDList()  # 获取所有车辆
for veh_id in vehicle_ids:
    traci.vehicle.setSpeed(veh_id, target_speed)  # 你决定控制谁
```

**干预成本公式：**
```python
C_int = (α×Σacmd + β×Σlc) / (T_total × N_ICV)

其中：
- acmd: 加速度指令数（α=1）
- lc: 换道指令数（β=5，成本极高！）
- T_total: 总仿真步数
- N_ICV: 你的算法控制的车辆数
```

**关键洞察：**
- ⭐ 控制5%车辆比控制25%车辆，干预成本降低5倍
- ⭐ 每10步决策一次比每步决策，成本降低10倍
- ⭐ 避免换道（β=5），成本是加速度的5倍

---

## 🏗️ 架构对比

| 维度 | 原架构 (v4.0) | 轻量级架构 (v5.0) |
|------|---------------|-------------------|
| **模型复杂度** | 世界模型 + GNN + 门控 | 3层GNN |
| **参数量** | 数百万 | 23K |
| **训练时间** | 3天（4阶段） | 6-8小时（2阶段） |
| **控制车辆比例** | 25% | 5% |
| **决策频率** | 每步 | 每10步 |
| **奖励函数** | 速度/吞吐量 | OCR直接优化 |
| **可解释性** | 黑盒（LSTM） | 可解释规则 |

---

## 📁 文件结构

```
src/
├── models/
│   └── v5_lightweight.py          # 轻量级OCR-GNN
│       ├── LightweightGraphConvolution
│       ├── VehicleInfluenceScorer
│       ├── LightweightOCRGNN
│       └── LightweightPolicyV5
│
├── training/
│   └── ocr_rewards.py             # OCR奖励计算器
│       ├── OCRRewardCalculator
│       └── BaselineStatisticsCollector
│
└── env/
    └── sparse_controller.py        # 稀疏控制器
        ├── SparseController (基类)
        ├── RuleBasedSparseController
        └── LearnedSparseController

train_phase1_lite.py                # 训练脚本
configs/phase1_lite.yaml           # 配置文件
test_v5_architecture.py            # 测试脚本（✅所有测试通过）
```

---

## 🚀 使用方法

### 1. **快速测试**

```bash
# 运行测试脚本
python test_v5_architecture.py

# 预期输出：所有测试通过
# ✅ 模型创建和前向传播
# ✅ Top-K车辆选择机制
# ✅ OCR奖励计算
# ✅ 稀疏控制器
# ✅ 端到端集成测试
```

### 2. **训练流程**

```bash
# 完整训练（2个阶段）
python train_phase1_lite.py --config configs/phase1_lite.yaml --stage all

# 只运行Stage 1（行为克隆）
python train_phase1_lite.py --config configs/phase1_lite.yaml --stage stage1

# 只运行Stage 2（PPO微调）
python train_phase1_lite.py --config configs/phase1_lite.yaml --stage stage2 --checkpoint checkpoints/phase1_lite_xxx/stage1_behavior_cloning.pth
```

### 3. **配置说明**

```yaml
# configs/phase1_lite.yaml

model:
  gnn:
    top_k_ratio: 0.05    # ⭐ 只控制5%车辆（关键！）
    hidden_dim: 64       # 隐藏层维度
    num_layers: 3        # GNN层数

training:
  stage1:               # 行为克隆
    num_episodes: 50
    epochs: 20

  stage2:               # PPO微调
    total_timesteps: 500000  # 50万步（vs 原方案300万步）
    num_envs: 8
    learning_rate: 0.0003

sparse_controller:
  decision_interval: 10  # ⭐ 每10步决策一次（关键！）
  top_k_ratio: 0.05
```

---

## 🔬 关键技术细节

### 1. **可解释的车辆选择**

```python
class VehicleInfluenceScorer(nn.Module):
    """
    70%学习特征 + 30%可解释规则
    """
    def forward(self, vehicle_states, gnn_embeddings):
        # 可解释特征
        in_bottleneck = (s > 0.7) & (lane < 3)     # 瓶颈区域
        is_slow = speed < (avg_speed - 1.5*std)    # 慢车
        in_merge_zone = in_ramp & in_merge_area    # 汇流区

        # 组合评分
        score = (
            0.3 * in_bottleneck +
            0.3 * is_slow +
            0.3 * in_merge_zone +
            0.1 * gnn_learned  # GNN学习的影响
        )

        return score
```

### 2. **稀疏控制机制**

```python
class SparseController:
    """
    按需决策，降低干预频率
    """
    def should_make_decision(self, step):
        # 规则1：定时决策（每10步）
        if (step - self.last_decision_step) >= 10:
            return True

        # 规则2：紧急情况立即干预
        if self.is_emergency():
            return True

        return False
```

### 3. **OCR直接奖励**

```python
class OCRRewardCalculator:
    """
    直接对齐评测公式
    """
    def update(self, vehicle_info, accel_commands, lane_changes, num_controlled):
        # 计算OCR
        ocr = (num_arrived + Σ(d_traveled / d_total)) / num_total

        # 计算效率得分
        s_efficiency = 100 * (ocr - baseline_ocr) / baseline_ocr

        # 计算稳定性得分
        s_stability = 100 * (0.4 * I_speed_std + 0.6 * I_accel)

        # 计算干预成本
        c_int = (1*Σacmd + 5*Σlc) / (T * N_ICV)
        p_int = exp(-0.1 * c_int)

        # 总分
        s_total = (0.7 * s_efficiency + 0.3 * s_stability) * p_int

        return s_total
```

---

## 📈 性能预估

### 干预成本对比

**场景A：暴力控制（错误）**
```
控制车辆：600辆
决策频率：每步
换道指令：每10步每车换道1次

C_int = (1*1,080,000 + 5*108,000) / (1800*600) = 1.5
P_int = e^(-0.4*1.5) ≈ 0.5  # ❌ 得分减半！
```

**场景B：智能稀疏控制（正确）**
```
控制车辆：30辆（5%）
决策频率：每10步
换道指令：只在紧急情况

C_int = (1*2,700 + 5*50) / (180*30) = 0.546
P_int = e^(-0.4*0.546) ≈ 0.85  # ✅ 只损失15%！
```

---

## 🎓 初赛建议

### ✅ **应该做的**

1. **控制比例：5-10%**
   - 5%足够大多数场景
   - 只在极端拥堵时提升到10%

2. **决策频率：每5-10步**
   - 降低T_total，直接降低成本
   - 不会影响效果（交通流变化缓慢）

3. **避免换道**
   - β=5，成本是加速度的5倍
   - 只在TTC<2秒的紧急情况换道

4. **直接优化OCR**
   - 不要优化速度/吞吐量
   - OCR是效率得分的唯一指标

### ❌ **不应该做的**

1. **控制25%车辆**
   - 成本增加5倍，惩罚因子大幅下降

2. **每步都决策**
   - 成本增加10倍，得不偿失

3. **频繁变道**
   - 换道成本是加速度的5倍

4. **使用世界模型**
   - 过度复杂，训练时间太长
   - 初赛不需要预测未来

---

## 🔧 调试技巧

### 1. **监控干预成本**

```python
# 训练时打印
info = calculator.get_statistics()
print(f"干预率: {info['num_controlled_vehicles']} / {info['num_total']}")
print(f"平均成本: {(info['num_accel_commands'] + 5*info['num_lane_changes']) / info['total_steps']:.4f}")
```

### 2. **可视化Top-K选择**

```python
selected_indices, info = policy.select_vehicles(obs)
print(f"选中车辆: {selected_indices}")
print(f"影响力评分: {info['influence_scores']}")
```

### 3. **验证OCR计算**

```python
scores = calculator.compute_episode_score()
print(f"OCR: {scores['ocr']:.4f}")
print(f"效率得分: {scores['s_efficiency']:.2f}")
print(f"惩罚因子: {scores['p_intervention']:.4f}")
```

---

## 📚 相关文档

- `docs/交通工程赛道-评测公式.md` - 官方评测公式
- `docs/赛题.md` - 赛题背景和挑战
- `docs/理想架构.md` - 原架构设计思路
- `src/models/v5_lightweight.py` - 模型实现（含详细注释）
- `src/training/ocr_rewards.py` - 奖励计算（含评测公式）
- `src/env/sparse_controller.py` - 稀疏控制器（含规则说明）

---

## ✅ 测试验证

所有测试通过：
```bash
$ python test_v5_architecture.py

✅ 模型创建和前向传播
✅ Top-K车辆选择机制（3.3% vs 目标5.0%）
✅ OCR奖励计算（OCR: 0.44, S_total: 29.52）
✅ 稀疏控制器（5/100辆选中）
✅ 端到端集成测试（最终得分: 28.69）
```

---

## 🎯 下一步

1. **运行基准测试**
   ```bash
   python train_phase1_lite.py --stage all
   ```

2. **分析结果**
   - 检查干预率是否 < 10%
   - 验证OCR是否提升
   - 对比baseline得分

3. **调优参数**
   - 如果OCR低 → 增加top_k_ratio
   - 如果惩罚因子低 → 降低decision_interval
   - 如果不稳定 → 增加w_stability权重

---

**总结**：这个架构通过"少控制、慢决策、直接优化OCR"的策略，在6-8小时内完成训练，同时最大化初赛得分。
