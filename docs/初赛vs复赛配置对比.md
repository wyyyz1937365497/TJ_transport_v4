# 初赛 vs 复赛配置对比

## 📋 核心区别

| 维度 | 初赛（以粒控流） | 复赛（粒流协控） |
|-----|----------------|----------------|
| **控制对象** | 仅车辆 | 车辆 + 设施（信号灯、VSL） |
| **智能车比例** | 25%固定 | 可变，设施控制增加灵活性 |
| **评分重点** | 效率为主 | 效率 + 稳定性 + 成本平衡 |
| **动作空间** | 连续（加速度、变道） | 混合（连续车辆 + 离散设施） |
| **Top-K选择** | 核心（稀疏控制） | 相对次要（有设施辅助） |

## ⚙️ 配置文件对比

### 1. 智能车比例
```yaml
# 初赛 (competition_preliminary.yaml)
environment:
  icv_ratio: 0.25  # 固定25%

# 复赛 (competition.yaml)
environment:
  icv_ratio: 0.20  # 可调整，配合设施控制
```

### 2. Top-K配置
```yaml
# 初赛
model:
  controller:
    top_k: 8              # 32辆中选8辆
    alpha_init: 0.6       # GNN权重
    beta_init: 0.4        # 预测权重

# 复赛
model:
  controller:
    top_k: 5              # 有设施辅助，减少车辆控制
    adaptive_top_k: true  # 自适应调整
```

### 3. 奖励权重
```yaml
# 初赛 - 侧重效率
environment:
  rewards:
    speed_weight: 2.0           # 提高
    efficiency_weight: 3.0      # 提高
    safety_weight: 1.0          # 降低
    intervention_cost_weight: 0.5  # 降低

# 复赛 - 平衡三者
environment:
  rewards:
    speed_weight: 1.0           # 标准
    efficiency_weight: 2.0      # 标准
    safety_weight: 3.0          # 提高（稳定性重要）
    intervention_cost_weight: 1.5  # 提高（成本重要）
```

### 4. 训练配置
```yaml
# 初赛 - 简化训练
training:
  phase2:
    total_timesteps: 1500000    # 150万步
    use_failure_bank: false     # 关闭高级功能
    target_kl: 0.01             # 标准KL控制

# 复赛 - 完整训练
training:
  phase2:
    total_timesteps: 2000000    # 200万步
    use_failure_bank: true      # 启用失败案例库
    use_per: true               # 启用优先经验回放
```

### 5. 评估指标
```yaml
# 初赛 - 侧重效率
evaluation:
  metrics:
    - avg_speed          # ⭐ 核心
    - throughput         # ⭐ 核心
    - intervention_rate  # 次要
    - episode_reward     # 参考

# 复赛 - 综合评估
evaluation:
  metrics:
    - avg_speed          # 核心
    - throughput         # 核心
    - intervention_rate  # 核心（成本惩罚）
    - safety_score       # ⭐ 核心（稳定性）
    - stability_metrics  # ⭐ 核心
```

## 🚀 训练脚本对比

### 初赛专用脚本
```bash
# 一键训练
./run_preliminary.sh

# 分步训练
python train_preliminary.py --config configs/competition_preliminary.yaml
python train_preliminary.py --skip-phase1 --start-level 3
```

### 复赛完整脚本
```bash
# 完整训练（包含设施控制）
./train_all.sh

# 分阶段训练
python train_phase1.py
python train_phase2.py --stage 1
python train_phase3.py --mode fine-tune
```

## 📊 模型架构对比

### 相同部分（两个阶段都使用）
- ✅ RiskSensitiveGNN（风险敏感图神经网络）
- ✅ MultiScaleRSSM（多尺度世界模型）
- ✅ InfluenceBasedController（影响力控制器）
- ✅ SafetyBarrier（双模态安全屏障）

### 不同部分

#### 初赛模型（IdealTrafficControllerV4）
```python
class PreliminaryController(nn.Module):
    def __init__(self):
        self.perception_layer = RiskSensitiveGNN()
        self.prediction_layer = MultiScaleRSSM()
        self.decision_layer = InfluenceBasedController(
            top_k=8,  # 固定Top-K
            action_dim=2  # [加速度, 变道]
        )
        self.safety_barrier = SafetyBarrier()
        # ❌ 无设施控制模块
```

#### 复赛模型（IdealTrafficControllerV4 + 设施层）
```python
class FinalController(nn.Module):
    def __init__(self):
        self.perception_layer = RiskSensitiveGNN()
        self.prediction_layer = MultiScaleRSSM()
        self.decision_layer = EnhancedInfluenceBasedController(
            top_k=5,  # 减少车辆控制
            adaptive_top_k=True
        )
        self.facility_controller = FacilityController()  # ✅ 新增
        self.safety_barrier = SafetyBarrier()
        self.lagrangian_optimizer = LagrangianOptimizer()  # ✅ 新增
```

## 🎯 训练策略对比

### 初赛策略
1. **快速验证优先**：先在Level 5训练，验证可行性
2. **效率驱动**：奖励函数侧重速度和吞吐量
3. **适度稀疏控制**：Top-K=8，平衡性能和成本
4. **简化流程**：Phase 1 + Phase 2，跳过Phase 3

### 复赛策略
1. **完整课程学习**：必须完成所有5个级别
2. **多目标平衡**：效率、稳定性、成本三者平衡
3. **更强稀疏控制**：Top-K=5 + 自适应调整
4. **拉格朗日优化**：Phase 3约束优化，最小化干预成本

## 📈 性能预期

### 初赛目标
- 平均速度：**> 12 m/s**（优秀：> 15 m/s）
- 吞吐量：**> 0.8 veh/step**
- 干预率：**< 30%**

### 复赛目标
- 平均速度：**> 10 m/s**（有设施辅助，可能略低）
- 稳定性：**速度方差 < 2.0**
- 干预率：**< 20%**（设施控制更高效）
- 成本惩罚因子：**> 0.7**

## 🔍 选择建议

### 使用初赛配置，如果：
- ✅ 正在参加初赛
- ✅ 只需控制车辆
- ✅ 评分侧重效率
- ✅ 希望快速迭代

### 使用复赛配置，如果：
- ✅ 正在参加复赛
- ✅ 需要控制设施（信号灯、VSL）
- ✅ 评分平衡效率、稳定性、成本
- ✅ 追求最高性能

## 📝 迁移指南

### 从初赛到复赛

1. **修改配置文件**
```bash
cp configs/competition_preliminary.yaml configs/competition_final.yaml
# 编辑competition_final.yaml，调整为复赛参数
```

2. **加载初赛模型**
```python
# 加载Level 5权重作为初始化
python train_phase3.py \
    --base-model checkpoints/competition/preliminary/level5/custom_ppo.zip \
    --mode fine-tune \
    --config configs/competition.yaml
```

3. **训练设施控制器**
```python
# 在预训练模型基础上，添加设施控制模块
# 使用较小的学习率微调
python train_phase3.py --mode fine-tune --lr 1e-4
```

4. **拉格朗日约束优化**
```python
# Phase 3：优化干预成本
python train_phase3.py \
    --mode constrained_opt \
    --cost-limit 0.1
```

## 🏆 总结

| 方面 | 初赛 | 复赛 |
|-----|------------------|------------------|
| **复杂度** | ⭐⭐ 简单 | ⭐⭐⭐⭐ 复杂 |
| **训练时间** | 10-15小时 | 20-30小时 |
| **调参难度** | 低 | 高 |
| **性能上限** | 中等 | 高 |
| **推荐场景** | 快速验证、初赛 | 冲刺排名、复赛 |

**建议**：
- 初赛阶段：使用`competition_preliminary.yaml`快速验证
- 复赛阶段：使用`competition.yaml`追求极致性能
