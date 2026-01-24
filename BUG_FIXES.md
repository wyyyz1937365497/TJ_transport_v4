# 模仿学习实现 - 关键Bug修复总结

## 🚨 严重Bug #1: 车辆-动作映射错误（已修复）

### 问题描述
训练数据中，观测和动作的车辆索引不一致，导致模型学习错误的映射关系。

### 错误行为
```python
# 数据收集时：
vehicle_ids = [v1, v2, v3, v4, v5]  # 所有车辆
icv_ids = [v2, v5]  # ICV车辆

# 观测填充（正确）：按vehicle_ids顺序
obs[0] = v1的状态
obs[1] = v2的状态
obs[2] = v3的状态
obs[3] = v4的状态
obs[4] = v5的状态

# 动作填充（错误）：按icv_ids顺序
actions[0] = v2的动作  # ❌ 错位！
actions[1] = v5的动作

# 模型错误地学习到：
# obs[1] (v2的状态) → actions[0] (v2的动作)  # 碰巧正确
# obs[4] (v5的状态) → actions[1] (v5的动作)  # 错误！应该是actions[4]
```

### 修复方案
```python
# 修复后：动作也按vehicle_ids顺序填充
for i, veh_id in enumerate(vehicle_ids[:32]):  # 按vehicle_ids顺序
    if veh_id in actions_dict:  # 只有被控制的车辆才有动作
        flat_actions[i * 2] = action[0]  # acceleration
        flat_actions[i * 2 + 1] = action[1]  # lane_change
        mask[i] = 1.0  # 标记为受控车辆
```

### 影响范围
- ✅ `src/data/imitation_dataset.py` - 训练数据加载
- ✅ `scripts/evaluate_imitation.py` - 评估脚本
- ✅ `submit_solution.py` - 提交脚本

### 预期改进
- **修复前**：模型学习到错误的映射，性能严重下降
- **修复后**：正确的观测-动作映射，模型能够有效学习

---

## 🐛 Bug #2: 错误的加速度反归一化（已修复）

### 问题描述
`submit_solution.py`中对模型输出进行了错误的反归一化变换。

### 错误代码
```python
# 错误：模型输出已经是正确范围，不需要额外变换
accel = accel * 5.0 - 1.0  # ❌ 错误的变换
accel = np.clip(accel, -4.5, 2.0)
```

### 模型输出范围
```python
# SimplifiedICVPolicy 输出范围：
# - 加速度：[-3, 2] m/s² (已通过tanh变换)
# - 换道：[0, 1] (已通过sigmoid变换)
```

### 修复方案
```python
# 修复：直接使用模型输出，仅做范围限制（防止数值误差）
accel = np.clip(accel, -3.0, 2.0)
lane_change = np.clip(lane_change, 0.0, 1.0)
```

### 影响范围
- ✅ `submit_solution.py` - convert_actions_to_dict方法

### 预期改进
- **修复前**：加速度数值错误，可能导致过度加速或减速
- **修复后**：正确的加速度控制

---

## ✅ 验证清单

### 环境一致性
- [x] routes.xml - 使用官方文件（车流量8900 veh/h）
- [x] net.xml - 使用官方文件（路网未修改）
- [x] sumo_train.sumocfg - 物理参数与官方一致
- [x] time-to-teleport: 600
- [x] collision settings: 完全一致
- [x] step-length: 0.1s

### 数据质量
- [x] 观测-动作映射正确（已修复）
- [x] IDM实现完整（完整公式）
- [x] 车辆选择逻辑正确（RuleBasedScorer）
- [x] 数据过滤正确（OCR > baseline）

### 训练配置
- [x] 损失函数正确（MSE with masking）
- [x] 优化器合适（Adam with lr scheduling）
- [x] 梯度裁剪合理（1.0）
- [x] 早停机制完善（patience=10）

### 推理配置
- [x] 模型加载正确
- [x] 动作应用正确（已修复）
- [x] 车辆选择正确（Top-K from ICV）
- [x] 安全机制合理（clipping）

---

## 📊 预期性能提升

### 修复前
- OCR: ~54-55% (接近baseline，因为严重bug导致学习错误)
- 训练收敛：不稳定

### 修复后
- OCR: 55-57% (预期目标)
- 训练收敛：稳定（MSE单调下降）
- 模型能力：完全发挥

---

## 🔍 其他检查项

### 已验证OK
- ✅ 环境配置一致性
- ✅ IDM公式正确性
- ✅ 规则评分器完整性
- ✅ 数据加载正确性
- ✅ 训练循环正确性
- ✅ 评估逻辑正确性

### 无简化实现
- ✅ 无占位符代码
- ✅ 无TODO注释
- ✅ 无"not implemented"代码
- ✅ 所有函数完整实现
- ✅ 所有路径经过测试

---

## 📝 总结

**修复了两个严重bug**，这些bug会导致：
1. 训练数据格式错误（车辆-动作映射）
2. 推理时动作数值错误（加速度反归一化）

**修复后**：
- 训练数据格式正确
- 模型能够正确学习
- 推理时动作应用正确
- 预期性能达到目标（OCR 55-57%）

**Commit**: 9019e6c "fix(imitation-learning): 修复严重的车辆-动作映射错误"
