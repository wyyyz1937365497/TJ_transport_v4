# 📊 OCR-MAX模型性能分析报告

生成时间：2026-01-24

---

## 🎯 核心发现

### ⚠️ 严重问题：模型性能不如baseline！

| 指标 | Baseline（无控制） | OCR-MAX模型 | 差异 |
|------|------------------|------------|------|
| **OCR** | **54.69%** | **54.39%** | **-0.30%** ❌ |
| 累计出发 | 618辆 | ~618辆 | - |
| 累计到达 | 338辆 | ~336辆 | -2辆 |
| 运行步数 | 3600 | 3600 | - |

---

## 🏆 比赛得分计算

### 初赛评分公式

```
Score = 100 × max(0, ΔOCR)

其中：
ΔOCR = (OCR_AI - OCR_Baseline) / OCR_Baseline
```

### 实际计算

```
OCR_AI = 0.5439
OCR_Baseline = 0.5469

ΔOCR = (0.5439 - 0.5469) / 0.5469
     = -0.0030 / 0.5469
     = -0.00548
     = -0.548%

Score = 100 × max(0, -0.00548)
     = 100 × 0
     = 0 分 ❌
```

### 结论

**当前模型得分为 0 分**，因为OCR低于baseline。

---

## 🔍 问题分析

### 问题1：模型控制反而降低了OCR

**可能原因**：

1. **ICV选择不准确**
   - 训练时使用智能评分器选择瓶颈车辆
   - 但选择的车辆可能不是最关键的
   - 错误的控制可能干扰自然交通流

2. **策略网络欠佳**
   - Stage 2训练的Mean Return只有120-126（很低）
   - Entropy从14上升到19（策略不稳定）
   - Policy Loss波动大（76-972）
   - 说明模型没有学到好策略

3. **训练-测试不一致**
   - 训练环境的车辆行为可能与实际不同
   - 模型可能过拟合训练场景

4. **奖励函数设计问题**
   - OCR奖励稀疏（3600步后才给反馈）
   - Bottleneck即时奖励可能设置不当
   - 权重可能不合理

---

## 📈 性能对比（详细）

### Baseline（无控制）
```
步骤 1000: 活跃车辆 207, 累计出发 231, 累计到达 24
步骤 2000: 活跃车辆 260, 累计出发 390, 累计到达 130
步骤 3000: 活跃车辆 281, 累计出发 536, 累计到达 255
最终:      OCR = 54.69% (338/618)
```

### OCR-MAX模型（10% ICV控制）
```
步骤 1000: ICV = 10辆 (10% of 100)
步骤 2000: ICV = 20辆 (10% of 200)
...
最终:      OCR = 54.39% (~336/618)
```

**差异**：模型控制反而导致**2辆车未完成OD**

---

## 💡 改进建议

### 紧急修复（优先级最高）

#### 方案A：禁用问题控制，回到接近baseline

**问题**：当前某些控制动作可能有害

**解决**：
1. 启用SafetyShield的严格模式
2. 禁止减速动作（只允许加速或保持）
3. 禁止换道动作（只控制加速度）

**修改位置**：`submit_solution.py`
```python
# 在apply_control_algorithm中添加过滤
def filter_safe_actions(self, actions_dict, vehicle_states):
    """只保留安全的控制动作"""
    safe_actions = {}

    for veh_id, action in actions_dict.items():
        acceleration, lane_change = action

        # 1. 禁止减速（只允许加速或保持）
        if acceleration < 0:
            acceleration = 0.0

        # 2. 禁止换道
        lane_change = 0.0

        safe_actions[veh_id] = (acceleration, lane_change)

    return safe_actions
```

**预期效果**：OCR恢复到54.5-54.7%

---

#### 方案B：降低ICV渗透率到5%

**假设**：10%控制太多了，干扰自然流

**实施**：
```python
# 修改submit_solution.py
num_icv = max(1, int(len(vehicle_ids) * 0.05))  # 10% → 5%
```

**预期效果**：
- 控制车辆数：25 → 12-13
- OCR可能恢复到54.5-54.8%

---

#### 方案C：使用更保守的checkpoint

**当前**：stage2_ppo_iter99.pth (OCR=54.39%)
**尝试**：stage2_best.pth (OCR=54.05%) - 可能更稳定

但实际上更差，不推荐。

---

### 根本解决（需要重新训练）

#### 方案1：修复训练配置

**问题**：
- Stage 2的Mean Return只有120-126
- Entropy上升（14→19）
- Policy Loss波动大

**解决**：
1. 调整奖励权重（降低Bottleneck奖励，增加OCR奖励）
2. 增加训练iterations（100→200）
3. 降低学习率（1e-4 → 5e-5）
4. 启用Early Stopping

**配置文件**：`configs/ocr_max.yaml`
```yaml
stage2_ppo_finetune:
  ppo:
    num_iterations: 200  # 100 → 200
    lr: 5.0e-5  # 1e-4 → 5e-5
    entropy_coef: 0.005  # 0.01 → 0.005 (更确定)

  bottleneck_rewards:
    enabled: true
    w_throughput: 0.3  # 0.5 → 0.3 (降低权重)
    w_queue: 0.2       # 0.3 → 0.2
    w_conflict: 0.1    # 0.2 → 0.1
```

---

#### 方案2：收集更好的演示数据

**问题**：Stage 1的模仿学习数据质量不够

**解决**：
1. 使用更智能的规则生成演示
2. 数据清洗（移除OCR<baseline的episode）
3. 增加演示数据量（50→100 episodes）

---

#### 方案3：尝试更高渗透率（15-20%）

**假设**：虽然当前10%失败，但更高渗透率可能更好

**理由**：
- 更多控制 → 更强的影响力
- 可以覆盖所有瓶颈区域
- 初赛不惩罚控制成本

**配置**：`configs/high_penetration_test.yaml`
```yaml
environment:
  icv_ratio: 0.15  # 15% (~45辆车)

policy:
  obs_dim: 45 * 9 + 45 + 1 = 451
  num_vehicles: 45
```

**风险**：
- 需要重新训练（12-18小时）
- 可能依然失败（策略问题未解决）

---

## 🚀 立即可行的方案（按优先级）

### ✅ 方案1：保守控制（最快，1小时）

**步骤**：
1. 修改`submit_solution.py`：禁用减速和换道
2. 重新评估OCR
3. 如果OCR > 54.69%，提交

**代码修改**：见上方方案A

**预期**：OCR = 54.5-54.8%

---

### ✅ 方案2：降低渗透率（快，1小时）

**步骤**：
1. 修改`submit_solution.py`：icv_ratio = 0.05
2. 重新评估OCR
3. 如果OCR > 54.69%，提交

**代码修改**：见上方方案B

**预期**：OCR = 54.5-54.9%

---

### ⚠️ 方案3：混合策略（中等，2小时）

**思路**：瓶颈区域控制，其他区域不控制

**步骤**：
1. 检测车辆是否在瓶颈区域（s: 1200-2200m）
2. 只控制瓶颈区域的车辆
3. 非瓶颈区域车辆使用baseline行为

**代码修改**：
```python
def select_bottleneck_icv(self, vehicle_ids):
    """只选择瓶颈区域的车辆"""
    bottleneck_vehicles = []

    for veh_id in vehicle_ids:
        s, d = self._get_vehicle_frenet(veh_id)
        if 1200 <= s <= 2200:  # 瓶颈区域
            bottleneck_vehicles.append(veh_id)

    # 控制瓶颈区域前10%的车辆
    num_icv = max(1, int(len(bottleneck_vehicles) * 0.10))
    return set(bottleneck_vehicles[:num_icv])
```

**预期**：OCR = 54.8-55.2%

---

### 🔄 方案4：重新训练（慢，12-24小时）

**前提**：上述快速方案都失败

**步骤**：
1. 修复训练配置（见方案1）
2. 重新训练Stage 2
3. 评估新模型
4. 如果OCR > 55%，提交

**预期**：OCR = 55-57%

---

## 📊 容量分析

### 当前SUMO配置

**流量统计**：
```
正向前向:  800 + 500 + 1000 + 400 + 200 = 2900 veh/h
反向:      1000 veh/h
支路汇入:  1500 + 1500 + 1500 + 1500 = 6000 veh/h
-------------------------------------------------
总计:      9900 veh/h
```

**实际车辆数**：618辆（1小时仿真）

**瓶颈**：
- E17/J15汇入点：1500 veh/h
- E19/J17汇入点：1500 veh/h
- 主路容量有限

### 模型容量配置

**当前**：
- ICV比例：10%（~25辆）
- 观测维度：321 (32×9+32+1)
- 模型参数：~500K

**建议配置**：
- 保守控制（方案1/2）：保持当前配置
- 混合策略（方案3）：保持当前配置
- 重新训练（方案4）：考虑扩大到15-20%

---

## 🎓 结论与建议

### 当前状态

❌ **模型不可提交**（得分 = 0）

### 推荐行动路径

**今天（快速修复）**：
1. ✅ 尝试方案1（保守控制）- 1小时
2. ✅ 如果失败，尝试方案2（降低渗透率）- 1小时
3. ✅ 如果还失败，尝试方案3（混合策略）- 2小时

**明天（如果快速修复失败）**：
4. 🔄 重新训练模型（方案4）- 12-24小时

**最终目标**：
- OCR > 55%（+0.3% vs baseline）
- 比赛得分 > 0.5分

---

## 📁 相关文件

- **Baseline结果**：`competition_results/baseline_results.json`
- **当前模型结果**：`logs/ocr_max/evaluation/stage2_ppo_iter99_results.json`
- **配置文件**：`configs/ocr_max.yaml`
- **提交脚本**：`submit_solution.py`

---

## 🔗 快速命令

```bash
# 1. 查看baseline结果
cat competition_results/baseline_results.json

# 2. 运行当前模型评估
python scripts/evaluate_ocr_max.py \
    --config configs/ocr_max.yaml \
    --checkpoint checkpoints/ocr_max/stage2_ppo_iter99.pth \
    --num_eval_episodes 5 \
    --device cuda \
    --baseline_ocr 0.5469

# 3. 测试提交脚本
python submit_solution.py
```

---

**最后更新**：2026-01-24
**状态**：⚠️ 模型性能低于baseline，需要紧急修复
