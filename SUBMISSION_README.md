# OCR-MAX模型提交指南

## 📋 概述

`submit_solution.py` 是用于提交OCR-MAX模型到官方评测系统的脚本。该脚本：

1. ✅ 参考官方 `main.py` 框架结构
2. ✅ 集成 `SimplifiedICVPolicy` 模型
3. ✅ 每个仿真步调用模型获取控制指令
4. ✅ 生成符合要求的 `submit.xlsx` 文件

## 🚀 快速开始

### 方式1: 直接运行（使用默认配置）

```bash
python submit_solution.py
```

### 方式2: 修改配置后运行

编辑 `submit_solution.py` 的 `main()` 函数：

```python
# SUMO配置文件
sumo_cfg = "仿真环境_初赛_1.0/仿真环境-初赛/sumo_train.sumocfg"

# 模型权重路径（选择你最好的checkpoint）
checkpoint_path = "checkpoints/ocr_max/stage2_ppo_iter99.pth"  # 推荐！

# 仿真参数
MAX_STEPS = 3600
USE_GUI = False  # 提交时必须设为False
DEVICE = 'cuda'  # 或 'cpu'
```

## 📊 输出文件

运行成功后，会生成 `competition_results/submit.xlsx`，包含以下sheets：

### 1. 数据汇总
- 理论总需求
- 实际累计出发/到达
- **OD完成率(OCR)** ⭐ - 比赛核心指标
- 唯一车辆数
- 仿真参数

### 2. 仿真参数
- 完整的仿真配置参数
- 模型权重路径

### 3. 时间步数据
- 每个时间步的系统状态
- 红绿灯状态
- 车辆统计

### 4. 车辆数据
- 每辆车每个时间步的详细数据
- OD信息
- 完成率

## 🔧 核心功能说明

### 1. 模型集成 (`load_model`)
```python
self.policy = SimplifiedICVPolicy(
    obs_dim=321,
    node_dim=9,
    hidden_dim=128,
    num_layers=3,
    num_vehicles=32,
    device='cuda',
    use_safety_shield=True  # 推理时启用
)
```

### 2. 观测处理 (`get_observation_dict`)
- 从SUMO获取车辆状态
- 选择ICV（前10%车辆）
- 计算Frenet坐标（简化版）
- 计算全局统计（32维）

### 3. 扁平化观测 (`flatten_observation`)
- 将观测字典转换为 `[321]` 扁平化向量
- 归一化车辆状态（9维特征）
- 拼接全局统计和车辆数量

### 4. 模型推理 (`apply_control_algorithm`)
```python
# 1. 获取观测
obs_dict = self.get_observation_dict()
obs_flat = self.flatten_observation(obs_dict)

# 2. 模型推理（确定性）
with torch.no_grad():
    outputs = self.policy(obs_tensor, deterministic=True)
    action = outputs['actions'][0].cpu().numpy()

# 3. 转换为动作字典
actions_dict = self.convert_actions_to_dict(action, vehicle_ids, icv_ids)

# 4. 应用到SUMO
self.apply_control_actions(actions_dict)
```

### 5. 动作应用 (`apply_control_actions`)
- **加速度控制**: 通过 `traci.vehicle.setSpeed()` 实现
- **换道控制**: 通过 `traci.vehicle.changeLane()` 实现（概率性）

## 🎯 Checkpoint选择建议

根据之前的评估结果：

| Checkpoint | OCR | 推荐 |
|-----------|-----|------|
| `stage2_ppo_iter99.pth` | 54.39% | ✅ **推荐！** |
| `stage2_best.pth` | 54.05% | ⚠️ 次优 |

**推荐使用**:
```python
checkpoint_path = "checkpoints/ocr_max/stage2_ppo_iter99.pth"
```

## ⚠️ 注意事项

### 1. 提交前检查清单

- [ ] 确认 `USE_GUI = False`（必须！）
- [ ] 确认 `checkpoint_path` 指向最佳模型
- [ ] 确认能成功运行完整个仿真（3600步）
- [ ] 确认生成的 `submit.xlsx` 包含4个sheets
- [ ] 确认OCR > 0.50（至少优于基线）

### 2. 常见问题

**Q1: 提示 "CUDA out of memory"**
```python
# 修改 DEVICE = 'cpu'
DEVICE = 'cpu'
```

**Q2: 提示 "模型权重不存在"**
```bash
# 检查权重文件是否存在
ls -lh checkpoints/ocr_max/*.pth

# 如果不存在，使用评估脚本中的路径
checkpoint_path = "logs/ocr_max/stage2_ppo_iter99.pth"
```

**Q3: OCR结果为0或异常低**
- 检查观测转换是否正确
- 检查动作应用是否有效
- 查看控制日志（每100步输出ICV数量）

**Q4: SUMO启动失败**
```bash
# 检查SUMO环境变量
echo $SUMO_HOME

# 或使用完整路径启动
export SUMO_HOME=/usr/share/sumo
```

### 3. 性能优化

如果需要加速仿真：

```python
# 在 submit_solution.py 中修改
USE_GUI = False  # 无GUI模式（已默认）
DEVICE = 'cuda'  # 使用GPU推理

# 或减少控制频率（每2步控制一次）
if step % 2 == 0:
    self.apply_control_algorithm(step)
```

## 📈 预期结果

基于之前的评估：

- **OCR**: 约 0.54 (54%)
- **累计到达**: 约 130-150 辆车
- **仿真时间**: 约 3600 秒 (100%)
- **ICV数量**: 约 25-30 辆车 (10%渗透率)

## 🔍 调试建议

### 查看详细日志

在 `collect_step_data` 中，每100步会输出：

```
[步骤 100] 活跃: 34, 累计出发: 30, 累计到达: 5, ICV数量: 3
```

关键指标：
- **活跃**: 当前路网中的车辆数
- **累计出发**: 已出发的车辆总数
- **累计到达**: 已完成OD的车辆总数（OCR分子）
- **ICV数量**: 当前控制的车辆数

### 本地测试

```bash
# 使用GUI模式观察仿真
# 修改 USE_GUI = True
python submit_solution.py

# 观察车辆行为是否合理
```

## 📝 提交流程

1. **本地测试**: 确保脚本正常运行
2. **生成xlsx**: 运行 `python submit_solution.py`
3. **验证文件**: 检查 `competition_results/submit.xlsx`
4. **提交**: 上传 `submit.xlsx` 到官方评测系统

## 🎓 技术细节

### 观测空间

```
obs_dim = 321 = 32车辆 × 9特征 + 32全局统计 + 1车辆数
```

9维车辆特征：
- s, d: Frenet坐标（归一化）
- vs, vd: Frenet速度（归一化）
- speed: 真实速度（归一化）
- acceleration: 加速度（归一化）
- lane_index: 车道索引（归一化）
- angle: 角度（归一化）
- is_icv: 是否ICV（0/1）

### 动作空间

```
action_dim = 32车辆 × 2动作 = 64
```

2个动作：
- acceleration: 加速度 [-4.5, 2.0] m/s²
- lane_change: 换道概率 [0, 1]

### 控制频率

- 每个SUMO步（0.1s）调用一次模型
- 只控制ICV（10%渗透率）
- 使用确定性推理（deterministic=True）

## 📞 支持

如有问题，请检查：

1. 模型权重是否正确加载
2. SUMO环境是否正确配置
3. GPU内存是否充足（如使用CUDA）
4. 仿真是否正常运行完3600步

---

**祝比赛顺利！🏆**
