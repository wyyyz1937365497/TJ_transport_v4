# SUMO错误修复说明

## 问题描述

在运行 `python train.py --phase all` 时出现以下错误：

```
Error: Answered with error to command 0xa4: Vehicle '('f_main_full.0', 0.8261915134695084)' is not known.
```

## 问题原因

**根本原因**：`traci.vehicle.getLeader()` 返回的是一个元组 `(leader_id, distance)`，但代码错误地将整个元组当作vehicle_id使用。

### 错误的代码（之前）

```python
# src/env/sumo_env.py 第350行
leader_id = traci.vehicle.getLeader(veh_id, 100.0)  # 返回 (veh_id, distance)
if leader_id:
    leader_speed = traci.vehicle.getSpeed(leader_id)  # ❌ 传入元组，导致错误
```

### 正确的代码（修复后）

```python
# src/env/sumo_env.py 第350-354行
leader_info = traci.vehicle.getLeader(veh_id, 100.0)  # 返回 (leader_id, distance)
if leader_info:
    leader_id = leader_info[0]  # ✅ 提取vehicle_id
    distance = leader_info[1]    # ✅ 提取distance
```

## 修复内容

**文件**：`src/env/sumo_env.py`

**位置**：第345-368行

**修改**：
1. ✅ 正确处理 `getLeader()` 返回的元组
2. ✅ 分别提取 `leader_id` 和 `distance`
3. ✅ 简化距离计算（直接使用getLeader返回的distance）

## 验证修复

现在可以重新运行训练：

```bash
python train.py --phase all
```

错误应该已经解决。

## 其他说明

### getLeader API 说明

根据SUMO文档：
```python
traci.vehicle.getLeader(veh_id, distance)
```

**返回值**：
- 如果有前车：`(leader_id, distance_to_leader)` 元组
- 如果无前车：`None`

**参数**：
- `veh_id`：当前车辆ID
- `distance`：查找距离（米）

### 正确的使用方式

```python
leader_info = traci.vehicle.getLeader(veh_id, 100.0)
if leader_info:
    leader_id = leader_info[0]      # 前车ID
    distance = leader_info[1]        # 与前车的距离

    # 使用leader_id进行后续操作
    leader_speed = traci.vehicle.getSpeed(leader_id)  # ✅ 正确
```

---

**问题已修复！** ✅

可以继续训练了。
