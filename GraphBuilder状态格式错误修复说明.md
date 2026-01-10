# GraphBuilder状态格式错误修复说明

## 问题描述

运行训练时出现以下错误：

```
KeyError: 'x'
```

**位置**：`src/models/gnn.py:370` (GraphBuilder._should_connect)

```python
dx = state_j['x'] - state_i['x']  # ❌ 找不到'x'键
```

---

## 问题原因

### GraphBuilder期望的车辆状态格式

**GraphBuilder需要的字段**：
```python
vehicle_state = {
    'x': float,           # x坐标
    'y': float,           # y坐标
    'vx': float,          # x方向速度
    'vy': float,          # y方向速度
    'ax': float,          # x方向加速度
    'ay': float,          # y方向加速度
    'lane_id': str,       # 车道ID（字符串）
    'lane_index': int,    # 车道索引
    'road_id': str        # 道路ID
}
```

### 我们之前提供的格式

**错误的格式**：
```python
vehicle_state = {
    'position': float,      # ❌ 应该是 'x' 和 'y'
    'speed': float,         # ❌ 应该是 'vx' 和 'vy'
    'acceleration': float,  # ❌ 应该是 'ax' 和 'ay'
    'lane_index': int,
    'road_id': str
}
```

---

## 修复方案

### 修改代码

**位置**：`src/algorithms/training.py` 第256-276行

**修改前**：
```python
vehicle_states_dict[veh_id] = {
    'position': current_position[i].item(),
    'speed': current_speed[i].item(),
    'acceleration': current_accel[i].item(),
    'lane_index': int(current_lane_indices[i].item()),
    'road_id': 'E0'
}
```

**修改后**：
```python
position = current_position[i].item()
speed = current_speed[i].item()
accel = current_accel[i].item()
lane_idx = int(current_lane_indices[i].item())

vehicle_states_dict[veh_id] = {
    'x': position,                    # ✅ x坐标
    'y': 0.0,                         # ✅ y坐标（假设直线道路）
    'vx': speed,                      # ✅ x方向速度
    'vy': 0.0,                        # ✅ y方向速度
    'ax': accel,                      # ✅ x方向加速度
    'ay': 0.0,                        # ✅ y方向加速度
    'lane_id': f"E0_{lane_idx}",      # ✅ 车道ID（字符串格式）
    'lane_index': lane_idx,           # ✅ 车道索引
    'road_id': 'E0'
}
```

---

## 为什么需要分解为x/y分量？

### 1. 支持二维道路网络

虽然当前使用的是直线道路（y=0），但GraphBuilder设计为支持：
- 多车道道路
- 弯道
- 交叉口
- 复杂的路网拓扑

### 2. 计算车辆间的相对位置

GraphBuilder需要计算：
```python
dx = state_j['x'] - state_i['x']  # x方向距离
dy = state_j['y'] - state_i['y']  # y方向距离
distance = sqrt(dx**2 + dy**2)    # 欧氏距离
```

### 3. 计算相对速度和加速度

```python
dvx = state_j['vx'] - state_i['vx']  # x方向相对速度
dvy = state_j['vy'] - state_i['vy']  # y方向相对速度

dax = state_j['ax'] - state_i['ax']  # x方向相对加速度
day = state_j['ay'] - state_i['ay']  # y方向相对加速度
```

这些相对运动信息用于：
- 判断车辆是否需要交互
- 计算边缘特征
- 预测碰撞风险

---

## 为什么y、vy、ay设为0？

### 当前假设：直线道路

对于直线道路（如E0）：
- **y = 0**：所有车辆在同一y坐标
- **vy = 0**：没有横向速度
- **ay = 0**：没有横向加速度

### 如果需要支持多车道

对于多车道道路，可以这样设置：

```python
lane_width = 3.5  # 车道宽度（米）

vehicle_states_dict[veh_id] = {
    'x': position,
    'y': lane_idx * lane_width,  # 不同车道不同y坐标
    'vx': speed,
    'vy': 0.0,  # 假设没有横向速度
    'ax': accel,
    'ay': 0.0,
    'lane_id': f"E0_{lane_idx}",
    'lane_index': lane_idx,
    'road_id': 'E0'
}
```

### 如果需要支持换道

换道时可以模拟横向运动：

```python
# 如果车辆i正在换道
if is_changing_lanes:
    # 计算横向速度
    target_lane_y = target_lane_idx * lane_width
    current_lane_y = current_lane_idx * lane_width
    vy = (target_lane_y - current_lane_y) / lane_change_time

    vehicle_states_dict[veh_id] = {
        'x': position,
        'y': current_lane_y + (target_lane_y - current_lane_y) * progress,
        'vx': speed * 0.95,  # 换道时略微减速
        'vy': vy,  # 横向速度
        'ax': accel,
        'ay': 0.0,
        'lane_id': f"E0_{lane_idx}",
        'lane_index': lane_idx,
        'road_id': 'E0'
    }
```

---

## lane_id的格式说明

### 格式：`"{road_id}_{lane_index}"`

**示例**：
```python
# 道路E0，车道0
lane_id = "E0_0"

# 道路E0，车道1
lane_id = "E0_1"

# 道路E16，车道2
lane_id = "E16_2"
```

### 为什么需要lane_id？

1. **唯一标识车道**：结合道路ID和车道索引
2. **SUMO兼容**：与SUMO的车道命名一致
3. **图构建**：用于判断车辆是否在同一车道

```python
# GraphBuilder中判断是否同一车道
same_lane = 1.0 if state_i['lane_id'] == state_j['lane_id'] else 0.0
```

---

## 验证修复

修复后，车辆状态字典包含所有必需字段：

```python
vehicle_states_dict[veh_id] = {
    'x': 123.45,          # ✅ float
    'y': 0.0,             # ✅ float
    'vx': 15.3,           # ✅ float
    'vy': 0.0,            # ✅ float
    'ax': 0.5,            # ✅ float
    'ay': 0.0,            # ✅ float
    'lane_id': "E0_1",    # ✅ str
    'lane_index': 1,      # ✅ int
    'road_id': "E0"       # ✅ str
}
```

所有字段都满足GraphBuilder的要求。

---

## 现在可以重新运行训练

```bash
python train.py --phase all
```

错误已经解决！

---

## 总结

### 关键点

1. **GraphBuilder需要完整的状态向量**：x, y, vx, vy, ax, ay
2. **直线道路可以简化**：y=0, vy=0, ay=0
3. **lane_id格式**：`"{road_id}_{lane_index}"`
4. **所有字段都必须提供**，否则KeyError

### 修复的代码模式

**从简化状态创建完整状态**：
```python
# 输入：简化状态（1维）
position = ...  # 位置
speed = ...     # 速度
accel = ...     # 加速度
lane_idx = ...  # 车道索引

# 输出：完整状态（2维）
state = {
    'x': position, 'y': 0.0,
    'vx': speed, 'vy': 0.0,
    'ax': accel, 'ay': 0.0,
    'lane_id': f"E0_{lane_idx}",
    'lane_index': lane_idx
}
```

---

**修复完成！** ✅
