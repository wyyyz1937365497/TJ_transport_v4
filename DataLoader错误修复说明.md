# DataLoader错误修复说明

## 问题描述

在运行 `python train.py --phase all` 时出现以下错误：

```
TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <U7
```

## 问题原因

**根本原因**：数据集中包含了numpy的Unicode字符串类型（`<U7`），PyTorch的DataLoader无法处理这种类型。

具体问题：
- `lane_id` 是字符串类型（如 "E0_1"）
- 当转换为numpy数组时，变成了Unicode字符串数组
- DataLoader的collate函数不支持Unicode字符串类型

## 修复方案

### 1. 移除字符串类型的 `lane_id` 字段

**修改文件**：`src/env/data_collector.py`

**修改内容**：

#### 初始化轨迹时（第142-149行）

**之前**：
```python
self.trajectories[veh_id] = {
    'id': veh_id,
    'timestamps': [],
    'positions': [],
    'speeds': [],
    'accelerations': [],
    'lane_ids': [],      # ❌ 字符串类型
    'lanes': []
}
```

**之后**：
```python
self.trajectories[veh_id] = {
    'id': veh_id,
    'timestamps': [],
    'positions': [],
    'speeds': [],
    'accelerations': [],
    'lane_indices': []  # ✅ 只保留整数类型的lane_index
}
```

#### 记录状态时（第174-175行）

**之前**：
```python
self.trajectories[veh_id]['lane_ids'].append(state['lane_id'])  # ❌ 字符串
self.trajectories[veh_id]['lanes'].append(state['lane_index'])
```

**之后**：
```python
# 只存储lane_index（整数），不存储lane_id（字符串）
self.trajectories[veh_id]['lane_indices'].append(state['lane_index'])  # ✅ 整数
```

#### 构建样本时（第362-424行）

**之前**：
```python
lane_ids = np.array(traj['lane_ids'])  # ❌ 变成Unicode数组
lane_indices = np.array(traj['lanes'])

samples.append({
    'current': current_seq,
    'future': future_seq,
    'current_lane_ids': current_lane_ids,      # ❌ 字符串
    'current_lane_indices': current_lane_indices,
    'current_timestamps': current_timestamps,
    'future_lane_ids': future_lane_ids,        # ❌ 字符串
    'future_lane_indices': future_lane_indices,
    'future_timestamps': future_timestamps
})
```

**之后**：
```python
# 使用lane_indices（整数），不使用lane_ids（字符串）
lane_indices = np.array(traj['lane_indices'])  # ✅ 整数数组

samples.append({
    'current': current_seq,
    'future': future_seq,
    'current_lane_indices': current_lane_indices,  # ✅ 只保留整数
    'current_timestamps': current_timestamps,
    'future_lane_indices': future_lane_indices,
    'future_timestamps': future_timestamps
})
```

### 2. 更新训练代码

**修改文件**：`src/algorithms/training.py`

**修改内容**：

#### 提取特征时（第196-214行）

**之前**：
```python
current_lane_ids = batch['current_lane_ids']      # ❌ 字符串
current_lane_indices = batch['current_lane_indices']
```

**之后**：
```python
current_lane_indices = batch['current_lane_indices']  # ✅ 只使用整数
```

#### 构建车辆状态时（第254-260行）

**之前**：
```python
vehicle_states_dict[veh_id] = {
    'position': float(current_position[i].cpu().numpy()),
    'speed': float(current_speed[i].cpu().numpy()),
    'acceleration': float(current_accel[i].cpu().numpy()),
    'lane_id': str(int(current_lane_ids[i])),  # ❌ 字符串
    'lane_index': int(current_lane_indices[i]),
    'road_id': 'E0'
}
```

**之后**：
```python
vehicle_states_dict[veh_id] = {
    'position': float(current_position[i].cpu().numpy()),
    'speed': float(current_speed[i].cpu().numpy()),
    'acceleration': float(current_accel[i].cpu().numpy()),
    'lane_index': int(current_lane_indices[i]),  # ✅ 只使用lane_index
    'road_id': 'E0'
}
```

## 为什么可以这样修复？

### lane_id vs lane_index

**lane_id**（字符串）：
- 示例：`"E0_1"`, `"E0_2"`
- SUMO内部使用
- 包含道路ID和车道编号
- **不适合作为特征输入**

**lane_index**（整数）：
- 示例：`0`, `1`, `2`
- 车道在道路上的索引
- **适合作为特征输入**

### 为什么只需要lane_index？

1. **数值型特征**：神经网络更适合处理整数
2. **唯一标识**：在单个道路场景中，lane_index已经能唯一标识车道
3. **简化处理**：避免字符串编码的复杂性

### 如果真的需要lane_id怎么办？

如果后续需要使用lane_id，可以使用以下方法：

1. **字符串编码**：将lane_id映射为整数
   ```python
   lane_id_to_index = {"E0_0": 0, "E0_1": 1, "E0_2": 2}
   lane_index = lane_id_to_index[lane_id]
   ```

2. **嵌入层**：使用nn.Embedding处理字符串
   ```python
   class LaneEmbedding(nn.Module):
       def __init__(self, num_lanes, embed_dim):
           self.embedding = nn.Embedding(num_lanes, embed_dim)
   ```

## 验证修复

修复后，数据集中只包含：
- ✅ 浮点数数组：`positions`, `speeds`, `accelerations`, `timestamps`
- ✅ 整数数组：`lane_indices`
- ❌ ~~字符串数组：`lane_ids`~~（已移除）

所有类型都是PyTorch DataLoader支持的类型。

---

## 现在可以重新运行训练

```bash
python train.py --phase all
```

错误应该已经解决了！

---

**修复的文件**：
1. ✅ `src/env/data_collector.py` - 移除lane_ids字段
2. ✅ `src/algorithms/training.py` - 更新特征提取代码
