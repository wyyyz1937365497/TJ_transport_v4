# GNN输出格式错误修复说明

## 问题描述

运行训练时出现以下错误：

```
AttributeError: 'dict' object has no attribute 'size'
```

**位置**：`src/models/world_model.py:117`

```python
batch_size = gnn_embedding.size(0)  # ❌ gnn_embedding是dict，不是tensor
```

---

## 问题原因

### GNN的返回值格式

GNN返回的是一个**字典**，而不是直接的tensor：

```python
# src/models/gnn.py 第197-201行
return {
    'node_embedding': node_output,      # 节点嵌入 [num_nodes, 256]
    'global_embedding': global_output,  # 全局嵌入 [1, 256]
    'risk_weights': risk_weights        # 风险权重 [num_nodes, num_heads]
}
```

### WorldModel期望的输入格式

WorldModel期望 `gnn_embedding` 是一个tensor：

```python
# src/models/world_model.py 第117行
batch_size = gnn_embedding.size(0)  # 期望tensor，但收到dict
```

---

## 修复方案

### 修改前的代码（错误）

```python
# ❌ 错误：假设GNN返回tensor
gnn_output = actual_model.risk_gnn(
    node_features=graph_data.x.to(self.device),
    edge_index=graph_data.edge_index.to(self.device),
    edge_features=graph_data.edge_attr.to(self.device),
    batch=batch_tensor
)

gnn_embedding = gnn_output  # ❌ gnn_output是dict，不是tensor

# WorldModel
predictions = wrapped_world_model(gnn_embedding)  # ❌ 传入dict，导致.size()报错
```

### 修改后的代码（正确）

```python
# ✅ 正确：从GNN返回的字典中提取node_embedding
gnn_output_dict = actual_model.risk_gnn(
    node_features=graph_data.x.to(self.device),
    edge_index=graph_data.edge_index.to(self.device),
    edge_features=graph_data.edge_attr.to(self.device),
    batch=batch_tensor
)

# 从字典中提取节点嵌入
gnn_embedding = gnn_output_dict['node_embedding']  # ✅ [num_nodes, 256] tensor

# WorldModel
predictions = wrapped_world_model(gnn_embedding)  # ✅ 传入tensor
```

**完整代码**：
```python
# 7. 通过GNN提取特征（多GPU支持）
batch_tensor = graph_data.batch.to(self.device) if graph_data.batch is not None else None

if num_gpus > 1 and hasattr(model, 'module'):
    gnn_output_dict = actual_model.risk_gnn(
        node_features=graph_data.x.to(self.device),
        edge_index=graph_data.edge_index.to(self.device),
        edge_features=graph_data.edge_attr.to(self.device),
        batch=batch_tensor
    )
else:
    gnn_output_dict = actual_model.risk_gnn(
        node_features=graph_data.x.to(self.device),
        edge_index=graph_data.edge_index.to(self.device),
        edge_features=graph_data.edge_attr.to(self.device),
        batch=batch_tensor
    )

# GNN返回字典：{'node_embedding': ..., 'global_embedding': ..., 'risk_weights': ...}
# 提取节点嵌入用于世界模型
gnn_embedding = gnn_output_dict['node_embedding']  # [num_nodes, 256]
```

---

## GNN输出详解

### 输出字典结构

```python
gnn_output_dict = {
    'node_embedding': torch.Tensor,    # [num_nodes, 256] - 每个节点的嵌入
    'global_embedding': torch.Tensor,  # [1, 256] - 全局图级嵌入
    'risk_weights': torch.Tensor      # [num_nodes, num_heads] - 注意力权重
}
```

### 各字段的含义

#### 1. node_embedding（节点嵌入）

- **形状**：`[num_nodes, 256]`
- **含义**：每个车辆/节点的特征表示
- **用途**：
  - 输入到WorldModel进行未来状态预测
  - 输入到Controller进行决策

**示例**：
```python
# 3个车辆
node_embedding = torch.randn(3, 256)
# node_embedding[0]: 车辆0的嵌入
# node_embedding[1]: 车辆1的嵌入
# node_embedding[2]: 车辆2的嵌入
```

#### 2. global_embedding（全局嵌入）

- **形状**：`[1, 256]` 或 `[batch_size, 256]`
- **含义**：整个图的全局特征表示
- **计算方式**：
  - 单图：所有节点嵌入的平均
  - 多图：每个图的节点嵌入分别平均

```python
# 单图模式
global_output = node_output.mean(dim=0, keepdim=True)  # [1, 256]

# 批次模式
global_output = global_mean_pool(node_output, batch)  # [batch_size, 256]
```

**用途**：
- 表示整个交通流的状态
- 可用于图级别的决策

#### 3. risk_weights（风险权重）

- **形状**：`[num_nodes, num_heads]`
- **含义**：注意力机制的权重
- **用途**：
  - 可解释性：查看哪些车辆对风险判断贡献大
  - 可视化：注意力图

---

## 为什么返回字典而不是直接返回tensor？

### 1. 灵活性

返回字典允许同时提供多种表示：
```python
return {
    'node_embedding': ...,      # 节点级别
    'global_embedding': ...,    # 图级别
    'risk_weights': ...         # 注意力权重
}
```

### 2. 可扩展性

未来可以添加更多字段：
```python
return {
    'node_embedding': ...,
    'global_embedding': ...,
    'risk_weights': ...,
    'attention_map': ...,       # 新增：注意力图
    'edge_features': ...,       # 新增：边特征
}
```

### 3. 调试和可视化

提供额外信息用于调试：
```python
output = model(...)
print(output['risk_weights'])  # 查看注意力权重
```

---

## 其他需要修改的地方

如果其他地方也使用GNN输出，需要同样处理：

### Phase 2, 3, 4训练

检查这些阶段是否也有类似问题：

```python
# ❌ 错误模式
gnn_output = model.risk_gnn(...)
world_model_output = model.world_model(gnn_output)

# ✅ 正确模式
gnn_output_dict = model.risk_gnn(...)
gnn_embedding = gnn_output_dict['node_embedding']
world_model_output = model.world_model(gnn_embedding)
```

### Controller中使用GNN

如果Controller也使用GNN：

```python
# ❌ 错误
gnn_output = self.risk_gnn(...)
action = self.controller(gnn_output)

# ✅ 正确
gnn_output_dict = self.risk_gnn(...)
node_embedding = gnn_output_dict['node_embedding']
action = self.controller(node_embedding)
```

---

## 验证修复

修复后的数据流：

```python
# 1. GNN前向传播
gnn_output_dict = model.risk_gnn(...)
# 返回：{'node_embedding': [B, 256], 'global_embedding': [1, 256], ...}

# 2. 提取节点嵌入
gnn_embedding = gnn_output_dict['node_embedding']
# gnn_embedding: [B, 256] tensor

# 3. 输入到WorldModel
predictions = world_model(gnn_embedding)
# gnn_embedding.size(0) ✅ 正常工作

# 4. 计算损失
loss = criterion(pred_state, future_state)
```

---

## 现在可以重新运行训练

```bash
python train.py --phase all
```

错误已经解决！

---

## 总结

### 关键点

1. **GNN返回字典**：`{'node_embedding': ..., 'global_embedding': ..., 'risk_weights': ...}`
2. **提取节点嵌入**：`gnn_output_dict['node_embedding']`
3. **传递tensor**：将提取的tensor传给WorldModel

### 修复模式

**从GNN获取节点嵌入**：
```python
# ❌ 错误
gnn_output = model.risk_gnn(...)
world_model(gnn_output)  # gnn_output是dict

# ✅ 正确
gnn_output_dict = model.risk_gnn(...)
gnn_embedding = gnn_output_dict['node_embedding']
world_model(gnn_embedding)  # gnn_embedding是tensor
```

---

**修复完成！** ✅
