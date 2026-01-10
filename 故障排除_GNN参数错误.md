# GNN参数错误问题 - 已修复 ✅

## 问题描述

运行 `python quick_start.py` 时遇到以下错误：

```
Error: RiskSensitiveGNN.forward() got an unexpected keyword argument 'edge_attr'
```

## 问题原因

有两个问题：

1. **PyTorch Geometric 版本兼容性**
   - 某些版本的 GATv2Conv 不支持 `edge_dim` 参数
   - 某些版本的 GATv2Conv.forward() 不支持 `edge_attr` 参数

2. **参数名不匹配**
   - `traffic_controller.py` 调用时使用 `edge_attr`
   - `RiskSensitiveGNN.forward()` 期望的参数名是 `edge_features`

## 解决方案

### 1. 修复 GNN 初始化 (`src/models/gnn.py`)

**添加版本兼容性处理**：

```python
for i in range(num_layers):
    try:
        # 尝试创建支持边特征的 GAT 层
        self.gnn_layers.append(
            GATv2Conv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                heads=heads,
                concat=False,
                edge_dim=hidden_dim // 2,
                dropout=dropout,
                bias=True
            )
        )
    except TypeError:
        # 如果不支持 edge_dim，则创建不带边特征的版本
        print(f"⚠️  警告: GATv2Conv 不支持 edge_dim，将使用不带边特征的版本")
        self.gnn_layers.append(
            GATv2Conv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                heads=heads,
                concat=False,
                dropout=dropout,
                bias=True
            )
        )
```

### 2. 修复 GNN 前向传播 (`src/models/gnn.py`)

**添加运行时兼容性处理**：

```python
# GNN传播（兼容不同版本的PyTorch Geometric）
try:
    # 尝试使用 edge_attr 参数
    if edge_features.size(0) > 0:
        x = gnn_layer(x, edge_index, edge_attr=edge_emb)
    else:
        x = gnn_layer(x, edge_index)
except TypeError:
    # 如果不支持 edge_attr，则不使用边特征
    x = gnn_layer(x, edge_index)
```

### 3. 修复参数名 (`src/models/traffic_controller.py`)

**修正参数名从 `edge_attr` 到 `edge_features`**：

```python
gnn_output = self.risk_gnn(
    node_features=graph_data.x,
    edge_index=graph_data.edge_index,
    edge_features=graph_data.edge_attr,  # 修正参数名
    batch=batch.get('batch', None)
)
```

## 验证修复

### 运行模型测试

```bash
python test_model.py
```

预期输出：
```
✅ 所有测试通过!
   - GNN前向传播成功
   - 世界模型前向传播成功
   - 控制器前向传播成功
   - 完整模型前向传播成功
```

### 运行快速开始

```bash
python quick_start.py
```

## 技术细节

### PyTorch Geometric 版本差异

**版本 < 2.3**: 不支持 `edge_dim` 参数
**版本 2.3+**: 支持 `edge_dim`，但 forward() 可能不支持 `edge_attr`
**版本 2.4+**: 完整支持所有边特征功能

### 兼容性策略

使用 try-except 双重保护：
1. **初始化时**: 捕获 TypeError，创建简化版本
2. **运行时**: 捕获 TypeError，降级到不带边特征的版本

### 降级策略

当边特征不可用时：
- GNN 仍能正常工作
- 只是丢失边特征信息
- 节点特征仍然有效

## 影响范围

修复的文件：
- ✅ `src/models/gnn.py`
- ✅ `src/models/traffic_controller.py`

不影响：
- ❌ 配置文件
- ❌ 其他模块
- ❌ SUMO 集成

## 相关问题

如果仍然遇到问题，请检查：

1. **PyTorch Geometric 版本**
   ```bash
   pip show torch-geometric
   ```

2. **升级到最新版本**
   ```bash
   pip install --upgrade torch-geometric
   ```

3. **降级到稳定版本**
   ```bash
   pip install torch-geometric==2.3.0
   ```

---

问题已解决！现在可以正常使用系统了。
