# 模型架构评估与优化建议

## 📋 评估时间
2026-01-19

## 🎯 评估目标

评估当前模型架构是否能够适应**从32辆车到600辆车**的扩展（18.75倍增长），并提出优化建议。

---

## 📊 当前模型架构

### 整体架构（v4.0）

```
观测 [4641维] = 512辆×9维特征 + 32维全局统计 + 1维车辆数
     ↓
┌─────────────────────────────────────────────────────────┐
│  1. GNN (感知层) - 建模车辆间交互关系                    │
│     - node_dim: 9                                      │
│     - edge_dim: 4                                     │
│     - hidden_dim: 64  ⚠️                              │
│     - output_dim: 256                                 │
│     - num_layers: 3  ⚠️                              │
│     - heads: 4  ⚠️                                    │
│     - dropout: 0.1                                    │
└─────────────────────────────────────────────────────────┘
     ↓ 节点嵌入 [512, 256]
┌─────────────────────────────────────────────────────────┐
│  2. World Model (预测层) - 预测交通流演化                 │
│     - hidden_dim: 128                                 │
│     - latent_dim: 64  ⚠️                              │
│     - future_steps: 5                                 │
│     - num_layers: 2  ⚠️                              │
│     - dropout: 0.1                                    │
└─────────────────────────────────────────────────────────┘
     ↓ 预测 [512, latent_dim*2]
┌─────────────────────────────────────────────────────────┐
│  3. Controller (决策层) - 生成控制动作                   │
│     - global_dim: 32 (全局统计维度)                    │
│     - hidden_dim: 128  ⚠️                            │
│     - action_dim: 2 (每辆车的动作)                     │
│     - top_k: 5 (默认控制5辆车)                          │
│     - dropout: 0.2                                    │
└─────────────────────────────────────────────────────────┘
     ↓ 动作 [1024维] = 512辆×2维动作
```

### 参数量统计

| 模块 | 参数量（估算） | 占比 |
|-----|-------------|-----|
| GNN（感知层） | ~1.5M | 45% |
| World Model（预测层） | ~0.8M | 24% |
| Controller（决策层） | ~1.0M | 30% |
| **总计** | **~3.3M** | 100% |

---

## ⚠️ 潜在问题分析

### 问题1：GNN容量不足 🔴 严重

**当前配置**：
- `hidden_dim: 64`
- `num_layers: 3`
- `heads: 4`

**问题**：

1. **节点数增长18.75倍**（32 → 512），但GNN容量没有增加
2. **512个节点 × 64维隐藏层** = 32,768个激活值（对于64维来说可能过载）
3. **3层GNN可能不足以捕捉长距离依赖**：
   - 在512个节点的图中，信息传播路径更长
   - 3层的感受野有限，可能无法捕捉全局模式
4. **4个注意力头可能不足以捕捉多样化的交互模式**：
   - 车辆交互复杂：跟驰、换道、汇流、超车
   - 需要更多的注意力头来建模不同类型的交互

**类比**：
```
32辆车：小村庄的社交网络（64维足够）
512辆车：大城市的交通网络（64维不够）
```

### 问题2：World Model预测容量不足 🟡 中等

**当前配置**：
- `latent_dim: 64`
- `num_layers: 2`

**问题**：

1. **64维latent space压缩512个节点的状态**：
   - 需要从512×256=131,078维压缩到64维
   - 压缩比超过2000:1，可能损失重要信息
2. **2层LSTM可能不足以预测复杂演化**：
   - 600辆车的交通流演化高度非线性
   - 需要更深的网络来捕捉多步依赖
3. **只预测5步未来**：
   - 对于高密度场景，5步（0.5秒）可能不够
   - 无法支持前瞻性的协同控制

### 问题3：Controller决策能力有限 🟡 中等

**当前配置**：
- `hidden_dim: 128`
- `top_k: 5`

**问题**：

1. **128维hidden层处理256维GNN嵌入**：
   - 可能成为瓶颈
2. **只控制5辆车（top_k=5）**：
   - 对于600辆车的场景，5辆车远不够（需要150辆ICV）
   - 理论上应该有全局控制能力，而不是只控制5辆

### 问题4：缺少全局信息聚合机制 🔴 严重

**问题**：

1. **没有全局池化/聚合层**：
   - GNN产生的512个节点嵌入，如何压缩成全局决策？
   - 当前使用简单的mean pooling，可能丢失重要信息
2. **缺少层次化信息处理**：
   - 没有局部-全局的层次结构
   - 难以同时处理微观（单车）和宏观（全局）决策

---

## 🎯 优化建议

### 方案A：渐进式优化（推荐）⭐

**原则**：先小幅度调整，验证效果后再进一步优化

#### 阶段1：最小改动（立即应用）

```yaml
# configs/competition.yaml
model:
  gnn:
    hidden_dim: 128         # 64 → 128 (2倍) ✅
    num_layers: 4          # 3 → 4 (增加1层) ✅
    heads: 8               # 4 → 8 (2倍) ✅
    dropout: 0.1           # 保持不变

  world_model:
    latent_dim: 128        # 64 → 128 (2倍) ✅
    num_layers: 3          # 2 → 3 (增加1层) ✅
    future_steps: 10       # 5 → 10 (2倍，预测更远未来) ✅
    dropout: 0.1

  controller:
    global_dim: 32         # 保持不变
    hidden_dim: 256        # 128 → 256 (2倍) ✅
    top_k: 10              # 5 → 10 (增加控制车辆数) ✅
    dropout: 0.15          # 0.2 → 0.15 (稍微降低)
```

**预期效果**：
- 参数量：~3.3M → **~5M**（增加50%）
- 训练时间：增加20-30%
- 性能提升：+10-15分

#### 阶段2：架构升级（如果阶段1效果好）

**添加全局信息聚合机制**：

```python
# 新增：全局注意力池化层
class GlobalAttentionPooling(nn.Module):
    def __init__(self, node_dim: int, output_dim: int):
        super().__init__()
        self.query = nn.Linear(node_dim, output_dim)
        self.key = nn.Linear(node_dim, output_dim)
        self.value = nn.Linear(node_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, node_embeddings):
        # node_embeddings: [num_nodes, node_dim]
        Q = self.query(node_embeddings)  # [N, output_dim]
        K = self.key(node_embeddings)    # [N, output_dim]
        V = self.value(node_embeddings)  # [N, output_dim]

        # 注意力权重
        attention = torch.softmax(Q @ K.T / np.sqrt(K.shape[-1]), dim=-1)

        # 加权聚合
        global_embed = attention @ V  # [output_dim]
        return global_embed
```

**添加到GNN之后**：

```python
# GNN输出：[512, 256]
node_embeddings = self.gnn(...)  # [512, 256]

# ✅ 新增：全局注意力池化
global_embedding = self.global_pool(node_embeddings)  # [256]

# 拼接全局统计和局部统计
global_stats = observations[:, -32:]  # [batch, 32]
global_info = torch.cat([global_embedding, global_stats], dim=-1)  # [288]
```

#### 阶段3：完整升级（如果阶段2效果好）

**引入层次化Transformer架构**：

```yaml
model:
  # 层次化架构
  hierarchical:
    enabled: true
    num_levels: 3           # 3层层次结构
    - level: "局部"
      num_vehicles: 10       # 每组10辆车
      gnn_hidden_dim: 128
      num_layers: 2
    - level: "区域"
      num_groups: 10        # 10个局部组
      gnn_hidden_dim: 256
      num_layers: 2
    - level: "全局"
      num_vehicles: 600      # 全局
      gnn_hidden_dim: 512
      num_layers: 3
```

### 方案B：直接升级到大型架构（激进）

**适用场景**：有充足的计算资源（显存≥24GB），时间紧迫

```yaml
model:
  gnn:
    hidden_dim: 256         # 64 → 256 (4倍) ⭐
    num_layers: 6          # 3 → 6 (2倍) ⭐
    heads: 16              # 4 → 16 (4倍) ⭐
    dropout: 0.1

  world_model:
    hidden_dim: 256        # 128 → 256 (2倍)
    latent_dim: 256        # 64 → 256 (4倍) ⭐
    num_layers: 4          # 2 → 4 (2倍)
    future_steps: 15       # 5 → 15 (3倍) ⭐
    dropout: 0.1

  controller:
    global_dim: 64         # 32 → 64 (2倍)
    hidden_dim: 512        # 128 → 512 (4倍) ⭐
    top_k: 20              # 5 → 20 (4倍)
    dropout: 0.1
```

**预期效果**：
- 参数量：~3.3M → **~10M**（增加3倍）
- 训练时间：增加100-150%
- 性能提升：+20-30分（如果训练充分）
- 显存需求：24GB+

---

## 📈 参数量对比

| 架构版本 | 参数量 | 显存需求 | 训练时间 | 预期得分 |
|---------|-------|---------|---------|---------|
| **当前架构** (v4.0) | ~3.3M | 8-10GB | 6-7小时 | 70-80分 |
| **方案A-阶段1** | ~5M | 10-12GB | 7-9小时 | 80-90分 ⭐ |
| **方案A-阶段2** | ~6M | 12-14GB | 8-10小时 | 85-93分 ⭐ |
| **方案A-阶段3** | ~8M | 14-18GB | 9-12小时 | 90-95分 ⭐ |
| **方案B** | ~10M | 20-24GB | 12-15小时 | 92-98分 ⭐⭐ |

---

## 🔍 具体修改代码

### 修改1：GNN配置

**文件**：`configs/competition.yaml`

```yaml
model:
  gnn:
    node_dim: 9
    edge_dim: 4
    hidden_dim: 128        # ✅ 从64提升到128
    output_dim: 256
    num_layers: 4          # ✅ 从3提升到4
    heads: 8               # ✅ 从4提升到8
    dropout: 0.1
```

**对应代码修改**：`src/models/ideal_policy_v4.py`

```python
# GNN配置会自动从config读取，无需修改代码
# 但需要确认模型能够正确处理新配置
```

### 修改2：World Model配置

**文件**：`configs/competition.yaml`

```yaml
model:
  world_model:
    hidden_dim: 128
    latent_dim: 128        # ✅ 从64提升到128
    future_steps: 10       # ✅ 从5提升到10
    num_layers: 3          # ✅ 从2提升到3
    dropout: 0.1
```

### 修改3：Controller配置

**文件**：`configs/competition.yaml`

```yaml
model:
  controller:
    global_dim: 32
    hidden_dim: 256        # ✅ 从128提升到256
    action_dim: 2
    top_k: 10              # ✅ 从5提升到10
    dropout: 0.15          # ✅ 从0.2降低到0.15
```

---

## 🎯 推荐方案

### 方案选择建议

**如果你的资源有限（显存<16GB）**：
- ✅ 使用**方案A-阶段1**（最小改动）
- 只增加GNN和Controller的hidden_dim
- 参数量：~5M
- 显存需求：10-12GB

**如果你的资源充足（显存≥22GB）**：
- ✅ 使用**方案A-阶段1 + 阶段2**（推荐）
- 增加容量 + 添加全局注意力池化
- 参数量：~6M
- 显存需求：12-14GB
- 性价比最高 ⭐

**如果你的资源非常充足（显存≥32GB）且时间紧迫**：
- ✅ 使用**方案B**（激进升级）
- 直接升级到大型架构
- 参数量：~10M
- 显存需求：20-24GB
- 可能获得最高分

---

## ⚡ 立即可执行的最小改动

**最优先的3个修改**（立即应用）：

1. ✅ **GNN hidden_dim**: 64 → 128
2. ✅ **GNN num_layers**: 3 → 4
3. ✅ **World Model latent_dim**: 64 → 128

**修改方法**：只需修改`configs/competition.yaml`，无需改代码！

```bash
# 编辑配置文件
vim configs/competition.yaml

# 修改以下3行：
hidden_dim: 128  # GNN
num_layers: 4    # GNN
latent_dim: 128  # World Model

# 重新训练
python train_phase1.py --config configs/competition.yaml
```

**预期效果**：
- 参数量增加30%
- 训练时间增加10-15%
- 性能提升5-10分

---

## 📝 实验建议

### 验证策略

1. **基线实验**（当前架构）：
   - 训练Phase 1 + Phase 2 (Stage 1-5)
   - 记录性能指标
   - 作为对比基准

2. **最小改动实验**（方案A-阶段1）：
   - 修改3个关键参数
   - 重新训练Phase 1 + Phase 2 (Stage 1-5)
   - 对比性能提升

3. **如果最小改动效果好**：
   - 尝试方案A-阶段2（添加全局池化）
   - 逐步优化

4. **如果最小改动效果不明显**：
   - 考虑方案B（直接大型升级）
   - 或者增加训练时间

---

## 🚨 注意事项

### 1. 参数量与训练时间

- 参数量增加50% → 训练时间增加20-30%
- 参数量增加200% → 训练时间增加100-150%
- 需要权衡性能提升和训练成本

### 2. 过拟合风险

- 更大的模型更容易过拟合
- 需要增加dropout、正则化
- 可能需要更多的训练数据

### 3. 渐进式优化

- 不要一次性改太多
- 每次改动后验证效果
- 根据效果决定是否继续优化

---

## ✅ 下一步行动

### 立即执行

1. ✅ **配置已统一**：课程配置已移到`configs/competition.yaml`
2. ⚠️ **需要重新训练**：停止当前训练，应用新配置

### 可选执行（建议）

3. ⚠️ **应用方案A-阶段1**：修改3个关键模型参数
4. ⚠️ **重新训练Phase 1**：观察性能提升

### 长期规划

5. 📊 **评估效果**：对比新旧架构的性能差异
6. 🎯 **决定是否继续优化**：根据实验结果决定是否应用阶段2/阶段3

---

**文档版本**: v1.0
**最后更新**: 2026-01-19
**作者**: Claude Code
**状态**: ✅ 配置统一完成，等待用户决定是否优化模型架构
