# 快速测试指南

本文档说明如何使用快速测试功能验证训练流程是否正常工作。

## 设计原则

**重要**: 快速测试配置 (`configs/quick_test.yaml`) 保持了与标准配置**完全相同的神经网络架构**，仅减少了数据量和训练轮数。这确保了：

- ✅ 快速验证训练流程的完整性
- ✅ 测试真实的神经网络结构（无简化）
- ✅ 快速迭代和调试配置问题
- ✅ 在短时间内完成完整4阶段训练流程

### 神经网络配置对比

| 配置项 | 标准配置 | 快速测试 | 是否相同 |
|--------|---------|---------|---------|
| **GNN节点维度** | 9 | 9 | ✅ |
| **GNN边维度** | 4 | 4 | ✅ |
| **GNN隐藏维度** | 64 | 64 | ✅ |
| **GNN输出维度** | 256 | 256 | ✅ |
| **GNN层数** | 3 | 3 | ✅ |
| **注意力头数** | 4 | 4 | ✅ |
| **World Model隐藏维度** | 128 | 128 | ✅ |
| **未来步数** | 5 | 5 | ✅ |
| **Controller隐藏维度** | 128 | 128 | ✅ |
| **Top-K选择** | 5 | 5 | ✅ |

### 训练参数对比

| 配置项 | 标准配置 | 快速测试 | 缩减比例 |
|--------|---------|---------|---------|
| **Phase 1 Episodes** | 20 | 2 | 10% |
| **Phase 1 Epochs** | 20 | 3 | 15% |
| **Phase 2步数** | 100,000 | 1,000 | 1% |
| **Phase 2环境数** | 4 | 2 | 50% |
| **Phase 3步数** | 50,000 | 500 | 1% |
| **Phase 4步数** | 50,000 | 500 | 1% |
| **每个Episode最大步数** | 3600 | 300 | 8% |

## 使用方法

### 方法1: 使用quick_train.py (推荐)

一键测试所有阶段：

```bash
# 测试单个阶段
python quick_train.py --phase 1

# 测试所有阶段（依次执行）
python quick_train.py --phase all

# 测试阶段2-4（如果已有Phase 1权重）
python quick_train.py --phase 2-4
```

### 方法2: 使用标准训练脚本

使用快速测试配置文件运行标准训练脚本：

```bash
# 运行完整4阶段训练
python train.py --config configs/quick_test.yaml

# 仅运行Phase 1
python train.py --config configs/quick_test.yaml --phase 1

# 仅运行Phase 2（需要Phase 1权重）
python train.py --config configs/quick_test.yaml --phase 2
```

### 方法3: 使用SB3加速版本

```bash
# Phase 2 SB3加速
python train_sb3.py --config configs/quick_test.yaml

# Phase 3 SB3加速
python train_sb3_phase3.py --config configs/quick_test.yaml
```

## 快速测试预期输出

### Phase 1: World Model预训练

```
预计时间: 3-5分钟
 Episodes: 2 (每个episode约1-2分钟)
 Epochs: 3

输出:
 - checkpoints/world_model_phase1.pth
 - 训练日志显示loss下降
```

### Phase 2: PPO训练

```
预计时间: 2-3分钟
 总步数: 1,000
 环境数: 2

输出:
 - checkpoints/ppo_phase2.pth
 - 平均reward逐步提升
```

### Phase 3: 端到端微调

```
预计时间: 1-2分钟
 总步数: 500

输出:
 - checkpoints/e2e_phase3.pth
 - 联合优化完成
```

### Phase 4: 约束优化

```
预计时间: 1-2分钟
 总步数: 500

输出:
 - checkpoints/final_model.pth
 - Cost指标下降
```

## 总预计时间

- **单阶段测试**: 约5分钟
- **完整4阶段**: 约10-15分钟

## 快速测试用途

### 1. 验证训练流程

```bash
python quick_train.py --phase all
```

检查所有阶段是否能正常完成，确认：
- 配置文件正确加载
- 环境创建成功
- 模型训练正常
- 权重保存正确

### 2. 调试单个阶段

```bash
# 调试Phase 1
python quick_train.py --phase 1

# 调试Phase 2
python quick_train.py --phase 2
```

### 3. 验证代码更改

修改代码后，运行快速测试确保没有破坏现有功能：

```bash
python quick_train.py --phase all
```

### 4. 测试新配置

创建新的配置文件后，先用快速测试验证：

```bash
# 1. 复制快速测试配置
cp configs/quick_test.yaml configs/my_test.yaml

# 2. 修改my_test.yaml（保持神经网络配置不变）

# 3. 运行测试
python train.py --config configs/my_test.yaml
```

## 常见问题

### Q: 快速测试能代表真实训练效果吗？

A: **不能**。快速测试仅用于验证流程和代码正确性，不能用于评估模型性能。因为：
- 数据量太少（1-10%）
- 训练步数不足
- 模型不会收敛

**正确用法**:
- 快速测试 → 验证流程 ✓
- 标准训练 → 获得性能 ✓

### Q: 为什么保持神经网络配置不变？

A: 保持相同架构确保：
1. 测试真实的数据流和计算
2. 验证内存占用是否合理
3. 检查所有组件是否正确集成
4. 发现潜在的架构问题

### Q: 快速测试失败怎么办？

1. **查看错误信息**: 定位具体阶段和错误
2. **检查SUMO配置**: 确认路径正确
3. **验证依赖**: 确保所有包已安装
4. **查看日志**: logs/目录下的详细日志

### Q: 如何调整快速测试参数？

编辑 `configs/quick_test.yaml`，但建议：
- ✅ 可以调整: 训练步数、episodes数、环境数
- ⚠️ 谨慎调整: batch_size、学习率
- ❌ 不要调整: 神经网络架构（GNN维度、层数等）

## 下一步

快速测试通过后，可以进行标准训练：

```bash
# Windows标准配置
python train.py --config configs/windows_base.yaml

# Windows高性能配置
python train.py --config configs/windows_high_performance.yaml

# Windows极限性能配置
python train.py --config configs/windows_extreme_performance.yaml
```

详见 `TRAINING_GUIDE.md` 获取完整训练指南。
