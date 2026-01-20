# 配置文件说明

## 📁 配置文件列表

### 1. `phase1_lite.yaml` - 主训练配置
**用途**：日常训练和开发使用的主配置文件

**特点**：
- ✅ 基于官方环境优化（10% ICV占比）
- ✅ 包含Stage 1（启发式引导）和Stage 2（PPO微调）的完整配置
- ✅ 适合快速迭代和实验

**使用场景**：
```bash
# Stage 1: 启发式引导
python train_smart_selector.py --config configs/phase1_lite.yaml

# Stage 2: PPO微调
python train_phase1_lite.py --config configs/phase1_lite.yaml --stage 2
```

**关键参数**：
- `icv_ratio: 0.10` - 控制10%车辆
- `max_steps: 36000` - 3600秒episode
- `瓶颈区域: 1200-2200米` - 覆盖J15、J17交叉口

---

### 2. `competition_optimized.yaml` - 比赛专用配置
**用途**：针对官方比赛环境深度优化的配置

**特点**：
- ✅ 完全匹配官方固定参数（流量9,900 veh/h，CV占比25%）
- ✅ 精确标定瓶颈区域（基于routes.xml分析）
- ✅ 包含多组对比实验配置（3%、5%、10%、25%）
- ✅ 详细的启发式规则和评估配置

**使用场景**：
```bash
# 完整训练流程
python train_smart_selector.py --config configs/competition_optimized.yaml

# 对比实验
python evaluate.py --config configs/competition_optimized.yaml --experiment optimized_10pct
python evaluate.py --config configs/competition_optimized.yaml --experiment aggressive_5pct
```

**关键优化**：
- 多瓶颈区域定义（E23/J5、E17/J15、E19/J17）
- 基于交通密度的动态ICV调整建议
- 完整的评估指标体系

---

## 🎯 配置文件选择指南

### 快速开发/实验 → 使用 `phase1_lite.yaml`
- 简洁清晰，易于修改
- 包含所有必要配置
- 适合日常训练

### 正式比赛/最终提交 → 使用 `competition_optimized.yaml`
- 深度优化
- 参数完全匹配官方环境
- 包含多组实验方案

---

## 📝 配置文件修改指南

### 修改ICV控制比例

```yaml
# 两个地方需要同步修改：
environment:
  icv_ratio: 0.10  # 环境中的ICV占比

model:
  gnn:
    top_k_ratio: 0.10  # 模型的Top-K选择比例
```

### 修改瓶颈区域

```yaml
heuristic_selector:
  bottleneck_s_min: 1200.0  # 起点（米）
  bottleneck_s_max: 2200.0  # 终点（米）
```

### 修改训练参数

```yaml
training:
  stage1:
    num_episodes: 100  # 数据收集episodes
    epochs: 20         # 训练轮数
  
  stage2:
    num_episodes: 200  # PPO训练episodes
    learning_rate: 0.0001
```

---

## 🔧 配置文件继承关系

```
competition_optimized.yaml (完整版，9.1KB)
    │
    ├─ 官方环境完整参数
    ├─ 多瓶颈区域定义
    ├─ 多组实验配置
    └─ 详细评估配置
    
phase1_lite.yaml (精简版，8.4KB)
    │
    ├─ 核心训练参数
    ├─ 简化的瓶颈定义
    └─ 基础评估配置
```

---

## 📊 推荐实验流程

### 1. 快速验证（使用phase1_lite.yaml）
```bash
# 快速训练验证想法
python train_smart_selector.py --config configs/phase1_lite.yaml --device cuda
```

### 2. 深度优化（使用competition_optimized.yaml）
```bash
# 完整训练流程
python train_smart_selector.py --config configs/competition_optimized.yaml --device cuda
```

### 3. 对比实验（修改配置文件）
```bash
# 创建不同占比的配置
cp configs/phase1_lite.yaml configs/test_5pct.yaml
# 修改 test_5pct.yaml 中的 icv_ratio: 0.05

# 训练对比
python train_smart_selector.py --config configs/test_5pct.yaml
```

---

## 🚀 快速开始

```bash
# 1. 默认配置（推荐新手）
python train_smart_selector.py --config configs/phase1_lite.yaml

# 2. 比赛优化配置（推荐最终训练）
python train_smart_selector.py --config configs/competition_optimized.yaml
```

选择任一配置，开始训练！🎯
