# 智能交通协同控制系统

基于深度强化学习的智能交通信号优化系统，用于智能交通协同控制竞赛。

## 项目概述

本项目实现了一个完整的交通控制神经网络系统，通过AI算法优化城市交通流，提升交通效率同时控制干预成本。

### 核心特性

- **Risk-Sensitive GNN**: 感知层，捕捉交通交互和风险模式
- **Progressive World Model**: 预测层，5步前瞻性状态预测
- **Influence-Driven Controller**: 决策层，智能选择Top-K关键车辆控制
- **Dual-mode Safety Shield**: 安全层，确保100%控制指令安全
- **Constrained RL**: 约束优化，自动平衡性能与成本

### 系统架构

```
┌─────────────────────────────────────────────────────────┐
│           智能交通协同控制系统 v4.0                        │
├─────────────┬─────────────┬─────────────┬───────────────┤
│   感知层     │   预测层     │   决策层     │    安全层      │
│ Risk-Sensitive│ Progressive│ Influence  │ Dual-mode     │
│     GNN      │ World Model │-Driven     │ Safety Shield │
│  (交通感知)   │ (5步预测)   │ Controller  │  (动作裁剪)    │
└─────────────┴─────────────┴─────────────┴───────────────┘
```

## 安装指南

### 1. 环境要求

- Python 3.8+
- CUDA 11.0+ (可选，用于GPU加速)
- Windows 10/11
- SUMO仿真器

### 2. SUMO安装

下载并安装SUMO:
- 官方网站: https://www.eclipse.org/sumo/
- 推荐版本: SUMO 1.14.0 或更高

安装后确保SUMO添加到系统PATH:
```bash
# 验证SUMO安装
sumo --version
```

### 3. Python依赖安装

```bash
# 克隆项目
git clone <repository-url>
cd TJ_transport_v4

# 安装依赖
pip install -r requirements.txt
```

## 快速开始

### 方式1: 快速测试（推荐新手）

```bash
# 运行简化版测试
python quick_start.py
```

这将运行一个简短的测试episode（1000步），验证系统是否正常工作。

### 方式2: 完整训练流程

```bash
# 运行完整的三阶段训练
python train.py

# 查看所有选项
python train.py --help
```

### 方式3: 分阶段训练

```bash
# 仅训练阶段1（世界模型预训练）
python train.py --phase 1

# 仅训练阶段2（安全RL训练）
python train.py --phase 2

# 仅训练阶段3（约束优化）
python train.py --phase 3
```

### 方式4: 仅评估并生成结果

```bash
# 加载已训练模型并评估
python train.py --eval-only

# 评估并生成XLSX结果文件
python train.py --eval-only --generate-xlsx
```

## 项目结构

```
TJ_transport_v4/
├── configs/                    # 配置文件
│   └── training_config.json   # 训练配置
├── src/
│   ├── models/                # 神经网络模型
│   │   ├── gnn.py            # GNN模块
│   │   ├── world_model.py    # 世界模型
│   │   ├── controller.py     # 控制器
│   │   ├── safety.py         # 安全屏障
│   │   └── traffic_controller.py  # 完整模型
│   ├── env/                  # SUMO环境接口
│   │   ├── sumo_env.py       # 环境基类
│   │   └── data_collector.py # 数据收集器
│   ├── algorithms/           # 训练算法
│   │   └── training.py       # 三阶段训练
│   └── evaluation/           # 评估工具
│       └── results_generator.py  # XLSX生成器
├── checkpoints/              # 模型检查点
├── logs/                     # 训练日志
├── results/                  # 结果文件
├── data/                     # 收集的数据
├── train.py                  # 主训练脚本
├── quick_start.py           # 快速开始脚本
├── requirements.txt         # Python依赖
└── README.md                # 本文件
```

## 配置说明

主配置文件: `configs/training_config.json`

### 关键配置项

```json
{
  "device": "cuda",                    // 设备: cuda/cpu
  "environment": {
    "sumo_cfg": "...",                 // SUMO配置文件路径
    "max_steps": 3600,                 // 每个episode最大步数
    "control_ratio": 0.25              // 控制车辆比例
  },
  "model": {
    "top_k": 5,                        // 每步控制的车辆数
    "future_steps": 5,                 // 预测步数
    "cost_limit": 0.1                  // 成本约束
  }
}
```

## 训练流程

### 三阶段训练

#### 阶段1: 世界模型预训练
- **目标**: 学习基础交通动力学
- **数据**: 收集随机驾驶数据
- **损失**: 状态预测MSE
- **输出**: 准确预测未来5步状态

#### 阶段2: 安全RL训练
- **目标**: 学习安全控制策略
- **算法**: PPO + 安全屏障
- **重点**: 最大化性能，忽略成本
- **输出**: 安全控制策略

#### 阶段3: 约束优化
- **目标**: 平衡性能与成本
- **方法**: 拉格朗日乘子法
- **约束**: 干预成本 ≤ 0.1
- **输出**: 最终优化模型

### 训练时间估计

| 阶段 | Episodes/步数 | 预计时间 (CPU) | 预计时间 (GPU) |
|------|---------------|----------------|----------------|
| 阶段1 | 20 episodes | 1-2小时 | 30-60分钟 |
| 阶段2 | 10K步 | 2-3小时 | 1-1.5小时 |
| 阶段3 | 5K步 | 1-2小时 | 30-60分钟 |
| **总计** | - | **4-7小时** | **2-3小时** |

*实际时间取决于硬件配置和SUMO仿真速度*

## 评估与结果

### 评估指标

1. **OD完成率 (OCR)**: 车辆按时到达目的地的比例
2. **速度标准差 (σv)**: 交通流稳定性指标
3. **平均绝对加速度 (|a|avg)**: 舒适度指标
4. **干预成本**: 控制指令的总成本

### 生成XLSX结果

```bash
python train.py --eval-only --generate-xlsx
```

结果将保存在 `results/` 目录下，包含:
- 车辆ID
- 时间步
- 加速度指令
- 换道指令
- 统计摘要

### 提交竞赛

只需上传生成的XLSX文件到竞赛平台，无需提交代码。

## 常见问题

### Q1: SUMO启动失败

**问题**: `TraCIException: TraCI is not connected`

**解决方案**:
1. 检查SUMO是否正确安装
2. 验证SUMO配置文件路径
3. 确保端口未被占用: 默认端口8813
4. 尝试使用GUI模式调试: 设置`use_gui: true`

### Q2: CUDA内存不足

**问题**: `RuntimeError: CUDA out of memory`

**解决方案**:
1. 减小batch size
2. 减少并行环境数量
3. 使用CPU训练: 设置`device: "cpu"`

### Q3: 训练速度慢

**优化建议**:
1. 使用GPU加速
2. 减少SUMO仿真步数
3. 降低future_steps参数
4. 使用更小的模型

### Q4: 导入错误

**问题**: `ModuleNotFoundError: No module named 'torch_geometric'`

**解决方案**:
```bash
# PyTorch Geometric需要特殊安装
pip install torch_geometric -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

## 性能优化

### GPU加速

1. 确保CUDA可用:
```python
import torch
print(torch.cuda.is_available())
```

2. 在配置中启用:
```json
{
  "device": "cuda"
}
```

### 多进程数据收集

```python
# 在data_collector.py中自动使用多核
num_processes = min(cpu_count(), num_episodes)
```

### 内存优化

```json
{
  "phase1": {
    "batch_size": 64  // 降低batch size
  },
  "phase2": {
    "num_envs": 2     // 减少并行环境
  }
}
```

## 技术栈

- **深度学习**: PyTorch 2.0+
- **图神经网络**: PyTorch Geometric
- **仿真器**: SUMO 1.14+
- **强化学习**: 自实现PPO
- **数值计算**: NumPy, SciPy
- **数据处理**: Pandas
- **结果导出**: openpyxl

## 贡献指南

欢迎贡献代码、报告问题或提出改进建议！

1. Fork项目
2. 创建特性分支
3. 提交更改
4. 发起Pull Request

## 许可证

本项目用于学术研究和竞赛目的。

## 联系方式

如有问题或建议，请通过以下方式联系:
- 提交Issue
- 发送邮件

## 致谢

- SUMO仿真器开发团队
- PyTorch和PyTorch Geometric社区
- 竞赛主办方

---

**祝你在竞赛中取得好成绩！** 🚗💨
