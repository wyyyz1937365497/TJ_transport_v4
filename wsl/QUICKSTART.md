# 快速开始指南

## 安装

### 1. 安装SUMO

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install sumo sumo-tools

# 验证安装
sumo --version
```

### 2. 创建虚拟环境

```bash
cd wsl
python -m venv venv
source venv/bin/activate
```

### 3. 安装Python依赖

```bash
# PyTorch (根据你的CUDA版本调整)
pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cu118

# PyTorch Geometric
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# 其他依赖
pip install -r requirements.txt
```

## 使用

### 数据收集 + 世界模型训练（阶段1）

```bash
# 完整流程（收集数据 + 训练）
python train.py --phase world_model

# 如果已有数据，跳过收集
python train.py --phase world_model --skip-data-collection
```

### 完整训练流程

```bash
# 训练所有阶段
python train.py

# 指定配置文件
python train.py --config config/base.yaml
```

### 评估

```bash
# 仅评估模式
python train.py --eval-only
```

## 目录结构

```
wsl/
├── config/
│   └── base.yaml          # 配置文件
├── src/
│   ├── models/            # 神经网络模型
│   │   ├── graph.py       # GPU优化的图构建器
│   │   └── networks.py    # 神经网络模块
│   ├── training/          # 训练相关
│   │   ├── lightning_module.py  # PyTorch Lightning模块
│   │   └── trainer.py     # 训练器封装
│   ├── env/               # 环境接口
│   │   ├── sumo_env.py    # SUMO环境
│   │   └── data_module.py # 数据模块
│   └── utils/             # 工具函数
│       ├── config.py      # 配置管理
│       ├── logging.py     # 日志
│       └── dataclass.py   # 数据结构
├── train.py               # 训练入口
├── requirements.txt       # 依赖
├── README.md             # 项目说明
└── QUICKSTART.md         # 本文件
```

## 配置

主要配置在 `config/base.yaml`:

```yaml
# 设备
device: cuda  # cuda, cpu
seed: 42
precision: bf16

# 路径
paths:
  data_dir: data
  log_dir: logs
  checkpoint_dir: checkpoints

# 训练配置
training:
  phase1:
    num_episodes: 5       # 数据收集episodes
    epochs: 10            # 训练轮数
    batch_size: 256
    learning_rate: 1.0e-4
```

## 与原版的对比

| 特性 | 原版 | WSL版 |
|-----|------|-------|
| 配置 | JSON | YAML |
| 训练框架 | 自定义 | PyTorch Lightning |
| 日志 | 文本 | Rich + WandB |
| 进度显示 | 文本 | 进度条 |
| 图构建 | CPU | GPU优化 |
| 类型提示 | 部分 | 完整 |
| 代码行数 | ~2500行 | ~1500行 |

## 常见问题

### Q: SUMO启动失败，提示端口占用
A: 这是因为之前的SUMO进程没有正常关闭。

**解决方案:**
```bash
# 方法1: 使用清理脚本
python kill_sumo.py
# 或
./kill_sumo.sh

# 方法2: 手动清理
killall -9 sumo sumo.exe
```

### Q: 并行数据收集时出现连接错误
A: 这通常是端口冲突问题。

**解决方案:**
```bash
# 1. 先清理所有SUMO进程
python kill_sumo.py

# 2. 减少并行worker数量
# 在config/base.yaml中设置:
environment:
  num_parallel_workers: 1  # 从4减少到1
```

### Q: 训练时CUDA内存不足
A: 减小batch_size或使用CPU训练
```yaml
training:
  phase1:
    batch_size: 128  # 减小batch size
device: cpu  # 或使用CPU
```

### Q: 数据收集失败
A: 检查SUMO配置文件路径
```yaml
environment:
  sumo_cfg: ../仿真环境_初赛_1.0/仿真环境-初赛/sumo.sumocfg
```

## 下一步

- 阶段2 (PPO训练): 待实现
- 阶段3 (端到端微调): 待实现
- 阶段4 (约束优化): 待实现
- 评估模块: 待实现
- XLSX结果生成: 待实现
