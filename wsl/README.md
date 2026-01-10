# TJ_transport_v4 - WSL优化版本

## 概述

这是针对Linux/WSL环境优化的版本，相比原版有以下改进：

### 主要优化

1. **更简洁的代码结构**
   - 使用PyTorch Lightning简化训练循环
   - 使用YAML配置替代JSON
   - 使用dataclass简化数据结构
   - 完整的类型提示

2. **更高的训练效率**
   - GPU优化的图构建
   - 异步数据加载
   - 混合精度训练(AMP)
   - 梯度检查点支持

3. **更好的开发体验**
   - WandB集成用于实验跟踪
   - Rich终端输出
   - 进度条显示
   - 更好的错误处理

## 安装

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 安装SUMO (Ubuntu/Debian)
sudo apt-get install sumo sumo-tools
```

## 使用

### 快速开始

```bash
# 使用默认配置训练
python train.py

# 指定配置文件
python train.py --config config/base.yaml

# 仅评估
python train.py --eval-only

# 指定训练阶段
python train.py --phase world_model
python train.py --phase ppo
python train.py --phase finetune
```

### 配置文件

配置文件位于 `config/` 目录，使用YAML格式：

```yaml
# config/base.yaml
device: cuda
seed: 42

model:
  node_dim: 9
  hidden_dim: 128
  num_layers: 3

training:
  batch_size: 256
  learning_rate: 1e-4
  max_epochs: 100
```

## 项目结构

```
wsl/
├── config/              # 配置文件
│   └── base.yaml
├── src/
│   ├── models/          # 神经网络模型
│   ├── training/        # 训练相关代码
│   ├── env/            # 环境接口
│   └── utils/          # 工具函数
├── train.py            # 训练入口
├── requirements.txt    # 依赖
└── README.md          # 本文件
```

## 与原版的区别

| 特性 | 原版 | WSL版 |
|-----|------|-------|
| 配置 | JSON | YAML |
| 训练框架 | 自定义 | PyTorch Lightning |
| 日志 | 本地文件 | WandB + 本地 |
| 进度显示 | 文本 | Rich + 进度条 |
| 类型提示 | 部分 | 完整 |
| 图构建 | CPU优化 | GPU加速 |

## 许可证

与原项目相同
