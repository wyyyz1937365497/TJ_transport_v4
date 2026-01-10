#!/bin/bash
# TJ_transport_v4 WSL版本 - 调试启动脚本
# 使用sumo conda环境，显示详细输出

# 激活sumo环境
eval "$(conda shell.bash hook)"
conda activate sumo

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export CUDA_VISIBLE_DEVICES=0

# 运行训练（详细模式）
python -u train.py "$@" 2>&1 | tee train_$(date +%Y%m%d_%H%M%S).log
