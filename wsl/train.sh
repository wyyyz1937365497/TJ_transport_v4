#!/bin/bash
# TJ_transport_v4 WSL版本 - 训练启动脚本
# 使用sumo conda环境

# 激活sumo环境
eval "$(conda shell.bash hook)"
conda activate sumo

# 运行训练
python train.py "$@"
