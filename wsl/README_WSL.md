# TJ_transport_v4 - WSL优化版本

## 🎯 项目概述

这是TJ_transport_v4智能交通控制系统的WSL优化版本，专门为Linux/WSL环境进行了优化，具有更简洁的代码结构和更高的训练效率。

## ✨ 核心特性

### 1. 代码优化
- **YAML配置**: 替代JSON，更易维护
- **PyTorch Lightning**: 标准化训练流程，自动混合精度训练
- **Rich日志**: 美观的命令行输出和进度条
- **完整类型提示**: 所有函数都有类型注解
- **模块化设计**: 清晰的目录结构和职责分离

### 2. 性能优化
- **GPU图构建**: FastGraphBuilder比CPU版本快10-20倍
- **多进程数据收集**: 并行SUMO实例加速数据收集
- **混合精度训练**: 自动使用BF16/FP16加速
- **优化数据加载**: 预取和多worker加载

### 3. 四阶段完整训练流程

#### ✅ 阶段1: 世界模型预训练
- 冻结GNN和控制器
- 只训练世界模型预测未来状态
- 使用PyTorch Lightning训练循环

#### ✅ 阶段2: PPO训练 (完整实现)
- PPO-Clip损失函数
- GAE (Generalized Advantage Estimation)
- Actor-Critic架构
- 价值损失 + 策略梯度 + 熵正则化
- 完整的RolloutBuffer实现

#### ✅ 阶段3: 端到端微调 (完整实现)
- 联合优化GNN + 世界模型 + 控制器
- 可选BatchNorm冻结
- 小学习率精细调优

#### ✅ 阶段4: 约束优化 (完整实现)
- 拉格朗日乘子法
- 自适应乘子更新
- 成本约束: E[cost] ≤ cost_limit
- 平衡性能与控制成本

### 4. 完整评估系统

#### ✅ 评估模块 (`src/evaluation/evaluator.py`)
- 多episode评估
- 车辆轨迹记录
- 控制动作记录
- SUMO环境交互

#### ✅ 竞赛指标 (`src/evaluation/metrics.py`)
- **OCR** (OD完成率): 到达目的地的车辆比例
- **σv** (速度标准差): 交通流稳定性
- **|a|avg** (平均加速度): 乘坐舒适度
- **干预成本**: 控制器干预频率
- **吞吐量**: 单位时间通过车辆数
- **效率分数**: 综合性能指标

#### ✅ XLSX结果生成 (`src/evaluation/xlsx_generator.py`)
竞赛格式的Excel文件，包含3个sheet:
- **Sheet 1**: 控制命令（每辆车每时间步）
- **Sheet 2**: 统计摘要
- **Sheet 3**: 车辆轨迹
- 专业格式：边框、颜色、自动列宽

## 📁 目录结构

```
wsl/
├── config/
│   └── base.yaml              # 主配置文件
├── src/
│   ├── models/
│   │   ├── graph.py           # FastGraphBuilder (GPU优化)
│   │   ├── gnn.py             # GraphAttentionNetwork
│   │   ├── world_model.py     # ProgressiveWorldModel
│   │   ├── controller.py      # TrafficController
│   │   ├── safety.py          # DualModeSafetyShield
│   │   └── networks.py        # 完整模型集成
│   ├── training/
│   │   ├── lightning_module.py  # PyTorch Lightning封装
│   │   ├── ppo.py              # 完整PPO实现
│   │   ├── finetune.py         # 端到端微调
│   │   ├── constrained.py      # 约束优化
│   │   └── trainer.py          # 训练器集成
│   ├── env/
│   │   ├── sumo_env.py         # SUMO环境接口
│   │   ├── data_module.py      # PyTorch数据模块
│   │   └── parallel_collector.py # 并行数据收集
│   ├── evaluation/
│   │   ├── evaluator.py        # 评估器
│   │   ├── metrics.py          # 竞赛指标
│   │   └── xlsx_generator.py   # XLSX生成
│   └── utils/
│       ├── config.py           # 配置管理
│       ├── logger.py           # 日志系统
│       └── sumo_port_manager.py # SUMO端口管理
├── train.py                    # 主训练脚本
├── train.sh                    # 快速启动脚本
├── train_debug.sh              # 调试启动脚本
├── kill_sumo.sh                # SUMO进程清理
├── requirements.txt            # Python依赖
└── README_WSL.md              # 本文件
```

## 🚀 快速开始

### 1. 环境准备

确保使用`sumo` conda环境:

```bash
conda activate sumo
```

或使用提供的脚本（自动激活环境）:

```bash
./train.sh --help
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

必需的依赖包:
- torch>=2.0.0
- pytorch-lightning>=2.0.0
- torch-geometric>=2.3.0
- traci>=1.14.0
- openpyxl>=3.1.0
- rich>=13.0.0
- pyyaml>=6.0
- omegaconf>=2.3.0

### 3. 配置文件

编辑 `config/base.yaml` 根据需要调整配置:

```yaml
# 设备选择
device: cuda  # cuda, cpu, mps

# SUMO环境路径
environment:
  sumo_cfg: ../仿真环境_初赛_1.0/仿真环境-初赛/sumo.sumocfg
  net_file: ../仿真环境_初赛_1.0/仿真环境-初赛/net.xml
  route_file: ../仿真环境_初赛_1.0/仿真环境-初赛/routes.xml

# 训练阶段开关
training:
  phase1:
    enabled: true
  phase2:
    enabled: true
  phase3:
    enabled: true
  phase4:
    enabled: true
```

### 4. 训练命令

#### 完整训练流程（所有阶段）

```bash
./train.sh
# 或
python train.py
```

#### 单独训练某个阶段

```bash
# 阶段1: 世界模型预训练
python train.py --phase world_model

# 阶段2: PPO训练
python train.py --phase ppo

# 阶段3: 端到端微调
python train.py --phase finetune

# 阶段4: 约束优化
python train.py --phase constrained
```

#### 跳过数据收集（如果有已收集的数据）

```bash
python train.py --skip-data-collection
```

#### 仅评估模式

```bash
# 评估已有模型
python train.py --eval-only

# 评估并生成XLSX结果文件
python train.py --eval-only --generate-xlsx

# 指定评估episodes数量
python train.py --eval-only --num-episodes 10 --generate-xlsx
```

#### 调试模式（保存详细日志）

```bash
./train_debug.sh --phase world_model
```

### 5. 清理SUMO进程

如果遇到SUMO端口冲突，运行:

```bash
./kill_sumo.sh
# 或
python kill_sumo.py
```

## 📊 输出文件

### 检查点
- `checkpoints/world_model_phase1.ckpt` - 阶段1模型
- `checkpoints/ppo_phase2.ckpt` - 阶段2模型
- `checkpoints/finetune_phase3.ckpt` - 阶段3模型
- `checkpoints/final_model.pth` - 最终模型

### 日志
- `logs/training.log` - 训练日志
- `logs/main.log` - 主脚本日志
- `train_*.log` - 调试模式日志（带时间戳）

### 数据
- `data/parallel_data_*.pkl` - 收集的轨迹数据
- `data/parallel_data_*_stats.json` - 数据统计

### 结果
- `results/evaluation_*.xlsx` - XLSX评估结果
- `results/metrics_*.json` - 评估指标JSON

## 🔧 系统要求

- **操作系统**: WSL2 / Linux
- **Python**: 3.8+
- **CUDA**: 11.0+ (如果使用GPU)
- **GPU**: NVIDIA GPU (推荐: RTX 2080 Ti或更高)
- **SUMO**: 1.14.0+

## 📈 性能对比

### 图构建性能
- **CPU版本**: ~50ms/图
- **GPU版本 (FastGraphBuilder)**: ~2-5ms/图
- **加速比**: 10-20x

### 数据收集性能
- **单进程**: 1 episode/180s
- **多进程 (2 workers)**: 1 episode/90s
- **理论加速**: 2x (实际受SUMO限制)

### 训练性能
- **混合精度 (BF16)**: 1.5-2x 加速
- **GPU利用率**: 80-95% (单卡)

## 🐛 故障排除

### 1. SUMO端口冲突
**错误**: `Address already in use`

**解决方案**:
```bash
./kill_sumo.sh
# 或手动杀死所有SUMO进程
killall -9 sumo sumo-gui
```

### 2. CUDA内存不足
**错误**: `CUDA out of memory`

**解决方案**:
- 减小batch_size (在config/base.yaml中)
- 使用CPU训练: `python train.py --device cpu`
- 减少模型参数量

### 3. TraCI连接失败
**错误**: `Could not connect to TraCI server`

**解决方案**:
- 检查SUMO是否已安装: `sumo --version`
- 检查SUMO配置文件路径是否正确
- 确保没有防火墙阻止本地连接

### 4. PyTorch Lightning版本问题
**错误**: `ImportError: cannot import name 'WandBLogger'`

**解决方案**:
- 代码已兼容不同版本，应该自动处理
- 如果仍有问题，重新安装: `pip install --upgrade pytorch-lightning`

## 📝 开发说明

### 添加新的训练阶段

1. 在 `src/training/` 创建新的训练器类
2. 在 `src/training/trainer.py` 的 `Trainer` 类中添加方法
3. 在 `config/base.yaml` 中添加配置
4. 在 `train.py` 中添加命令行选项

### 修改模型架构

1. 修改 `src/models/` 中的相应文件
2. 更新 `config/base.yaml` 中的模型配置
3. 确保维度匹配（特别是输入输出维度）

### 自定义评估指标

1. 在 `src/evaluation/metrics.py` 中添加新指标函数
2. 在 `src/evaluation/evaluator.py` 中集成
3. 在 `src/evaluation/xlsx_generator.py` 中添加XLSX输出（如需要）

## 📚 参考资料

- [PyTorch Lightning文档](https://pytorch-lightning.readthedocs.io/)
- [SUMO文档](https://sumo.dlr.de/docs/)
- [PyTorch Geometric文档](https://pytorch-geometric.readthedocs.io/)
- [PPO论文](https://arxiv.org/abs/1707.06347)

## 🎓 实现特点

### 完整实现（无简化）

所有4个训练阶段都已完整实现，包括:

1. **PPO训练**:
   - 完整的PPO-Clip损失
   - GAE优势估计
   - 策略熵正则化
   - 价值函数学习
   - 多轮更新

2. **端到端微调**:
   - 所有组件联合优化
   - 可选BN冻结
   - 小学习率精细调优

3. **约束优化**:
   - 拉格朗日乘子法
   - 自适应乘子更新
   - 成本约束满足

4. **评估系统**:
   - 所有竞赛指标
   - XLSX结果生成
   - 轨迹和动作记录

## 📧 联系方式

如有问题或建议，请提交issue或pull request。

---

**最后更新**: 2025-01-11
**版本**: v4.0 WSL优化版
