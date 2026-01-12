# 训练命令完整指南

## 📋 目录

1. [环境准备](#环境准备)
2. [快速测试训练（5-10分钟）](#快速测试训练)
3. [完整训练流程](#完整训练流程)
4. [模型评估](#模型评估)
5. [常见问题](#常见问题)

---

## 环境准备

### 1. 检查依赖

```bash
# 检查Python版本（需要3.8+）
python --version

# 检查PyTorch（需要1.10+）
python -c "import torch; print(torch.__version__)"

# 检查CUDA（可选）
python -c "import torch; print(torch.cuda.is_available())"

# 检查SUMO
sumo --version
```

### 2. 安装依赖

```bash
# 安装Python包
pip install -r requirements.txt

# 或单独安装关键依赖
pip install torch torchvision torchaudio
pip install stable-baselines3
pip install sumolib
pip install traci
pip install pyyaml
pip install pandas openpyxl
```

### 3. 验证环境

```bash
# 快速验证SUMO环境
python -c "import traci; sumo -v"
```

---

## 快速测试训练

**用途**：验证代码正确性、功能测试、快速迭代
**时间**：5-10分钟
**资源**：2-4GB GPU内存

### 单阶段快速测试

```bash
# 仅Phase 1（世界模型预训练）- 约2分钟
python train_v4_ideal.py \
    --config configs/competition_quick_test.yaml \
    --phase 1

# 仅Phase 2（PPO训练）- 约3分钟
python train_v4_ideal.py \
    --config configs/competition_quick_test.yaml \
    --phase 2

# 仅Phase 3（约束优化）- 约2分钟
python train_v4_ideal.py \
    --config configs/competition_quick_test.yaml \
    --phase 3
```

### 完整快速测试（推荐）

```bash
# 运行所有阶段 - 约5-10分钟
python train_v4_ideal.py \
    --config configs/competition_quick_test.yaml \
    --phase all
```

### 自定义快速测试

```bash
# 更少的数据收集
python train_v4_ideal.py \
    --config configs/competition_quick_test.yaml \
    --phase all \
    --episodes 1

# 更少的训练步数
python train_v4_ideal.py \
    --config configs/competition_quick_test.yaml \
    --phase all \
    --timesteps 500

# 使用CPU（如果GPU不可用）
python train_v4_ideal.py \
    --config configs/competition_quick_test.yaml \
    --phase all \
    --device cpu
```

---

## 完整训练流程

**用途**：正式训练、比赛准备
**时间**：约2-4小时（取决于硬件）
**资源**：8-16GB GPU内存推荐

### 方法1：一键完整训练（推荐）

```bash
# 运行所有阶段（Phase 1+2+3）
python train_v4_ideal.py \
    --config configs/competition_preliminary.yaml \
    --phase all
```

**预期输出**：
```
Phase 1: 收集50个episodes数据，训练30轮...
Phase 2: PPO训练200000步...
Phase 3: 约束优化100000步...
[SUCCESS] Complete training pipeline finished!
```

### 方法2：分阶段训练

```bash
# Phase 1: 世界模型预训练（约30-45分钟）
python train_v4_ideal.py \
    --config configs/competition_preliminary.yaml \
    --phase 1

# Phase 2: 带安全屏障的PPO训练（约60-90分钟）
python train_v4_ideal.py \
    --config configs/competition_preliminary.yaml \
    --phase 2

# Phase 3: 拉格朗日约束优化（约30-45分钟）
python train_v4_ideal.py \
    --config configs/competition_preliminary.yaml \
    --phase 3
```

**优点**：
- 可以分批训练，灵活控制时间
- 每个阶段后可以检查中间结果
- 便于调试和优化

### 方法3：自定义训练参数

```bash
# 自定义数据收集量
python train_v4_ideal.py \
    --config configs/competition_preliminary.yaml \
    --phase 1 \
    --episodes 100

# 自定义训练步数
python train_v4_ideal.py \
    --config configs/competition_preliminary.yaml \
    --phase all \
    --timesteps 300000

# 禁用缓存强制重新收集数据
python train_v4_ideal.py \
    --config configs/competition_preliminary.yaml \
    --phase all \
    --no-cache
```

---

## 训练阶段详解

### Phase 1: 世界模型预训练

**目的**：让模型学会交通流演化规律
**时间**：30-45分钟
**输出**：`checkpoints/competition/world_model_phase1.pth`

```bash
python train_v4_ideal.py \
    --config configs/competition_preliminary.yaml \
    --phase 1
```

**配置参数**（在`competition_preliminary.yaml`中）：
```yaml
training:
  phase1:
    num_episodes: 50    # 数据收集episodes
    epochs: 30          # 训练轮数
    batch_size: 256     # 批次大小
    learning_rate: 1.0e-4
```

### Phase 2: PPO训练

**目的**：训练决策策略
**时间**：60-90分钟
**输出**：`checkpoints/competition/ppo_phase2.zip`

```bash
python train_v4_ideal.py \
    --config configs/competition_preliminary.yaml \
    --phase 2
```

**配置参数**：
```yaml
training:
  phase2:
    total_timesteps: 200000  # 训练步数
    learning_rate: 3.0e-4
    num_envs: 8              # 并行环境数
    n_steps: 4096             # Rollout步数
```

### Phase 3: 约束优化

**目的**：满足成本约束，优化干预成本
**时间**：30-45分钟
**输出**：`checkpoints/competition/final_model_competition.pth`

```bash
python train_v4_ideal.py \
    --config configs/competition_preliminary.yaml \
    --phase 3
```

**配置参数**：
```yaml
training:
  phase3:
    total_timesteps: 100000
    cost_limit: 0.1         # 成本上限
    learning_rate: 1.0e-4
```

---

## 模型评估

### 快速评估

```bash
# 使用训练好的模型进行评估
python evaluate_v4_ideal.py \
    --config configs/competition_preliminary.yaml \
    --model checkpoints/competition/final_model_competition.pth \
    --episodes 10
```

### 生成比赛提交文件

```bash
# 评估并生成XLSX结果
python evaluate_v4_ideal.py \
    --config configs/competition_preliminary.yaml \
    --model checkpoints/competition/final_model_competition.pth \
    --episodes 10 \
    --save_xlsx
```

### 查看详细分析

```bash
# 评估并显示详细分析
python evaluate_v4_ideal.py \
    --config configs/competition_preliminary.yaml \
    --model checkpoints/competition/final_model_competition.pth \
    --episodes 10 \
    --verbose
```

---

## 输出文件位置

### 训练输出

```
checkpoints/
├── competition/              # 正式训练输出
│   ├── world_model_phase1.pth
│   ├── ppo_phase2.zip
│   └── final_model_competition.pth
│
└── quick_test/              # 快速测试输出
    ├── world_model_phase1.pth
    ├── ppo_phase2.zip
    └── final_model.pth

logs/
├── competition/              # 正式训练日志
└── quick_test/              # 快速测试日志
```

### 评估输出

```
results/
├── competition/              # 正式评估结果
│   └── evaluation_results.xlsx
│
└── quick_test/              # 快速测试结果
    └── evaluation_results.xlsx
```

---

## 常见问题

### Q1: 训练时出现CUDA内存不足

**解决方案**：
```bash
# 方案1：减少并行环境数
# 在配置文件中设置
environment:
  num_parallel_workers: 4  # 从8减少到4

training:
  phase2:
    num_envs: 4  # 从8减少到4

# 方案2：减小批次大小
training:
  phase1:
    batch_size: 128  # 从256减少到128
  phase2:
    batch_size: 128  # 从256减少到128

# 方案3：使用CPU训练
python train_v4_ideal.py \
    --config configs/competition_preliminary.yaml \
    --phase all \
    --device cpu
```

### Q2: SUMO环境启动失败

**检查**：
```bash
# 验证SUMO安装
sumo --version

# 验证环境文件路径
ls -la "仿真环境_初赛_1.0/仿真环境-初赛/sumo.sumocfg"
ls -la "仿真环境_初赛_1.0/仿真环境-初赛/net.xml"
ls -la "仿真环境_初赛_1.0/仿真环境-初赛/routes.xml"

# 验证SUMO可执行文件
which sumo
which sumo-gui
```

**解决方案**：
```bash
# 方案1：设置SUMO_HOME环境变量
export SUMO_HOME=/path/to/sumo
export PATH=$SUMO_HOME/bin:$PATH

# 方案2：在配置中指定完整路径
# 编辑 configs/competition_preliminary.yaml
environment:
  sumo_home: "C:/Program Files (x86)/SUMO"  # Windows
```

### Q3: 训练过程中断

**从checkpoint恢复**：
```bash
# 如果Phase 1已完成，直接从Phase 2开始
python train_v4_ideal.py \
    --config configs/competition_preliminary.yaml \
    --phase 2

# 如果Phase 2已完成，直接从Phase 3开始
python train_v4_ideal.py \
    --config configs/competition_preliminary.yaml \
    --phase 3
```

### Q4: 如何使用已训练的模型

**加载模型进行推理**：
```python
import torch
from src.models.v4_architecture import IdealTrafficControllerV4

# 创建模型
model = IdealTrafficControllerV4(
    top_k=5,
    device='cuda'
)

# 加载checkpoint
checkpoint = torch.load('checkpoints/competition/final_model_competition.pth')
model.load_state_dict(checkpoint)
model.eval()

# 推理
output = model(observation, deterministic=True)
```

### Q5: 训练速度慢

**加速方法**：

1. **增加并行度**：
```yaml
environment:
  num_parallel_workers: 16  # 从8增加到16

training:
  phase2:
    num_envs: 16  # 从8增加到16
```

2. **使用混合精度**：
```yaml
precision: bf16  # 或fp16
```

3. **减少训练步数**：
```bash
python train_v4_ideal.py \
    --config configs/competition_preliminary.yaml \
    --phase all \
    --timesteps 100000  # 从200k减少到100k
```

### Q6: 如何监控训练进度

**方法1：查看日志**
```bash
# 实时查看训练日志
tail -f logs/competition/*.log

# 查看最近的输出
tail -100 logs/competition/*.log
```

**方法2：使用TensorBoard（如果启用）**
```bash
# 启动TensorBoard
tensorboard --logdir logs/competition

# 浏览器访问
# http://localhost:6006
```

**方法3：检查checkpoint**
```bash
# 查看最新checkpoint
ls -lht checkpoints/competition/*.pth
ls -lht checkpoints/competition/*.zip
```

---

## 高级用法

### 自定义配置文件

```bash
# 1. 复制现有配置
cp configs/competition_preliminary.yaml configs/my_custom_config.yaml

# 2. 编辑配置文件
vim configs/my_custom_config.yaml

# 3. 使用自定义配置训练
python train_v4_ideal.py \
    --config configs/my_custom_config.yaml \
    --phase all
```

### 仅训练特定阶段

```bash
# 仅重新训练Phase 3（约束优化）
python train_v4_ideal.py \
    --config configs/competition_preliminary.yaml \
    --phase 3

# 覆盖训练步数
python train_v4_ideal.py \
    --config configs/competition_preliminary.yaml \
    --phase 2 \
    --timesteps 500000  # 更多训练步数
```

### 调试模式

```bash
# 启用详细日志
export PYTHONUNBUFFERED=1
python train_v4_ideal.py \
    --config configs/competition_quick_test.yaml \
    --phase all \
    2>&1 | tee training_log.txt
```

---

## 推荐工作流程

### 初次使用

```bash
# 1. 快速测试（验证环境）
python train_v4_ideal.py \
    --config configs/competition_quick_test.yaml \
    --phase all

# 2. 快速评估
python evaluate_v4_ideal.py \
    --config configs/competition_quick_test.yaml \
    --model checkpoints/quick_test/final_model.pth \
    --episodes 2

# 3. 如果一切正常，开始完整训练
python train_v4_ideal.py \
    --config configs/competition_preliminary.yaml \
    --phase all
```

### 日常开发

```bash
# 代码修改后快速验证
python train_v4_ideal.py \
    --config configs/competition_quick_test.yaml \
    --phase all \
    --no-cache

# 验证通过后完整训练
python train_v4_ideal.py \
    --config configs/competition_preliminary.yaml \
    --phase all
```

---

## 总结

### 快速测试（5-10分钟）

```bash
python train_v4_ideal.py --config configs/competition_quick_test.yaml --phase all
```

### 完整训练（2-4小时）

```bash
python train_v4_ideal.py --config configs/competition_preliminary.yaml --phase all
```

### 评估模型

```bash
python evaluate_v4_ideal.py \
    --config configs/competition_preliminary.yaml \
    --model checkpoints/competition/final_model_competition.pth \
    --episodes 10
```

---

**所有增强功能（可学习权重、场景识别、自适应Top-K、动态拉格朗日）均已默认启用！**

直接运行上述命令即可享受所有性能提升！🚀
