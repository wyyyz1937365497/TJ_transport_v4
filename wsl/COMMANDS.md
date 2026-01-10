# 🚀 常用命令快速参考

## 环境激活
```bash
conda activate sumo
cd wsl
```

## 训练命令

### 完整训练（所有阶段）
```bash
python train.py
./train.sh
```

### 单阶段训练
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

### 仅评估模式
```bash
# 评估已有模型
python train.py --eval-only

# 评估并生成XLSX
python train.py --eval-only --generate-xlsx

# 指定episodes数量
python train.py --eval-only --num-episodes 10 --generate-xlsx
```

### 跳过数据收集
```bash
python train.py --skip-data-collection
```

### 使用不同配置
```bash
python train.py --config config/custom.yaml
```

### 指定设备
```bash
python train.py --device cuda
python train.py --device cpu
```

## 调试命令

### 调试模式（保存日志）
```bash
./train_debug.sh --phase world_model
```

### 查看日志
```bash
tail -f logs/training.log
tail -f train_*.log
```

## 清理命令

### 清理SUMO进程
```bash
./kill_sumo.sh
python kill_sumo.py
```

### 查看SUMO进程
```bash
ps aux | grep sumo
```

## 测试命令

### 测试系统初始化
```bash
python -c "
import sys
sys.path.insert(0, 'src')
from src.utils import load_config
from src.training import Trainer
config = load_config('config/base.yaml')
trainer = Trainer(config)
print('✅ 系统初始化成功')
"
```

### 查看GPU状态
```bash
nvidia-smi
```

### 查看conda环境
```bash
conda env list
```

## 配置编辑

### 编辑主配置
```bash
nano config/base.yaml
vim config/base.yaml
```

## 帮助

### 查看训练脚本帮助
```bash
python train.py --help
```

## 文件查看

### 查看训练日志
```bash
less logs/training.log
```

### 查看检查点
```bash
ls -lh checkpoints/
```

### 查看结果
```bash
ls -lh results/
ls -lh data/
```

## Bash别名（可选）

添加到 `~/.bashrc`:
```bash
alias tjs='cd ~/TJ_transport_v4/wsl && conda activate sumo'
alias train='python train.py'
alias train-debug='./train_debug.sh'
alias kill-sumo='./kill_sumo.sh'
```

使用:
```bash
tjs           # 进入目录并激活环境
train --help  # 运行训练
train-debug   # 调试模式
kill-sumo     # 清理SUMO
```
