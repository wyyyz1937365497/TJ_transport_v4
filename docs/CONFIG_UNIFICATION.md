# 训练配置统一规范

## 概述

所有训练阶段统一使用 `configs/competition.yaml` 作为主配置文件。

## 配置文件结构

```
configs/
├── competition.yaml              # ⭐ 主配置文件（所有训练脚本默认使用）
├── competition_preliminary.yaml  # 初赛专用配置（已弃用，仅作参考）
└── default.yaml                  # 默认配置模板
```

## 训练脚本配置映射

| 训练脚本 | 默认配置文件 | 说明 |
|---------|-------------|------|
| `train_phase1.py` | `configs/competition.yaml` | ✅ 已统一 |
| `train_phase2.py` | `configs/competition.yaml` | ✅ 已统一 |
| `train_phase3.py` | `configs/competition.yaml` | ✅ 已统一 |
| `train_all.sh` | `configs/competition.yaml` | ✅ 已统一 |
| `train.py` | `configs/competition.yaml` | ✅ 已统一 |

## 主配置文件结构

`configs/competition.yaml` 包含所有训练阶段的配置：

```yaml
# 路径配置
paths:
  checkpoint_dir: checkpoints/competition
  log_dir: logs
  tensorboard_dir: runs

# 环境配置
environment:
  max_vehicles: 32
  inflow_rate: 2000
  # ... 其他环境配置

# 模型配置
model:
  gnn:
    node_dim: 9
    # ... 模型架构配置

# 训练配置
training:
  # 课程学习
  curriculum:
    enabled: true

  # Phase 1: 世界模型预训练
  phase1:
    num_episodes: 50
    epochs: 30
    batch_size: 256
    # ... Phase 1 配置

  # Phase 2: PPO训练
  phase2:
    total_timesteps: 2000000
    num_envs: 4
    n_steps: 1024
    batch_size: 512
    update_epochs: 10
    # ... Phase 2 配置

  # Phase 3: 拉格朗日优化
  phase3:
    total_timesteps: 100000
    # ... Phase 3 配置
```

## 使用方法

### 标准训练（推荐）

使用默认配置文件 `configs/competition.yaml`：

```bash
# 单独训练各阶段
python train_phase1.py                    # Phase 1
python train_phase2.py --stage 1          # Phase 2 - Stage 1
python train_phase3.py                    # Phase 3

# 完整训练流程
./train_all.sh                            # 所有阶段
```

### 使用自定义配置

```bash
# 指定配置文件
python train_phase1.py --config configs/custom_config.yaml
python train_phase2.py --stage 1 --config configs/custom_config.yaml
python train_phase3.py --config configs/custom_config.yaml
```

## 配置文件修改规范

### 1. 修改全局配置

直接编辑 `configs/competition.yaml`：

```yaml
# 例如：修改Phase 2的并行环境数
training:
  phase2:
    num_envs: 8  # 从4改为8
```

### 2. 创建实验配置

基于主配置创建实验专用配置：

```bash
# 复制主配置
cp configs/competition.yaml configs/experiment_001.yaml

# 修改实验参数
vim configs/experiment_001.yaml

# 使用实验配置训练
python train_phase2.py --stage 1 --config configs/experiment_001.yaml
```

## 配置验证

### 检查配置文件完整性

```bash
# 检查所有必需的配置项
python -c "
import yaml
with open('configs/competition.yaml') as f:
    config = yaml.safe_load(f)

# 验证必需的配置节
required_sections = ['paths', 'environment', 'model', 'training']
for section in required_sections:
    assert section in config, f'Missing section: {section}'
    print(f'✓ {section} section found')

# 验证训练阶段配置
training = config['training']
for phase in ['phase1', 'phase2', 'phase3']:
    assert phase in training, f'Missing phase: {phase}'
    print(f'✓ {phase} config found')
"
```

### 验证训练脚本使用的配置

```bash
# 查看Phase 1使用的配置
python train_phase1.py --help | grep "config"

# 查看Phase 2使用的配置
python train_phase2.py --help | grep "config"

# 查看Phase 3使用的配置
python train_phase3.py --help | grep "config"
```

应该都显示：
```
--config CONFIG  配置文件路径 (默认: configs/competition.yaml)
```

## 历史变更记录

### 2024-01-19: 配置统一

**问题**: 不同训练脚本使用不同的配置文件
- `train_phase1.py` 使用 `configs/competition_preliminary.yaml`
- `train_phase2.py` 使用 `configs/competition.yaml`
- `train_phase3.py` 使用 `configs/competition.yaml`

**解决方案**: 统一所有训练脚本使用 `configs/competition.yaml`

**变更**:
- ✅ 更新 `train_phase1.py` 默认配置
- ✅ 更新所有文档和示例
- ✅ 确保 `configs/competition.yaml` 包含所有必要配置
- ✅ 保留 `configs/competition_preliminary.yaml` 作为参考

## 故障排查

### 问题1: 训练参数没有生效

**症状**: 修改了配置文件，但训练时参数仍然是旧值

**原因**:
1. 使用了错误的配置文件
2. 缓存的.pyc文件
3. 配置文件格式错误

**解决方案**:
```bash
# 1. 确认使用的配置文件
python train_phase1.py --help  # 查看默认配置

# 2. 清理Python缓存
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete

# 3. 验证配置文件语法
python -c "import yaml; yaml.safe_load(open('configs/competition.yaml'))"

# 4. 查看训练开始时的配置输出
# 训练日志会显示使用的配置参数
```

### 问题2: 配置文件不存在

**症状**: `FileNotFoundError: configs/competition.yaml`

**解决方案**:
```bash
# 检查配置文件是否存在
ls -la configs/competition.yaml

# 如果不存在，从备份恢复
cp configs/competition_preliminary.yaml configs/competition.yaml

# 或创建默认配置
cp configs/default.yaml configs/competition.yaml
```

### 问题3: Phase 1/2/3 配置不一致

**症状**: Phase 1训练正常，但Phase 2加载失败

**原因**: 不同阶段的checkpoint路径不匹配

**解决方案**:
```yaml
# 确保configs/competition.yaml中的路径配置一致
paths:
  checkpoint_dir: checkpoints/competition  # 所有阶段使用同一目录

training:
  phase1:
    phase1_model_path: checkpoints/competition/phase1/world_model_final.pth
  phase2:
    phase2_model_path: checkpoints/competition/phase2/shielded_ppo.zip
  phase3:
    phase3_model_path: checkpoints/competition/phase3/final.zip
```

## 最佳实践

### 1. 版本控制

- ✅ `configs/competition.yaml` 始终提交到Git
- ✅ 修改配置时创建备份
- ✅ 重要实验使用专用配置文件

### 2. 配置管理

```bash
# 创建实验配置
cp configs/competition.yaml configs/exp_001_baseline.yaml
cp configs/competition.yaml configs/exp_002_large_batch.yaml

# 记录配置差异
git diff configs/competition.yaml configs/exp_001_baseline.yaml > exp_001_changes.patch
```

### 3. 配置文档化

在配置文件中添加注释：

```yaml
training:
  phase2:
    num_envs: 4  # 4个并行环境（根据CPU核心数调整）
    n_steps: 1024  # 单次rollout步数（1024*4=4096 transitions）
    batch_size: 512  # PPO batch size（根据GPU显存调整）
    # 修改这些参数会影响：
    # - 训练速度
    # - 内存占用
    # - 训练稳定性
```

### 4. 测试配置

在正式训练前测试配置：

```bash
# 干运行测试（不实际训练）
python train_phase2.py --stage 1 --config configs/test_config.yaml --dry-run

# 短时间测试（只训练几个update）
# 修改配置文件中的total_timesteps为较小的值
```

## 相关文档

- `docs/performance_optimization.md` - 性能优化指南
- `docs/training_guide.md` - 训练完整指南
- `TRAINING_FIXES.md` - 训练问题修复记录
