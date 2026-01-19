"""
初赛专用训练脚本 - 完整流程（Phase 1 + Phase 2）

使用方法：
    # 完整训练（Phase 1 + Phase 2 所有级别）
    python train_preliminary.py --config configs/competition_preliminary.yaml

    # 仅训练Phase 2（跳过Phase 1）
    python train_preliminary.py --config configs/competition_preliminary.yaml --skip-phase1

    # 从指定级别开始训练
    python train_preliminary.py --config configs/competition_preliminary.yaml --start-level 3

初赛特点：
    - 仅控制25%智能车辆，不控制设施
    - 以效率为主要评分指标
    - 5级课程学习（从简单到复杂）
    - 自动保存最佳模型用于提交
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.ideal_policy_v4 import create_ideal_traffic_policy_v4, IdealTrafficPolicyV4
from src.env.competition_env import CompetitionSumoEnv
from src.training.custom_ppo_trainer import CustomPPOTrainer

# 导入恢复的Phase 1训练器
try:
    from train_phase1 import Phase1WorldModelTrainer

    # 创建缺失的依赖类
    class MultiGPUManager:
        """简化的多GPU管理器"""
        def __init__(self, config):
            self.config = config
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            self.multi_gpu = torch.cuda.device_count() > 1

        def wrap_model(self, model):
            if self.multi_gpu:
                return torch.nn.DataParallel(model)
            return model

        def get_model_state_dict(self, model):
            if self.multi_gpu:
                return model.module.state_dict()
            return model.state_dict()

        def get_effective_batch_size(self, batch_size):
            return batch_size * max(1, torch.cuda.device_count())

    class TrainingMetrics:
        """简化的训练指标记录器"""
        def __init__(self):
            self.history = {'train_loss': [], 'val_loss': [], 'learning_rate': []}

        def update(self, **kwargs):
            for key, value in kwargs.items():
                if key in self.history:
                    self.history[key].append(value)

        def print_summary(self):
            print("[TRAINING METRICS SUMMARY]")
            print(f"  Total epochs: {len(self.history['train_loss'])}")
            if self.history['train_loss']:
                print(f"  Final train loss: {self.history['train_loss'][-1]:.4f}")

except ImportError:
    # Fallback to current version
    from src.training.world_model_train_v4 import WorldModelTrainer as Phase1WorldModelTrainer

    class MultiGPUManager:
        """简化的多GPU管理器"""
        def __init__(self, config):
            self.config = config
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            self.multi_gpu = torch.cuda.device_count() > 1

        def wrap_model(self, model):
            if self.multi_gpu:
                return torch.nn.DataParallel(model)
            return model

        def get_model_state_dict(self, model):
            if self.multi_gpu:
                return model.module.state_dict()
            return model.state_dict()

        def get_effective_batch_size(self, batch_size):
            return batch_size * max(1, torch.cuda.device_count())

    class TrainingMetrics:
        """简化的训练指标记录器"""
        def __init__(self):
            self.history = {'train_loss': [], 'val_loss': [], 'learning_rate': []}

        def update(self, **kwargs):
            for key, value in kwargs.items():
                if key in self.history:
                    self.history[key].append(value)

        def print_summary(self):
            print("[TRAINING METRICS SUMMARY]")
            print(f"  Total epochs: {len(self.history['train_loss'])}")
            if self.history['train_loss']:
                print(f"  Final train loss: {self.history['train_loss'][-1]:.4f}")


# =============================================================================
# 课程级别配置（从配置文件读取）
# =============================================================================
DEFAULT_CURRICULUM_LEVELS = [
    {
        'level': 1,
        'name': '基础场景',
        'max_vehicles': 10,
        'inflow_rate': 800,
        'icv_ratio': 0.3,
        'disturbance_level': 0.0,
        'total_timesteps': 126000,
    },
    {
        'level': 2,
        'name': '中等流量',
        'max_vehicles': 15,
        'inflow_rate': 1200,
        'icv_ratio': 0.25,
        'disturbance_level': 0.2,
        'total_timesteps': 126000,
    },
    {
        'level': 3,
        'name': '高流量场景',
        'max_vehicles': 20,
        'inflow_rate': 1800,
        'icv_ratio': 0.25,
        'disturbance_level': 0.4,
        'total_timesteps': 126000,
    },
    {
        'level': 4,
        'name': '极端场景',
        'max_vehicles': 32,
        'inflow_rate': 2400,
        'icv_ratio': 0.15,
        'disturbance_level': 0.7,
        'total_timesteps': 126000,
    },
    {
        'level': 5,
        'name': '赛题场景',
        'max_vehicles': 32,
        'inflow_rate': 2000,
        'icv_ratio': 0.25,
        'disturbance_level': 0.5,
        'total_timesteps': 996000,
    },
]


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    print(f"\n{'='*80}")
    print(f"加载配置文件: {config_path}")
    print(f"{'='*80}\n")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 使用配置文件中的课程级别
    if 'curriculum_levels' in config:
        curriculum = config['curriculum_levels']
        print(f"[CONFIG] 从配置文件读取 {len(curriculum)} 个课程级别")
        return config, curriculum

    return config, DEFAULT_CURRICULUM_LEVELS


def set_seed(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# =============================================================================
# Phase 1: 世界模型预训练
# =============================================================================
def train_phase1(config: Dict[str, Any], device: torch.device):
    """
    Phase 1: 世界模型预训练（使用恢复的正确实现）

    训练目标：
        - 学习车辆状态编码器（GNN）
        - 学习交通流预测器（RSSM）
        - 学习动态权重门控（场景识别）
    """
    print("\n" + "="*80)
    print("[PHASE 1] 世界模型预训练")
    print("="*80)
    print("训练目标：")
    print("  1. 学习车辆状态编码器（RiskSensitiveGNN）")
    print("  2. 学习交通流预测器（MultiScaleRSSM）")
    print("  3. 学习动态权重门控（场景识别）")
    print("  4. 为Phase 2提供良好的初始化")
    print("="*80 + "\n")

    # 导入必要的依赖
    from train_phase1 import Phase1WorldModelTrainer, EnhancedTrainingManager

    # 创建增强训练管理器
    enhanced_manager = EnhancedTrainingManager(config)

    # 创建Phase 1训练器
    phase1_trainer = Phase1WorldModelTrainer(
        config=config,
        enhanced_manager=enhanced_manager
    )

    # 开始训练
    checkpoint_path = phase1_trainer.train()

    print(f"\n[SUCCESS] Phase 1 训练完成！")
    print(f"  - 检查点: {checkpoint_path}")
    print(f"  - 将用于Phase 2的初始化\n")

    return checkpoint_path


# =============================================================================
# Phase 2: PPO课程学习训练
# =============================================================================
def train_phase2_level(
    config: Dict[str, Any],
    level_config: Dict[str, Any],
    level: int,
    prev_checkpoint: str = None,
    phase1_checkpoint: str = None,
    device: torch.device = None
):
    """
    Phase 2: 训练单个课程级别

    Args:
        config: 全局配置
        level_config: 当前级别配置
        level: 级别编号 (1-5)
        prev_checkpoint: 上一级别检查点
        phase1_checkpoint: Phase 1检查点
        device: 计算设备

    Returns:
        检查点路径
    """
    print("\n" + "="*80)
    print(f"[PHASE 2] Level {level}/{len(config.get('curriculum_levels', DEFAULT_CURRICULUM_LEVELS))}")
    print(f"           {level_config['name']}")
    print("="*80)
    print(f"参数配置：")
    print(f"  - max_vehicles:      {level_config['max_vehicles']}")
    print(f"  - inflow_rate:       {level_config['inflow_rate']} veh/h")
    print(f"  - icv_ratio:         {level_config['icv_ratio']*100}%")
    print(f"  - disturbance_level: {level_config['disturbance_level']}")
    print(f"  - total_timesteps:   {level_config['total_timesteps']:,}")
    print("="*80 + "\n")

    # 更新环境配置
    env_config = config['environment'].copy()
    env_config.update({
        'max_vehicles': level_config['max_vehicles'],
        'inflow_rate': level_config['inflow_rate'],
        'icv_ratio': level_config['icv_ratio'],
        'disturbance_level': level_config['disturbance_level'],
    })

    # 创建环境
    print("[ENV] 创建训练环境...")
    from src.env.vec_env import create_parallel_envs
    env = create_parallel_envs(
        config=env_config,
        num_envs=config['training']['phase2']['num_envs'],
        seed=config['seed']
    )

    # 创建策略网络
    print("[MODEL] 创建策略网络...")
    # 获取观察空间和动作空间
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"[MODEL] 观测维度: {obs_dim}, 动作维度: {action_dim}")

    # 创建策略实例
    policy = IdealTrafficPolicyV4(
        obs_dim=obs_dim,
        action_dim=action_dim,
        config=config
    )

    # 加载Phase 1权重（如果是Level 1）
    if level == 1 and phase1_checkpoint is not None:
        print(f"[CHECKPOINT] 加载Phase 1预训练权重: {phase1_checkpoint}")
        phase1_state = torch.load(phase1_checkpoint, map_location=device)
        # TODO: 提取并加载GNN和RSSM的权重
        # 这里需要根据实际的checkpoint格式进行调整

    # 加载上一级别权重（如果是Level 2-5）
    if prev_checkpoint is not None:
        print(f"[CHECKPOINT] 加载Level {level-1}权重: {prev_checkpoint}")
        # CustomPPOTrainer会自动加载

    # 创建PPO训练器
    checkpoint_dir = Path(config['paths']['checkpoint_dir']) / 'preliminary' / f'level{level}'
    log_dir = Path(config['paths']['log_dir']) / 'preliminary' / f'level{level}'

    trainer = CustomPPOTrainer(
        policy=policy,
        env=env,
        config=config['training']['phase2'],
        device=device,
        checkpoint_dir=str(checkpoint_dir)
    )

    # 如果有上一级别检查点，加载它
    if prev_checkpoint is not None:
        trainer.load(prev_checkpoint)
        print(f"[OK] 成功加载Level {level-1}检查点")

    # 开始训练
    print(f"\n[TRAIN] 开始训练Level {level}...")
    print(f"预计步数: {level_config['total_timesteps']:,}\n")

    trainer.learn(
        total_timesteps=level_config['total_timesteps']
    )

    # 保存检查点
    checkpoint_path = checkpoint_dir / 'custom_ppo.zip'
    trainer.save(str(checkpoint_path))

    print(f"\n[SUCCESS] Level {level} 训练完成！")
    print(f"  - 检查点保存至: {checkpoint_path}\n")

    env.close()

    return str(checkpoint_path)


def train_phase2_curriculum(
    config: Dict[str, Any],
    curriculum_levels: List[Dict[str, Any]],
    phase1_checkpoint: str = None,
    start_level: int = 1,
    device: torch.device = None
):
    """
    Phase 2: 完整课程学习训练（Level 1-5）

    Args:
        config: 全局配置
        curriculum_levels: 课程级别配置
        phase1_checkpoint: Phase 1检查点
        start_level: 从哪个级别开始
        device: 计算设备
    """
    print("\n" + "="*80)
    print("[PHASE 2] PPO课程学习训练")
    print("="*80)
    print(f"课程级别：{len(curriculum_levels)} 个")
    print(f"起始级别：Level {start_level}")
    print("="*80 + "\n")

    current_checkpoint = None
    phase1_used = False

    for level_config in curriculum_levels:
        level = level_config['level']

        # 跳过之前的级别
        if level < start_level:
            print(f"[SKIP] 跳过Level {level}（从Level {start_level}开始）")
            if level == 1:
                # 尝试找到Level 1的检查点
                checkpoint_path = Path(config['paths']['checkpoint_dir']) / 'preliminary' / 'level1' / 'custom_ppo.zip'
                if checkpoint_path.exists():
                    current_checkpoint = str(checkpoint_path)
                    print(f"[OK] 找到Level 1检查点: {current_checkpoint}")
            continue

        # 决定使用哪个检查点
        if level == 1 and phase1_checkpoint is not None:
            checkpoint_for_training = phase1_checkpoint
            phase1_used = True
        else:
            checkpoint_for_training = current_checkpoint

        # 训练当前级别
        current_checkpoint = train_phase2_level(
            config=config,
            level_config=level_config,
            level=level,
            prev_checkpoint=None if level == 1 else current_checkpoint,
            phase1_checkpoint=phase1_checkpoint if level == 1 else None,
            device=device
        )

        print(f"\n[PROGRESS] 完成Level {level}/{len(curriculum_levels)}")
        print(f"{'='*80}\n")

    print("\n" + "="*80)
    print("[SUCCESS] 所有课程级别训练完成！")
    print("="*80)
    print(f"最终模型: {current_checkpoint}")
    print("\n可以运行以下命令评估模型：")
    print(f"  python evaluate_v4_ideal.py --checkpoint {current_checkpoint}")
    print("="*80 + "\n")

    return current_checkpoint


# =============================================================================
# 主训练流程
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='初赛专用训练脚本')

    parser.add_argument(
        '--config',
        type=str,
        default='configs/competition_preliminary.yaml',
        help='配置文件路径'
    )

    parser.add_argument(
        '--skip-phase1',
        action='store_true',
        help='跳过Phase 1，直接进行Phase 2'
    )

    parser.add_argument(
        '--start-level',
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help='从哪个级别开始训练（1-5）'
    )

    parser.add_argument(
        '--phase1-checkpoint',
        type=str,
        default=None,
        help='Phase 1检查点路径（如果跳过Phase 1训练）'
    )

    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='计算设备（cuda:0, cpu等），默认使用配置文件中的设置'
    )

    args = parser.parse_args()

    # 加载配置
    config, curriculum_levels = load_config(args.config)

    # 设置设备
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device_str = config.get('device', 'cuda:0')
        device = torch.device(device_str if torch.cuda.is_available() else 'cpu')

    print(f"\n[DEVICE] 使用设备: {device}")
    if device.type == 'cuda':
        print(f"[GPU]     GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"[GPU]     显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")

    # 设置随机种子
    set_seed(config['seed'])

    # 创建必要的目录
    Path(config['paths']['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['paths']['log_dir']).mkdir(parents=True, exist_ok=True)

    # 记录开始时间
    start_time = datetime.now()
    print(f"\n[START] 训练开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ========================================================================
    # Phase 1: 世界模型预训练（可选）
    # ========================================================================
    phase1_checkpoint = None

    if not args.skip_phase1:
        # 检查是否已有Phase 1检查点
        default_phase1_path = Path(config['paths']['checkpoint_dir']) / 'preliminary' / 'phase1' / 'world_model_final.pth'

        if default_phase1_path.exists() and args.start_level > 1:
            print(f"[CHECKPOINT] 发现已存在的Phase 1检查点: {default_phase1_path}")
            user_input = input("是否使用已有检查点？[Y/n] ").strip().lower()
            if user_input != 'n':
                phase1_checkpoint = str(default_phase1_path)
                print("[OK] 使用已有Phase 1检查点\n")
            else:
                print("[INFO] 将重新训练Phase 1\n")

        if phase1_checkpoint is None:
            phase1_checkpoint = train_phase1(config, device)
    else:
        print("\n[SKIP] 跳过Phase 1训练\n")
        if args.phase1_checkpoint is not None:
            phase1_checkpoint = args.phase1_checkpoint
            print(f"[CHECKPOINT] 使用指定的Phase 1检查点: {phase1_checkpoint}\n")

    # ========================================================================
    # Phase 2: PPO课程学习训练
    # ========================================================================
    final_checkpoint = train_phase2_curriculum(
        config=config,
        curriculum_levels=curriculum_levels,
        phase1_checkpoint=phase1_checkpoint,
        start_level=args.start_level,
        device=device
    )

    # 训练完成
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "="*80)
    print("[SUCCESS] 初赛训练完成！")
    print("="*80)
    print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时:   {duration}")
    print("="*80)
    print(f"\n最终模型: {final_checkpoint}")
    print(f"\n下一步：")
    print(f"  1. 评估模型性能")
    print(f"     python evaluate_v4_ideal.py --checkpoint {final_checkpoint}")
    print(f"  2. 如果满意，用于提交比赛")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
