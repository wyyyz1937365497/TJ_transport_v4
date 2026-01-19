"""
Phase 2 PPO训练 - 简化版本（支持课程学习）

使用配置文件控制所有课程级别，无需创建多个脚本。

使用方法：
    # Stage 1（基础场景）
    python train_phase2.py --stage 1

    # Stage 2-5（加载上一阶段权重）
    python train_phase2.py --stage 2 --prev-checkpoint checkpoints/competition/curriculum/level1/custom_ppo.zip
    python train_phase2.py --stage 3 --prev-checkpoint checkpoints/competition/curriculum/level2/custom_ppo.zip
    python train_phase2.py --stage 4 --prev-checkpoint checkpoints/competition/curriculum/level3/custom_ppo.zip
    python train_phase2.py --stage 5 --prev-checkpoint checkpoints/competition/curriculum/level4/custom_ppo.zip

    # 或者使用bat文件自动训练所有阶段
    train_all_stages.bat
"""

import os
import sys
import argparse
import yaml
import torch
from pathlib import Path
from typing import Dict, Any

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.ideal_policy_v4 import create_ideal_traffic_policy_v4
from src.env.vec_env import create_parallel_envs
from src.training.custom_ppo_trainer import CustomPPOTrainer


# 课程级别配置（从配置文件读取）
CURRICULUM_LEVELS = [
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
        'icv_ratio': 0.2,
        'disturbance_level': 0.5,
        'total_timesteps': 1494000,
    },
]


def load_config(config_path: str = 'configs/competition.yaml') -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def train_stage(
    config: dict,
    stage: int,
    prev_checkpoint: str = None,
    phase1_checkpoint: str = None
):
    """训练指定阶段"""

    stage_config = CURRICULUM_LEVELS[stage - 1]

    print("\n" + "=" * 80)
    print(f"[PHASE 2] Stage {stage}/{len(CURRICULUM_LEVELS)}: {stage_config['name']}")
    print("=" * 80)

    print(f"\n[CONFIG] {stage_config['name']}:")
    print(f"  - Max vehicles: {stage_config['max_vehicles']}")
    print(f"  - Inflow rate: {stage_config['inflow_rate']}")
    print(f"  - ICV ratio: {stage_config['icv_ratio']}")
    print(f"  - Disturbance: {stage_config['disturbance_level']}")
    print(f"  - Total timesteps: {stage_config['total_timesteps']:,}")

    # 设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"\n[DEVICE] {device}")

    # 环境配置
    env_config = config.get('environment', {}).copy()
    env_config.update({
        'max_vehicles': stage_config['max_vehicles'],
        'inflow_rate': stage_config['inflow_rate'],
        'icv_ratio': stage_config['icv_ratio'],
        'disturbance_level': stage_config['disturbance_level'],
    })

    # Phase 2配置
    phase2_config = config.get('training', {}).get('phase2', {})

    num_envs = int(phase2_config.get('num_envs', 4))
    n_steps = int(phase2_config.get('n_steps', 2048))
    batch_size = int(phase2_config.get('batch_size', 64))
    n_epochs = int(phase2_config.get('update_epochs', 10))
    learning_rate = float(phase2_config.get('learning_rate', 3e-4))

    print(f"\n[TRAINING] Parallel envs: {num_envs}, Steps: {n_steps}, Batch: {batch_size}, Epochs: {n_epochs}")

    # 创建环境
    print(f"\n[ENV] Creating {num_envs} parallel environments...")
    vec_env_wrapper = create_parallel_envs(
        config=env_config,
        num_envs=num_envs,
        seed=config.get('seed', 42),
        device=str(device),
    )
    vec_env = vec_env_wrapper.vec_env
    print(f"[OK] Environments created")

    # 创建策略
    print("\n[MODEL] Creating policy network...")
    PolicyClass = create_ideal_traffic_policy_v4(config)

    # 实例化策略网络（不需要传入model配置，已通过config属性传递）
    policy = PolicyClass(
        observation_space=vec_env.observation_space,
        action_space=vec_env.action_space,
        lr_schedule=lambda _: learning_rate,
    )

    # 将策略移动到设备
    policy = policy.to(device)

    print("[INFO] torch.compile() is disabled to avoid dynamic shape issues")

    # 加载权重
    # 智能识别checkpoint类型：Phase 1 (.pth) 或 Phase 2 (.zip)
    checkpoint_to_load = None
    checkpoint_type = None

    if prev_checkpoint and os.path.exists(prev_checkpoint):
        checkpoint_to_load = prev_checkpoint
        # 判断checkpoint类型
        if prev_checkpoint.endswith('.pth'):
            checkpoint_type = 'phase1'
        elif prev_checkpoint.endswith('.zip'):
            checkpoint_type = 'phase2'
        else:
            # 通过内容判断
            checkpoint = torch.load(prev_checkpoint, map_location=str(device), weights_only=False)
            if 'model_state_dict' in checkpoint:
                checkpoint_type = 'phase1'
            elif 'policy_state_dict' in checkpoint:
                checkpoint_type = 'phase2'

    elif phase1_checkpoint and os.path.exists(phase1_checkpoint):
        checkpoint_to_load = phase1_checkpoint
        checkpoint_type = 'phase1'

    # 加载checkpoint
    if checkpoint_to_load:
        print(f"\n[LOAD] Loading checkpoint: {checkpoint_to_load} (type: {checkpoint_type})")
        checkpoint = torch.load(checkpoint_to_load, map_location=str(device), weights_only=False)

        if checkpoint_type == 'phase2':
            # Phase 2 checkpoint - 加载PPO策略
            if 'policy_state_dict' in checkpoint:
                policy.load_state_dict(checkpoint['policy_state_dict'])
                print(f"[OK] Loaded Phase 2 policy weights")
            else:
                print(f"[WARN] No 'policy_state_dict' found, trying direct load")
                try:
                    policy.load_state_dict(checkpoint)
                    print(f"[OK] Loaded directly")
                except Exception as e:
                    print(f"[ERROR] Failed to load: {e}")

        elif checkpoint_type == 'phase1':
            # Phase 1 checkpoint - 从完整模型中提取匹配的权重
            print(f"[DEBUG] Checkpoint keys: {list(checkpoint.keys())}")

            if 'model_state_dict' in checkpoint:
                phase1_state_dict = checkpoint['model_state_dict']
                if phase1_state_dict is not None:
                    # 从Phase 1的完整模型state_dict中提取匹配的权重
                    policy_state_dict = policy.state_dict()
                    matched_weights = {}

                    for key in policy_state_dict.keys():
                        if key in phase1_state_dict:
                            if phase1_state_dict[key].shape == policy_state_dict[key].shape:
                                matched_weights[key] = phase1_state_dict[key]
                                print(f"[MATCH] {key}: {phase1_state_dict[key].shape} -> {policy_state_dict[key].shape}")
                            else:
                                print(f"[SKIP] {key}: shape mismatch {phase1_state_dict[key].shape} vs {policy_state_dict[key].shape}")

                    print(f"[INFO] Matched {len(matched_weights)}/{len(policy_state_dict)} weights")

                    if len(matched_weights) > 0:
                        policy.load_state_dict(matched_weights, strict=False)
                        print(f"[OK] Loaded Phase 1 weights ({len(matched_weights)} parameters)")
                    else:
                        print(f"[WARN] No matching weights found, starting from scratch")
                else:
                    print(f"[ERROR] 'model_state_dict' is None!")
            else:
                print(f"[ERROR] No 'model_state_dict' found in checkpoint!")
    else:
        print(f"\n[INFO] No checkpoint found, starting from scratch")

    # 冻结感知层和预测层
    print("\n[FREEZE] Freezing perception and prediction layers...")
    for name, param in policy.named_parameters():
        if 'gnn' in name or 'world_model' in name:
            param.requires_grad = False
    print(f"[OK] Frozen")

    # 创建训练器
    print("\n[TRAINER] Creating Custom PPO Trainer...")

    trainer_config = {
        'learning_rate': learning_rate,
        'n_steps': n_steps,
        'batch_size': batch_size,
        'n_epochs': n_epochs,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
    }

    checkpoint_dir = Path(config.get('paths', {}).get('checkpoint_dir', 'checkpoints/competition'))
    checkpoint_path = checkpoint_dir / 'curriculum' / f'level{stage}'

    trainer = CustomPPOTrainer(
        policy=policy,
        env=vec_env,
        config=trainer_config,
        device=device,
        checkpoint_dir=str(checkpoint_path),
    )

    print(f"[OK] Trainer ready")

    # 开始训练
    print("\n" + "=" * 80)
    print("[TRAIN] Starting training...")
    print("=" * 80)

    trainer.learn(total_timesteps=stage_config['total_timesteps'])

    print(f"\n[DONE] Stage {stage} completed!")
    print(f"[SAVE] Checkpoint: {checkpoint_path / 'custom_ppo.zip'}")


def main():
    parser = argparse.ArgumentParser(description='Phase 2 PPO Training')
    parser.add_argument('--stage', type=int, choices=[1, 2, 3, 4, 5],
                        help='Curriculum level (1-5)')
    parser.add_argument('--prev-checkpoint', type=str,
                        help='Previous stage checkpoint')
    parser.add_argument('--phase1-checkpoint', type=str,
                        default='checkpoints/competition/phase1/world_model_final.pth',
                        help='Phase 1 checkpoint (for stage 1)')
    parser.add_argument('--config', type=str, default='configs/competition.yaml',
                        help='Config file path')

    args = parser.parse_args()

    if args.stage is None:
        parser.print_help()
        print("\n[ERROR] --stage is required")
        print("[INFO] Use train_all_stages.bat to train all stages automatically")
        return

    config = load_config(args.config)

    train_stage(
        config=config,
        stage=args.stage,
        prev_checkpoint=args.prev_checkpoint,
        phase1_checkpoint=args.phase1_checkpoint,
    )


if __name__ == '__main__':
    main()
