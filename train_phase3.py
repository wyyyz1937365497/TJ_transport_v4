"""
Phase 3: 微调与优化

训练目标：
- 基于Phase 2的最佳模型进行微调
- 针对赛题特定场景优化
- 提升模型在真实仿真中的表现
- 集成学习（可选）

使用方法：
    python train_phase3.py --base-model checkpoints/competition/curriculum/level5/custom_ppo.zip
    python train_phase3.py --base-model checkpoints/competition/best_model.zip --fine-tune
    python train_phase3.py --base-model checkpoints/competition/best_model.zip --ensemble
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.ideal_policy_v4 import create_ideal_traffic_policy_v4
from src.env.vec_env import create_parallel_envs
from src.training.custom_ppo_trainer import CustomPPOTrainer


def load_config(config_path: str = 'configs/competition.yaml') -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def train_fine_tuning(
    config: dict,
    base_model_path: str,
    output_dir: str = None
):
    """
    微调训练（Phase 3a）

    在赛题真实场景上微调，提升Sim-to-Real性能
    """
    print("\n" + "=" * 80)
    print("[PHASE 3a] Fine-tuning on Competition Scenario")
    print("=" * 80)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"\n[DEVICE] {device}")

    # 赛题场景配置
    env_config = config.get('environment', {}).copy()
    env_config.update({
        'max_vehicles': 32,
        'inflow_rate': 2000,
        'icv_ratio': 0.2,
        'disturbance_level': 0.5,
        'scenario': 'competition',  # 使用真实赛题场景
    })

    # 微调训练配置（更保守的学习率）
    phase3_config = config.get('training', {}).get('phase3', {})

    num_envs = int(phase3_config.get('num_envs', 4))
    n_steps = int(phase3_config.get('n_steps', 2048))
    batch_size = int(phase3_config.get('batch_size', 64))
    n_epochs = int(phase3_config.get('update_epochs', 10))
    learning_rate = float(phase3_config.get('learning_rate', 1e-4))  # 更小的学习率
    total_timesteps = int(phase3_config.get('total_timesteps', 500000))

    print(f"\n[CONFIG] Fine-tuning Configuration:")
    print(f"  - Base model: {base_model_path}")
    print(f"  - Learning rate: {learning_rate} (smaller for fine-tuning)")
    print(f"  - Total timesteps: {total_timesteps:,}")
    print(f"  - Parallel envs: {num_envs}")

    # 创建环境
    print(f"\n[ENV] Creating {num_envs} parallel competition environments...")
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

    policy = PolicyClass(
        observation_space=vec_env.observation_space,
        action_space=vec_env.action_space,
        lr_schedule=lambda _: learning_rate,
    )

    # 将策略移动到设备
    policy = policy.to(device)

    # 加载基础模型权重
    print(f"\n[LOAD] Loading base model from: {base_model_path}")
    checkpoint = torch.load(base_model_path, map_location=device)

    # 加载策略权重
    if 'policy_state_dict' in checkpoint:
        policy.load_state_dict(checkpoint['policy_state_dict'])
        print(f"[OK] Policy weights loaded")
    else:
        print("[WARN] Checkpoint format not recognized, trying direct load...")
        policy.load_state_dict(checkpoint)

    # 创建PPO训练器
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
        'target_kl': 0.01,
        'warmup_steps': 500,
    }

    trainer = CustomPPOTrainer(
        policy=policy,
        env=vec_env,
        config=trainer_config,
        device=device,
        checkpoint_dir=output_dir or 'checkpoints/competition/phase3/fine_tune',
    )

    # 开始微调
    print("\n" + "=" * 80)
    print("[TRAIN] Starting Fine-tuning...")
    print("=" * 80)

    trainer.learn(total_timesteps=total_timesteps)

    # 保存微调后的模型
    checkpoint_dir = Path(output_dir or 'checkpoints/competition/phase3/fine_tune')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    final_model = checkpoint_dir / 'fine_tuned_model.zip'
    trainer.save_checkpoint(final_model)

    print("\n" + "=" * 80)
    print("[DONE] Fine-tuning Completed!")
    print(f"[SAVE] Fine-tuned model: {final_model}")
    print("=" * 80)


def train_ensemble(
    config: dict,
    base_models: list,
    output_dir: str = None
):
    """
    集成学习（Phase 3b - 可选）

    融合多个模型的预测，提升鲁棒性
    """
    print("\n" + "=" * 80)
    print("[PHASE 3b] Ensemble Learning")
    print("=" * 80)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f"\n[CONFIG] Ensemble Configuration:")
    print(f"  - Base models: {len(base_models)}")
    for i, model_path in enumerate(base_models):
        print(f"    Model {i+1}: {model_path}")

    # TODO: 实现集成学习
    print("\n[INFO] Ensemble learning is under development")
    print("[INFO] Skipping ensemble training for now")


def evaluate_model(
    config: dict,
    model_path: str,
    num_episodes: int = 10
):
    """
    评估模型性能
    """
    print("\n" + "=" * 80)
    print("[EVAL] Model Evaluation")
    print("=" * 80)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 创建评估环境
    env_config = config.get('environment', {}).copy()
    env_config.update({
        'max_vehicles': 32,
        'inflow_rate': 2000,
        'icv_ratio': 0.2,
        'disturbance_level': 0.5,
    })

    vec_env_wrapper = create_parallel_envs(
        config=env_config,
        num_envs=1,  # 评估时使用单环境
        seed=config.get('seed', 42) + 1000,  # 不同的种子
        device=str(device),
    )
    vec_env = vec_env_wrapper.vec_env

    # 创建策略
    PolicyClass = create_ideal_traffic_policy_v4(config)
    policy = PolicyClass(
        observation_space=vec_env.observation_space,
        action_space=vec_env.action_space,
        lr_schedule=lambda _: 1e-4,
    )
    policy = policy.to(device)

    # 加载模型
    print(f"\n[LOAD] Loading model: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    if 'policy_state_dict' in checkpoint:
        policy.load_state_dict(checkpoint['policy_state_dict'])

    # 评估循环
    print(f"\n[EVAL] Running {num_episodes} episodes...")

    episode_rewards = []
    episode_lengths = []

    for episode in range(num_episodes):
        obs, _ = vec_env.reset()
        done = False
        truncated = False
        episode_reward = 0
        episode_length = 0

        while not (done or truncated):
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32).to(device)
                actions, _, _ = policy(obs_tensor)

            obs, reward, done, truncated, info = vec_env.step(actions.cpu().numpy())
            episode_reward += reward.sum()
            episode_length += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        print(f"  Episode {episode+1}/{num_episodes}: Reward={episode_reward:.2f}, Length={episode_length}")

    # 统计结果
    print("\n" + "=" * 80)
    print("[RESULTS] Evaluation Statistics")
    print("=" * 80)
    print(f"Mean Reward: {sum(episode_rewards)/len(episode_rewards):.2f}")
    print(f"Std Reward:  {(sum((x - sum(episode_rewards)/len(episode_rewards))**2 for x in episode_rewards)/len(episode_rewards))**0.5:.2f}")
    print(f"Mean Length: {sum(episode_lengths)/len(episode_lengths):.2f}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Phase 3: 微调与优化')

    parser.add_argument('--base-model', type=str, required=True,
                       help='基础模型路径（Phase 2的Level 5模型）')
    parser.add_argument('--mode', type=str, choices=['fine-tune', 'ensemble', 'eval'],
                       default='fine-tune', help='训练模式')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='输出目录')
    parser.add_argument('--config', type=str, default='configs/competition.yaml',
                       help='配置文件路径')
    parser.add_argument('--num-episodes', type=int, default=10,
                       help='评估时的episode数量')
    parser.add_argument('--ensemble-models', type=str, nargs='+',
                       help='集成学习的模型列表')

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    if args.mode == 'fine-tune':
        train_fine_tuning(
            config=config,
            base_model_path=args.base_model,
            output_dir=args.output_dir
        )
    elif args.mode == 'ensemble':
        if not args.ensemble_models:
            args.ensemble_models = [args.base_model]
        train_ensemble(
            config=config,
            base_models=args.ensemble_models,
            output_dir=args.output_dir
        )
    elif args.mode == 'eval':
        evaluate_model(
            config=config,
            model_path=args.base_model,
            num_episodes=args.num_episodes
        )


if __name__ == '__main__':
    main()
