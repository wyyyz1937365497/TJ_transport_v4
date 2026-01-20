"""
初赛轻量级训练脚本 - Phase 1 Lite

核心特性:
1. 简化训练流程：行为克隆 → PPO微调（2个阶段）
2. 直接优化OCR奖励
3. 轻量级GNN架构（无世界模型）
4. 稀疏控制机制（只控制5%车辆）

训练时间预估：6-8小时（vs 原架构3天）
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import argparse

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.env.gym_wrapper import GymSumoEnv
from src.env.vec_env import ParallelSumoEnvs
from src.models.v5_lightweight import create_lightweight_policy_v5
from src.training.ocr_rewards import create_ocr_reward_calculator
from src.env.sparse_controller import create_sparse_controller
from src.training.custom_ppo_trainer import CustomPPOTrainer


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_environment(config: dict):
    """创建训练环境"""
    # ✅ 修复：提取 environment 配置并与顶层配置合并
    # 因为环境代码期望从顶层 config 读取 sumo_config
    env_config = config.get('environment', {})

    # 将相对路径转换为绝对路径（基于项目根目录）
    if 'sumo_config' in env_config:
        import os
        sumo_config = env_config['sumo_config']
        if not os.path.isabs(sumo_config):
            # 转换为绝对路径
            project_root = Path(__file__).parent.resolve()
            env_config['sumo_config'] = str(project_root / sumo_config)

    # 合并配置：environment 配置覆盖顶层配置（如果存在）
    merged_config = {**config, **env_config}

    env = GymSumoEnv(
        config=merged_config,
        seed=config.get('seed', 42),
        device=config.get('device', 'cuda:0')
    )

    return env


def stage1_behavior_cloning(
    env,
    policy: nn.Module,
    config: dict,
    checkpoint_dir: Path
):
    """
    Stage 1: 行为克隆

    从人类驾驶数据（IDM模型）学习基础驾驶行为
    无需强化学习，快速收敛

    使用并行环境加速数据收集
    """
    print("\n" + "=" * 70)
    print("Stage 1: 行为克隆 (从IDM模型学习)")
    print("=" * 70)

    stage1_config = config['training']['stage1']
    num_episodes = stage1_config['num_episodes']
    batch_size = stage1_config['batch_size']
    learning_rate = stage1_config['learning_rate']
    num_epochs = stage1_config['epochs']
    num_envs = stage1_config.get('num_envs', 8)  # 并行环境数（从配置读取，默认为8）

    # 优化器
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    # 损失函数
    criterion = nn.MSELoss()

    # ========== 使用并行环境收集数据 ==========
    print(f"\n创建并行环境 ({num_envs}个环境)...")
    parallel_envs = ParallelSumoEnvs(
        config=config,
        num_envs=num_envs,
        seed=config.get('seed', 42),
        device=config.get('device', 'cuda:0')
    )

    print(f"\n并行收集数据 ({num_episodes} episodes, {num_envs}个并行环境)...")
    observations = []
    actions = []

    episode_count = 0
    current_obs = parallel_envs.reset()

    with tqdm(total=num_episodes, desc="收集数据") as pbar:
        while episode_count < num_episodes:
            # 随机动作（探索）
            actions_list = [parallel_envs.action_space.sample() for _ in range(num_envs)]
            actions_array = np.stack(actions_list)

            # 执行一步
            obs, rewards, dones, infos = parallel_envs.step(actions_array)

            # 收集数据
            for env_idx in range(num_envs):
                # 检查是否有专家动作（IDM模型）
                info = infos[env_idx] if isinstance(infos, list) else infos
                expert_action = info.get('expert_action', None) if isinstance(info, dict) else None

                if expert_action is not None:
                    observations.append(current_obs[env_idx].copy())
                    actions.append(expert_action.copy())

                # 检查episode是否结束
                done = dones[env_idx]
                if done:
                    episode_count += 1
                    pbar.update(1)
                    pbar.set_postfix({'episodes': episode_count})

            current_obs = obs

            # 提前退出
            if episode_count >= num_episodes:
                break

    # 关闭并行环境
    parallel_envs.close()

    print(f"收集到 {len(observations)} 个样本")

    # 转换为tensor
    observations = torch.FloatTensor(np.array(observations))
    actions = torch.FloatTensor(np.array(actions))

    # 训练
    print(f"\n训练行为克隆模型 ({num_epochs} epochs)...")
    policy.train()

    device = next(policy.parameters()).device
    observations = observations.to(device)
    actions = actions.to(device)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        # 随机打乱数据
        indices = np.random.permutation(len(observations))

        for i in range(0, len(observations), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_obs = observations[batch_indices]
            batch_actions = actions[batch_indices]

            # 前向传播
            policy_actions, _, _ = policy(batch_obs, deterministic=True)

            # 计算损失
            loss = criterion(policy_actions, batch_actions)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")

    # 保存checkpoint
    checkpoint_path = checkpoint_dir / "stage1_behavior_cloning.pth"
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
    }, checkpoint_path)
    print(f"\n✅ Stage 1 完成！Checkpoint保存至: {checkpoint_path}")


def stage2_ppo_finetuning(
    env,
    policy: nn.Module,
    config: dict,
    checkpoint_dir: Path
):
    """
    Stage 2: PPO微调

    使用PPO算法直接优化OCR奖励
    """
    print("\n" + "=" * 70)
    print("Stage 2: PPO微调 (直接优化OCR)")
    print("=" * 70)

    stage2_config = config['training']['stage2']

    # 创建PPO训练器
    trainer = CustomPPOTrainer(
        env=env,
        policy=policy,
        config=stage2_config,
        checkpoint_dir=checkpoint_dir,
        use_tensorboard=True
    )

    # 创建OCR奖励计算器
    reward_calculator = create_ocr_reward_calculator(
        baseline_stats=None,  # 初赛没有基准
        **config.get('ocr_rewards', {})
    )

    # 训练
    print(f"\n开始PPO训练 ({stage2_config['total_timesteps']} steps)...")
    trainer.train(reward_calculator=reward_calculator)

    print(f"\n✅ Stage 2 完成！Final checkpoint保存至: {checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(description='初赛轻量级训练脚本')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/phase1_lite.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--stage',
        type=str,
        default='all',
        choices=['stage1', 'stage2', 'all'],
        help='训练阶段'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='预训练checkpoint路径（用于stage2）'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='训练设备'
    )

    args = parser.parse_args()

    # 加载配置
    print(f"\n加载配置文件: {args.config}")
    config = load_config(args.config)

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建checkpoint目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = Path(config['paths']['checkpoint_dir']) / f"phase1_lite_{timestamp}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint目录: {checkpoint_dir}")

    # 保存配置
    config_path = checkpoint_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    # 创建环境
    print("\n创建训练环境...")
    env = create_environment(config)

    # 获取观测和动作空间维度
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print(f"观测维度: {obs_dim}")
    print(f"动作维度: {action_dim}")

    # 创建策略网络
    print("\n创建轻量级策略网络...")
    policy = create_lightweight_policy_v5(
        obs_dim=obs_dim,
        action_dim=action_dim,
        config=config
    )
    policy = policy.to(device)

    # 统计参数量
    num_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"策略网络参数量: {num_params:,}")

    # ========== Stage 1: 行为克隆 ==========
    if args.stage in ['stage1', 'all']:
        stage1_behavior_cloning(env, policy, config, checkpoint_dir)

    # ========== Stage 2: PPO微调 ==========
    if args.stage in ['stage2', 'all']:
        # 加载预训练权重（如果有）
        if args.checkpoint is not None:
            print(f"\n加载预训练checkpoint: {args.checkpoint}")
            checkpoint = torch.load(args.checkpoint, map_location=device)
            policy.load_state_dict(checkpoint['policy_state_dict'])
            print("✅ 预训练权重加载完成")

        stage2_ppo_finetuning(env, policy, config, checkpoint_dir)

    print("\n" + "=" * 70)
    print("🎉 训练完成！")
    print(f"所有checkpoint保存在: {checkpoint_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
