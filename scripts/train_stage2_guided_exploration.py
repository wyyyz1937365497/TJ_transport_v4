#!/usr/bin/env python3
"""
Stage 2: 引导探索训练（Guided Exploration Training）

功能：
1. 加载Stage 1预训练的WorldModel
2. PPO训练优化OCR奖励
3. 固定奖励权重

使用方法：
    python scripts/train_stage2_guided_exploration.py --config configs/v5_complete.yaml --resume checkpoints/v5_complete/stage1_best.pth

输出：
    - checkpoints/v5_complete/stage2_guided.pth
    - logs/v5_complete/stage2/
"""

import os
import sys
import argparse
import yaml
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.joint_icv_policy import create_joint_icv_policy
from src.models.world_model import WorldModel, create_world_model
from src.env.competition_env import CompetitionSumoEnv
from src.training.ocr_rewards import OCRRewardComputer


class PPOTrainer:
    """
    PPO训练器（Stage 2）
    """

    def __init__(self, policy, env, config, device='cuda'):
        """
        Args:
            policy: JointICVPolicy
            env: CompetitionSumoEnv
            config: 配置字典
            device: 设备
        """
        self.policy = policy
        self.env = env
        self.config = config
        self.device = device

        # PPO配置
        ppo_config = config['stage2_guided_exploration']['ppo']

        self.num_iterations = ppo_config['num_iterations']
        self.num_steps_per_iteration = ppo_config['num_steps_per_iteration']
        self.batch_size = ppo_config['batch_size']
        self.num_minibatches = ppo_config['num_minibatches']

        self.gamma = ppo_config['gamma']
        self.gae_lambda = ppo_config['gae_lambda']
        self.clip_param = ppo_config['clip_param']

        self.update_epochs = ppo_config['update_epochs']
        self.entropy_coef = ppo_config['entropy_coef']
        self.vf_coef = ppo_config['vf_coef']

        # 优化器
        self.optimizer = optim.Adam(
            policy.parameters(),
            lr=ppo_config['optimizer']['lr'],
            weight_decay=ppo_config['optimizer']['weight_decay']
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=ppo_config['lr_schedule']['start_factor'],
            total_iters=ppo_config['lr_schedule']['total_iters'],
            end_factor=ppo_config['lr_schedule']['end_factor']
        )

        # 混合精度训练
        self.scaler = GradScaler()

        # 奖励计算
        self.reward_computer = OCRRewardComputer()

        # Rollout buffer
        self.buffer = {
            'observations': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'values': [],
            'dones': []
        }

    def collect_rollouts(self, num_steps):
        """
        收集rollout数据

        Args:
            num_steps: 收集的步数

        Returns:
            rollouts: Dict
        """
        self.policy.eval()

        obs, _ = self.env.reset()
        episode_rewards = []

        for step in tqdm(range(num_steps), desc="收集rollouts"):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)

            # 前向传播
            with torch.no_grad():
                outputs = self.policy(obs_tensor, deterministic=False)

            action = outputs['actions'][0].cpu().numpy()
            value = outputs['value'][0, 0].cpu().item()
            log_prob = outputs['log_prob'][0].cpu().item()

            # 执行动作
            next_obs, reward, done, truncated, info = self.env.step(action)

            # 存储到buffer
            self.buffer['observations'].append(obs.copy())
            self.buffer['actions'].append(action.copy())
            self.buffer['log_probs'].append(log_prob)
            self.buffer['rewards'].append(reward)
            self.buffer['values'].append(value)
            self.buffer['dones'].append(done or truncated)

            obs = next_obs

            if done or truncated:
                obs, _ = self.env.reset()

        # 转换为tensor
        for key in self.buffer.keys():
            self.buffer[key] = np.array(self.buffer[key])

        return self.buffer

    def compute_gae(self, rewards, values, dones):
        """
        计算GAE优势

        Args:
            rewards: [T]
            values: [T]
            dones: [T]

        Returns:
            advantages: [T]
            returns: [T]
        """
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        returns = np.zeros_like(rewards)
        last_return = 0

        for t in reversed(range(rewards.shape[0])):
            if t == rewards.shape[0] - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_advantage = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            returns[t] = last_return = rewards[t] + self.gamma * (1 - dones[t]) * last_return

        # 归一化advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def update_policy(self, rollouts):
        """
        更新策略

        Args:
            rollouts: rollout数据

        Returns:
            metrics: Dict
        """
        observations = rollouts['observations']
        actions = rollouts['actions']
        old_log_probs = rollouts['log_probs']
        rewards = rollouts['rewards']
        values = rollouts['values']
        dones = rollouts['dones']

        # 计算GAE
        advantages, returns = self.compute_gae(rewards, values, dones)

        # 转换为tensor
        obs_tensor = torch.as_tensor(observations, dtype=torch.float32).to(self.device)
        actions_tensor = torch.as_tensor(actions, dtype=torch.float32).to(self.device)
        old_log_probs_tensor = torch.as_tensor(old_log_probs, dtype=torch.float32).to(self.device)
        advantages_tensor = torch.as_tensor(advantages, dtype=torch.float32).to(self.device)
        returns_tensor = torch.as_tensor(returns, dtype=torch.float32).to(self.device)

        # Minibatch更新
        batch_size = observations.shape[0]
        minibatch_size = batch_size // self.num_minibatches

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0

        for epoch in range(self.update_epochs):
            indices = np.random.permutation(batch_size)

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_indices = indices[start:end]

                # 获取minibatch数据
                mb_obs = obs_tensor[mb_indices]
                mb_actions = actions_tensor[mb_indices]
                mb_old_log_probs = old_log_probs_tensor[mb_indices]
                mb_advantages = advantages_tensor[mb_indices]
                mb_returns = returns_tensor[mb_indices]

                # 重新计算log_prob
                self.policy.train()
                outputs = self.policy(mb_obs, deterministic=False)

                new_log_probs = outputs['log_prob']
                entropy = outputs['log_prob']  # 简化：使用log_prob作为熵的代理
                value = outputs['value']

                # 计算ratio
                ratio = torch.exp(new_log_probs - mb_old_log_probs)

                # PPO clip loss
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(value.squeeze(1), mb_returns)

                # 总损失
                loss = policy_loss + self.vf_coef * value_loss - self.entropy_coef * entropy.mean()

                # 反向传播
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config['stage2_guided_exploration']['ppo']['grad_clip'])
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # 统计
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        # 更新学习率
        self.scheduler.step()

        metrics = {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates,
            'mean_reward': rewards.mean(),
            'mean_return': returns.mean()
        }

        return metrics

    def train(self):
        """训练循环"""
        print("\n" + "=" * 80)
        print("开始Stage 2 PPO训练")
        print("=" * 80)

        best_reward = float('-inf')
        checkpoint_dir = Path(self.config['global']['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for iteration in range(self.num_iterations):
            print(f"\n迭代 {iteration + 1}/{self.num_iterations}")

            # 收集rollouts
            rollouts = self.collect_rollouts(self.num_steps_per_iteration)

            # 更新策略
            metrics = self.update_policy(rollouts)

            print(f"  Policy Loss: {metrics['policy_loss']:.4f}")
            print(f"  Value Loss: {metrics['value_loss']:.4f}")
            print(f"  Entropy: {metrics['entropy']:.4f}")
            print(f"  Mean Reward: {metrics['mean_reward']:.4f}")
            print(f"  Mean Return: {metrics['mean_return']:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            # 保存检查点
            if (iteration + 1) % self.config['stage2_guided_exploration']['checkpoint']['save_interval'] == 0:
                checkpoint_path = checkpoint_dir / f'stage2_iter_{iteration+1}.pth'
                torch.save(self.policy.state_dict(), checkpoint_path)
                print(f"  检查点已保存: {checkpoint_path}")

            # 保存最佳模型
            if metrics['mean_return'] > best_reward:
                best_reward = metrics['mean_return']
                best_checkpoint = checkpoint_dir / 'stage2_best.pth'
                torch.save(self.policy.state_dict(), best_checkpoint)
                print(f"  ✅ 最佳模型已保存: {best_checkpoint}")

        print("\n" + "=" * 80)
        print("🎉 Stage 2 训练完成！")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Guided Exploration Training")
    parser.add_argument('--config', type=str, default='configs/v5_complete.yaml',
                        help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复检查点路径')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备')

    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 80)
    print("Stage 2: 引导探索训练")
    print("=" * 80)
    print(f"\n配置文件: {args.config}")
    print(f"设备: {args.device}")

    if args.resume:
        print(f"恢复检查点: {args.resume}")

    # 创建环境
    print("\n创建环境...")
    env = CompetitionSumoEnv(
        net_file=config['environment']['net_file'],
        route_file=config['environment']['route_file'],
        max_vehicles=config['environment']['icv_config']['max_vehicles']
    )

    # 创建策略
    print("\n创建策略...")
    policy = create_joint_icv_policy(
        obs_dim=config['policy']['obs_dim'],
        node_dim=config['policy']['node_dim'],
        hidden_dim=config['policy']['hidden_dim'],
        num_layers=config['policy']['num_layers'],
        initial_k_ratio=config['policy']['sparse_gate']['initial_k_ratio'],
        device=args.device
    )

    print(f"策略参数量: {sum(p.numel() for p in policy.parameters()):,}")

    # 加载Stage 1权重（如果有）
    if args.resume and config['stage2_guided_exploration']['resume_from_stage1']:
        print(f"\n加载Stage 1 WorldModel权重...")
        # 注意：这里需要从Stage 1检查点加载WorldModel权重
        # 简化：创建一个新的WorldModel并加载权重
        world_model = create_world_model(
            hidden_dim=config['policy']['hidden_dim'],
            latent_dim=config['stage1_world_observer']['world_model']['latent_dim'],
            num_vehicles=config['environment']['icv_config']['max_vehicles'],
            device=args.device
        )

        stage1_checkpoint = torch.load(args.resume, map_location=args.device)
        world_model.load_state_dict(stage1_checkpoint)
        print(f"  ✅ WorldModel权重已加载")

        # 启用WorldModel
        policy.use_world_model = True
        policy.world_model = world_model
        print(f"  ✅ WorldModel已启用")

    # 创建训练器
    trainer = PPOTrainer(policy, env, config, args.device)

    # 训练
    start_time = time.time()
    trainer.train()
    elapsed_time = time.time() - start_time

    print(f"\n训练完成！")
    print(f"  训练时长: {elapsed_time/3600:.2f} 小时")


if __name__ == '__main__':
    main()
