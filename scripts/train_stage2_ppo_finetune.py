#!/usr/bin/env python3
"""
Stage 2: PPO微调训练（Fine-tuning）

功能：
1. 加载Stage 1预训练的SimplifiedICVPolicy
2. 使用PPO算法在真实环境中微调
3. 集成Bottleneck即时奖励

使用方法：
    python scripts/train_stage2_ppo_finetune.py --config configs/ocr_max.yaml --stage1_checkpoint checkpoints/ocr_max/stage1_best.pth

输出：
    - checkpoints/ocr_max/stage2_ppo_best.pth
    - logs/ocr_max/stage2_ppo/
"""

import os
import sys
import argparse
import yaml
import time
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.simplified_icv_policy import SimplifiedICVPolicy
from src.env.competition_env import CompetitionSumoEnv
from src.training.ocr_rewards import OCRRewardCalculator


def flatten_observation(obs_dict: dict, max_vehicles: int = 32) -> np.ndarray:
    """
    将观测字典转换为扁平化张量

    Args:
        obs_dict: 环境返回的观测字典
        max_vehicles: 最大车辆数（用于padding）

    Returns:
        flattened_obs: [obs_dim] 扁平化观测
    """
    vehicle_states = obs_dict.get('vehicle_states', {})
    vehicle_ids = obs_dict.get('vehicle_ids', [])
    icv_ids = obs_dict.get('icv_ids', set())
    global_stats = obs_dict.get('global_stats', np.zeros(32))

    num_vehicles = len(vehicle_ids)

    # 提取9维车辆特征（归一化）
    vehicle_features = np.zeros((max_vehicles, 9), dtype=np.float32)

    for i, veh_id in enumerate(vehicle_ids[:max_vehicles]):
        if veh_id in vehicle_states:
            state = vehicle_states[veh_id]
            vehicle_features[i, 0] = state.get('s', 0.0) / 1000.0
            vehicle_features[i, 1] = state.get('d', 0.0) / 10.0
            vehicle_features[i, 2] = state.get('vs', 0.0) / 30.0
            vehicle_features[i, 3] = state.get('vd', 0.0) / 10.0
            vehicle_features[i, 4] = state.get('speed', 0.0) / 30.0
            vehicle_features[i, 5] = state.get('acceleration', 0.0) / 3.0
            vehicle_features[i, 6] = state.get('lane_index', 0.0) / 10.0
            vehicle_features[i, 7] = state.get('angle', 0.0) / 360.0
            vehicle_features[i, 8] = 1.0 if veh_id in icv_ids else 0.0

    # 确保global_stats是32维
    global_stats_flat = global_stats.flatten()
    if len(global_stats_flat) < 32:
        global_stats_flat = np.concatenate([
            global_stats_flat,
            np.zeros(32 - len(global_stats_flat), dtype=np.float32)
        ])
    elif len(global_stats_flat) > 32:
        global_stats_flat = global_stats_flat[:32]

    # 扁平化并拼接
    vehicle_features_flat = vehicle_features.flatten()
    num_vehicles_array = np.array([num_vehicles], dtype=np.float32)

    flattened_obs = np.concatenate([
        vehicle_features_flat,
        global_stats_flat,
        num_vehicles_array
    ])

    return flattened_obs


def convert_actions_to_dict(action_array: np.ndarray, vehicle_ids: list, icv_ids: set) -> dict:
    """
    将扁平动作数组转换为字典格式（只控制ICV）

    Args:
        action_array: [max_vehicles * 2] 扁平动作数组
        vehicle_ids: 车辆ID列表
        icv_ids: ICV ID集合

    Returns:
        actions_dict: {vehicle_id: [acceleration, lane_change]}
    """
    max_vehicles = action_array.shape[0] // 2
    actions_reshaped = action_array.reshape(max_vehicles, 2)

    actions_dict = {}
    for i, veh_id in enumerate(vehicle_ids[:max_vehicles]):
        if veh_id in icv_ids and i < max_vehicles:
            actions_dict[veh_id] = actions_reshaped[i]

    return actions_dict


class PPOFinetuneTrainer:
    """
    PPO微调训练器（Stage 2）
    """

    def __init__(
        self,
        policy,
        env,
        reward_calculator,
        config: dict,
        device: str = 'cuda',
        log_dir: str = 'logs/ocr_max/stage2_ppo',
        checkpoint_dir: str = 'checkpoints/ocr_max'
    ):
        """
        Args:
            policy: SimplifiedICVPolicy (预训练)
            env: CompetitionSumoEnv
            reward_calculator: OCRRewardCalculator
            config: 配置字典
            device: 设备
            log_dir: 日志目录
            checkpoint_dir: 检查点目录
        """
        self.policy = policy.to(device)
        self.env = env
        self.reward_calculator = reward_calculator
        self.config = config
        self.device = device
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir

        # 创建目录
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # PPO配置
        ppo_config = config.get('stage2_ppo_finetune', {}).get('ppo', {})

        self.num_iterations = ppo_config.get('num_iterations', 100)
        self.num_episodes_per_iter = ppo_config.get('num_episodes_per_iter', 10)
        self.max_steps_per_episode = ppo_config.get('max_steps_per_episode', 3600)

        self.gamma = ppo_config.get('gamma', 0.99)
        self.gae_lambda = ppo_config.get('gae_lambda', 0.95)
        self.clip_epsilon = ppo_config.get('clip_epsilon', 0.2)
        self.entropy_coef = ppo_config.get('entropy_coef', 0.01)
        self.value_coef = ppo_config.get('value_coef', 0.5)

        self.num_minibatches = ppo_config.get('num_minibatches', 4)
        self.ppo_epochs = ppo_config.get('ppo_epochs', 10)
        self.batch_size = ppo_config.get('batch_size', 64)

        # 优化器（更小的学习率用于微调）
        lr = ppo_config.get('lr', 1e-4)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.num_iterations,
            eta_min=lr * 0.1
        )

        # TensorBoard
        self.writer = SummaryWriter(log_dir)

        # 训练统计
        self.best_mean_return = -float('inf')
        self.global_step = 0

    def collect_rollouts(self, num_episodes: int) -> dict:
        """
        收集rollouts

        Args:
            num_episodes: Episode数量

        Returns:
            rollouts: {
                'observations': [T, B, obs_dim],
                'actions': [T, B, N*2],
                'rewards': [T, B],
                'values': [T, B],
                'log_probs': [T, B],
                'dones': [T, B]
            }
        """
        self.policy.eval_mode()  # 推理模式，不使用SafetyShield

        all_obs = []
        all_actions = []
        all_rewards = []
        all_values = []
        all_log_probs = []
        all_dones = []

        episode_returns = []

        for episode_idx in tqdm(range(num_episodes), desc="Collecting rollouts"):
            obs_dict = self.env.reset()
            episode_reward = 0.0
            episode_obs = []
            episode_actions = []
            episode_rewards = []
            episode_values = []
            episode_log_probs = []

            for step in range(self.max_steps_per_episode):
                # 解析观测
                obs_flat = flatten_observation(obs_dict)
                obs_tensor = torch.from_numpy(obs_flat).unsqueeze(0).float().to(self.device)

                # 策略前向传播
                with torch.no_grad():
                    outputs = self.policy(obs_tensor, deterministic=False)
                    action = outputs['actions'][0].cpu().numpy()  # [N*2]
                    value = outputs['value'][0, 0].item()
                    log_prob = outputs['log_prob'][0].item()

                # 转换为动作字典
                vehicle_ids = obs_dict.get('vehicle_ids', [])
                icv_ids = obs_dict.get('icv_ids', set())
                actions_dict = convert_actions_to_dict(action, vehicle_ids, icv_ids)

                # 执行动作（环境内部计算奖励）
                next_obs_dict, reward, done, info = self.env.step(actions_dict)

                # 存储转换
                episode_obs.append(obs_flat)
                episode_actions.append(action)
                episode_rewards.append(reward)
                episode_values.append(value)
                episode_log_probs.append(log_prob)

                # 更新观测
                obs_dict = next_obs_dict

                episode_reward += reward

                if done:
                    break

            episode_returns.append(episode_reward)
            all_obs.extend(episode_obs)
            all_actions.extend(episode_actions)
            all_rewards.extend(episode_rewards)
            all_values.extend(episode_values)
            all_log_probs.extend(episode_log_probs)
            all_dones.extend([1.0] * len(episode_rewards))
            all_dones[-1] = 1.0  # 最后一个done设为1

        # 转换为张量
        rollouts = {
            'observations': torch.from_numpy(np.array(all_obs)).float().to(self.device),
            'actions': torch.from_numpy(np.array(all_actions)).float().to(self.device),
            'rewards': torch.from_numpy(np.array(all_rewards)).float().to(self.device),
            'values': torch.from_numpy(np.array(all_values)).float().to(self.device),
            'log_probs': torch.from_numpy(np.array(all_log_probs)).float().to(self.device),
            'dones': torch.from_numpy(np.array(all_dones)).float().to(self.device)
        }

        print(f"收集完成: {len(all_obs)}步, 平均回报: {np.mean(episode_returns):.2f}")

        return rollouts

    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor,
                    dones: torch.Tensor) -> tuple:
        """
        计算Generalized Advantage Estimation

        Args:
            rewards: [T] 奖励
            values: [T] 价值估计
            dones: [T] 终止标志

        Returns:
            advantages: [T] 优势
            returns: [T] 回报
        """
        T = len(rewards)
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        last_advantage = 0.0
        last_return = values[-1].item()

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0.0
                next_non_terminal = 1.0 - dones[t]
            else:
                next_value = values[t + 1].item()
                next_non_terminal = 1.0 - dones[t + 1]

            delta = rewards[t].item() + self.gamma * next_value * next_non_terminal - values[t].item()
            advantages[t] = last_advantage = delta + self.gamma * self.gae_lambda * next_non_terminal * last_advantage
            returns[t] = last_return = rewards[t].item() + self.gamma * next_non_terminal * last_return

        return advantages, returns

    def update_policy(self, rollouts: dict) -> dict:
        """
        更新策略

        Args:
            rollouts: 收集的rollouts

        Returns:
            metrics: 训练指标
        """
        self.policy.train_mode()

        observations = rollouts['observations']
        actions = rollouts['actions']
        old_log_probs = rollouts['log_probs']
        values = rollouts['values']
        rewards = rollouts['rewards']
        dones = rollouts['dones']

        # 计算GAE
        advantages, returns = self.compute_gae(rewards, values, dones)

        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 训练多个epoch
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        num_updates = 0

        batch_size = self.batch_size
        num_samples = len(observations)

        for epoch in range(self.ppo_epochs):
            # 随机打乱
            indices = torch.randperm(num_samples)

            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                batch_indices = indices[start:end]

                # 准备批次数据
                batch_obs = observations[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # 前向传播
                outputs = self.policy(batch_obs, deterministic=False)
                new_log_probs = outputs['log_prob']
                new_values = outputs['value'].squeeze(-1)
                entropy = outputs['entropy']

                # 计算ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # PPO损失
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # 价值损失
                value_loss = nn.MSELoss()(new_values, batch_returns)

                # 熵奖励
                entropy_loss = -entropy.mean()

                # 总损失
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)

                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        avg_policy_loss = total_policy_loss / num_updates
        avg_value_loss = total_value_loss / num_updates
        avg_entropy = total_entropy / num_updates

        metrics = {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy': avg_entropy
        }

        return metrics

    def train(self):
        """
        完整训练流程
        """
        print(f"\n{'='*80}")
        print("Stage 2: PPO微调训练")
        print(f"{'='*80}\n")

        print(f"配置:")
        print(f"  Iterations: {self.num_iterations}")
        print(f"  Episodes per iteration: {self.num_episodes_per_iter}")
        print(f"  Max steps per episode: {self.max_steps_per_episode}")
        print(f"  Learning rate: {self.optimizer.param_groups[0]['lr']}")
        print(f"  Device: {self.device}\n")

        for iteration in range(self.num_iterations):
            print(f"\n--- Iteration {iteration + 1}/{self.num_iterations} ---")

            # 收集rollouts
            rollouts = self.collect_rollouts(self.num_episodes_per_iter)

            # 更新策略
            metrics = self.update_policy(rollouts)

            # 计算统计
            mean_return = rollouts['rewards'].sum().item() / self.num_episodes_per_iter

            print(f"\n训练指标:")
            print(f"  Policy Loss: {metrics['policy_loss']:.4f}")
            print(f"  Value Loss: {metrics['value_loss']:.4f}")
            print(f"  Entropy: {metrics['entropy']:.4f}")
            print(f"  Mean Return: {mean_return:.2f}")

            # TensorBoard
            self.writer.add_scalar('train/policy_loss', metrics['policy_loss'], iteration)
            self.writer.add_scalar('train/value_loss', metrics['value_loss'], iteration)
            self.writer.add_scalar('train/entropy', metrics['entropy'], iteration)
            self.writer.add_scalar('train/mean_return', mean_return, iteration)
            self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], iteration)

            # 学习率调度
            self.scheduler.step()

            # 保存检查点
            if mean_return > self.best_mean_return:
                self.best_mean_return = mean_return
                self.save_checkpoint(iteration, metrics, mean_return, 'stage2_best.pth')
                print(f"  ✅ 新的最佳模型: mean_return={mean_return:.2f}")

            # 定期保存
            if (iteration + 1) % 20 == 0:
                self.save_checkpoint(iteration, metrics, mean_return)

        print(f"\n{'='*80}")
        print("训练完成！")
        print(f"  最佳平均回报: {self.best_mean_return:.2f}")
        print(f"{'='*80}\n")

        self.writer.close()

    def save_checkpoint(self, iteration: int, metrics: dict, mean_return: float,
                       filename: str = None):
        """
        保存检查点

        Args:
            iteration: 当前迭代
            metrics: 训练指标
            mean_return: 平均回报
            filename: 文件名
        """
        if filename is None:
            filename = f'stage2_ppo_iter{iteration}.pth'

        checkpoint = {
            'iteration': iteration,
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'mean_return': mean_return,
            'best_mean_return': self.best_mean_return,
            'config': self.config
        }

        save_path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, save_path)

        print(f"  检查点已保存: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Stage 2: PPO微调训练')
    parser.add_argument('--config', type=str, default='configs/ocr_max.yaml',
                        help='配置文件路径')
    parser.add_argument('--stage1_checkpoint', type=str, required=True,
                        help='Stage 1检查点路径')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda/cpu)')

    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 创建策略网络
    print("创建策略网络...")
    policy_config = config.get('policy', {})

    policy = SimplifiedICVPolicy(
        obs_dim=policy_config.get('obs_dim', 321),
        node_dim=policy_config.get('node_dim', 9),
        hidden_dim=policy_config.get('hidden_dim', 128),
        num_layers=policy_config.get('num_layers', 3),
        num_vehicles=policy_config.get('num_vehicles', 32),
        device=args.device,
        use_safety_shield=False  # 训练时不使用SafetyShield
    )

    # 加载Stage 1权重
    print(f"加载Stage 1权重: {args.stage1_checkpoint}")
    checkpoint = torch.load(args.stage1_checkpoint, map_location=args.device)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    print(f"  Stage 1 loss: {checkpoint['best_loss']:.4f}")

    # 创建环境
    print("\n创建环境...")
    env_config = config.get('environment', {})

    # 设置默认值
    if 'sumocfg_file' not in env_config:
        env_config['sumocfg_file'] = '仿真环境_初赛_1.0/仿真环境-初赛/sumo_train.sumocfg'
    if 'max_steps' not in env_config:
        env_config['max_steps'] = 3600
    if 'icv_ratio' not in env_config:
        env_config['icv_ratio'] = 0.10

    # CompetitionSumoEnv接受完整的config字典
    env = CompetitionSumoEnv(
        config=env_config,
        use_gui=False,
        device=args.device
    )

    # 创建奖励计算器
    print("创建奖励计算器...")
    reward_config = config.get('stage2_ppo_finetune', {}).get('bottleneck_rewards', {})

    reward_calculator = OCRRewardCalculator(
        num_icv_total=int(600 * 0.10),
        use_improved_reward=True,
        use_bottleneck_rewards=reward_config.get('enabled', True),
        bottleneck_reward_config={
            'bottleneck_s_min': reward_config.get('bottleneck_s_min', 1200.0),
            'bottleneck_s_max': reward_config.get('bottleneck_s_max', 2200.0),
            'w_throughput': reward_config.get('w_throughput', 0.5),
            'w_queue': reward_config.get('w_queue', 0.3),
            'w_conflict': reward_config.get('w_conflict', 0.2),
            'ttc_threshold': reward_config.get('ttc_threshold', 3.0),
            'min_speed': reward_config.get('min_speed', 5.0)
        }
    )

    # 创建训练器
    log_dir = config.get('logging', {}).get('log_dir', 'logs/ocr_max') + '/stage2_ppo'
    checkpoint_dir = config.get('checkpoints', {}).get('dir', 'checkpoints/ocr_max')

    trainer = PPOFinetuneTrainer(
        policy=policy,
        env=env,
        reward_calculator=reward_calculator,
        config=config,
        device=args.device,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir
    )

    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()
