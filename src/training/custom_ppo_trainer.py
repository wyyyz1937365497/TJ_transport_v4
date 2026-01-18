"""
优化的PPO训练实现 - 高效GPU版本

优化点：
- 消除所有不必要的CPU-GPU数据传输
- 向量化所有可能的操作
- 减少同步点（.item()调用）
- 修复学习率调度器初始化bug

参考：
- https://github.com/DLR-RM/stable-baselines3
- Schulman et al. 2017: "Proximal Policy Optimization Algorithms"
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Tuple, Optional, List, Iterable
import numpy as np
import time
from pathlib import Path
import gymnasium as gym
from tqdm import tqdm


class GPURolloutBuffer:
    """
    优化的GPU Rollout Buffer

    数据从收集到存储都在GPU上，消除CPU-GPU传输瓶颈。
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: torch.device,
        n_envs: int = 1,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        self.n_envs = n_envs
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # 获取obs和action的维度
        if isinstance(observation_space, gym.spaces.Box):
            self.obs_dim = observation_space.shape[0]
        else:
            raise ValueError(f"Unsupported observation space: {observation_space}")

        if isinstance(action_space, gym.spaces.Box):
            self.action_dim = action_space.shape[0]
        else:
            raise ValueError(f"Unsupported action space: {action_space}")

        # 在GPU上预分配buffer（使用pinned memory加速传输）
        self.observations = torch.zeros((buffer_size, n_envs, self.obs_dim), device=device)
        self.actions = torch.zeros((buffer_size, n_envs, self.action_dim), device=device)
        self.rewards = torch.zeros((buffer_size, n_envs), device=device)
        self.returns = torch.zeros((buffer_size, n_envs), device=device)
        self.episode_starts = torch.zeros((buffer_size, n_envs), device=device)
        self.values = torch.zeros((buffer_size, n_envs), device=device)
        self.log_probs = torch.zeros((buffer_size, n_envs), device=device)
        self.advantages = torch.zeros((buffer_size, n_envs), device=device)

        self.pos = 0
        self.full = False

    def add(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        episode_starts: torch.Tensor,
        values: torch.Tensor,
        log_probs: torch.Tensor,
    ) -> None:
        """
        添加transition到buffer（输入已经是GPU tensor）

        优化：直接接收GPU tensor，无需转换
        """
        # 存储到GPU buffer（断开计算图）
        self.observations[self.pos] = obs.detach()
        self.actions[self.pos] = actions.detach()
        self.rewards[self.pos] = rewards.detach()
        self.episode_starts[self.pos] = episode_starts.detach()
        self.values[self.pos] = values.detach().flatten()
        self.log_probs[self.pos] = log_probs.detach().flatten()

        self.pos += 1
        if self.pos >= self.buffer_size:
            self.pos = 0
            self.full = True

    def compute_returns_and_advantage(
        self,
        last_values: torch.Tensor,
        last_dones: torch.Tensor,
    ) -> None:
        """
        计算returns和advantages（GAE）- 向量化版本

        完全在GPU上进行，无数据传输
        """
        # 确定buffer的大小
        buffer_size = self.buffer_size if self.full else self.pos

        # 向量化GAE计算
        # 从后往前计算GAE（反向循环是必要的，因为GAE是递归的）
        last_gae = torch.zeros(self.n_envs, device=self.device)
        for step in reversed(range(buffer_size)):
            if step == buffer_size - 1:
                next_values = last_values
                next_non_terminal = 1.0 - last_dones
            else:
                next_values = self.values[step + 1]
                next_non_terminal = 1.0 - self.episode_starts[step + 1]

            # GAE计算（向量化所有envs）
            delta = (
                self.rewards[step]
                + self.gamma * next_values * next_non_terminal
                - self.values[step]
            )
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae

            self.advantages[step] = last_gae
            self.returns[step] = last_gae + self.values[step]

    def get(self, batch_size: int) -> Iterable[Dict[str, torch.Tensor]]:
        """
        生成mini-batches - 优化版本

        数据已在GPU上，直接返回，无需传输
        """
        # 确定buffer大小
        buffer_size = self.buffer_size if self.full else self.pos

        # 展平所有维度并随机打乱
        indices = torch.randperm(buffer_size * self.n_envs, device=self.device)

        # 生成mini-batches（完全向量化）
        for start in range(0, buffer_size * self.n_envs, batch_size):
            end = start + batch_size
            mb_indices = indices[start:end]

            # 向量化索引转换
            step_indices = mb_indices // self.n_envs
            env_indices = mb_indices % self.n_envs

            yield {
                'observations': self.observations[step_indices, env_indices],
                'actions': self.actions[step_indices, env_indices],
                'values': self.values[step_indices, env_indices],
                'log_probs': self.log_probs[step_indices, env_indices],
                'advantages': self.advantages[step_indices, env_indices],
                'returns': self.returns[step_indices, env_indices],
            }


class CustomPPOTrainer:
    """
    优化的PPO训练器

    实现了PPO论文中的所有组件：
    1. Clipped Surrogate Objective
    2. Value Function Loss
    3. Entropy Bonus
    4. GAE (Generalized Advantage Estimation)
    5. Multiple Epochs with Mini-batches

    优化：
    - 消除不必要的CPU-GPU传输
    - 向量化所有可能操作
    - 减少同步点
    """

    def __init__(
        self,
        policy: nn.Module,
        env,
        config: Dict[str, Any],
        device: torch.device,
        checkpoint_dir: Optional[str] = None,
    ):
        self.policy = policy.to(device)
        self.env = env
        self.device = device
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

        # PPO超参数
        self.learning_rate = config.get('learning_rate', 3e-4)
        self.n_steps = config.get('n_steps', 2048)
        self.batch_size = config.get('batch_size', 64)
        self.n_epochs = config.get('n_epochs', 10)
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_range = config.get('clip_range', 0.2)
        self.ent_coef = config.get('ent_coef', 0.01)
        self.vf_coef = config.get('vf_coef', 0.5)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)

        # Rollout buffer（在optimizer之前初始化，用于计算total_updates）
        self.buffer = GPURolloutBuffer(
            buffer_size=self.n_steps,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
            n_envs=env.num_envs if hasattr(env, 'num_envs') else 1,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        # 优化器
        self.optimizer = optim.Adam(
            list(self.policy.parameters()),
            lr=self.learning_rate,
            eps=1e-8,
        )

        # 学习率调度器（修复：在buffer初始化后计算）
        total_updates = 1000  # 默认值，会在learn中重新计算
        self.lr_schedule = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=total_updates
        )
        self.total_updates = total_updates

        # 训练统计
        self.ep_info_buffer = []
        self.n_updates = 0

    def collect_rollouts(self) -> Dict[str, float]:
        """
        收集rollout数据 - 优化版本

        减少CPU-GPU传输，向量化操作
        """
        start_time = time.perf_counter()

        # 重置环境
        obs = self.env.reset()
        episode_starts = torch.ones(self.env.num_envs, device=self.device)

        # 预分配tensor存储
        obs_tensor = torch.empty((self.env.num_envs, self.buffer.obs_dim), device=self.device)

        # 收集rollout
        for step in range(self.n_steps):
            # numpy → GPU tensor（唯一的传输点）
            obs_tensor.copy_(torch.from_numpy(obs).to(self.device))

            with torch.no_grad():
                # Policy forward pass
                actions, values, log_probs = self.policy(obs_tensor)

                # GPU tensor → numpy（环境需要）
                actions_np = actions.cpu().numpy()

            # 环境step
            next_obs, rewards, dones, infos = self.env.step(actions_np)

            # 处理episode信息
            for idx, info in enumerate(infos):
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    self.ep_info_buffer.append(maybe_ep_info)

            # 更新episode_starts（GPU tensor）
            episode_starts = torch.from_numpy(dones).to(self.device).float()

            # 存储到buffer（已经是GPU tensor）
            rewards_tensor = torch.from_numpy(rewards).to(self.device).float()
            self.buffer.add(
                obs=obs_tensor,
                actions=actions,
                rewards=rewards_tensor,
                episode_starts=episode_starts,
                values=values,
                log_probs=log_probs,
            )

            obs = next_obs

        # 最后的value估计（用于GAE）
        with torch.no_grad():
            obs_tensor.copy_(torch.from_numpy(obs).to(self.device))
            _, last_values, _ = self.policy(obs_tensor)
            if last_values.dim() > 1:
                last_values = last_values.flatten()
            last_values = last_values[:self.buffer.n_envs]

        rollout_time = time.perf_counter() - start_time

        return {
            'rollout_time': rollout_time,
            'ep_rew_mean': self._get_episode_statistics('r'),
            'ep_len_mean': self._get_episode_statistics('l'),
        }

    def _get_episode_statistics(self, key: str) -> float:
        """计算episode统计"""
        if len(self.ep_info_buffer) > 0:
            values = [ep_info[key] for ep_info in self.ep_info_buffer]
            return float(np.mean(values))
        return 0.0

    def train(self) -> Dict[str, float]:
        """
        更新策略网络 - 优化版本

        向量化所有可能的操作，减少同步点
        """
        start_time = time.perf_counter()

        # 1. 计算advantages（GAE）
        gae_start = time.perf_counter()
        with torch.no_grad():
            # 使用buffer中最后一个obs计算last_values
            last_obs = self.buffer.observations[-1]
            _, last_values, _ = self.policy(last_obs)
            if last_values.dim() > 1:
                last_values = last_values.flatten()

            # 最后的done标志（全是False，因为episode在rollout中处理）
            last_dones = torch.zeros(self.buffer.n_envs, device=self.device)

            self.buffer.compute_returns_and_advantage(last_values, last_dones)

        gae_time = time.perf_counter() - gae_start

        # 2. 准备advantages（标准化）- 向量化
        advantages = self.buffer.advantages.flatten()
        # 标准化advantages（稳定训练）
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        if adv_std > 1e-8:
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        # 3. 多个epoch的更新
        policy_losses = []
        value_losses = []
        entropy_losses = []

        for epoch in range(self.n_epochs):
            # 生成mini-batches
            for minibatch in self.buffer.get(self.batch_size):
                # 准备数据（已经是GPU tensor）
                observations = minibatch['observations']
                actions = minibatch['actions']
                old_values = minibatch['values'].flatten()
                old_log_probs = minibatch['log_probs'].flatten()
                advantages_mb = minibatch['advantages'].flatten()
                returns_mb = minibatch['returns'].flatten()

                # Forward pass - 重新计算log_probs和values
                values, log_probs, entropy = self.policy.evaluate_actions(observations, actions)

                # 确保形状正确
                values = values.flatten()
                log_probs = log_probs.flatten()

                # 数值稳定性：清理NaN/Inf（向量化）
                log_probs = torch.nan_to_num(log_probs, nan=0.0, posinf=10.0, neginf=-10.0)
                old_log_probs = torch.nan_to_num(old_log_probs, nan=0.0, posinf=10.0, neginf=-10.0)

                # 计算ratio（带数值稳定性保护）
                log_ratio = log_probs - old_log_probs
                log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
                ratio = torch.exp(log_ratio)

                # 计算losses（向量化）
                policy_loss = self._compute_policy_loss(ratio, advantages_mb)
                value_loss = self._compute_value_loss(values, returns_mb, old_values)
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                    + self.ent_coef * entropy_loss
                )

                # 检查loss是否异常（向量化检查）
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"[WARNING] Loss is NaN or Inf! Skipping minibatch.")
                    continue

                # 记录losses（减少.item()调用）
                policy_losses.append(policy_loss.detach())
                value_losses.append(value_loss.detach())
                entropy_losses.append(entropy_loss.detach())

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

                self.optimizer.step()

        # 学习率调度
        self.lr_schedule.step()
        self.n_updates += 1

        total_time = time.perf_counter() - start_time

        # 清空episode info buffer
        self.ep_info_buffer = []

        # 计算平均losses（向量化）
        return {
            'update_time': total_time,
            'gae_time': gae_time,
            'policy_loss': torch.stack(policy_losses).mean().item() if policy_losses else 0.0,
            'value_loss': torch.stack(value_losses).mean().item() if value_losses else 0.0,
            'entropy_loss': torch.stack(entropy_losses).mean().item() if entropy_losses else 0.0,
        }

    def _compute_policy_loss(
        self,
        ratio: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算PPO policy loss（clipped surrogate objective）

        向量化版本
        """
        # 清理异常值（向量化）
        ratio = torch.nan_to_num(ratio, nan=1.0, posinf=10.0, neginf=0.0)
        advantages = torch.nan_to_num(advantages, nan=0.0, posinf=10.0, neginf=-10.0)

        # Clipped surrogate objective（向量化）
        policy_loss = -torch.min(
            ratio * advantages,
            torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
        ).mean()

        return policy_loss

    def _compute_value_loss(
        self,
        values: torch.Tensor,
        returns: torch.Tensor,
        old_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算value function loss（带Clipped Value Loss）

        向量化版本
        """
        # 清理异常值（向量化）
        values = torch.nan_to_num(values, nan=0.0, posinf=10.0, neginf=-10.0)
        returns = torch.nan_to_num(returns, nan=0.0, posinf=10.0, neginf=-10.0)
        old_values = torch.nan_to_num(old_values, nan=0.0, posinf=10.0, neginf=-10.0)

        # Clipped Value Loss（向量化）
        value_pred_clipped = old_values + torch.clamp(
            values - old_values,
            -self.clip_range,
            self.clip_range
        )

        value_loss = torch.max(
            nn.functional.mse_loss(values, returns, reduction='none'),
            nn.functional.mse_loss(value_pred_clipped, returns, reduction='none')
        ).mean()

        return value_loss

    def learn(
        self,
        total_timesteps: int,
        callback: Optional[Any] = None,
    ) -> None:
        """
        主训练循环

        Args:
            total_timesteps: 总训练步数
            callback: 回调函数（可选）
        """
        print("\n" + "=" * 80)
        print("[TRAIN] Optimized PPO Training")
        print("=" * 80)
        print(f"[INFO] Total timesteps: {total_timesteps:,}")
        print(f"[INFO] Device: {self.device}")
        print(f"[INFO] Batch size: {self.batch_size}")
        print(f"[INFO] Epochs per update: {self.n_epochs}")
        print(f"[INFO] Learning rate: {self.learning_rate}")
        print()

        # 重新计算total_updates并更新学习率调度器
        self.total_updates = total_timesteps // (self.n_steps * self.env.num_envs)
        self.lr_schedule = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=self.total_updates
        )

        # 创建进度条
        pbar = tqdm(
            range(self.total_updates),
            desc=f"[TRAIN]",
            unit="update",
            ncols=120,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )

        for update in pbar:
            update_start = time.perf_counter()

            # Rollout
            rollout_metrics = self.collect_rollouts()

            # Train
            train_metrics = self.train()

            # 更新进度条
            steps = (update + 1) * self.n_steps * self.env.num_envs
            total_time = time.perf_counter() - update_start

            pbar.set_postfix({
                'Steps': f'{steps:,}',
                'Reward': f'{rollout_metrics.get("ep_rew_mean", 0):.1f}',
                'Loss': f'{train_metrics["policy_loss"]:.3f}',
                'Time': f'{total_time:.1f}s'
            })

            # 详细打印（每10次update）
            if (update + 1) % 10 == 0:
                tqdm.write(f"\n[Update {update+1}/{self.total_updates}] Steps: {steps:,}/{total_timesteps:,}")
                tqdm.write(f"  [TIMING] Rollout: {rollout_metrics['rollout_time']:.2f}s | Update: {train_metrics['update_time']:.2f}s")
                tqdm.write(f"  [METRICS] Episode Reward: {rollout_metrics.get('ep_rew_mean', 0):.2f}")
                tqdm.write(f"  [LOSS] Policy: {train_metrics['policy_loss']:.4f} | Value: {train_metrics['value_loss']:.4f} | Entropy: {train_metrics['entropy_loss']:.4f}")

            # 保存checkpoint
            if self.checkpoint_dir is not None and (update + 1) % 100 == 0:
                self.save_checkpoint(self.checkpoint_dir / f"update_{update+1}")

        pbar.close()

        print("\n" + "=" * 80)
        print("[DONE] Training completed!")
        print("=" * 80)

    def save_checkpoint(self, path: Path) -> None:
        """保存checkpoint"""
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_schedule.state_dict(),
            'n_updates': self.n_updates,
            'config': self.config,
        }, path)

        print(f"[SAVE] Checkpoint saved: {path}")
