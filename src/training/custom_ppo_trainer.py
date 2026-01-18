"""
完整的PPO训练实现 - GPU优化版本

基于Stable-Baselines3的PPO实现，但优化了数据流：
- RolloutBuffer在GPU上
- 消除更新阶段的CPU-GPU传输

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
    GPU上的Rollout Buffer

    数据从收集到存储都在GPU上，消除CPU-GPU传输瓶颈。
    完全实现，无简化逻辑。
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

        # 在GPU上预分配buffer
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
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        episode_starts: np.ndarray,
        values: torch.Tensor,
        log_probs: torch.Tensor,
    ) -> None:
        """
        添加transition到buffer

        关键：numpy array → GPU tensor转换在这里一次性完成
        """
        # 转换为GPU tensor（唯一的CPU→GPU传输点）
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        actions_tensor = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        episode_starts_tensor = torch.as_tensor(episode_starts, dtype=torch.float32, device=self.device)

        # 存储到GPU buffer
        # 关键：使用detach()断开计算图，避免在update时重复backward
        self.observations[self.pos] = obs_tensor
        self.actions[self.pos] = actions_tensor
        self.rewards[self.pos] = rewards_tensor
        self.episode_starts[self.pos] = episode_starts_tensor
        self.values[self.pos] = values.detach().flatten()      # ✅ 断开计算图
        self.log_probs[self.pos] = log_probs.detach().flatten()  # ✅ 断开计算图

        self.pos += 1
        if self.pos >= self.buffer_size:
            self.pos = 0
            self.full = True

    def compute_returns_and_advantage(
        self,
        last_values: torch.Tensor,
        last_dones: np.ndarray,
    ) -> None:
        """
        计算returns和advantages（GAE）

        完全在GPU上进行，无数据传输

        Args:
            last_values: 最后状态的value估计 [n_envs]
            last_dones: 最后状态的done标志 [n_envs]
        """
        # 转换last_dones为GPU tensor
        last_dones_tensor = torch.as_tensor(last_dones, dtype=torch.float32, device=self.device)

        # 确定buffer的大小
        buffer_size = self.buffer_size if self.full else self.pos

        # 从后往前计算GAE
        last_gae = 0
        for step in reversed(range(buffer_size)):
            if step == buffer_size - 1:
                next_values = last_values
                next_non_terminal = 1.0 - last_dones_tensor
            else:
                next_values = self.values[step + 1]
                next_non_terminal = 1.0 - self.episode_starts[step + 1]

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
        生成mini-batches

        数据已在GPU上，直接返回，无需传输

        Args:
            batch_size: mini-batch大小

        Yields:
            minibatch字典，包含所有需要的tensors
        """
        # 确定buffer大小
        buffer_size = self.buffer_size if self.full else self.pos

        # 展平所有维度
        indices = torch.randperm(buffer_size * self.n_envs, device=self.device)

        # 生成mini-batches
        for start in range(0, buffer_size * self.n_envs, batch_size):
            end = start + batch_size
            mb_indices = indices[start:end]

            # 将flat indices转为 (step, env) 对
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
    完整的PPO训练器 - GPU优化版本

    实现了PPO论文中的所有组件：
1. Clipped Surrogate Objective
2. Value Function Loss
3. Entropy Bonus
4. GAE (Generalized Advantage Estimation)
5. Multiple Epochs with Mini-batches

优化：
- 所有数据在GPU上
- 详细的性能分析
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

        # 优化器（添加eps提高数值稳定性）
        self.optimizer = optim.Adam(
            list(self.policy.parameters()),
            lr=self.learning_rate,
            eps=1e-8,  # 提高数值稳定性
        )

        # ✅ 学习率调度器（线性衰减，提升收敛速度）
        total_updates = (total_timesteps // (self.n_steps * self.buffer.n_envs)) if hasattr(self, 'buffer') else 1000
        self.lr_schedule = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.1,  # 衰减到10%
            total_iters=total_updates
        )
        self.current_lr = self.learning_rate

        # Rollout buffer
        self.buffer = GPURolloutBuffer(
            buffer_size=self.n_steps,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
            n_envs=env.num_envs if hasattr(env, 'num_envs') else 1,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        # 性能分析
        self.timings = {
            'rollout': [],
            'compute_gae': [],
            'update_epoch': [],
            'forward': [],
            'backward': [],
            'data_transfer': [],
            'value_pred': [],  # value function prediction
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
        }

        # 训练统计
        self.ep_info_buffer = []  # episode info buffer
        self.n_updates = 0

    def collect_rollouts(self) -> Dict[str, float]:
        """
        收集rollout数据

        完整实现，包含：
- 环境交互
- Value function prediction
- Action sampling
- Data collection
        """
        start_time = time.perf_counter()

        # 重置环境
        obs = self.env.reset()
        episode_starts = np.ones(self.env.num_envs, dtype=bool)

        # 收集rollout
        for step in range(self.n_steps):
            transfer_start = time.perf_counter()

            # obs: numpy array → GPU tensor
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

                # Policy forward pass
                # 注意：policy.forward返回 (actions, values, log_prob)
                actions, values, log_probs = self.policy(obs_tensor)

                # actions: GPU tensor → numpy array (环境需要)
                actions_np = actions.cpu().numpy()

            transfer_time = time.perf_counter() - transfer_start
            self.timings['data_transfer'].append(transfer_time)

            # 环境step
            next_obs, rewards, dones, infos = self.env.step(actions_np)

            # 处理episode信息
            for idx, info in enumerate(infos):
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    self.ep_info_buffer.append(maybe_ep_info)

            # 更新episode_starts
            episode_starts = dones

            # 存储到buffer
            self.buffer.add(
                obs=obs,
                actions=actions_np,
                rewards=rewards,
                episode_starts=episode_starts,
                values=values,
                log_probs=log_probs,
            )

            obs = next_obs

            # 每256步打印一次进度
            if (step + 1) % 256 == 0:
                elapsed = time.perf_counter() - start_time
                progress = (step + 1) / self.n_steps * 100
                print(f"[ROLLOUT] {step+1}/{self.n_steps} steps ({progress:.1f}%) | Elapsed: {elapsed:.1f}s")

        # 最后的value估计（用于GAE）
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            # 调用policy的evaluate_actions或forward方法获取last_values
            # policy.forward返回 (actions, values, log_probs)
            _, last_values, _ = self.policy(obs_tensor)

            # 确保last_values是正确的形状
            if last_values.dim() > 1:
                last_values = last_values.flatten()
            last_values = last_values[:self.buffer.n_envs]

        rollout_time = time.perf_counter() - start_time
        self.timings['rollout'].append(rollout_time)

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
        更新策略网络

        完整的PPO更新循环：
1. Compute advantages (GAE)
2. For each epoch:
   For each minibatch:
     - Forward pass (compute new log_probs, values, entropy)
     - Compute policy loss (clipped surrogate)
     - Compute value loss
     - Compute entropy loss
     - Backward pass
     - Clip gradients
     - Optimizer step

        Returns:
            训练统计字典
        """
        start_time = time.perf_counter()

        # 1. 计算advantages
        gae_start = time.perf_counter()
        with torch.no_grad():
            # 获取最后的values和dones
            obs_tensor = torch.as_tensor(self.buffer.observations[-1], dtype=torch.float32, device=self.device)
            # policy.forward返回 (actions, values, log_probs)
            _, last_values, _ = self.policy(obs_tensor)

            if last_values.dim() > 1:
                last_values = last_values.flatten()

            last_dones = np.zeros(self.buffer.n_envs, dtype=bool)

            self.buffer.compute_returns_and_advantage(last_values, last_dones)

        gae_time = time.perf_counter() - gae_start
        self.timings['compute_gae'].append(gae_time)

        # 2. 准备advantages（标准化）
        advantages = self.buffer.advantages.flatten()
        # 标准化advantages（稳定训练）
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 3. 多个epoch的更新
        policy_losses = []
        value_losses = []
        entropy_losses = []

        total_minibatches = (self.buffer.buffer_size * self.buffer.n_envs) // self.batch_size * self.n_epochs
        current_minibatch = 0

        for epoch in range(self.n_epochs):
            epoch_start = time.perf_counter()

            # 生成mini-batches
            minibatches = self.buffer.get(self.batch_size)

            for minibatch in minibatches:
                current_minibatch += 1

                # 每32个minibatch打印一次进度
                if current_minibatch % 32 == 0:
                    progress = current_minibatch / total_minibatches * 100
                    print(f"[UPDATE] Epoch {epoch+1}/{self.n_epochs} | Minibatch {current_minibatch}/{total_minibatches} ({progress:.1f}%)")
                forward_start = time.perf_counter()

                # 准备数据
                observations = minibatch['observations']
                actions = minibatch['actions']
                old_values = minibatch['values'].flatten()
                old_log_probs = minibatch['log_probs'].flatten()
                advantages_mb = minibatch['advantages'].flatten()
                returns_mb = minibatch['returns'].flatten()

                # Forward pass - 重新计算log_probs和values
                # 注意：这里需要调用policy的evaluate_actions方法
                values, log_probs, entropy = self.policy.evaluate_actions(observations, actions)

                # 确保形状正确
                values = values.flatten()
                log_probs = log_probs.flatten()

                # 数值稳定性：清理NaN/Inf
                log_probs = torch.nan_to_num(log_probs, nan=0.0, posinf=10.0, neginf=-10.0)
                old_log_probs = torch.nan_to_num(old_log_probs, nan=0.0, posinf=10.0, neginf=-10.0)

                forward_time = time.perf_counter() - forward_start
                self.timings['forward'].append(forward_time)

                # 计算ratio（带数值稳定性保护）
                log_ratio = log_probs - old_log_probs
                # 限制log_ratio范围，防止exp爆炸
                log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
                ratio = torch.exp(log_ratio)

                # PPO clip loss
                policy_loss = self._compute_policy_loss(ratio, advantages_mb)
                value_loss = self._compute_value_loss(values, returns_mb, old_values)
                entropy_loss = -entropy.mean()

                # Total loss（带数值稳定性检查）
                loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                    + self.ent_coef * entropy_loss
                )

                # 检查loss是否为NaN或Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"[WARNING] Loss is NaN or Inf! Skipping this minibatch update.")
                    print(f"  policy_loss: {policy_loss.item()}, value_loss: {value_loss.item()}, entropy_loss: {entropy_loss.item()}")
                    continue  # 跳过这个minibatch

                # 记录损失
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())

                # Backward pass
                backward_start = time.perf_counter()

                self.optimizer.zero_grad()
                loss.backward()

                # 检查梯度是否异常
                total_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                if torch.isnan(total_norm) or torch.isinf(total_norm):
                    print(f"[WARNING] Gradient norm is NaN or Inf! Skipping optimizer step.")
                    print(f"  Total norm: {total_norm.item()}")
                    continue  # 跳过优化器步骤

                self.optimizer.step()

                backward_time = time.perf_counter() - backward_start
                self.timings['backward'].append(backward_time)

            epoch_time = time.perf_counter() - epoch_start
            self.timings['update_epoch'].append(epoch_time)

        # 学习率调度
        if self.lr_schedule is not None:
            self.lr_schedule.step()

        self.n_updates += 1

        total_time = time.perf_counter() - start_time

        # 清空episode info buffer
        self.ep_info_buffer = []

        return {
            'update_time': total_time,
            'gae_time': gae_time,
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy_loss': np.mean(entropy_losses),
        }

    def _compute_policy_loss(
        self,
        ratio: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算PPO policy loss（clipped surrogate objective）

        L^CLIP = E[min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)]

        数值稳定性：清理NaN/Inf
        """
        # 清理ratio中的异常值
        ratio = torch.nan_to_num(ratio, nan=1.0, posinf=10.0, neginf=0.0)
        # 清理advantages中的异常值
        advantages = torch.nan_to_num(advantages, nan=0.0, posinf=10.0, neginf=-10.0)

        # Clipped surrogate objective
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

        数值稳定性：清理NaN/Inf
        使用Clipped Value Loss提升稳定性（PPO论文建议）
        """
        # 清理values和returns中的异常值
        values = torch.nan_to_num(values, nan=0.0, posinf=10.0, neginf=-10.0)
        returns = torch.nan_to_num(returns, nan=0.0, posinf=10.0, neginf=-10.0)
        old_values = torch.nan_to_num(old_values, nan=0.0, posinf=10.0, neginf=-10.0)

        # 标准MSE loss
        value_loss = nn.functional.mse_loss(values, returns)

        # ✅ 使用Clipped Value Loss（更稳定，PPO论文建议）
        value_pred_clipped = old_values + torch.clamp(
            values - old_values,
            -self.clip_range,
            self.clip_range
        )
        value_loss_clipped = nn.functional.mse_loss(value_pred_clipped, returns)
        value_loss = torch.max(value_loss, value_loss_clipped).mean()

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
        print("[TRAIN] Custom PPO Training (GPU Optimized)")
        print("=" * 80)
        print(f"[INFO] Total timesteps: {total_timesteps:,}")
        print(f"[INFO] Device: {self.device}")
        print(f"[INFO] Batch size: {self.batch_size}")
        print(f"[INFO] Epochs per update: {self.n_epochs}")
        print(f"[INFO] Learning rate: {self.learning_rate}")
        print()

        # 计算update次数
        n_updates = total_timesteps // (self.n_steps * self.env.num_envs)

        # 创建总进度条
        pbar = tqdm(
            range(n_updates),
            desc=f"[TRAIN] Phase 2",
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

            # 更新进度条显示的指标
            steps = (update + 1) * self.n_steps * self.env.num_envs
            total_time = time.perf_counter() - update_start

            pbar.set_postfix({
                'Steps': f'{steps:,}',
                'Reward': f'{rollout_metrics.get("ep_rew_mean", 0):.1f}',
                'Len': f'{rollout_metrics.get("ep_len_mean", 0):.0f}',
                'Loss': f'{train_metrics["policy_loss"]:.3f}',
                'Time': f'{total_time:.1f}s'
            })

            # 详细打印（每10次update）
            if (update + 1) % 10 == 0:
                tqdm.write(f"\n[Update {update+1}/{n_updates}] Steps: {steps:,}/{total_timesteps:,}")
                tqdm.write(f"  [TIMING] Rollout: {rollout_metrics['rollout_time']:.2f}s | Update: {train_metrics['update_time']:.2f}s | Total: {total_time:.2f}s")
                tqdm.write(f"  [METRICS] Episode Reward: {rollout_metrics.get('ep_rew_mean', 0):.2f} | Length: {rollout_metrics.get('ep_len_mean', 0):.2f}")
                tqdm.write(f"  [LOSS] Policy: {train_metrics['policy_loss']:.4f} | Value: {train_metrics['value_loss']:.4f} | Entropy: {train_metrics['entropy_loss']:.4f}")

            # 保存checkpoint
            if self.checkpoint_dir is not None and (update + 1) % 100 == 0:
                self.save_checkpoint(self.checkpoint_dir / f"update_{update+1}")

            # 性能分析
            if (update + 1) % 50 == 0:
                self.print_performance_summary()

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
            'n_updates': self.n_updates,
            'config': self.config,
        }, path)

        print(f"[SAVE] Checkpoint saved: {path}")

    def print_performance_summary(self) -> None:
        """打印详细的性能分析"""
        print("\n" + "=" * 80)
        print("[PERF] Performance Analysis")
        print("=" * 80)

        total_time = sum([sum(times) for times in self.timings.values() if times])

        for name, times in self.timings.items():
            if not times:
                continue

            avg_time = np.mean(times)
            total = np.sum(times)
            pct = (total / total_time) * 100 if total_time > 0 else 0

            print(f"\n{name}:")
            print(f"  - Calls: {len(times)}")
            print(f"  - Avg: {avg_time*1000:.2f} ms")
            print(f"  - Total: {total:.2f} s ({pct:.1f}%)")

        print("=" * 80 + "\n")
