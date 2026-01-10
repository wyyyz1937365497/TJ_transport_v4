"""
PPO训练模块 - 完整实现
阶段2: 使用PPO算法训练控制器
"""

import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from ..models import TrafficController
from ..env import SumoEnvironment
from ..utils import Config, setup_logger, get_logger
from ..utils.dataclass import Observation, Action, StepResult

logger = get_logger()


class RolloutBuffer:
    """
    PPO经验回放缓冲区

    存储一个完整rollout的轨迹数据
    """

    def __init__(
        self,
        buffer_size: int = 2048,
        gae_lambda: float = 0.95,
        gamma: float = 0.99,
    ):
        self.buffer_size = buffer_size
        self.gae_lambda = gae_lambda
        self.gamma = gamma

        # 缓冲区
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.advantages = []
        self.returns = []

        # 元数据
        self.episode_rewards = []
        self.episode_lengths = []

    def clear(self):
        """清空缓冲区"""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.advantages = []
        self.returns = []
        self.episode_rewards = []
        self.episode_lengths = []

    def add(
        self,
        state: Dict[str, Any],
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: float,
        value: torch.Tensor,
        done: bool,
    ):
        """添加一个步骤"""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def finish_episode(self, episode_reward: float, episode_length: int):
        """完成一个episode"""
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)

    def compute_gae(self, last_value: torch.Tensor = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算广义优势估计(GAE)

        Args:
            last_value: 最后一个状态的值估计（用于bootstrapping）

        Returns:
            (advantages, returns)
        """
        # 转换为numpy
        rewards = np.array(self.rewards, dtype=np.float32)
        values = np.array(self.values + [last_value], dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)

        # 计算returns
        returns = np.zeros_like(rewards)
        last_return = last_value

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]

            returns[t] = rewards[t] + self.gamma * next_non_terminal * next_value

        # 计算GAE优势
        advantages = np.zeros_like(rewards)
        last_gae = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]

            # TD误差
            delta = rewards[t] + self.gamma * next_non_terminal * next_value - values[t]

            # GAE
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae

        # 转换为tensor
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)

        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def get_batches(self, batch_size: int) -> List[Dict]:
        """获取mini-batches"""
        indices = np.random.permutation(len(self.states))

        batches = []
        for start in range(0, len(indices), batch_size):
            end = min(start + batch_size, len(indices))
            batch_indices = indices[start:end]

            batch = {
                "states": [self.states[i] for i in batch_indices],
                "actions": torch.stack([self.actions[i] for i in batch_indices]),
                "old_log_probs": torch.stack([self.log_probs[i] for i in batch_indices]),
                "advantages": self.advantages[batch_indices],
                "returns": self.returns[batch_indices],
            }

            batches.append(batch)

        return batches

    def __len__(self) -> int:
        return len(self.states)


class PPOPolicy(nn.Module):
    """
    PPO策略网络

    包含actor和critic
    """

    def __init__(
        self,
        controller: nn.Module,
        action_dim: int = 2,
        action_scale: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.controller = controller
        self.action_dim = action_dim

        # 动作缩放（将[-1,1]映射到实际动作范围）
        if action_scale is None:
            # [acceleration, lane_change]
            # acceleration: [-3, 2] -> scale by [2.5, 0.5]
            # lane_change: [0, 1]
            action_scale = torch.tensor([2.5, 1.0])
        self.action_scale = action_scale

        # 动作偏移
        self.action_bias = torch.tensor([-0.5, 0.0])

    def forward(self, state: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            state: 环境状态

        Returns:
            (actions, log_probs, values)
        """
        # 通过控制器获取动作
        output = self.controller(state)

        # 如果没有选中的车辆，返回默认动作
        if not output["selected_vehicle_ids"]:
            batch_size = 1
            actions = torch.zeros(batch_size, self.action_dim, device=next(self.parameters()).device)
            log_probs = torch.zeros(batch_size, device=next(self.parameters()).device)
            values = torch.zeros(batch_size, device=next(self.parameters()).device)
            return actions, log_probs, values

        raw_actions = output["raw_actions"]  # [N, 2]
        value_estimates = output.get("value_estimates")

        # 缩放动作
        actions = raw_actions * self.action_scale.to(raw_actions.device) + self.action_bias.to(raw_actions.device)

        # 计算log概率（使用高斯分布作为策略分布）
        # 对于连续动作空间，我们假设策略是高斯分布
        # 均值由网络给出，标准差作为可学习参数
        if not hasattr(self, "log_std"):
            self.log_std = nn.Parameter(torch.zeros(self.action_dim))

        std = torch.exp(self.log_std)
        dist = Normal(actions, std)

        # 采样动作
        sampled_actions = dist.rsample()
        log_probs = dist.log_prob(sampled_actions).sum(dim=-1)

        # 价值估计
        if value_estimates is not None:
            values = value_estimates
        else:
            values = torch.zeros(actions.size(0), device=actions.device)

        return sampled_actions, log_probs, values

    def evaluate_actions(
        self,
        state: Dict[str, Any],
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        评估动作

        Args:
            state: 环境状态
            actions: 要评估的动作

        Returns:
            (log_probs, values)
        """
        output = self.controller(state)

        if not output["selected_vehicle_ids"]:
            return torch.zeros(1), torch.zeros(1)

        raw_actions = output["raw_actions"]
        value_estimates = output.get("value_estimates")

        # 缩放动作
        mean_actions = raw_actions * self.action_scale.to(raw_actions.device) + self.action_bias.to(raw_actions.device)

        # 计算log概率
        std = torch.exp(self.log_std)
        dist = Normal(mean_actions, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)

        # 价值估计
        if value_estimates is not None:
            values = value_estimates
        else:
            values = torch.zeros(mean_actions.size(0), device=mean_actions.device)

        return log_probs, values


class PPOTrainer:
    """
    PPO训练器 - 完整实现

    实现PPO-Clip算法
    """

    def __init__(
        self,
        model: TrafficController,
        config: Config,
    ):
        self.model = model
        self.config = config

        # 提取PPO配置
        ppo_cfg = config.training.get("phase2", {})

        self.num_envs = ppo_cfg.get("num_envs", 1)
        self.total_timesteps = ppo_cfg.get("total_timesteps", 20000)
        self.learning_rate = ppo_cfg.get("learning_rate", 3e-4)
        self.gamma = ppo_cfg.get("gamma", 0.99)
        self.gae_lambda = ppo_cfg.get("gae_lambda", 0.95)
        self.clip_epsilon = ppo_cfg.get("clip_epsilon", 0.2)
        self.entropy_coef = ppo_cfg.get("entropy_coef", 0.01)
        self.value_loss_coef = ppo_cfg.get("value_loss_coef", 0.5)
        self.max_grad_norm = ppo_cfg.get("max_grad_norm", 0.5)
        self.update_epochs = ppo_cfg.get("update_epochs", 4)
        self.batch_size = ppo_cfg.get("batch_size", 64)

        self.rollout_size = ppo_cfg.get("rollout_size", 2048)

        # 设备
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        # 创建策略网络
        self.policy = PPOPolicy(model.controller).to(self.device)

        # 冻结其他组件
        model.freeze_component("gnn")
        model.freeze_component("world_model")

        # 优化器（只优化控制器）
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.learning_rate,
        )

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.total_timesteps,
            eta_min=self.learning_rate * 0.01,
        )

        # 回放缓冲区
        self.buffer = RolloutBuffer(
            buffer_size=self.rollout_size,
            gae_lambda=self.gae_lambda,
            gamma=self.gamma,
        )

        # 统计
        self.stats = {
            "episode_rewards": [],
            "episode_lengths": [],
            "losses": [],
        }

        # 日志
        self.logger = setup_logger(
            log_file=config.log_dir / "ppo_training.log",
        )

    def collect_rollout(
        self,
        env: SumoEnvironment,
        num_steps: int,
    ) -> Dict[str, float]:
        """
        收集一个rollout

        Args:
            env: SUMO环境
            num_steps: 收集步数

        Returns:
            统计信息
        """
        self.buffer.clear()
        episode_rewards = []
        episode_lengths = []

        # 重置环境
        state = env.reset()
        episode_reward = 0
        episode_length = 0

        for step in range(num_steps):
            # 准备状态
            batch = self._prepare_batch(state, env)

            # 获取动作
            with torch.no_grad():
                actions, log_probs, values = self.policy(batch)

            # 转换为字典格式
            action_dict = self._convert_actions_to_dict(
                actions,
                batch.get("vehicle_ids", []),
            )

            # 执行动作
            next_state, reward, done, info = env.step(action_dict)

            # 存储到缓冲区
            self.buffer.add(
                state=batch,
                action=actions,
                log_prob=log_probs,
                reward=reward,
                value=values,
                done=done,
            )

            episode_reward += reward
            episode_length += 1

            # 如果episode结束，记录统计
            if done:
                self.buffer.finish_episode(episode_reward, episode_length)
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)

                # 重置
                state = env.reset()
                episode_reward = 0
                episode_length = 0
            else:
                state = next_state

        # 计算GAE
        with torch.no_grad():
            last_value = values if not done else 0.0
            advantages, returns = self.buffer.compute_gae(last_value)
            self.buffer.advantages = advantages
            self.buffer.returns = returns

        # 统计
        stats = {
            "mean_reward": np.mean(episode_rewards) if episode_rewards else 0.0,
            "mean_length": np.mean(episode_lengths) if episode_lengths else 0.0,
            "total_steps": len(self.buffer),
        }

        return stats

    def update(self) -> Dict[str, float]:
        """
        更新策略网络

        Returns:
            损失统计
        """
        total_losses = []
        policy_losses = []
        value_losses = []
        entropy_losses = []

        # 获取batches
        batches = self.buffer.get_batches(self.batch_size)

        # 多次更新（PPO的epochs）
        for epoch in range(self.update_epochs):
            for batch in batches:
                # 前向传播
                states = batch["states"]
                actions = batch["actions"].to(self.device)
                old_log_probs = batch["old_log_probs"].to(self.device)
                advantages = batch["advantages"].to(self.device)
                returns = batch["returns"].to(self.device)

                # 评估动作
                log_probs = []
                values = []
                for state in states:
                    state = self._move_batch_to_device(state)
                    lp, v = self.policy.evaluate_actions(state, actions[:len(state.get("vehicle_ids", []))])
                    log_probs.append(lp)
                    values.append(v)

                log_probs = torch.cat(log_probs) if log_probs else torch.zeros(1, device=self.device)
                values = torch.cat(values) if values else torch.zeros(1, device=self.device)

                # PPO-Clip损失
                ratio = torch.exp(log_probs - old_log_probs)

                # 策略损失
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # 价值损失
                value_loss = F.mse_loss(values, returns)

                # 熵损失（鼓励探索）
                entropy = -self.entropy_coef * torch.log(torch.exp(self.policy.log_std) + 1e-8).mean()

                # 总损失
                loss = policy_loss + self.value_loss_coef * value_loss + entropy

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_losses.append(loss.item())
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy.item())

        # 更新学习率
        self.scheduler.step()

        return {
            "loss": np.mean(total_losses),
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy": np.mean(entropy_losses),
            "lr": self.scheduler.get_last_lr()[0],
        }

    def train(self) -> TrafficController:
        """
        完整训练流程

        Returns:
            训练好的模型
        """
        self.logger.info("=" * 70)
        self.logger.info("PPO训练开始")
        self.logger.info("=" * 70)

        # 创建环境
        env = SumoEnvironment(
            self.config.environment,
            use_gui=False,
        )

        # 训练循环
        timestep = 0
        num_updates = self.total_timesteps // self.rollout_size

        with tqdm(total=self.total_timesteps, desc="PPO训练") as pbar:
            while timestep < self.total_timesteps:
                # 收集rollout
                rollout_steps = min(self.rollout_size, self.total_timesteps - timestep)

                collect_stats = self.collect_rollout(env, rollout_steps)

                # 更新策略
                update_stats = self.update()

                # 更新时间步
                timestep += rollout_steps
                pbar.update(rollout_steps)

                # 记录统计
                self.stats["episode_rewards"].append(collect_stats["mean_reward"])
                self.stats["episode_lengths"].append(collect_stats["mean_length"])
                self.stats["losses"].append(update_stats["loss"])

                # 打印进度
                pbar.set_postfix({
                    "reward": f"{collect_stats['mean_reward']:.2f}",
                    "loss": f"{update_stats['loss']:.4f}",
                    "lr": f"{update_stats['lr']:.2e}",
                })

                self.logger.info(
                    f"Timestep {timestep}/{self.total_timesteps} | "
                    f"Reward: {collect_stats['mean_reward']:.2f} | "
                    f"Loss: {update_stats['loss']:.4f} | "
                    f"LR: {update_stats['lr']:.2e}"
                )

        env.close()

        self.logger.info("=" * 70)
        self.logger.info("PPO训练完成")
        self.logger.info(f"最终平均奖励: {np.mean(self.stats['episode_rewards'][-10:]):.2f}")
        self.logger.info("=" * 70)

        # 保存检查点
        self._save_checkpoint()

        return self.model

    def _prepare_batch(self, observation: Observation, env: SumoEnvironment) -> Dict[str, Any]:
        """准备训练batch"""
        # 提取车辆状态
        vehicle_states = observation.vehicle_states
        vehicle_ids = observation.vehicle_ids
        icv_ids = observation.icv_ids

        # 构建batch
        batch = {
            "vehicle_states": vehicle_states,
            "vehicle_ids": vehicle_ids,
            "icv_ids": icv_ids,
            "global_metrics": observation.global_stats.unsqueeze(0),
            "is_icv": torch.tensor(
                [1 if vid in icv_ids else 0 for vid in vehicle_ids],
                dtype=torch.float32,
            ),
        }

        # 构建图数据
        graph_data = self._build_graph(observation)
        batch["graph_data"] = graph_data

        return batch

    def _build_graph(self, observation: Observation) -> Any:
        """从观测构建图数据"""
        # 提取车辆状态
        states = observation.vehicle_states
        vehicle_ids = observation.vehicle_ids

        if not vehicle_ids:
            # 返回空图
            return self.model.graph_builder.build_graph(
                positions=torch.zeros(0, 2),
                velocities=torch.zeros(0, 2),
                accelerations=torch.zeros(0, 2),
                lane_indices=torch.zeros(0, dtype=torch.long),
            )

        # 准备数据
        positions = []
        velocities = []
        accelerations = []
        lane_indices = []

        for vid in vehicle_ids:
            if vid not in states:
                continue
            state = states[vid]

            positions.append([state.x, state.y])
            velocities.append([state.speed * 0.707, state.speed * 0.707])  # 假设45度角
            accelerations.append([state.acceleration * 0.707, state.acceleration * 0.707])
            lane_indices.append(state.lane_index)

        # 转换为tensor
        positions = torch.tensor(positions, dtype=torch.float32)
        velocities = torch.tensor(velocities, dtype=torch.float32)
        accelerations = torch.tensor(accelerations, dtype=torch.float32)
        lane_indices = torch.tensor(lane_indices, dtype=torch.long)

        # 构建图
        graph_data = self.model.graph_builder.build_graph(
            positions=positions,
            velocities=velocities,
            accelerations=accelerations,
            lane_indices=lane_indices,
        )

        return graph_data

    def _convert_actions_to_dict(
        self,
        actions: torch.Tensor,
        vehicle_ids: List[str],
    ) -> Dict[str, np.ndarray]:
        """将tensor动作转换为字典"""
        action_dict = {}

        if not vehicle_ids:
            return action_dict

        actions_np = actions.cpu().numpy()

        # 假设actions已经通过控制器选择了top-k车辆
        # 这里我们需要使用模型实际的输出
        # 暂时简化：将第一个动作应用到第一个ICV
        num_actions = min(len(vehicle_ids), actions_np.shape[0])

        for i in range(num_actions):
            action_dict[vehicle_ids[i]] = actions_np[i]

        return action_dict

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """移动batch到设备"""
        result = {}

        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(self.device)
            elif isinstance(value, dict):
                result[key] = self._move_batch_to_device(value)
            else:
                result[key] = value

        return result

    def _save_checkpoint(self):
        """保存检查点"""
        checkpoint_path = self.config.checkpoint_dir / "ppo_phase2.ckpt"

        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "stats": self.stats,
            "config": self.config.to_dict(),
        }, checkpoint_path)

        self.logger.info(f"检查点已保存: {checkpoint_path}")

    def _load_checkpoint(self, checkpoint_path: Path):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.stats = checkpoint["stats"]

        self.logger.info(f"检查点已加载: {checkpoint_path}")
