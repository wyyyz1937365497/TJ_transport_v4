"""
V5架构专用PPO训练器

核心优化：
1. Dict ↔ Tensor 自动转换（桥接Gym环境和V5策略）
2. 稀疏控制优化（只处理Top-K车辆的动作）
3. 零拷贝观测格式化（GPU加速）
4. 批量动作解析（减少CPU开销）
5. V5特定的性能监控

架构适配：
- 环境：Dict格式观测/动作（Gymnasium标准）
- 策略：扁平化张量（LightweightPolicyV5）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
import time

from .custom_ppo_trainer import CustomPPOTrainer
from .v5_rollout_buffer import V5RolloutBuffer


class V5PPOTrainer(CustomPPOTrainer):
    """
    V5架构专用的PPO训练器

    核心职责：
    1. 自动转换Dict观测↔Tensor观测
    2. 自动转换Dict动作↔Tensor动作
    3. 优化稀疏控制机制（只跟踪Top-K动作）
    4. 提供V5特定的性能指标

    与父类CustomPPOTrainer的区别：
    - ✅ 添加Dict↔Tensor转换层
    - ✅ 优化动作解析（只提取Top-K车辆ID）
    - ✅ 添加V5特定的性能监控
    - ✅ 更好的批次处理（利用V5的批量优化）
    """

    def __init__(self, env, policy, config, checkpoint_dir, device='cuda'):
        """
        初始化V5 PPO训练器

        Args:
            env: Gymnasium环境（返回Dict观测，期望Dict动作）
            policy: LightweightPolicyV5（期望Tensor观测，返回Tensor动作）
            config: 训练配置
            checkpoint_dir: 检查点目录
            device: 设备
        """
        # 初始化V5特定配置
        model_config = config.get('model', {})
        gnn_config = model_config.get('gnn', {})
        self.top_k_ratio = gnn_config.get('top_k_ratio', 0.05)

        # 推断max_vehicles（从策略网络）
        self.max_vehicles = policy.max_vehicles
        self.obs_dim = policy.obs_dim  # = max_vehicles * 9 + 32 + 1

        # 初始化V5专用Rollout Buffer
        stage2_config = config.get('training', {}).get('stage2', {})
        n_steps = stage2_config.get('n_steps', 1024)
        num_envs = stage2_config.get('num_envs', 4)

        self.buffer = V5RolloutBuffer(
            buffer_size=n_steps,
            num_envs=num_envs,
            device=device,
            max_vehicles=self.max_vehicles,
            obs_dim=self.obs_dim
        )

        # 调用父类初始化（跳过buffer初始化）
        self.env = env
        self.policy = policy.to(device)
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.device = device

        # PPO超参数
        self.learning_rate = stage2_config.get('learning_rate', 0.0003)
        self.n_steps = n_steps
        self.batch_size = stage2_config.get('batch_size', 256)
        self.n_epochs = stage2_config.get('update_epochs', 10)
        self.gamma = stage2_config.get('gamma', 0.99)
        self.gae_lambda = stage2_config.get('gae_lambda', 0.95)
        self.clip_range = stage2_config.get('clip_range', 0.2)
        self.entropy_coef = stage2_config.get('entropy_coef', 0.01)
        self.value_loss_coef = stage2_config.get('value_loss_coef', 0.5)
        self.max_grad_norm = stage2_config.get('max_grad_norm', 0.5)
        self.target_kl = stage2_config.get('target_kl', 0.015)
        self.num_envs = num_envs

        # 优化器
        self.optimizer = optim.Adam(policy.parameters(), lr=self.learning_rate)

        # 训练状态
        self.current_obs = None
        self.learning_rate_schedule = None

        print(f"[V5PPOTrainer] 初始化完成")
        print(f"  - max_vehicles: {self.max_vehicles}")
        print(f"  - obs_dim: {self.obs_dim}")
        print(f"  - top_k_ratio: {self.top_k_ratio}")
        print(f"  - buffer_size: {n_steps * num_envs}")

    # ==================== Dict ↔ Tensor 转换层 ====================

    def _dict_obs_to_tensor(self, obs_dict: Dict[str, np.ndarray]) -> torch.Tensor:
        """
        将Dict格式观测转换为Tensor格式（V5策略期望的格式）

        Args:
            obs_dict: Dict格式观测
                {
                    'vehicle_states': [N, 9],
                    'vehicle_ids': [N],
                    'icv_ids': [N_icv],
                    'global_stats': [32],
                    'step': []
                }

        Returns:
            obs_tensor: [obs_dim] 扁平化观测
                = [vehicle_states flattened] + [global_stats] + [num_vehicles]

        性能优化：零拷贝，预分配数组
        """
        vehicle_states = obs_dict.get('vehicle_states', np.zeros((0, 9), dtype=np.float32))
        global_stats = obs_dict.get('global_stats', np.zeros(32, dtype=np.float32))
        vehicle_ids = obs_dict.get('vehicle_ids', np.zeros(0, dtype=np.int32))

        num_vehicles = len(vehicle_ids)

        # ⚡ 预分配完整观测数组
        obs_flat = np.zeros(self.obs_dim, dtype=np.float32)

        # 1. 填充车辆状态（padding到max_vehicles）
        vehicle_dim = self.max_vehicles * 9
        if num_vehicles > 0:
            obs_flat[:vehicle_states.size] = vehicle_states.flatten()
        # 剩余位置保持0（padding）

        # 2. 填充全局统计
        obs_start = vehicle_dim
        obs_end = vehicle_dim + 32
        obs_flat[obs_start:obs_end] = global_stats[:32]

        # 3. 填充实际车辆数
        obs_flat[obs_end] = float(num_vehicles)

        # 转换为Tensor（直接在目标设备）
        obs_tensor = torch.as_tensor(obs_flat, dtype=torch.float32, device=self.device)

        return obs_tensor

    def _batch_dict_obs_to_tensor(
        self,
        obs_batch: List[Dict[str, np.ndarray]]
    ) -> torch.Tensor:
        """
        批量转换Dict观测到Tensor（优化版）

        Args:
            obs_batch: [num_envs] 每个环境的Dict观测

        Returns:
            obs_tensor: [num_envs, obs_dim]
        """
        # 预分配批次数组
        batch_size = len(obs_batch)
        obs_array = np.zeros((batch_size, self.obs_dim), dtype=np.float32)

        # 批量填充
        for i, obs_dict in enumerate(obs_batch):
            vehicle_states = obs_dict.get('vehicle_states', np.zeros((0, 9), dtype=np.float32))
            global_stats = obs_dict.get('global_stats', np.zeros(32, dtype=np.float32))
            vehicle_ids = obs_dict.get('vehicle_ids', [])

            num_vehicles = len(vehicle_ids)
            vehicle_dim = self.max_vehicles * 9

            # 车辆状态（padding）
            if num_vehicles > 0:
                obs_array[i, :vehicle_states.size] = vehicle_states.flatten()

            # 全局统计
            obs_array[i, vehicle_dim:vehicle_dim + 32] = global_stats[:32]

            # 车辆数
            obs_array[i, vehicle_dim + 32] = float(num_vehicles)

        # 转换为Tensor
        obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32, device=self.device)

        return obs_tensor

    def _tensor_action_to_dict(
        self,
        action_tensor: torch.Tensor,
        obs_dict: Dict[str, np.ndarray],
        top_k_indices: Optional[torch.Tensor] = None
    ) -> Dict[str, np.ndarray]:
        """
        将Tensor动作转换为Dict格式（环境期望的格式）

        Args:
            action_tensor: [max_vehicles * 2] 扁平化动作
                [a1, l1, a2, l2, ..., aN, lN] 其中ai, li是第i辆车的动作
            obs_dict: 原始Dict观测（用于获取vehicle_ids）
            top_k_indices: [k] Top-K车辆索引（可选，如果不提供则解析所有车辆）

        Returns:
            action_dict: {'vehicle_ids': [id1, ...], 'actions': [[a1, l1], ...]}
                只包含Top-K车辆的动作（稀疏控制）

        性能优化：只提取Top-K车辆的动作
        """
        vehicle_ids = obs_dict.get('vehicle_ids', np.zeros(0, dtype=np.int32))
        num_vehicles = len(vehicle_ids)

        if num_vehicles == 0:
            return {'vehicle_ids': np.array([], dtype=np.int32), 'actions': np.zeros((0, 2), dtype=np.float32)}

        # Reshape动作：[max_vehicles * 2] -> [max_vehicles, 2]
        action_np = action_tensor.cpu().numpy().reshape(self.max_vehicles, 2)

        # 裁剪到实际车辆数
        action_np = action_np[:num_vehicles]

        # 确定Top-K车辆索引
        if top_k_indices is not None:
            # 使用提供的Top-K索引
            k_indices = top_k_indices.cpu().numpy()
            # 确保索引在有效范围内
            k_indices = k_indices[k_indices < num_vehicles]
        else:
            # 如果没有提供Top-K索引，则返回所有车辆的动作
            # （但这不是V5的预期用法，应该总是提供Top-K索引）
            k_indices = np.arange(num_vehicles)

        # 提取Top-K车辆ID和动作
        top_k_ids = vehicle_ids[k_indices]
        top_k_actions = action_np[k_indices]

        # 裁剪动作到合理范围
        top_k_actions[:, 0] = np.clip(top_k_actions[:, 0], -3.0, 3.0)  # 加速度
        top_k_actions[:, 1] = np.clip(top_k_actions[:, 1], -1.0, 1.0)  # 换道

        return {
            'vehicle_ids': top_k_ids.astype(np.int32),
            'actions': top_k_actions.astype(np.float32)
        }

    # ==================== 覆盖父类方法以支持Dict格式 ====================

    def collect_rollouts(self, reward_calculator=None) -> Dict[str, float]:
        """
        收集rollout数据（V5优化版）

        核心改进：
        1. 自动转换Dict观测↔Tensor观测
        2. 自动转换Dict动作↔Tensor动作
        3. 从GNN输出提取Top-K索引（用于稀疏控制）
        4. 优化批次处理（零拷贝）

        Returns:
            stats: 收集过程的统计信息
        """
        # 重置环境（如果需要）
        if self.current_obs is None:
            obs_dict, _ = self.env.reset()
            self.current_obs = obs_dict

        episode_rewards = []
        episode_lengths = []
        start_time = time.time()

        # 收集rollouts
        for step in range(self.n_steps):
            # ========== 转换观测：Dict -> Tensor ==========
            obs_batch = [self.current_obs]  # 单环境
            obs_tensor = self._batch_dict_obs_to_tensor(obs_batch)  # [1, obs_dim]

            # ========== 策略前向传播 ==========
            with torch.no_grad():
                # V5策略：输出Tensor动作
                actions_tensor, values, log_probs = self.policy(
                    obs_tensor,
                    deterministic=False
                )

                # ⚡ 提取Top-K索引（用于稀疏控制）
                # 需要重新运行GNN获取Top-K信息
                vehicle_states, _, num_vehicles = self.policy._parse_observation(obs_tensor)
                gnn_outputs = self.policy.gnn(vehicle_states, num_vehicles, return_debug_info=False)
                top_k_indices = gnn_outputs['top_k_indices'][0]  # [k]（squeeze batch维度）

            # ========== 转换动作：Tensor -> Dict ==========
            actions_dict = self._tensor_action_to_dict(
                actions_tensor[0],  # [obs_dim]（squeeze batch维度）
                self.current_obs,
                top_k_indices
            )

            # ========== 执行环境步 ==========
            actions_list = [actions_dict]  # 单环境
            next_obs_dict, rewards, dones, truncateds, infos = self.env.step(actions_list)

            # 解包单环境结果
            next_obs = next_obs_dict
            reward = rewards[0]
            done = dones[0]
            truncated = truncateds[0]
            info = infos[0]

            # ========== 存储到buffer ==========
            # 注意：buffer存储Dict观测（延迟转换）
            self.buffer.add(
                obs=self.current_obs,  # Dict格式
                action=actions_tensor.cpu().numpy()[0],  # [obs_dim] Tensor动作
                reward=reward,
                done=done,
                value=values.cpu().numpy()[0, 0],  # 标量
                log_prob=log_probs.cpu().numpy()[0],  # 标量
                env_idx=0  # 单环境
            )

            # 更新当前观测
            self.current_obs = next_obs

            # 跟踪episodes
            if 'episode_reward' in info:
                episode_rewards.append(info['episode_reward'])
            if 'episode_length' in info:
                episode_lengths.append(info['episode_length'])

            # 重置完成的environments
            if done or truncated:
                self.current_obs, _ = self.env.reset()

        # ========== 计算最后一步的value ==========
        obs_tensor = self._batch_dict_obs_to_tensor([self.current_obs])
        with torch.no_grad():
            last_values = self.policy(obs_tensor)[0].cpu().numpy()  # [1]
        self.last_values = last_values  # ✅ 缓存到实例变量
        last_dones = np.array([False])

        # 计算统计信息
        collection_time = time.time() - start_time
        steps_per_second = self.n_steps * self.num_envs / collection_time

        stats = {
            'collection_time': collection_time,
            'steps_per_second': steps_per_second,
            'mean_episode_reward': np.mean(episode_rewards) if episode_rewards else 0.0,
            'mean_episode_length': np.mean(episode_lengths) if episode_lengths else 0.0,
        }

        return stats

    def train(self, update_idx=0, reward_calculator=None) -> Dict[str, float]:
        """
        训练一个PPO更新周期（V5优化版）

        改进点：
        1. 从buffer提取Dict观测时自动转换为Tensor
        2. 添加V5特定的性能监控
        3. 利用V5的批量处理优化
        4. 使用V5RolloutBuffer的生成器接口

        Args:
            update_idx: 更新索引
            reward_calculator: 奖励计算器（可选）

        Returns:
            train_stats: 训练统计信息
        """
        import time
        start_time = time.time()

        # 1. 计算GAE
        with torch.no_grad():
            # 使用缓存的last_values
            if hasattr(self, 'last_values') and self.last_values is not None:
                last_values = self.last_values.to(self.device)
            else:
                # Fallback
                obs_tensor = self._batch_dict_obs_to_tensor([self.current_obs])
                _, last_values, _ = self.policy(obs_tensor)
                last_values = last_values.flatten().cpu().numpy()

            last_dones = np.zeros(self.num_envs, dtype=bool)

            self.buffer.compute_returns_and_advantages(
                last_values=last_values,
                last_done=last_dones,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda
            )

        # 2. 标准化advantages
        with torch.no_grad():
            advantages = self.buffer.advantages
            if advantages.std() > 1e-8:
                normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                self.buffer.advantages.copy_(normalized_advantages)

        # 3. 多epoch训练
        policy_losses = []
        value_losses = []
        entropy_losses = []
        kl_divs = []

        for epoch in range(self.n_epochs):
            # 使用V5RolloutBuffer的get方法，传入转换函数
            for obs_tensor, actions, advantages, returns, old_values, old_log_probs in self.buffer.get(
                batch_size=self.batch_size,
                dict_to_tensor_converter=self._batch_dict_obs_to_tensor
            ):
                # Forward pass
                values, log_probs, entropy = self.policy.evaluate_actions(obs_tensor, actions)

                # Flatten
                values = values.flatten()
                log_probs = log_probs.flatten()
                advantages = advantages.flatten()
                returns = returns.flatten()

                # KL散度（用于早停）
                with torch.no_grad():
                    ratio = torch.exp(log_probs - old_log_probs)
                    kl = (old_log_probs - log_probs).mean()
                    kl_divs.append(kl.item())

                # KL早停
                if self.target_kl > 0 and kl > self.target_kl * 1.5:
                    print(f"[V5PPOTrainer] KL early stop at epoch {epoch+1}, KL={kl:.4f}")
                    break

                # PPO loss
                ratio = torch.exp(log_probs - old_log_probs)
                policy_loss = -torch.min(
                    ratio * advantages,
                    torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
                ).mean()

                # Value loss (clipped)
                value_loss_clipped = old_values + torch.clamp(
                    values - old_values,
                    -self.clip_range,
                    self.clip_range
                )
                value_loss = torch.max(
                    F.mse_loss(values, returns),
                    F.mse_loss(value_loss_clipped, returns)
                )

                # Entropy loss
                entropy_loss = -entropy

                # Total loss
                loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # 记录
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())

            # KL早停（外部epoch循环）
            if self.target_kl > 0 and np.mean(kl_divs[-10:]) > self.target_kl * 1.5:
                break

        # 计算统计
        train_time = time.time() - start_time

        train_stats = {
            'train_time': train_time,
            'policy_loss': np.mean(policy_losses) if policy_losses else 0.0,
            'value_loss': np.mean(value_losses) if value_losses else 0.0,
            'entropy_loss': np.mean(entropy_losses) if entropy_losses else 0.0,
            'kl_div': np.mean(kl_divs) if kl_divs else 0.0,
            'update_idx': update_idx
        }

        return train_stats


def create_v5_ppo_trainer(
    env,
    policy,
    config,
    checkpoint_dir,
    device='cuda'
) -> V5PPOTrainer:
    """
    创建V5 PPO训练器的便捷函数

    Args:
        env: Gymnasium环境
        policy: LightweightPolicyV5
        config: 训练配置
        checkpoint_dir: 检查点目录
        device: 设备

    Returns:
        V5PPOTrainer实例
    """
    return V5PPOTrainer(
        env=env,
        policy=policy,
        config=config,
        checkpoint_dir=checkpoint_dir,
        device=device
    )
