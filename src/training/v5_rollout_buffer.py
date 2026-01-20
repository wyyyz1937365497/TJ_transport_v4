"""
V5架构专用的Rollout Buffer

核心优化：
1. 支持Dict格式观测存储（直接存储，无需padding）
2. 延迟转换（只在训练时转换为Tensor）
3. 零拷贝访问（GPU加速）
4. 动态维度支持（节省内存）
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Iterator
from collections import deque


class V5RolloutBuffer:
    """
    V5架构专用的Rollout Buffer

    核心特性：
    - 存储Dict格式观测（保持动态维度，无需padding）
    - 延迟转换：只在训练时转换为Tensor（节省内存）
    - 零拷贝：直接在GPU上构建训练批次
    - GAE支持：广义优势估计

    存储格式：
    - observations: List[Dict] - 每个时间步的Dict观测
    - actions: [buffer_size, action_dim] - 动作（Tensor格式）
    - rewards: [buffer_size] - 奖励
    - advantages: [buffer_size] - 优势（GAE计算后）
    - returns: [buffer_size] - 回报（GAE计算后）
    - values: [buffer_size] - 价值估计
    - log_probs: [buffer_size] - 动作对数概率
    """

    def __init__(
        self,
        buffer_size: int,
        num_envs: int,
        device: str = 'cuda',
        max_vehicles: int = 512,
        obs_dim: int = 321,  # max_vehicles * 9 + 32 + 1
        action_dim: int = 2  # 加速度 + 换道
    ):
        """
        初始化Rollout Buffer

        Args:
            buffer_size: 每个环境收集的步数
            num_envs: 并行环境数
            device: 设备
            max_vehicles: 最大车辆数（用于Tensor转换）
            obs_dim: 观测维度（用于Tensor转换）
            action_dim: 动作维度
        """
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.device = device
        self.max_vehicles = max_vehicles
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # ========== 存储数组（GPU加速） ==========
        # 注意：observations存储为List[Dict]以保持动态维度
        self.observations = []  # List[Dict] - 延迟存储
        self.actions = torch.zeros((buffer_size, num_envs, action_dim * max_vehicles), dtype=torch.float32)
        self.rewards = torch.zeros((buffer_size, num_envs), dtype=torch.float32)
        self.dones = torch.zeros((buffer_size, num_envs), dtype=torch.float32)
        self.values = torch.zeros((buffer_size, num_envs), dtype=torch.float32)
        self.log_probs = torch.zeros((buffer_size, num_envs), dtype=torch.float32)

        # GAE计算后填充
        self.advantages = torch.zeros((buffer_size, num_envs), dtype=torch.float32)
        self.returns = torch.zeros((buffer_size, num_envs), dtype=torch.float32)

        # 指针
        self.pos = 0
        self.full = False

        print(f"[V5RolloutBuffer] 初始化完成")
        print(f"  - buffer_size: {buffer_size}")
        print(f"  - num_envs: {num_envs}")
        print(f"  - max_vehicles: {max_vehicles}")
        print(f"  - 总容量: {buffer_size * num_envs} transitions")

    def add(
        self,
        obs: Dict,
        action: np.ndarray,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
        env_idx: int = 0
    ):
        """
        添加一个transition

        Args:
            obs: Dict格式观测
            action: [action_dim * max_vehicles] 动作（Tensor格式）
            reward: 奖励
            done: 是否终止
            value: 价值估计
            log_prob: 对数概率
            env_idx: 环境索引
        """
        # 存储Dict观测（保持原始格式，无需padding）
        # 第一次到达这个位置时存储
        if env_idx == 0 and len(self.observations) <= self.pos:
            self.observations.append(obs)

        # 存储其他数据（GPU张量）
        self.actions[self.pos, env_idx] = torch.as_tensor(action, dtype=torch.float32)
        self.rewards[self.pos, env_idx] = reward
        self.dones[self.pos, env_idx] = float(done)
        self.values[self.pos, env_idx] = value
        self.log_probs[self.pos, env_idx] = log_prob

        # 更新指针
        if env_idx == self.num_envs - 1:
            self.pos += 1
            if self.pos >= self.buffer_size:
                self.pos = 0
                self.full = True

    def compute_returns_and_advantages(
        self,
        last_value: np.ndarray,
        last_done: np.ndarray,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ):
        """
        计算GAE优势估计和回报

        Args:
            last_value: [num_envs] 最后一步的价值估计
            last_done: [num_envs] 最后一步的done标志
            gamma: 折扣因子
            gae_lambda: GAE参数
        """
        # 转换为numpy数组计算
        rewards = self.rewards.cpu().numpy()
        values = self.values.cpu().numpy()
        dones = self.dones.cpu().numpy()

        last_value = np.asarray(last_value).reshape(-1)
        last_done = np.asarray(last_done).reshape(-1)

        # 初始化
        advantages = np.zeros_like(rewards)
        last_advantage = np.zeros(self.num_envs)
        last_return = last_value.copy()

        # 反向计算GAE（从后往前）
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_values = last_value
                next_non_terminal = 1.0 - last_done
            else:
                next_values = values[step + 1]
                next_non_terminal = 1.0 - dones[step + 1]

            # δ = r + γV(s') - V(s)
            delta = rewards[step] + gamma * next_values * next_non_terminal - values[step]

            # A = δ + γλ(1 - done)A'
            advantages[step] = delta + gamma * gae_lambda * next_non_terminal * last_advantage
            last_advantage = advantages[step].copy()

        # 计算回报：R = A + V
        returns = advantages + values

        # 存储到GPU
        self.advantages = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        self.returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

    def get(
        self,
        batch_size: int,
        dict_to_tensor_converter: Optional[callable] = None
    ) -> Iterator[Tuple[torch.Tensor, ...]]:
        """
        生成训练批次（带shuffle）

        Args:
            batch_size: 批大小
            dict_to_tensor_converter: Dict观测转Tensor的函数

        Yields:
            (obs_tensor, actions, advantages, returns, values, log_probs)
        """
        # 计算实际大小
        buffer_size = self.buffer_size if self.full else self.pos

        # Flatten: [buffer_size, num_envs] -> [buffer_size * num_envs]
        actions_flat = self.actions[:buffer_size].reshape(-1, self.actions.shape[-1])
        advantages_flat = self.advantages[:buffer_size].reshape(-1)
        returns_flat = self.returns[:buffer_size].reshape(-1)
        values_flat = self.values[:buffer_size].reshape(-1)
        log_probs_flat = self.log_probs[:buffer_size].reshape(-1)

        # 生成索引
        indices = np.random.permutation(buffer_size * self.num_envs)

        # 分批yield
        for start in range(0, buffer_size * self.num_envs, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]

            # 转换Dict观测到Tensor
            if dict_to_tensor_converter is not None:
                obs_batch = []
                for idx in batch_indices:
                    step_idx = idx // self.num_envs
                    obs_dict = self.observations[step_idx]
                    obs_batch.append(obs_dict)

                # 使用转换器批量转换
                obs_tensor = dict_to_tensor_converter(obs_batch)
            else:
                # 如果没有转换器，返回None（调用者需要处理）
                obs_tensor = None

            actions = actions_flat[batch_indices].to(self.device)
            advantages = advantages_flat[batch_indices].to(self.device)
            returns = returns_flat[batch_indices].to(self.device)
            old_values = values_flat[batch_indices].to(self.device)
            old_log_probs = log_probs_flat[batch_indices].to(self.device)

            yield obs_tensor, actions, advantages, returns, old_values, old_log_probs

    def __len__(self):
        """返回buffer中的样本数"""
        buffer_size = self.buffer_size if self.full else self.pos
        return buffer_size * self.num_envs
