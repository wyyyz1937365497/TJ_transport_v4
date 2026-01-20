"""
并行环境管理器 - 原生Gymnasium并行环境

使用Gymnasium的VectorEnv实现真正的并行
每个SUMO实例在独立进程中运行，自动处理端口分配
"""

import gymnasium as gym
import numpy as np
import multiprocessing as mp
from typing import Dict, Any, List, Optional
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

from .gym_wrapper import GymSumoEnv, make_gym_env

# 设置spawn模式以避免CUDA fork问题
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # 已经设置过了


class ParallelSumoEnvs:
    """
    SUMO并行环境管理器

    使用Gymnasium的VectorEnv实现真正的并行
    每个SUMO实例在独立进程中运行，自动处理端口分配

    特性：
    - 自动端口分配（避免冲突）
    - 每个环境独立进程
    - 标准Gymnasium接口
    - GPU加速支持
    - 自动错误恢复
    """

    def __init__(
        self,
        config: Dict[str, Any],
        num_envs: int = 4,
        monitor_dir: Optional[str] = None,
        seed: Optional[int] = None,
        device: str = 'cuda'
    ):
        """
        初始化并行环境

        Args:
            config: SUMO配置
            num_envs: 并行环境数量
            monitor_dir: 监控日志目录
            seed: 随机种子
            device: GPU设备 ('cuda' or 'cpu')
        """
        self.config = config
        self._num_envs = num_envs  # 使用私有名称避免冲突
        self.device = device  # 保存设备信息
        self.seed = seed

        # 创建环境
        self.vec_env = self._make_vec_env(monitor_dir)

    def _make_env(self, rank: int, seed: Optional[int] = None) -> callable:
        """
        创建单个环境的工厂函数

        Args:
            rank: 环境排名（用于端口分配）
            seed: 随机种子

        Returns:
            环境工厂函数
        """
        def _init():
            # 子进程中使用CPU模式，避免CUDA fork问题
            # 神经网络评分器会自动降级为规则评分
            env = make_gym_env(
                config=self.config,
                seed=seed,
                device='cpu'  # 强制使用CPU，避免CUDA fork错误
            )

            return env

        return _init

    def _make_vec_env(self, monitor_dir: Optional[str] = None):
        """
        创建向量化环境（使用Gymnasium VectorEnv）

        Args:
            monitor_dir: 监控目录（暂不支持，Gymnasium使用RecordVideo）

        Returns:
            VectorEnv实例（SyncVectorEnv或AsyncVectorEnv）
        """
        # 创建环境函数列表
        env_fns = [
            self._make_env(i, self.seed if self.seed is None else self.seed + i)
            for i in range(self._num_envs)
        ]

        # 如果只有1个环境，使用SyncVectorEnv（单进程）
        if self._num_envs == 1:
            vec_env = SyncVectorEnv(env_fns)
        else:
            # 多个环境使用AsyncVectorEnv（并行多进程）
            # 使用spawn context避免CUDA fork问题
            # 使用different observation_mode支持动态观测空间
            ctx = mp.get_context('spawn')
            vec_env = AsyncVectorEnv(
                env_fns, 
                shared_memory=False,
                context=ctx,
                observation_mode='different'  # 支持动态观测空间
            )

        # 注意：Gymnasium不使用VecMonitor，而是使用RecordVideo或其他监控方式
        # 如果需要监控，可以添加wrapper

        return vec_env

    def reset(self, seed: Optional[int] = None):
        """
        重置所有环境

        Args:
            seed: 随机种子（可选）

        Returns:
            observations: 初始观测
        """
        if seed is not None:
            return self.vec_env.reset(seed=seed)
        return self.vec_env.reset()

    def step(self, actions: List[Dict]):
        """
        在所有环境中执行动作

        Args:
            actions: [num_envs] 每个环境的动作（Dict格式）

        Returns:
            (observations, rewards, terminateds, truncateds, infos)
        """
        return self.vec_env.step(actions)

    def close(self):
        """关闭所有环境"""
        if hasattr(self, 'vec_env'):
            self.vec_env.close()

    def __getattr__(self, name):
        """转发未知属性到底层vec_env"""
        # 避免在序列化时触发递归
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        if 'vec_env' in self.__dict__:
            return getattr(self.vec_env, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    @property
    def num_envs(self) -> int:
        """环境数量"""
        return self._num_envs

    @property
    def observation_space(self) -> gym.Space:
        """观测空间"""
        return self.vec_env.observation_space

    @property
    def action_space(self) -> gym.Space:
        """动作空间"""
        return self.vec_env.action_space

    def get_attr(self, attr_name: str, indices=None):
        """获取环境属性"""
        return self.vec_env.get_attr(attr_name, indices)

    def set_attr(self, attr_name: str, value: Any, indices=None):
        """设置环境属性"""
        return self.vec_env.set_attr(attr_name, value, indices)

    def env_method(self, method_name: str, *args, indices=None, **kwargs):
        """在环境中调用方法"""
        return self.vec_env.env_method(method_name, *args, indices=indices, **kwargs)

    def seed(self, seed: int):
        """设置种子"""
        self.seed_val = seed
        return self.vec_env.seed(seed)

    def call(self, method_name: str, *args, **kwargs):
        """在所有环境中调用方法"""
        return self.vec_env.call(method_name, *args, **kwargs)


def create_parallel_envs(
    config: Dict[str, Any],
    num_envs: int = 4,
    monitor_dir: Optional[str] = None,
    seed: Optional[int] = None,
    device: str = 'cuda'
) -> ParallelSumoEnvs:
    """
    创建并行SUMO环境的便捷函数

    Args:
        config: SUMO配置
        num_envs: 并行环境数量
        monitor_dir: 监控目录（暂不支持）
        seed: 随机种子
        device: GPU设备 ('cuda' or 'cpu')

    Returns:
        ParallelSumoEnvs实例
    """
    return ParallelSumoEnvs(
        config=config,
        num_envs=num_envs,
        monitor_dir=monitor_dir,
        seed=seed,
        device=device
    )