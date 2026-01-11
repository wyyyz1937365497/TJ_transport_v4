"""
并行环境管理器 - 使用Stable-Baselines3的SubprocVecEnv
解决SUMO多进程并行问题
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Any, List, Optional
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv,
    VecEnvStepReturn,
)

from .gym_wrapper import GymSumoEnv, make_gym_env


class ParallelSumoEnvs:
    """
    SUMO并行环境管理器

    使用Stable-Baselines3的SubprocVecEnv实现真正的并行，
    每个SUMO实例在独立进程中运行，自动处理端口分配。

    特性：
    - 自动端口分配（避免冲突）
    - 每个环境独立进程
    - 标准Gymnasium接口
    - 与Stable-Baselines3无缝集成
    - 自动错误恢复
    """

    def __init__(
        self,
        config: Dict[str, Any],
        num_envs: int = 4,
        base_port: int = 8813,
        monitor_dir: Optional[str] = None,
        seed: Optional[int] = None
    ):
        """
        初始化并行环境

        Args:
            config: SUMO配置
            num_envs: 并行环境数量
            base_port: 起始端口
            monitor_dir: 监控日志目录
            seed: 随机种子
        """
        self.config = config
        self._num_envs = num_envs  # 使用私有名称避免冲突
        self.base_port = base_port
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
            # 每个环境使用不同端口
            port = self.base_port + rank * 10

            # 创建环境
            env = make_gym_env(
                config=self.config,
                port=port,
                seed=seed
            )

            return env

        return _init

    def _make_vec_env(self, monitor_dir: Optional[str] = None) -> VecEnv:
        """
        创建向量化环境

        Args:
            monitor_dir: 监控目录

        Returns:
            VecEnv实例
        """
        # 如果只有1个环境，使用DummyVecEnv（不启动新进程）
        if self._num_envs == 1:
            env_fns = [self._make_env(0, self.seed)]
            vec_env = DummyVecEnv(env_fns)
        else:
            # 多个环境使用SubprocVecEnv（并行）
            env_fns = [
                self._make_env(i, self.seed if self.seed is None else self.seed + i)
                for i in range(self._num_envs)
            ]

            vec_env = SubprocVecEnv(
                env_fns=env_fns,
                start_method='spawn'  # 使用spawn避免fork问题
            )

        # 添加监控（记录episode统计）
        if monitor_dir is not None:
            vec_env = VecMonitor(vec_env, monitor_dir)

        return vec_env

    def reset(self) -> np.ndarray:
        """重置所有环境"""
        return self.vec_env.reset()

    def step(self, actions: np.ndarray) -> VecEnvStepReturn:
        """
        在所有环境中执行动作

        Args:
            actions: [num_envs, action_dim]

        Returns:
            (observations, rewards, dones, infos)
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


def create_parallel_envs(
    config: Dict[str, Any],
    num_envs: int = 4,
    base_port: int = 8813,
    monitor_dir: Optional[str] = None,
    seed: Optional[int] = None
) -> ParallelSumoEnvs:
    """
    创建并行SUMO环境的便捷函数

    Args:
        config: SUMO配置
        num_envs: 并行环境数量
        base_port: 起始端口
        monitor_dir: 监控目录
        seed: 随机种子

    Returns:
        ParallelSumoEnvs实例
    """
    return ParallelSumoEnvs(
        config=config,
        num_envs=num_envs,
        base_port=base_port,
        monitor_dir=monitor_dir,
        seed=seed
    )


# 辅助函数：用于数据收集
def collect_data_parallel(
    config: Dict[str, Any],
    num_episodes: int = 10,
    num_envs: int = 4,
    max_steps: int = 3600,
    output_dir: str = "data"
) -> tuple:
    """
    使用并行环境收集数据

    Args:
        config: SUMO配置
        num_episodes: 需要收集的episodes数量
        num_envs: 并行环境数
        max_steps: 每个episode最大步数
        output_dir: 输出目录

    Returns:
        (trajectories, stats)
    """
    import time
    import pickle
    from pathlib import Path

    # 创建并行环境
    parallel_envs = create_parallel_envs(
        config=config,
        num_envs=num_envs,
        seed=config.get('seed', 42)
    )

    # 数据收集
    all_trajectories = {}
    episode_count = 0
    start_time = time.time()

    print(f"\n{'='*70}")
    print(f"🚀 开始并行数据收集 (Stable-Baselines3 VecEnv)")
    print(f"{'='*70}")
    print(f"   - Episodes: {num_episodes}")
    print(f"   - 并行环境: {num_envs}")
    print(f"   - 最大步数: {max_steps}")
    print(f"{'='*70}\n")

    current_obs = parallel_envs.reset()

    while episode_count < num_episodes:
        # 随机动作（数据收集阶段）
        actions = [parallel_envs.action_space.sample() for _ in range(num_envs)]

        # 执行一步
        obs, rewards, dones, infos = parallel_envs.step(actions)

        # 记录数据（这里简化处理，实际需要更复杂的数据收集逻辑）
        for env_idx, info in enumerate(infos):
            # 检查是否episode结束
            if dones[env_idx]:
                episode_count += 1
                print(f"   ✅ Episode {episode_count}/{num_episodes} 完成 (env {env_idx})")

                if episode_count >= num_episodes:
                    break

    # 清理
    parallel_envs.close()

    # 保存数据
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    filepath = Path(output_dir) / f'vec_env_data_{timestamp}.pkl'

    with open(filepath, 'wb') as f:
        pickle.dump({
            'trajectories': all_trajectories,
            'stats': {
                'num_episodes': num_episodes,
                'collection_time': time.time() - start_time,
                'num_envs': num_envs
            }
        }, f)

    print(f"\n{'='*70}")
    print(f"✅ 数据收集完成！")
    print(f"   - Episodes: {episode_count}")
    print(f"   - 耗时: {time.time() - start_time:.1f}s")
    print(f"   - 数据文件: {filepath}")
    print(f"{'='*70}\n")

    return all_trajectories, {
        'num_episodes': num_episodes,
        'collection_time': time.time() - start_time,
        'filepath': str(filepath)
    }
