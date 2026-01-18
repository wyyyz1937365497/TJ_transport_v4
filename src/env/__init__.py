"""
SUMO环境模块

核心环境：
- GPUSumoEnvironmentOptimized: GPU加速的SUMO环境（基类）
- CompetitionSumoEnv: 比赛专用环境（完整奖励函数）

包装器：
- GymSumoEnv: Gymnasium接口包装
- ParallelSumoEnvs: 并行环境管理器
"""

from .competition_env import CompetitionSumoEnv
from .gym_wrapper import GymSumoEnv, make_gym_env
from .vec_env import create_parallel_envs, ParallelSumoEnvs

__all__ = [
    'CompetitionSumoEnv',
    'GymSumoEnv',
    'make_gym_env',
    'create_parallel_envs',
    'ParallelSumoEnvs',
]
