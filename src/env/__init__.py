"""
SUMO环境模块

已清理：
- 删除了重复的 gpu_sumo_env.py（与 gpu_sumo_env_optimized.py 功能重复）
- 删除了过时的 sumo_env.py（使用旧版 TraCI，性能较低）
"""

from .competition_env import CompetitionSumoEnv
from .gym_wrapper import GymSumoEnv, make_gym_env
from .vec_env import create_parallel_envs, ParallelSumoEnvs
from .gpu_sumo_env_optimized import GPUSumoEnvironmentOptimized

__all__ = [
    'GPUSumoEnvironmentOptimized',
    'CompetitionSumoEnv',
    'GymSumoEnv',
    'make_gym_env',
    'create_parallel_envs',
    'ParallelSumoEnvs',
]
