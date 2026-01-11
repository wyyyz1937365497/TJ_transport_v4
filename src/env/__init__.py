"""
SUMO环境模块
"""

from .sumo_env import SumoEnvironment
try:
    from .data_collector import EfficientDataCollector, TrajectoryDataset
    _data_collector_available = True
except ImportError:
    _data_collector_available = False

try:
    from .parallel_collector import (
        ParallelDataCollector,
        collect_parallel_data_optimized,
        collect_single_episode
    )
    _parallel_collector_available = True
except ImportError:
    _parallel_collector_available = False

__all__ = ['SumoEnvironment']

if _data_collector_available:
    __all__.extend(['EfficientDataCollector', 'TrajectoryDataset'])

if _parallel_collector_available:
    __all__.extend(['ParallelDataCollector', 'collect_parallel_data_optimized', 'collect_single_episode'])
