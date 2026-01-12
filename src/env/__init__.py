"""
SUMO环境模块
"""

from .sumo_env import SumoEnvironment
try:
    from .data_collector import EfficientDataCollector, TrajectoryDataset
    _data_collector_available = True
except ImportError:
    _data_collector_available = False

__all__ = ['SumoEnvironment']

if _data_collector_available:
    __all__.extend(['EfficientDataCollector', 'TrajectoryDataset'])
