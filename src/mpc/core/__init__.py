"""
MPC核心模块

包含交通流模型和MPC控制器。
"""

from .traffic_model import (
    SimplifiedTrafficModel,
    IDMController,
    TrafficTopology
)

from .mpc_controller import (
    MPCController,
    DistributedMPCController,
    MPCConfig
)

# GPU加速版本
from .traffic_model_gpu import GPUTrafficModel, BatchIDMModel
from .mpc_controller_gpu import GPUMPCController

__all__ = [
    # CPU版本
    'SimplifiedTrafficModel',
    'IDMController',
    'TrafficTopology',
    'MPCController',
    'DistributedMPCController',
    'MPCConfig',
    # GPU版本
    'GPUTrafficModel',
    'BatchIDMModel',
    'GPUMPCController'
]
