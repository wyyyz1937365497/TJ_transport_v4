"""
训练模块包

包含：
1. CustomPPOTrainer - GPU优化的PPO训练器
2. PhaseCheckpointManager - 阶段检查点管理器
"""

from .custom_ppo_trainer import CustomPPOTrainer, GPURolloutBuffer
from .checkpoint_manager import PhaseCheckpointManager

__all__ = [
    'CustomPPOTrainer',
    'GPURolloutBuffer',
    'PhaseCheckpointManager',
]
